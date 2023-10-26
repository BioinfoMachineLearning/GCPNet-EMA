import ssl

import hydra
import rootutils
from beartype.typing import Any, Dict, List, Tuple
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies.strategy import Strategy
from omegaconf import DictConfig, open_dict
from proteinworkshop import register_custom_omegaconf_resolvers
from proteinworkshop.configs.config import validate_config
from proteinworkshop.models.base import BenchMarkModel

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src import (
    register_custom_omegaconf_resolvers as src_register_custom_omegaconf_resolvers,
)
from src import resolve_omegaconf_variable
from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Predicts with given checkpoint on a datamodule predictset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    if getattr(cfg, "create_unverified_ssl_context", False):
        log.info("Creating unverified SSL context!")
        ssl._create_default_https_context = ssl._create_unverified_context

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    with open_dict(cfg):
        cfg.model.model_cfg = validate_config(cfg.model.model_cfg)
        cfg.model.model_cfg.ablate_af2_plddt = cfg.data.ablate_af2_plddt
        cfg.model.model_cfg.ablate_esm_embeddings = cfg.data.ablate_esm_embeddings
        cfg.model.model_cfg.ablate_ankh_embeddings = cfg.data.ablate_ankh_embeddings
        cfg.model.model_cfg.ablate_gtn = cfg.model.ablate_gtn
        cfg.model.model_cfg.gtn_walk_length = cfg.model.gtn_walk_length
        cfg.model.model_cfg.gtn_emb_dim = cfg.model.gtn_emb_dim
        cfg.model.model_cfg.gtn_attn_type = cfg.model.gtn_attn_type
        cfg.model.model_cfg.gtn_dropout = cfg.model.gtn_dropout
        cfg.model.model_cfg.gtn_pe_dim = cfg.model.gtn_pe_dim
        cfg.model.model_cfg.gtn_num_layers = cfg.model.gtn_num_layers
    benchmark_model = BenchMarkModel(cfg.model.model_cfg)
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        model=benchmark_model,
        path_cfg=cfg.paths,
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    plugins = None
    if "_target_" in cfg.environment:
        log.info(f"Instantiating environment <{cfg.environment._target_}>")
        plugins: ClusterEnvironment = hydra.utils.instantiate(cfg.environment)

    strategy = getattr(cfg.trainer, "strategy", None)
    if "_target_" in cfg.strategy:
        log.info(f"Instantiating strategy <{cfg.strategy._target_}>")
        strategy: Strategy = hydra.utils.instantiate(cfg.strategy)
        if "mixed_precision" in strategy.__dict__:
            strategy.mixed_precision.param_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.param_dtype)
                if cfg.strategy.mixed_precision.param_dtype is not None
                else None
            )
            strategy.mixed_precision.reduce_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.reduce_dtype)
                if cfg.strategy.mixed_precision.reduce_dtype is not None
                else None
            )
            strategy.mixed_precision.buffer_dtype = (
                resolve_omegaconf_variable(cfg.strategy.mixed_precision.buffer_dtype)
                if cfg.strategy.mixed_precision.buffer_dtype is not None
                else None
            )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = (
        hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            plugins=plugins,
            strategy=strategy,
        )
        if strategy is not None
        else hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            plugins=plugins,
        )
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Loading checkpoint!")
    model = model.load_from_checkpoint(
        checkpoint_path=cfg.ckpt_path,
        map_location="cpu",
        strict=True,
        path_cfg=hydra.utils.instantiate(cfg.paths),
    )

    log.info("Starting predictions!")
    trainer.predict(model=model, datamodule=datamodule)
    log.info(f"Predictions saved to: {trainer.model.predictions_csv_path}")

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for prediction.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    predict(cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    src_register_custom_omegaconf_resolvers()
    main()
