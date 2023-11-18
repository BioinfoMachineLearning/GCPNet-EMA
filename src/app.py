# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import copy
import os
import shutil
import tempfile
import uuid

import hydra
import pandas as pd
import rootutils
from beartype.typing import Any, Dict, Optional, Tuple
from flask import (
    Flask,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_wtf.csrf import CSRFProtect
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
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
from src.utils import RankedLogger
from src.utils.email_utils import send_email_with_attachment
from src.utils.form_utils import PredictForm, ServerPredictForm

log = RankedLogger(__name__, rank_zero_only=True)

app = Flask(__name__)
app.secret_key = os.environ["SERVER_SECRET_KEY"]  # set the secret key for CSRF protection
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # e.g., 16 MB limit

csrf = CSRFProtect(app)

GlobalHydra.instance().clear()

register_custom_omegaconf_resolvers()
src_register_custom_omegaconf_resolvers()

# load configuration using Hydra
config_dir = "../configs"  # adjust the path to your configuration directory
config_name = "app.yaml"  # adjust the configuration file name
hydra.initialize(config_dir, version_base="1.3")
cfg = hydra.compose(config_name=config_name, return_hydra_config=True)
HydraConfig().cfg = cfg

predict_cfg: DictConfig = cfg
af2_predict_cfg: DictConfig = copy.deepcopy(cfg)
with open_dict(af2_predict_cfg):
    af2_predict_cfg.ckpt_path = af2_predict_cfg.af2_ckpt_path
    af2_predict_cfg.model.ablate_af2_plddt = False

datamodule: Optional[LightningDataModule] = None
model: Optional[LightningModule] = None
af2_model: Optional[LightningModule] = None
plugins: Optional[ClusterEnvironment] = None
strategy: Optional[Strategy] = None
trainer: Optional[Trainer] = None

predict_input_dir: str = os.path.join(tempfile.gettempdir(), "gcpnet-ema", "inputs")
predict_output_dir: str = os.path.join(tempfile.gettempdir(), "gcpnet-ema", "outputs")


@app.route("/")
def index():
    """Hosts a homepage."""
    form = PredictForm()
    return render_template("index.html", form=form)


@app.route("/about")
def about():
    """Hosts an about page."""
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def success():
    """Hosts an endpoint to make predictions for a given PDB file using a pre-trained
    checkpoint."""
    if request.method == "POST":
        try:
            # create an instance of the prediction form
            form = PredictForm()

            # validate the form data
            if form.validate_on_submit():
                pdb_file = form.file.data
                af2_input = form.af2_input.data
                cameo_output = form.cameo_output.data

                global predict_cfg, af2_predict_cfg, model, af2_model, plugins, strategy, trainer
                save_location = pdb_file.filename
                pdb_file.save(save_location)
                os.makedirs(predict_input_dir, exist_ok=True)
                os.makedirs(predict_output_dir, exist_ok=True)
                new_save_location = os.path.join(
                    predict_input_dir, os.path.basename(save_location)
                )
                with open_dict(predict_cfg):
                    predict_cfg.data.predict_input_dir = predict_input_dir
                    predict_cfg.data.predict_true_dir = None
                    predict_cfg.data.predict_output_dir = predict_output_dir
                with open_dict(af2_predict_cfg):
                    af2_predict_cfg.data.predict_input_dir = predict_input_dir
                    af2_predict_cfg.data.predict_true_dir = None
                    af2_predict_cfg.data.predict_output_dir = predict_output_dir
                shutil.move(save_location, new_save_location)
                predict(
                    predict_cfg, af2_predict_cfg, af2_input=af2_input, cameo_output=cameo_output
                )
                prediction_df = pd.read_csv(trainer.model.predictions_csv_path)
                annotated_pdb_filepath = prediction_df["predicted_annotated_pdb_filepath"].iloc[-1]
                global_score = prediction_df["global_score"].iloc[-1].item()
                shutil.rmtree(predict_input_dir)
                shutil.rmtree(predict_output_dir)
                return render_template(
                    "prediction.html",
                    annotated_pdb_name=os.path.basename(annotated_pdb_filepath),
                    global_score=f"{global_score:.2f}",
                )
            else:
                # form data is not valid, handle the validation errors
                flash("Form validation failed. Please check the form fields.")
                return redirect(url_for("index"))

        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return redirect(url_for("index"))


@app.route("/download_prediction/<filename>")
def download_prediction(filename: str):
    """Hosts an endpoint to download predicted PDB annotations previously made for an input PDB
    file."""
    filepath = os.path.join(tempfile.gettempdir(), filename)
    pdb_file = open(filepath)
    response = make_response(pdb_file.read())
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["Content-Type"] = "chemical/x-pdb"
    return response


@app.route("/server_predict", methods=["POST"])
@csrf.exempt
def predict_and_send_email():
    try:
        global predict_cfg, af2_predict_cfg, datamodule, model, af2_model, plugins, strategy, trainer

        # create an instance of the server prediction form
        form = ServerPredictForm()

        # validate form data
        if form.validate_on_submit():
            # extract input parameters from the request
            title = form.title.data
            structure_upload = form.data["structure_upload"]
            sequence = form.sequence.data  # NOTE: we currently do not use the provided sequence
            results_email = form.results_email.data
            other_parameters = form.other_parameters.data

            # make predictions
            unique_id = str(uuid.uuid4())
            predict_input_dir_ = os.path.join(predict_input_dir, f"{title}_{unique_id}")
            predict_output_dir_ = os.path.join(predict_output_dir, f"{title}_{unique_id}")
            os.makedirs(predict_input_dir_, exist_ok=True)
            os.makedirs(predict_output_dir_, exist_ok=True)
            save_location = os.path.join(
                predict_input_dir_, f"{unique_id}_{structure_upload.filename}"
            )
            new_save_location = os.path.join(predict_input_dir_, os.path.basename(save_location))
            structure_upload.save(save_location)
            with open_dict(predict_cfg):
                predict_cfg.data.predict_input_dir = predict_input_dir_
                predict_cfg.data.predict_true_dir = None
                predict_cfg.data.predict_output_dir = predict_output_dir_
            with open_dict(af2_predict_cfg):
                af2_predict_cfg.data.predict_input_dir = predict_input_dir_
                af2_predict_cfg.data.predict_true_dir = None
                af2_predict_cfg.data.predict_output_dir = predict_output_dir_
            shutil.move(save_location, new_save_location)
            af2_input = other_parameters is not None and "af2_input" in other_parameters
            # NOTE: for email-based responses, we default to returning a CAMEO-style accuracy metric
            af2_output = other_parameters is not None and "af2_output" in other_parameters
            predict(predict_cfg, af2_predict_cfg, af2_input=af2_input, cameo_output=not af2_output)
            prediction_df = pd.read_csv(trainer.model.predictions_csv_path)
            annotated_pdb_filepath = prediction_df["predicted_annotated_pdb_filepath"].iloc[-1]
            shutil.rmtree(predict_input_dir_)
            shutil.rmtree(predict_output_dir_)

            # send the annotated PDB file as an email attachment
            send_email_with_attachment(
                subject=title,
                body="job complete",
                sender=os.environ["SERVER_EMAIL_ADDRESS"],
                recipients=[results_email],
                output_file=annotated_pdb_filepath,
                smtp_server=os.environ["SERVER_EMAIL_SMTP_SERVER"],
                port=int(os.environ["SERVER_EMAIL_PORT"]),
            )

            return jsonify({"message": "Prediction completed and email sent."})

        else:
            # form data is not valid, return validation errors
            return jsonify({"Validation server error": form.errors})

    except Exception as e:
        return jsonify({"General server error": str(e)})


def predict(
    cfg: DictConfig, af2_cfg: DictConfig, af2_input: bool = False, cameo_output: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Predicts with given checkpoint on a datamodule predictset.

    :param cfg: DictConfig configuration composed by Hydra.
    :param af2_cfg: DictConfig configuration composed by Hydra for AlphaFold 2 inputs.
    :param af2_input: Whether an AlphaFold 2 structure has been provided for assessment.
    :param cameo_output: Whether to return a CAMEO-style accuracy metric.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path and af2_cfg.ckpt_path, "Checkpoint paths not provided!"

    global datamodule, model, af2_model, plugins, strategy, trainer

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    local_datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        load_esm_model=False,
        load_ankh_model=False,
        return_cameo_accuracy=cameo_output,
    )
    datamodule = local_datamodule

    # load the general-purpose model
    if model is None:
        log.info(f"Instantiating model <{cfg.model._target_}>")
        with open_dict(cfg):
            cfg.model.model_cfg = validate_config(cfg.model.model_cfg)
        benchmark_model = BenchMarkModel(cfg.model.model_cfg)
        with open_dict(cfg):
            # remove unpickleable `nn.Modules` from `cfg.model.model_cfg`
            model_cfg_finetune = DictConfig(
                {
                    "encoder": DictConfig(
                        {
                            "load_weights": cfg.model.model_cfg.finetune.encoder.load_weights,
                            "freeze": cfg.model.model_cfg.finetune.encoder.freeze,
                        }
                    ),
                    "decoder": DictConfig(
                        {
                            "load_weights": cfg.model.model_cfg.finetune.decoder.load_weights,
                            "freeze": cfg.model.model_cfg.finetune.decoder.freeze,
                        }
                    ),
                }
            )
            cfg.model.model_cfg = DictConfig(
                {
                    "ckpt_path": cfg.model.model_cfg.ckpt_path,
                    "ablate_esm_embeddings": cfg.data.ablate_esm_embeddings,
                    "ablate_ankh_embeddings": cfg.data.ablate_ankh_embeddings,
                    "ablate_af2_plddt": cfg.model.ablate_af2_plddt,
                    "ablate_gtn": cfg.model.ablate_gtn,
                    "gtn_walk_length": cfg.model.gtn_walk_length,
                    "gtn_emb_dim": cfg.model.gtn_emb_dim,
                    "gtn_attn_type": cfg.model.gtn_attn_type,
                    "gtn_dropout": cfg.model.gtn_dropout,
                    "gtn_pe_dim": cfg.model.gtn_pe_dim,
                    "gtn_num_layers": cfg.model.gtn_num_layers,
                }
            )
            cfg.model.model_cfg.finetune = model_cfg_finetune
        local_model: LightningModule = hydra.utils.instantiate(
            cfg.model,
            model=benchmark_model,
            path_cfg=cfg.paths,
        )
        log.info("Loading checkpoint!")
        local_model = local_model.__class__.load_from_checkpoint(
            checkpoint_path=cfg.ckpt_path,
            map_location="cpu",
            strict=True,
            path_cfg=hydra.utils.instantiate(cfg.paths),
            is_inference_run=True,
        )
        model = local_model

    # load the AlphaFold-specialized model
    if af2_model is None:
        log.info(f"Instantiating model <{af2_cfg.model._target_}>")
        with open_dict(af2_cfg):
            af2_cfg.model.model_cfg = validate_config(af2_cfg.model.model_cfg)
        benchmark_model = BenchMarkModel(af2_cfg.model.model_cfg)
        with open_dict(af2_cfg):
            # remove unpickleable `nn.Modules` from `af2_cfg.model.model_cfg`
            model_cfg_finetune = DictConfig(
                {
                    "encoder": DictConfig(
                        {
                            "load_weights": af2_cfg.model.model_cfg.finetune.encoder.load_weights,
                            "freeze": af2_cfg.model.model_cfg.finetune.encoder.freeze,
                        }
                    ),
                    "decoder": DictConfig(
                        {
                            "load_weights": af2_cfg.model.model_cfg.finetune.decoder.load_weights,
                            "freeze": af2_cfg.model.model_cfg.finetune.decoder.freeze,
                        }
                    ),
                }
            )
            af2_cfg.model.model_cfg = DictConfig(
                {
                    "ckpt_path": af2_cfg.model.model_cfg.ckpt_path,
                    "ablate_esm_embeddings": af2_cfg.data.ablate_esm_embeddings,
                    "ablate_ankh_embeddings": af2_cfg.data.ablate_ankh_embeddings,
                    "ablate_af2_plddt": af2_cfg.model.ablate_af2_plddt,
                    "ablate_gtn": af2_cfg.model.ablate_gtn,
                    "gtn_walk_length": af2_cfg.model.gtn_walk_length,
                    "gtn_emb_dim": af2_cfg.model.gtn_emb_dim,
                    "gtn_attn_type": af2_cfg.model.gtn_attn_type,
                    "gtn_dropout": af2_cfg.model.gtn_dropout,
                    "gtn_pe_dim": af2_cfg.model.gtn_pe_dim,
                    "gtn_num_layers": af2_cfg.model.gtn_num_layers,
                }
            )
            af2_cfg.model.model_cfg.finetune = model_cfg_finetune
        af2_local_model: LightningModule = hydra.utils.instantiate(
            af2_cfg.model,
            model=benchmark_model,
            path_cfg=af2_cfg.paths,
        )
        log.info("Loading checkpoint!")
        af2_local_model = af2_local_model.__class__.load_from_checkpoint(
            checkpoint_path=af2_cfg.ckpt_path,
            map_location="cpu",
            strict=True,
            path_cfg=hydra.utils.instantiate(af2_cfg.paths),
            is_inference_run=True,
        )
        af2_model = af2_local_model

    if plugins is None:
        local_plugins = None
        if "_target_" in cfg.environment:
            log.info(f"Instantiating environment <{cfg.environment._target_}>")
            local_plugins: ClusterEnvironment = hydra.utils.instantiate(cfg.environment)
        plugins = local_plugins

    if strategy is None:
        local_strategy = getattr(cfg.trainer, "strategy", None)
        if "_target_" in cfg.strategy:
            log.info(f"Instantiating strategy <{cfg.strategy._target_}>")
            local_strategy: Strategy = hydra.utils.instantiate(cfg.strategy)
            if "mixed_precision" in strategy.__dict__:
                local_strategy.mixed_precision.param_dtype = (
                    resolve_omegaconf_variable(cfg.strategy.mixed_precision.param_dtype)
                    if cfg.strategy.mixed_precision.param_dtype is not None
                    else None
                )
                local_strategy.mixed_precision.reduce_dtype = (
                    resolve_omegaconf_variable(cfg.strategy.mixed_precision.reduce_dtype)
                    if cfg.strategy.mixed_precision.reduce_dtype is not None
                    else None
                )
                local_strategy.mixed_precision.buffer_dtype = (
                    resolve_omegaconf_variable(cfg.strategy.mixed_precision.buffer_dtype)
                    if cfg.strategy.mixed_precision.buffer_dtype is not None
                    else None
                )
        strategy = local_strategy

    if trainer is None:
        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        local_trainer: Trainer = (
            hydra.utils.instantiate(
                cfg.trainer,
                plugins=plugins,
                strategy=strategy,
            )
            if strategy is not None
            else hydra.utils.instantiate(
                cfg.trainer,
                plugins=plugins,
            )
        )
        trainer = local_trainer

    effective_model = af2_model if af2_model is not None and af2_input else model
    object_dict = {
        "cfg": cfg,
        "af2_cfg": af2_cfg,
        "datamodule": datamodule,
        "model": effective_model,
        "trainer": trainer,
    }

    log.info("Starting predictions!")
    trainer.predict(model=effective_model, datamodule=datamodule)
    log.info(f"Predictions saved to: {trainer.model.predictions_csv_path}")

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 80))  # nosec
