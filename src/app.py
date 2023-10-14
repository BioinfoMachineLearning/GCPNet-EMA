import os
import shutil
import smtplib
import tempfile
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import hydra
import pandas as pd
import rootutils
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple
from flask import Flask, jsonify, make_response, render_template, request
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
from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

predict_cfg: Optional[DictConfig] = None
model: Optional[LightningModule] = None
plugins: Optional[ClusterEnvironment] = None
strategy: Optional[Strategy] = None
trainer: Optional[Trainer] = None

predict_input_dir: str = os.path.join(tempfile.gettempdir(), "gcpnet-ema", "inputs")
predict_output_dir: str = os.path.join(tempfile.gettempdir(), "gcpnet-ema", "outputs")


@app.route("/")
def index():
    """Hosts a homepage."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Hosts an about page."""
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def success():
    """Hosts an endpoint to make predictions for a given PDB file using a pre-trained
    checkpoint."""
    if request.method == "POST":
        global predict_cfg, model, plugins, strategy, trainer
        f = request.files["file"]
        save_location = f.filename
        f.save(save_location)
        os.makedirs(predict_input_dir, exist_ok=True)
        os.makedirs(predict_output_dir, exist_ok=True)
        new_save_location = os.path.join(predict_input_dir, os.path.basename(save_location))
        with open_dict(predict_cfg):
            predict_cfg.data.predict_input_dir = predict_input_dir
            predict_cfg.data.predict_true_dir = None
            predict_cfg.data.predict_output_dir = predict_output_dir
        shutil.move(save_location, new_save_location)
        predict(predict_cfg)
        prediction_df = pd.read_csv(trainer.model.predictions_csv_path)
        annotated_pdb_filepath = prediction_df["predicted_annotated_pdb_filepath"].iloc[-1]
        global_plddt = prediction_df["global_plddt"].iloc[-1].item()
        shutil.rmtree(predict_input_dir)
        shutil.rmtree(predict_output_dir)
        return render_template(
            "prediction.html",
            annotated_pdb_name=os.path.basename(annotated_pdb_filepath),
            global_plddt=f"{global_plddt:.2f}",
        )


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
def predict_and_send_email():
    try:
        global predict_cfg, model, plugins, strategy, trainer

        # extract input parameters from the request
        title = request.form.get("Title")
        structure_upload = request.files["Structure Upload"]
        sequence = request.form.get("Sequence")
        results_email = request.form.get("Results Email")
        other_parameters = request.form.get("Other Parameters")

        # make predictions
        save_location = structure_upload.filename
        structure_upload.save(save_location)
        os.makedirs(predict_input_dir, exist_ok=True)
        os.makedirs(predict_output_dir, exist_ok=True)
        new_save_location = os.path.join(predict_input_dir, os.path.basename(save_location))
        with open_dict(predict_cfg):
            predict_cfg.data.predict_input_dir = predict_input_dir
            predict_cfg.data.predict_true_dir = None
            predict_cfg.data.predict_output_dir = predict_output_dir
        shutil.move(save_location, new_save_location)
        predict(predict_cfg)
        prediction_df = pd.read_csv(trainer.model.predictions_csv_path)
        annotated_pdb_filepath = prediction_df["predicted_annotated_pdb_filepath"].iloc[-1]
        shutil.rmtree(predict_input_dir)
        shutil.rmtree(predict_output_dir)

        @beartype
        def send_email(
            subject: str,
            body: str,
            sender: str,
            recipients: List[str],
            password: str,
            output_file: str,
        ):
            # craft message
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = ", ".join(recipients)
            msg.attach(MIMEText(body, "plain"))
            # craft attachment
            with open(output_file, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {os.path.basename(output_file)}",
            )
            msg.attach(part)
            # send email with message and attachment
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
                smtp_server.login(sender, password)
                smtp_server.sendmail(sender, recipients, msg.as_string())

        # send the annotated PDB file as an email attachment
        send_email(
            subject=title,
            body="job complete",
            sender=os.environ["SERVER_EMAIL_ADDRESS"],
            recipients=[results_email],
            password=os.environ["SERVER_EMAIL_PASSWORD"],
            output_file=annotated_pdb_filepath,
        )

        return jsonify({"message": "Prediction completed and email sent."})

    except Exception as e:
        return jsonify({"error": str(e)})


@task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Predicts with given checkpoint on a datamodule predictset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    global model, plugins, strategy, trainer

    if model is None:
        log.info(f"Instantiating model <{cfg.model._target_}>")
        with open_dict(cfg):
            cfg.model.model_cfg = validate_config(cfg.model.model_cfg)
            cfg.model.model_cfg.ablate_af2_plddt = cfg.data.ablate_af2_plddt
            cfg.model.model_cfg.ablate_esm_embeddings = cfg.data.ablate_esm_embeddings
        benchmark_model = BenchMarkModel(cfg.model.model_cfg)
        local_model: LightningModule = hydra.utils.instantiate(
            cfg.model,
            model=benchmark_model,
            path_cfg=cfg.paths,
        )
        log.info("Loading checkpoint!")
        local_model = local_model.load_from_checkpoint(
            checkpoint_path=cfg.ckpt_path,
            map_location="cpu",
            strict=True,
            path_cfg=hydra.utils.instantiate(cfg.paths),
        )
        model = local_model

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

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

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

    global predict_cfg
    if predict_cfg is None:
        predict_cfg = cfg

    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)  # nosec


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    src_register_custom_omegaconf_resolvers()
    main()
