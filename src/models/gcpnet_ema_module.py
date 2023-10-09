# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import collections
import os
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torchmetrics
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple
from lightning import LightningModule
from omegaconf import DictConfig
from proteinworkshop.configs.config import validate_config
from proteinworkshop.datasets.utils import create_example_batch
from proteinworkshop.models.base import BenchMarkModel
from torch_scatter import scatter

from src.data.components.ema_dataset import MAX_PLDDT_VALUE
from src.models import HALT_FILE_EXTENSION, annotate_pdb_with_new_column_values
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class GCPNetEMALitModule(LightningModule):
    """A `LightningModule` for estimation of protein structure model accuracy (EMA) using GCPNet.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        model_cfg: DictConfig,
        path_cfg: DictConfig = None,
        **kwargs,
    ):
        """Initialize a `GCPNetEMALitModule` instance.

        :param optimizer: An optimizer instance.
        :param scheduler: A scheduler instance.
        :param compile: Whether to compile the model.
        :param model_cfg: A `ProteinWorkshop` model checkpoint `DictConfig` containing values for the following keys.
            - `ckpt_path`: Path to the `ProteinWorkshop` model checkpoint.
            - `dataset`: Dataset `DictConfig`.
            - `decoder`: Decoder `DictConfig`.
            - `encoder`: Encoder `DictConfig`.
            - `features`: Features `DictConfig`.
            - `metrics`: Metrics `DictConfig`.
            - `name`: Name of the run.
            - `num_workers`: Number of workers for the data loader.
            - `optimiser`: Optimiser `DictConfig`.
            - `scheduler`: Scheduler `DictConfig`.
            - `seed`: Random seed.
            - `task`: Task `DictConfig`.
            - `task_name`: Name of the task.
            - `trainer`: Trainer `DictConfig`.
            - `finetune`: Finetuning `DictConfig` which contains the values for the following keys.
                - `finetune.encoder.load_weights`: Whether to load the encoder weights from the checkpoint.
                - `finetune.encoder.freeze`: Whether to freeze the encoder weights.
                - `finetune.decoder.load_weights`: Whether to load the decoder weights from the checkpoint.
                - `finetune.decoder.freeze`: Whether to freeze the decoder weights.
        :param path_cfg: A dictionary-like object containing paths to various directories and
            files.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # `ProteinWorkshop` encoder-decoder model weights #
        validate_config(model_cfg)

        self.model: LightningModule = BenchMarkModel(model_cfg)

        # Initialize lazy layers for parameter counts.
        # This is also required for the `BenchMarkModel` model to be able to load weights.
        # Otherwise, lazy layers will have their parameters reset.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin
        print("Initializing `BenchMarkModel` lazy layers...")
        with torch.no_grad():
            batch = create_example_batch()
            batch = self.model.featurise(batch)
            out = self.model.forward(batch)
            del batch, out

        # NOTE: we only want to load weights
        if (
            model_cfg.ckpt_path
            and model_cfg.ckpt_path != "none"
            and os.path.exists(model_cfg.ckpt_path)
        ):
            log.info(f"Loading `BenchMarkModel` weights from checkpoint {model_cfg.ckpt_path}...")
            state_dict = torch.load(model_cfg.ckpt_path)["state_dict"]

            if model_cfg.finetune.encoder.load_weights:
                encoder_weights = collections.OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("encoder"):
                        encoder_weights[k.replace("encoder.", "")] = v
                log.info(f"Loading `BenchMarkModel` encoder weights: {encoder_weights}")
                err = self.model.encoder.load_state_dict(encoder_weights, strict=False)
                log.info(f"Error loading `BenchMarkModel` encoder weights: {err}")

            if model_cfg.finetune.decoder.load_weights:
                decoder_weights = collections.OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("decoder"):
                        decoder_weights[k.replace("decoder.", "")] = v
                log.info(f"Loading `BenchMarkModel` decoder weights: {decoder_weights}")
                err = self.model.decoder.load_state_dict(decoder_weights, strict=False)
                log.info(f"Error loading `BenchMarkModel` decoder weights: {err}")

            if model_cfg.finetune.encoder.freeze:
                log.info("Freezing `BenchMarkModel` encoder!")
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

            if model_cfg.finetune.decoder.freeze:
                log.info("Freezing `BenchMarkModel` decoder!")
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
        else:
            log.info(
                "A valid checkpoint path was not found. Training a new set of weights for the `BenchMarkModel`..."
            )

        # loss function and metrics #
        self.criterion = torch.nn.SmoothL1Loss()
        # note: for averaging loss across batches
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        # use separate metrics instances for the steps
        # of each phase (e.g., train, val and test)
        # to ensure a proper reduction over the epoch
        self.train_phase, self.val_phase, self.test_phase, self.predict_phase = (
            "train",
            "val",
            "test",
            "predict",
        )
        phases = [self.train_phase, self.val_phase, self.test_phase]

        metrics = {
            "PerResidueMSE": partial(torchmetrics.regression.mse.MeanSquaredError),
            "PerResidueMAE": partial(torchmetrics.regression.mae.MeanAbsoluteError),
            "PerResiduePearsonCorrCoef": partial(torchmetrics.PearsonCorrCoef),
            "PerModelMSE": partial(torchmetrics.regression.mse.MeanSquaredError),
            "PerModelMAE": partial(torchmetrics.regression.mae.MeanAbsoluteError),
            "PerModelPearsonCorrCoef": partial(torchmetrics.PearsonCorrCoef),
        }

        for phase in phases:
            for k, v in metrics.items():
                setattr(self, f"{phase}_{k}_metric", v())
        self.metrics = {
            phase: nn.ModuleDict(
                {k: getattr(self, f"{phase}_{k}_metric") for k, _ in metrics.items()}
            )
            for phase in phases
        }

        # note: for logging best-so-far validation metrics
        self.val_per_residue_mse_best = torchmetrics.MinMetric()
        self.val_per_residue_mae_best = torchmetrics.MinMetric()
        self.val_per_residue_pearson_corr_coef_best = torchmetrics.MaxMetric()
        self.val_per_model_mse_best = torchmetrics.MinMetric()
        self.val_per_model_mae_best = torchmetrics.MinMetric()
        self.val_per_model_pearson_corr_coef_best = torchmetrics.MaxMetric()

        setattr(self, f"{self.predict_phase}_step_outputs", [])

    @staticmethod
    def get_labels(batch: Any) -> Any:
        """Get labels from a batch.

        :param batch: A batch of data.
        :return: Labels from the batch.
        """
        if type(batch) in [list, tuple]:
            return batch[0].label
        return batch.label

    @beartype
    def forward(self, batch: Any) -> Tuple[Any, torch.Tensor]:
        """Make a forward pass through the model.

        :param batch: A batch of data.
        :return: A tuple containing the batch and the predictions.
        """
        # # correct residue-wise graph metadata for batch context
        # batch.ca_atom_idx, ca_atom_batch_index = convert_idx_from_batch_local_to_global(
        #     batch.ca_atom_idx, batch.batch, batch.num_graphs
        # )
        # batch.atom_residue_idx, _ = convert_idx_from_batch_local_to_global(
        #     batch.atom_residue_idx, ca_atom_batch_index, batch.num_graphs
        # )

        # # centralize node positions to make them translation-invariant
        # _, batch.x = centralize(batch, key="x", batch_index=batch.batch, node_mask=batch.mask)

        # # craft complete local frames corresponding to each edge
        # batch.f_ij = localize(
        #     batch.x,
        #     batch.edge_index,
        #     norm_x_diff=self.hparams.module_cfg.norm_x_diff,
        #     node_mask=batch.mask,
        # )

        # embed node and edge input
        batch.h = torch.cat((batch.h, self.atom_embedding(batch.atom_types)), dim=-1)
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi) = layer((h, chi), (e, xi), batch.edge_index, batch.f_ij, node_mask=batch.mask)

        # record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi = h, chi, e, xi

        # summarize intermediate node representations as final predictions
        out = self.invariant_node_projection[0]((h, chi))  # e.g., GCPLayerNorm()
        out = self.invariant_node_projection[1](
            out, batch.edge_index, batch.f_ij, node_inputs=True, node_mask=batch.mask
        )  # e.g., GCP((h, chi)) -> h'
        res_out = scatter(
            out[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean"
        )  # get batch-wise plDDT for each residue
        res_out = self.dense(res_out).squeeze()

        return batch, res_out

    def model_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take a step with the model.

        :param batch: A batch of data.
        :return: A tuple containing the loss, predictions, and labels.
        """
        # make a forward pass and score it
        labels = self.get_labels(batch)
        _, preds = self.forward(batch)
        loss = self.criterion(preds, labels)
        return loss, preds, labels

    def on_train_start(self):
        """Lightning hook that is called when the train begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_`metric`_best doesn't store any values from these checks
        self.val_loss.reset()
        self.metrics[self.val_phase]["PerResidueMSE"].reset()
        self.metrics[self.val_phase]["PerResidueMAE"].reset()
        self.metrics[self.val_phase]["PerResiduePearsonCorrCoef"].reset()
        self.metrics[self.val_phase]["PerModelMSE"].reset()
        self.metrics[self.val_phase]["PerModelMAE"].reset()
        self.metrics[self.val_phase]["PerModelPearsonCorrCoef"].reset()
        self.val_per_residue_mse_best.reset()
        self.val_per_residue_mae_best.reset()
        self.val_per_residue_pearson_corr_coef_best.reset()
        self.val_per_model_mse_best.reset()
        self.val_per_model_mae_best.reset()
        self.val_per_model_pearson_corr_coef_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Make a training step.

        :param batch: A batch of data.
        :param batch_idx: The batch index.
        :return: The loss.
        """
        # by default, do not skip the current batch
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting

        try:
            loss, preds, labels = self.model_step(batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                log.warning(
                    f"Ran out of memory in the forward pass. Skipping current training batch with index {batch_idx}"
                )
                if not torch_dist.is_initialized():
                    # NOTE: for skipping batches in a single-device setting
                    for p in self.net.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return None
                skip_flag = torch.ones((), device=self.device, dtype=torch.bool)
            else:
                raise e

        # NOTE: for skipping batches in a multi-device setting
        # credit: https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1553404417
        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                for p in self.net.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None

        # update metrics
        self.train_loss(loss.detach())

        # log per-residue metrics
        preds = preds.detach()
        for metric in self.metrics[self.train_phase].keys():
            if "residue" in metric.lower():
                self.metrics[self.train_phase][metric](preds, labels)

        # log per-model metrics
        ca_batch = scatter(
            batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean"
        ).long()  # get node-batch indices for Ca atoms
        preds_out = scatter(preds, ca_batch, dim=0, reduce="mean")  # get batch-wise global plDDT
        labels_out = scatter(labels, ca_batch, dim=0, reduce="mean")
        for metric in self.metrics[self.train_phase].keys():
            if "model" in metric.lower():
                self.metrics[self.train_phase][metric](preds_out, labels_out)

        return loss

    def on_train_epoch_end(self):
        """Lightning hook that is called when the train epoch ends."""
        # log metrics
        self.log(f"{self.train_phase}/loss", self.train_loss, prog_bar=False)
        for metric in self.metrics[self.train_phase].keys():
            self.log(
                f"{self.train_phase}/" + metric,
                self.metrics[self.train_phase][metric],
                metric_attribute=self.metrics[self.train_phase][metric],
                prog_bar=True,
            )

    def validation_step(self, batch: Any, batch_idx: int):
        """Make a validation step.

        :param batch: A batch of data.
        :param batch_idx: The batch index.
        """
        # by default, do not skip the current batch
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting

        try:
            loss, preds, labels = self.model_step(batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                log.warning(
                    f"Ran out of memory in the forward pass. Skipping current validation batch with index {batch_idx}"
                )
                if not torch_dist.is_initialized():
                    # NOTE: for skipping batches in a single-device setting
                    for p in self.net.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return None
                skip_flag = torch.ones((), device=self.device, dtype=torch.bool)
            else:
                raise e

        # NOTE: for skipping batches in a multi-device setting
        # credit: https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1553404417
        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                for p in self.net.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None

        # update metrics
        self.val_loss(loss.detach())

        # log per-residue metrics
        preds = preds.detach()
        for metric in self.metrics[self.val_phase].keys():
            if "residue" in metric.lower():
                self.metrics[self.val_phase][metric](preds, labels)

        # log per-model metrics
        ca_batch = scatter(
            batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean"
        ).long()  # get node-batch indices for Ca atoms
        preds_out = scatter(preds, ca_batch, dim=0, reduce="mean")  # get batch-wise global plDDT
        labels_out = scatter(labels, ca_batch, dim=0, reduce="mean")
        for metric in self.metrics[self.val_phase].keys():
            if "model" in metric.lower():
                self.metrics[self.val_phase][metric](preds_out, labels_out)

        return loss

    def on_validation_epoch_end(self):
        """Lightning hook that is called when the validation epoch ends."""
        # update best-so-far validation metrics according to the current epoch's results
        self.val_per_residue_mse_best.update(
            self.metrics[self.val_phase]["PerResidueMSE"].compute()
        )
        self.val_per_residue_mae_best.update(
            self.metrics[self.val_phase]["PerResidueMAE"].compute()
        )
        self.val_per_residue_pearson_corr_coef_best.update(
            self.metrics[self.val_phase]["PerResiduePearsonCorrCoef"].compute()
        )
        self.val_per_model_mse_best.update(self.metrics[self.val_phase]["PerModelMSE"].compute())
        self.val_per_model_mae_best.update(self.metrics[self.val_phase]["PerModelMAE"].compute())
        self.val_per_model_pearson_corr_coef_best.update(
            self.metrics[self.val_phase]["PerModelPearsonCorrCoef"].compute()
        )

        # log metrics
        self.log(f"{self.val_phase}/loss", self.val_loss, prog_bar=True)
        for metric in self.metrics[self.val_phase].keys():
            self.log(
                f"{self.val_phase}/" + metric,
                self.metrics[self.val_phase][metric],
                metric_attribute=self.metrics[self.val_phase][metric],
                prog_bar=True,
            )

        # log best-so-far metrics as a value through `.compute()` method, instead of as a metric object;
        # otherwise, metric would be reset by Lightning after each epoch
        # note: when logging as a value, set `sync_dist=True` for proper reduction over processes in DDP mode
        self.log(
            f"{self.val_phase}/PerResidueMSE_best",
            self.val_per_residue_mse_best.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{self.val_phase}/PerResidueMAE_best",
            self.val_per_residue_mae_best.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{self.val_phase}/PerResiduePearsonCorrCoef_best",
            self.val_per_residue_pearson_corr_coef_best.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{self.val_phase}/PerModelMSE_best",
            self.val_per_model_mse_best.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{self.val_phase}/PerModelMAE_best",
            self.val_per_model_mae_best.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{self.val_phase}/PerModelPearsonCorrCoef_best",
            self.val_per_model_pearson_corr_coef_best.compute(),
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        """Make a test step.

        :param batch: A batch of data.
        :param batch_idx: The batch index.
        """
        loss, preds, labels = self.model_step(batch)

        # update loss
        self.test_loss(loss.detach())

        # log per-residue metrics
        preds = preds.detach()
        for metric in self.metrics[self.test_phase].keys():
            if "residue" in metric.lower():
                self.metrics[self.test_phase][metric](preds, labels)

        # log per-model metrics
        ca_batch = scatter(
            batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean"
        ).long()  # get node-batch indices for Ca atoms
        preds_out = scatter(preds, ca_batch, dim=0, reduce="mean")  # get batch-wise global plDDT
        labels_out = scatter(labels, ca_batch, dim=0, reduce="mean")
        for metric in self.metrics[self.test_phase].keys():
            if "model" in metric.lower():
                self.metrics[self.test_phase][metric](preds_out, labels_out)

        return loss

    def on_test_epoch_end(self):
        """Lightning hook that is called when the test epoch ends."""
        # log metrics
        self.log(f"{self.test_phase}/loss", self.test_loss, prog_bar=False)
        for metric in self.metrics[self.test_phase].keys():
            self.log(
                f"{self.test_phase}/" + metric,
                self.metrics[self.test_phase][metric],
                metric_attribute=self.metrics[self.test_phase][metric],
                prog_bar=True,
            )

    def on_predict_epoch_start(self):
        """Lightning hook that is called when the predict epoch begins."""
        # configure loss function for inference to report loss values per batch element
        self.criterion = torch.nn.SmoothL1Loss(reduction="none")
        # define where the final predictions should be recorded
        self.predictions_csv_path = os.path.join(
            self.trainer.default_root_dir,
            f"{self.predict_phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_rank_{self.global_rank}_predictions.csv",
        )

    @torch.inference_mode()
    @beartype
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Make a predict step.

        :param batch: A batch of data.
        :param batch_idx: The batch index.
        :param dataloader_idx: The dataloader index.
        """
        batch.init_h = batch.h.clone()
        if hasattr(batch, "true_pdb_filepath") and all(batch.true_pdb_filepath):
            # note: currently, we can only score the loss for batches without any missing true (i.e., native) PDB files
            loss, preds, labels = self.model_step(batch)
        else:
            _, preds = self.forward(batch)
            loss, labels = None, None

        # collect per-model predictions
        batch.ca_batch = scatter(
            batch.batch[batch.mask], batch.atom_residue_idx[batch.mask], dim=0, reduce="mean"
        ).long()  # get node-batch indices for Ca atoms
        global_preds = scatter(
            preds, batch.ca_batch, dim=0, reduce="mean"
        )  # get batch-wise global plDDT

        if loss is not None:
            # get batch-wise global plDDT loss
            loss = scatter(loss, batch.ca_batch, dim=0, reduce="mean")
            # get initial residue-wise plDDT values from AlphaFold
            batch.initial_res_scores = scatter(
                batch.init_h[:, -1][batch.mask],
                batch.atom_residue_idx[batch.mask],
                dim=0,
                reduce="mean",
            )

        # collect outputs, and visualize predicted lDDT scores
        step_outputs = self.record_ema_preds(
            batch=batch, res_preds=preds, global_preds=global_preds, loss=loss, labels=labels
        )
        # collect outputs
        step_outputs_list = getattr(self, f"{self.predict_phase}_step_outputs")
        step_outputs_list.append(step_outputs)

    @torch.inference_mode()
    @beartype
    def on_predict_epoch_end(self):
        """Lightning hook that is called when the predict epoch ends."""
        step_outputs = getattr(self, f"{self.predict_phase}_step_outputs")
        # compile predictions collected by the current device (e.g., rank zero)
        predictions_csv_df = pd.DataFrame(step_outputs)
        predictions_csv_df.to_csv(self.predictions_csv_path, index=False)

    @torch.inference_mode()
    @beartype
    def record_ema_preds(
        self,
        batch: Any,
        res_preds: torch.Tensor,
        global_preds: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        plddt_scale_factor: float = MAX_PLDDT_VALUE,
    ) -> List[Dict[str, Any]]:
        """Record EMA predictions for protein structure models.

        :param batch: A batch of data.
        :param res_preds: Residue-wise predictions.
        :param global_preds: Global predictions.
        :param loss: The loss.
        :param labels: The labels.
        :param plddt_scale_factor: The scale factor for plDDT values.
        :return: A list of dictionaries containing the EMA predictions.
        """
        # create temporary output PDB files for predictions
        batch_metrics = []
        initial_res_scores = (
            batch.initial_res_scores.detach().cpu().numpy()
            if hasattr(batch, "initial_res_scores")
            else None
        )
        pred_res_scores = res_preds.detach().cpu().numpy()
        pred_global_scores = global_preds.detach().cpu().numpy()
        batch_loss = None if loss is None else loss.detach().cpu().numpy()
        batch_labels = None if labels is None else labels.detach().cpu().numpy()
        res_batch_index = batch.ca_batch.detach().cpu().numpy()
        for b_index in range(batch.num_graphs):
            metrics = {}
            temp_pdb_dir = tempfile._get_default_tempdir()
            temp_pdb_code = next(tempfile._get_candidate_names())
            initial_pdb_filepath = batch.decoy_pdb_filepath[b_index]
            prediction_path = str(
                temp_pdb_dir / Path(f"predicted_{temp_pdb_code}").with_suffix(".pdb")
            )
            true_path = str(temp_pdb_dir / Path(f"true_{temp_pdb_code}").with_suffix(".pdb"))
            # isolate each individual example within the current batch
            if initial_res_scores is not None:
                initial_res_scores_ = (
                    initial_res_scores[res_batch_index == b_index] * plddt_scale_factor
                )
            pred_res_scores_ = pred_res_scores[res_batch_index == b_index] * plddt_scale_factor
            pred_global_score_ = pred_global_scores[b_index] * plddt_scale_factor
            loss_ = np.nan if batch_loss is None else batch_loss[b_index]
            labels_ = (
                None
                if batch_labels is None
                else batch_labels[res_batch_index == b_index] * plddt_scale_factor
            )
            annotate_pdb_with_new_column_values(
                input_pdb_filepath=initial_pdb_filepath,
                output_pdb_filepath=prediction_path,
                column_name="b_factor",
                new_column_values=pred_res_scores_,
            )
            if labels_ is not None and initial_res_scores is not None:
                annotate_pdb_with_new_column_values(
                    input_pdb_filepath=initial_pdb_filepath,
                    output_pdb_filepath=true_path,
                    column_name="b_factor",
                    new_column_values=labels_,
                )
                initial_per_res_plddt_ae = (
                    np.abs(initial_res_scores_ - labels_).mean() / plddt_scale_factor
                )
                pred_per_res_plddt_ae = (
                    np.abs(pred_res_scores_ - labels_).mean() / plddt_scale_factor
                )
            else:
                true_path = None
                initial_per_res_plddt_ae = None
                pred_per_res_plddt_ae = None
            metrics["input_annotated_pdb_filepath"] = initial_pdb_filepath
            metrics["predicted_annotated_pdb_filepath"] = prediction_path
            metrics["true_annotated_pdb_filepath"] = true_path
            metrics["global_plddt"] = pred_global_score_
            metrics["plddt_loss"] = loss_
            metrics["input_per_residue_plddt_absolute_error"] = initial_per_res_plddt_ae
            metrics["predicted_per_residue_plddt_absolute_error"] = pred_per_res_plddt_ae
            batch_metrics.append(metrics)
        return batch_metrics

    @beartype
    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any):
        """Overrides Lightning's `backward` step to add an out-of-memory (OOM) check."""
        # by default, do not skip the current batch
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting

        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                log.warning(f"Ran out of memory in the backward pass. Skipping batch due to: {e}")
                if not torch_dist.is_initialized():
                    # NOTE: for skipping batches in a single-device setting
                    for p in self.net.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return None
                skip_flag = torch.ones((), device=self.device, dtype=torch.bool)
            else:
                raise e

        # NOTE: for skipping batches in a multi-device setting
        # credit: https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1553404417
        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                for p in self.net.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()

    def setup(self, stage: str):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        try:
            optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        except TypeError:
            # NOTE: strategies such as DeepSpeed require `params` to be specified as `model_params`
            optimizer = self.hparams.optimizer(model_params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_fit_end(self):
        """Lightning calls this upon completion of the user's call to `trainer.fit()` for model
        training.

        For example, Lightning will call this hook upon exceeding `trainer.max_epochs` in model training.
        """
        if self.trainer.is_global_zero:
            path_cfg = self.hparams.path_cfg
            if path_cfg is not None and path_cfg.grid_search_script_dir is not None:
                # uniquely record when model training is concluded
                grid_search_script_dir = self.hparams.path_cfg.grid_search_script_dir
                run_id = self.logger.experiment.id
                fit_end_indicator_filename = f"{run_id}.{HALT_FILE_EXTENSION}"
                fit_end_indicator_filepath = os.path.join(
                    grid_search_script_dir, fit_end_indicator_filename
                )
                os.makedirs(grid_search_script_dir, exist_ok=True)
                with open(fit_end_indicator_filepath, "w") as f:
                    f.write("`on_fit_end` has been called.")
        return super().on_fit_end()


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gcpnet_ema.yaml")
    _ = hydra.utils.instantiate(cfg)
