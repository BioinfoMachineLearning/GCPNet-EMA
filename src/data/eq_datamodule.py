# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import torch
import torch_geometric
import pytorch_lightning as pl

from beartype import beartype
from typing import List, Optional, Dict, Any

from src.data.components.ema_dataset import EMADataset


class EMADataModule(pl.LightningDataModule):
    def __init__(
            self,
            splits_dir: str = os.path.join("data", "EMA", "splits"),
            decoy_dir: str = os.path.join("data", "EMA", "decoy_model"),
            true_dir: str = os.path.join("data", "EMA", "true_model"),
            model_data_cache_dir: str = os.path.join("data", "EMA", "model_data_cache"),
            edge_cutoff: float = 4.5,
            max_neighbors: int = 32,
            rbf_edge_dist_cutoff: float = 4.5,
            num_rbf: int = 16,
            python_exec_path: Optional[str] = None,
            lddt_exec_path: Optional[str] = None,
            pdbtools_dir: Optional[str] = None,
            subset_to_ca_atoms_only: bool = False,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = True,
            # arguments for model inference
            predict_input_dir: str = os.path.join("data", "EMA", "examples", "decoy_model"),
            predict_true_dir: Optional[str] = os.path.join("data", "EMA", "examples", "true_model"),
            predict_output_dir: str = os.path.join("data", "EMA", "examples", "outputs"),
            predict_batch_size: int = 1,
            predict_pin_memory: bool = True,
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # features - ESM protein sequence embeddings #
        self.esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.esm_model = self.esm_model.eval().cpu()
        self.esm_batch_converter = esm_alphabet.get_batch_converter()

    @staticmethod
    @beartype
    def parse_split_pdbs(
        decoy_dir: str,
        true_dir: str,
        splits_dir: str,
        split_filename: str
    ) -> List[Dict[str, Any]]:
        split_entries = []
        with open(os.path.join(splits_dir, split_filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip().split(" ")
                target = line[0]
                split_entries.append({
                    "decoy_pdb": os.path.join(decoy_dir, f"{target}.pdb"),
                    "true_pdb": os.path.join(true_dir, f"{target}.pdb")
                })
        return split_entries
    
    @staticmethod
    @beartype
    def parse_inference_pdbs(
        decoy_dir: str,
        true_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        split_entries = []
        for item in os.listdir(decoy_dir):
            decoy_pdb_filepath = os.path.join(decoy_dir, item)
            true_pdb_filepath = (
                os.path.join(true_dir, item)
                if true_dir and os.path.exists(os.path.join(true_dir, item))
                else None
            )
            split_entries.append({
                "decoy_pdb": decoy_pdb_filepath,
                "true_pdb": true_pdb_filepath
            })
        return split_entries

    def setup(self, stage: Optional[str] = None):
        train_pdbs = self.parse_split_pdbs(
            self.hparams.decoy_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            "train.lst"
        )
        valid_pdbs = self.parse_split_pdbs(
            self.hparams.decoy_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            "valid.lst"
        )
        test_pdbs = self.parse_split_pdbs(
            self.hparams.decoy_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            "test.lst"
        )
        predict_pdbs = self.parse_inference_pdbs(
            decoy_dir=self.hparams.predict_input_dir,
            true_dir=self.hparams.predict_true_dir
        )

        if stage in ["predict"]:
            assert len(predict_pdbs) > 0, "PDB inputs must be provided during model inference."

        self.data_train = EMADataset(
            decoy_pdbs=train_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            edge_cutoff=self.hparams.edge_cutoff,
            max_neighbors=self.hparams.max_neighbors,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            lddt_exec_path=self.hparams.lddt_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            subset_to_ca_atoms_only=self.hparams.subset_to_ca_atoms_only,
        )
        self.data_val = EMADataset(
            decoy_pdbs=valid_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            edge_cutoff=self.hparams.edge_cutoff,
            max_neighbors=self.hparams.max_neighbors,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            lddt_exec_path=self.hparams.lddt_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            subset_to_ca_atoms_only=self.hparams.subset_to_ca_atoms_only,
        )
        self.data_test = EMADataset(
            decoy_pdbs=test_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            edge_cutoff=self.hparams.edge_cutoff,
            max_neighbors=self.hparams.max_neighbors,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            lddt_exec_path=self.hparams.lddt_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            subset_to_ca_atoms_only=self.hparams.subset_to_ca_atoms_only,
        )
        self.data_predict = EMADataset(
            decoy_pdbs=predict_pdbs,
            model_data_cache_dir=self.hparams.predict_output_dir,
            edge_cutoff=self.hparams.edge_cutoff,
            max_neighbors=self.hparams.max_neighbors,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            lddt_exec_path=self.hparams.lddt_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            subset_to_ca_atoms_only=self.hparams.subset_to_ca_atoms_only,
        )

    @beartype
    def get_dataloader(
        self,
        dataset: EMADataset,
        batch_size: int,
        pin_memory: bool,
        shuffle: bool,
        drop_last: bool
    ) -> torch_geometric.loader.DataLoader:
        return torch_geometric.loader.DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last
        )

    def train_dataloader(self):
        return self.get_dataloader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return self.get_dataloader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self):
        return self.get_dataloader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False
        )
    
    def predict_dataloader(self):
        return self.get_dataloader(
            self.data_predict,
            batch_size=self.hparams.predict_batch_size,
            pin_memory=self.hparams.predict_pin_memory,
            shuffle=False,
            drop_last=False
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "ema.yaml")
    _ = hydra.utils.instantiate(cfg)
