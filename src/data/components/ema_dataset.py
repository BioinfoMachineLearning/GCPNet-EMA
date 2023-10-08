# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import re
import subprocess  # nosec
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import prody as pr
import torch
import torch.nn as nn
import torch_cluster
from beartype import beartype
from biopandas.pdb import PandasPdb
from jaxtyping import Float, Int64
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
from sidechainnet.utils.measure import get_seq_coords_and_angles
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.components.helper import _normalize, _rbf
from src.data.components.protein_graph_dataset import ProteinGraphDataset
from src.utils.pylogger import get_pylogger

pr.confProDy(verbosity="none")


log = get_pylogger(__name__)


ALPHABET = [
    "#",
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
ATOM_TYPES = [
    "",
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
RES_ATOM14 = [
    [""] * 14,
    ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
]
ONE_TO_THREE_LETTER_MAP = {
    "#": "UNK",
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
NUM_COORDINATES_PER_RESIDUE = NUM_COORDS_PER_RES
MAX_PLDDT_VALUE = 100


@beartype
def _edge_features(
    coords: Float[torch.Tensor, "num_nodes 3"],  # noqa: F722
    edge_index: Int64[torch.Tensor, "2 num_edges"],  # noqa: F722
    scalar_edge_feats: Float[torch.Tensor, "num_edges num_edge_scalar_features"],  # noqa: F722
    D_max: float = 4.5,
    num_rbf: int = 16,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[
    Float[torch.Tensor, "num_edges num_cat_edge_scalar_features"],  # noqa: F722
    Float[torch.Tensor, "num_edges num_edge_vector_features", 3],  # noqa: F722
]:
    """Builds edge features for a geometric graph.

    :param coords: (num_nodes, 3) tensor of node coordinates
    :param edge_index: (2, num_edges) tensor of edge indices
    :param scalar_edge_feats: (num_edges, num_edge_scalar_features) tensor of scalar edge features
    :param D_max: maximum distance for radial basis functions
    :param num_rbf: number of radial basis functions
    :param device: device to use for tensor operations
    :return: (num_edges, num_cat_edge_scalar_features) tensor of concatenated edge features
    """
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)

    edge_s = torch.cat((scalar_edge_feats, rbf), dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


@beartype
def _node_features(
    coords: Float[torch.Tensor, "num_nodes 3"]  # noqa: F722
) -> Float[torch.Tensor, "num_nodes num_cat_node_vector_features 3"]:  # noqa: F722
    """Node features for a geometric graph.

    :param coords: (num_nodes, 3) tensor of node coordinates
    :return: (num_nodes, num_cat_node_vector_features, 3) tensor of concatenated node features
    """
    node_v = ProteinGraphDataset._orientations(coords)
    return node_v


@beartype
def merge_pdb_chains_within_output_file(
    input_pdb_filepath,
    output_pdb_filepath,
    new_chain_id: str = "A",
    new_start_index: int = 1,
    python_exec_path: Optional[str] = None,
    pdbtools_dir: Optional[str] = None,
):
    """Merges chains within a PDB file into a single chain, and renumbers the atoms within the
    merged chain.

    :param input_pdb_filepath: path to input PDB file
    :param output_pdb_filepath: path to output PDB file
    :param new_chain_id: new chain ID to assign to the merged chain
    :param new_start_index: new starting index to assign to the merged chain
    :param python_exec_path: path to a Python executable (i.e., binary) file
    :param pdbtools_dir: path to the `pdb-tools` directory
    :return: None
    """
    with open(input_pdb_filepath) as f:
        x = f.readlines()
    filtered = [i for i in x if re.match(r"^ATOM.+", i)]
    chains = list({i[21] for i in x if re.match(r"^ATOM.+", i)})
    chains.sort()
    with open(output_pdb_filepath + ".tmp", "w") as f:
        f.writelines(filtered)
    if pdbtools_dir is not None:
        assert (
            python_exec_path is not None
        ), "Path to a Python executable (i.e., binary) file must be provided to run individual scripts within `pdb-tools`."
    pdb_selchain_cmd = (
        f"{python_exec_path} {os.path.join(pdbtools_dir, 'pdb_selchain.py')}"
        if pdbtools_dir is not None
        else "pdb_selchain"
    )
    pdb_chain_cmd = (
        f"{python_exec_path} {os.path.join(pdbtools_dir, 'pdb_chain.py')}"
        if pdbtools_dir is not None
        else "pdb_chain"
    )
    pdb_reres_cmd = (
        f"{python_exec_path} {os.path.join(pdbtools_dir, 'pdb_reres.py')}"
        if pdbtools_dir is not None
        else "pdb_reres"
    )
    merge_cmd = f"{pdb_selchain_cmd} -{','.join(chains)} {output_pdb_filepath + '.tmp'} | {pdb_chain_cmd} -{new_chain_id} | {pdb_reres_cmd} -{new_start_index} > {output_pdb_filepath}"
    try:
        subprocess.run(args=merge_cmd, shell=True)
    except Exception as e:
        log.error(
            f"Unable to chain-merge PDB file {input_pdb_filepath} via pdb-tools. Please investigate the validity of this input file."
        )
        raise e
    os.remove(output_pdb_filepath + ".tmp")


@beartype
def generate_lddt_score(
    input_model_filepath: str,
    target_model_filepath: str,
    lddt_exec_path: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Generates a lDDT score for a given input model.

    :param input_model_filepath: path to input model
    :param target_model_filepath: path to target model
    :param lddt_exec_path: path to the lDDT executable
    :param verbose: whether to log messages
    :return: lDDT score
    """
    # note: if a path to an executable is not provided, it is assumed that lDDT has already been installed within e.g., the currently-activated Conda environment
    lddt_cmd = lddt_exec_path if lddt_exec_path is not None else "lddt"
    proc = subprocess.Popen(
        [lddt_cmd, input_model_filepath, target_model_filepath], stdout=subprocess.PIPE
    )  # nosec
    output = proc.stdout.read().decode("utf-8")
    try:
        df = pd.read_csv(StringIO(output), sep="\t", skiprows=10)
        score = df["Score"].to_numpy()
        if score.dtype.name == "object":
            score[score == "-"] = -1
    except Exception as e:
        if verbose:
            log.error(f"Skipping calculation of lDDT score for {input_model_filepath} due to: {e}")
        score = np.array([-1])
    score = score.astype(np.float32)
    return score


class EMADataset(Dataset):
    def __init__(
        self,
        decoy_pdbs: List[Dict[str, Any]],
        model_data_cache_dir: str,
        edge_cutoff: float,
        max_neighbors: int,
        rbf_edge_dist_cutoff: float,
        num_rbf: int,
        esm_model: Optional[Any] = None,
        esm_batch_converter: Optional[Any] = None,
        python_exec_path: Optional[str] = None,
        lddt_exec_path: Optional[str] = None,
        pdbtools_dir: Optional[str] = None,
        subset_to_ca_atoms_only: bool = False,
    ):
        self.decoy_pdbs = decoy_pdbs
        self.model_data_cache_dir = model_data_cache_dir
        self.edge_cutoff = edge_cutoff
        self.max_neighbors = max_neighbors
        self.rbf_edge_dist_cutoff = rbf_edge_dist_cutoff
        self.num_rbf = num_rbf
        self.esm_model = esm_model
        self.esm_batch_converter = esm_batch_converter
        self.python_exec_path = python_exec_path
        self.lddt_exec_path = lddt_exec_path
        self.pdbtools_dir = pdbtools_dir
        self.subset_to_ca_atoms_only = subset_to_ca_atoms_only
        self.num_pdbs = len(self.decoy_pdbs)

        os.makedirs(self.model_data_cache_dir, exist_ok=True)

    def __len__(self):
        """Return the number of PDBs in the dataset."""
        return self.num_pdbs

    @beartype
    def __getitem__(self, idx: int) -> Data:
        """Get the data for a single PDB.

        :param idx: index of the PDB to retrieve
        :return: data for a single PDB
        """
        return self._featurize_as_graph(self.decoy_pdbs[idx])

    @staticmethod
    @beartype
    def _safe_index_of_value(
        alphabet: List[str],
        value: str,
        default_value: str = "#",
        granularity: str = "residue",
        verbose: bool = True,
    ) -> int:
        """Safely identify the index of a value within a given alphabet.

        :param alphabet: alphabet to search
        :param value: value to search for
        :param default_value: default value to return if the value is not found
        :param granularity: granularity of the value to search for
        :param verbose: whether to log messages
        :return: index of the value within the alphabet
        """
        try:
            return alphabet.index(value)
        except ValueError:
            if verbose:
                log.info(
                    f"PDBDataset: Unable to identify {granularity} type value {value} -> defaulting to {granularity} type value '{default_value}'"
                )
            return alphabet.index(default_value)

    @staticmethod
    @beartype
    def _extract_protein_features(
        esm_model: Optional[nn.Module] = None,
        esm_batch_converter: Optional[Any] = None,
        decoy_protein_pdb_filepath: Optional[Path] = None,
        true_protein_pdb_filepath: Optional[Path] = None,
        protein_data: Optional[Any] = None,
        protein_data_filepath: Optional[Path] = None,
        protein_chain_ids: Optional[List[str]] = None,
        python_exec_path: Optional[str] = None,
        lddt_exec_path: Optional[str] = None,
        pdbtools_dir: Optional[str] = None,
        cache_processed_data: bool = True,
        force_process_data: bool = False,
    ) -> Tuple[Data, Optional[pr.AtomGroup]]:
        """Extracts protein features from a PDB file.

        :param esm_model: ESM model
        :param esm_batch_converter: ESM batch converter
        :param decoy_protein_pdb_filepath: path to decoy protein PDB file
        :param true_protein_pdb_filepath: path to true protein PDB file
        :param protein_data: protein data
        :param protein_data_filepath: path to protein data
        :param protein_chain_ids: chain IDs of the protein
        :param python_exec_path: path to a Python executable (i.e., binary) file
        :param lddt_exec_path: path to the lDDT executable
        :param pdbtools_dir: path to the `pdb-tools` directory
        :param cache_processed_data: whether to cache processed data
        :param force_process_data: whether to force processing of data
        :return: protein data
        """
        protein = None
        if protein_data is None or force_process_data:
            # gather protein features #
            assert (
                len(protein_chain_ids) > 0
            ), "There must be at least one chain ID present in each input PDB to enable PDB processing."

            protein_chain_index = 0
            (
                protein_coords,
                protein_sequences,
                protein_atom_types,
                protein_atom_chain_indices,
                protein_bfactors,
            ) = ([] for _ in range(5))
            for protein_chain_id in protein_chain_ids:
                if len(protein_chain_id) == 0:
                    assert (
                        len(protein_chain_ids) == 1
                    ), f"To impute a missing chain ID for {decoy_protein_pdb_filepath}, there must be only a single input PDB chain."

                # within each input protein, isolate the heavy atoms (i.e., the side-chain and Ca atoms) and their corresponding amino acid sequence
                protein_chain = pr.parsePDB(
                    str(decoy_protein_pdb_filepath),
                    chain=protein_chain_id if len(protein_chain_id) > 0 else "A",
                    model=1,
                )
                _, protein_chain_coords, protein_chain_sequence, _, _ = get_seq_coords_and_angles(
                    protein_chain
                )

                # collect side-chain atom coordinates corresponding to heavy atoms
                protein_coords.append(torch.from_numpy(protein_chain_coords))

                # collect protein sequences by chain ID
                protein_sequences.append(("", protein_chain_sequence))

                # collect the type of each side-chain heavy atom
                protein_atom_types.append(
                    torch.tensor(
                        [
                            [
                                ATOM_TYPES.index(a)
                                for a in RES_ATOM14[EMADataset._safe_index_of_value(ALPHABET, s)]
                            ]
                            for s in protein_chain_sequence
                        ]
                    )
                )

                # collect chain index of each atom
                protein_atom_chain_indices.append(
                    torch.tensor(
                        [
                            [
                                protein_chain_index
                                for _ in RES_ATOM14[EMADataset._safe_index_of_value(ALPHABET, s)]
                            ]
                            for s in protein_chain_sequence
                        ]
                    )
                )

                # collect beta-factors of each residue
                protein_bfactors.append(
                    torch.from_numpy(protein_chain.select("name CA").getBetas()) / MAX_PLDDT_VALUE
                )

                protein_chain_index += 1

            # derive node mask to denote "missing" side-chain atoms
            protein_coords = torch.cat(protein_coords, dim=0).float()
            protein_mask = (protein_coords.norm(dim=-1) > 1e-6).bool()
            assert not protein_coords.isnan().any(), "NaN protein coordinates must not be present."

            # encode atom types corresponding to each residue
            protein_atom_types = torch.cat(protein_atom_types, dim=0).view(-1)
            protein_atom_types *= protein_mask.long()

            # encode chain indices corresponding to each atom
            protein_atom_chain_indices = torch.cat(protein_atom_chain_indices, dim=0).view(-1)

            # collate AlphaFold's plDDT values for each residue
            af2_plddt_per_residue = torch.cat(protein_bfactors, dim=0).float()

            # identify indices of Ca atoms
            protein_ca_atom_idx = torch.where(protein_atom_types == 2)[0]

            # add ESM sequence embeddings as scalar atom features that are shared between atoms of the same residue
            batch_tokens = esm_batch_converter(protein_sequences)[2]
            with torch.inference_mode():
                results = esm_model(batch_tokens, repr_layers=[esm_model.num_layers])
            token_representations = results["representations"][esm_model.num_layers].cpu()
            protein_atom_representations = []
            for i, (_, protein_sequence) in enumerate(protein_sequences):
                representations = token_representations[i, 1 : len(protein_sequence) + 1]
                protein_atom_representations.append(representations)
            protein_atom_representations = torch.cat(protein_atom_representations, dim=0)
            assert protein_atom_representations.size(0) * NUM_COORDINATES_PER_RESIDUE == len(
                protein_coords
            ), "Number of side-chain atoms must match."

            # associate atoms belonging to the same residue using unique atom-residue indices
            total_num_residues = sum([len(s[1]) for s in protein_sequences])
            protein_atom_residue_idx = torch.arange(
                start=0, end=total_num_residues
            ).repeat_interleave(NUM_COORDINATES_PER_RESIDUE, 0)
            assert protein_atom_residue_idx.size(0) == len(
                protein_coords
            ), "Number of atom-residue indices must match number of atoms."

            lddt_per_residue = None
            if true_protein_pdb_filepath is not None:
                is_multi_chain_true_protein = len(protein_chain_ids) > 1
                if is_multi_chain_true_protein:
                    tmp_decoy_protein_pdb_filepath = (
                        Path(tempfile._get_default_tempdir())
                        / Path(next(tempfile._get_candidate_names()))
                    ).with_suffix(".pdb")
                    tmp_true_protein_pdb_filepath = (
                        Path(tempfile._get_default_tempdir())
                        / Path(next(tempfile._get_candidate_names()))
                    ).with_suffix(".pdb")
                    merge_pdb_chains_within_output_file(
                        decoy_protein_pdb_filepath,
                        str(tmp_decoy_protein_pdb_filepath),
                        python_exec_path=python_exec_path,
                        pdbtools_dir=pdbtools_dir,
                    )
                    merge_pdb_chains_within_output_file(
                        true_protein_pdb_filepath,
                        str(tmp_true_protein_pdb_filepath),
                        python_exec_path=python_exec_path,
                        pdbtools_dir=pdbtools_dir,
                    )
                    lddt_per_residue = generate_lddt_score(
                        str(tmp_decoy_protein_pdb_filepath),
                        str(tmp_true_protein_pdb_filepath),
                        lddt_exec_path=lddt_exec_path,
                    )
                else:
                    lddt_per_residue = generate_lddt_score(
                        str(decoy_protein_pdb_filepath),
                        str(true_protein_pdb_filepath),
                        lddt_exec_path=lddt_exec_path,
                    )

            # save protein data #
            protein_data = Data(
                protein_atom_rep=protein_atom_representations,
                protein_atom_types=protein_atom_types,
                protein_atom_chain_indices=protein_atom_chain_indices,
                protein_atom_residue_idx=protein_atom_residue_idx,
                protein_ca_atom_idx=protein_ca_atom_idx,
                protein_x=protein_coords,
                protein_mask=protein_mask,
                protein_num_atoms=torch.tensor([len(protein_coords)]),
                protein_decoy_alphafold_per_residue_plddt=af2_plddt_per_residue,
                protein_decoy_per_residue_lddt=torch.from_numpy(lddt_per_residue)
                if lddt_per_residue is not None
                else None,
            )
            if cache_processed_data:
                torch.save(protein_data, str(protein_data_filepath))

        # convert ESM residue-wise embeddings and AlphaFold plDDTs into atom-wise embeddings at runtime to reduce storage requirements
        protein_data.protein_atom_rep = protein_data.protein_atom_rep.repeat_interleave(
            NUM_COORDINATES_PER_RESIDUE, 0
        )
        protein_data.protein_decoy_alphafold_per_residue_plddt = (
            protein_data.protein_decoy_alphafold_per_residue_plddt.repeat_interleave(
                NUM_COORDINATES_PER_RESIDUE, 0
            )
        )
        return protein_data, protein

    @staticmethod
    @beartype
    def _prot_to_data(
        esm_model: Optional[nn.Module] = None,
        esm_batch_converter: Optional[Any] = None,
        decoy_protein_pdb_filepath: Optional[Path] = None,
        true_protein_pdb_filepath: Optional[Path] = None,
        protein_data: Optional[Any] = None,
        protein_data_filepath: Optional[Path] = None,
        protein_chain_ids: Optional[List[str]] = None,
        python_exec_path: Optional[str] = None,
        lddt_exec_path: Optional[str] = None,
        pdbtools_dir: Optional[str] = None,
        cache_processed_data: bool = True,
        force_process_data: bool = False,
    ) -> Tuple[Data, Optional[pr.AtomGroup]]:
        """Processes a protein PDB file into a `Data` collection.

        :param esm_model: ESM model
        :param esm_batch_converter: ESM batch converter
        :param decoy_protein_pdb_filepath: path to decoy protein PDB file
        :param true_protein_pdb_filepath: path to true protein PDB file
        :param protein_data: protein data
        :param protein_data_filepath: path to protein data
        :param protein_chain_ids: chain IDs of the protein
        :param python_exec_path: path to a Python executable (i.e., binary) file
        :param lddt_exec_path: path to the lDDT executable
        :param pdbtools_dir: path to the `pdb-tools` directory
        :param cache_processed_data: whether to cache processed data
        :param force_process_data: whether to force processing of data
        :return: protein data
        """
        protein_data, protein = EMADataset._extract_protein_features(
            esm_model,
            esm_batch_converter,
            decoy_protein_pdb_filepath=decoy_protein_pdb_filepath,
            true_protein_pdb_filepath=true_protein_pdb_filepath,
            protein_data=protein_data,
            protein_data_filepath=protein_data_filepath,
            protein_chain_ids=protein_chain_ids,
            python_exec_path=python_exec_path,
            lddt_exec_path=lddt_exec_path,
            pdbtools_dir=pdbtools_dir,
            cache_processed_data=cache_processed_data,
            force_process_data=force_process_data,
        )

        # organize protein metadata and features into a single `Data` collection #
        data = Data(
            atom_rep=protein_data.protein_atom_rep,
            atom_types=protein_data.protein_atom_types,
            atom_chain_indices=protein_data.protein_atom_chain_indices,
            atom_residue_idx=protein_data.protein_atom_residue_idx,
            ca_atom_idx=protein_data.protein_ca_atom_idx,
            x=protein_data.protein_x,
            mask=protein_data.protein_mask,
            num_atoms=protein_data.protein_num_atoms,
            decoy_protein_alphafold_per_residue_plddt=protein_data.protein_decoy_alphafold_per_residue_plddt,
            decoy_protein_per_residue_lddt=getattr(
                protein_data, "protein_decoy_per_residue_lddt", None
            ),
        )
        return data, protein

    @staticmethod
    @beartype
    def finalize_graph_topology_and_features_within_data(
        data: Data,
        edge_cutoff: float = 4.5,
        max_neighbors: int = 32,
        rbf_edge_dist_cutoff: float = 4.5,
        num_rbf: int = 16,
    ) -> Data:
        """Finalizes the graph topology and features within a `Data` collection.

        :param data: `Data` collection
        :param edge_cutoff: edge cutoff distance
        :param max_neighbors: maximum number of neighbors
        :param rbf_edge_dist_cutoff: radial basis function edge distance cutoff
        :param num_rbf: number of radial basis functions
        :return: `Data` collection
        """
        # design graph topology
        edge_index = torch_cluster.radius_graph(
            data.x, r=edge_cutoff, max_num_neighbors=max_neighbors
        )

        # build geometric graph features
        h = torch.cat(
            (data.atom_rep, data.decoy_protein_alphafold_per_residue_plddt.unsqueeze(-1)), dim=-1
        )
        chi = _node_features(data.x)

        edge_chain_encodings = (
            (
                # note: atom pairs from the same chain are assigned a value of 1.0; all others are assigned 0.0
                data.atom_chain_indices[edge_index[0]]
                == data.atom_chain_indices[edge_index[1]]
            )
            .float()
            .unsqueeze(-1)
        )
        edge_ca_atom_encodings = (
            (
                # note: atom pairs from the same residue are assigned a value of 1.0; all others are assigned 0.0
                data.atom_residue_idx[edge_index[0]]
                == data.atom_residue_idx[edge_index[1]]
            )
            .float()
            .unsqueeze(-1)
        )
        edge_encodings = torch.cat((edge_chain_encodings, edge_ca_atom_encodings), dim=-1)
        e, xi = _edge_features(
            data.x,
            edge_index,
            scalar_edge_feats=edge_encodings,
            D_max=rbf_edge_dist_cutoff,
            num_rbf=num_rbf,
        )

        # standardize graph features
        standardized_data = Data(
            h=h,
            chi=chi,
            e=e,
            xi=xi,
            x=data.x,
            label=getattr(data, "decoy_protein_per_residue_lddt", None),
            edge_index=edge_index,
            mask=data.mask,
            atom_types=data.atom_types,
            atom_chain_indices=data.atom_chain_indices,
            atom_residue_idx=data.atom_residue_idx,
            ca_atom_idx=data.ca_atom_idx,
        )

        return standardized_data

    @staticmethod
    @beartype
    def subset_data_to_ca_atoms_only(
        data: Data,
        edge_cutoff: float = 8.0,
        max_neighbors: int = 128,
        rbf_edge_dist_cutoff: float = 8.0,
        num_rbf: int = 16,
    ) -> Data:
        """Subsets a `Data` collection to only the Ca atoms available for each residue.

        :param data: `Data` collection
        :param edge_cutoff: edge cutoff distance
        :param max_neighbors: maximum number of neighbors
        :param rbf_edge_dist_cutoff: radial basis function edge distance cutoff
        :param num_rbf: number of radial basis functions
        :return: `Data` collection
        """
        # reduce the nodes (and edges) in `data` to only the Ca atoms available for each residue
        nodes_to_keep_idx = data.ca_atom_idx
        nodes_to_keep_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        nodes_to_keep_mask[nodes_to_keep_idx] = 1
        data.x = data.x[nodes_to_keep_mask]
        data.h = data.h[nodes_to_keep_mask]
        data.mask = data.mask[nodes_to_keep_mask]
        data.atom_types = data.atom_types[nodes_to_keep_mask]
        data.atom_chain_indices = data.atom_chain_indices[nodes_to_keep_mask]
        data.chi = _node_features(
            data.x
        )  # note: `chi` relies on the sequence of atoms, so it must be recomputed
        data.atom_residue_idx = torch.arange(
            0, len(data.ca_atom_idx), device=data.ca_atom_idx.device, dtype=torch.long
        )
        data.ca_atom_idx = data.atom_residue_idx
        # redesign graph topology using Ca atom positions
        data.edge_index = torch_cluster.radius_graph(
            data.x, r=edge_cutoff, max_num_neighbors=max_neighbors
        )
        edge_chain_encodings = (
            (
                # note: atom pairs from the same chain are assigned a value of 1.0; all others are assigned 0.0
                data.atom_chain_indices[data.edge_index[0]]
                == data.atom_chain_indices[data.edge_index[1]]
            )
            .float()
            .unsqueeze(-1)
        )
        edge_ca_atom_encodings = (
            (
                # note: atom pairs from the same residue are assigned a value of 1.0; all others are assigned 0.0
                data.atom_residue_idx[data.edge_index[0]]
                == data.atom_residue_idx[data.edge_index[1]]
            )
            .float()
            .unsqueeze(-1)
        )
        edge_encodings = torch.cat((edge_chain_encodings, edge_ca_atom_encodings), dim=-1)
        data.e, data.xi = _edge_features(
            data.x,
            data.edge_index,
            scalar_edge_feats=edge_encodings,
            D_max=rbf_edge_dist_cutoff,
            num_rbf=num_rbf,
        )
        return data

    @beartype
    def _featurize_as_graph(
        self,
        pdb_filename_dict: Dict[str, Any],
        atom_df_name: str = "ATOM",
        chain_id_col: str = "chain_id",
    ) -> Data:
        """Featurizes a PDB file as a graph.

        :param pdb_filename_dict: dictionary of PDB filenames
        :param atom_df_name: name of the atom DataFrame
        :param chain_id_col: name of the chain ID column
        :return: data for a single PDB
        """
        decoy_pdb_path = Path(pdb_filename_dict["decoy_pdb"])
        true_pdb_path = (
            None if pdb_filename_dict["true_pdb"] is None else Path(pdb_filename_dict["true_pdb"])
        )
        pdb_id = decoy_pdb_path.stem

        protein_data_filepath = Path(self.model_data_cache_dir) / f"{pdb_id}.pt"
        protein_data = (
            torch.load(protein_data_filepath)
            if os.path.exists(str(protein_data_filepath))
            else None
        )

        protein_chain_ids = (
            # note: the current version of Pandas ensures that `unique()` preserves ordering in a `pd.Series`
            PandasPdb()
            .read_pdb(str(decoy_pdb_path))
            .df[atom_df_name][chain_id_col]
            .unique()
            .tolist()
            if protein_data is None
            else None
        )

        data, _ = self._prot_to_data(
            esm_model=self.esm_model,
            esm_batch_converter=self.esm_batch_converter,
            decoy_protein_pdb_filepath=decoy_pdb_path,
            true_protein_pdb_filepath=true_pdb_path,
            protein_data=protein_data,
            protein_data_filepath=protein_data_filepath,
            protein_chain_ids=protein_chain_ids,
            python_exec_path=self.python_exec_path,
            lddt_exec_path=self.lddt_exec_path,
            pdbtools_dir=self.pdbtools_dir,
        )

        # finalize graph features according to current graph topology and model specifications
        data = self.finalize_graph_topology_and_features_within_data(
            data,
            edge_cutoff=self.edge_cutoff,
            max_neighbors=self.max_neighbors,
            rbf_edge_dist_cutoff=self.rbf_edge_dist_cutoff,
            num_rbf=self.num_rbf,
        )
        if self.subset_to_ca_atoms_only:
            data = self.subset_data_to_ca_atoms_only(
                data,
                num_rbf=self.num_rbf,
            )
        data["decoy_pdb_filepath"] = str(decoy_pdb_path)
        data["true_pdb_filepath"] = None if true_pdb_path is None else str(true_pdb_path)

        return data
