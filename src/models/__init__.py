# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import subprocess  # nosec

import numpy as np
from beartype import beartype
from beartype.typing import Dict
from biopandas.pdb import PandasPdb

HALT_FILE_EXTENSION = "done"


@beartype
def annotate_pdb_with_new_column_values(
    input_pdb_filepath: str,
    output_pdb_filepath: str,
    column_name: str,
    new_column_values: np.ndarray,
    pdb_df_key: str = "ATOM",
):
    """In-place annotates a PDB file with new column values.

    :param input_pdb_filepath: Path to input PDB file.
    :param output_pdb_filepath: Path to output PDB file.
    :param column_name: Name of column to annotate.
    :param new_column_values: New column values.
    :param pdb_df_key: Key of dataframe to annotate.
    """
    pdb = PandasPdb().read_pdb(input_pdb_filepath)
    if len(pdb.df[pdb_df_key]) > 0 and column_name in pdb.df[pdb_df_key]:
        if column_name in ["b_factor"]:
            residue_indices = (
                pdb.df[pdb_df_key]["residue_number"].values
                - pdb.df[pdb_df_key]["residue_number"].values.min()
            )
            pdb.df[pdb_df_key].loc[:, column_name] = new_column_values[residue_indices]
        else:
            raise NotImplementedError(f"PDB column {column_name} is currently not supported.")
    pdb.to_pdb(output_pdb_filepath)


@beartype
def calculate_tmscore_metrics(
    pred_pdb_filepath: str, native_pdb_filepath: str, tmscore_exec_path: str
) -> Dict[str, float]:
    """Calculates TM-score structural metrics between predicted and native protein structures.

    :param pred_pdb_filepath (str): Filepath to predicted protein structure in PDB format.
    :param native_pdb_filepath (str): Filepath to native protein structure in PDB format.
    :param tmscore_exec_path (str): Path to TM-score executable.
    :return: Dictionary containing TM-score structural metrics (e.g., GDT-HA).
    """
    # run TM-score with subprocess and capture output
    cmd = [tmscore_exec_path, pred_pdb_filepath, native_pdb_filepath]
    output = subprocess.check_output(cmd, text=True)  # nosec

    # parse TM-score output to extract structural metrics
    metrics = {}
    for line in output.splitlines():
        if line.startswith("TM-score"):
            metrics["TM-score"] = float(line.split()[-3])
        elif line.startswith("MaxSub"):
            metrics["MaxSub"] = float(line.split()[-3])
        elif line.startswith("GDT-TS"):
            metrics["GDT-TS"] = float(line.split()[-5])
        elif line.startswith("RMSD"):
            metrics["RMSD"] = float(line.split()[-1])
        elif line.startswith("GDT-HA"):
            metrics["GDT-HA"] = float(line.split()[-5])

    return metrics
