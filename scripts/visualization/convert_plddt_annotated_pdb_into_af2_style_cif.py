import argparse

from Bio import PDB
from Bio.PDB import MMCIFIO, MMCIF2Dict


def main():
    """Convert PDB to CIF with AFDB-style plDDT residue annotations."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Convert PDB to CIF")

    # Add the arguments
    parser.add_argument(
        "pdb_file", metavar="input", type=str, help="The input (annotated) PDB file"
    )
    parser.add_argument(
        "ref_cif_file", metavar="ref", type=str, help="The input (reference) mmCIF file"
    )
    parser.add_argument("output_cif_file", metavar="output", type=str, help="The output CIF file")

    # Parse the arguments
    args = parser.parse_args()

    # Load PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", args.pdb_file)

    # Load reference mmCIF file
    mmcif_dict = MMCIF2Dict.MMCIF2Dict(args.ref_cif_file)
    mmcif_ma_qa_metric_global_columns = [
        "_ma_qa_metric.id",
        "_ma_qa_metric.mode",
        "_ma_qa_metric.name",
        "_ma_qa_metric.software_group_id",
        "_ma_qa_metric.type",
    ]
    mmcif_ma_qa_metric_local_columns = [
        "_ma_qa_metric_local.label_asym_id",
        "_ma_qa_metric_local.label_comp_id",
        "_ma_qa_metric_local.label_seq_id",
        "_ma_qa_metric_local.metric_id",
        "_ma_qa_metric_local.metric_value",
        "_ma_qa_metric_local.model_id",
        "_ma_qa_metric_local.ordinal_id",
    ]
    for column in mmcif_ma_qa_metric_global_columns:
        if column not in mmcif_dict:
            if column == "_ma_qa_metric.id":
                mmcif_dict[column] = ["1", "2"]
            if column == "_ma_qa_metric.mode":
                mmcif_dict[column] = ["global", "local"]
            if column == "_ma_qa_metric.name":
                mmcif_dict[column] = ["plDDT", "pLDDT"]
            if column == "_ma_qa_metric.software_group_id":
                mmcif_dict[column] = ["1", "1"]
            if column == "_ma_qa_metric.type":
                mmcif_dict[column] = ["plDDT", "pLDDT"]
    if "_ma_qa_metric_global.metric_id" not in mmcif_dict:
        mmcif_dict["_ma_qa_metric_global.metric_id"] = "1"
    if "_ma_qa_metric_global.metric_value" not in mmcif_dict:
        mmcif_dict["_ma_qa_metric_global.metric_value"] = "0.0"
    if "_ma_qa_metric_global.model_id" not in mmcif_dict:
        mmcif_dict["_ma_qa_metric_global.model_id"] = "1"
    if "_ma_qa_metric_global.ordinal_id" not in mmcif_dict:
        mmcif_dict["_ma_qa_metric_global.ordinal_id"] = "1"
    for column in mmcif_ma_qa_metric_local_columns:
        if column not in mmcif_dict:
            mmcif_dict[column] = ["?"] * len(list(structure.get_residues()))

    # Iterate through the structure and update mmCIF atom and residue entries
    for model in structure:
        for chain in model:
            for residue_list_index, residue in enumerate(chain):
                for atom_index, atom in enumerate(residue):
                    mmcif_dict["_atom_site.B_iso_or_equiv"][atom_index] = str(atom.bfactor)
                residue_index = str(residue.get_id()[1])
                mmcif_dict["_ma_qa_metric_local.label_asym_id"][
                    residue_list_index
                ] = chain.get_id()
                mmcif_dict["_ma_qa_metric_local.label_comp_id"][
                    residue_list_index
                ] = residue.get_resname()
                mmcif_dict["_ma_qa_metric_local.label_seq_id"][residue_list_index] = residue_index
                mmcif_dict["_ma_qa_metric_local.metric_id"][
                    residue_list_index
                ] = "2"  # NOTE: global mmCIF metrics are always first
                mmcif_dict["_ma_qa_metric_local.metric_value"][residue_list_index] = str(
                    atom.bfactor
                )
                mmcif_dict["_ma_qa_metric_local.model_id"][residue_list_index] = "1"
                mmcif_dict["_ma_qa_metric_local.ordinal_id"][residue_list_index] = residue_index

    # Write the modified mmCIF to a new file
    mmcif_writer = MMCIFIO()
    mmcif_writer.set_dict(mmcif_dict)
    mmcif_writer.save(args.output_cif_file)


if __name__ == "__main__":
    main()
