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

    # Iterate through the structure and update mmCIF atom and residue entries
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_index = atom.get_serial_number() - 1
                    mmcif_dict["_atom_site.B_iso_or_equiv"][atom_index] = str(atom.bfactor)
                residue_index = residue.get_id()[1] - 1
                mmcif_dict["_ma_qa_metric_local.metric_value"][residue_index] = str(atom.bfactor)

    # Write the modified mmCIF to a new file
    mmcif_writer = MMCIFIO()
    mmcif_writer.set_dict(mmcif_dict)
    mmcif_writer.save(args.output_cif_file)


if __name__ == "__main__":
    main()
