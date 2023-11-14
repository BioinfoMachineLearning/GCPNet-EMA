# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField
from wtforms import BooleanField
from wtforms.validators import InputRequired


class PredictForm(FlaskForm):
    """Predict Form."""

    file = FileField(
        "Choose PDB File",
        validators=[
            FileAllowed(["pdb"], "Only PDB files are allowed."),
            InputRequired("File is required."),
        ],
    )
    af2_input = BooleanField("Is AlphaFold Structure")
