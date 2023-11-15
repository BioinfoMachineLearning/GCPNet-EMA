# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import re

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import BooleanField, StringField
from wtforms.validators import (
    DataRequired,
    Email,
    InputRequired,
    Length,
    ValidationError,
)


class PdbFileValidator:
    """Custom validator for PDB files."""

    def __call__(self, form, field):
        # reset the file position to the beginning
        field.data.seek(0)

        # check if the file content follows the PDB file format (a basic check)
        pdb_content = field.data.read().decode("utf-8")

        # NOTE: you may want to implement a more sophisticated check based on the PDB file format
        if not re.search(r"^ATOM|^HETATM", pdb_content, re.MULTILINE):
            raise ValidationError("Invalid PDB file format")


class PredictForm(FlaskForm):
    """UI Prediction Form."""

    file = FileField(
        "Choose PDB File",
        validators=[
            InputRequired("File is required."),
            FileAllowed(["pdb"], "Only PDB (.pdb) files are allowed."),
            PdbFileValidator(),
        ],
    )
    af2_input = BooleanField("Is AlphaFold Structure")


class ServerPredictForm(FlaskForm):
    """Server Prediction Form."""

    title = StringField("Title", validators=[DataRequired(), Length(min=1, max=10000)])
    structure_upload = FileField(
        "Structure Upload",
        validators=[
            FileRequired(),
            FileAllowed(["pdb"], "Only PDB (.pdb) files are allowed."),
        ],
    )
    sequence = StringField("Sequence", validators=[Length(min=0, max=10000)])
    results_email = StringField("Results Email", validators=[DataRequired(), Email()])
    other_parameters = StringField("Other Parameters", validators=[Length(min=0, max=10000)])

    # disable CSRF protection for this form only
    class Meta:
        """Meta class."""

        csrf = False
