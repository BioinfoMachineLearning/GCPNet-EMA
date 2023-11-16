# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import BooleanField, StringField
from wtforms.validators import DataRequired, Email, InputRequired, Length


class PredictForm(FlaskForm):
    """UI Prediction Form."""

    file = FileField(
        "Choose PDB File",
        validators=[
            InputRequired("File is required."),
            FileAllowed(["pdb"], "Only PDB (.pdb) files are allowed."),
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
