# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from beartype import beartype
from beartype.typing import List


@beartype
def send_email(
    subject: str,
    body: str,
    sender: str,
    recipients: List[str],
    password: str,
    output_file: str,
):
    """Send email with attachment.

    :param subject: Subject of email.
    :param body: Body of email.
    :param sender: Sender email address.
    :param recipients: List of recipient email addresses.
    :param password: Password for sender email address.
    :param output_file: Path to output file to attach to email.
    """
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
