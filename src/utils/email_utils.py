# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet-EMA (https://github.com/BioinfoMachineLearning/GCPNet-EMA):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from beartype import beartype
from beartype.typing import List


@beartype
def send_email_with_attachment(
    subject: str,
    body: str,
    sender: str,
    recipients: List[str],
    output_file: str,
    output_file_ext_type: str,
    smtp_server: str = "massmail.missouri.edu",
    port: int = 587,
):
    """Send email with attachment using `massmail`.

    :param subject: Subject of email.
    :param body: Body of email.
    :param sender: Sender email address.
    :param recipients: List of recipient email addresses.
    :param output_file: Path to output file to attach to email.
    :param output_file_ext_type: Extension type of output file.
    :param smtp_server: SMTP server to use for sending email.
    :param port: Port to use for SMTP. This is required for `starttls()`.
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
    # manually specify the attachment's filetype for `massmail` attachment support
    filename = os.path.splitext(os.path.basename(output_file))[0] + f".{output_file_ext_type}"
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )
    msg.attach(part)
    # send email with message and attachment
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls(context=context)
        server.sendmail(sender, recipients, msg.as_string())
