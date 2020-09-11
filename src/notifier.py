
import smtplib, time, traceback
from src.constants import gmail_sender as SENDER
from src.constants import gmail_password as PASSWORD
from src.constants import error_recipients as ME
from src.constants import new_qcs_recipients as RECIPIENTS
from src import logger
from email.mime.text import MIMEText


def send_new_qc_notification(qualities, info):
    """ This method sends a notification of a successful execution on a new QC file."""

    SUBJECT = 'New QC added'

    TEXT = 'Hi there,\n\n' \
           'A new QC run with {} buffer has just been processed.\n'.format(info['buffer']) + \
           'Score: ' + str(sum(qualities)) + '/' + str(len(qualities)) + '\n' + \
           'Total: ' + str(info['total_qcs']) + \
           '\nMore details:\n' \
           'http://imsb-nz-crazy/qc' \
           '\n\nCheers,\n' \
           'Mass Spec Monitor'

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(SENDER, PASSWORD)

    message = MIMEText(TEXT)

    message['Subject'] = SUBJECT
    message['From'] = SENDER
    message['To'] = ", ".join(RECIPIENTS)

    try:
        server.sendmail(SENDER, RECIPIENTS, message.as_string())
    except Exception:
        logger.print_qc_info("Notification failed!")
        logger.print_qc_info(traceback.format_exc())

    server.quit()


def send_error_notification(filename, trace):
    """ This method sends a notification of an error caused by a new QC file."""

    SUBJECT = 'New QC crashed'

    TEXT = 'Hi there,\n\n' \
           'The file ' + filename + ' caused an unexpected error:\n' + trace + \
           '\n\nCheck out \'qc_logs.txt\' on the server.' \
           '\n\nCheers,\n' \
           'Mass Spec Monitor'

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(SENDER, PASSWORD)

    message = MIMEText(TEXT)

    message['Subject'] = SUBJECT
    message['From'] = SENDER
    message['To'] = ", ".join(ME)

    try:
        server.sendmail(SENDER, ME, message.as_string())
    except Exception:
        logger.print_qc_info("Notification failed!")
        logger.print_qc_info(traceback.format_exc())

    server.quit()


if __name__ == "__main__":

    start_time = time.time()
    fake_qualities = [1 for x in range(16)]
    # send_new_qc_notification(fake_qualities)
    send_error_notification("new_file", "Value error")
    print("sending e-mail takes", time.time() - start_time, "s")