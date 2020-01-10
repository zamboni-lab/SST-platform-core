
from src.msfe.constants import qc_log_location


def print_qc_info(info, file=qc_log_location):
    with open(file, 'a') as log:
        log.write(info + "\n")