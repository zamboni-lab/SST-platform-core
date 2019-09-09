
from src.msfe.constants import qc_log_location, tune_log_location


def print_qc_info(info, file=qc_log_location):
    with open(file, 'a') as log:
        log.write(info + "\n")


def print_tune_info(info, file=tune_log_location):
    with open(file, 'a') as log:
        log.write(info + "\n")
