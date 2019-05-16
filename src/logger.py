
from src.constants import log_location


def print_info(info, file=log_location):
    with open(file, 'a') as log:
        log.write(info + "\n")
