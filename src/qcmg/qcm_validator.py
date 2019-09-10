
import os

""" QC values are logged into following files. """

folder = "/Users/andreidm/ETH/projects/qc_metrics/val/"
csv = ".csv"


def print_qcm_names(qc_type, names):
    """ This method prints the first line (names of the tracked values) to each file. """

    if os.path.isfile(folder + qc_type + csv):
        # file already exists
        pass
    else:
        # write a line of names
        single_str = ",".join(names)+'\n'
        with open(folder + qc_type + csv, 'a') as file:
            file.write(single_str)


def print_qcm_values(qc_type, values):
    """ This method prints values that are used to calculate QC characteristics. """

    single_str = ",".join(map(str, values))+'\n'

    with open(folder + qc_type + csv, 'a') as file:
        file.write(single_str)
