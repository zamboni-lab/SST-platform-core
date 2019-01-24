
from src.stuff.structures import Peak


def parse_expected_peaks(file_path):
    """ This method parses information about expected peaks and creates instances of Peak class. """

    expected_peaks = []

    with open(file_path) as file:
        lines = file.readlines()

    names = lines[0].split(",")

    for i in range(1,len(lines)):

        new_peak = Peak()
        new_peak_info = (names, lines[i].split(","))
        new_peak._set_expected_peak_info(new_peak_info)

        expected_peaks.append(new_peak)

    return expected_peaks
