
def parse_expected_peaks(file_path):
    """ This method parses information about expected peaks. """

    expected_mzs = []
    expected_intensities = []

    with open(file_path) as file:
        lines = file.readlines()

    for i in range(1,len(lines)):

        mz, intensity = lines[i].split(",")

        expected_mzs.append(mz)
        expected_intensities.append(intensity)

    return expected_mzs, expected_intensities
