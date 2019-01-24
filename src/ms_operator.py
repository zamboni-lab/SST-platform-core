
import numpy


def extract_mz_region(spectrum, mz_interval):
    """ Method extracts spectrum region based on m/z interval specified. """

    mz_values = []
    intensities = []

    for i in range(len(spectrum['m/z array'])):

        if mz_interval[0] <= spectrum['m/z array'][i] <= mz_interval[1]:
            mz_values.append(spectrum['m/z array'][i])
            intensities.append(spectrum['intensity array'][i])

        elif spectrum['m/z array'][i] > mz_interval[1]:
            break

    return numpy.array(mz_values), numpy.array(intensities)


def find_closest_centroids(centroids_list, expected_peaks_list):
    """ This method looks for all the expected peaks in the list of centroids. """
    pass


def locate_annotated_peak(mz_region, spectrum):
    """ This method finds the accurate m/z and intensity values
    for visually chosen peaks and hardcoded m/z region for them. """

    local_max_intensity = -1

    for i in range(len(spectrum['m/z array'])):

        if mz_region[0] <= spectrum['m/z array'][i] <= mz_region[1]:

            if local_max_intensity <= spectrum['intensity array'][i]:
                local_max_intensity = spectrum['intensity array'][i]
            else:
                pass

        elif spectrum['m/z array'][i] > mz_region[1]:
            break

    if local_max_intensity < 0:
        raise ValueError

    accurate_intensity_value = local_max_intensity
    accurate_mz_value = spectrum['m/z array'][list(spectrum['intensity array']).index(local_max_intensity)]

    return accurate_mz_value, accurate_intensity_value


def get_peak_fitting_region(spectrum, index):
    """ This method extract the peak region (peak with tails) for a peak of the given index. """

    left_border = -1

    step = 0
    while left_border < 0:

        if spectrum['intensity array'][index-step-1] <= spectrum['intensity array'][index-step]:
            step += 1
        else:
            left_border = index-step

    right_border = -1

    step = 0
    while right_border < 0:

        if spectrum['intensity array'][index+step] >= spectrum['intensity array'][index+step+1]:
            step += 1
        else:
            right_border = index+step

    return [left_border, right_border]
