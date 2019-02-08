
import numpy
from src.constants import allowed_ppm_error


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


def find_closest_centroids(mz_spectrum, centroids_indexes, expected_ions_info):
    """ This method looks for all the expected peaks in the list of centroids. """

    expected_peaks_list = expected_ions_info['expected mzs']
    expected_intensities_list = expected_ions_info['expected intensities']

    # actual peaks out of expected ones
    actual_peaks = []

    for i in range(len(expected_peaks_list)):

        closest_peak_index, centroid_ppm = find_closest_peak_index(mz_spectrum, centroids_indexes, expected_peaks_list[i])

        isotopes, fragments = find_expected_isotopes_and_fragments(expected_peaks_list[i],
                                                                   expected_ions_info['isotopes mzs'],
                                                                   expected_ions_info['fragments mzs'])

        if closest_peak_index < 0:
            another_peak = {
                'present': False,
                'expected mz': expected_peaks_list[i],
                'expected isotopes': isotopes, # expected (theoretical) isotopes related to this ion (incl. itself)
                'expected fragments': fragments, # expected (theoretical) fragments related to this ion (incl. itself)
            }
        else:
            another_peak = {
                'present': True,
                'expected mz': expected_peaks_list[i],  # expected (theoretical) mz
                'expected intensity': expected_intensities_list[i],  # expected (theoretical) ion abundance
                'mz': mz_spectrum[centroids_indexes[closest_peak_index]],  # measured mz
                'index': centroids_indexes[closest_peak_index],
                'expected isotopes': isotopes,  # expected (theoretical) isotopes related to this ion (incl. itself)
                'expected fragments': fragments,  # expected (theoretical) fragments related to this ion (incl. itself)
                'centroid ppm': centroid_ppm  # ppm between expected peak and actual peak centroid
            }

        actual_peaks.append(another_peak)

    return actual_peaks


def find_expected_isotopes_and_fragments(expected_peak_mz, isotopes_mz_lists, fragments_mz_lists):
    """ This method returns a list of expected isotopes and a list of expected fragments if there are any. """

    isotopes, fragments = [], []

    for isotope_list in isotopes_mz_lists:
        # if this is a major ion having the other isotopes
        if expected_peak_mz == isotope_list[0]:
            # add to the list the other isotopes
            isotopes = isotope_list[1:len(isotope_list)]

    for fragments_list in fragments_mz_lists:
        # if this is a major ion having potential fragments (may be fragmented)
        if expected_peak_mz == fragments_list[0]:
            # add to the list these potential fragments
            fragments = fragments_list[1:len(fragments_list)]

    return isotopes, fragments


def find_closest_peak_index(mz_spectrum, peaks_indexes, expected_peak_mz):
    """ This method finds the closest peak to the expected one within centroids list.
        If in the vicinity of allowed ppm there is no peak, the peak is considered to be missing. """

    closest_index = 0
    while mz_spectrum[peaks_indexes[closest_index]] < expected_peak_mz:
        closest_index += 1

    previous_peak_ppm = abs(mz_spectrum[peaks_indexes[closest_index-1]] - expected_peak_mz) / expected_peak_mz * 10 ** 6
    next_peak_ppm = abs(mz_spectrum[peaks_indexes[closest_index]] - expected_peak_mz) / expected_peak_mz * 10 ** 6

    if previous_peak_ppm <= next_peak_ppm and previous_peak_ppm <= allowed_ppm_error:
        return closest_index-1, previous_peak_ppm

    elif previous_peak_ppm > next_peak_ppm and next_peak_ppm <= allowed_ppm_error:
        return closest_index, next_peak_ppm

    else:
        return -1, None


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
    """ This method extracts the peak region indexes (peak with tails) for a peak of the given index.
        The region is being prolonged unless the intensity value goes up again. So the tails are always descending. """

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


def get_peak_width_and_predicted_mz(peak_region, spectrum, fitted_model):
    """ This method calculates peak resolution. """

    # define the intensity of the desired mz range
    intensity_at_half_height = max(spectrum['intensity array'][peak_region[0]:peak_region[-1] + 1]) / 2

    # find predicted peak mz
    xc = numpy.linspace(spectrum['m/z array'][peak_region[0]], spectrum['m/z array'][peak_region[-1] + 1], 100)
    yc = fitted_model.eval(x=xc)

    predicted_peak_mz = xc[numpy.where(yc == max(yc))]

    # define step to evaluate model to the left and to the right of the peak
    mz_step = max(
        abs(predicted_peak_mz - spectrum['m/z array'][peak_region[0]]),
        abs(predicted_peak_mz - spectrum['m/z array'][peak_region[-1]])
    )

    i = 1
    while True:

        # extend region with i mz-steps to look for desired mz value
        xc = numpy.linspace(predicted_peak_mz - i * mz_step, predicted_peak_mz + i * mz_step, i*100)
        yc = fitted_model.eval(x=xc)

        # if current region covers the intensity of the desired mz
        if min(yc) < intensity_at_half_height:

            residuals = abs(numpy.array(yc) - intensity_at_half_height)

            # find mz value of desired intensity
            half_height_mz = xc[numpy.where(residuals == min(residuals))]

            half_peak_width = predicted_peak_mz - half_height_mz

            return 2 * half_peak_width, predicted_peak_mz

        else:
            i += 1


def get_integration_arrays(mz_array, intensity_array, left_point_mz, right_point_mz):
    """ This method gets raw data arrays for integration given the desired boundaries (left and right point). """

    # find boundaries in the raw data to ingrate within
    mzs = []
    intensities = []

    for i in range(len(mz_array)):

        if left_point_mz <= mz_array[i] <= right_point_mz:

            mzs.append(mz_array[i])
            intensities.append(intensity_array[i])

        elif mz_array[i] > right_point_mz:
            break

    return intensities, mzs
