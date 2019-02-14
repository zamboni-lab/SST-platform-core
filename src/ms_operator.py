
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


def get_saturated_peak_mz_range(mz_spectrum, intensities, peak_index):
    """ This method looks for mz range of the saturated peak, i.e. mz range of a peak within which all the intensity
        values are equal. """

    saturated_peak_mz_values = []
    saturated_peak_mz_indexes = []

    # go to the left
    there_is_equal_intensity_on_the_left = True

    step = 0
    while there_is_equal_intensity_on_the_left:

        if intensities[peak_index - step - 1] == intensities[peak_index - step]:
            step += 1
            saturated_peak_mz_values.append(mz_spectrum[peak_index - step - 1])
            saturated_peak_mz_indexes.append(peak_index - step - 1)
        else:
            there_is_equal_intensity_on_the_left = False

    # go to the right
    there_is_equal_intensity_on_the_right = True

    step = 0
    while there_is_equal_intensity_on_the_right:

        if intensities[peak_index + step] == intensities[peak_index + step + 1]:
            step += 1
            saturated_peak_mz_values.append(mz_spectrum[peak_index + step + 1])
            saturated_peak_mz_indexes.append(peak_index + step + 1)
        else:
            there_is_equal_intensity_on_the_right = False

    return saturated_peak_mz_values, saturated_peak_mz_indexes


def correct_centroids_indexes(mz_spectrum, intensities, centroids_indexes, expected_ions_info):
    """ This method corrects peak-picking result. It looks for saturated peaks (with the same maximum intensity value)
        among expected ones. If there is one, its centroid index is corrected so that it's closer to the expected value. """

    left_corrected_peak_index, right_corrected_peak_index, left_min_mz_diffs, right_min_mz_diffs = None, None, None, None

    expected_peaks_list = expected_ions_info['expected mzs']

    for i in range(len(expected_peaks_list)):

        # find closest peak index to the expected one
        closest_index = 0
        while mz_spectrum[centroids_indexes[closest_index]] < expected_peaks_list[i]:
            closest_index += 1

        # check two neighboring peaks (left and right) for saturation
        left_peak_is_saturated = False
        right_peak_is_saturated = False

        if intensities[centroids_indexes[closest_index-1]-1] == intensities[centroids_indexes[closest_index-1]] or \
                intensities[centroids_indexes[closest_index-1]] == intensities[centroids_indexes[closest_index-1]+1]:

            left_peak_is_saturated = True

            saturation_mz_values, saturation_mz_indexes = get_saturated_peak_mz_range(mz_spectrum, intensities,
                                                                                      centroids_indexes[closest_index-1])
            # find closest mz
            mz_diffs = numpy.array(saturation_mz_values) - expected_peaks_list[i]
            left_min_mz_diffs = min(mz_diffs)
            closest_mz_index = int(mz_diffs[numpy.where(mz_diffs == left_min_mz_diffs)])

            left_corrected_peak_index = saturation_mz_indexes[closest_mz_index]

        else:
            pass

        if intensities[centroids_indexes[closest_index]-1] == intensities[centroids_indexes[closest_index]] or \
                intensities[centroids_indexes[closest_index]] == intensities[centroids_indexes[closest_index]+1]:

            right_peak_is_saturated = True

            saturation_mz_values, saturation_mz_indexes = get_saturated_peak_mz_range(mz_spectrum, intensities,
                                                                                      centroids_indexes[closest_index])
            # find closest mz
            mz_diffs = numpy.array(saturation_mz_values) - expected_peaks_list[i]
            right_min_mz_diffs = min(mz_diffs)
            closest_mz_index = int(mz_diffs[numpy.where(mz_diffs == right_min_mz_diffs)])

            right_corrected_peak_index = saturation_mz_indexes[closest_mz_index]

        else:
            pass

        # nothing to correct if both peaks are not saturated
        if not left_peak_is_saturated and not right_peak_is_saturated:
            continue

        # make correction of single saturated peak
        elif not left_peak_is_saturated and right_peak_is_saturated:
            centroids_indexes[closest_index] = right_corrected_peak_index
        # make correction of saturated peak
        elif left_peak_is_saturated and not right_peak_is_saturated:
            centroids_indexes[closest_index-1] = left_corrected_peak_index
        else:
            # choose the closest saturated peak and make correction then
            if left_min_mz_diffs < right_min_mz_diffs:
                centroids_indexes[closest_index-1] = left_corrected_peak_index
            else:
                centroids_indexes[closest_index] = right_corrected_peak_index

    return centroids_indexes


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
                'expected isotopes': isotopes,  # expected (theoretical) isotopes related to this ion (incl. itself)
                'expected fragments': fragments,  # expected (theoretical) fragments related to this ion (incl. itself)
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
            break

    for fragments_list in fragments_mz_lists:
        # if this is a major ion having potential fragments (may be fragmented)
        if expected_peak_mz == fragments_list[0]:
            # add to the list these potential fragments
            fragments = fragments_list[1:len(fragments_list)]
            break

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


def get_peak_fitting_values(spectrum, peak_region):
    """ This method returns mz and intensity values according to the given peak region.
        It also checks the peak for saturation and returns boolean. """

    mzs, intensities = spectrum['m/z array'][peak_region[0]:peak_region[-1] + 1], \
                       spectrum['intensity array'][peak_region[0]:peak_region[-1] + 1]

    is_peak_saturated = False

    # correction is made in case the peak is saturated
    corrected_mzs = [mzs[0]]  # add first value
    corrected_intensities = [intensities[0]]  # add first value

    for i in range(1, len(intensities) - 1):

        if (intensities[i] == intensities[i-1] or intensities[i] == intensities[i+1]) \
                and intensities[i] == max(intensities):

            is_peak_saturated = True

        else:
            # these are normal values -> append
            corrected_intensities.append(intensities[i])
            corrected_mzs.append(mzs[i])

    # add last values
    corrected_intensities.append(intensities[-1])
    corrected_mzs.append(mzs[-1])

    return numpy.array(corrected_mzs), numpy.array(corrected_intensities), int(is_peak_saturated)


def get_peak_width_and_predicted_mz(peak_region, spectrum, fitted_model):
    """ This method calculates peak resolution. """

    # define the intensity of the desired mz range
    intensity_at_half_height = max(spectrum['intensity array'][peak_region[0]:peak_region[-1] + 1]) / 2

    # find predicted peak mz
    xc = numpy.linspace(spectrum['m/z array'][peak_region[0]], spectrum['m/z array'][peak_region[-1]], 100)
    yc = fitted_model.eval(x=xc)

    predicted_peak_mz = float(xc[numpy.where(yc == max(yc))])
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
            half_height_mz = float(xc[numpy.where(residuals == min(residuals))])

            half_peak_width = abs(predicted_peak_mz - half_height_mz)

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
