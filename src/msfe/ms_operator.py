
import numpy
from src.msfe.constants import allowed_ppm_error, number_of_normal_scans, normal_scans_indexes_window


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

    saturated_peak_mz_values = [mz_spectrum[peak_index]]
    saturated_peak_mz_indexes = [peak_index]

    step = 1
    # go to the left looking for the same intensity values
    while intensities[peak_index] == intensities[peak_index - step]:
        saturated_peak_mz_indexes.append(peak_index - step)
        saturated_peak_mz_values.append(mz_spectrum[peak_index - step])
        step += 1

    step = 1
    # go to the right looking for the same intensity values
    while intensities[peak_index] == intensities[peak_index + step]:
        saturated_peak_mz_indexes.append(peak_index + step)
        saturated_peak_mz_values.append(mz_spectrum[peak_index + step])
        step += 1

    return saturated_peak_mz_values, saturated_peak_mz_indexes


def correct_centroids_indexes(mz_spectrum, intensities, centroids_indexes, expected_ions_info):
    """ This method corrects peak-picking result. It looks for flat peaks (with the same maximum intensity value)
        among expected ones. If there is one, its centroid index is corrected so that it's closer to the expected value. """

    left_corrected_peak_index, right_corrected_peak_index, left_min_mz_diffs, right_min_mz_diffs = None, None, None, None

    expected_peaks_list = expected_ions_info['expected_mzs']

    for i in range(len(expected_peaks_list)):

        # find closest peak index to the expected one
        closest_index = 0
        while mz_spectrum[centroids_indexes[closest_index]] < expected_peaks_list[i] and closest_index+1 < len(centroids_indexes):
            closest_index += 1

        # check two neighboring peaks (left and right) for flat apex
        left_peak_is_flat = False
        right_peak_is_flat = False

        if intensities[centroids_indexes[closest_index-1]-1] == intensities[centroids_indexes[closest_index-1]] or \
                intensities[centroids_indexes[closest_index-1]] == intensities[centroids_indexes[closest_index-1]+1]:

            left_peak_is_flat = True

            flat_apex_mz_values, flat_apex_mz_indexes = get_saturated_peak_mz_range(mz_spectrum, intensities,
                                                                                      centroids_indexes[closest_index-1])
            # find closest mz
            mz_diffs = abs(numpy.array(flat_apex_mz_values) - expected_peaks_list[i])
            left_min_mz_diffs = min(abs(mz_diffs))
            closest_mz_index = int(mz_diffs[numpy.where(mz_diffs == left_min_mz_diffs)])

            left_corrected_peak_index = flat_apex_mz_indexes[closest_mz_index]

        else:
            pass

        if intensities[centroids_indexes[closest_index]-1] == intensities[centroids_indexes[closest_index]] or \
                intensities[centroids_indexes[closest_index]] == intensities[centroids_indexes[closest_index]+1]:

            right_peak_is_flat = True

            flat_apex_mz_values, flat_apex_mz_indexes = get_saturated_peak_mz_range(mz_spectrum, intensities,
                                                                                      centroids_indexes[closest_index])
            # find closest mz
            mz_diffs = abs(numpy.array(flat_apex_mz_values) - expected_peaks_list[i])
            right_min_mz_diffs = min(abs(mz_diffs))
            closest_mz_index = int(mz_diffs[numpy.where(mz_diffs == right_min_mz_diffs)])

            right_corrected_peak_index = flat_apex_mz_indexes[closest_mz_index]

        else:
            pass

        # nothing to correct if both peaks are not flat
        if not left_peak_is_flat and not right_peak_is_flat:
            continue

        # make correction for right one
        elif not left_peak_is_flat and right_peak_is_flat:
            centroids_indexes[closest_index] = right_corrected_peak_index
        # make correction for left one
        elif left_peak_is_flat and not right_peak_is_flat:
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

    expected_peaks_list = expected_ions_info['expected_mzs']
    expected_peaks_ids = expected_ions_info['ions_ids']

    # actual peaks out of expected ones
    actual_peaks = []

    for i in range(len(expected_peaks_list)):

        closest_peak_index, centroid_ppm = find_closest_peak_index(mz_spectrum, centroids_indexes, expected_peaks_list[i])

        isotopes, isotopic_ratios, fragments = find_expected_isotopes_and_fragments(expected_peaks_list[i],
                                                                                    expected_ions_info['isotopes_mzs'],
                                                                                    expected_ions_info['fragments_mzs'],
                                                                                    expected_ions_info['expected_isotopic_ratios'])

        if closest_peak_index < 0:
            another_peak = {
                'present': False,
                'expected_mz': expected_peaks_list[i],
                'id': expected_peaks_ids[i],
                'expected_isotopes': isotopes,  # expected (theoretical) isotopes related to this ion (incl. itself)
                'expected_fragments': fragments,  # expected (theoretical) fragments related to this ion (incl. itself)
                'expected_isotopic_ratios': isotopic_ratios
            }
        else:
            another_peak = {
                'present': True,
                'expected_mz': expected_peaks_list[i],  # expected (theoretical) mz
                'id': expected_peaks_ids[i],
                'mz': mz_spectrum[centroids_indexes[closest_peak_index]],  # measured mz
                'index': centroids_indexes[closest_peak_index],
                'expected_isotopes': isotopes,  # expected (theoretical) isotopes related to this ion (incl. itself)
                'expected_isotopic_ratios': isotopic_ratios,
                'expected_fragments': fragments,  # expected (theoretical) fragments related to this ion (incl. itself)
                'centroid_ppm': centroid_ppm  # ppm between expected peak and actual peak centroid
            }

        actual_peaks.append(another_peak)

    return actual_peaks


def find_expected_isotopes_and_fragments(expected_peak_mz, isotopes_mz_lists, fragments_mz_lists, isotopic_ratios_lists):
    """ This method returns a list of expected isotopes and a list of expected fragments if there are any. """

    # TODO: review & refactor: method was revisited since v.0.2.15, may be redundant now

    isotopes, fragments, ratios = [], [], []

    for isotope_list in isotopes_mz_lists:
        # if this is a major ion having the other isotopes
        if expected_peak_mz == isotope_list[0]:
            # add to the list the other isotopes
            isotopes = isotope_list[0:len(isotope_list)]
            index = isotopes_mz_lists.index(isotope_list)
            ratios = isotopic_ratios_lists[index][0:len(isotopic_ratios_lists[index])]
            break

    for fragments_list in fragments_mz_lists:
        # if this is a major ion having potential fragments (may be fragmented)
        if expected_peak_mz == fragments_list[0]:
            # add to the list these potential fragments
            fragments = fragments_list[0:len(fragments_list)]
            break

    return isotopes, ratios, fragments


def find_closest_peak_index(mz_spectrum, peaks_indexes, expected_peak_mz):
    """ This method finds the closest peak to the expected one within centroids list.
        If in the vicinity of allowed ppm there is no peak, the peak is considered to be missing. """

    closest_index = 0
    while mz_spectrum[peaks_indexes[closest_index]] < expected_peak_mz and closest_index+1 < len(peaks_indexes):
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

    return left_border, right_border


def get_peak_fitting_region_2(spectrum, index):
    """ This method extracts the peak region indexes (peak with tails) for a peak of the given index.
        This version of the method considers two following points instead of one. So the tails may ascend locally,
        but globally they are also descending. """

    local_maximum = spectrum['intensity array'][index]

    left_border = -1

    step_left = 0
    while left_border < 0:

        if spectrum['intensity array'][index-step_left-1] <= spectrum['intensity array'][index-step_left]:
            step_left += 1

        elif spectrum['intensity array'][index-step_left] < spectrum['intensity array'][index-step_left-1] < local_maximum \
                and spectrum['intensity array'][index-step_left-2] <= spectrum['intensity array'][index-step_left-1]:
            step_left += 1

        else:
            break

    right_border = -1

    step_right = 0
    while right_border < 0:

        if spectrum['intensity array'][index+step_right] >= spectrum['intensity array'][index+step_right+1]:
            step_right += 1

        elif spectrum['intensity array'][index+step_right] < spectrum['intensity array'][index+step_right+1] < local_maximum \
                and spectrum['intensity array'][index+step_right+2] <= spectrum['intensity array'][index+step_right+1]:
            step_right += 1

        else:
            break

    # a way to guarantee equal number of point to the left and to the right from the peak
    left_border = index - min(step_left, step_right)
    right_border = index + min(step_left, step_right)

    return left_border, right_border


def get_peak_fitting_values(spectrum, peak_region):
    """ This method returns mz and intensity values according to the given peak region.
        The flat apex intensity values, however, are not included. """

    mzs, intensities = spectrum['m/z array'][peak_region[0]:peak_region[-1] + 1], \
                       spectrum['intensity array'][peak_region[0]:peak_region[-1] + 1]

    is_peak_flat = False

    # correction is made in case the peak is flat
    corrected_mzs = [mzs[0]]  # add first value
    corrected_intensities = [intensities[0]]  # add first value

    max_intensity = max(intensities)

    for i in range(1, len(intensities) - 1):

        if intensities[i] == intensities[i-1] and intensities[i] == max_intensity:
            is_peak_flat = True

        else:
            # these are normal values -> append
            corrected_intensities.append(intensities[i])
            corrected_mzs.append(mzs[i])

    # add last values
    corrected_intensities.append(intensities[-1])
    corrected_mzs.append(mzs[-1])

    return numpy.array(corrected_mzs), numpy.array(corrected_intensities), is_peak_flat


def get_peak_width_and_predicted_mz(peak_region, spectrum, fitted_model):
    """ This method calculates peak resolution. """

    # define the intensity of the desired mz range
    intensity_at_half_height = max(spectrum['intensity array'][peak_region[0]:peak_region[-1] + 1]) / 2

    # find predicted peak mz
    xc = numpy.linspace(spectrum['m/z array'][peak_region[0]], spectrum['m/z array'][peak_region[-1]], 101)
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
        xc = numpy.linspace(predicted_peak_mz - i * mz_step, predicted_peak_mz + i * mz_step, i*100+1)
        yc = fitted_model.eval(x=xc)

        # if current region covers the intensity of the desired mz
        if min(yc) < intensity_at_half_height:

            residuals = abs(numpy.array(yc) - intensity_at_half_height)

            # find mz value of desired intensity
            half_height_mz = float(xc[numpy.where(residuals == min(residuals))][0])  # for ideally symmetrical peak the left point is taken

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


def get_best_tic_scans_indexes(spectra, n=number_of_normal_scans, in_test_mode=False):
    """ This method finds max TIC within the spectra and returns n following scans indexes. """

    if in_test_mode:
        # for mzxml data structure
        tic_field_name = "totIonCurrent"
    else:
        # for custom data structure built on mz5
        tic_field_name = "tic"

    max_tic_scan = (0, spectra[0][tic_field_name])

    number_from, number_to = normal_scans_indexes_window

    for i in range(number_from, number_to):
        if spectra[i][tic_field_name] > max_tic_scan[1]:
            max_tic_scan = (i, spectra[i][tic_field_name])

    best_tic_scans_indexes = [max_tic_scan[0]+i for i in range(n)]

    if in_test_mode:
        # # add saturated scans for testing
        # best_tic_scans_indexes.append(18)  # for file 007 from test1
        # best_tic_scans_indexes.append(60)  # for file 042 from test1
        pass

    return best_tic_scans_indexes
