""" MS feature extractor """

import time, numpy
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import parser, ms_operator
from src.constants import peak_region_factor as prf
from src.constants import peak_widths_levels_of_interest as widths_levels
from src.constants import minimal_normal_peak_intensity
from src.constants import maximum_number_of_subsequent_peaks_to_consider as max_sp_number
from src.constants import normal_scan_mz_frame_size, normal_scan_number_of_frames
from src.constants import chemical_noise_scan_mz_frame_size, chemical_noise_scan_number_of_frames
from src.constants import instrument_noise_mz_frame_size, instrument_noise_scan_number_of_frames
from src.constants import number_of_top_noisy_peaks_to_consider as n_top_guys
from src.constants import frame_intensity_percentiles
from src.constants import chemical_noise_features_scans_indexes, instrument_noise_features_scans_indexes
from src.constants import expected_peaks_file_path
from src.constants import minimal_background_peak_intensity as min_bg_peak_intensity
from lmfit.models import GaussianModel


def extract_peak_features(continuous_mz, fitted_intensity, fit_info, spectrum, centroids_indexes, peak_id):
    """ This method extracts features related to expected ions of interest and expected mixture chemicals. """

    predicted_peak_mz = float(continuous_mz[numpy.where(fitted_intensity == max(fitted_intensity))])

    # extract information about subsequent (following) peaks after the major one
    sp_ratios = extract_sp_features(predicted_peak_mz, max(fitted_intensity), continuous_mz[-1], spectrum,
                                    centroids_indexes)

    left_tail_auc, right_tail_auc = extract_auc_features(spectrum, continuous_mz, fitted_intensity, predicted_peak_mz)

    # TODO: think about a better metric
    symmetry = (left_tail_auc + right_tail_auc) / (2 * max(left_tail_auc, right_tail_auc))

    peak_features = {
        # # we don't have expected ("theoretical") intensity actually,
        # # we only have abundancy ratios for isotopes
        # 'expected_intensity_diff': max(fitted_intensity) - expected_intensity,
        # 'expected_intensity_ratio': expected_intensity / max(fitted_intensity),

        'is_missing_'+peak_id: 0,
        'saturation_'+peak_id: fit_info['saturation'],
        'intensity_'+peak_id: max(fitted_intensity),
        'absolute_mass_accuracy_'+peak_id: fit_info['fit_theory_absolute_ma'],
        'ppm_'+peak_id: fit_info['fit_theory_ppm'],
        'widths_'+peak_id: extract_width_features(continuous_mz, fitted_intensity),  # 20%, 50%, 80% of max intensity
        'subsequent_peaks_number_'+peak_id: int(sum([ratio > 0 for ratio in sp_ratios])),
        'subsequent_peaks_ratios_'+peak_id: sp_ratios,
        'left_tail_auc_'+peak_id: left_tail_auc,
        'right_tail_auc_'+peak_id: right_tail_auc,
        'symmetry_'+peak_id: symmetry,
        'goodness-of-fit_'+peak_id: fit_info['goodness-of-fit']
    }

    return peak_features


def extract_auc_features(spectrum, continuous_mz, fitted_intensity, predicted_peak_mz):
    """ This method extracts AUC (area under curve) features between real peak signal and fitted peak values. """

    # get raw peak data for integration within regions of interest
    l_tail_y, l_tail_x = ms_operator.get_integration_arrays(spectrum['m/z array'], spectrum['intensity array'],continuous_mz[0], predicted_peak_mz)
    r_tail_y, r_tail_x = ms_operator.get_integration_arrays(spectrum['m/z array'], spectrum['intensity array'],predicted_peak_mz, continuous_mz[-1])

    # integrate raw peak data within boundaries
    left_raw_data_integral = numpy.trapz(l_tail_y, l_tail_x)
    right_raw_data_integral = numpy.trapz(r_tail_y, r_tail_x)

    # get predicted peak data for integration within regions of interest
    l_tail_y, l_tail_x = ms_operator.get_integration_arrays(continuous_mz, fitted_intensity, continuous_mz[0],predicted_peak_mz)
    r_tail_y, r_tail_x = ms_operator.get_integration_arrays(continuous_mz, fitted_intensity, predicted_peak_mz,continuous_mz[-1])

    # integrate predicted peak data within boundaries
    left_predicted_data_integral = numpy.trapz(l_tail_y, l_tail_x)
    right_predicted_data_integral = numpy.trapz(r_tail_y, r_tail_x)

    # calculate features
    left_tail_auc = left_raw_data_integral - left_predicted_data_integral
    right_tail_auc = right_raw_data_integral - right_predicted_data_integral

    return left_tail_auc, right_tail_auc


def extract_sp_features(major_peak_mz, major_peak_intensity, right_boundary_mz, spectrum, centroids_indexes):
    """ This method extracts features of the following (subsequent) lower peaks after the major peak. """

    sp_number, sp_ratios = 0, []

    for index in centroids_indexes:

        if major_peak_mz <= spectrum['m/z array'][index] <= right_boundary_mz:

            sp_number += 1
            sp_ratios.append(spectrum['intensity array'][index] / major_peak_intensity)

        elif spectrum['m/z array'][index] > right_boundary_mz:
            break

    # always keep sp_ratios of the same size
    if sp_number < max_sp_number:
        # if there is less than a fixed size, then extend with null values
        ratios_extension = [-1 for value in range(max_sp_number-sp_number)]
        sp_ratios.extend(ratios_extension)

    elif sp_number > max_sp_number:
        # if there is more than a fixed size, then cut first fixed number
        sp_ratios = sp_ratios[0:max_sp_number]

    else:
        pass

    return sp_ratios


def extract_width_features(continuous_mz, fitted_intensity):
    """ This method extract widths of different levels of the peak height. """

    widths = []
    for percent in widths_levels:
        # intensity on the desired level
        intensity = max(fitted_intensity) * percent
        residuals = abs(fitted_intensity - intensity)

        # find mz value of desired intensity
        mz = continuous_mz[numpy.where(residuals == min(residuals))]

        width = float(2 * abs((continuous_mz[numpy.where(fitted_intensity == max(fitted_intensity))] - mz)))  # symmetry -> * 2

        widths.append(width)

    return widths


def get_peak_fit(peak_region, spectrum, theoretical_mz):
    """ This method fits the peak with a model and returns the fitted curve with fit information. """

    x, y, saturation = ms_operator.get_peak_fitting_values(spectrum, peak_region)

    if len(x) == len(y) == 3:
        # I doubt it ever happens with large expected peaks
        raise ValueError("Don't fit function to 3 points. Better calculate features from raw data.")

    else:

        g_model = GaussianModel()
        g_pars = g_model.guess(y, x=x)
        g_out = g_model.fit(y, g_pars, x=x)

        # define d as peak resolution (i.e. width on the 50% of the height)
        d, predicted_peak_mz = ms_operator.get_peak_width_and_predicted_mz(peak_region, spectrum, g_out)

        xc = numpy.linspace(predicted_peak_mz - prf * d, predicted_peak_mz + prf * d, 5000)
        yc = g_out.eval(x=xc)

        # now compose fit information

        # find absolute mass accuracy and ppm for signal related to fit
        signal_fit_mass_diff = float(x[numpy.where(y == max(y))] - predicted_peak_mz)
        signal_fit_ppm = signal_fit_mass_diff / predicted_peak_mz * 10 ** 6

        # find absolute mass accuracy and ppm for fit related to expected (theoretical) value
        fit_theory_mass_diff = predicted_peak_mz - theoretical_mz
        fit_theory_ppm = fit_theory_mass_diff / theoretical_mz * 10 ** 6

        fit_info = {
            'model': 'gaussian',
            'goodness-of-fit': [g_out.redchi, g_out.aic, g_out.bic],  # goodness-of-fit is reduced chi-squared
            'fit_theory_absolute_ma': fit_theory_mass_diff,  # fitted absolute mass accuracy
            'fit_theory_ppm': fit_theory_ppm,  # ppm between fitted peak mz and expected (theoretical) mz
            'resolution': d,
            'raw_intensity_array': y,
            'saturation': saturation,

            # probably redundant information
            'signal_fit_absolute_ma': signal_fit_mass_diff,
            'signal_fit_ppm': signal_fit_ppm
        }

        return xc, yc, fit_info


def fit_peak_and_extract_features(actual_peak, spectrum, centroids_indexes):
    """ This method takes index of peak, gets fitting region, fits the pick
        and extracts information out of fitted function. """

    peak_region_indexes = ms_operator.get_peak_fitting_region(spectrum, actual_peak['index'])

    fitted_mz, fitted_intensity, fit_info = get_peak_fit(peak_region_indexes, spectrum, actual_peak['expected_mz'])

    peak_features = extract_peak_features(fitted_mz, fitted_intensity, fit_info,
                                          spectrum, centroids_indexes, actual_peak['id'])

    peak_fit = {
        'expected_mz': actual_peak['expected_mz'],  # this is an id of the peak
        'peak_id': actual_peak['id'],
        'mz': fitted_mz,
        'intensity': fitted_intensity,
        'info': fit_info
    }

    return peak_fit, peak_features


def extract_non_expected_features_from_one_frame(mz_frame, spectrum, centroids_indexes, actual_peaks, scan_type):
    """ This method extracts non-expected features of a given frame. Expected peaks are excluded. """

    frame_peaks_intensities = []

    i = 0
    # go until left boundary of the frame is reached
    while mz_frame[0] > spectrum['m/z array'][centroids_indexes[i]]:
        i += 1
    # collect peaks between left and right boundaries
    while mz_frame[0] < spectrum['m/z array'][centroids_indexes[i]] < mz_frame[1]:

        is_non_expected_peak = True
        # try looking for this peak among expected ones
        for j in range(len(actual_peaks)):

            if actual_peaks[j]['present']:
                if actual_peaks[j]['index'] == centroids_indexes[i]:
                    # if found, then it's expected one
                    is_non_expected_peak = False
                    break
            else:
                pass

        if is_non_expected_peak:
            # append only non-expected peaks
            frame_peaks_intensities.append(spectrum['intensity array'][centroids_indexes[i]])
        else:
            pass

        # iterate further
        i += 1
        # exit loop if there's no more centroids
        if i == len(centroids_indexes):
            break

    top_peaks_intensities = sorted(frame_peaks_intensities, reverse=True)[0:n_top_guys]

    features_id = scan_type[0:4] + "_" + str(mz_frame[0]) + "_" + str(mz_frame[1])

    frame_features = {
        'number_of_peaks_'+features_id: len(frame_peaks_intensities),
        'intensity_sum_'+features_id: sum(frame_peaks_intensities),
        'percentiles_'+features_id: list(numpy.percentile(frame_peaks_intensities, frame_intensity_percentiles)),
        'top_peaks_intensities_'+features_id: top_peaks_intensities,
        'top_percentiles_'+features_id: list(numpy.percentile(top_peaks_intensities, frame_intensity_percentiles))
    }

    return frame_features


def extract_instrument_noise_features_from_one_frame(mz_frame, spectrum, centroids_indexes):
    """ This method extracts background (instrument noise) features of a given frame.
        No expected peaks here. """

    frame_peaks_intensities = []

    i = 0
    # go until left boundary of the frame is reached
    while mz_frame[0] > spectrum['m/z array'][centroids_indexes[i]]:
        i += 1

    # collect peaks between left and right boundaries
    while mz_frame[0] < spectrum['m/z array'][centroids_indexes[i]] < mz_frame[1]:
        frame_peaks_intensities.append(spectrum['intensity array'][centroids_indexes[i]])
        i += 1

        # exit loop if there's no more centroids
        if i == len(centroids_indexes):
            break

    top_peaks_intensities = sorted(frame_peaks_intensities, reverse=True)[0:n_top_guys]

    features_id = 'bg_' + str(mz_frame[0]) + "_" + str(mz_frame[1])

    frame_features = {
        'number_of_peaks_'+features_id: len(frame_peaks_intensities),
        'intensity_sum_'+features_id: sum(frame_peaks_intensities),
        'percentiles_'+features_id: list(numpy.percentile(frame_peaks_intensities, frame_intensity_percentiles)),
        'top_peaks_intensities_'+features_id: top_peaks_intensities,
        'top_percentiles_'+features_id: list(numpy.percentile(top_peaks_intensities, frame_intensity_percentiles))
    }

    return frame_features


def form_frames_and_extract_instrument_noise_features(spectrum, centroids_indexes):
    """ This method forms frames for extraction of instrument noise (background) features
        and then calls the extracting function. Frames here are specific background scans. """

    non_expected_features = []

    # define mz ranges to extract features from
    frames = []

    ranges = [i * instrument_noise_mz_frame_size for i in range(1, instrument_noise_scan_number_of_frames+2)]
    for i in range(instrument_noise_scan_number_of_frames):
        frames.append([ranges[i], ranges[i + 1]])

    # for each frame extract features
    for frame in frames:
        frame_features = extract_instrument_noise_features_from_one_frame(frame, spectrum, centroids_indexes)
        non_expected_features.append(frame_features)

    return non_expected_features


def form_frames_and_extract_non_expected_features(spectrum, centroids_indexes, actual_peaks, scan_type):
    """ This method forms m/z frames and then extracts non-expected features
        related to normal or chemical noise scan out of each frame.
        Different frames are used depending on the type of the scan. """

    non_expected_features = []

    # define mz ranges to extract features from
    frames = []

    if scan_type == 'normal':
        ranges = [i * normal_scan_mz_frame_size for i in range(1, normal_scan_number_of_frames+2)]
        for i in range(normal_scan_number_of_frames):
            frames.append([ranges[i], ranges[i+1]])

    elif scan_type == 'chemical_noise':
        ranges = [i * chemical_noise_scan_mz_frame_size for i in range(1, chemical_noise_scan_number_of_frames+2)]
        for i in range(chemical_noise_scan_number_of_frames):
            frames.append([ranges[i], ranges[i + 1]])

    else:
        pass

    # for each frame extract features
    for frame in frames:
        frame_features = extract_non_expected_features_from_one_frame(frame, spectrum, centroids_indexes, actual_peaks, scan_type)
        non_expected_features.append(frame_features)

    return non_expected_features


def find_isotope_and_extract_features(major_peak_index, actual_peaks_info, peak_fits):
    """ This method looks for the isotope in the list of peaks fits, gets its predicted intensity and mz,
        and calculates features using the major peak fit (major peak). """

    major_peak_fitted_intensity = peak_fits[major_peak_index]['intensity']
    major_peak_continuous_mz = peak_fits[major_peak_index]['mz']

    major_peak_max_intensity = max(major_peak_fitted_intensity)
    major_peak_mz = float(major_peak_continuous_mz[numpy.where(major_peak_fitted_intensity == major_peak_max_intensity)])

    isotope_intensity_ratios = []
    isotope_mass_diff_values = []

    for j in range(len(actual_peaks_info[major_peak_index]['expected_isotopes'])):

        # find each isotope in the peak fits list
        for k in range(len(peak_fits)):
            if peak_fits[k]['expected_mz'] == actual_peaks_info[major_peak_index]['expected_isotopes'][j]:

                # if the peak was present and was fitted actually
                if peak_fits[k]['mz'][0] != -1:

                    # ratio between isotope intensity and its major ions intensity
                    max_isotope_intensity = max(peak_fits[k]['intensity'])
                    ratio = max_isotope_intensity / major_peak_max_intensity

                    # m/z diff between isotope and its major ion (how far is the isotope)
                    isotope_mz = float(peak_fits[k]['mz'][numpy.where(peak_fits[k]['intensity'] == max_isotope_intensity)])
                    mass_diff = isotope_mz - major_peak_mz

                    isotope_intensity_ratios.append(ratio)
                    isotope_mass_diff_values.append(mass_diff)

                    break

                else:
                    # otherwise it means that this expected isotope is missing actually
                    isotope_intensity_ratios.append(-1)
                    isotope_mass_diff_values.append(-1)
                    break

    peak_id = actual_peaks_info[major_peak_index]['id']

    isotopic_features = {
        # 'isotopes mzs': actual_peaks_info[major_peak_index]['expected isotopes'],  # in case id is needed
        'intensity_ratios_'+peak_id: isotope_intensity_ratios,
        'mass_diff_values_'+peak_id: isotope_mass_diff_values
    }

    return isotopic_features


def find_fragment_and_extract_features(major_peak_index, actual_peaks_info, peak_fits):
    """ This method looks for the fragment in the list of peaks fits, gets its predicted intensity and mz,
        and calculates features using the major peak fit (major peak). """

    major_peak_fitted_intensity = peak_fits[major_peak_index]['intensity']
    major_peak_continuous_mz = peak_fits[major_peak_index]['mz']

    major_peak_max_intensity = max(major_peak_fitted_intensity)
    major_peak_mz = float(major_peak_continuous_mz[numpy.where(major_peak_fitted_intensity == major_peak_max_intensity)])

    fragment_intensity_ratios = []
    fragment_mass_diff_values = []

    for j in range(len(actual_peaks_info[major_peak_index]['expected_fragments'])):

        # find each fragment in the peak fits list
        for k in range(len(peak_fits)):
            if peak_fits[k]['expected_mz'] == actual_peaks_info[major_peak_index]['expected_fragments'][j]:

                # if the peak was present and was fitted actually
                if peak_fits[k]['mz'][0] != -1:

                    # ratio between fragment intensity and its major ions intensity
                    max_fragment_intensity = max(peak_fits[k]['intensity'])
                    ratio = max_fragment_intensity / major_peak_max_intensity

                    # m/z diff between fragment and its major ion (how far is the fragment)
                    fragment_mz = float(peak_fits[k]['mz'][numpy.where(peak_fits[k]['intensity'] == max_fragment_intensity)])
                    mass_diff = major_peak_mz - fragment_mz

                    fragment_intensity_ratios.append(ratio)
                    fragment_mass_diff_values.append(mass_diff)
                    break

                else:
                    # otherwise it means that this expected isotope is missing actually
                    fragment_intensity_ratios.append(-1)
                    fragment_mass_diff_values.append(-1)
                    break

    peak_id = actual_peaks_info[major_peak_index]['id']

    fragmentation_features = {
        # 'fragments mzs': actual_peaks_info[major_peak_index]['expected fragments'],  # in case id is needed
        'intensity_ratios_'+peak_id: fragment_intensity_ratios,
        'mass_diff_values_'+peak_id: fragment_mass_diff_values
    }

    return fragmentation_features


def get_null_peak_features(peak_id):
    """ Compose the empty dictionary with peak features
        to keep the whole features matrix of the same dimensionality. """

    missing_peak_features = {
        # # we don't have expected ("theoretical") intensity actually,
        # # we only have abundancy ratios for isotopes
        # 'expected_intensity_diff': -1,
        # 'expected_intensity_ratio': -1,

        'is_missing_'+peak_id: 1,
        'saturation_'+peak_id: -1,
        'intensity_'+peak_id: -1,
        'absolute_mass_accuracy_'+peak_id: -1,
        'ppm_'+peak_id: -1,
        'widths_'+peak_id: [-1, -1, -1],  # 20%, 50%, 80% of max intensity
        'subsequent_peaks_number_'+peak_id: -1,
        'subsequent_peaks_ratios_'+peak_id: [-1 for value in range(max_sp_number)],
        'left_tail_auc_'+peak_id: -1,
        'right_tail_auc_'+peak_id: -1,
        'symmetry_'+peak_id: -1,
        'goodness-of-fit_'+peak_id: [-1, -1, -1]
    }

    return missing_peak_features


def get_null_peak_fit(actual_peak):
    """ Compose the empty dictionary with peak fit for a missing peak to keep dimensionality of the data structure. """

    missing_peak_fit = {
        'expected_mz': actual_peak['expected_mz'],  # this is an id of the peak
        'peak_id': actual_peak['id'],
        'mz': [-1],
        'intensity': [-1],
        'info': {}
    }

    return missing_peak_fit


def get_null_isotopic_features(actual_peak_info):
    """ Compose the empty dictionary with isotopic features for a missing peak
        to keep the whole features matrix of the same dimensionality. """

    peak_id = actual_peak_info['id']

    missing_isotopic_features = {
        # 'isotopes mzs': actual_peak_info['expected isotopes'],  # in case id is needed
        'intensity_ratios_'+peak_id: [-1 for value in actual_peak_info['expected_isotopes']],
        'mass_diff_values_'+peak_id: [-1 for value in actual_peak_info['expected_isotopes']]
    }

    return missing_isotopic_features


def get_null_fragmentation_features(actual_peak_info):
    """ Compose the empty dictionary with isotopic features for a missing peak
        to keep the whole features matrix of the same dimensionality. """

    peak_id = actual_peak_info['id']

    missing_fragmentation_features = {
        # 'fragments mzs': actual_peak_info['expected fragments'],  # in case id is needed
        'intensity_ratios_'+peak_id: [-1 for value in actual_peak_info['expected_fragments']],
        'mass_diff_values_'+peak_id: [-1 for value in actual_peak_info['expected_fragments']]
    }

    return missing_fragmentation_features


def extend_scan_features(general_scan_features, general_features_names, some_new_features, new_features_type, we_need_features_names=True):

    if we_need_features_names:
        # this is the first scan, so we are collecting features values and names
        for features in some_new_features:
            for feature_name in list(features.keys()):

                if isinstance(features[feature_name], int) or isinstance(features[feature_name], float):
                    general_scan_features.append(features[feature_name])
                    general_features_names.append(feature_name)

                elif isinstance(features[feature_name], list):
                    general_scan_features.extend(features[feature_name])
                    for i in range(len(features[feature_name])):
                        general_features_names.append(feature_name + "_" + str(i))

                elif isinstance(features[feature_name], numpy.ndarray):
                    general_scan_features.extend(list(features[feature_name]))
                    for i in range(len(features[feature_name])):
                        general_features_names.append(feature_name + "_" + str(i))

                else:
                    print(feature_name, ": ", features[feature_name])
                    raise ValueError("Unknown feature type encountered for: " + new_features_type)
    else:
        # collecting just features values (features names were collected during previous iteration, for previous scan)
        for features in some_new_features:
            for feature_name in list(features.keys()):

                if isinstance(features[feature_name], int) or isinstance(features[feature_name], float):
                    general_scan_features.append(features[feature_name])

                elif isinstance(features[feature_name], list):
                    general_scan_features.extend(features[feature_name])

                elif isinstance(features[feature_name], numpy.ndarray):
                    general_scan_features.extend(list(features[feature_name]))

                else:
                    print(feature_name, ": ", features[feature_name])
                    raise ValueError("Unknown feature type encountered for: " + new_features_type)


def merge_features(all_independent_features, all_isotopic_features, all_fragmentation_features, all_non_expected_features, get_names=True):
    """ This method combines all the different features
        and effectively builds one row (out of one scan) for the feature matrix. """

    scan_features = []
    features_names = []  # for feature matrix readability

    extend_scan_features(scan_features, features_names, all_independent_features, "independent", we_need_features_names=get_names)
    extend_scan_features(scan_features, features_names, all_isotopic_features, "isotopic", we_need_features_names=get_names)
    extend_scan_features(scan_features, features_names, all_fragmentation_features, "fragmentation", we_need_features_names=get_names)
    extend_scan_features(scan_features, features_names, all_non_expected_features, "non-expected", we_need_features_names=get_names)

    return scan_features, features_names


def extract_background_features_from_scan(spectrum, get_names=True):
    """ This method extracts background (related to instrument noise) features from one scan. """

    scan_features = []
    features_names = []

    # peak picking here
    centroids_indexes, properties = signal.find_peaks(spectrum['intensity array'], height=min_bg_peak_intensity)

    all_background_features = form_frames_and_extract_instrument_noise_features(spectrum, centroids_indexes)

    extend_scan_features(scan_features, features_names, all_background_features, "instrument noise", we_need_features_names=get_names)

    return scan_features, features_names


def extract_main_features_from_scan(spectrum, scan_type, get_names=True):
    """ This method extracts all the features from one scan.
        There are slight differences between normal scan and chemical noise scan. """

    # peak picking here
    centroids_indexes, properties = signal.find_peaks(spectrum['intensity array'], height=minimal_normal_peak_intensity)

    # # debug
    # import matplotlib.pyplot as plt
    # plt.plot(spectrum['m/z array'], spectrum['intensity array'], 'b-')
    # plt.plot(spectrum['m/z array'][centroids_indexes[22746]], spectrum['intensity array'][centroids_indexes[22746]], 'gx')
    # plt.plot(spectrum['m/z array'][centroids_indexes[22745]], spectrum['intensity array'][centroids_indexes[22745]], 'gx')
    # plt.plot(spectrum['m/z array'][centroids_indexes[22747]], spectrum['intensity array'][centroids_indexes[22747]], 'gx')
    # plt.plot(spectrum['m/z array'][216960], spectrum['intensity array'][216960], 'rx')
    # plt.show()

    # parse expected peaks info
    expected_ions_info = parser.parse_expected_ions(expected_peaks_file_path, scan_type=scan_type)

    # TODO: review & debug, as it's causing problems: wrike+351004892
    # # correct indexes of peaks (currently only saturated peaks are processed)
    # corrected_centroids_indexes = ms_operator.correct_centroids_indexes(spectrum['m/z array'], spectrum['intensity array'],
    #                                                                     centroids_indexes, expected_ions_info)

    # get information about actual peaks in the spectrum in relation to expected ones and centroiding results
    actual_peaks = ms_operator.find_closest_centroids(spectrum['m/z array'], centroids_indexes, expected_ions_info)

    independent_peaks_features = []
    independent_peak_fits = []  # there is a need to store temporarily peak fitting results

    # extract peaks features independently
    for i in range(len(actual_peaks)):

        if actual_peaks[i]['present']:

            peak_fit, peak_features = fit_peak_and_extract_features(actual_peaks[i], spectrum, centroids_indexes)

            independent_peaks_features.append(peak_features)
            independent_peak_fits.append(peak_fit)

        else:
            # save the same dimensionality with actual peaks structure
            # ans keep the size of the feature matrix constant

            null_peak_features = get_null_peak_features(actual_peaks[i]['id'])
            null_peak_fit = get_null_peak_fit(actual_peaks[i])

            independent_peaks_features.append(null_peak_features)
            independent_peak_fits.append(null_peak_fit)

    isotopic_peaks_features = []
    fragmentation_peaks_features = []

    # extract features related to ions isotopic abundance and fragmentation
    for i in range(len(actual_peaks)):

        if actual_peaks[i]['present']:
            if len(actual_peaks[i]['expected_isotopes']) > 0:
                isotope_features = find_isotope_and_extract_features(i, actual_peaks, independent_peak_fits)
                isotopic_peaks_features.append(isotope_features)

            elif len(actual_peaks[i]['expected_fragments']) > 0:
                fragmentation_features = find_fragment_and_extract_features(i, actual_peaks, independent_peak_fits)
                fragmentation_peaks_features.append(fragmentation_features)
            else:
                pass

        else:
            # fill the data structure with null values
            if len(actual_peaks[i]['expected_isotopes']) > 0:
                isotope_features = get_null_isotopic_features(actual_peaks[i])
                isotopic_peaks_features.append(isotope_features)

            elif len(actual_peaks[i]['expected_fragments']) > 0:
                fragmentation_features = get_null_fragmentation_features(actual_peaks[i])
                fragmentation_peaks_features.append(fragmentation_features)
            else:
                pass

    # extract non-expected features from a scan
    non_expected_features = form_frames_and_extract_non_expected_features(spectrum, centroids_indexes,
                                                                          actual_peaks, scan_type=scan_type)

    # merge independent, isotopic, fragmentation and non-expected features
    scan_features, features_names = merge_features(independent_peaks_features, isotopic_peaks_features,
                                                   fragmentation_peaks_features, non_expected_features, get_names=get_names)

    return scan_features, features_names


def aggregate_features(list_of_scans_features, features_names):
    """ This method takes list of scans features and returns one feature list of n scans feature lists:
        for each feature average value is calculated and variance metric is added as another feature. """

    aggregated_main_features_names = []

    # in case there was only 1 scan processed
    if len(list_of_scans_features) == 1:
        return list_of_scans_features[0]
    # otherwise do aggregate
    else:
        aggregated_main_features = []

        for j in range(len(list_of_scans_features[0])):
            feature_values = []
            for i in range(len(list_of_scans_features)):
                # ! adding -1 values (e.g., missing features) we allow estimations to be shifted significantly
                feature_values.append(list_of_scans_features[i][j])

            # simple averaging and variance estimation
            # TODO: consider using bootstrap or other mean estimates instead
            mean_estimate = numpy.mean(feature_values)
            dispersion_estimate = numpy.std(feature_values)

            feature_names = [features_names[j]+"_mean", features_names[j]+"_std"]

            aggregated_main_features.extend([mean_estimate, dispersion_estimate])
            aggregated_main_features_names.extend(feature_names)

        return aggregated_main_features, aggregated_main_features_names


def extract_features_from_ms_run(spectra, ms_run_ids, in_test_mode=False):
    """ This is main method of the module. It extracts all the features from a single ms run (several scans)
        and returns single row of feature matrix (list) together with feature names (list)."""

    start_time = time.time()

    if in_test_mode:

        # chemical mix by Michelle
        chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix/20190405_QCmeth_Mix30_013.mzXML'
        spectra = list(mzxml.read(chemical_standard))

        print('\n', time.time() - start_time, "seconds elapsed for reading")

    else:
        pass

    feature_matrix_row = []
    feature_matrix_row_names = []

    main_features_scans_indexes = ms_operator.get_best_tic_scans_indexes(spectra, in_test_mode=in_test_mode)

    # get main features for every scan
    main_features = []
    main_features_names = []
    for scan_index in main_features_scans_indexes:

        if len(main_features_names) > 0:
            scan_features, _ = extract_main_features_from_scan(spectra[scan_index], scan_type='normal', get_names=False)
        else:
            scan_features, main_features_names = extract_main_features_from_scan(spectra[scan_index],scan_type='normal')

        main_features.append(scan_features)

    # aggregate main features and add to the feature matrix
    aggregated_main_features, aggregated_main_features_names = aggregate_features(main_features, main_features_names)

    # get chemical noise features for every scan
    chemical_noise_features = []
    chemical_noise_features_names = []
    for scan_index in chemical_noise_features_scans_indexes:

        if len(chemical_noise_features_names) > 0:
            scan_features, _ = extract_main_features_from_scan(spectra[scan_index], scan_type='chemical_noise', get_names=False)
        else:
            scan_features, chemical_noise_features_names = extract_main_features_from_scan(spectra[scan_index], scan_type='chemical_noise')

        chemical_noise_features.append(scan_features)

    # aggregate chemical noise features and add to the feature matrix
    aggregated_chemical_noise_features, aggregated_chemical_noise_features_names = aggregate_features(chemical_noise_features, chemical_noise_features_names)

    # get features related to instrument noise for every scan
    instrument_noise_features = []
    instrument_noise_features_names = []
    for scan_index in instrument_noise_features_scans_indexes:
        if len(instrument_noise_features_names) > 0:
            scan_features, _ = extract_background_features_from_scan(spectra[scan_index], get_names=False)
        else:
            scan_features, instrument_noise_features_names = extract_background_features_from_scan(spectra[scan_index])

        instrument_noise_features.append(scan_features)

    # aggregate instrument noise features and add to the feature matrix
    aggregated_instrument_noise_features, aggregated_instrument_noise_features_names = aggregate_features(instrument_noise_features, instrument_noise_features_names)

    # compose feature matrix row (values)
    feature_matrix_row.extend(aggregated_main_features)
    feature_matrix_row.extend(aggregated_chemical_noise_features)
    feature_matrix_row.extend(aggregated_instrument_noise_features)

    # compose feature matrix row (names)
    feature_matrix_row_names.extend(aggregated_main_features_names)
    feature_matrix_row_names.extend(aggregated_chemical_noise_features_names)
    feature_matrix_row_names.extend(aggregated_instrument_noise_features_names)

    print('\n', time.time() - start_time, "seconds elapsed in total")

    parser.update_feature_matrix(feature_matrix_row, feature_matrix_row_names, ms_run_ids)


if __name__ == '__main__':

    ms_run_ids = {'date': 'next_time', 'original_filename': 'another_new_file'}
    extract_features_from_ms_run([], ms_run_ids, in_test_mode=True)
