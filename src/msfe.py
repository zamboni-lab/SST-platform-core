""" MS feature extractor """

import time, numpy
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import parser, ms_operator
from src.constants import peak_region_factor as prf
from src.constants import peak_widths_levels_of_interest as widths_levels
from src.constants import minimal_peak_intensity
from lmfit.models import GaussianModel


def extract_peak_features(continuous_mz, fitted_intensity, fit_info, spectrum, centroids_indexes, actual_peak_info):
    """ This method extracts features related to expected ions of interest and expected mixture chemicals. """

    expected_intensity = actual_peak_info['expected intensity']

    predicted_peak_mz = continuous_mz[numpy.where(fitted_intensity == max(fitted_intensity))]

    # extract information about subsequent (following) peaks after the major one
    # TODO: guarantee that sp_ratios is always of the same size (to keep the feature matrix of the same size)
    sp_number, sp_ratios = extract_sp_features(predicted_peak_mz,
                                               max(fitted_intensity),
                                               continuous_mz[-1],
                                               spectrum,
                                               centroids_indexes)

    left_tail_auc, right_tail_auc = extract_auc_features(spectrum, continuous_mz, fitted_intensity, predicted_peak_mz)

    peak_features = {
        # 'intensity': max(fitted_intensity, max(fit_info['raw intensity array'])),
        # # in this case goodness-of-fit does not tell much

        'intensity': max(fitted_intensity),
        'expected intensity diff': max(fitted_intensity) - expected_intensity,
        'expected intensity ratio': expected_intensity / max(fitted_intensity),
        'absolute mass accuracy': fit_info['fit_theory_absolute_ma'],
        'ppm': fit_info['fit_theory_ppm'],
        'widths': extract_width_features(continuous_mz, fitted_intensity),  # 20%, 50%, 80% of max intensity
        'subsequent peaks number': sp_number,
        'subsequent peaks ratios': sp_ratios,
        'left tail auc': left_tail_auc,
        'right tail auc': right_tail_auc,
        'symmetry': (left_tail_auc + right_tail_auc) / (2 * max(left_tail_auc, right_tail_auc)),
        'goodness-of-fit': fit_info['goodness-of-fit']
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
    """ This method extracts features of the following lower peaks after the major peak. """

    sp_number, sp_ratios = 0, []

    for index in centroids_indexes:

        if major_peak_mz <= spectrum['m/z array'][index] <= right_boundary_mz:

            sp_number += 1
            sp_ratios.append(spectrum['intensity array'][index] / major_peak_intensity)

        elif spectrum['m/z array'][index] > right_boundary_mz:
            break

    return sp_number, sp_ratios


def extract_width_features(continuous_mz, fitted_intensity):
    """ This method extract widths of different levels of the peak height. """

    widths = []
    for percent in widths_levels:
        # intensity on the desired level
        intensity = max(fitted_intensity) * percent
        residuals = abs(fitted_intensity - intensity)

        # find mz value of desired intensity
        mz = continuous_mz[residuals[numpy.where(residuals == min(residuals))]]

        width = 2 * (continuous_mz[numpy.where(fitted_intensity == max(fitted_intensity))] - mz)  # symmetry -> * 2

        widths.append(width)

    return widths


def get_peak_fit(peak_region, spectrum, theoretical_mz):
    """ This method fits the peak with a model and returns the fitted curve with fit information. """

    x, y = spectrum['m/z array'][peak_region[0]:peak_region[-1] + 1], \
           spectrum['intensity array'][peak_region[0]:peak_region[-1] + 1]

    if len(x) == len(y) == 3:

        # I should find out whether this happens at all with large expected peaks.

        # TODO: implement feature extraction out of raw data and remove ValueError
        raise ValueError("Don't fit function to 3 points. Better calculate features from raw data.")

    else:

        g_model = GaussianModel()
        g_pars = g_model.guess(y, x=x)
        g_out = g_model.fit(y, g_pars, x=x)

        # define d as peak resolution (i.e. width on the 50% of the height)
        d, predicted_peak_mz = ms_operator.get_peak_width_and_predicted_mz(peak_region, spectrum, g_out)

        xc = numpy.linspace(predicted_peak_mz - prf * d, predicted_peak_mz + prf * d, 5000)
        yc = g_out.eval(xc)

        # now compose fit information

        # find absolute mass accuracy and ppm for signal related to fit
        signal_fit_mass_diff = x[numpy.where(y == max(y))] - predicted_peak_mz
        signal_fit_ppm = signal_fit_mass_diff / predicted_peak_mz * 10 ** 6

        # find absolute mass accuracy and ppm for fit related to expected (theoretical) value
        fit_theory_mass_diff = predicted_peak_mz - theoretical_mz
        fit_theory_ppm = fit_theory_mass_diff / theoretical_mz * 10 ** 6

        fit_info = {
            'model': 'gaussian',
            'goodness-of-fit': g_out.redchi,  # goodness-of-fit is reduced chi-squared
            'fit_theory_absolute_ma': fit_theory_mass_diff,  # fitted absolute mass accuracy
            'fit_theory_ppm': fit_theory_ppm,  # ppm between fitted peak mz and expected (theoretical) mz
            'resolution': d,
            'raw intensity array': y,

            # probably redundant information
            'signal_fit_absolute_ma': signal_fit_mass_diff,
            'signal_fit_ppm': signal_fit_ppm
        }

        return xc, yc, fit_info


def fit_peak_and_extract_features(actual_peak, spectrum, centroids_indexes):
    """ This method takes index of peak, gets fitting region, fits the pick
        and extracts information out of fitted function. """

    peak_region_indexes = ms_operator.get_peak_fitting_region(spectrum, actual_peak['index'])

    fitted_mz, fitted_intensity, fit_info = get_peak_fit(peak_region_indexes, spectrum, actual_peak['expected mz'])

    peak_features = extract_peak_features(fitted_mz, fitted_intensity, fit_info,
                                          spectrum, centroids_indexes, actual_peak)

    peak_fit = {
        'expected mz': actual_peak['expected mz'],  # this is an id of the peak
        'mz': fitted_mz,
        'intensity': fitted_intensity,
        'info': fit_info
    }

    return peak_fit, peak_features


def extract_unexpected_noise_features():
    """ This method extracts features related to unexpected noise of instrumental and chemical nature. """

    # TODO

    pass


def find_isotope_and_extract_features(major_peak_index, actual_peaks_info, peak_fits):
    """ This method looks for the isotope in the list of peaks fits, gets its predicted intensity and mz,
        and calculates features using the major peak fit (major peak). """

    major_peak_fitted_intensity = peak_fits[major_peak_index]['intensity']
    major_peak_continuous_mz = peak_fits[major_peak_index]['mz']

    major_peak_max_intensity = max(major_peak_fitted_intensity)
    major_peak_mz = major_peak_continuous_mz[numpy.where(major_peak_fitted_intensity == major_peak_max_intensity)]

    isotope_intensity_ratios = []
    isotope_mass_diff_values = []

    for j in range(len(actual_peaks_info[major_peak_index]['expected isotopes'])):

        # find each isotope in the peak fits list
        for k in range(len(peak_fits)):
            if peak_fits[k]['expected mz'] == actual_peaks_info[major_peak_index]['expected isotopes'][j]:

                # ratio between isotope intensity and its major ions intensity
                max_isotope_intensity = max(peak_fits[k]['intensity'])
                ratio = max_isotope_intensity / major_peak_max_intensity

                # m/z diff between isotope and its major ion (how far is the isotope)
                isotope_mz = peak_fits[k]['mz'][numpy.where(peak_fits[k]['intensity'] == max_isotope_intensity)]
                mass_diff = isotope_mz - major_peak_mz

                isotope_intensity_ratios.append(ratio)
                isotope_mass_diff_values.append(mass_diff)

                break

    isotopic_features = {
        'isotopes mzs': actual_peaks_info[major_peak_index]['expected isotopes'],  # in case id is needed
        'intensity ratios': isotope_intensity_ratios,
        'mass diff values': isotope_mass_diff_values
    }

    return isotopic_features


def find_fragment_and_extract_features(major_peak_index, actual_peaks_info, peak_fits):
    """ This method looks for the fragment in the list of peaks fits, gets its predicted intensity and mz,
        and calculates features using the major peak fit (major peak). """

    major_peak_fitted_intensity = peak_fits[major_peak_index]['intensity']
    major_peak_continuous_mz = peak_fits[major_peak_index]['mz']

    major_peak_max_intensity = max(major_peak_fitted_intensity)
    major_peak_mz = major_peak_continuous_mz[numpy.where(major_peak_fitted_intensity == major_peak_max_intensity)]

    fragment_intensity_ratios = []
    fragment_mass_diff_values = []

    for j in range(len(actual_peaks_info[major_peak_index]['expected fragments'])):

        # find each fragment in the peak fits list
        for k in range(len(peak_fits)):
            if peak_fits[k]['expected mz'] == actual_peaks_info[major_peak_index]['expected fragments'][j]:

                # ratio between fragment intensity and its major ions intensity
                max_fragment_intensity = max(peak_fits[k]['intensity'])
                ratio = max_fragment_intensity / major_peak_max_intensity

                # m/z diff between fragment and its major ion (how far is the fragment)
                fragment_mz = peak_fits[k]['mz'][numpy.where(peak_fits[k]['intensity'] == max_fragment_intensity)]
                mass_diff = major_peak_mz - fragment_mz

                fragment_intensity_ratios.append(ratio)
                fragment_mass_diff_values.append(mass_diff)

                break

    fragmentation_features = {
        'fragments mzs': actual_peaks_info[major_peak_index]['expected fragments'],  # in case id is needed
        'intensity ratios': fragment_intensity_ratios,
        'mass diff values': fragment_mass_diff_values
    }

    return fragmentation_features


def get_null_peak_features():
    """ Compose the empty dictionary with peak features
        to keep the whole features matrix of the same dimensionality. """

    missing_peak_features = {
        # 'intensity': max(fitted_intensity, max(fit_info['raw intensity array'])),
        # # in this case goodness-of-fit does not tell much

        'intensity': -1,
        'expected intensity diff': -1,
        'expected intensity ratio': -1,
        'absolute mass accuracy': -1,
        'ppm': -1,
        'widths': [-1, -1, -1],  # 20%, 50%, 80% of max intensity
        'subsequent peaks number': -1,
        'subsequent peaks ratios': [],  # TODO: to add as many -1s as sp_rations length is fixed
        'left tail auc': -1,
        'right tail auc': -1,
        'symmetry': -1,
        'goodness-of-fit': -1
    }

    return missing_peak_features


def get_null_peak_fit(actual_peak):
    """ Compose the empty dictionary with peak fit for a missing peak to keep dimensionality of the data structure. """

    missing_peak_fit = {
        'expected mz': actual_peak['expected mz'],  # this is an id of the peak
        'mz': -1,
        'intensity': -1,
        'info': {}
    }

    return missing_peak_fit


def get_null_isotopic_features(actual_peak_info):
    """ Compose the empty dictionary with isotopic features for a missing peak
        to keep the whole features matrix of the same dimensionality. """

    missing_isotopic_features = {
        'isotopes mzs': actual_peak_info['expected isotopes'],  # in case id is needed
        'intensity ratios': [-1 for value in actual_peak_info['expected isotopes']],
        'mass diff values': [-1 for value in actual_peak_info['expected isotopes']]
    }

    return missing_isotopic_features


def get_null_fragmentation_features(actual_peak_info):
    """ Compose the empty dictionary with isotopic features for a missing peak
        to keep the whole features matrix of the same dimensionality. """

    missing_fragmentation_features = {
        'fragments mzs': actual_peak_info['expected fragments'],  # in case id is needed
        'intensity ratios': [-1 for value in actual_peak_info['expected fragments']],
        'mass diff values': [-1 for value in actual_peak_info['expected fragments']]
    }

    return missing_fragmentation_features


if __name__ == '__main__':

    start_time = time.time()

    good_example = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'
    spectra = list(mzxml.read(good_example))

    print('\n', time.time() - start_time, "seconds elapsed for reading")

    mid_spectrum = spectra[43]  # nice point on chromatogram

    # peak picking here
    centroids_indexes, properties = signal.find_peaks(mid_spectrum['intensity array'], height=minimal_peak_intensity)

    example_file = "/Users/andreidm/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"
    expected_ions_info = parser.parse_expected_ions(example_file)

    actual_peaks = ms_operator.find_closest_centroids(mid_spectrum['m/z array'], centroids_indexes, expected_ions_info)

    independent_peaks_features = []
    independent_peak_fits = []  # there is a need to store temporarily peak fitting results

    # extract peaks features independently
    for i in range(len(actual_peaks)):

        if actual_peaks[i]['present']:

            peak_fit, peak_features = fit_peak_and_extract_features(actual_peaks[i], mid_spectrum, centroids_indexes)

            independent_peaks_features.append(peak_features)
            independent_peak_fits.append(peak_fit)

        else:
            # save the same dimensionality with actual peaks structure
            # ans keep the size of the feature matrix constant

            null_peak_features = get_null_peak_features()
            null_peak_fit = get_null_peak_fit(actual_peaks[i])

            independent_peaks_features.append(null_peak_features)
            independent_peak_fits.append(null_peak_fit)

    # extract features related to ions isotopic abundance and fragmentation
    for i in range(len(actual_peaks)):

        if actual_peaks[i]['present']:
            if len(actual_peaks[i]['expected isotopes']) > 0:
                isotopic_features = find_isotope_and_extract_features(i, actual_peaks, independent_peak_fits)

            elif len(actual_peaks[i]['expected fragments']) > 0:
                fragmentation_features = find_fragment_and_extract_features(i, actual_peaks, independent_peak_fits)

            else:
                pass

        else:
            # fill the data structure with null values
            if len(actual_peaks[i]['expected isotopes']) > 0:
                isotopic_features = get_null_isotopic_features(actual_peaks[i])

            elif len(actual_peaks[i]['expected fragments']) > 0:
                fragmentation_features = get_null_fragmentation_features(actual_peaks[i])

            else:
                pass

    # merge isotopic and fragmentation features with independent peaks features

    # TODO

    print('\n', time.time() - start_time, "seconds elapsed in total")
