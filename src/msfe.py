""" MS feature extractor """

import time, numpy
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import parser, ms_operator
from src.constants import peak_region_factor as prf
from lmfit.models import GaussianModel


def extract_peak_features(continuous_mz, fitted_intensity, fit_info, spectrum, centroids_indexes):
    """ This method extracts features related to expected ions of interest and expected mixture chemicals. """

    # TODO

    predicted_peak_mz = continuous_mz[numpy.where(fitted_intensity == max(fitted_intensity))]

    # extract information about subsequent (following) peaks after the major one
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
        'mass accuracy': fit_info['ma'],
        'widths': extract_width_features(continuous_mz, fitted_intensity),  # 20%, 50%, 80% of max intensity
        'subsequent peaks number': sp_number,
        'subsequent peaks ratios': sp_ratios,
        'left tail auc': left_tail_auc,
        'right tail auc': right_tail_auc,
        'goodness-of-fit': fit_info['gof']
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
    for percent in [0.2, 0.5, 0.8]:
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

        xc = numpy.linspace(predicted_peak_mz - prf * d, predicted_peak_mz + prf * d, 1000)
        yc = g_out.eval(xc)

        # find mass accuracy and ppm
        mass_accuracy = abs(x[numpy.where(y == min(y))] - predicted_peak_mz)
        fitted_ppm = (predicted_peak_mz - theoretical_mz) / theoretical_mz * 10 ** 6

        # compose fit information
        fit_info = {
            'model': 'gaussian',
            'gof': g_out.redchi,  # goodness-of-fit is reduced chi-squared
            'ma': mass_accuracy,  # fitted mass accuracy
            'fitted ppm': fitted_ppm,  # ppm between fitted peak mz and expected (theoretical) mz
            'resolution': d,
            'raw intensity array': y
        }

        return xc, yc, fit_info


def fit_peak_and_extract_features(actual_peak, spectrum, centroids_indexes):
    """ This method takes index of peak, gets fitting region, fits the pick
        and extracts information out of fitted function. """

    peak_region_indexes = ms_operator.get_peak_fitting_region(spectrum, actual_peak['index'])

    fitted_mz, fitted_intensity, fit_info = get_peak_fit(peak_region_indexes, spectrum, actual_peak['expected mz'])

    peak_features = extract_peak_features(fitted_mz, fitted_intensity, fit_info, spectrum, centroids_indexes)

    return peak_features


def extract_unexpected_noise_features():
    """ This method extracts features related to unexpected noise of instrumental and chemical nature. """
    pass


if __name__ == '__main__':

    start_time = time.time()

    good_example = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'
    spectra = list(mzxml.read(good_example))

    print('\n', time.time() - start_time, "seconds elapsed for reading")

    mid_spectrum = spectra[43]  # nice point on chromatogram

    # peak picking here
    peaks_indexes, properties = signal.find_peaks(mid_spectrum['intensity array'], height=100)

    example_file = "/Users/andreidm/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"
    expected_mzs, expected_intensities = parser.parse_expected_peaks(example_file)

    actual_peaks = ms_operator.find_closest_centroids(mid_spectrum['m/z array'], peaks_indexes, expected_mzs)

    peaks_feature_matrix = []
    for i in range(len(actual_peaks)):

        if actual_peaks[i]['present']:

            peak_features = fit_peak_and_extract_features(actual_peaks[i], mid_spectrum, peaks_indexes)
            peaks_feature_matrix.append(peak_features)

    print('\n', time.time() - start_time, "seconds elapsed in total")
