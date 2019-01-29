""" MS feature extractor """

import time, numpy
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import parser, ms_operator
from src.constants import peak_region_factor as prf
from lmfit.models import GaussianModel


def extract_peak_features(continuous_mz, fitted_intensity, fit_info):
    """ This method extracts features related to expected ions of interest and expected mixture chemicals. """

    peak_features = {}

    # TODO

    return peak_features


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
            'fitted ppm': fitted_ppm  # ppm between fitted peak mz and expected (theoretical) mz
        }

        return xc, yc, fit_info


def fit_peak_and_extract_features(actual_peak, spectrum):
    """ This method takes index of peak, gets fitting region, fits the pick
        and extracts information out of fitted function. """

    peak_region = ms_operator.get_peak_fitting_region(spectrum, actual_peak['index'])

    fitted_mz, fitted_intensity, fit_info = get_peak_fit(peak_region, spectrum, actual_peak['expected mz'])

    peak_features = extract_peak_features(fitted_mz, fitted_intensity, fit_info)

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

            peak_features = fit_peak_and_extract_features(actual_peaks[i], mid_spectrum)
            peaks_feature_matrix.append(peak_features)

    print('\n', time.time() - start_time, "seconds elapsed in total")
