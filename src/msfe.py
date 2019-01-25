""" MS feature extractor """

import time, numpy
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import parser, ms_operator
from lmfit.models import DampedOscillatorModel, SkewedGaussianModel, SkewedVoigtModel, DonaichModel, GaussianModel


def extract_peak_features(continuous_mz, fitted_intensity):
    """ This method extracts features related to expected ions of interest and expected mixture chemicals. """

    peak_features = {}

    # TODO

    return peak_features


def get_peak_fit(peak_region, spectrum):
    """ This method fits the peak with models and returns the best fitted curve. """

    x, y = spectrum['m/z array'][peak_region[0]:peak_region[-1] + 1], \
           spectrum['intensity array'][peak_region[0]:peak_region[-1] + 1]

    if len(x) == len(y) == 3:
        # TODO: implement feature extraction out of raw data and remove ValueError
        raise ValueError("Don't fit function to 3 points. Better calculate features from raw data.")

    else:

        g_model = GaussianModel()
        g_pars = g_model.guess(y, x=x)
        g_out = g_model.fit(y, g_pars, x=x)

        do_model = DampedOscillatorModel()
        do_pars = do_model.guess(y, x=x)
        do_out = do_model.fit(y, do_pars, x=x)

        sg_model = SkewedGaussianModel()
        sg_pars = sg_model.guess(y, x=x)
        sg_out = sg_model.fit(y, sg_pars, x=x)

        sv_model = SkewedVoigtModel()
        sv_pars = sv_model.guess(y, x=x)
        sv_out = sv_model.fit(y, sv_pars, x=x)

        d_model = DonaichModel()
        d_pars = d_model.guess(y, x=x)
        d_out = d_model.fit(y, d_pars, x=x)

        fitted_models = [g_out, do_out, sg_out, sv_out, d_out]
        fitting_metrics = [g_out.redchi, do_out.redchi, sg_out.redchi, sv_out.redchi, d_out.redchi]

        best_model = fitted_models[fitting_metrics.index(min(fitting_metrics))]

    xc = numpy.linspace(spectrum['m/z array'][peak_region[0]], spectrum['m/z array'][peak_region[-1]+1], 50)
    yc = best_model.eval(xc)

    return xc, yc


def fit_peak_and_extract_features(actual_peak, spectrum):
    """ This method takes index of peak, gets fitting region, fits the pick
        and extracts information out of fitted function. """

    peak_region = ms_operator.get_peak_fitting_region(mid_spectrum, actual_peak['index'])

    fitted_mz, fitted_intensity = get_peak_fit(peak_region, mid_spectrum)

    peak_features = extract_peak_features(fitted_mz, fitted_intensity)

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
