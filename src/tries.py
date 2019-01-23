
import numpy, operator
from scipy import signal
from matplotlib import pyplot as plt
from pyteomics import mzxml
import time
from src import ms_operator, matlab_caller
import random

from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from lmfit.models import Pearson7Model
from lmfit.models import DampedOscillatorModel
from lmfit.models import SkewedGaussianModel, SkewedVoigtModel, DonaichModel

from src.ms_operator import extract_mz_region


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


spectra = list(mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))
mid_spectrum = spectra[43]  # nice point on chromatogram

peaks, properties = signal.find_peaks(mid_spectrum['intensity array'], height=100)

models = ['Gauss', 'Lorenz', 'Voigt', 'Pearson', 'Damped Oscillator', 'Skewed Gauss', 'Skewed Voigt', 'Donaich']
total_score = [0 for name in models]
wins = [0 for name in models]

for i in range(100):

    random_index = random.randint(0,len(peaks))
    random_peak_region = get_peak_fitting_region(mid_spectrum, peaks[random_index])

    x, y = mid_spectrum['m/z array'][random_peak_region[0]:random_peak_region[-1]+1],\
           mid_spectrum['intensity array'][random_peak_region[0]:random_peak_region[-1]+1]

    if len(x) == len(y) == 3:
        continue
    else:

        g_model = GaussianModel()

        g_pars = g_model.guess(y, x=x)
        g_out = g_model.fit(y, g_pars, x=x)

        # print(g_out.fit_report(min_correl=0.25))

        l_model = LorentzianModel()

        l_pars = l_model.guess(y, x=x)
        l_out = l_model.fit(y, l_pars, x=x)

        # print(l_out.fit_report(min_correl=0.25))

        v_model = VoigtModel()

        v_pars = v_model.guess(y, x=x)
        v_out = v_model.fit(y, v_pars, x=x)

        # print(v_out.fit_report(min_correl=0.25))

        # other models to try

        # TODO: try-catch for nan input values or figure out the error

        p_model = Pearson7Model()
        p_pars = p_model.guess(y, x=x)
        p_out = p_model.fit(y, p_pars, x=x)

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

        print("Reduced chi-squared:", g_out.redchi, l_out.redchi, v_out.redchi,
                                      p_out.redchi, do_out.redchi, sg_out.redchi, sv_out.redchi, d_out.redchi)

        scores = [g_out.redchi, l_out.redchi, v_out.redchi, p_out.redchi, do_out.redchi, sg_out.redchi, sv_out.redchi, d_out.redchi]

        for j in range(len(models)):
            total_score[j] += scores[j]

        wins[scores.index(min(scores))] += 1

        # xc = numpy.linspace(mid_spectrum['m/z array'][random_peak_region[0]],
        #                     mid_spectrum['m/z array'][random_peak_region[-1]+1],
        #                     50)
    #     plt.figure()
    #     plt.plot(x, y, 'k.', lw=1)
    #     plt.plot(xc, g_out.eval(x=xc), lw=1, label='Gauss')
    #     plt.plot(xc, l_out.eval(x=xc), lw=1, label='Lorenz')
    #     plt.plot(xc, v_out.eval(x=xc), lw=1, label='Voigt')
    #     plt.plot(xc, p_out.eval(x=xc), lw=1, label='Pearson')
    #     plt.plot(xc, do_out.eval(x=xc), lw=1, label='Damped Oscillator')
    #     plt.plot(xc, sg_out.eval(x=xc), lw=1, label='Skewed Gauss')
    #     plt.plot(xc, sv_out.eval(x=xc), lw=1, label='Skewed Voigt')
    #     plt.plot(xc, d_out.eval(x=xc), lw=1, label='Donaich')
    #
    #     plt.legend()
#
# plt.show()

print(models)
print(wins)
print(numpy.array(total_score) / 1000000)
