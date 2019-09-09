

import numpy, random
from scipy import signal
from pyteomics import mzxml
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from lmfit.models import Pearson7Model
from lmfit.models import DampedOscillatorModel
from lmfit.models import SkewedGaussianModel, SkewedVoigtModel, DonaichModel
from src.msfe.ms_operator import get_peak_fitting_region
from matplotlib import pyplot as plt


spectra = list(mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))
mid_spectrum = spectra[43]  # nice point on chromatogram

peaks, properties = signal.find_peaks(mid_spectrum['intensity array'], height=100)

models = ['Gauss', 'Lorenz', 'Voigt', 'Pearson', 'Damped Oscillator', 'Skewed Gauss', 'Skewed Voigt', 'Donaich']
# # best models are 'Donaich', 'Skewed Voigt', 'Skewed Gauss', 'Damped Oscillator', 'Gauss'

with open('/Users/andreidm/ETH/projects/ms_feature_extractor/res/models_race.txt', 'a') as file:

    champions = []

    for k in range(1):
        # 10 rounds

        total_score = [0 for name in models]
        wins = [0 for name in models]

        for i in range(10):

            random_index = random.randint(5,len(peaks)-5)
            random_peak_region = get_peak_fitting_region(mid_spectrum, peaks[random_index])

            x, y = mid_spectrum['m/z array'][random_peak_region[0]:random_peak_region[-1]+1],\
                   mid_spectrum['intensity array'][random_peak_region[0]:random_peak_region[-1]+1]

            if len(x) == len(y) == 3:
                continue
            else:

                g_model = GaussianModel()

                g_pars = g_model.guess(y, x=x)
                g_out = g_model.fit(y, g_pars, x=x)
                g_qc = g_out.redchi

                # print(g_out.fit_report(min_correl=0.25))

                l_model = LorentzianModel()

                l_pars = l_model.guess(y, x=x)
                l_out = l_model.fit(y, l_pars, x=x)
                l_qc = l_out.redchi

                # print(l_out.fit_report(min_correl=0.25))

                v_model = VoigtModel()

                v_pars = v_model.guess(y, x=x)
                v_out = v_model.fit(y, v_pars, x=x)
                v_qc = v_out.redchi

                # print(v_out.fit_report(min_correl=0.25))

                # other models to try

                do_model = DampedOscillatorModel()
                do_pars = do_model.guess(y, x=x)
                do_out = do_model.fit(y, do_pars, x=x)
                do_qc = do_out.redchi

                sg_model = SkewedGaussianModel()
                sg_pars = sg_model.guess(y, x=x)
                sg_out = sg_model.fit(y, sg_pars, x=x)
                sg_qc = sg_out.redchi

                sv_model = SkewedVoigtModel()
                sv_pars = sv_model.guess(y, x=x)
                sv_out = sv_model.fit(y, sv_pars, x=x)
                sv_qc = sv_out.redchi

                d_model = DonaichModel()
                d_pars = d_model.guess(y, x=x)
                d_out = d_model.fit(y, d_pars, x=x)
                d_qc = d_out.redchi

                # try-except for Pearson as if fails sometimes
                try:
                    p_model = Pearson7Model()
                    p_pars = p_model.guess(y, x=x)
                    p_out = p_model.fit(y, p_pars, x=x)
                    p_qc = p_out.redchi

                except ValueError:
                    p_qc = max([g_qc, l_qc, v_qc, do_qc, sg_qc, sv_qc, d_qc]) \
                           + min([g_qc, l_qc, v_qc, do_qc, sg_qc, sv_qc, d_qc])  # fine for failure

                scores = [g_qc, l_qc, v_qc, p_qc, do_qc, sg_qc, sv_qc, d_qc]

                print("Reduced chi-squared:", g_qc, l_qc, v_qc, p_qc, do_qc, sg_qc, sv_qc, d_qc)

                for j in range(len(models)):
                    total_score[j] += scores[j]

                wins[scores.index(min(scores))] += 1

                xc = numpy.linspace(mid_spectrum['m/z array'][random_peak_region[0]],
                                    mid_spectrum['m/z array'][random_peak_region[-1]+1],
                                    50)
                plt.figure()
                plt.plot(x, y, 'k.', lw=1)
                plt.plot(xc, g_out.eval(x=xc), lw=1, label='Gauss')
                plt.plot(xc, l_out.eval(x=xc), lw=1, label='Lorenz')
                plt.plot(xc, v_out.eval(x=xc), lw=1, label='Voigt')
                plt.plot(xc, p_out.eval(x=xc), lw=1, label='Pearson')
                plt.plot(xc, do_out.eval(x=xc), lw=1, label='Damped Oscillator')
                plt.plot(xc, sg_out.eval(x=xc), lw=1, label='Skewed Gauss')
                plt.plot(xc, sv_out.eval(x=xc), lw=1, label='Skewed Voigt')
                plt.plot(xc, d_out.eval(x=xc), lw=1, label='Donaich')

                plt.legend()

        plt.show()

        champions.append( (models[total_score.index(min(total_score))], models[wins.index(max(wins))]) )
        total_score = numpy.array(total_score) / 1000000

        # file.write(", ".join(map(str, models)) + "\n")
        # file.write(", ".join(map(str, wins)) + " (" + str(sum(wins)) + " out of 500)\n")
        # file.write(", ".join(map(str, total_score)) + "\n")

        print(models)
        print(wins, ",", sum(wins), "out of 100")
        print(numpy.array(total_score) / 1000000)

print()
for champion_pair in champions:
    print(champion_pair)
