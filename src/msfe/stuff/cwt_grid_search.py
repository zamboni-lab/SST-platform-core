from src.msfe import ms_operator
import numpy
from scipy import signal
from pyteomics import mzxml
import operator

if __name__ == '__main__':

    spectra = list(mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))

    mid_spectrum = spectra[43]

    mz_region, intensities = ms_operator.extract_mz_region(mid_spectrum, [600, 900])

    # optimize peak picking with grid search

    experiments = []

    snr_values = [1, 3, 5, 10, 15, 20, 25, 30, 35, 50, 60, 70, 80, 90]
    noise_percentages = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]

    progress = 0.
    for snr in snr_values:
        for noise in noise_percentages:

            progress = progress + 100 / len(snr_values) / len(noise_percentages)
            print("\nProgressing: ", progress, "%\n")

            peak_indices = signal.find_peaks_cwt(intensities, numpy.arange(1,32), min_snr=snr, noise_perc=noise)

            print("SNR = ", snr, ", Noise level = ", noise, "%")
            print("Total number of peaks = ", len(peak_indices))

            experiments.append(tuple([snr, noise, len(peak_indices)]))

    print()
    experiments.sort(key=operator.itemgetter(2), reverse=True)

    with open("/Users/andreidm/ETH/projects/ms_feature_extractor/results_mz_600_900.txt", 'a') as file:

        for experiment in experiments:
            print(experiment)

            file.write(str(experiment)+"\n")
