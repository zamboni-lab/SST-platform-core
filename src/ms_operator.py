
import numpy
from scipy import signal
from matplotlib import pyplot as plt
from pyteomics import mzxml
import time


def extract_mz_region(spectrum, interval):

    mz_values = []
    intensities = []

    for i in range(len(spectrum['m/z array'])):

        if interval[0] <= spectrum['m/z array'][i] <= interval[1]:
            mz_values.append(spectrum['m/z array'][i])
            intensities.append(spectrum['intensity array'][i])

        elif spectrum['m/z array'][i] > interval[1]:
            break

    return numpy.array(mz_values), numpy.array(intensities)


if __name__ == '__main__':

    spectra = list(mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))

    mid_spectrum = spectra[43]  # nice point on chromatogram

    mz_region, intensities = extract_mz_region(mid_spectrum,[200,400])

    # peak picking

    plt.plot(mz_region, intensities, lw=1)

    start_time = time.time()

    # peak_indices = signal.find_peaks_cwt(intensities, numpy.arange(1,32), min_snr=1, noise_perc=55)

    # this pair of widths and noise percent allows identification of everything beyond 100 intensity value (visually)
    # the larger widths the less number of relevant peaks identified
    # the larger noise percent the more number of redundant peaks identified
    peak_indices = signal.find_peaks_cwt(intensities, [0.5], min_snr=1, noise_perc=5)

    print('\n',time.time() - start_time, "seconds elapsed\n")

    print(peak_indices, mz_region[peak_indices], intensities[peak_indices])

    print("\nTotal number of peaks = ", len(peak_indices))

    plt.plot(mz_region[peak_indices], intensities[peak_indices], 'gx', lw=1)

    plt.show()
