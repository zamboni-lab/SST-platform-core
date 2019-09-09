from src.msfe.stuff import matlab_caller
from src.msfe.ms_operator import extract_mz_region
from pyteomics import mzxml
from matplotlib import pyplot as plt
import time, signal


def get_corrected_peak_indices(cwt_peak_indices, intensities, step=3, min_intensity=50):
    """ This method takes results of CWT centroiding (indices of peaks in a spectrum) and corrects them.
        @ step is number of values to check to the left and to the right from CWT peak index. """

    corrected_peak_indices = []

    for index in cwt_peak_indices:

        for i in range(1-step, step):
            is_corrected = False

            if is_negligible_intensity(index+i, intensities, boundary=min_intensity):
                pass

            else:
                # check this is a peak
                if is_true_peak(index+i, intensities):

                    # check if this peak has been already counted
                    if not corrected_peak_indices.count(index+i) > 0:
                        corrected_peak_indices.append(index+i)
                        is_corrected = True
                    else:
                        pass

                if i == step-1 and not is_corrected:
                    # append peak index anyway to be inclusive processing peak shoulders
                    corrected_peak_indices.append(index)

    return corrected_peak_indices


def is_negligible_intensity(index, intensities, boundary=50):
    """ This method returns True if the intensity is too small to consider in peak picking.
        @ boundary is minimal intensity value to consider as potential peak, not noise"""

    if intensities[index] <= boundary:
        return True
    else:
        return False


def is_true_peak(index, intensities):
    """ This method returns True if ribs of the peak index are both descending. """

    if intensities[index-1] < intensities[index] and intensities[index] > intensities[index+1]:
        return True
    else:
        return False


def test_cwt_peak_picking():
    """ This is an old implementation of peak-picking with CWT with peaks correction and plotting.
        Super long (~306 s per scan). Thrashed."""

    spectra = list(mzxml.read(
        '/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))

    mid_spectrum = spectra[43]  # nice point on chromatogram

    # mz_region, intensities = extract_mz_region(mid_spectrum,[200,400])
    mz_region, intensities = extract_mz_region(mid_spectrum, [200, 250])

    # peak picking

    plt.plot(mz_region, intensities, lw=1)

    start_time = time.time()

    # peak_indices = signal.find_peaks_cwt(intensities, numpy.arange(1,32), min_snr=1, noise_perc=55)

    # this pair of widths and noise percent allows identification of everything beyond 100 intensity value (visually)
    # the larger widths the less number of relevant peaks identified
    # the larger noise percent the more number of redundant peaks identified
    cwt_peak_indices = signal.find_peaks_cwt(intensities, [0.5], min_snr=1, noise_perc=5)

    corrected_peak_indices = get_corrected_peak_indices(cwt_peak_indices, intensities, step=3, min_intensity=100)

    print('\n', time.time() - start_time, "seconds elapsed\n")

    # print(cwt_peak_indices, mz_region[cwt_peak_indices], intensities[cwt_peak_indices])

    print("\nTotal number of CWT peaks = ", len(cwt_peak_indices))
    print("\nTotal number of corrected peaks = ", len(corrected_peak_indices))

    plt.plot(mz_region[cwt_peak_indices], intensities[cwt_peak_indices], 'gx', lw=1)

    plt.plot(mz_region[corrected_peak_indices], intensities[corrected_peak_indices], 'r.', lw=1)

    plt.show()


def test_matlab_peak_picking(mid_spectrum):
    """ This is another old implementation of peak-picking with CWT of Matlab and plotting.
        Quite long (~16 s per scan). There is also a problem with matching peak indices from Matlab
        and actual raw data peaks. Thrashed."""

    # spectra = list(
    #     mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))
    #
    # mid_spectrum = spectra[43]  # nice point on chromatogram

    peaks = matlab_caller.call_peak_picking(scriptpath='/Users/andreidm/ETH/projects/fiaminer_peak_picking.m',
                                            filepath='/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML')

    mz_values = [peak[0] for peak in peaks]
    intensities = [peak[1] for peak in peaks]

    plt.plot(mid_spectrum['m/z array'], mid_spectrum['intensity array'], lw=1)
    plt.plot(mz_values, intensities, 'rx', lw=1)

    plt.show()
