
import numpy, time
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src.parser import parse_expected_peaks

start_time = time.time()

spectra = list(mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))

print('\n', time.time() - start_time, "seconds elapsed for reading")

mid_spectrum = spectra[43]  # nice point on chromatogram

# peak picking here
peaks, properties = signal.find_peaks(mid_spectrum['intensity array'], height=100)




expected_mzs, expected_intensities = parse_expected_peaks("/Users/andreidm/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt")





# plt.plot(mid_spectrum['m/z array'], mid_spectrum['intensity array'], lw=1)
# plt.plot(mid_spectrum['m/z array'][peaks], mid_spectrum['intensity array'][peaks], 'gx', lw=1)
#
# plt.show()

print('\n',time.time() - start_time, "seconds elapsed in total")