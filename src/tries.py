
import numpy, time
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src.parser import parse_expected_peaks
from src import ms_operator


# good_example = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'
# spectra = list(mzxml.read(good_example))
#
# mid_spectrum = spectra[43]
#
# print(mid_spectrum['m/z array'].index(mid_spectrum['m/z array'][3]))

x1 = numpy.linspace(1,10,100)

x2 = numpy.linspace(0.5,10.5,10)

y1 = numpy.log(x1)

y2 = numpy.log(x2) + numpy.sin(x2)


# integral1 = numpy.trapz(y1, x1)
# integral2 = numpy.trapz(y2, x2)
#
# integral3 = integral2 - integral1

# y3 = y2 - y1


fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

ax0.plot(x1, y1, 'r.')
ax0.plot(x2, y2, 'b*')

ax1.fill_between(x1, 0, y1, alpha=0.5)
ax1.set_ylabel('between y1 and 0')

ax1.fill_between(x2, 0, y2, alpha=0.5)
ax1.set_ylabel('between y2 and 0')

plt.show()