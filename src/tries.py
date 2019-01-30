
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

a = numpy.array([-1,0,1])

print(abs(a))