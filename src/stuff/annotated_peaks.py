
import numpy, operator
from scipy import signal
from matplotlib import pyplot as plt
from pyteomics import mzxml
import time
from src import ms_operator

annotated_peaks = [
    [204.92, 204.94],
    [205.15, 205.17],
    [205.92, 205.94],
    [207.055, 207.075],
    [207.092, 207.110],
    [208.085, 208.10],
    [208.135, 208.150],
    [214.99, 215.015],
    [216.93, 216.95],
    [239.055, 239.075]
]

accurate_peak_locations = []

spectra = list(mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))
mid_spectrum = spectra[43]  # nice point on chromatogram

for mz_region in annotated_peaks:
    accurate_peak_locations.append(ms_operator.locate_annotated_peak(mz_region,mid_spectrum))

print(accurate_peak_locations)
