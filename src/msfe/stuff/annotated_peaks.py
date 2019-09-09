
from pyteomics import mzxml
from src.msfe import ms_operator

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

expected_peaks = [
    [126.90, 126.92],
    [139.01, 139.03],
    [276.79, 276.81],
    [271.84, 271.85],
    [1047.9, 1047.92],
    [1048.9, 1048.92]

]

spectra = list(mzxml.read('/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'))
mid_spectrum = spectra[43]  # nice point on chromatogram

accurate_peak_locations = []

for mz_region in expected_peaks:
    print(mz_region, "is being processed...")
    accurate_peak_locations.append(ms_operator.locate_annotated_peak(mz_region, mid_spectrum))

print("Done!")
print()
print(accurate_peak_locations)
