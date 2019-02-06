
import numpy, time
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import ms_operator
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets


missing_peak_features = {
        # 'intensity': max(fitted_intensity, max(fit_info['raw intensity array'])),
        # # in this case goodness-of-fit does not tell much

        'intensity': -1,
        'expected intensity diff': -1,
        'expected intensity ratio': -1,
        'absolute mass accuracy': -1,
        'ppm': -1,
        'widths': [-1, -1, -1],  # 20%, 50%, 80% of max intensity
        'subsequent peaks number': -1,
        'subsequent peaks ratios': [-1 for value in range(10)],
        'left tail auc': -1,
        'right tail auc': -1,
        'symmetry': -1,
        'goodness-of-fit': -1
    }

print(list(missing_peak_features.keys()))

print(isinstance(missing_peak_features['widths'],float))