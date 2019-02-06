
import numpy, time
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import ms_operator
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets

a = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]

print(numpy.percentile(a,[25,50, 75, 100]))