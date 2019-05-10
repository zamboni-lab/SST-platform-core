
import numpy, time
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import ms_operator
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets
from lmfit.models import GaussianModel

from pyopenms import *

# isotopes = EmpiricalFormula("C13H11F2N6O-").getIsotopeDistribution(CoarseIsotopePatternGenerator(3))
# isotopes1 = EmpiricalFormula("C7H7N4O2-").getIsotopeDistribution(CoarseIsotopePatternGenerator(3))
# isotopes2 = EmpiricalFormula("C7H7N4O2").getIsotopeDistribution(CoarseIsotopePatternGenerator(3))
#
# isotopes1 = isotopes1.getContainer()
# isotopes2 = isotopes2.getContainer()
#
# for i in range(len(isotopes1)):
#     if isotopes1[i].getMZ() == isotopes2[i].getMZ() and isotopes1[i].getIntensity() == isotopes2[i].getIntensity():
#         print(True)
#


# ions_ids = ["damn" + "it" + str(i) for i in range(len(isotopes))]


# llist = ["Caffeine", "C8H9N4O2-", "C7H7N4O2-"]
#
# # correct ions names: all uppercase + no - in the end
# for i in range(1,len(llist)):
#     if llist[i][-1] == '-':
#         llist[i] = llist[i][:-1].upper()
#     else:
#         llist[i] = llist[i].upper()
#
# print(llist)

id = '1'

d = {'peak'+id: 3 }

print(d)

a = sum([1 > 0, 2. > 0, 3 < 1. ])

print(isinstance(a, int))