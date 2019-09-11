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

a = []
import numpy
print(sum(numpy.array(a) != -1.))