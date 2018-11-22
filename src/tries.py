
import numpy, operator
from scipy import signal
from matplotlib import pyplot as plt

widths = numpy.arange(4,6)

xs = numpy.arange(0, 50 * numpy.pi, 0.05)
data = numpy.sin(xs)
peakind = signal.find_peaks_cwt(data, widths)


print(widths)
print(peakind, xs[peakind], data[peakind])

plt.plot(xs, data)
plt.plot(xs[peakind], data[peakind], 'gx')
plt.show()