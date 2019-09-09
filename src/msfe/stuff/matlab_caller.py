import time
import zmq
from pymatbridge import Matlab


def call_peak_picking(filepath, scriptpath):
    """ Method connects to local Matlab via ZeroMQ and calls peak picking function there implemented in script path. """

    start_time = time.time()

    mlab = Matlab(executable='/Applications/MATLAB_R2018a_floating.app/bin/matlab')

    mlab.start()
    response = mlab.run_func(scriptpath, {'path': filepath})
    mlab.stop()

    peaks = response['result']

    print(time.time() - start_time, " seconds elapsed for peak picking")

    print("Total number of peaks", len(peaks))

    return peaks
