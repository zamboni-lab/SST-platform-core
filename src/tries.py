
import numpy, time
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import ms_operator
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets
from lmfit.models import GaussianModel


y = [5909.0, 164092.0, 1183368.0, 1493003.0, 1493535.0, 1493535.0, 1493535.0, 1493535.0, 1493535.0, 1493535.0, 1493535.0, 1493535.0, 1493452.0, 1491589.0, 1471912.0, 1423399.0, 1352991.0, 1255359.0, 914992.0, 690754.0]
x = [126.89574972982028, 126.89771016928717, 126.89967062389752, 126.9016310936514, 126.90359159020302, 126.9055520902439, 126.90751260542828, 126.90947313575616, 126.91143369288325, 126.91339425349814, 126.91535482925651, 126.91731542015839, 126.91927603786093, 126.9212366590498, 126.92319729538217, 126.92515794685802, 126.92711862513607, 126.92907930689894, 126.93104000380529, 126.93300071585516]

cy = [y[0]]  # corrected y
cx = [x[0]]  # corrected x

for i in range(1,len(y)-1):
    if (y[i] == y[i-1] or y[i] == y[i+1]) and y[i] == max(y):
        # this is saturation
        pass
    else:
        cy.append(y[i])
        cx.append(x[i])

cy.append(y[-1])
cx.append(x[-1])

ncy = cy[0:4]
ncy.extend(cy[-4:len(cx)])

ncx = cx[0:4]
ncx.extend(cx[-4:len(cx)])

ncy = numpy.array(ncy)
ncx = numpy.array(ncx)

cy = numpy.array(cy)
cx = numpy.array(cx)

g_model = GaussianModel()
g_pars = g_model.guess(ncy, x=ncx)
g_out_2 = g_model.fit(ncy, g_pars, x=ncx)

g_model = GaussianModel()
g_pars = g_model.guess(cy, x=cx)
g_out_1 = g_model.fit(cy, g_pars, x=cx)

ncxc = numpy.linspace(ncx[0], ncx[-1], 1000)
ncyc = g_out_2.eval(x=ncxc)

cxc = numpy.linspace(cx[0], cx[-1], 1000)
cyc = g_out_1.eval(x=cxc)


plt.plot(x,y)
plt.plot(ncx,ncy, 'r.')
plt.plot(cx,cy, 'b.')
# plt.plot(ncxc, ncyc)
plt.plot(cxc, cyc)
plt.show()
