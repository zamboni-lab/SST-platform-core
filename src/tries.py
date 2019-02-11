
import numpy, time
from scipy import signal
from pyteomics import mzxml
from matplotlib import pyplot as plt
from src import ms_operator
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets


# x = [139.01505911, 139.01711103, 139.01916297, 139.02121493, 139.0232669,
#  139.02531888, 139.02737088, 139.0294229,  139.03147493, 139.03352697,
#  139.03557903, 139.03763112, 139.03968321, 139.04173531, 139.04378743]
#
# y = [   146. ,  1345. , 27248.  , 91233. ,114453.  ,74362.  ,20396.   ,8070.   ,7523.,
#    6772.   ,6452.   ,5653.   ,3262.   ,1207.    ,910.]
#
# expected = 139.02314287
#
# xc = [139.00249830786206, 139.00291536967762, 139.0033324314932, 139.00374949330873, 139.0041665551243,
#       139.00458361693987, 139.00500067875544, 139.00541774057098, 139.00583480238654, 139.0062518642021,
#       139.00666892601768, 139.00708598783322, 139.0075030496488, 139.00792011146436, 139.00833717327993,
#       139.0087542350955, 139.00917129691103, 139.0095883587266, 139.01000542054217, 139.01042248235774,
#       139.01083954417328, 139.01125660598885, 139.01167366780442, 139.01209072961998, 139.01250779143552,
#       139.0129248532511, 139.01334191506666, 139.01375897688223, 139.01417603869777, 139.01459310051334,
#       139.0150101623289, 139.01542722414447, 139.01584428596004, 139.01626134777558, 139.01667840959115,
#       139.01709547140672, 139.0175125332223, 139.01792959503783, 139.0183466568534, 139.01876371866896,
#       139.01918078048453, 139.01959784230007, 139.02001490411564, 139.0204319659312, 139.02084902774678,
#       139.02126608956235, 139.02168315137789, 139.02210021319345, 139.02251727500902, 139.0229343368246,
#       139.02335139864013, 139.0237684604557, 139.02418552227127, 139.02460258408684, 139.02501964590238,
#       139.02543670771794, 139.0258537695335, 139.02627083134908, 139.02668789316465, 139.0271049549802,
#       139.02752201679576, 139.02793907861133, 139.0283561404269, 139.02877320224243, 139.029190264058,
#       139.02960732587357, 139.03002438768914, 139.03044144950468, 139.03085851132025, 139.03127557313582,
#       139.03169263495138, 139.03210969676695, 139.0325267585825, 139.03294382039806, 139.03336088221363,
#       139.0337779440292, 139.03419500584474, 139.0346120676603, 139.03502912947587, 139.03544619129144,
#       139.03586325310698, 139.03628031492255, 139.03669737673812, 139.0371144385537, 139.03753150036923,
#       139.0379485621848, 139.03836562400036, 139.03878268581593, 139.0391997476315, 139.03961680944704,
#       139.0400338712626, 139.04045093307818, 139.04086799489374, 139.04128505670928, 139.04170211852485,
#       139.04211918034042, 139.042536242156, 139.04295330397153, 139.0433703657871, 139.04378742760267]
#
# yc = [2.6153127977259974e-12, 1.2225190833830208e-11, 5.536599841831198e-11, 2.4293277184629033e-10,
#       1.0327246795132827e-09, 4.253422132682911e-09, 1.697258132070813e-08, 6.561645907515148e-08,
#       2.457724299656509e-07, 8.918853227572594e-07, 3.1357422052954828e-06, 1.0681375096154317e-05,
#       3.525083933052869e-05, 0.00011271124279689241, 0.00034915679279512533, 0.0010479226644560817,
#       0.003047147322126985, 0.008584463352827162, 0.023430865485895028, 0.061961081753622894,
#       0.15874685589622106, 0.39404584676756593, 0.9476410319697455, 2.2079866350432455,
#       4.9843037876957865, 10.901042902419164, 23.09867531154278, 47.4200033397223,
#       94.31734497779831, 181.7510942788079, 339.32666003841126, 613.7823719465268,
#       1075.6384560134138, 1826.3066932651382, 3004.2535401366245, 4788.008724951517,
#       7393.137192106573, 11060.075158619884, 16030.3450960903, 22510.396567896816,
#       30625.197705568007, 40367.34038023207, 51550.971362779295, 63782.134646811784,
#       76456.90747744024, 88795.28103893758, 99912.20074805223, 108918.74910512015,
#       115038.24442947129, 117716.4960940576, 116704.57688249779, 112096.98476855055,
#       104317.0877232861, 94052.9578250553, 82157.07002780866, 69530.10862351766,
#       57010.69741917592, 45289.265822483576, 34856.97442550196, 25991.99266678965,
#       18777.810177162886, 13143.3415513032, 8912.963416954224, 5855.903815831705,
#       3727.5306760284598, 2298.8148732522927, 1373.542972221813, 795.126054767901,
#       445.94900461952824, 242.32035344538383, 127.57044467397334, 65.06774044730727,
#       32.154137351259564, 15.39442528685522, 7.140778773810944, 3.2090993130262793,
#       1.3972567530634337, 0.5894198501165719, 0.24089555070302213, 0.09538680022072388,
#       0.03659344213238931, 0.013601090366360825, 0.004897784471179395, 0.0017087600994144412,
#       0.0005775877844489083, 0.0001891517595252694, 6.001478465549549e-05, 1.8448522887706465e-05,
#       5.494401946088673e-06, 1.5853849676596298e-06, 4.432048714834401e-07, 1.2004105269498845e-07,
#       3.149999943421311e-08, 8.008418817118504e-09, 1.9725975504070726e-09, 4.707449635360164e-10,
#       1.088399555321338e-10, 2.4380719045163004e-11, 5.291272700180811e-12, 1.1125749057707498e-12]
#
#
# plt.plot(x,y,'k-')
# plt.plot(xc,yc,'b-')
# plt.axvline(x=expected)
# plt.show()

good_example = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'
# good_example = '/Users/dmitrav/ETH/projects/ms_feature_extractor/data/CsI_NaI_best_conc_mzXML/CsI_NaI_neg_08.mzXML'
spectra = list(mzxml.read(good_example))

centroids_indexes, properties = signal.find_peaks(spectra[43]['intensity array'], height=100)

plt.plot(spectra[43]['m/z array'], spectra[43]['intensity array'], lw=1)
plt.plot(spectra[43]['m/z array'][centroids_indexes], spectra[43]['intensity array'][centroids_indexes], 'rx',  lw=1)
plt.show()