
ms_spectrum_names = ['num', 'scanType', 'centroided', 'msLevel', 'peaksCount', 'polarity', 'retentionTime',\
                    'lowMz', 'highMz', 'basePeakMz', 'basePeakIntensity', 'totIonCurrent', 'msInstrumentID',\
                    'id', 'm/z array', 'intensity array']

parser_comment_symbol = '#'
parser_description_symbols = '()'

minimal_peak_intensity = 100

peak_widths_levels_of_interest = [0.2, 0.5, 0.8]

allowed_ppm_error = 0.25

peak_region_factor = 3  # 3 times resolution is a region for extracting information

maximum_number_of_subsequent_peaks_to_consider = 50  # initial guess
