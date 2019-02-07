
ms_spectrum_names = ['num', 'scanType', 'centroided', 'msLevel', 'peaksCount', 'polarity', 'retentionTime',\
                    'lowMz', 'highMz', 'basePeakMz', 'basePeakIntensity', 'totIonCurrent', 'msInstrumentID',\
                    'id', 'm/z array', 'intensity array']

parser_comment_symbol = '#'
parser_description_symbols = '()'

minimal_peak_intensity = 100

# expected_peaks_file_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"
expected_peaks_file_path = "/Users/dmitrav/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"

# scan number 43 is a good example for current testing mzXML file
main_features_scans_indexes = [43]  # indexes for independent, isotopic, fragmentation, non-expected features extraction

background_features_scans_indexes = []  # scans indexes used for instrument noise features extraction

peak_widths_levels_of_interest = [0.2, 0.5, 0.8]

allowed_ppm_error = 0.25

peak_region_factor = 3  # 3 times resolution is a region for extracting information

maximum_number_of_subsequent_peaks_to_consider = 50  # initial guess

mz_frame_size = 50  # for frames [0, 50], [50, 100], ...
number_of_frames = 11

number_of_top_noisy_peaks_to_consider = 5
frame_intensity_percentiles = [25, 50, 75]
