
version = '0.1.6'

feature_matrix_file_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/feature_matrix.json"

parser_comment_symbol = '#'
parser_description_symbols = '()'

minimal_normal_peak_intensity = 100
minimal_background_peak_intensity = 1

normal_scan_expected_peaks_file_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"
chemical_noise_scan_expected_peaks_file_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"
# normal_scan_expected_peaks_file_path = "/Users/dmitrav/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"
# chemical_noise_scan_expected_peaks_file_path = "/Users/dmitrav/ETH/projects/ms_feature_extractor/data/expected_peaks_example.txt"

number_of_normal_scans = 3  # for Michelle's method main scans are defined by TIC maxima
# chemical_noise_features_scans_indexes = [17, 18]  # for Michelle's method
# instrument_noise_features_scans_indexes = [174, 175]  # for Michelle's method

# scan number 43 is a good example for current testing mzXML file
# main_features_scans_indexes = [43, 42]  # indexes for independent, isotopic, fragmentation, non-expected features extraction
chemical_noise_features_scans_indexes = [42, 41]  # indexes for chemical noise features extraction
instrument_noise_features_scans_indexes = [41, 40]  # indexes for instrument noise features extraction

peak_widths_levels_of_interest = [0.2, 0.5, 0.8]

allowed_ppm_error = 25

peak_region_factor = 3  # 3 times resolution is a region for extracting information

maximum_number_of_subsequent_peaks_to_consider = 10  # initial guess

normal_scan_mz_frame_size = 50  # for frames [50, 100], [100, 150] ... [1000, 1050]
normal_scan_number_of_frames = 20

chemical_noise_scan_mz_frame_size = 100  # for frames [50, 150], [150, 250] ... [950, 1050]
chemical_noise_scan_number_of_frames = 10

instrument_noise_mz_frame_size = 200  # for frames [50, 250], [250, 450] ... [850, 1050]
instrument_noise_scan_number_of_frames = 5

number_of_top_noisy_peaks_to_consider = 10
frame_intensity_percentiles = [25, 50, 75]
