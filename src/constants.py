
msfe_version = '0.2.13'
chemical_mix_id = '1'

# qc_log_location = "/mnt/nas2/fiaqc-out/qc_logs.txt"
qc_log_location = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/qc_logs.txt"
# tune_log_location = "/mnt/nas2/fiaqc-out/tune_logs.txt"
tune_log_location = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/tune_logs.txt"

# feature_matrix_file_path = "/mnt/nas2/fiaqc-out/f_matrix.json"
feature_matrix_file_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/feature_matrix.json"
# ms_settings_matrix_file_path = "/mnt/nas2/fiaqc-out/s_matrix.json"
ms_settings_matrix_file_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/ms_settings_matrix.json"

# expected_peaks_file_path = "/home/nzadmin/msqc/msfe/data/expected_peaks_v" + chemical_mix_id + ".json"
expected_peaks_file_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/data/expected_peaks_v" + chemical_mix_id + ".json"

parser_comment_symbol = '#'
parser_description_symbols = '()'

minimal_normal_peak_intensity = 100
minimal_background_peak_intensity = 1

number_of_normal_scans = 3  # for Michelle's method main scans are defined by TIC maxima

chemical_noise_features_scans_indexes = [17]  # for Michelle's method
instrument_noise_features_scans_indexes = [174]  # for Michelle's method

peak_widths_levels_of_interest = [0.2, 0.5, 0.8]

saturation_intensity = 1000000

allowed_ppm_error = 25

peak_region_factor = 3  # 3 times resolution is a region for extracting information

maximum_number_of_subsequent_peaks_to_consider = 5  # initial guess

normal_scan_mz_frame_size = 50  # for frames [50, 100], [100, 150] ... [1000, 1050]
normal_scan_number_of_frames = 20

chemical_noise_scan_mz_frame_size = 100  # for frames [50, 150], [150, 250] ... [950, 1050]
chemical_noise_scan_number_of_frames = 10

instrument_noise_mz_frame_size = 200  # for frames [50, 250], [250, 450] ... [850, 1050]
instrument_noise_scan_number_of_frames = 5

number_of_top_noisy_peaks_to_consider = 10
frame_intensity_percentiles = [25, 50, 75]
