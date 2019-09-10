
msfe_version = '0.3.3'
chemical_mix_id = '1'

""" Mass-spec features extractor constants """

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

# for Michelle's method
chemical_noise_features_scans_indexes = [17]
instrument_noise_features_scans_indexes = [174]
normal_scans_indexes_window = [25, 75]

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
no_signal_intensity_value = 0.
frame_intensity_percentiles = [25, 50, 75]


""" QC metrics generator constants """

qc_matrix_file_path = feature_matrix_file_path.replace('feature_matrix','qc_matrix')

qc_database_path = '/Users/andreidm/ETH/projects/qc_metrics/res/qc_matrix.db'

# features names
resolution_200_features_names = ['absolute_mass_accuracy_Caffeine_i1_mean', 'widths_Caffeine_i1_1_mean']  # 193.0725512871
resolution_700_features_names = ['absolute_mass_accuracy_Perfluorotetradecanoic_acid_i1_mean', 'widths_Perfluorotetradecanoic_acid_i1_1_mean']  # 712.94671694


accuracy_features_names = ['absolute_mass_accuracy_Caffeine_i1_mean', 'absolute_mass_accuracy_Caffeine_i2_mean', 'absolute_mass_accuracy_Caffeine_i3_mean', 'absolute_mass_accuracy_Caffeine_f1_mean',
                           'absolute_mass_accuracy_Fluconazole_i1_mean', 'absolute_mass_accuracy_Fluconazole_i2_mean', 'absolute_mass_accuracy_Fluconazole_i3_mean', 'absolute_mass_accuracy_Fluconazole_f1_mean',
                           'absolute_mass_accuracy_3-(Heptadecafluorooctyl)aniline_i1_mean', 'absolute_mass_accuracy_3-(Heptadecafluorooctyl)aniline_i2_mean', 'absolute_mass_accuracy_3-(Heptadecafluorooctyl)aniline_i3_mean',
                           'absolute_mass_accuracy_Albendazole_i1_mean', 'absolute_mass_accuracy_Albendazole_i2_mean', 'absolute_mass_accuracy_Albendazole_i3_mean', 'absolute_mass_accuracy_Albendazole_f1_mean', 'absolute_mass_accuracy_Albendazole_f2_mean',
                           'absolute_mass_accuracy_Triamcinolone_acetonide_i1_mean', 'absolute_mass_accuracy_Triamcinolone_acetonide_i2_mean', 'absolute_mass_accuracy_Triamcinolone_acetonide_i3_mean', 'absolute_mass_accuracy_Triamcinolone acetonide_f1_mean', 'absolute_mass_accuracy_Triamcinolone acetonide_f2_mean',
                           'absolute_mass_accuracy_Perfluorodecanoic_acid_i1_mean', 'absolute_mass_accuracy_Perfluorodecanoic_acid_i2_mean', 'absolute_mass_accuracy_Perfluorodecanoic_acid_i3_mean', 'absolute_mass_accuracy_Perfluorodecanoic acid_f1_mean',
                           'absolute_mass_accuracy_Tricosafluorododecanoic_acid_i1_mean', 'absolute_mass_accuracy_Tricosafluorododecanoic_acid_i2_mean', 'absolute_mass_accuracy_Tricosafluorododecanoic_acid_i3_mean', 'absolute_mass_accuracy_Tricosafluorododecanoic acid_f1_mean',
                           'absolute_mass_accuracy_Perfluorotetradecanoic_acid_i1_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic_acid_i2_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic_acid_i3_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic acid_f1_mean', 'absolute_mass_accuracy_Perfluorotetradecanoic acid_f2_mean',
                           'absolute_mass_accuracy_Pentadecafluoroheptyl_i1_mean', 'absolute_mass_accuracy_Pentadecafluoroheptyl_i2_mean', 'absolute_mass_accuracy_Pentadecafluoroheptyl_i3_mean']

dirt_features_names = ['intensity_sum_chem_50_150', 'intensity_sum_chem_150_250', 'intensity_sum_chem_250_350', 'intensity_sum_chem_350_450', 'intensity_sum_chem_450_550',
                       'intensity_sum_chem_550_650','intensity_sum_chem_650_750', 'intensity_sum_chem_750_850', 'intensity_sum_chem_850_950', 'intensity_sum_chem_950_1050']


instrument_noise_tic_features_names = ['intensity_sum_bg_50_250', 'intensity_sum_bg_250_450', 'intensity_sum_bg_450_650', 'intensity_sum_bg_650_850', 'intensity_sum_bg_850_1050']

instrument_noise_percentiles_features_names = ['percentiles_bg_50_250_1', 'percentiles_bg_250_450_1', 'percentiles_bg_450_650_1', 'percentiles_bg_650_850_1', 'percentiles_bg_850_1050_1']

isotopic_presence_features_names = ['isotopes_ratios_diffs_Caffeine_i1_0_mean', 'isotopes_ratios_diffs_Caffeine_i1_1_mean', 'isotopes_ratios_diffs_Caffeine_i1_2_mean',
                                    'isotopes_ratios_diffs_Fluconazole_i1_0_mean', 'isotopes_ratios_diffs_Fluconazole_i1_1_mean', 'isotopes_ratios_diffs_Fluconazole_i1_2_mean',
                                    'isotopes_ratios_diffs_3-(Heptadecafluorooctyl)aniline_i1_0_mean', 'isotopes_ratios_diffs_3-(Heptadecafluorooctyl)aniline_i1_1_mean', 'isotopes_ratios_diffs_3-(Heptadecafluorooctyl)aniline_i1_2_mean',
                                    'isotopes_ratios_diffs_Albendazole_i1_0_mean', 'isotopes_ratios_diffs_Albendazole_i1_1_mean', 'isotopes_ratios_diffs_Albendazole_i1_2_mean',
                                    'isotopes_ratios_diffs_Triamcinolone_acetonide_i1_0_mean', 'isotopes_ratios_diffs_Triamcinolone_acetonide_i1_1_mean', 'isotopes_ratios_diffs_Triamcinolone_acetonide_i1_2_mean',
                                    'isotopes_ratios_diffs_Perfluorodecanoic_acid_i1_0_mean', 'isotopes_ratios_diffs_Perfluorodecanoic_acid_i1_1_mean', 'isotopes_ratios_diffs_Perfluorodecanoic_acid_i1_2_mean',
                                    'isotopes_ratios_diffs_Tricosafluorododecanoic_acid_i1_0_mean', 'isotopes_ratios_diffs_Tricosafluorododecanoic_acid_i1_1_mean', 'isotopes_ratios_diffs_Tricosafluorododecanoic_acid_i1_2_mean',
                                    'isotopes_ratios_diffs_Perfluorotetradecanoic_acid_i1_0_mean', 'isotopes_ratios_diffs_Perfluorotetradecanoic_acid_i1_1_mean', 'isotopes_ratios_diffs_Perfluorotetradecanoic_acid_i1_2_mean',
                                    'isotopes_ratios_diffs_Pentadecafluoroheptyl_i1_0_mean', 'isotopes_ratios_diffs_Pentadecafluoroheptyl_i1_1_mean', 'isotopes_ratios_diffs_Pentadecafluoroheptyl_i1_2_mean']

transmission_features_names = ['intensity_Perfluorotetradecanoic_acid_i1_mean', 'intensity_Fluconazole_i1_mean']

fragmentation_features_names = ['fragments_ratios_Fluconazole_i1_0_mean',
                                'fragments_ratios_Perfluorotetradecanoic_acid_i1_0_mean']

baseline_150_250_features_names = ['percentiles_chem_150_250_0', 'percentiles_chem_150_250_1']
baseline_650_750_features_names = ['percentiles_chem_650_750_0', 'percentiles_chem_650_750_1']

signal_features_names = [feature_name.replace('absolute_mass_accuracy', 'intensity') for feature_name in accuracy_features_names]


s2b_features_names = ['intensity_3-(Heptadecafluorooctyl)aniline_i1_mean', 'percentiles_norm_500_550_0_mean']

s2n_features_names = ['intensity_3-(Heptadecafluorooctyl)aniline_i1_mean', 'percentiles_norm_500_550_0_mean', 'percentiles_norm_500_550_1_mean']

