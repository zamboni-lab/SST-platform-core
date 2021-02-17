
import numpy
from src.analysis import features_analysis
from src.constants import qc_mix_ions, chem_bg_ions, expected_peaks_file_path
from src.msfe import parser

# hardcoded ION FEATURES TYPES of the following structure {'string key': 'ion type'}
# categorical features dropped
ion_types_dict = {
    'intensity': 'ion_count',

    'widths': 'peak_width',

    'absolute_mass_accuracy': 'mass_accuracy',
    'ppm': 'mass_accuracy',

    'subsequent_peaks_number': 'peak_vicinity',
    'subsequent_peaks_ratios': 'peak_vicinity',

    'left_tail_auc': 'peak_shape',
    'right_tail_auc': 'peak_shape',
    'symmetry': 'peak_shape',
    'goodness_of_fit': 'peak_shape',

    'isotopes_ratios': 'ion_isotopes',
    'isotopes_ratios_diffs': 'ion_isotopes',
    'isotopes_mass_diffs': 'ion_isotopes',

    'fragments_ratios': 'ion_fragments',
    'fragments_ratios_diffs': 'ion_fragments',
    'fragments_mass_diffs': 'ion_fragments'
}

# hardcoded FRAME FEATURES TYPES of the following structure {'string key': 'frame type'}
# categorical features dropped
frame_types_dict = {

    'number_of_peaks_norm': 'qc_mix_bg',
    'intensity_sum_norm': 'qc_mix_bg',
    'percentiles_norm': 'qc_mix_bg',
    'top_peaks_intensities_norm': 'qc_mix_bg',
    'top_percentiles_norm': 'qc_mix_bg',

    'number_of_peaks_chem': 'chemical_bg',
    'intensity_sum_chem': 'chemical_bg',
    'percentiles_chem': 'chemical_bg',
    'top_peaks_intensities_chem': 'chemical_bg',
    'top_percentiles_chem': 'chemical_bg',

    'number_of_peaks_bg': 'detector_noise',
    'intensity_sum_bg': 'detector_noise',
    'percentiles_bg': 'detector_noise',
    'top_peaks_intensities_bg': 'detector_noise',
    'top_percentiles_bg': 'detector_noise'
}

# hardcoded ION MASS TYPES of the following structure {'string key': 'ion type'}
# categorical features dropped
ion_masses_dict = {
    'HOT_i1': '050_150',
    'HOT_i2': '050_150',
    'HOT_i3': '050_150',

    'Caffeine_i1': '150_250',
    'Caffeine_i2': '150_250',
    'Caffeine_i3': '150_250',
    'Caffeine_f1': '150_250',
    'Fluconazole_f1': '150_250',
    'Albendazole_f1': '150_250',
    'Albendazole_f2': '150_250',

    'Fluconazole_i1': '250_350',
    'Fluconazole_i2': '250_350',
    'Fluconazole_i3': '250_350',
    'Albendazole_i1': '250_350',
    'Albendazole_i2': '250_350',
    'Albendazole_i3': '250_350',
    'Triamcinolone_acetonide_f2': '250_350',

    'Triamcinolone_acetonide_i1': '350_450',
    'Triamcinolone_acetonide_i2': '350_450',
    'Triamcinolone_acetonide_i3': '350_450',
    'Pentadecafluoroheptyl_i1': '350_450',
    'Pentadecafluoroheptyl_i2': '350_450',
    'Pentadecafluoroheptyl_i3': '350_450',

    '3Heptadecafluorooctylaniline_i1': '450_550',
    '3Heptadecafluorooctylaniline_i2': '450_550',
    '3Heptadecafluorooctylaniline_i3': '450_550',
    'Triamcinolone_acetonide_f1': '450_550',
    'Perfluorodecanoic_acid_i1': '450_550',
    'Perfluorodecanoic_acid_i2': '450_550',
    'Perfluorodecanoic_acid_i3': '450_550',
    'Perfluorodecanoic_acid_f1': '450_550',

    'Tricosafluorododecanoic_acid_i1': '550_650',
    'Tricosafluorododecanoic_acid_i2': '550_650',
    'Tricosafluorododecanoic_acid_i3': '550_650',
    'Tricosafluorododecanoic_acid_f1': '550_650',
    'Perfluorotetradecanoic_acid_f2': '550_650',

    'Perfluorotetradecanoic_acid_i1': '650_750',
    'Perfluorotetradecanoic_acid_i2': '650_750',
    'Perfluorotetradecanoic_acid_i3': '650_750',
    'Perfluorotetradecanoic_acid_f1': '650_750',

    'HEX_i1': '850_950',
    'HEX_i2': '850_950',
    'HEX_i3': '850_950'
}

# hardcoded FRAME MASS TYPES of the following structure {'string key': 'frame type'}
# categorical features dropped
frame_masses_dict = {
    '50_100': '050_150',
    '100_150': '050_150',
    '50_150': '050_150',
    '50_250': ('050_150', '150_250'),

    '150_200': '150_250',
    '200_250': '150_250',
    '150_250': '150_250',

    '250_300': '250_350',
    '300_350': '250_350',
    '250_350': '250_350',
    '250_450': ('250_350', '350_450'),

    '350_400': '350_450',
    '400_450': '350_450',
    '350_450': '350_450',

    '450_500': '450_550',
    '500_550': '450_550',
    '450_550': '450_550',
    '450_650': ('450_550', '550_650'),

    '550_600': '550_650',
    '600_650': '550_650',
    '550_650': '550_650',

    '650_700': '650_750',
    '700_750': '650_750',
    '650_750': '650_750',
    '650_850': ('650_750', '750_850'),

    '750_800': '750_850',
    '800_850': '750_850',
    '750_850': '750_850',

    '850_900': '850_950',
    '900_950': '850_950',
    '850_950': '850_950',
    '850_1050': ('850_950', '950_1050'),

    '950_1000': '950_1050',
    '1000_1050': '950_1050',
    '950_1050': '950_1050'
}


def get_feature_types(continuous_features_names):

    feature_types = []
    for feature in continuous_features_names:

        ion_related = False
        for ion in [*qc_mix_ions, *chem_bg_ions]:
            if ion in feature:
                type = ion_types_dict[feature.split(ion)[0][:-1]]  # + '_{}'.format(ion)
                feature_types.append(type)
                ion_related = True
                break

        if not ion_related:
            for feature_name in frame_types_dict.keys():
                if feature_name in feature:
                    type = frame_types_dict[feature_name]
                    feature_types.append(type)
                    break

    return feature_types


def get_mass_types(continuous_features_names):

    mass_types = []
    for feature in continuous_features_names:

        ion_related = False
        for ion in ion_masses_dict.keys():
            if ion in feature:
                type = ion_masses_dict[ion]
                mass_types.append(type)
                ion_related = True
                break

        if not ion_related:
            for feature_name in frame_masses_dict.keys():
                if feature_name in feature:
                    type = frame_masses_dict[feature_name]
                    mass_types.append(type)
                    break

    return mass_types


if __name__ == "__main__":

    _, features, colnames = features_analysis.get_features_data()
    _, continuous_features_names, _, categorical_features_names = features_analysis.split_features_to_cont_and_cat(features, numpy.array(colnames[4:]))

    all_types = get_feature_types(continuous_features_names)

    mass_types = get_mass_types(continuous_features_names)


    print()


