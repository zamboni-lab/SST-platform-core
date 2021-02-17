
import numpy
from src.analysis import features_analysis
from src.constants import qc_mix_ions, chem_bg_ions

# hardcoded ion features type of the following structure {'string key': 'ion type'}
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

# hardcoded frame features type of the following structure {'string key': 'frame type'}
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


if __name__ == "__main__":

    _, features, colnames = features_analysis.get_features_data()
    _, continuous_features_names, _, categorical_features_names = features_analysis.split_features_to_cont_and_cat(features, numpy.array(colnames[4:]))

    all_types = get_feature_types(continuous_features_names)

    # TODO: implement logic to define:
    #  - features' mass types
