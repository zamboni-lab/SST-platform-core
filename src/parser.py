
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets
from src.constants import feature_matrix_file_path, ms_settings_matrix_file_path
from src.constants import chemical_mix_id, msfe_version
from src.constants import number_of_normal_scans
from src.constants import chemical_noise_features_scans_indexes as chem_scans
from src.constants import instrument_noise_features_scans_indexes as bg_scans
from pyopenms import EmpiricalFormula, CoarseIsotopePatternGenerator
import json, os


def parse_expected_ions(file_path, scan_type):
    """ Since >v.0.1.8 JSON file is used for input. The information about expected ions is extracted from there.
        The resulting data structure is almost the same with the old version (to integrate to old code). """

    assert scan_type == "normal" or scan_type == "chemical_noise"

    ions_ids = []
    expected_ions_mzs = []
    expected_isotopic_ratios = []
    fragmentation_lists = []
    isotopic_lists = []

    with open(file_path) as input_file:
        all_expected_ions = json.load(input_file)

    # correct ions names: all uppercase + no - in the end
    for i in range(len(all_expected_ions[scan_type])):
        for j in range(1, len(all_expected_ions[scan_type][i])):
            if all_expected_ions[scan_type][i][j][-1] == '-':
                # if there is - in the end of the ion name, remove it
                all_expected_ions[scan_type][i][j] = all_expected_ions[scan_type][i][j][:-1].upper()
            else:
                # if not, just make sure it's uppercase
                all_expected_ions[scan_type][i][j] = all_expected_ions[scan_type][i][j].upper()

    # iterate over ions to parse information
    for ion in all_expected_ions[scan_type]:
        # get the list of mzs of isotopes of the main guy + isotopic intensity ratios
        # (abundance of isotope in relation to the main guy)
        ion_isotopes = EmpiricalFormula(ion[1]).getIsotopeDistribution(CoarseIsotopePatternGenerator(3)).getContainer()
        isotopes_mzs = [iso.getMZ() for iso in ion_isotopes]
        isotopes_intensity_ratios = [iso.getIntensity() for iso in ion_isotopes]

        # add ids of the ion isotopes
        ids = [ion[0] + "_i" + str(i+1) for i in range(len(ion_isotopes))]

        # if there is any expected fragment
        if len(ion) > 1:
            # get the list of mzs of ions fragments including mz of the main guy
            fragments_list = [EmpiricalFormula(fragment).getMonoWeight() for fragment in ion[1:]]
            # add ids of the ion fragments
            ids.extend([ion[0] + "_f" + str(i+1) for i in range(len(fragments_list[1:]))])

        else:
            fragments_list = []

        # append / extend
        # the following two lists are of the same size == number of all mz values incl. main guys, isotopes, fragments
        ions_ids.extend(ids)
        expected_ions_mzs.extend([*isotopes_mzs, *fragments_list[1:]])

        # the following three lists are of the same size == number of main guys
        expected_isotopic_ratios.append(isotopes_intensity_ratios)
        fragmentation_lists.append(fragments_list)
        isotopic_lists.append(isotopes_mzs)

    # compose and return info
    expected_ions_info = {
        'ions_ids': ions_ids,
        'expected_mzs': expected_ions_mzs,
        'expected_isotopic_ratios': expected_isotopic_ratios,  # instead of "theoretical" intensities
        'fragments_mzs': fragmentation_lists,
        'isotopes_mzs': isotopic_lists
    }

    return expected_ions_info


def parse_expected_ions_old_version(file_path):
    """ Deprecated since v.0.1.8.
        This method parses information about expected ions. """

    with open(file_path) as file:
        all_of_it = file.read()

    pieces = all_of_it.split(sharp)

    expected_ions_mzs = []
    expected_ions_intensities = []

    fragmentation_lists = []
    isotopic_lists = []

    for piece in pieces:

        if piece.split(brackets[0])[0].lower().find('ions') >= 0:

            ions_info = piece.split(brackets[1])[1].split('\n')

            for ion in ions_info:
                if ion != "":
                    expected_ions_mzs.append(float(ion.split(",")[0]))
                    expected_ions_intensities.append(float(ion.split(",")[1]))

        elif piece.split(brackets[0])[0].lower().find('fragments') >= 0:

            fragmentation_info = piece.split(brackets[1])[1].split('\n')

            for fragments_list_info in fragmentation_info:
                if fragments_list_info != "":
                    fragments_list = [float(value) for value in fragments_list_info.split(',')]
                    fragmentation_lists.append(fragments_list)

        elif piece.split(brackets[0])[0].lower().find('isotopes') >= 0:

            isotopes_info = piece.split(brackets[1])[1].split('\n')

            for isotope_list_info in isotopes_info:
                if isotope_list_info != "":
                    isotopes_list = [float(value) for value in isotope_list_info.split(',')]
                    isotopic_lists.append(isotopes_list)

    expected_ions_info = {
        'expected_mzs': expected_ions_mzs,
        'expected_intensities': expected_ions_intensities,
        'fragments_mzs': fragmentation_lists,
        'isotopes_mzs': isotopic_lists
    }

    return expected_ions_info


def parse_instrument_settings_from_multiple_ms_runs(list_of_paths):
    """ This method reads instrument settings from previously generated files (paths provided),
        and adds information to the general ms_settings_matrix, which is stored as another json. """

    if not os.path.isfile(ms_settings_matrix_file_path):
        # if the file does not exist yet, create empty one
        s_matrix = {'ms_runs': []}
        with open(ms_settings_matrix_file_path, 'w') as new_file:
            json.dump(s_matrix, new_file)
    else:
        pass

    for path in list_of_paths:
        parse_ms_run_instrument_settings(path)


def parse_ms_run_instrument_settings(file_path):
    """ This method reads instrument settings from newly generated file (after it's uploaded on server)
        and adds information to the general ms_settings_matrix, which is stored as another json. """

    # read newly generated ms settings file
    with open(file_path) as file:
        new_data = json.load(file)

    # compose data structure to collect data
    meta = {'keys': [], 'values': []}
    actuals = {'keys': [], 'values': []}
    cals = {'keys': [], 'values': []}

    for key in new_data:

        if key == "Actuals":
            for actual in new_data[key]:
                actuals['keys'].append(actual.replace(" ","_"))
                actuals['values'].append(new_data[key][actual])

        elif key == "Cal":
            for mode in ['defaultPos', 'defaultNeg']:
                for type in ['traditional', 'polynomial']:
                    for i in range(len(new_data[key][mode][type])):
                        cals['keys'].append(mode + "_" + type + "_" + str(i))
                        cals['values'].append(new_data[key][mode][type][i])
        else:
            meta["keys"].append(key)
            meta["values"].append(new_data[key])

    # open old ms settings file
    with open(ms_settings_matrix_file_path) as general_file:
        s_matrix = json.load(general_file)

    # add new data to old file
    s_matrix['ms_runs'].append({
        'meta': meta,
        'actuals': actuals,
        'cals': cals
    })

    # dump updated file to the same place
    with open(ms_settings_matrix_file_path, 'w') as updated_file:
        json.dump(s_matrix, updated_file)


def update_feature_matrix(extracted_features, features_names, ms_run_ids):
    """ This method gets results of single MS run feature extraction
        and updates the general feature matrix. """

    if not os.path.isfile(feature_matrix_file_path):
        # if the file does not exist yet, create empty one
        f_matrix = {'ms_runs': []}
        with open(feature_matrix_file_path, 'w') as new_file:
            json.dump(f_matrix, new_file)
    else:
        pass

    with open(feature_matrix_file_path) as general_file:
        f_matrix = json.load(general_file)

    f_matrix['ms_runs'].append({
        'date': ms_run_ids['date'],
        'original_filename': ms_run_ids['original_filename'],
        'chemical_mix_id': chemical_mix_id,
        'msfe_version': msfe_version,
        'scans_processed': [number_of_normal_scans, len(chem_scans), len(bg_scans)],
        'features_values': extracted_features,
        'features_names': features_names
    })

    # dump updated file to the same place
    with open(feature_matrix_file_path, 'w') as updated_file:
        json.dump(f_matrix, updated_file)


if __name__ == "__main__":

    # parse_instrument_settings_from_multiple_ms_runs(["/Users/andreidm/ETH/projects/ms_feature_extractor/data/ms_settings.json"])

    print()

