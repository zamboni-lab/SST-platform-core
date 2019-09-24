
from src.msfe.constants import parser_comment_symbol as sharp
from src.msfe.constants import parser_description_symbols as brackets
from src.msfe.constants import feature_matrix_file_path, tunings_matrix_file_path
from src.msfe.constants import chemical_mix_id, msfe_version
from src.msfe.constants import qc_database_path
from src.qcmg import metrics_generator
from src.msfe import logger
from pyopenms import EmpiricalFormula, CoarseIsotopePatternGenerator
import json, os, datetime


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

    # correct ions names: all uppercase with no '-' in the end
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
        ids = [ion[0].replace(" ", "_") + "_i" + str(i+1) for i in range(len(ion_isotopes))]

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

    if not os.path.isfile(tunings_matrix_file_path):
        # if the file does not exist yet, create empty one
        s_matrix = {'ms_runs': []}
        with open(tunings_matrix_file_path, 'w') as new_file:
            json.dump(s_matrix, new_file)
    else:
        pass

    for path in list_of_paths:
        parse_ms_run_instrument_settings(path)


def parse_ms_run_instrument_settings(file_path, empty=False):
    """ This method reads instrument settings from newly generated file
        and adds information to the general tunings_matrix, which is stored as another json. """

    # compose data structure to collect data
    meta = {'keys': [], 'values': []}
    actuals = {'keys': [], 'values': []}
    cals = {'keys': [], 'values': []}

    if not empty:

        # read newly generated ms settings file
        with open(file_path) as file:
            new_data = json.load(file)

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

        logger.print_tune_info(datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S") + ": new tunes collected")
    else:
        logger.print_tune_info(datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S") + ": tunes are missing")

    if not os.path.isfile(tunings_matrix_file_path):
        # if the file does not exist yet, create empty one
        s_matrix = {'ms_runs': [{'meta': meta, 'actuals': actuals, 'cals': cals}]}

        with open(tunings_matrix_file_path, 'w') as new_file:
            json.dump(s_matrix, new_file)
    else:
        # open old ms settings file
        with open(tunings_matrix_file_path) as general_file:
            s_matrix = json.load(general_file)

            # add new data to old file
            s_matrix['ms_runs'].append({
                'meta': meta,
                'actuals': actuals,
                'cals': cals
            })

        # dump updated file to the same place
        with open(tunings_matrix_file_path, 'w') as updated_file:
            json.dump(s_matrix, updated_file)

    logger.print_tune_info("MS settings matrix updated\n")


def update_feature_matrix(extracted_features, features_names, ms_run_ids, scans_processed):
    """ This method gets results of single MS run feature extraction
        and updates the general feature matrix. """

    new_ms_run = {
        'processing_date': ms_run_ids['original_filename'],
        'acquisition_date': ms_run_ids['acquisition_date'],
        'chemical_mix_id': chemical_mix_id,
        'msfe_version': msfe_version,
        'scans_processed': scans_processed,
        'features_values': extracted_features,
        'features_names': features_names
    }

    # entry point for qcm to process new_ms_run and insert into QC database
    metrics_generator.calculate_and_save_qc_metrics_for_ms_run(new_ms_run)

    if not os.path.isfile(feature_matrix_file_path):
        # if the file does not exist yet, create empty one
        f_matrix = {'ms_runs': []}
        with open(feature_matrix_file_path, 'w') as new_file:
            json.dump(f_matrix, new_file)
    else:
        pass

    # TODO: substitute JSON with SQLite as well, to save time reading (eventually) large files
    # read existing file
    with open(feature_matrix_file_path) as general_file:
        f_matrix = json.load(general_file)

    # add new processed ms run
    f_matrix['ms_runs'].append(new_ms_run)

    # dump updated file to the same place
    with open(feature_matrix_file_path, 'w') as updated_file:
        json.dump(f_matrix, updated_file)


if __name__ == "__main__":

    path = '/Volumes/biol_imsb_sauer_2/fiaqc-data/'
    filename = '/all.json'

    for dir in sorted(os.listdir(path)):

        print(dir, "is being processed")

        # errorful and ancient files excluded
        if dir not in ['.DS_Store', '2019-05-17T115518', '2019-05-17T115432', '2019-05-17T115246', '2019-09-10T124004',
                       '2019-05-17T115021', '2019-05-17T114715', '2019-05-16T165425', '2019-04-12T152701',
                       '2019-04-12T152608', '2019-04-12T151912', '2019-04-11T200719', '2019-04-11T200714',
                       '2019-06-10T113612']:

            full_path = path + dir + filename
            if not os.path.isfile(full_path):
                # create empty data structure
                parse_ms_run_instrument_settings(full_path, empty=True)
            else:
                parse_ms_run_instrument_settings(full_path)

    print("All settings are pulled out.")

