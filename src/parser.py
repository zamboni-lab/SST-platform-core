
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets
from src.constants import feature_matrix_file_path, ms_settings_matrix_file_path
import json, os


def parse_expected_ions(file_path):
    """ This method parses information about expected ions. """

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


def parse_instrument_settings_from_multiple_old_files(list_of_paths):
    """ This method reads instrument settings from previously generated files (paths provided),
        and adds information to the general ms_settings_matrix, which is stored as another json. """

    if not os.path.isfile(ms_settings_matrix_file_path):
        # if the file does not exist yet, create empty one
        ms_matrix = {'ms_runs': []}
        with open(ms_settings_matrix_file_path, 'w') as new_file:
            json.dump(ms_matrix, new_file)
    else:
        pass

    for path in list_of_paths:
        parse_instrument_settings(path)


def parse_instrument_settings(file_path):
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
        ms_matrix = json.load(general_file)

    # add new data to old file
    ms_matrix['ms_runs'].append({
        'meta': meta,
        'actuals': actuals,
        'cals': cals
    })

    # dump updated file to the same place
    with open(ms_settings_matrix_file_path, 'w') as updated_file:
        json.dump(ms_matrix, updated_file)


if __name__ == "__main__":

    parse_instrument_settings_from_multiple_old_files(["/Users/andreidm/ETH/projects/ms_feature_extractor/data/ms_settings.json"])

    print()

