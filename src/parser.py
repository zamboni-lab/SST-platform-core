
from src.constants import parser_comment_symbol as sharp
from src.constants import parser_description_symbols as brackets


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
