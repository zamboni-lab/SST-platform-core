import datetime
import time, os, json
from pyteomics import mzxml
from src.msfe.constants import tunings_matrix_file_path
from src.msfe.constants import instrument_noise_features_scans_indexes
from src.msfe import msfe


def process_all_tunes_and_files_at_once():
    """ This method runs msfe on all the files given full tunings matrix for them.
        Used usually to process all QC files from the beginning, to create updated f_matrix. """

    # get all instrument settings to provide correct acquisition dates later
    with open(tunings_matrix_file_path) as tunes:
        tunings_data = json.load(tunes)

    acquisition_dates = []
    for run in tunings_data['ms_runs']:
        acquisition_dates.append(run['meta']['values'][1].split(".")[0].replace(":", ""))

    path_to_files = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/nas2/'

    for root, dirs, files in os.walk(path_to_files):

        dirs = sorted(dirs)  # to make it chronological

        # for filename in files:
        for dir in dirs:

            # if filename != '.DS_Store':
            if dir != '.DS_Store':  # and dir > "2019-11-12T102201":
                start_time = time.time()
                print(dir, 'file is being processed')

                spectra = list(mzxml.read(path_to_files + dir + '/raw.mzXML'))

                acq_date = acquisition_dates[dirs.index(dir)]

                ms_run_ids = {'acquisition_date': acq_date, 'original_filename': dir}

                msfe.extract_features_from_ms_run(spectra, ms_run_ids, in_test_mode=True)

                # print(files.index(filename)+1, '/', len(files), 'is processed within', time.time() - start_time, 's\n')
                print(dirs.index(dir) + 1, '/', len(dirs), 'is processed within', time.time() - start_time, 's\n')

    print('All done. Well done!')


def check_instrument_noise_outliers():
    """ This method checks that there's no contradiction between TIC levels and instrument noise sum
        for two most noisy runs. """

    bigger_noise_file_path = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/nas2/2019-05-29T164545/raw.mzXML'
    smaller_noise_file_path = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/nas2/2019-07-12T101256/raw.mzXML'

    bigger_noise_spectra = list(mzxml.read(bigger_noise_file_path))
    smaller_noise_spectra = list(mzxml.read(smaller_noise_file_path))

    from matplotlib import pyplot

    pyplot.plot(bigger_noise_spectra[174]['m/z array'], bigger_noise_spectra[174]['intensity array'])
    pyplot.plot(smaller_noise_spectra[174]['m/z array'], smaller_noise_spectra[174]['intensity array'])
    pyplot.show()

    return True


def run_msfe_in_test_mode():
    """ This method runs msfe on a specified mzXML file. """

    start_time = time.time()

    # # chemical mix by Michelle
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix_v1/20190405_QCmeth_Mix30_013.mzXML'

    # scan 19 should have almost all the expected peaks saturated
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix_v1_saturation/20190523_RefMat_007.mzXML'

    # # scan 61 should have some expected peaks saturated
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix_v1_saturation/20190523_RefMat_042.mzXML'

    # # Duncan's last qc
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix_v1_debug/duncan_3_points_fit_bug.mzXML'

    # # file from test2 causing bug
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix_v1_debug/20190523_RefMat_131.mzXML'

    # file from test2 causing warning
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix_v1_debug/20190523_RefMat_134.mzXML'

    # file from test2 causing another bug
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/chem_mix_v1_debug/20190523_RefMat_012.mzXML'

    # file from nas2 causing index out of range bug (only 86 scans in file)
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/nas2/2019-06-10T113612/raw.mzXML'

    # file from nas2 causing error fitting peaks
    # chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/nas2/2019-09-05T212603/raw.mzXML'

    # file from nas2 causing index out of range bug (less number of scans than supposed to be)
    chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/nas2/2019-11-08T124046/raw.mzXML'

    chemical_standard = '/Users/andreidm/ETH/projects/ms_feature_extractor/data/nas2/2019-11-08T131837/raw.mzXML'

    spectra = list(mzxml.read(chemical_standard))
    print(time.time() - start_time, " seconds elapsed for reading", sep="")

    ms_run_ids = {'date': datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S"), 'original_filename': '2019-11-08T131837'}

    msfe.extract_features_from_ms_run(spectra, ms_run_ids, in_test_mode=True)


if __name__ == '__main__':

    # run_msfe_in_test_mode()
    process_all_tunes_and_files_at_once()





