
import json, os, numpy

from src.qcmg import db_connector
from src.msfe.constants import feature_matrix_file_path as f_matrix_path
from src.msfe.constants import qc_matrix_file_path as qc_matrix_path
from src.msfe.constants import resolution_200_features_names, resolution_700_features_names
from src.msfe.constants import accuracy_features_names, dirt_features_names, isotopic_presence_features_names
from src.msfe.constants import instrument_noise_tic_features_names as noise_features_names
from src.msfe.constants import transmission_features_names, fragmentation_features_names, signal_features_names
from src.msfe.constants import baseline_150_250_features_names, baseline_650_750_features_names
from src.msfe.constants import s2b_features_names, s2n_features_names
from src.msfe.constants import qc_database_path
from src.msfe import logger
from src.qcmg import qcm_validator


def add_resolution_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates resolutions metric for two ions (at around 200 m/z and 700 m/z).
        It's m/z divided by width of the peak at 50% height."""

    # get resolution at mz ~200
    mz200, width200 = resolution_200_features_names

    mz200_value = ms_run['features_values'][ms_run['features_names'].index(mz200)]
    width200_value = ms_run['features_values'][ms_run['features_names'].index(width200)]

    if mz200_value == -1. or width200_value == -1.:
        resolution200 = -1.
    else:
        resolution200 = int((193.0725512871 + mz200_value) / width200_value)

    # get resolution at mz ~700
    mz700, width700 = resolution_700_features_names

    mz700_value = ms_run['features_values'][ms_run['features_names'].index(mz700)]
    width700_value = ms_run['features_values'][ms_run['features_names'].index(width700)]

    if mz700_value == -1. or width700_value == -1.:
        resolution700 = -1.
    else:
        resolution700 = int((712.94671694 + mz700_value) / width700_value)

    qc_values.extend([resolution200, resolution700])
    qc_names.extend(['resolution_200', 'resolution_700'])

    if in_debug_mode:
        qcm_validator.print_qcm_names('resolution_200', ['ion_mz', mz200, width200, 'resolution_200'])
        qcm_validator.print_qcm_values('resolution_200', [193.0725512871, mz200_value, width200_value, resolution200])
        qcm_validator.print_qcm_names('resolution_700', ['ion_mz', mz700, width700, 'resolution_700'])
        qcm_validator.print_qcm_values('resolution_700', [712.94671694, mz700_value, width700_value, resolution700])


def add_accuracy_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates accuracy metrics for QC run.
        It's average of the absolute m/z diff values for all the expected ions. """

    values = []
    values_sum = 0.

    for feature in accuracy_features_names:
        value = ms_run['features_values'][ms_run['features_names'].index(feature)]

        values.append(value)
        if value != -1.:  # if this is not a missing value
            values_sum += value

    total_non_missing = sum(numpy.array(values) != -1.)
    average_accuracy = values_sum / total_non_missing

    qc_values.append(average_accuracy)
    qc_names.append('average_accuracy')

    if in_debug_mode:
        qcm_validator.print_qcm_names('average_accuracy', [*accuracy_features_names, 'total_non_missing', 'average_accuracy'])
        qcm_validator.print_qcm_values('average_accuracy', [*values, total_non_missing, average_accuracy])


def add_dirt_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates metric of the dirtiness.
        It sums up absolute intensities of the chemical noise scan. All the expected peaks there are excluded. """

    chem_noise_signal_sum = 0
    values = []  # for debugging only

    for feature in dirt_features_names:
        chem_noise_signal_sum += ms_run['features_values'][ms_run['features_names'].index(feature)]

        if in_debug_mode:
            values.append(ms_run['features_values'][ms_run['features_names'].index(feature)])

    qc_values.append(int(chem_noise_signal_sum))
    qc_names.append('chemical_dirt')

    if in_debug_mode:
        qcm_validator.print_qcm_names('chemical_dirt', [*dirt_features_names, 'chemical_dirt'])
        qcm_validator.print_qcm_values('chemical_dirt', [*values, int(chem_noise_signal_sum)])


def add_noise_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates metric of the instrument noise.
        It sums up absolute intensities of the instrument noise scan."""

    instrument_noise_signal_sum = 0
    values = []  # for debugging only

    for feature in noise_features_names:
        instrument_noise_signal_sum += ms_run['features_values'][ms_run['features_names'].index(feature)]

        if in_debug_mode:
            values.append(ms_run['features_values'][ms_run['features_names'].index(feature)])

    qc_values.append(int(instrument_noise_signal_sum))
    qc_names.append('instrument_noise')

    if in_debug_mode:
        qcm_validator.print_qcm_names('instrument_noise', [*noise_features_names, 'instrument_noise'])
        qcm_validator.print_qcm_values('instrument_noise', [*values, int(instrument_noise_signal_sum)])


def add_isotopic_abundance_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates metrics of the isotopic presence.
        It finds the average of isotopic intensities ratios diffs (absolute percent diffs for all the isotopes). """

    values = []
    values_sum = 0.

    for feature in isotopic_presence_features_names:
        value = ms_run['features_values'][ms_run['features_names'].index(feature)]

        values.append(value)
        if value != -1.:
            values_sum += abs(value)

    total_non_missing = sum(numpy.array(values) != -1.)
    ratios_diffs_mean = values_sum / total_non_missing

    qc_values.append(ratios_diffs_mean)
    qc_names.append('isotopic_presence')

    if in_debug_mode:
        qcm_validator.print_qcm_names('isotopic_presence', [*isotopic_presence_features_names, 'total_non_missing', 'isotopic_presence'])
        qcm_validator.print_qcm_values('isotopic_presence', [*values, total_non_missing, ratios_diffs_mean])


def add_transmission_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates the metric of transmission.
        It finds the ratio of the intensities of two ions: the light one (~ mz305) and the heavy one (~ mz712). """

    intensity712, intensity305 = transmission_features_names

    intensity712_value = ms_run['features_values'][ms_run['features_names'].index(intensity712)]
    intensity305_value = ms_run['features_values'][ms_run['features_names'].index(intensity305)]

    if intensity305_value != -1. and intensity712_value != -1.:
        transmission = intensity712_value / intensity305_value
    else:
        transmission = -1.

    qc_values.append(transmission)
    qc_names.append('transmission')

    if in_debug_mode:
        qcm_validator.print_qcm_names('transmission', [intensity712, intensity305, 'transmission'])
        qcm_validator.print_qcm_values('transmission', [intensity712_value, intensity305_value, transmission])


def add_fragmentation_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method saves the metrics of fragmentation (nothing is calculated, just directly passed).
        Fragmentation intensity ratios of two ions are taken: mz305(to mz191) & mz712 (to mz668). """

    ratio305, ratio712 = fragmentation_features_names

    fragmentation305 = ms_run['features_values'][ms_run['features_names'].index(ratio305)]
    fragmentation712 = ms_run['features_values'][ms_run['features_names'].index(ratio712)]

    qc_values.extend([fragmentation305, fragmentation712])
    qc_names.extend(['fragmentation_305', 'fragmentation_712'])

    if in_debug_mode:
        qcm_validator.print_qcm_names('fragmentation_305', [ratio305])
        qcm_validator.print_qcm_values('fragmentation_305', [fragmentation305])
        qcm_validator.print_qcm_names('fragmentation_712', [ratio712])
        qcm_validator.print_qcm_values('fragmentation_712', [fragmentation712])


def add_baseline_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method saves the metrics of baseline (nothing is calculated, just directly passed).
        25th and 50th percentiles are taken from two chemical noise scan frames: [150, 250], [650, 750]. """

    percentile_25_from_150, percentile_50_from_150 = baseline_150_250_features_names
    percentile_25_from_650, percentile_50_from_650 = baseline_650_750_features_names

    baseline_25_from_150 = int(ms_run['features_values'][ms_run['features_names'].index(percentile_25_from_150)])
    baseline_50_from_150 = int(ms_run['features_values'][ms_run['features_names'].index(percentile_50_from_150)])

    baseline_25_from_650 = int(ms_run['features_values'][ms_run['features_names'].index(percentile_25_from_650)])
    baseline_50_from_650 = int(ms_run['features_values'][ms_run['features_names'].index(percentile_50_from_650)])

    qc_values.extend([baseline_25_from_150, baseline_50_from_150, baseline_25_from_650, baseline_50_from_650])
    qc_names.extend(['baseline_25_150', 'baseline_50_150', 'baseline_25_650', 'baseline_50_650'])

    if in_debug_mode:
        qcm_validator.print_qcm_names('baseline_25_150', [percentile_25_from_150])
        qcm_validator.print_qcm_values('baseline_25_150', [baseline_25_from_150])
        qcm_validator.print_qcm_names('baseline_50_150', [percentile_50_from_150])
        qcm_validator.print_qcm_values('baseline_50_150', [baseline_50_from_150])
        qcm_validator.print_qcm_names('baseline_25_650', [percentile_25_from_650])
        qcm_validator.print_qcm_values('baseline_25_650', [baseline_25_from_650])
        qcm_validator.print_qcm_names('baseline_50_650', [percentile_50_from_650])
        qcm_validator.print_qcm_values('baseline_50_650', [baseline_50_from_650])


def add_signal_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates metric of the overall signal.
        It sums up absolute intensities of all the expected peaks."""

    values = []
    signal_sum = 0.

    for feature in signal_features_names:
        value = ms_run['features_values'][ms_run['features_names'].index(feature)]

        values.append(value)
        if value != -1.:
            signal_sum += value

    signal_sum = int(signal_sum)
    total_non_missing = sum(numpy.array(values) != -1.)

    qc_values.append(signal_sum)
    qc_names.append('signal')

    if in_debug_mode:
        qcm_validator.print_qcm_names('signal', [*signal_features_names, 'total_non_missing', 'signal'])
        qcm_validator.print_qcm_values('signal', [*values, total_non_missing, signal_sum])


def add_signal_to_background_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates metric of signal to background as follows:
        intensity at mz510 / 25th percentile in [500, 550]. """

    intensity510, percentile_25_from_500 = s2b_features_names

    intensity510_value = ms_run['features_values'][ms_run['features_names'].index(intensity510)]
    percentile_25_from_500_value = ms_run['features_values'][ms_run['features_names'].index(percentile_25_from_500)]

    if intensity510_value != -1.:
        s2b = intensity510_value / percentile_25_from_500_value
    else:
        s2b = -1.

    qc_values.append(s2b)
    qc_names.append('s2b')

    if in_debug_mode:
        qcm_validator.print_qcm_names('s2b', [intensity510, percentile_25_from_500, 's2b'])
        qcm_validator.print_qcm_values('s2b', [intensity510_value, percentile_25_from_500_value, s2b])


def add_signal_to_noise_metrics(qc_values, qc_names, ms_run, in_debug_mode=False):
    """ This method calculates metric of signal to background as follows:
        intensity at mz510 / ( 50th percentile in [500, 550] - 25th percentile in [500, 550]) . """

    intensity510, percentile_25_from_500, percentile_50_from_500 = s2n_features_names

    intensity510_value = ms_run['features_values'][ms_run['features_names'].index(intensity510)]
    percentile_50_from_500_value = ms_run['features_values'][ms_run['features_names'].index(percentile_50_from_500)]
    percentile_25_from_500_value = ms_run['features_values'][ms_run['features_names'].index(percentile_25_from_500)]

    if intensity510_value != -1.:
        s2n = intensity510_value / (percentile_50_from_500_value - percentile_25_from_500_value)
    else:
        s2n = -1.

    qc_values.append(s2n)
    qc_names.append('s2n')

    if in_debug_mode:
        qcm_validator.print_qcm_names('s2n', [intensity510, percentile_25_from_500, percentile_50_from_500, 's2n'])
        qcm_validator.print_qcm_values('s2n', [intensity510_value, percentile_25_from_500_value, percentile_50_from_500_value, s2n])


def calculate_and_save_qc_matrix(path=None, output='sqlite'):
    """ This method creates a new QC matrix out of the feature matrix and fills it with the QC characteristics
        calculated out of the feature matrix. """

    if path is None:
        path = f_matrix_path

    with open(path) as input:
        f_matrix = json.load(input)

    qc_matrix = {'qc_runs': []}

    print('Start processing...')

    for run in f_matrix['ms_runs']:

        qc_values = []
        qc_names = []

        add_resolution_metrics(qc_values, qc_names, run)
        add_accuracy_metrics(qc_values, qc_names, run)
        add_dirt_metrics(qc_values, qc_names, run)
        add_noise_metrics(qc_values, qc_names, run)
        add_isotopic_abundance_metrics(qc_values, qc_names, run)
        add_transmission_metrics(qc_values, qc_names, run)
        add_fragmentation_metrics(qc_values, qc_names, run)
        add_baseline_metrics(qc_values, qc_names, run)
        add_signal_metrics(qc_values, qc_names, run)
        add_signal_to_background_metrics(qc_values, qc_names, run)
        add_signal_to_noise_metrics(qc_values, qc_names, run)

        qc_matrix['qc_runs'].append({
            'date': run['date'],
            'original_filename': run['original_filename'],
            'chemical_mix_id': run['chemical_mix_id'],
            'msfe_version': run['msfe_version'],
            'scans_processed': run['scans_processed'],
            'qc_values': qc_values,
            'qc_names': qc_names
        })

        print('File', run['original_filename'], 'has been processed successfully.')

    # two options for dumping
    if output == 'json':
        with open(qc_matrix_path, 'w') as output:
            json.dump(qc_matrix, output)

    elif output == 'sqlite':
        db_connector.create_and_fill_qc_database(qc_matrix, debug=True)

    else:
        pass

    print('Processing is done! Results saved to', qc_matrix_path)


def calculate_and_save_qc_metrics_for_ms_run(ms_run):
    """ This method computes QC metrics for a new ms_run and calls method to insert them into a database. """

    qc_values = []
    qc_names = []

    in_debug_mode = False

    add_resolution_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_accuracy_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_dirt_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_noise_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_isotopic_abundance_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_transmission_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_fragmentation_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_baseline_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_signal_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_signal_to_background_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)
    add_signal_to_noise_metrics(qc_values, qc_names, ms_run, in_debug_mode=in_debug_mode)

    new_qc_run = {
        'processing_date': ms_run['processing_date'],
        'acquisition_date': ms_run['acquisition_date'],
        'chemical_mix_id': ms_run['chemical_mix_id'],
        'msfe_version': ms_run['msfe_version'],
        'scans_processed': ms_run['scans_processed'],
        'qc_values': qc_values,
        'qc_names': qc_names
    }

    logger.print_qc_info('QC characteristics has been computed successfully')

    if not os.path.isfile(qc_database_path):
        # if there's yet no database
        db_connector.create_and_fill_qc_database({'qc_runs': [new_qc_run]})
        logger.print_qc_info('New QC database has been created (SQLite)')
    else:
        # if the database already exists
        db_connector.insert_new_qc_run(new_qc_run)
        logger.print_qc_info('QC database has been updated')


if __name__ == '__main__':

    pass


