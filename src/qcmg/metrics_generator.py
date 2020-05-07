
import json, os, numpy, seaborn, pandas
from matplotlib import pyplot
from sklearn.ensemble import IsolationForest
from src.constants import resolution_200_features_names, resolution_700_features_names
from src.constants import accuracy_features_names, dirt_features_names, isotopic_presence_features_names
from src.constants import instrument_noise_tic_features_names as noise_features_names
from src.constants import transmission_features_names, fragmentation_features_names, signal_features_names
from src.constants import baseline_150_250_features_names, baseline_650_750_features_names
from src.constants import s2b_features_names, s2n_features_names
from src.constants import qc_metrics_database_path, qc_features_database_path, qc_tunes_database_path
from src.constants import anomaly_detection_method
from src.constants import min_number_of_metrics_to_assess_quality as min_number_of_runs
from src.constants import get_buffer_id, all_metrics
from src.analysis import anomaly_detector
from src.msfe import logger, db_connector
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
            values_sum += abs(value)  # abs for absolute mass accuracy values

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
    """ Outdated in v.0.3.24. This method creates a new QC matrix out of the feature matrix and fills it with the QC characteristics
        calculated out of the feature matrix. """

    qc_matrix_path = ""  # added to avoid declaration error here

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
        db_connector.create_and_fill_qc_databases(qc_matrix, in_debug_mode=True)

    else:
        pass

    print('Processing is done! Results saved to', qc_matrix_path)


def calculate_metrics_and_update_qc_databases(ms_run, in_debug_mode=False):
    """ This method computes QC metrics for a new ms_run and calls method to insert them into a database. """

    metrics_values = []
    metrics_names = []

    add_resolution_metrics(metrics_values, metrics_names, ms_run)
    add_accuracy_metrics(metrics_values, metrics_names, ms_run)
    add_dirt_metrics(metrics_values, metrics_names, ms_run)
    add_noise_metrics(metrics_values, metrics_names, ms_run)
    add_isotopic_abundance_metrics(metrics_values, metrics_names, ms_run)
    add_transmission_metrics(metrics_values, metrics_names, ms_run)
    add_fragmentation_metrics(metrics_values, metrics_names, ms_run)
    add_baseline_metrics(metrics_values, metrics_names, ms_run)
    add_signal_metrics(metrics_values, metrics_names, ms_run)
    add_signal_to_background_metrics(metrics_values, metrics_names, ms_run)
    add_signal_to_noise_metrics(metrics_values, metrics_names, ms_run)

    # assign quality for each metric based on previous records
    metrics_qualities = assign_metrics_qualities(metrics_values, metrics_names, ms_run, in_debug_mode=in_debug_mode)
    metrics_qualities = [int(x) for x in metrics_qualities]
    quality = int(sum(metrics_qualities) >= (len(metrics_qualities)+1) // 2)  # '1' if at least half of metrics are '1'

    new_qc_run = {

        'md5': ms_run['md5'],
        'original_filename': ms_run['original_filename'],
        'instrument': ms_run['instrument'],
        'user': ms_run['user'],

        'processing_date': ms_run['processing_date'],
        'acquisition_date': ms_run['acquisition_date'],
        'chemical_mix_id': ms_run['chemical_mix_id'],
        'buffer_id': ms_run['buffer_id'],
        'msfe_version': ms_run['msfe_version'],
        'scans_processed': ms_run['scans_processed'],

        'features_values': ms_run['features_values'],
        'features_names': ms_run['features_names'],
        'metrics_values': metrics_values,
        'metrics_names': metrics_names,
        'metrics_qualities': metrics_qualities,
        'tunes_values': ms_run['tunes_values'],
        'tunes_names': ms_run['tunes_names'],

        'user_comment': "",
        'quality': quality
    }

    logger.print_qc_info('QC characteristics have been computed successfully')

    if not (os.path.isfile(qc_metrics_database_path) or os.path.isfile(qc_features_database_path)
            or os.path.isfile(qc_tunes_database_path)):

        # if there's yet no databases
        db_connector.create_and_fill_qc_databases(new_qc_run, in_debug_mode=in_debug_mode)
        logger.print_qc_info('New QC databases have been created (SQLite)\n')
    else:
        # if the databases already exist
        db_connector.insert_new_qc_run(new_qc_run, in_debug_mode=in_debug_mode)
        logger.print_qc_info('QC databases have been updated\n')


def create_and_fill_quality_table_using_percentiles(data):
    """ This method applies initial method to assign qualities to metrics.
        All metrics are grouped in three, based on hand-designed ranges of "good" values.
        Ranges are defined using percentiles. (No predictions are made). """

    quality_table = data[:]

    for metric in ["resolution_200", "resolution_700", "signal", "s2b", "s2n"]:

        all_values = data.loc[:, metric]
        q1, q2 = numpy.percentile(all_values, [5, 95])

        filtered = all_values[(all_values > q1) & (all_values < q2)]  # remove outliers
        upper_boundary = numpy.percentile(filtered, 25)  # set lowest "good" value

        # define quality for each metric individually
        quality_table.loc[:, metric] = all_values > upper_boundary

    for metric in ["average_accuracy", "chemical_dirt", "instrument_noise", "baseline_25_150", "baseline_50_150", "baseline_25_650", "baseline_50_650"]:

        all_values = data.loc[:, metric]
        q1, q2 = numpy.percentile(all_values, [5, 95])

        filtered = all_values[(all_values > q1) & (all_values < q2)]  # remove outliers
        upper_boundary = numpy.percentile(filtered, 75)  # set highest "good" value

        # define quality for each metric individually
        quality_table.loc[:, metric] = all_values < upper_boundary

    for metric in ["isotopic_presence", "transmission", "fragmentation_305", "fragmentation_712"]:

        all_values = data.loc[:, metric]
        q1, q2 = numpy.percentile(all_values, [5, 95])

        filtered = all_values[(all_values > q1) & (all_values < q2)]  # remove outliers
        lower_boundary, upper_boundary = numpy.percentile(filtered, [5, 95])  # set interval of "good" values

        # define quality for each metric individually
        quality_table.loc[:, metric] = (all_values > lower_boundary) & (all_values < upper_boundary)

    # summarise individual qualities
    quality_table.insert(0, "quality", quality_table.sum(axis=1) > 7)

    return quality_table


def create_and_fill_quality_table_using_iforest(data):
    """ Introduced in v.0.3.79: a method to assign qualities to metrics using Isolation Forest. """

    quality_table = data[:]

    for metric in all_metrics:

        all_metric_values = numpy.array(data.loc[:, metric]).reshape(-1, 1)

        # effectively, allows ~15% of outliers
        iforest = IsolationForest()
        # train and predict on the same values, since it's the first time
        predicted_outliers = iforest.fit_predict(all_metric_values)
        # correct depending on the metrics, get array of {0,1}
        corrected_outliers = anomaly_detector.correct_outlier_prediction_for_metric(metric, predicted_outliers, all_metric_values, all_metric_values)

        # define quality for each metric individually
        quality_table.loc[:, metric] = corrected_outliers

    # summarise individual qualities
    quality_table.insert(0, "quality", quality_table.sum(axis=1) > 7)

    return quality_table


def recompute_quality_table_for_all_runs(last_run_metrics, metrics_data):
    """ This method calculates quality table for a QC metrics database:
        1) it stacks together last (new) run metrics and previous runs metrics,
        2) calculates quality table, according to method,
        3) updates existing databases and returns quality table. """

    # add last run metrics to previously acquired to compute quality_table altogether
    last_run_metrics = pandas.DataFrame([last_run_metrics], columns=metrics_data.columns[4:])
    data = pandas.concat([metrics_data.iloc[:, 4:], last_run_metrics])

    if anomaly_detection_method == "iforest":
        quality_table = create_and_fill_quality_table_using_iforest(data)
    else:
        # TODO: test for backward compatibility after refactoring
        quality_table = create_and_fill_quality_table_using_percentiles(data)

    return quality_table


def recompute_quality_table_and_predict_new_qualities(last_run_metrics, metrics_names, previous_metrics_data, previous_qualities_data):
    """ This method computes qualities of the new run, based on previous metrics data.
        Decision boundaries in both approaches are learnt from previously recorded metrics,
        and evaluated on new run data to assign qualities for each metric. """

    if anomaly_detection_method == "iforest":
        # recompute quality table for old runs and predict new qualities based on that
        quality_table, new_metrics_qualities = estimate_qualities_using_iforest(last_run_metrics, previous_metrics_data)

        db_connector.update_all_databases_with_qualities(quality_table, previous_metrics_data)

    else:
        # this method doesn't recompute quality table, it relies on first N records to to compute new metrics qualities
        # TODO: test for backward compatibility after refactoring
        new_metrics_qualities = estimate_qualities_using_percentiles(last_run_metrics, metrics_names, previous_metrics_data, previous_qualities_data)

    return new_metrics_qualities


def estimate_qualities_using_iforest(last_run_metrics, previous_metrics_data):
    """ Introduced in v.0.3.79: a method to recompute qualities for the metrics database (for all previously acquired
        measurement), and then to predict qualities for last (new) run metrics based on previous ones. """

    last_run_metrics = pandas.DataFrame([last_run_metrics], columns=previous_metrics_data.columns[4:])
    quality_table = previous_metrics_data[:]

    for metric in all_metrics:

        old_metric_values = numpy.array(previous_metrics_data.loc[:, metric]).reshape(-1, 1)
        new_metric = numpy.array(last_run_metrics.loc[:, metric]).reshape(-1, 1)

        # effectively, allows ~15% of outliers
        iforest = IsolationForest()
        # train and predict on the same values, since it's the first time
        iforest.fit(old_metric_values)

        # recompute quality table first
        predicted_outliers = iforest.predict(old_metric_values)
        # correct depending on the metrics, get array of {0,1}
        corrected_outliers = anomaly_detector.correct_outlier_prediction_for_metric(metric, predicted_outliers, old_metric_values, old_metric_values)
        # define quality for each metric individually
        quality_table.loc[:, metric] = corrected_outliers

        # now predict qualities of the last (new) run
        new_prediction = iforest.predict(new_metric)
        # correct depending on the metrics, get array of {0,1}
        corrected_prediction = anomaly_detector.correct_outlier_prediction_for_metric(metric, new_prediction, new_metric, old_metric_values)

        last_run_metrics.loc[:, metric] = corrected_prediction

    # summarise individual qualities
    quality_table.insert(0, "quality", quality_table.iloc[:, 1:].sum(axis=1) > 7)

    return quality_table, list(last_run_metrics.iloc[0,:])


def estimate_qualities_using_percentiles(last_run_metrics, metrics_names, previous_metrics_data, previous_qualities_data):
    """ This method takes new (last uploaded) run metrics and assigns qualities to it using percentiles
        of the previously acquired runs from QC metrics database (metrics, qualities dataframes). """

    # create dataframe with values and names to avoid any issues of ordering
    new_run_data = pandas.DataFrame([last_run_metrics, [0 for x in last_run_metrics]], columns=metrics_names, index=["value", "quality"])

    for metric in ["resolution_200", "resolution_700", "signal", "s2b", "s2n"]:

        good_values_indexes = previous_qualities_data.loc[:, metric] == 1  # get indexes based on previous qualities

        # choose values of "good" quality to compute a boundary
        good_values = previous_metrics_data.loc[good_values_indexes, metric]
        lower_boundary = numpy.percentile(good_values, 25)  # set lowest "good" value

        # define quality for each metric individually
        new_run_data.loc["quality", metric] = int(new_run_data.loc["value", metric] > lower_boundary)

    for metric in ["average_accuracy", "chemical_dirt", "instrument_noise", "baseline_25_150", "baseline_50_150", "baseline_25_650", "baseline_50_650"]:

        good_values_indexes = previous_qualities_data.loc[:, metric] == 1  # get indexes based on previous qualities

        # choose values of "good" quality to compute a boundary
        good_values = previous_metrics_data.loc[good_values_indexes, metric]
        upper_boundary = numpy.percentile(good_values, 75)  # set highest "good" value

        # define quality for each metric individually
        new_run_data.loc["quality", metric] = int(new_run_data.loc["value", metric] < upper_boundary)

    for metric in ["isotopic_presence", "transmission", "fragmentation_305", "fragmentation_712"]:

        good_values_indexes = previous_qualities_data.loc[:, metric] == 1  # get indexes based on previous qualities

        # choose values of "good" quality to compute a boundaries
        good_values = previous_metrics_data.loc[good_values_indexes, metric]
        lower_boundary, upper_boundary = numpy.percentile(good_values, [5, 95])  # set interval of "good" values

        # define quality for each metric individually
        metric_value = new_run_data.loc["value", metric]
        new_run_data.loc["quality", metric] = int(lower_boundary < metric_value < upper_boundary)

    metrics_qualities = list(new_run_data.loc['quality', :])

    return metrics_qualities


def assign_metrics_qualities(last_run_metrics, metrics_names, last_ms_run, in_debug_mode=False):
    """ This method calculates quality values for metrics of the last run,
        based on the previous entries of QC metrics database. """

    if not (os.path.isfile(qc_metrics_database_path) or os.path.isfile(qc_features_database_path) or os.path.isfile(qc_tunes_database_path)):
        # if there's yet no databases, return all "good"
        qualities = [1 for x in last_run_metrics]

    else:
        metrics_db = db_connector.create_connection(qc_metrics_database_path)

        meta_data, colnames = db_connector.fetch_table(metrics_db, "qc_meta")
        meta_data = pandas.DataFrame(meta_data, columns=colnames)
        # get meta_ids of all runs corresponding to the same buffer
        meta_ids = meta_data.loc[meta_data['buffer_id'] == last_ms_run['buffer_id'], 'id']

        # get metrics data with meta_ids corresponding to the same buffer
        metrics_data, colnames = db_connector.fetch_table(metrics_db, "qc_metrics")
        metrics_data = pandas.DataFrame(metrics_data, columns=colnames)
        metrics_data = metrics_data[metrics_data['meta_id'].isin(meta_ids)]

        # get metrics data with meta_ids corresponding to the same buffer
        qualities_data, _ = db_connector.fetch_table(metrics_db, "qc_metrics_qualities")
        qualities_data = pandas.DataFrame(qualities_data, columns=colnames)
        qualities_data = qualities_data[qualities_data['meta_id'].isin(meta_ids)]

        if metrics_data.shape[0] < min_number_of_runs[anomaly_detection_method]:
            # it's not enough data to assign quality, return all "good"
            qualities = [1 for x in last_run_metrics]

        elif metrics_data.shape[0] == min_number_of_runs[anomaly_detection_method]:
            # compute qualities for all the runs in the db, including this one

            # compute quality table for the first time
            quality_table = recompute_quality_table_for_all_runs(last_run_metrics, metrics_data)

            db_connector.update_all_databases_with_qualities(quality_table, metrics_data)

            # last run metrics qualities are in the last row of quality table now
            qualities = list(quality_table.iloc[-1, 1:])

        else:
            # recompute qualities for all the previous runs in the db,
            # and predict qualities of this run, based on previous runs

            # TODO: test this method (add "new" run locally)
            qualities = recompute_quality_table_and_predict_new_qualities(last_run_metrics, metrics_names, metrics_data, qualities_data)

    return qualities


if __name__ == '__main__':

    # TODO: refactoring: split (metrics generation), (working with databases) and (old methods)

    pass


