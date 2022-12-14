
import os, numpy, pandas
from src.constants import min_number_of_metrics_to_assess_quality as min_number_of_runs
from src.constants import percent_of_good_metrics_for_good_quality as percent_of_good
from src.constants import anomaly_detection_method, all_metrics
from src.qcmg import metrics_generator, sqlite_connector
from src.analysis import features_analysis, metrics_tunes_analysis

# a few constants here
folder = '/Users/andreidm/ETH/projects/monitoring_system/res/nas2/190_injections_15_12_20/'
new_metrics_database_path = folder + 'qc_metrics_15_12_20.sqlite'
new_features_database_path = folder + 'qc_features_15_12_20.sqlite'
new_tunes_database_path = folder + 'qc_tunes_15_12_20.sqlite'
in_debug_mode = True

qc_tunes_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_tunes_database.sqlite"
qc_metrics_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_metrics_database.sqlite"
qc_features_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"


def assign_metrics_qualities_for_local_db(last_run_metrics, metrics_names, buffer):
    """ This method calculates quality values for metrics of the last run,
        based on the previous entries of QC metrics database.

        COPY of the same method in metrics_generator with a few adjustments to serve db_splitter needs. """

    total_qcs = 0

    if not (os.path.isfile(new_metrics_database_path) or os.path.isfile(new_features_database_path) or os.path.isfile(new_tunes_database_path)):
        # if there's yet no databases, return all "good"
        qualities = [1 for x in last_run_metrics]
        total_qcs += 1

    else:
        metrics_db = sqlite_connector.create_connection(new_metrics_database_path)

        meta_data, colnames = sqlite_connector.fetch_table(metrics_db, "qc_meta")
        meta_data = pandas.DataFrame(meta_data, columns=colnames)
        # get meta_ids of all runs corresponding to the same buffer
        meta_ids = meta_data.loc[meta_data['buffer_id'] == buffer, 'id']
        total_qcs = meta_ids.shape[0] + 1

        # get metrics data with meta_ids corresponding to the same buffer
        metrics_data, colnames = sqlite_connector.fetch_table(metrics_db, "qc_metrics")
        metrics_data = pandas.DataFrame(metrics_data, columns=colnames)
        metrics_data = metrics_data[metrics_data['meta_id'].isin(meta_ids)]

        # get metrics data with meta_ids corresponding to the same buffer
        qualities_data, _ = sqlite_connector.fetch_table(metrics_db, "qc_metrics_qualities")
        qualities_data = pandas.DataFrame(qualities_data, columns=colnames)
        qualities_data = qualities_data[qualities_data['meta_id'].isin(meta_ids)]

        if metrics_data.shape[0] < min_number_of_runs[anomaly_detection_method]:
            # it's not enough data to assign quality, return all "good"
            qualities = [1 for x in last_run_metrics]

        elif metrics_data.shape[0] == min_number_of_runs[anomaly_detection_method]:
            # compute qualities for all the runs in the db, including this one

            # compute quality table for the first time
            quality_table = metrics_generator.recompute_quality_table_for_all_runs(last_run_metrics, metrics_data)

            sqlite_connector.update_all_databases_with_qualities(quality_table, metrics_data,
                                                                 metrics_db_path=new_metrics_database_path,
                                                                 features_db_path=new_features_database_path,
                                                                 tunes_db_path=new_tunes_database_path)

            # last run metrics qualities are in the last row of quality table now
            qualities = list(quality_table.iloc[-1, 1:])

        else:
            # recompute qualities for all the previous runs in the db,
            # and predict qualities of this run, based on previous runs
            quality_table, qualities = metrics_generator.estimate_qualities_using_iforest(last_run_metrics, metrics_data)

            sqlite_connector.update_all_databases_with_qualities(quality_table, metrics_data,
                                                                 metrics_db_path=new_metrics_database_path,
                                                                 features_db_path=new_features_database_path,
                                                                 tunes_db_path=new_tunes_database_path)

    return qualities, {'buffer': buffer, 'total_qcs': total_qcs}


def split_databases():
    """ This method moves 191 single day (15-12-2020) QC injections to a new database.
        6 QC injections (of 2 batches) from that day are kept in the common database. """

    # read qc metrics
    metrics, metrics_names, acquisition, quality = metrics_tunes_analysis.get_metrics_data(qc_metrics_database_path)
    # read tunes
    tunes, tunes_names = metrics_tunes_analysis.get_tunes_and_names(qc_tunes_database_path, no_filter=True)
    # read meta data
    full_meta_data = features_analysis.get_meta_data(path=qc_features_database_path)
    # read features
    features_meta, features, features_names = features_analysis.get_features_data(path=qc_features_database_path)

    # FILTER
    rep_indices = [numpy.where(full_meta_data['acquisition_date'].values == x)[0][0] for x in
                   full_meta_data['acquisition_date'].values if x[:10] == '2020-12-15']

    tunes = tunes[numpy.array(rep_indices), :]
    metrics = metrics[numpy.array(rep_indices), :]
    features = features[numpy.array(rep_indices), :]
    full_meta_data = full_meta_data.iloc[numpy.array(rep_indices), :]

    for i in range(metrics.shape[0]):

        # assign quality for each metric based on previous records
        metrics_qualities, info = assign_metrics_qualities_for_local_db(list(metrics[i, :]), metrics_names, 'IPA_H2O')
        metrics_qualities = [int(x) for x in metrics_qualities]
        quality = int(sum(metrics_qualities) >= int(len(all_metrics) * percent_of_good[anomaly_detection_method]))

        # prepare information for copying
        new_qc_run = {

            'md5': full_meta_data['md5'].values[i],
            'instrument': full_meta_data['instr'].values[i],
            'user': full_meta_data['user'].values[i],
            'processing_date': full_meta_data['processing_date'].values[i],
            'acquisition_date': full_meta_data['acquisition_date'].values[i],
            'chemical_mix_id': full_meta_data['chemical_mix_id'].values[i],
            'buffer_id': full_meta_data['buffer_id'].values[i],
            'msfe_version': full_meta_data['msfe_version'].values[i],
            'scans_processed': {
                'normal': [int(full_meta_data['norm_scan_1'].values[i]),
                           int(full_meta_data['norm_scan_2'].values[i]),
                           int(full_meta_data['norm_scan_3'].values[i])],
                'chemical_noise': [17],  # constant
                'instrument_noise': [174]  # constant
            },

            'features_values': list(features[i, :]),
            'features_names': features_names[4:],
            'metrics_values': list(metrics[i, :]),
            'metrics_names': metrics_names,
            'metrics_qualities': metrics_qualities,
            'tunes_values': [
                list(tunes[i, 4:19]),  # meta
                list(tunes[i, 19:104]),  # actuals
                list(tunes[i, 104:])  # cals
            ],
            'tunes_names': [
                tunes_names[4:19],
                tunes_names[19:104],
                tunes_names[104:]
            ],

            'user_comment': full_meta_data['user_comment'].values[i],
            'quality': quality
        }

        # COPY to a new db
        if not (os.path.isfile(new_metrics_database_path) or os.path.isfile(new_features_database_path)
                or os.path.isfile(new_tunes_database_path)):

            # if there's yet no databases
            sqlite_connector.create_and_fill_qc_databases(new_qc_run,
                                                          metrics_db_path=new_metrics_database_path,
                                                          features_db_path=new_features_database_path,
                                                          tunes_db_path=new_tunes_database_path,
                                                          in_debug_mode=in_debug_mode)
            print('New QC databases have been created (SQLite)')
        else:
            # if the databases already exist
            sqlite_connector.insert_new_qc_run(new_qc_run,
                                               metrics_db_path=new_metrics_database_path,
                                               features_db_path=new_features_database_path,
                                               tunes_db_path=new_tunes_database_path,
                                               in_debug_mode=in_debug_mode)
            print('QC databases have been updated')

        # REMOVE from the old db
        run_id = str(full_meta_data['id'].values[i])

        if run_id not in ['157', '158', '159', '242', '249', '250']:
            sqlite_connector.remove_row_from_all_databases_by_id(run_id,
                                                                 metrics_db_path=qc_metrics_database_path,
                                                                 features_db_path=qc_features_database_path,
                                                                 tunes_db_path=qc_tunes_database_path)
            print('Run ID {} has been removed from old databases\n'.format(run_id))


if __name__ == '__main__':
    split_databases()