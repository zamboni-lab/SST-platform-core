
import pandas, numpy
from matplotlib import pyplot
from src.qcmg import metrics_generator, db_connector
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from PyAstronomy import pyasl
from sklearn.linear_model import LinearRegression
from src.constants import qc_metrics_database_path, all_metrics, time_periods
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


def explore_linear_trends_in_data():
    """ This method runs linear regression for metrics in the database, split by buffer.
        NOT USED in the project code here, but implemented almost the same way in shiny-qc.  """

    last_ms_run = {'buffer_id': 'IPA_H2O_DMSO'}  # for testing here

    metrics_db = db_connector.create_connection(qc_metrics_database_path)

    meta_data, colnames = db_connector.fetch_table(metrics_db, "qc_meta")
    meta_data = pandas.DataFrame(meta_data, columns=colnames)
    # get meta_ids of all runs corresponding to the same buffer
    meta_ids = meta_data.loc[meta_data['buffer_id'] == last_ms_run['buffer_id'], 'id']

    # get metrics data with meta_ids corresponding to the same buffer
    metrics_data, colnames = db_connector.fetch_table(metrics_db, "qc_metrics")
    metrics_data = pandas.DataFrame(metrics_data, columns=colnames)
    metrics_data = metrics_data[metrics_data['meta_id'].isin(meta_ids)]

    # get trends for certain periods of time

    linear_trends = {}

    for period, number_of_days in time_periods:

        linear_trends[period] = []

        # sort by dates!
        metrics_data = metrics_data.sort_values(by='acquisition_date')

        # define the starting date to take records from
        date_from = str(datetime.fromisoformat(metrics_data["acquisition_date"].values[-1]) - timedelta(days=number_of_days))
        # select records for specified time period
        metrics_subset = metrics_data[metrics_data['acquisition_date'] >= date_from]

        # convert dates to X (= number of days since 0 time point)
        dates = [datetime.fromisoformat(date) for date in metrics_subset['acquisition_date'].values]
        X = [0]

        for i in range(1, len(dates)):
            delta = dates[i] - dates[i-1]
            delta_in_days = delta.days + delta.seconds / 3600 / 24
            X.append(X[i-1] + delta_in_days)

        for metric in all_metrics:

            X = numpy.array(X).reshape(-1,1)
            y = metrics_subset[metric].values.reshape(-1,1)

            y = StandardScaler().fit_transform(y)
            reg = LinearRegression().fit(X, y)

            score = round(reg.score(X, y), 4)
            coef = round(reg.coef_[0][0], 4)

            if score < 0.01:
                trend_value = "unchanged"
            else:
                if abs(coef) < 0.05:
                    trend_value = "unchanged"
                else:
                    if coef < 0:
                        trend_value = "decreasing"
                    else:
                        trend_value = "increasing"

            linear_trends[period].append({metric: trend_value})

            subscript = metric + ": " + trend_value + "\n" + period + ": score = " + str(score) + ", coef = " + str(coef)
            pyplot.figure()
            pyplot.plot(X, y, 'ko')
            pyplot.plot(X, reg.predict(X), 'b-')
            pyplot.title(subscript)
            pyplot.xlabel("days")
            pyplot.grid()
            pyplot.savefig("/Users/andreidm/ETH/papers_posters/monitoring_system/img/" + period + "_" + metric + ".pdf")
            # pyplot.show()

    for period in linear_trends:
        print(period)
        for metric in linear_trends[period]:
            print('\t', metric)


def correct_outlier_prediction_for_metric(metric, prediction, test_values, train_values):
    """ This method corrects outlier prediction of methods, that don't take into account the nature of the metric.
        E.g., predicted outliers with very high resolution are marked as normal values,
              predicted outliers with very low chemical_dirt are marked as normal values, as well. """

    prediction[prediction == -1] = 0  # reformat predictions to {0,1}

    if metric in ["resolution_200", "resolution_700", "signal", "s2b", "s2n"]:

        train_values = train_values.reshape(1,-1)[0]
        test_values = test_values.reshape(1,-1)[0]
        # only low values can be marked as outliers, while high values are ok
        values_higher_than_threshold = test_values > numpy.percentile(train_values, 10)  # was median before
        corrected_prediction = prediction | values_higher_than_threshold  # vectorized bitwise 'or' operator

    elif metric in ["average_accuracy", "chemical_dirt", "instrument_noise", "baseline_25_150", "baseline_50_150", "baseline_25_650", "baseline_50_650"]:

        train_values = train_values.reshape(1,-1)[0]
        test_values = test_values.reshape(1,-1)[0]
        # only high values can be marked as outliers, while low values are ok
        values_lower_than_threshold = test_values < numpy.percentile(train_values, 90)  # was median before
        corrected_prediction = prediction | values_lower_than_threshold  # vectorized bitwise 'or' operator
    else:
        # no correction is done for metrics:
        # "isotopic_presence", "transmission", "fragmentation_305", "fragmentation_712"
        return prediction

    return corrected_prediction


def compare_outlier_prediction_methods():
    """ This method evaluates outlier prediction for each QC metric using methods:
        - Isolation Forest (seems the best),
        - Local Outlier Factor,
        - Quantile-based (in-house),
        - Generalized ESD (Extreme Studentized Deviate).
        Minimal data preprocessing and plotting is done. """

    # qualities_data, _ = db_connector.fetch_table(conn, "qc_metrics_qualities")
    # qualities_data = pandas.DataFrame(qualities_data, columns=colnames)

    metrics_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_metrics_database.sqlite"

    conn = db_connector.create_connection(metrics_path)
    metrics_data, colnames = db_connector.fetch_table(conn, "qc_metrics")

    # convert to dataframes for convenience
    metrics_data = pandas.DataFrame(metrics_data, columns=colnames)
    metrics_data = metrics_data.loc[metrics_data["acquisition_date"] < "2020-03-29", :]  # remove Mauro's dataset

    quality_table = metrics_generator.recompute_quality_table_for_all_runs(metrics_data)

    # pandas.set_option('display.max_rows', None)
    # pandas.set_option('display.max_columns', None)
    # print(quality_table)

    for metric_name in all_metrics:
        # reshape data to feed to models
        single_metric = numpy.array(metrics_data.loc[:, metric_name]).reshape(-1, 1)

        # detect outliers with isolation forest
        forest = IsolationForest(random_state=0)  # effectively, allows ~15% of outliers
        forest.fit(single_metric)
        forest_prediction = forest.predict(single_metric)
        forest_corrected_prediction = correct_outlier_prediction_for_metric(metric_name, forest_prediction, single_metric, single_metric)

        # detect outliers with local outlier factor
        lof = LocalOutlierFactor()
        lof_prediction = lof.fit_predict(single_metric)
        lof_corrected_prediction = correct_outlier_prediction_for_metric(metric_name, lof_prediction, single_metric, single_metric)

        # detect outliers with GESD
        gesd_prediction_indices = pyasl.generalizedESD(single_metric, int(single_metric.shape[0] * 0.2), 0.05)[1]
        gesd_prediction = numpy.ones(shape=(single_metric.shape[0]))  # make an empty ("all good") array
        gesd_prediction[gesd_prediction_indices] = 0  # add predicted outliers by indices
        gesd_corrected_prediction = correct_outlier_prediction_for_metric(metric_name, gesd_prediction, single_metric, single_metric)

        # prepare data for plotting
        dates = metrics_data.loc[:, "acquisition_date"]
        dates_labels = numpy.array([str(date)[0:10] for date in metrics_data.loc[:, "acquisition_date"]])
        values = metrics_data.loc[:, metric_name]

        # plot
        fig, axs = pyplot.subplots(4, 1, sharex='col', figsize=(12, 8))

        # the strictest, non-adaptable
        axs[0].plot(dates, values, 'k-o')
        axs[0].plot(dates[quality_table[metric_name] == 0], values[quality_table[metric_name] == 0], 'r.')
        axs[0].title.set_text("quantile-based")
        axs[0].set_ylabel(metric_name)
        axs[0].grid()

        # less strict, adaptable -> optimal?
        axs[1].plot(dates, values, 'k-o')
        axs[1].plot(dates[forest_corrected_prediction == 0], values[forest_corrected_prediction == 0], 'r.')
        axs[1].title.set_text("isolation forest")
        axs[1].set_ylabel(metric_name)
        axs[1].grid()

        # more tolerant, adaptable
        axs[2].plot(dates, values, 'k-o')
        axs[2].plot(dates[lof_corrected_prediction == 0], values[lof_corrected_prediction == 0], 'r.')
        axs[2].title.set_text("local outlier factor")
        axs[2].set_ylabel(metric_name)
        axs[2].grid()

        # most tolerant, adaptable
        axs[3].plot(dates, values, 'k-o')
        axs[3].plot(dates[gesd_corrected_prediction == 0], values[gesd_corrected_prediction == 0], 'r.')
        axs[3].title.set_text("generalized ESD")
        axs[3].set_ylabel(metric_name)
        axs[3].grid()

        pyplot.xticks(dates[::2], dates_labels[::2], rotation='vertical')
        pyplot.tight_layout()
        pyplot.show()


def test_outliers_prediction():
    """ This method evaluates performance of Isolation Forest in prediction of outliers.
        Model is trained on previous records and tested on "newly generated" measurement(s) / run(s).
        Results of train and test are plotted together.

        (This logic was tested for different subsets of all available data. Method seemed to work decently). """

    metrics_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_metrics_database.sqlite"

    conn = db_connector.create_connection(metrics_path)
    metrics_data, colnames = db_connector.fetch_table(conn, "qc_metrics")

    # convert to dataframes for convenience
    metrics_data = pandas.DataFrame(metrics_data, columns=colnames)
    test_data = metrics_data.loc[(metrics_data["acquisition_date"] >= "2020-04-29"),:]
    metrics_data = metrics_data.loc[metrics_data["acquisition_date"] < "2020-03-30", :]

    for metric_name in all_metrics:
        # reshape data to feed to models
        single_metric = numpy.array(metrics_data.loc[:, metric_name]).reshape(-1, 1)
        test_metric = numpy.array(test_data.loc[:, metric_name]).reshape(-1, 1)

        # detect outliers with isolation forest
        forest = IsolationForest(random_state=0)  # effectively, allows ~15% of outliers
        forest.fit(single_metric)

        train_prediction = forest.predict(single_metric)
        train_corrected_prediction = correct_outlier_prediction_for_metric(metric_name, train_prediction, single_metric, single_metric)

        test_prediction = forest.predict(test_metric)
        test_corrected_prediction = correct_outlier_prediction_for_metric(metric_name, test_prediction, test_metric, single_metric)

        # prepare data for plotting
        dates = metrics_data.loc[:, "acquisition_date"]
        dates_labels = numpy.array([str(date)[0:10] for date in metrics_data.loc[:, "acquisition_date"]])
        values = metrics_data.loc[:, metric_name]

        test_dates = test_data.loc[:, "acquisition_date"]
        test_dates_labels = numpy.array([str(date)[0:10] for date in test_data.loc[:, "acquisition_date"]])
        test_values = test_data.loc[:, metric_name]

        # plot
        fig, axs = pyplot.subplots(1, 2, sharey=True, figsize=(12, 6))

        pyplot.setp(axs, xticks=[], xticklabels=[])

        axs[0].plot(dates, values, 'k-o')
        axs[0].plot(dates[train_corrected_prediction == 0], values[train_corrected_prediction == 0], 'y.')
        axs[0].title.set_text("training")
        axs[0].set_ylabel(metric_name)
        axs[0].grid()

        axs[1].plot(test_dates, test_values, 'k-o')
        axs[1].plot(test_dates[test_corrected_prediction == 0], test_values[test_corrected_prediction == 0], 'r.')
        axs[1].title.set_text("test")
        axs[1].set_ylabel(metric_name)
        axs[1].grid()

        pyplot.sca(axs[0])
        pyplot.xticks(dates[::3], dates_labels[::3], rotation='vertical')
        pyplot.sca(axs[1])
        pyplot.xticks(test_dates, test_dates_labels, rotation='vertical')

        pyplot.tight_layout()
        # pyplot.savefig("/Users/andreidm/ETH/papers_posters/monitoring_system/img/" + metric_name + ".pdf")
        pyplot.show()


if __name__ == "__main__":

    # qualities_data, _ = db_connector.fetch_table(conn, "qc_metrics_qualities")
    # qualities_data = pandas.DataFrame(qualities_data, columns=colnames)

    test_outliers_prediction()
    # explore_linear_trends_in_data()
