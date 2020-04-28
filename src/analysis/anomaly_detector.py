
import pandas, numpy
from matplotlib import pyplot
from src.msfe import db_connector
from src.qcmg import metrics_generator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


if __name__ == "__main__":

    # qualities_data, _ = db_connector.fetch_table(conn, "qc_metrics_qualities")
    # qualities_data = pandas.DataFrame(qualities_data, columns=colnames)

    metrics_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_metrics_database.sqlite"

    conn = db_connector.create_connection(metrics_path)
    metrics_data, colnames = db_connector.fetch_table(conn, "qc_metrics")

    # convert to dataframes for convenience
    metrics_data = pandas.DataFrame(metrics_data, columns=colnames)

    quality_table = metrics_generator.compute_quality_table_first_time(metrics_data)

    # pandas.set_option('display.max_rows', None)
    # pandas.set_option('display.max_columns', None)
    # print(quality_table)

    # reshape data to feed to models
    single_metric = numpy.array(metrics_data.loc[:, "resolution_200"]).reshape(-1, 1)

    # detect outliers with isolation forest
    forest = IsolationForest(random_state=0)
    forest.fit(single_metric)
    forest_prediction = forest.predict(single_metric)
    # TODO: adjust predictions so that high "resolution" and low "accuracy" values are not marked as outliers

    # detect outliers with local outlier factor
    lof = LocalOutlierFactor()
    lof_prediction = lof.fit_predict(single_metric)

    # prepare data for plotting
    dates = metrics_data.loc[:, "acquisition_date"]
    dates_labels = numpy.array([str(date)[0:10] for date in metrics_data.loc[:, "acquisition_date"]])
    values = metrics_data.loc[:, "resolution_200"]

    # plot
    fig, axs = pyplot.subplots(3, 1, sharex='col', figsize=(10,8))

    axs[0].plot(dates, values, 'k-o')
    axs[0].plot(dates[quality_table["resolution_200"] == 0], values[quality_table["resolution_200"] == 0], 'r.')
    axs[0].title.set_text("quantile-based")
    axs[0].set_ylabel('resolution_200')
    axs[0].grid()

    axs[1].plot(dates, values, 'k-o')
    axs[1].plot(dates[forest_prediction == -1], values[forest_prediction == -1], 'r.')
    axs[1].title.set_text("isolation forest")
    axs[1].set_ylabel('resolution_200')
    axs[1].grid()

    axs[2].plot(dates, values, 'k-o')
    axs[2].plot(dates[lof_prediction == -1], values[lof_prediction == -1], 'r.')
    axs[2].title.set_text("local outlier factor")
    axs[2].set_ylabel('resolution_200')
    axs[2].grid()

    pyplot.xticks(dates[::2], dates_labels[::2], rotation='vertical')

    pyplot.tight_layout()
    pyplot.show()