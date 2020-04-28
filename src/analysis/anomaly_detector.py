
import pandas, numpy
from matplotlib import pyplot
from src.msfe import db_connector
from src.qcmg import metrics_generator


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

    dates = metrics_data.loc[:, "acquisition_date"]
    dates_labels = numpy.array([str(date)[0:10] for date in metrics_data.loc[:, "acquisition_date"]])
    values = metrics_data.loc[:, "resolution_200"]

    fig, axs = pyplot.subplots(3, 1, sharex='col', figsize=(10,8))
    fig.suptitle("resolution_200")

    axs[0].plot(dates, values, 'k-o')
    axs[0].plot(dates[quality_table["resolution_200"] == 0], values[quality_table["resolution_200"] == 0], 'r.')
    axs[0].grid()

    axs[1].plot(dates, values, 'k-o')
    axs[1].plot(dates[quality_table["resolution_200"] == 0], values[quality_table["resolution_200"] == 0], 'r.')
    axs[1].grid()

    axs[2].plot(dates, values, 'k-o')
    axs[2].plot(dates[quality_table["resolution_200"] == 0], values[quality_table["resolution_200"] == 0], 'r.')
    axs[2].grid()

    pyplot.xticks(dates[::2], dates_labels[::2], rotation='vertical')
    pyplot.tight_layout()
    pyplot.subplots_adjust(top=0.95)
    pyplot.show()