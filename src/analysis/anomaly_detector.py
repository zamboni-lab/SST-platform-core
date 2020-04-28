
import pandas
from src.msfe import db_connector
from src.constants import qc_metrics_database_path


if __name__ == "__main__":

    # create and fill metric quality table for existing qc_metrics_database
    conn = db_connector.create_connection(qc_metrics_database_path)
    metrics_data, colnames = db_connector.fetch_table(conn, "qc_metrics")
    qualities_data, _ = db_connector.fetch_table(conn, "qc_metrics_qualities")

    # convert to dataframes for convenience
    metrics_data = pandas.DataFrame(metrics_data, columns=colnames)
    qualities_data = pandas.DataFrame(qualities_data, columns=colnames)

