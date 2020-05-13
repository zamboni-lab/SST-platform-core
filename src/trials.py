import numpy, pandas
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from src.msfe import db_connector
from src.constants import qc_metrics_database_path, all_metrics, time_periods
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler, StandardScaler

if __name__ == "__main__":

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
                pass
            else:
                if abs(coef) < 0.05:
                    pass
                else:
                    if coef < 0:
                        linear_trends[period].append({metric: "decreasing"})
                    else:
                        linear_trends[period].append({metric: "increasing"})

            # subscript = period + ": score = " + str(score) + ", coef = " + str(coef)
            # pyplot.plot(X, y, 'ko')
            # pyplot.plot(X, reg.predict(X), 'b-')
            # pyplot.title(subscript)
            # pyplot.grid()
            # pyplot.show()

    for period in linear_trends:
        print(period)
        for metric in linear_trends[period]:
            print('\t', metric)

