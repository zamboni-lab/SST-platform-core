import numpy, pandas, matplotlib, scipy, seaborn
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from src.msfe import db_connector
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def perform_sparse_pca():
    """ This method performs sparse PCA on the qc_features_database (local file of some version provided),
        prints sparsity value and variance fraction explained by first N components. """

    qc_features_database_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/nas2/qc_features_database.sqlite"

    conn = db_connector.create_connection(qc_features_database_path)
    database_1, _ = db_connector.fetch_table(conn, "qc_features_1")
    database_2, _ = db_connector.fetch_table(conn, "qc_features_2")

    features_1 = numpy.array(database_1)
    features_2 = numpy.array(database_2)

    features = numpy.hstack([features_1[:,5:].astype(numpy.float), features_2[:,5:].astype(numpy.float)])

    number_of_components = 5

    transformer = SparsePCA(n_components=number_of_components)
    transformer.fit(features)

    features_transformed = transformer.transform(features)
    print(features_transformed.shape)

    # fraction of zero values in the components_ (sparsity)
    print(numpy.mean(transformer.components_ == 0))

    # get variance explained
    variances = [numpy.var(features_transformed[:,i]) for i in range(number_of_components)]
    fraction_explained = [var / sum(variances) for var in variances]

    print(fraction_explained)


if __name__ == "__main__":

    qc_tunes_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_tunes_database.sqlite"

    conn = db_connector.create_connection(qc_tunes_database_path)
    database, colnames = db_connector.fetch_table(conn, "qc_tunes")

    tunes = numpy.array(database)

    # extract numeric values only
    indices = [10, 13, 16]
    indices.extend([i for i in range(18,151)])

    tunes = numpy.vstack([tunes[:, i].astype(numpy.float) for i in indices]).T
    colnames = numpy.array(colnames)[indices]

    # set full display
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)

    # get numbers of unique values for each tune
    unique_values_numbers = numpy.array([pandas.DataFrame(tunes).iloc[:,i].unique().shape[0] for i in range(pandas.DataFrame(tunes).shape[1])])

    # get only tunes with at least 2 different values
    informative_tunes = tunes[:, numpy.where(unique_values_numbers > 1)[0]]
    informative_colnames = colnames[numpy.where(unique_values_numbers > 1)[0]]

    # split tunes into two groups
    continuous_tunes, categorical_tunes = [], []
    continuous_names, categorical_names = [], []

    for i in range(informative_tunes.shape[1]):
        if len(set(informative_tunes[:, i])) > 12:
            continuous_tunes.append(informative_tunes[:, i])
            continuous_names.append(informative_colnames[i])
        else:
            categorical_tunes.append(informative_tunes[:, i])
            categorical_names.append(informative_colnames[i])

    continuous_tunes = numpy.array(continuous_tunes).T
    categorical_tunes = numpy.array(categorical_tunes).T

    # get correlation matrix for continuous
    df = pandas.DataFrame(continuous_tunes).corr()

    # plot a heatmap
    seaborn.heatmap(df, xticklabels=df.columns, yticklabels=df.columns)
    plt.show()

    pass

