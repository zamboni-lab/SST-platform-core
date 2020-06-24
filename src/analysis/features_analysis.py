
import numpy, pandas, scipy, seaborn, math
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from src.qcmg import db_connector
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency
from collections import Counter
from statsmodels.stats.multitest import multipletests


def perform_sparse_pca():
    """ This method performs sparse PCA on the qc_features_database (local file of some version provided),
        prints sparsity value and variance fraction explained by first N components. """

    _, features, _ = get_features_data()

    number_of_components = 10

    transformer = SparsePCA(n_components=number_of_components)
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)
    features_transformed = transformer.fit_transform(scaled_features)

    print(features_transformed.shape)

    # fraction of zero values in the components_ (sparsity)
    print(numpy.mean(transformer.components_ == 0))

    # get variance explained
    variances = [numpy.var(features_transformed[:,i]) for i in range(number_of_components)]
    fraction_explained = [var / sum(variances) for var in variances]

    print(fraction_explained)


def get_features_data(path="/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"):
    """ This method read metrics database,
        returns a matrix with metrics, metrics names, arrays of quality and acquisitions dates. """

    conn = db_connector.create_connection(path)
    database_1, colnames = db_connector.fetch_table(conn, "qc_features_1")
    database_2, _ = db_connector.fetch_table(conn, "qc_features_2")

    features_1 = numpy.array(database_1)
    features_2 = numpy.array(database_2)

    meta = features_1[:, :5]
    features = numpy.hstack([features_1[:, 5:].astype(numpy.float), features_2[:, 5:].astype(numpy.float)])

    return meta, features, colnames


if __name__ == "__main__":

    # read qc features
    # meta_info, features, colnames = get_features_data()
    perform_sparse_pca()