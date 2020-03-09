
import numpy, pandas
from sklearn.datasets import make_friedman1
from sklearn.decomposition import SparsePCA
from src.msfe import db_connector


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

    qc_tunes_database_path = "/Users/andreidm/ETH/projects/ms_feature_extractor/res/nas2/qc_tunes_database.sqlite"

    conn = db_connector.create_connection(qc_tunes_database_path)
    database_1, _ = db_connector.fetch_table(conn, "qc_tunes")
    tunes = numpy.array(database_1)

    indices = [10, 13, 16]
    indices.extend([i for i in range(18,151)])

    tunes = numpy.vstack([tunes[:, i].astype(numpy.float) for i in indices]).T

    description = pandas.DataFrame(tunes).describe()

    pass