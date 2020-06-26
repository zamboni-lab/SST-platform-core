
import numpy, pandas, scipy, seaborn, math
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from src.qcmg import db_connector
from matplotlib import pyplot
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import ks_2samp, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency
from collections import Counter
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import ward, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy


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

    pandas.DataFrame(features_transformed).to_csv('/Users/andreidm/ETH/projects/monitoring_system/res/nas2/sparse_pca_features.csv', index=False)


def get_features_data(path="/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"):
    """ This method read metrics database,
        returns a matrix with metrics, metrics names, arrays of quality and acquisitions dates. """

    conn = db_connector.create_connection(path)
    database_1, colnames = db_connector.fetch_table(conn, "qc_features_1")
    database_2, _ = db_connector.fetch_table(conn, "qc_features_2")

    features_1 = numpy.array(database_1)
    features_2 = numpy.array(database_2)

    meta = features_1[:, :4]
    features = numpy.hstack([features_1[:, 4:].astype(numpy.float), features_2[:, 4:].astype(numpy.float)])

    return meta, features, colnames


def get_meta_data(path="/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"):
    """ This method read metrics database,
        returns a matrix with metrics, metrics names, arrays of quality and acquisitions dates. """

    conn = db_connector.create_connection(path)
    meta_data, colnames = db_connector.fetch_table(conn, "qc_meta")

    return pandas.DataFrame(meta_data, columns=colnames)


def perform_k_means(features):

    range_n_clusters = [5]  # has the best score

    for n_clusters in range_n_clusters:

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(features)

        silhouette_avg = silhouette_score(features, cluster_labels)

        print("For n_clusters =", n_clusters, "\nThe average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(features, cluster_labels)
        print("Sample silhouette values:", list(sample_silhouette_values), "\n")


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = numpy.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = numpy.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def perform_sklearn_hierarchical_clustering(features):
    """ I tried to vary distance threshold with n_clusters=None. That gave the best score for 5 clusters.
        Then I tried varying n_clusters, and it also delivered 5 as the best. So, it's hardcoded now. """

    model = AgglomerativeClustering(n_clusters=5)  # gives the best score
    labels = model.fit_predict(features)

    score = silhouette_score(features, labels)
    print("n_clusters:", model.n_clusters_, "score:", score)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(features, labels)
    print("sample silhouette values:", list(sample_silhouette_values), "\n")

    pyplot.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    pyplot.xlabel("Number of points in node (or index of point if no parenthesis).")
    pyplot.show()


def plot_sklearn_dendrogram(features):

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(features)

    print(model.n_clusters_)
    print(model.labels_)

    pyplot.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=3)
    pyplot.xlabel("Number of points in node (or index of point if no parenthesis).")
    pyplot.show()


def perform_hierarchical_clustering_and_plot(features):

    Z = linkage(features, 'ward')

    labels = fcluster(Z, t=5, criterion='maxclust')
    score = silhouette_score(features, labels)
    print("silhouette score:", score)
    print(labels)

    dn = hierarchy.dendrogram(Z)
    pyplot.show()


def get_tunes_and_names(path="/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_tunes_database.sqlite"):
    """ This method reads a database with tunes, makes some preprocessing and returns
        categorical and continuous tunes with names. """

    conn = db_connector.create_connection(path)
    database, colnames = db_connector.fetch_table(conn, "qc_tunes")

    tunes = numpy.array(database)

    # extract numeric values only
    indices = [10, 13, 16]
    indices.extend([i for i in range(18, 151)])

    # compose arrays
    tunes = numpy.vstack([tunes[:, i].astype(numpy.float) for i in indices]).T
    colnames = numpy.array(colnames)[indices]

    # remove nans
    tunes = numpy.delete(tunes, range(89, 116), 1)  # these columns contain only zeros and nans
    colnames = numpy.delete(colnames, range(89, 116), 0)

    # get numbers of unique values for each tune
    unique_values_numbers = numpy.array(
        [pandas.DataFrame(tunes).iloc[:, i].unique().shape[0] for i in range(pandas.DataFrame(tunes).shape[1])])

    # get only tunes with at least 2 different values
    informative_tunes = tunes[:, numpy.where(unique_values_numbers > 1)[0]]
    informative_colnames = colnames[numpy.where(unique_values_numbers > 1)[0]]

    # split tunes into two groups
    continuous_tunes, categorical_tunes = [], []
    continuous_names, categorical_names = [], []

    for i in range(informative_tunes.shape[1]):
        # let 12 be a max number of values for a tune to be categorical
        if len(set(informative_tunes[:, i])) > 12:
            continuous_tunes.append(informative_tunes[:, i])
            continuous_names.append(informative_colnames[i])
        else:
            categorical_tunes.append(informative_tunes[:, i])
            categorical_names.append(informative_colnames[i])

    continuous_tunes = numpy.array(continuous_tunes).T
    categorical_tunes = numpy.array(categorical_tunes).T

    return continuous_tunes, continuous_names, categorical_tunes, categorical_names


def perform_global_clustering(condensed_features, title=""):
    """ Global clustering: reflects two buffers """

    t = 2

    Z = linkage(condensed_features, 'ward')
    labels = fcluster(Z, t=t, criterion='maxclust')
    score = round(silhouette_score(condensed_features, labels), 3)
    print("t =", t, "silhouette score =", score)
    print(labels)

    # TODO: assess accuracy of separating buffers

    dn = hierarchy.dendrogram(Z)
    pyplot.title(title + ", t={}".format(t) + ", score={}".format(score))
    pyplot.show()


def find_and_perform_best_clustering(features, title=""):
    """ More special clustering:
        1) finding the best score,
        2) filtering out single classes,
        3) finding the best score again """

    Z = linkage(features, 'ward')

    results = {'t': [], 'score': [], 'labels': []}

    for t in range(2, 30):
        labels = fcluster(Z, t=t, criterion='maxclust')
        score = silhouette_score(features, labels)

        results['t'].append(t)
        results['score'].append(score)
        results['labels'].append(list(labels))

    # find best clustering
    max_score = max(results['score'])
    print("best score before filtering:", max_score)
    labels = results['labels'][results['score'].index(max_score)]

    # find unique classes and how frequent they are
    unique_labels = list(set(labels))
    occurencies = [labels.count(label) for label in unique_labels]

    qc_outliers_indices = []
    for i in range(len(occurencies)):
        # if some class appears only once, we treat it as outlier
        if occurencies[i] == 1:
            qc_outliers_indices.append(labels.index(unique_labels[i]))

    # drop outliers
    features = features.drop(features.index[qc_outliers_indices])
    print("dropped:", qc_outliers_indices)

    # recompute Z
    Z = linkage(features, 'ward')

    results = {'t': [], 'score': []}

    # find best score again
    for t in range(2, 30):
        labels = fcluster(Z, t=t, criterion='maxclust')
        score = silhouette_score(features, labels)
        # print("t=", t, "score=", score)

        results['t'].append(t)
        results['score'].append(score)

    max_score = max(results['score'])
    best_t = results['t'][results['score'].index(max_score)]

    labels = fcluster(Z, t=best_t, criterion='maxclust')
    score = round(silhouette_score(features, labels), 3)
    print("best score after filtering =", score)
    print("t =", best_t)

    dn = hierarchy.dendrogram(Z)
    pyplot.title(title + ", t={}".format(best_t) + ", score={}".format(score))
    pyplot.show()


if __name__ == "__main__":

    if False:
        # condense features and save features
        perform_sparse_pca()

    condensed_features = pandas.read_csv("/Users/andreidm/ETH/projects/monitoring_system/res/nas2/sparse_pca_features.csv")

    meta_info, features, colnames = get_features_data()
    full_meta_data = get_meta_data()
    continuous_tunes, continuous_names, categorical_tunes, categorical_names = get_tunes_and_names()

    if False:
        # testing of k-means
        perform_k_means(condensed_features)

    if True:
        # cluster by buffer for the whole dataset
        perform_global_clustering(condensed_features, title="All data")

    if False:
        # find the best clustering for the whole dataset
        find_and_perform_best_clustering(condensed_features, title='All data')

    if False:
        # clustering of "IPA_H2O" buffer
        ipa_h2o_subset = condensed_features.iloc[numpy.where(full_meta_data.iloc[:, 15] == 'IPA_H2O')[0], :]
        find_and_perform_best_clustering(ipa_h2o_subset, title="IPA_H2O")

    if False:
        # clustering of "IPA_H2O_DMSO" buffer
        dmso_subset = condensed_features.iloc[numpy.where(full_meta_data.iloc[:, 15] == 'IPA_H2O_DMSO')[0], :]
        find_and_perform_best_clustering(dmso_subset, title="IPA_H2O_DMSO")
