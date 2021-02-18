
import numpy, pandas, scipy, seaborn, math, time, umap
from src.qcmg import db_connector
from src.constants import user
from src.analysis import metrics_tunes_analysis
from src.msfe import type_generator

from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize, MinMaxScaler
from matplotlib import pyplot
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix
from scipy.stats import ks_2samp, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency
from collections import Counter
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import ward, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import Ridge, Lasso
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn import manifold


def plot_sparse_pca_variance_explained(filter_dmso=True):

    _, features, _ = get_features_data()

    if filter_dmso:
        # filter out DMSO samples
        full_meta_data = get_meta_data()
        ipa_h20_indices = numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O')[0]
        features = features[ipa_h20_indices,:]

        file_mark = "without_dmso"
    else:
        file_mark = "with_dmso"

    total_variance_percent = []
    number_of_components = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for n in number_of_components:

        transformer = SparsePCA(n_components=n)
        scaler = StandardScaler()

        scaled_features = scaler.fit_transform(features)
        features_transformed = transformer.fit_transform(scaled_features)

        print(features_transformed.shape)

        # calculate properly total variance explained
        P_hat = transformer.components_.T  # loadings
        T_hat = numpy.dot(numpy.dot(scaled_features, P_hat), numpy.linalg.pinv(numpy.dot(P_hat.T, P_hat)))  # scores
        E = scaled_features - numpy.dot(T_hat, P_hat.T)  # errors

        trace_PT_squared = numpy.trace(numpy.dot(numpy.dot(P_hat, T_hat.T), numpy.dot(T_hat, P_hat.T)))
        trace_E_squared = numpy.trace(numpy.dot(E.T, E))
        trace_features_squared = numpy.trace(numpy.dot(scaled_features.T, scaled_features))

        TotPT = (trace_PT_squared + trace_E_squared) / trace_features_squared
        print("N components = {}, TotPT = {}".format(n, TotPT))
        print("Fraction of total variance explained:", round(trace_PT_squared / trace_features_squared, 3))
        total_variance_percent.append(trace_PT_squared / trace_features_squared)

        # fraction of zero values in the components_ (sparsity)
        print("Sparsity:", round(numpy.mean(transformer.components_ == 0), 3))

    pyplot.plot(number_of_components, total_variance_percent, '--bo')
    pyplot.grid()
    pyplot.title("Sparse PCA: explained variance")
    pyplot.show()


def plot_pca_variance_explained(filter_dmso=True):

    _, features, _ = get_features_data()

    if filter_dmso:
        # filter out DMSO samples
        full_meta_data = get_meta_data()
        ipa_h20_indices = numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O')[0]
        features = features[ipa_h20_indices,:]

        file_mark = "without_dmso"
    else:
        file_mark = "with_dmso"

    transformer = PCA()
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)
    features_transformed = transformer.fit_transform(scaled_features)

    # x = range(len(transformer.explained_variance_ratio_))
    # y = [sum(transformer.explained_variance_ratio_[:i]) for i in x]

    x = range(1, 1 + len(transformer.explained_variance_ratio_))[::5]
    y = transformer.explained_variance_ratio_[::5]

    pyplot.plot(x, y, '--b.')
    pyplot.grid()
    pyplot.title("PCA: explained variance")
    pyplot.xlabel("Principal components")
    pyplot.ylabel("Percent of total variance")
    pyplot.xticks(x)
    pyplot.tight_layout()
    pyplot.show()
    # pyplot.savefig("/Users/dmitrav/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/monitoring_system/img/fig1a.pdf")


def perform_sparse_pca(filter_dmso=True, n=None):
    """ This method performs sparse PCA on the qc_features_database (local file of some version provided),
        prints sparsity value and variance fraction explained by first N components. """

    _, features, _ = get_features_data()

    if filter_dmso:
        # filter out DMSO samples
        full_meta_data = get_meta_data()
        ipa_h20_indices = numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O')[0]
        features = features[ipa_h20_indices,:]
        print('dmso filtered: {} entries remained'.format(features.shape[0]))

        file_mark = "without_dmso"
    else:
        file_mark = "with_dmso"

    if n is None:
        number_of_components = 15
    else:
        number_of_components = n

    transformer = SparsePCA(n_components=number_of_components)
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)
    features_transformed = transformer.fit_transform(scaled_features)

    print(features_transformed.shape)

    # calculate properly total variance explained
    P_hat = transformer.components_.T
    T_hat = numpy.dot(numpy.dot(scaled_features, P_hat), numpy.linalg.pinv(numpy.dot(P_hat.T, P_hat)))
    E = scaled_features - numpy.dot(T_hat, P_hat.T)

    trace_PT_squared = numpy.trace(numpy.dot(numpy.dot(P_hat, T_hat.T), numpy.dot(T_hat, P_hat.T)))
    trace_E_squared = numpy.trace(numpy.dot(E.T, E))
    trace_features_squared = numpy.trace(numpy.dot(scaled_features.T, scaled_features))

    TotPT = (trace_PT_squared + trace_E_squared) / trace_features_squared
    print("N components = {}, TotPT = {}".format(number_of_components, TotPT))
    print("Fraction of total variance explained:", round(trace_PT_squared / trace_features_squared, 3))

    # fraction of zero values in the components_ (sparsity)
    print("Sparsity:", round(numpy.mean(transformer.components_ == 0), 3))

    pandas.DataFrame(transformer.components_).to_csv('/Users/andreidm/ETH/projects/monitoring_system/res/nas2/sparse_pca_loadings_{}_n={}.csv'.format(file_mark, number_of_components), index=False)
    pandas.DataFrame(features_transformed).to_csv('/Users/andreidm/ETH/projects/monitoring_system/res/nas2/sparse_pca_features_{}_n={}.csv'.format(file_mark, number_of_components), index=False)


def perform_pca(filter_dmso=True):
    """ This method performs PCA on the qc_features_database (local file of some version provided),
        prints sparsity value and variance fraction explained by first N components. """

    _, features, _ = get_features_data()

    if filter_dmso:
        # filter out DMSO samples
        full_meta_data = get_meta_data()
        ipa_h20_indices = numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O')[0]
        features = features[ipa_h20_indices,:]

        file_mark = "without_dmso"
    else:
        file_mark = "with_dmso"

    transformer = PCA(n_components=100)
    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(features)
    features_transformed = transformer.fit_transform(scaled_features)

    print(features_transformed.shape)

    # # fraction of zero values in the components_ (sparsity)
    # print(numpy.mean(transformer.components_ == 0))

    # percent of variance explained
    print(transformer.explained_variance_ratio_ * 100)

    # # pandas.DataFrame(transformer.components_).to_csv('/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/sparse_pca_loadings_{}.csv'.format(file_mark), index=False)
    #
    # # get variance explained
    # variances = [numpy.var(features_transformed[:,i]) for i in range(number_of_components)]
    # fraction_explained = [var / sum(variances) for var in variances]
    #
    # print(fraction_explained)
    #
    # # pandas.DataFrame(features_transformed).to_csv('/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/sparse_pca_features_{}.csv'.format(file_mark), index=False)


def get_features_data(path="/Users/{}/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite".format(user)):
    """ This method read metrics database,
        returns a matrix with metrics, metrics names, arrays of quality and acquisitions dates. """

    conn = db_connector.create_connection(path)
    if conn is None:
        raise ValueError("Database connection unsuccessful. Check out path. ")

    database_1, colnames_1 = db_connector.fetch_table(conn, "qc_features_1")
    database_2, colnames_2 = db_connector.fetch_table(conn, "qc_features_2")

    features_1 = numpy.array(database_1)
    features_2 = numpy.array(database_2)

    meta = features_1[:, :4]
    features = numpy.hstack([features_1[:, 4:].astype(numpy.float), features_2[:, 4:].astype(numpy.float)])
    colnames = [*colnames_1, *colnames_2[4:]]

    return meta, features, colnames


def get_meta_data(path="/Users/{}/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite".format(user)):
    """ This method read metrics database,
        returns a matrix with metrics, metrics names, arrays of quality and acquisitions dates. """

    conn = db_connector.create_connection(path)
    if conn is None:
        raise ValueError("Database connection unsuccessful. Check out path.")

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


def get_tunes_and_names(path="/Users/{}/ETH/projects/monitoring_system/res/nas2/qc_tunes_database.sqlite".format(user)):
    """ This method reads a database with tunes, makes some preprocessing and returns
        categorical and continuous tunes with names. """

    conn = db_connector.create_connection(path)
    if conn is None:
        raise ValueError("Database connection unsuccessful. Check out path. ")
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

        is_calibration_coef = False
        for n in ['2', '3', '4', '5', '6', '7']:
            if 'polynomial_' + n in informative_colnames[i]:
                is_calibration_coef = True
                break

        if not is_calibration_coef:
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

    dn = hierarchy.dendrogram(Z)
    pyplot.title(title + ", t={}".format(t) + ", score={}".format(score))
    pyplot.show()

    return labels


def perform_best_clustering(samples_labels, features, best_t, method='ward', title="", no_labels=False):

    # recompute Z
    Z = linkage(features, method)

    clusters = fcluster(Z, t=best_t, criterion='maxclust')
    score = round(silhouette_score(features, clusters), 3)
    print("clustering score =", score)

    samples_indices = features.index

    if no_labels:
        # plot with no labels
        dn = hierarchy.dendrogram(Z)
        plot = pyplot.gca()
        plot.axes.get_xaxis().set_visible(False)
    else:
        dn = hierarchy.dendrogram(Z, labels=samples_labels, leaf_rotation=90)

    pyplot.title(title + ", t={}".format(best_t) + ", score={}".format(score))
    pyplot.tight_layout()
    pyplot.show()

    # print cluster groups
    cluster_groups = {}
    for i in range(len(samples_indices)):
        if clusters[i] not in cluster_groups.keys():
            cluster_groups[clusters[i]] = {'labels': [samples_labels[i]], 'indices': [samples_indices[i]]}
        else:
            cluster_groups[clusters[i]]['labels'].append(samples_labels[i])
            cluster_groups[clusters[i]]['indices'].append(samples_indices[i])

    return cluster_groups


def find_and_perform_best_clustering(samples_labels, features, max_t=30, method='ward', title="", no_labels=False):
    """ More special clustering:
        1) finding the best score,
        2) filtering out single classes,
        3) finding the best score again """

    Z = linkage(features, method)

    results = {'t': [], 'score': [], 'labels': []}

    for t in range(2, max_t):
        clusters = fcluster(Z, t=t, criterion='maxclust')
        score = silhouette_score(features, clusters)

        results['t'].append(t)
        results['score'].append(score)
        results['labels'].append(list(clusters))

    # find best clustering
    max_score = max(results['score'])
    print("best score before filtering:", max_score)
    clusters = results['labels'][results['score'].index(max_score)]

    # find unique classes and how frequent they are
    unique_groups = list(set(clusters))
    occurencies = [clusters.count(label) for label in unique_groups]

    qc_outliers_indices = []
    for i in range(len(occurencies)):
        # if some class appears only once, we treat it as outlier
        if occurencies[i] == 1:
            qc_outliers_indices.append(clusters.index(unique_groups[i]))

    # drop outliers
    features.insert(0, 'label', samples_labels)
    # SHOULDN'T indices be +1, because of insertion?
    features = features.drop(features.index[qc_outliers_indices])
    print("dropped:", qc_outliers_indices)

    # assign index for dendrogram plotting
    samples_labels = features['label']
    features = features.drop(columns=['label'])

    # recompute Z
    Z = linkage(features, 'ward')

    results = {'t': [], 'score': []}

    # find best score again
    for t in range(2, max_t):
        clusters = fcluster(Z, t=t, criterion='maxclust')
        score = silhouette_score(features, clusters)
        print("t =", t, "score =", score)

        results['t'].append(t)
        results['score'].append(score)

    max_score = max(results['score'])
    best_t = results['t'][results['score'].index(max_score)]

    clusters = fcluster(Z, t=best_t, criterion='maxclust')
    score = round(silhouette_score(features, clusters), 3)
    print("best score after filtering =", score)
    print("t =", best_t)

    samples_indices = features.index

    if no_labels:
        # plot with no labels
        dn = hierarchy.dendrogram(Z)
        plot = pyplot.gca()
        plot.axes.get_xaxis().set_visible(False)
    else:
        dn = hierarchy.dendrogram(Z, labels=samples_labels, leaf_rotation=90)

    pyplot.title(title + ", t={}".format(best_t) + ", score={}".format(score))
    pyplot.tight_layout()
    pyplot.show()

    # print cluster groups
    cluster_groups = {}
    for i in range(len(samples_indices)):
        if clusters[i] not in cluster_groups.keys():
            cluster_groups[clusters[i]] = {'labels': [samples_labels[i]], 'indices': [samples_indices[i]]}
        else:
            cluster_groups[clusters[i]]['labels'].append(samples_labels[i])
            cluster_groups[clusters[i]]['indices'].append(samples_indices[i])

    return cluster_groups


def split_features_to_cont_and_cat(features, names):

    # get numbers of unique values for each feature
    unique_values_numbers = numpy.array([numpy.unique(features[:,i]).shape[0] for i in range(features.shape[1])])

    # get only tunes with at least 2 different values
    informative_features = features[:, numpy.where(unique_values_numbers > 1)[0]]
    informative_colnames = names[numpy.where(unique_values_numbers > 1)[0]]

    # split tunes into two groups
    continuous_features, categorical_features = [], []
    continuous_names, categorical_names = [], []

    for i in range(informative_features.shape[1]):
        # let 12 be a max number of values for a tune to be categorical
        if len(set(informative_features[:, i])) > 12:
            continuous_features.append(informative_features[:, i])
            continuous_names.append(informative_colnames[i])
        else:
            categorical_features.append(informative_features[:, i])
            categorical_names.append(informative_colnames[i])

    continuous_features = numpy.array(continuous_features).T
    categorical_features = numpy.array(categorical_features).T

    return continuous_features, continuous_names, categorical_features, categorical_names


def compare_cat_features_with_cat_tunes(features_cat, tunes_cat, tunes_names_cat, features_names_cat):
    """ Comment: looks like catergotical features don't bring any useful information on the tunes. """

    # create empty correlation matrix for categorical tunes
    df = pandas.DataFrame(numpy.empty([features_cat.shape[1], tunes_cat.shape[1]]))
    df.columns = tunes_names_cat

    for i in range(features_cat.shape[1]):
        for j in range(tunes_cat.shape[1]):

            correlation = metrics_tunes_analysis.get_theils_u_correlation(features_cat[:, i], tunes_cat[:, j])
            df.iloc[i, j] = correlation

            # look into variables closer if correlation is high
            if abs(correlation) > 0.7:

                fig, ax = pyplot.subplots(figsize=(10, 5))

                ax.scatter(tunes_cat[:, j], features_cat[:, i])

                # adds a title and axes labels
                ax.set_title(tunes_names_cat[j] + ' vs ' + features_names_cat[i] + ": theils u = " + str(correlation))
                ax.set_xlabel(tunes_names_cat[j])
                ax.set_ylabel(features_names_cat[i])

                # removing top and right borders
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                pyplot.show()

    seaborn.heatmap(df, xticklabels=df.columns, yticklabels=False)
    pyplot.tight_layout()
    pyplot.show()


def calculate_MI(x, y):
    """ Sklearn implementation of MI is used. There, the log base is e, so in fact ln is used, instead of log2.
        It means, that MI is bound to 2.71 (?) at max. """

    c_xy = numpy.histogram2d(x, y)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi / numpy.log(2)


def compute_mutual_info_between_tunes_and_features(features_cont, features_names_cont, tunes, tunes_names, features_type='features', tunes_type='continuous', inspection_mode=False):
    """ This method calculates mutual information between tunes and features. """

    # create dataframe to store results
    df = pandas.DataFrame(numpy.empty([len(features_names_cont), len(tunes_names)]), index=features_names_cont)

    # change names for better display
    for i in range(len(tunes_names)):
        tunes_names[i] = tunes_names[i].replace("default", "").replace("traditional", "trad").replace("polynomial", "poly")

    df.columns = tunes_names

    # calculate MIs and fill the dataframe
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):

            mi = calculate_MI(features_cont[:, i], tunes[:, j])
            df.iloc[i, j] = mi

            # look into variables closer if MI > 1 bit
            if inspection_mode and mi > 1.:
                fig, ax = pyplot.subplots(figsize=(10, 5))

                ax.scatter(tunes[:, j], features_cont[:, i])

                # adds a title and axes labels
                ax.set_title(df.index[i] + ' vs ' + df.columns[j] + ": MI = " + str(mi))
                ax.set_xlabel(df.columns[j])
                ax.set_ylabel(df.index[i])

                # removing top and right borders
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                pyplot.show()

    print("number of pairs with MI > 0.9 bit:", numpy.sum(df.values > .9))
    print("number of pairs with MI < 0.1 bit:", numpy.sum(df.values < 0.1))

    # get types and filter, depending on features / metrics
    if features_type == 'features':
        features_types = sorted(type_generator.get_feature_types(df.index))
        df = df.iloc[numpy.where(df.values > 1.)[0], :]

    elif features_type == 'metrics':
        features_types = features_names_cont
        # filter out calibration coefs, as they are not informative at all
        df = df.loc[:, numpy.array([name for name in tunes_names if name[-1] not in '34567'])]

    else:
        raise ValueError('Unknown feature type')

    pyplot.figure(figsize=(10,6))
    ax = pyplot.axes()
    res = seaborn.heatmap(df, xticklabels=df.columns, yticklabels=features_types, ax=ax)
    # res.set_yticklabels(res.get_ymajorticklabels(), fontsize=6)

    if features_type == 'features':
        ax.set_title("Mutual information: QC features vs machine settings")
    elif features_type == 'metrics':
        ax.set_title("Mutual information: QC indicators vs machine settings")
    else:
        raise ValueError('Unknown feature type')

    pyplot.tight_layout()
    # pyplot.show()
    pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/mi_{}_tunes_{}_with_types.pdf'.format(user, features_type, tunes_type))
    pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/mi_{}_tunes_{}_with_types.png'.format(user, features_type, tunes_type))

    if features_type == 'features':

        features_masses = type_generator.get_mass_types(df.index)
        # substitute tuples with the first element
        for i, mass_type in enumerate(features_masses):
            if isinstance(mass_type, tuple):
                features_masses[i] = mass_type[0]
        features_masses = sorted(features_masses)

        pyplot.figure(figsize=(10,6))
        ax = pyplot.axes()
        res = seaborn.heatmap(df, xticklabels=df.columns, yticklabels=features_masses, ax=ax)
        # res.set_yticklabels(res.get_ymajorticklabels(), fontsize=6)

        ax.set_title("Mutual information: QC features vs machine settings")
        pyplot.tight_layout()
        pyplot.show()
        # pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/mi_{}_tunes_{}_with_masses.pdf'.format(user, features_type, tunes_type))
        # pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/mi_{}_tunes_{}_with_masses.png'.format(user, features_type, tunes_type))


def calc_MI(x, y, bins=None):

    if bins:
        c_xy = numpy.histogram2d(x, y, bins)[0]
    else:
        c_xy = numpy.histogram2d(x, y)[0]

    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi


if __name__ == "__main__":

    if True:
        # GET DATA

        # # not updated with the latest db
        # condensed_features = pandas.read_csv("/Users/{}/ETH/projects/monitoring_system/res/nas2/sparse_pca_features_without_dmso_n=15.csv".format(user))
        # loadings = pandas.read_csv("/Users/{}/ETH/projects/monitoring_system/res/nas2/sparse_pca_loadings_without_dmso_n=15.csv".format(user))

        meta_info, features, colnames = get_features_data()
        features_cont, features_names_cont, features_cat, features_names_cat = split_features_to_cont_and_cat(features, numpy.array(colnames[4:]))

        full_meta_data = get_meta_data()

        tunes_cont, tunes_names_cont, tunes_cat, tunes_names_cat = get_tunes_and_names()

    if True:
        # FILTER OUT DMSO
        ipa_h20_indices = numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O')[0]

        features_cat = features_cat[ipa_h20_indices, :]
        features_cont = features_cont[ipa_h20_indices, :]
        tunes_cat = tunes_cat[ipa_h20_indices, :]
        tunes_cont = tunes_cont[ipa_h20_indices, :]

    if False:
        # CONDENSE AND SAVE FEATURES

        # from multiprocessing import Process
        # for n in [500, 1000]:
        #     p = Process(target=perform_sparse_pca, args=(True, n,))
        #     p.start()

        perform_sparse_pca(filter_dmso=True)

    if False:
        perform_pca(filter_dmso=False)

    if False:
        plot_pca_variance_explained(filter_dmso=True)

    if False:

        if True:
            # FIND INFORMATIVE FEATURES
            loading_sums = numpy.sum(numpy.abs(loadings.values), axis=0)

            # normalize
            loading_sums = loading_sums / numpy.max(loading_sums)

            print("number of features with 0 contribution:", numpy.where(loading_sums == 0)[0].shape[0])

            _, _, colnames = get_features_data()
            colnames = numpy.array(colnames[4:])

            highest_loadings = numpy.sort(loading_sums)[::-1][:100]

            for i in range(highest_loadings.shape[0]):
                print(colnames[numpy.where(loading_sums == highest_loadings[i])], ": ", highest_loadings[i], sep="")

            print("Normality test p =", scipy.stats.shapiro(loading_sums[numpy.where(loading_sums > 0)]).pvalue)

            seaborn.distplot(loading_sums[numpy.where(loading_sums >= 0)])
            pyplot.title("Sparse PCA features' loadings")
            pyplot.grid()
            pyplot.show()
            # pyplot.savefig("/Users/dmitrav/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/monitoring_system/img/fig1b.pdf")

        if False:
            # PLOT PC DISTRIBUTIONS
            fig = pyplot.figure()
            for i in range(10):
                ax = fig.add_subplot(2,5,i+1)
                ax.hist(loadings.iloc[:,i])
                ax.set_title('PC{}'.format(i+1))
                ax.grid()

            pyplot.tight_layout()
            pyplot.show()

    if False:
        # CLUSTERING FULL DATA NO FILTERING
        predicted_buffers = perform_global_clustering(condensed_features, title="All data")

        pairs = [pair for pair in zip(full_meta_data['buffer_id'], predicted_buffers)]
        unique_pairs = list(set(pairs))

        confusion_matrix = {}
        for pair in unique_pairs:
            confusion_matrix[pair] = pairs.count(pair)

        print(confusion_matrix)  # accuracy >= 0.98, only few false negatives (IPA_H2O assigned to DMSO by mistake)

    if False:
        # CLUSTERING FULL DATA WITH FILTERING

        dates = [date[:10] for date in full_meta_data['acquisition_date']]
        # find the best clustering for the whole dataset
        predictions = find_and_perform_best_clustering(dates, condensed_features, title='All data')

    if False:
        # CLUSTERING "IPA_H2O_DMSO"
        dmso_subset = condensed_features.iloc[numpy.where(full_meta_data.iloc[:, 15] == 'IPA_H2O_DMSO')[0], :]
        dates = [date[:10] for date in full_meta_data['acquisition_date'][numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O_DMSO')[0]]]

        predictions = find_and_perform_best_clustering(dates, dmso_subset, title="IPA_H2O_DMSO")

        print(predictions)

    if False:
        # K-MEANS TESTING
        perform_k_means(condensed_features)

    if False:
        # CLUSTERING "IPA_H2O"
        dates = [date[:10] for date in full_meta_data['acquisition_date'][ipa_h20_indices]]

        predictions = find_and_perform_best_clustering(dates, condensed_features, title="IPA_H2O")

        print(predictions)

    if False:
        # CROSS CORRELATIONS FEATURES: CLUSTER ENRICHMENTS WITH FEATURES TYPES

        df = pandas.DataFrame(features_cont)
        df = df.corr()
        df = df.fillna(0)

        n_clusters = 88  # the best is 88, the second best is 3

        predictions = perform_best_clustering(features_names_cont, df[:], n_clusters, title="QC features cross-correlations", no_labels=True)

        all_types = sorted(list(set(type_generator.get_feature_types(features_names_cont))))
        cluster_enrichments = pandas.DataFrame(columns=[x+1 for x in range(n_clusters)], index=all_types)
        cluster_enrichments = cluster_enrichments.fillna(0)

        medians = []
        groups = []
        for group in predictions:
            print("\nGroup", group)
            print('size:', len(predictions[group]['indices']))
            values = df.iloc[predictions[group]['indices'], predictions[group]['indices']].values.flatten()

            groups.append(group)
            medians.append(numpy.median(values))
            print("median =",  numpy.median(values))

            group_masses = type_generator.get_feature_types(predictions[group]['labels'])
            unique_masses = list(set(group_masses))
            counts = [group_masses.count(x) for x in unique_masses]

            for i, group_type in enumerate(unique_masses):
                # fill in the heatmap
                cluster_enrichments.loc[group_type, group] += counts[i]

        # sort columns by increasing cross-correlation median
        sorted_groups = numpy.array([groups[medians.index(x)] for x in sorted(medians)])
        cluster_enrichments = cluster_enrichments.loc[:, sorted_groups]

        print('\ntotal sum =', cluster_enrichments.sum().sum())  # debug

        pyplot.figure(figsize=(10,6))
        # seaborn.heatmap(cluster_enrichments, xticklabels=cluster_enrichments.columns, yticklabels=cluster_enrichments.index)
        seaborn.heatmap(cluster_enrichments, yticklabels=cluster_enrichments.index)
        pyplot.title("Cluster enrichments with types: features' cross-correlations")
        pyplot.tight_layout()
        pyplot.xticks(rotation=45)
        pyplot.show()
        # pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/corr_enrichments_types_{}.pdf'.format(user, n_clusters))
        # pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/corr_enrichments_types_{}.png'.format(user, n_clusters))

        # save medians as well
        medians_df = pandas.DataFrame({'cluster': [str(x) for x in sorted_groups], 'median_corr': sorted(medians)})
        pyplot.figure(figsize=(12,6))
        res = seaborn.barplot(x='cluster', y='median_corr', data=medians_df, order=medians_df['cluster'])
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=8)
        pyplot.xlabel('Cluster')
        pyplot.ylabel('Median correlation')
        pyplot.grid(True)
        pyplot.xticks(rotation=45)
        pyplot.tight_layout()
        # pyplot.show()
        pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/cluster_corr_medians_{}.png'.format(user, n_clusters))

    if False:
        # CROSS CORRELATIONS FEATURES: CLUSTER ENRICHMENTS WITH MASS TYPES

        df = pandas.DataFrame(features_cont)
        df = df.corr()
        df = df.fillna(0)

        n_clusters = 3  # the best is 88, the second best is 3

        predictions = perform_best_clustering(features_names_cont, df[:], n_clusters, title="QC features cross-correlations", no_labels=True)

        all_types = [x for x in type_generator.get_mass_types(features_names_cont) if not isinstance(x, tuple)]
        all_types = sorted(list(set(all_types)))

        cluster_enrichments = pandas.DataFrame(columns=[x+1 for x in range(n_clusters)], index=all_types)
        cluster_enrichments = cluster_enrichments.fillna(0)

        medians = []
        groups = []
        for group in predictions:
            print("\nGroup", group)
            print('size:', len(predictions[group]['indices']))
            values = df.iloc[predictions[group]['indices'], predictions[group]['indices']].values.flatten()

            groups.append(group)
            medians.append(numpy.median(values))
            print("median =",  numpy.median(values))

            group_masses = type_generator.get_mass_types(predictions[group]['labels'])
            unique_masses = list(set(group_masses))
            counts = [group_masses.count(x) for x in unique_masses]

            for i, group_type in enumerate(unique_masses):

                if isinstance(group_type, str):
                    # fill in the heatmap
                    cluster_enrichments.loc[group_type, group] += counts[i]
                elif isinstance(group_type, tuple):
                    # fill in the heatmap for two types with half count for each
                    cluster_enrichments.loc[group_type[0], group] += counts[i]
                    cluster_enrichments.loc[group_type[1], group] += counts[i]
                else:
                    raise ValueError('Unrecognized group type')

        # sort columns by increasing cross-correlation median
        sorted_groups = numpy.array([groups[medians.index(x)] for x in sorted(medians)])
        cluster_enrichments = cluster_enrichments.loc[:, sorted_groups]

        print('\ntotal sum =', cluster_enrichments.sum().sum())  # debug

        pyplot.figure(figsize=(10,6))
        # seaborn.heatmap(cluster_enrichments, xticklabels=cluster_enrichments.columns, yticklabels=cluster_enrichments.index)
        seaborn.heatmap(cluster_enrichments, yticklabels=cluster_enrichments.index)
        pyplot.title("Cluster enrichments with masses: features' cross-correlations")
        pyplot.tight_layout()
        # pyplot.xticks(rotation=45)
        # pyplot.show()
        pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/corr_enrichments_masses_{}.pdf'.format(user, n_clusters))
        pyplot.savefig('/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/corr_enrichments_masses_{}.png'.format(user, n_clusters))

    if False:
        # TRENDS FOR FEATURES = f(TUNES):
        # Ion Focus

        x = [x for x in range(tunes_cat.shape[0])][21:88][::2]
        y = tunes_cat[:, 18][21:88][::2]

        dates = numpy.sort(full_meta_data.iloc[ipa_h20_indices, :]['acquisition_date'].values)[21:88][::2]
        dates = [str(date)[:10] for date in dates]

        pyplot.plot(x, y, 'o-')
        pyplot.xticks(ticks=x, labels=dates, rotation='vertical')
        pyplot.title("Ion Focus")
        pyplot.grid()
        pyplot.tight_layout()
        pyplot.savefig("/Users/dmitrav/ETH/projects/monitoring_system/res/analysis/features_different_split_by_tunes/ion_focus_drop.pdf")
        # pyplot.show()

        first_group_indices = numpy.concatenate([[False for x in range(55)], [True for x in range(88 - 55)], [False for x in range(100 - 88)]])
        second_group_indices = numpy.concatenate([[False for x in range(21)], [True for x in range(55 - 21)], [False for x in range(100 - 55)]])

        comparisons = {
            "continuous": metrics_tunes_analysis.test_tunes_for_statistical_differences(
                features_cont, features_names_cont,
                first_group_indices, second_group_indices,
                ["group 1", "group 2"], "Ion Focus", level=0.01, tunes_type="continuous", inspection_mode=True)
        }

        print("Number of significantly different QC features:", comparisons['continuous'].iloc[3,:].sum())

    if False:
        # TRENDS FOR FEATURES = f(TUNES):
        # TOF Vac

        x = [x for x in range(tunes_cont.shape[0])][70:97]
        y = tunes_cont[:, 1][70:97]

        dates = numpy.sort(full_meta_data.iloc[ipa_h20_indices, :]['acquisition_date'].values)[70:97]
        dates = [str(date)[:10] for date in dates]

        pyplot.plot(x, y, 'o-')
        pyplot.xticks(ticks=x, labels=dates, rotation='vertical')
        pyplot.title("TOF Vac")
        pyplot.grid()
        pyplot.tight_layout()
        pyplot.savefig("/Users/dmitrav/ETH/projects/monitoring_system/res/analysis/features_different_split_by_tunes/tof_vac_drop.pdf")
        # pyplot.show()

        first_group_indices = numpy.concatenate([[False for x in range(70)], [True for x in range(88 - 70)], [False for x in range(100 - 88)]])
        second_group_indices = numpy.concatenate([[False for x in range(88)], [True for x in range(97 - 88)], [False for x in range(100 - 97)]])

        print()

        comparisons = {
            "continuous": metrics_tunes_analysis.test_tunes_for_statistical_differences(
                features_cont, features_names_cont,
                first_group_indices, second_group_indices,
                ["group 1", "group 2"], "TOF_Vac", level=0.05, tunes_type="continuous", inspection_mode=True)
        }

        print("Number of significantly different QC features:", comparisons['continuous'].iloc[3,:].sum())

    if False:
        # CROSS CORRELATIONS FEATURES

        df = pandas.DataFrame(features_cont)
        df = df.iloc[:, 800:1500]

        df = df.corr()

        """ some key observations:
            1) ~97% of pairs correlate with r < 0.5, 
            2) those pairs with r > 0.9 are related to "top 10 noisy peaks",
            3) those pairs with moderate correlation around 0.6 < r < 0.8 are related to frames features:
               (i.e. top peaks, num of peaks, int sums, percentiles, top percentiles...
               within frames 350-400 and 400-450, 950-1000 and 1000-1050, etc.),
            4) majority of continuous features don't cross-correlate much """

        # plot a heatmap
        seaborn.heatmap(df)
        pyplot.title("Cross-correlations: features")
        pyplot.tight_layout()
        pyplot.show()
        # pyplot.savefig("/Users/dmitrav/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/monitoring_system/img/fig2.pdf")

    if False:
        # CORRELATIONS FEATURES-TUNES

        metrics_tunes_analysis.assess_correlations_between_tunes_and_metrics(
            features_cont, features_names_cont, tunes_cont, tunes_names_cont, tunes_type='continuous', method="pearson",
            inspection_mode=False
        )

        # metrics_tunes_analysis.assess_correlations_between_tunes_and_metrics(
        #     features_cont, features_names_cont, tunes_cat, tunes_names_cat, tunes_type='categorical',
        #     inspection_mode=True
        # )

        # metrics_tunes_analysis.assess_correlations_between_tunes_and_metrics(
        #     features_cont, features_names_cont, tunes_cat, tunes_names_cat, tunes_type='continuous',
        #     inspection_mode=True
        # )

    if False:
        # MUTUAL INFO FEATURES-TUNES

        compute_mutual_info_between_tunes_and_features(
            features_cont, features_names_cont, tunes_cont, tunes_names_cont, tunes_type='cont', inspection_mode=False
        )

        compute_mutual_info_between_tunes_and_features(
            features_cont, features_names_cont, tunes_cat, tunes_names_cat, tunes_type='cat', inspection_mode=False
        )

    if False:
        # CLASSIFICATION
        random_seed = 905

        results = pandas.DataFrame(columns=['full size', '% resampled', 'method', 'val size', 'n classes', 'accuracy'], index=tunes_names_cat)

        for i in range(tunes_cat.shape[1]):

            le = LabelEncoder().fit(tunes_cat[:,i])
            classes = le.transform(tunes_cat[:,i])

            results.loc[tunes_names_cat[i], 'n classes'] = len(le.classes_)

            start = time.time()

            upsampling_successful = False

            # upsample positive class
            try:
                resampler = SMOTETomek(random_state=random_seed)
                X_resampled, y_resampled = resampler.fit_resample(condensed_features, classes)
                print("SMOTETomek resampling was done")
                results.loc[tunes_names_cat[i], 'method'] = 'SMOTE'
                upsampling_successful = True
            except ValueError:

                try:
                    resampler = RandomOverSampler(random_state=random_seed)
                    X_resampled, y_resampled = resampler.fit_resample(condensed_features, classes)
                    print("Random upsampling was done")
                    results.loc[tunes_names_cat[i], 'method'] = 'random'
                    upsampling_successful = True
                except ValueError:
                    print("upsampling failed")

            if upsampling_successful:
                # print("resampling for", tunes_names_cat[i], "took", round(time.time() - start) // 60 + 1, 'min')
                print("X before: ", condensed_features.shape[0], ', X after: ', X_resampled.shape[0], sep="")
                results.loc[tunes_names_cat[i], 'full size'] = X_resampled.shape[0]
                results.loc[tunes_names_cat[i], '% resampled'] = int(100 * (X_resampled.shape[0] / condensed_features.shape[0] - 1))

                X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, stratify=y_resampled, random_state=random_seed)

                results.loc[tunes_names_cat[i], 'val size'] = X_val.shape[0]

                scoring = {'roc_auc': make_scorer(roc_auc_score, average='weighted'),
                           'f1': make_scorer(f1_score, average='weighted'),
                           'accuracy': make_scorer(accuracy_score)}

                clf = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_seed),
                                   param_grid={'splitter': ['best', 'random'], 'criterion': ['gini', 'entropy']},
                                   scoring=scoring, refit='f1',
                                   cv=3, n_jobs=-1)

                start = time.time()

                y = label_binarize(y_train, classes=numpy.sort(numpy.unique(y_train)))

                clf.fit(X_train, y)
                # print("training for ", tunes_names_cat[i], ' took ', round(time.time() - start) // 60 + 1, ' min', sep="")

                y = label_binarize(y_val, classes=numpy.sort(numpy.unique(y_train)))

                val_score = clf.score(X_val, y)
                print(tunes_names_cat[i], ", validation set size: ", X_val.shape[0], sep="")
                print('val auc: ', round(clf.cv_results_['mean_test_roc_auc'].mean(), 3),
                      ', val f1: ', round(clf.cv_results_['mean_test_f1'].mean(), 3),
                      ', val accuracy: ', round(clf.cv_results_['mean_test_accuracy'].mean(), 3), sep="")
                print("feature importances:", clf.best_estimator_.feature_importances_.tolist())

                results.loc[tunes_names_cat[i], 'accuracy'] = round(clf.cv_results_['mean_test_accuracy'].mean(), 3)

                print("best params:", clf.best_params_, '\n')

                inspect_feature_contributions = True
                if inspect_feature_contributions:

                    feature_contributions = (clf.best_estimator_.feature_importances_ * loadings.T).T

                    loading_sums = numpy.sum(numpy.abs(feature_contributions.values), axis=0)

                    normalised_loadings = loading_sums / numpy.max(loading_sums)

                    print("number of features with 0 contribution:", numpy.where(normalised_loadings == 0)[0].shape[0])

                    _, _, colnames = get_features_data()
                    colnames = numpy.array(colnames[4:])

                    highest_loadings = numpy.sort(normalised_loadings)[::-1][:100]

                    for j in range(highest_loadings.shape[0]):
                        print(colnames[numpy.where(normalised_loadings == highest_loadings[j])], ": ", highest_loadings[j],
                              sep="")
                    print()

                    pyplot.figure()
                    seaborn.distplot(normalised_loadings[numpy.where(loading_sums > 0)])
                    pyplot.title("Features importances to predict {}".format(tunes_names_cat[i]))
                    pyplot.grid()
                    # pyplot.show()
                    pyplot.savefig("/Users/andreidm/ETH/projects/monitoring_system/res/analysis/feature_importances_in_tunes_prediction/{}.pdf".format(tunes_names_cat[i]))

                inspect_decision_tree = True
                if inspect_decision_tree:

                    import graphviz
                    dot_data = tree.export_graphviz(clf.best_estimator_, out_file=None,
                                                    feature_names=["PC{}".format(x) for x in range(1,16)],
                                                    class_names=numpy.array([str(x) for x in list(set(y_val))]),
                                                    filled=True, rounded=True,
                                                    special_characters=True)
                    graph = graphviz.Source(dot_data)

                    # TODO: I can parse and edit graph representation to add classes in multiclass cases:
                    #   - find nodes with entropy or gini == 0,
                    #   - convert value representation to class (where it's [0 N], index of this array is the class)
                    #   Easy, but haven't tried or tested.

                    graph.render('/Users/andreidm/ETH/projects/monitoring_system/res/analysis/decision_trees/{}.gv.pdf'.format(tunes_names_cat[i]), view=False)

            else:
                continue

        results.to_csv("/Users/andreidm/ETH/projects/monitoring_system/res/analysis/tunes_predictions_without_dmso_n=15.csv")
        print("predictions saved")

    if False:
        # REGRESSION: sucks (why?)
        # - tried condensed features, continuous features
        # - tried with/without feature selection
        random_seed = 905

        features_to_fit = ['Duration', 'TOF_Vac', 'Quad_Vac', 'Rough_Vac', 'Turbo1_Power', 'Turbo2_Power',
                           'defaultPos_traditional_0', 'defaultPos_traditional_1', 'defaultPos_polynomial_0', 'defaultPos_polynomial_1',
                           'defaultNeg_traditional_0', 'defaultNeg_traditional_1', 'defaultNeg_polynomial_0', 'defaultNeg_polynomial_1']

        for i in range(len(features_to_fit)):

            # seaborn.distplot(tunes_cont[:,i])
            # pyplot.title(tunes_names_cont[i])
            # pyplot.show()

            X_train, X_val, y_train, y_val = train_test_split(condensed_features, tunes_cont[:, tunes_names_cont.index(features_to_fit[i])], random_state=random_seed)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                # ('selector', SelectKBest(score_func=f_regression)),
                ('regressor', Lasso(random_state=random_seed))
            ])

            param_grid = {
                'regressor__alpha': numpy.concatenate((numpy.linspace(1e-3, 1, 100), numpy.linspace(1, 1000, 100))),
                'regressor__fit_intercept': [True, False]
                # 'selector__k': [10]
            }

            scoring = {'r2': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)}

            reg = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring=scoring, refit='mse',
                               cv=3, n_jobs=-1)

            start = time.time()

            reg.fit(X_train, y_train)
            print("training for ", tunes_names_cont[i], ' took ', round(time.time() - start) // 60 + 1, ' min', sep="")

            val_score = reg.score(X_val, y_val)
            print(tunes_names_cont[i], ", validation set size: ", X_val.shape[0], sep="")
            print('val r2: ', reg.cv_results_['mean_test_r2'].mean(),
                  ', val mse: ', reg.cv_results_['mean_test_mse'].mean(), sep="")

            print("best params:", reg.best_params_, '\n')

    if False:
        # T-SNE FEATURES
        random_seed = 905

        perplexities = [20]

        for i, perplexity in enumerate(perplexities):

            tsne = manifold.TSNE(n_components=2, init='random',
                                 random_state=random_seed, perplexity=perplexity)

            Y = tsne.fit_transform(features_cont)

            fig, ax = pyplot.subplots(figsize=(10, 5))
            ax.scatter(Y[:, 0], Y[:, 1], c='#A9A9A9', marker='o')
            # adds a title and axes labels
            ax.set_title("t-SNE, perplexity=%d" % perplexity)

            for i, txt in enumerate(meta_info[:,0]):
                ax.annotate(txt, xy=(Y[i, 0], Y[i, 1]), xytext=(3, 3), textcoords='offset points')

            pyplot.show()

    if False:

        random_seed = 905

        neighbors = [15, 15, 15]
        metrics = ['correlation', 'euclidean', 'cosine']
        min_dist = 0.3

        start = time.time()
        scaled_data = StandardScaler().fit_transform(features_cont)
        print('scaling took {} s'.format(time.time() - start))

        pyplot.subplots(nrows=1, ncols=3, figsize=(12, 4))
        seaborn.set(font_scale=.8)
        seaborn.color_palette('colorblind')
        seaborn.axes_style('whitegrid')

        # for annotation
        dates = []
        for date in full_meta_data.loc[ipa_h20_indices, 'acquisition_date'].values:
            if '' in date:
                dates.append(date[:10])
            else:
                dates.append('')

        for i, n in enumerate(neighbors):
            reducer = umap.UMAP(n_neighbors=n, metric=metrics[i], min_dist=min_dist, random_state=random_seed)
            start = time.time()
            embedding = reducer.fit_transform(scaled_data)
            print('umap transform with n = {} took {} s'.format(n, time.time() - start))

            pyplot.subplot(1, 3, i + 1)
            seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], alpha=0.8)
            pyplot.title('UMAP: n={}, metric={}'.format(n, metrics[i]), fontsize=12)

            # annotate points
            for i in range(len(dates)):
                pyplot.annotate(dates[i],  # this is the text
                                (embedding[i, 0], embedding[i, 1]),  # this is the point to label
                                textcoords="offset points",  # how to position the text
                                xytext=(0, 3),  # distance from text to points (x,y)
                                ha='center',  # horizontal alignment can be left, right or center
                                fontsize=4)

        # pyplot.show()
        pyplot.tight_layout()
        pyplot.savefig('/Users/andreidm/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/monitoring_system/img/umap/umap_features.pdf')

    if False:

        random_seed = 905

        neighbors = 15
        metric = 'cosine'
        min_dist = 0.1

        start = time.time()
        scaled_data = StandardScaler().fit_transform(features_cont)
        print('scaling took {} s'.format(time.time() - start))

        seaborn.set(font_scale=.8)
        seaborn.color_palette('colorblind')
        seaborn.axes_style('whitegrid')

        # for annotation
        dates = []
        for date in full_meta_data.loc[ipa_h20_indices, 'acquisition_date'].values:
            if '' in date:
                dates.append(date[:10])
            else:
                dates.append('')

        reducer = umap.UMAP(n_neighbors=neighbors, metric=metric, min_dist=min_dist, random_state=random_seed)
        start = time.time()
        embedding = reducer.fit_transform(scaled_data)
        print('umap transform with n = {} took {} s'.format(neighbors, time.time() - start))

        # tune_values = tunes_cat[:, tunes_names_cat.index('InstrumentFW')]
        # tune_names = ['InstrumentFW='+str(val) for val in tune_values]
        #
        # pyplot.subplot(121)
        # seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=tune_names, alpha=1)
        # pyplot.title('UMAP on QC features: n={}, metric={}'.format(neighbors, metric), fontsize=12)
        #
        # # annotate points
        # for i in range(len(dates)):
        #     pyplot.annotate(dates[i],  # this is the text
        #                     (embedding[i, 0], embedding[i, 1]),  # this is the point to label
        #                     textcoords="offset points",  # how to position the text
        #                     xytext=(0, 3),  # distance from text to points (x,y)
        #                     ha='center',  # horizontal alignment can be left, right or center
        #                     fontsize=4)

        tune_values = tunes_cat[:, tunes_names_cat.index('Mirror_Mid')]
        tune_names = ['value='+str(round(val, 1)) for val in tune_values]

        pyplot.figure(figsize=(8,6))
        seaborn.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=tune_names, alpha=1)
        pyplot.title('UMAP on QC features: n={}, metric={}'.format(neighbors, metric), fontsize=12)

        # annotate points
        for i in range(len(dates)):
            pyplot.annotate(dates[i],  # this is the text
                            (embedding[i, 0], embedding[i, 1]),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 3),  # distance from text to points (x,y)
                            ha='center',  # horizontal alignment can be left, right or center
                            fontsize=4)

        pyplot.legend(title='Mirror_Mid', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
        pyplot.tight_layout()
        pyplot.show()
        # pyplot.savefig('/Users/andreidm/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/monitoring_system/img/umap/umap_mirror_mid.pdf')