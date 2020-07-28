
import numpy, pandas, scipy, seaborn, math, time
from sklearn.decomposition import SparsePCA, PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize, MinMaxScaler
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
from src.analysis import metrics_tunes_analysis
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.tree import DecisionTreeClassifier
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

        P_hat = transformer.components_.T
        T_hat = numpy.dot(numpy.dot(scaled_features, P_hat), numpy.linalg.pinv(numpy.dot(P_hat.T, P_hat)))
        E = scaled_features - numpy.dot(T_hat, P_hat.T)

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


def perform_sparse_pca(filter_dmso=True):
    """ This method performs sparse PCA on the qc_features_database (local file of some version provided),
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

    number_of_components = 15  # try also 30

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

    pandas.DataFrame(transformer.components_).to_csv('/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/sparse_pca_loadings_{}.csv'.format(file_mark), index=False)
    pandas.DataFrame(features_transformed).to_csv('/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/sparse_pca_features_{}.csv'.format(file_mark), index=False)


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

    number_of_components = 2850

    transformer = PCA(n_components=10)
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


def get_features_data(path="/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"):
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


def get_meta_data(path="/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/qc_features_database.sqlite"):
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


def get_tunes_and_names(path="/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/qc_tunes_database.sqlite"):
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


def find_and_perform_best_clustering(samples_labels, features, title=""):
    """ More special clustering:
        1) finding the best score,
        2) filtering out single classes,
        3) finding the best score again """

    Z = linkage(features, 'ward')

    results = {'t': [], 'score': [], 'labels': []}

    for t in range(2, 30):
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
    features = features.drop(features.index[qc_outliers_indices])
    print("dropped:", qc_outliers_indices)

    # assign index for dendrogram plotting
    samples_labels = features['label']
    features = features.drop(columns=['label'])

    # recompute Z
    Z = linkage(features, 'ward')

    results = {'t': [], 'score': []}

    # find best score again
    for t in range(2, 30):
        clusters = fcluster(Z, t=t, criterion='maxclust')
        score = silhouette_score(features, clusters)
        # print("t=", t, "score=", score)

        results['t'].append(t)
        results['score'].append(score)

    max_score = max(results['score'])
    best_t = results['t'][results['score'].index(max_score)]

    clusters = fcluster(Z, t=best_t, criterion='maxclust')
    score = round(silhouette_score(features, clusters), 3)
    print("best score after filtering =", score)
    print("t =", best_t)

    samples_indices = features.index
    features.index = samples_labels
    dn = hierarchy.dendrogram(Z, labels=features.index, leaf_rotation=90)
    pyplot.title(title + ", t={}".format(best_t) + ", score={}".format(score))
    pyplot.tight_layout()
    pyplot.show()

    cluster_groups = {}
    for i in range(len(samples_indices)):
        if clusters[i] not in cluster_groups.keys():
            cluster_groups[clusters[i]] = [samples_indices[i]]
        else:
            cluster_groups[clusters[i]].append(samples_indices[i])

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
    return mi


def compute_mutual_info_between_tunes_and_features(features_cont, features_names_cont, tunes_cont, tunes_names_cont, inspection_mode=False):
    """ This method calculates mutual information between tunes and features. """

    # create dataframe to store correlations
    df = pandas.DataFrame(numpy.empty([len(features_names_cont), len(tunes_names_cont)]), index=features_names_cont)

    # change names for better display
    for i in range(len(tunes_names_cont)):
        tunes_names_cont[i] = tunes_names_cont[i].replace("default", "").replace("traditional", "trad").replace("polynomial", "poly")

    df.columns = tunes_names_cont

    # calculate correlations and fill the dataframe
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):

            mi = calculate_MI(features_cont[:, i], tunes_cont[:, j])
            df.iloc[i, j] = mi

            # look into variables closer if MI > some value
            if inspection_mode and mi > 0.9:
                fig, ax = pyplot.subplots(figsize=(10, 5))

                ax.scatter(tunes_cont[:, j], features_cont[:, i])

                # adds a title and axes labels
                ax.set_title(df.index[i] + ' vs ' + df.columns[j] + ": MI = " + str(mi))
                ax.set_xlabel(df.columns[j])
                ax.set_ylabel(df.index[i])

                # removing top and right borders
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                pyplot.show()

    print("number of pairs with MI > 0.9:", numpy.sum(df.values > 0.9))

    # plot a heatmap
    seaborn.heatmap(df, xticklabels=df.columns, yticklabels=False)
    pyplot.tight_layout()
    pyplot.show()


from scipy.stats import chi2_contingency


def calc_MI(x, y, bins=None):

    if bins:
        c_xy = numpy.histogram2d(x, y, bins)[0]
    else:
        c_xy = numpy.histogram2d(x, y)[0]

    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi


if __name__ == "__main__":

    if False:
        # GET DATA
        condensed_features = pandas.read_csv("/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/sparse_pca_features_with_dmso.csv")

        meta_info, features, colnames = get_features_data()
        features_cont, features_names_cont, features_cat, features_names_cat = split_features_to_cont_and_cat(features, numpy.array(colnames[4:]))

        full_meta_data = get_meta_data()

        tunes_cont, tunes_names_cont, tunes_cat, tunes_names_cat = get_tunes_and_names()

    if False:
        # FILTER OUT DMSO
        ipa_h20_indices = numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O')[0]

        condensed_features = condensed_features.iloc[ipa_h20_indices, :]
        features_cat = features_cat[ipa_h20_indices, :]
        features_cont = features_cont[ipa_h20_indices, :]
        tunes_cat = tunes_cat[ipa_h20_indices, :]
        tunes_cont = tunes_cont[ipa_h20_indices, :]

    if True:
        # CONDENSE AND SAVE FEATURES
        perform_sparse_pca(filter_dmso=False)

    if False:
        perform_pca(filter_dmso=False)

    if False:
        # PCA LOADINGS ANALYSIS
        loadings = pandas.read_csv("/Users/dmitrav/ETH/projects/monitoring_system/res/nas2/sparse_pca_loadings_without_dmso.csv")

        if True:
            # FIND INFORMATIVE FEATURES
            loading_sums = numpy.sum(numpy.abs(loadings.values), axis=0)

            print("number of features with 0 contribution:", numpy.where(loading_sums == 0)[0].shape[0])

            _, _, colnames = get_features_data()
            colnames = numpy.array(colnames[4:])

            highest_loadings = numpy.sort(loading_sums)[::-1][:100]

            for i in range(highest_loadings.shape[0]):
                print(colnames[numpy.where(loading_sums == highest_loadings[i])], ": ", highest_loadings[i], sep="")

            seaborn.distplot(loading_sums[numpy.where(loading_sums > 0)])
            pyplot.title("Total features' contributions")
            pyplot.grid()
            pyplot.show()

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

        print(confusion_matrix)  # accuracy = 0.98, only 2 false negatives (IPA_H2O assigned to DMSO by mistake)

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
        # CROSS CORRELATIONS FEATURES
        df = pandas.DataFrame(features_cont).corr()

        """ some key observations:
            1) ~97% of pairs correlate with r < 0.5, 
            2) those pairs with r > 0.9 are related to "top 10 noisy peaks",
            3) those pairs with moderate correlation around 0.6 < r < 0.8 are related to frames features:
               (i.e. top peaks, num of peaks, int sums, percentiles, top percentiles...
               within frames 350-400 and 400-450, 950-1000 and 1000-1050, etc.),
            4) majority of continuous features don't cross-correlate much """

        # plot a heatmap
        seaborn.heatmap(df)
        pyplot.tight_layout()
        pyplot.show()

    if False:
        # CORRELATIONS FEATURES-TUNES

        metrics_tunes_analysis.assess_correlations_between_tunes_and_metrics(
            features_cont, features_names_cont, tunes_cont, tunes_names_cont, tunes_type='continuous', method="pearson",
            inspection_mode=False
        )

        metrics_tunes_analysis.assess_correlations_between_tunes_and_metrics(
            features_cont, features_names_cont, tunes_cat, tunes_names_cat, tunes_type='categorical',
            inspection_mode=False
        )

    if False:
        # MUTUAL INFO FEATURES-TUNES

        # compute_mutual_info_between_tunes_and_features(
        #     features_cont, features_names_cont, tunes_cont, tunes_names_cont, inspection_mode=True
        # )

        compute_mutual_info_between_tunes_and_features(
            features_cont, features_names_cont, tunes_cat, tunes_names_cat, inspection_mode=True
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

            else:
                continue

        results.to_csv("/Users/dmitrav/ETH/projects/monitoring_system/res/analysis/tunes_predictions_without_dmso.csv")
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
                ('scaler', MinMaxScaler()),
                ('selector', SelectKBest(score_func=f_regression)),
                ('regressor', Ridge(random_state=random_seed))
            ])

            param_grid = {
                'regressor__alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                'regressor__fit_intercept': [True, False],
                'selector__k': [10]
            }

            scoring = {'r2': make_scorer(r2_score), 'mse': make_scorer(mean_squared_error)}

            reg = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring=scoring, refit='mse',
                               cv=5, n_jobs=-1)

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