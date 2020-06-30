
import numpy, pandas, scipy, seaborn, math, time
from sklearn.decomposition import SparsePCA
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
    database_1, colnames_1 = db_connector.fetch_table(conn, "qc_features_1")
    database_2, colnames_2 = db_connector.fetch_table(conn, "qc_features_2")

    features_1 = numpy.array(database_1)
    features_2 = numpy.array(database_2)

    meta = features_1[:, :4]
    features = numpy.hstack([features_1[:, 4:].astype(numpy.float), features_2[:, 4:].astype(numpy.float)])
    colnames = [*colnames_1, *colnames_2[4:]]

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
        It means, that MI is bound to 2.71 at max. """

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
            if inspection_mode and mi > 1:
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

    print("number of pairs with MI > 1:", numpy.sum(df.values > 1))

    # plot a heatmap
    seaborn.heatmap(df, xticklabels=df.columns, yticklabels=False)
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":

    if False:
        # condense features and save features
        perform_sparse_pca()

    condensed_features = pandas.read_csv("/Users/andreidm/ETH/projects/monitoring_system/res/nas2/sparse_pca_features.csv")

    meta_info, features, colnames = get_features_data()
    features_cont, features_names_cont, features_cat, features_names_cat = split_features_to_cont_and_cat(features, numpy.array(colnames[4:]))

    full_meta_data = get_meta_data()

    tunes_cont, tunes_names_cont, tunes_cat, tunes_names_cat = get_tunes_and_names()

    if False:
        # testing of k-means
        perform_k_means(condensed_features)

    if False:
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

    if False:

        metrics_tunes_analysis.assess_correlations_between_tunes_and_metrics(
            features_cont, features_names_cont, tunes_cont, tunes_names_cont, tunes_type='continuous', method="pearson",
            inspection_mode=False
        )

        metrics_tunes_analysis.assess_correlations_between_tunes_and_metrics(
            features_cont, features_names_cont, tunes_cat, tunes_names_cat, tunes_type='categorical',
            inspection_mode=True
        )

    if False:

        compute_mutual_info_between_tunes_and_features(
            features_cont, features_names_cont, tunes_cont, tunes_names_cont, inspection_mode=False
        )

        # compute_mutual_info_between_tunes_and_features(
        #     features_cont, features_names_cont, tunes_cat, tunes_names_cat, inspection_mode=False
        # )

    if False:

        random_seed = 905

        # CLASSIFICATION
        for i in range(tunes_cat.shape[1]):

            le = LabelEncoder().fit(tunes_cat[:,i])
            classes = le.transform(tunes_cat[:,i])

            start = time.time()
            # upsample positive class
            try:
                resampler = SMOTETomek(random_state=random_seed)
                X_resampled, y_resampled = resampler.fit_resample(condensed_features, classes)
                print("SMOTETomek resampling was done")
            except ValueError:
                resampler = RandomOverSampler(random_state=random_seed)
                X_resampled, y_resampled = resampler.fit_resample(condensed_features, classes)
                print("Random upsampling was done")

            # print("resampling for", tunes_names_cat[i], "took", round(time.time() - start) // 60 + 1, 'min')
            print("X before: ", condensed_features.shape[0], ', X after: ', X_resampled.shape[0], sep="")

            X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, stratify=y_resampled, random_state=random_seed)

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

            print("best params:", clf.best_params_, '\n')

    if False:

        random_seed = 905

        features_to_fit = ['Duration', 'TOF_Vac', 'Quad_Vac', 'Rough_Vac', 'Turbo1_Power', 'Turbo2_Power',
                           'defaultPos_traditional_0', 'defaultPos_traditional_1', 'defaultPos_polynomial_0', 'defaultPos_polynomial_1',
                           'defaultNeg_traditional_0', 'defaultNeg_traditional_1', 'defaultNeg_polynomial_0', 'defaultNeg_polynomial_1']

        # REGRESSION
        for i in range(len(features_to_fit)):

            # seaborn.distplot(tunes_cont[:,i])
            # pyplot.title(tunes_names_cont[i])
            # pyplot.show()

            X_train, X_val, y_train, y_val = train_test_split(features_cont, tunes_cont[:, tunes_names_cont.index(features_to_fit[i])], random_state=random_seed)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(score_func=f_regression)),
                ('regressor', Ridge(random_state=random_seed))
            ])

            param_grid = {
                'regressor__alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                'regressor__fit_intercept': [True, False],
                'selector__k': range(2,11)
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

    if True:

        pass