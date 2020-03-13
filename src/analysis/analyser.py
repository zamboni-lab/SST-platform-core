import numpy, pandas, matplotlib, scipy, seaborn, math
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from src.msfe import db_connector
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, kruskal
from collections import Counter
from statsmodels.stats.multitest import multipletests


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


def assess_cross_correlations(data_matrix, columns_names, type='continuous', method="", level=0.7):

    if type == 'continuous':

        # get correlation matrix for continuous
        if method in ['pearson', 'spearman']:
            df = pandas.DataFrame(data_matrix).corr(method=method)
        else:
            df = pandas.DataFrame(data_matrix).corr()

        # change names for better display
        for i in range(len(columns_names)):
            columns_names[i] = columns_names[i].replace("default", "").replace("traditional", "trad").replace(
                "polynomial", "poly")

        df.columns = columns_names

        # list cross correlated features
        correlated_groups = []

        for i in range(df.shape[0]):
            i_correlated_with = [i]
            for j in range(i + 1, df.shape[0]):
                if abs(df.iloc[i, j]) > level:
                    i_correlated_with.append(j)

            if len(i_correlated_with) > 1:
                # now check if this group is fully in another group
                is_part_of_another_group = False
                for group in correlated_groups:
                    # if all indices are founf inside any group
                    if sum([index in group for index in i_correlated_with]) == len(i_correlated_with):
                        is_part_of_another_group = True

                if not is_part_of_another_group:
                    correlated_groups.append(i_correlated_with)

        # get names of correlated tunes
        correlated_group_names = []
        for group in correlated_groups:
            tunes_names = [columns_names[index] for index in group]
            group_name = "-".join(tunes_names)
            correlated_group_names.append(group_name)

        print("Correlated groups:")
        print(correlated_group_names)

        # plot a heatmap
        seaborn.heatmap(df, xticklabels=df.columns, yticklabels=df.columns)
        plt.tight_layout()
        plt.show()

    elif type == 'categorical':

        # change names for better display
        for i in range(len(columns_names)):
            if len(columns_names[i]) > 12:
                # shorten too long names
                elements = columns_names[i].split("_")
                shortened_name = "_".join([element[:3] for element in elements])
                columns_names[i] = shortened_name

        # create empty correlation matrix for categorical tunes
        df = pandas.DataFrame(numpy.empty([len(columns_names), len(columns_names)]))
        df.columns = columns_names

        if method == 'cramers_v':
            # calculate Cramer's V correlation and fill the dataframe
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    df.iloc[i,j] = get_cramers_v_correlation(data_matrix[:,i], data_matrix[:,j])
        else:
            # calculate Theil's U correlation and fill the dataframe
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    df.iloc[i,j] = get_theils_u_correlation(data_matrix[:,i], data_matrix[:,j])

        print(df)

        # list cross correlated features
        correlated_groups = []

        for i in range(df.shape[0]):
            i_correlated_with = [i]
            for j in range(i + 1, df.shape[0]):
                if abs(df.iloc[i, j]) > level:
                    i_correlated_with.append(j)

            if len(i_correlated_with) > 1:
                # now check if this group is fully in another group
                is_part_of_another_group = False
                for group in correlated_groups:
                    # if all indices are founf inside any group
                    if sum([index in group for index in i_correlated_with]) == len(i_correlated_with):
                        is_part_of_another_group = True

                if not is_part_of_another_group:
                    correlated_groups.append(i_correlated_with)

        # get names of correlated tunes
        correlated_group_names = []
        for group in correlated_groups:
            tunes_names = [columns_names[index] for index in group]
            group_name = "-".join(tunes_names)
            correlated_group_names.append(group_name)

        print("Correlated groups:")
        print(correlated_group_names)

        # plot a heatmap
        seaborn.heatmap(df, xticklabels=df.columns, yticklabels=df.columns)
        plt.tight_layout()
        plt.show()

    else:
        pass


def get_cramers_v_correlation(x, y):
    """ Adapted from: https://github.com/shakedzy/dython/blob/master/dython/nominal.py

        This method calculates Cramer's V statistic for categorical-categorical association.
        Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013): 323-328.
        This is a symmetric coefficient: V(x,y) = V(y,x)
        Original function taken from: https://stackoverflow.com/a/46498792/5863503

        **Returns:** float in the range of [0,1] """

    confusion_matrix = pandas.crosstab(x,y)
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)

    return numpy.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def get_conditional_entropy(x,y):
    """ Adapted from: https://github.com/shakedzy/dython/blob/master/dython/nominal.py
        This method calculates the conditional entropy of x given y: S(x|y).
        It's used to calculate Theil's u correlation.

        **Returns:** float """

    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0

    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)

    return entropy


def get_theils_u_correlation(x,y):
    """ Adapted from: https://github.com/shakedzy/dython/blob/master/dython/nominal.py

        This method calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
        This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information
        about x, and 1 means y provides full information about x.
        This is an asymmetric coefficient: U(x,y) != U(y,x)

        **Returns:** float in the range of [0,1] """

    s_xy = get_conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = scipy.stats.entropy(p_x)

    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def get_correlation_ratio(categories, measurements):
    """ Adapted from: https://github.com/shakedzy/dython/blob/master/dython/nominal.py

        Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
        Answers the question - given a continuous value of a measurement, is it possible to know which category is it associated with?

        Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement,
        and 1 means a category can be determined with absolute certainty.

        **Returns:** float in the range of [0,1] """

    fcat, _ = pandas.factorize(categories)
    cat_num = numpy.max(fcat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)

    for i in range(0, cat_num):
        cat_measures = measurements[numpy.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)

    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(numpy.multiply(n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)))
    denominator = numpy.sum(numpy.power(numpy.subtract(measurements, y_total_avg), 2))

    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)

    return eta


def assess_correlations_between_tunes_and_metrics(metrics, metrics_names, tunes, tunes_names, tunes_type="continuous", method=""):
    """ This method calculates correlations between tunes and metrics. """

    if tunes_type == "continuous":

        # create dataframe to store correlations
        df = pandas.DataFrame(numpy.empty([len(metrics_names), len(tunes_names)]), index=metrics_names)

        # change names for better display
        for i in range(len(tunes_names)):
            tunes_names[i] = tunes_names[i].replace("default", "").replace("traditional", "trad").replace("polynomial", "poly")

        df.columns = tunes_names

        if method == "pearson":
            # calculate correlations and fill the dataframe
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    df.iloc[i, j] = scipy.stats.pearsonr(metrics[:, i], tunes[:, j])[0]
        else:
            # use spearman correlation coefficient
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    df.iloc[i, j] = scipy.stats.spearmanr(metrics[:, i], tunes[:, j])[0]

        print(df)

        # plot a heatmap
        seaborn.heatmap(df, xticklabels=df.columns, yticklabels=df.index)
        plt.tight_layout()
        plt.show()

    else:
        # tunes are categorical here

        # create dataframe to store correlations
        df = pandas.DataFrame(numpy.empty([len(metrics_names), len(tunes_names)]), index=metrics_names)

        # change names for better display
        for i in range(len(tunes_names)):
            if len(tunes_names[i]) > 12:
                # shorten too long names
                elements = tunes_names[i].split("_")
                shortened_name = "_".join([element[:3] for element in elements])
                tunes_names[i] = shortened_name

        df.columns = tunes_names

        # calculate correlations and fill the dataframe
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                df.iloc[i, j] = get_correlation_ratio(tunes[:, j], metrics[:, i])

        print(df)

        # plot a heatmap
        seaborn.heatmap(df, xticklabels=df.columns, yticklabels=df.index)
        plt.tight_layout()
        plt.show()


def test_tunes_for_statistical_differences(tunes, tunes_names, good_quality, bad_quality, tunes_type="continuous", level=0.05):
    """ This method conducts testing of hypothesis that
        tunes corresponding to "good" and "bad" runs differ statistically. """

    if tunes_type == "continuous":

        # prepare dataframe for statistical tests results
        df = pandas.DataFrame(numpy.empty([3, continuous_tunes.shape[1]]), index=["kolmogorov", "wilcoxon", "kruskall"])
        df.columns = tunes_names

        for i in range(tunes.shape[1]):
            # test continuous tunes of "good" and "bad" runs
            p1 = ks_2samp(tunes[good_quality, i], tunes[bad_quality, i])[1]
            p2 = mannwhitneyu(tunes[good_quality, i], tunes[bad_quality, i])[1]
            p3 = kruskal(tunes[good_quality, i], tunes[bad_quality, i])[1]

            df.iloc[:, i] = numpy.array([p1, p2, p3])

        for i in range(df.shape[0]):
            # correct p values for multiple testing
            df.iloc[i,:] = multipletests(df.iloc[i,:], method="fdr_bh")[1]

        # just add a row with True / False
        boolean_result = pandas.DataFrame(numpy.array([False for i in range(df.shape[1])])).T
        boolean_result.index = ["different"]
        boolean_result.columns = tunes_names

        for i in range(df.shape[1]):
            if sum(df.iloc[:,i] <= level) >= 2:
                boolean_result.iloc[0,i] = True

        return pandas.concat([df, boolean_result], axis=0)


def get_contingency_table(tunes, index, good_quality, bad_quality):
    """ This method create a continjency table for chi2 testing. """

    all_possible_values = set(tunes[:, index])

    good_values = list(tunes[good_quality, index])
    bad_values = list(tunes[bad_quality, index])





    pass



if __name__ == "__main__":

    qc_tunes_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_tunes_database.sqlite"

    conn = db_connector.create_connection(qc_tunes_database_path)
    database, colnames = db_connector.fetch_table(conn, "qc_tunes")

    tunes = numpy.array(database)

    # extract numeric values only
    indices = [10, 13, 16]
    indices.extend([i for i in range(18,151)])

    # compose arrays
    tunes = numpy.vstack([tunes[:, i].astype(numpy.float) for i in indices]).T
    colnames = numpy.array(colnames)[indices]

    # remove nans
    tunes = numpy.delete(tunes, range(89, 116), 1)  # these columns contain only zeros and nans
    colnames = numpy.delete(colnames, range(89, 116), 0)

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
        # let 12 be a max number of values for a tune to be categorical
        if len(set(informative_tunes[:, i])) > 12:
            continuous_tunes.append(informative_tunes[:, i])
            continuous_names.append(informative_colnames[i])
        else:
            categorical_tunes.append(informative_tunes[:, i])
            categorical_names.append(informative_colnames[i])

    continuous_tunes = numpy.array(continuous_tunes).T
    categorical_tunes = numpy.array(categorical_tunes).T

    if False:
        # check cross-correlations in tunes
        assess_cross_correlations(continuous_tunes, continuous_names, type='continuous', level=0.65)
        assess_cross_correlations(categorical_tunes, categorical_names, type='categorical', level=0.65)

    # read qc metrics
    qc_metrics_database_path = "/Users/andreidm/ETH/projects/monitoring_system/res/nas2/qc_metrics_database.sqlite"

    conn = db_connector.create_connection(qc_metrics_database_path)
    database, colnames = db_connector.fetch_table(conn, "qc_metrics")

    metrics = numpy.array(database)

    good_quality = metrics[:, 3] == '1'
    bad_quality = metrics[:, 3] == '0'

    # remove meta info columns
    metrics = numpy.delete(metrics, range(4), 1)
    metrics_names = colnames[4:]

    # convert to float
    metrics = metrics.astype(numpy.float)

    if False:
        # explore general correlations between tunes and metrics
        assess_correlations_between_tunes_and_metrics(metrics, metrics_names, continuous_tunes, continuous_names,
                                                      tunes_type='continuous', method="spearman")
        assess_correlations_between_tunes_and_metrics(metrics, metrics_names, categorical_tunes, categorical_names,
                                                  tunes_type='categorical')

    if False:
        # test tunes grouped by quality
        testing_results = test_tunes_for_statistical_differences(continuous_tunes, continuous_names, good_quality, bad_quality, tunes_type="continuous")


    for i in range(categorical_tunes.shape[1]):

        categorical_tunes[good_quality, i]
        categorical_tunes[bad_quality, i]



    pass


