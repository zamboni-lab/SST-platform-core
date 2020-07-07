import numpy, pandas, scipy, seaborn, math
from sklearn.decomposition import SparsePCA
from src.qcmg import db_connector
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, kruskal
from scipy.stats import chi2_contingency
from collections import Counter
from statsmodels.stats.multitest import multipletests


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


def assess_correlations_between_tunes_and_metrics(metrics, metrics_names, tunes, tunes_names, tunes_type="continuous", method="", inspection_mode=False):
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

                    correlation = scipy.stats.pearsonr(metrics[:, i], tunes[:, j])[0]
                    df.iloc[i, j] = correlation

                    # look into variables closer if correlation is high
                    if inspection_mode and abs(correlation) > 0.6:
                        fig, ax = plt.subplots(figsize=(10, 5))

                        ax.scatter(tunes[:, j], metrics[:, i])

                        # adds a title and axes labels
                        ax.set_title(df.index[i] + ' vs ' + df.columns[j] + ": r = " + str(correlation))
                        ax.set_xlabel(df.columns[j])
                        ax.set_ylabel(df.index[i])

                        # removing top and right borders
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        plt.show()

        else:
            # use spearman correlation coefficient
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):

                    correlation = scipy.stats.spearmanr(metrics[:, i], tunes[:, j])[0]
                    df.iloc[i, j] = correlation

                    # look into variables closer if correlation is high
                    if inspection_mode and abs(correlation) > 0.6:
                        fig, ax = plt.subplots(figsize=(10, 5))

                        ax.scatter(tunes[:, j], metrics[:, i])

                        # adds a title and axes labels
                        ax.set_title(df.index[i] + ' vs ' + df.columns[j] + ": r = " + str(correlation))
                        ax.set_xlabel(df.columns[j])
                        ax.set_ylabel(df.index[i])

                        # removing top and right borders
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        plt.show()

        print("number of pairs with correlation > 0.7:", numpy.sum(numpy.abs(df.values) > 0.7))

        # plot a heatmap
        seaborn.heatmap(df, xticklabels=df.columns, yticklabels=False)
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

                correlation_ratio = get_correlation_ratio(tunes[:, j], metrics[:, i])

                # look into variables closer if correlation is high
                if inspection_mode and correlation_ratio > 0.9:

                    fig, ax = plt.subplots(figsize=(10, 5))

                    ax.scatter(tunes[:, j], metrics[:, i])

                    # adds a title and axes labels
                    ax.set_title(df.index[i] + ' vs ' + df.columns[j] + ": eta = " + str(correlation_ratio))
                    ax.set_xlabel(df.columns[j])
                    ax.set_ylabel(df.index[i])

                    # removing top and right borders
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.show()

                df.iloc[i, j] = correlation_ratio

        print("number of pairs with correlation ratio > 0.9:", numpy.sum(numpy.abs(df.values) > 0.9))

        # plot a heatmap
        seaborn.heatmap(df, xticklabels=df.columns, yticklabels=False)
        plt.tight_layout()
        plt.show()


def plot_tunes_values_by_groups(tunes, index, group_1_indices, group_2_indices, group_names, metric_split_by, testing_result):
    """ This method visualises samples of tunes that were statistically different.
        It makes scatter plots for two groups of values. """

    fig, ax = plt.subplots(figsize=(8, 5))

    groups = numpy.empty(tunes.shape[0]).astype(str)
    groups[group_1_indices] = group_names[0] + " (N = " + str(numpy.sum(group_1_indices)) + ")"
    groups[group_2_indices] = group_names[1] + " (N = " + str(numpy.sum(group_2_indices)) + ")"

    # add formatted title
    plt.title(r"$\bf{" + testing_result.columns[index].replace("_", "\_") + "}$" + ", split by " + r"$\bf{" + metric_split_by.replace("_", "\_") + "}$")

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    data = pandas.DataFrame({"groups": groups[group_1_indices + group_2_indices],
                             testing_result.columns[index]: tunes[group_1_indices + group_2_indices, index]})

    seaborn.stripplot(x="groups", y=testing_result.columns[index], data=data)
    plt.grid()
    plt.show()


def test_tunes_for_statistical_differences(tunes, tunes_names, group_1_indices, group_2_indices, group_names, metric_split_by, tunes_type="continuous", level=0.05, inspection_mode=False):
    """ This method conducts testing of hypothesis that
        tunes corresponding to "good" and "bad" runs differ statistically. """

    if tunes_type == "continuous":

        # prepare dataframe for statistical tests results
        df = pandas.DataFrame(numpy.empty([3, tunes.shape[1]]), index=["kolmogorov", "wilcoxon", "kruskall"])
        df.columns = tunes_names

        for i in range(tunes.shape[1]):
            # test continuous tunes of group 1 and group 2 runs
            p1 = ks_2samp(tunes[group_1_indices, i], tunes[group_2_indices, i])[1]
            p2 = mannwhitneyu(tunes[group_1_indices, i], tunes[group_2_indices, i])[1]
            p3 = kruskal(tunes[group_1_indices, i], tunes[group_2_indices, i])[1]

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

                # look into variables closer if they are statistically different
                if inspection_mode:
                    plot_tunes_values_by_groups(tunes, i, group_1_indices, group_2_indices, group_names, metric_split_by, boolean_result)

        return pandas.concat([df, boolean_result], axis=0)

    elif tunes_type == 'categorical':

        df = pandas.DataFrame(numpy.empty([1, tunes.shape[1]]), index=["chi2"])
        df.columns = tunes_names

        # now go over categorical tunes and perform chi2
        for i in range(tunes.shape[1]):
            continjency_table = get_contingency_table(tunes, i, group_1_indices, group_2_indices)
            print(i, tunes_names[i])
            print(continjency_table)
            df.iloc[0, i] = chi2_contingency(continjency_table)[1]

        # correct p values for multiple testing
        df.iloc[0, :] = multipletests(df.iloc[0, :], method="fdr_bh")[1]

        # just add a row with True / False
        boolean_result = pandas.DataFrame(numpy.array([False for i in range(df.shape[1])])).T
        boolean_result.index = ["different"]
        boolean_result.columns = tunes_names

        for i in range(df.shape[1]):
            if round(df.iloc[0, i], 2) <= level:
                boolean_result.iloc[0, i] = True

                # look into variables closer if they are statistically different
                if inspection_mode:
                    plot_tunes_values_by_groups(tunes, i, group_1_indices, group_2_indices, group_names, metric_split_by, boolean_result)

        return pandas.concat([df, boolean_result], axis=0)

    else:
        raise ValueError("Tunes type not specified.")


def get_contingency_table(tunes, index, group_1_indices, group_2_indices):
    """ This method create a continjency table for chi2 testing. """

    all_possible_values = sorted(list(set(tunes[:, index])))

    good_values = list(tunes[group_1_indices, index])
    bad_values = list(tunes[group_2_indices, index])

    good_values_occurencies = [good_values.count(value) for value in all_possible_values]
    bad_values_occurencies = [bad_values.count(value) for value in all_possible_values]

    # remove 0/0 cases causing errors in chi2
    i = 0
    while i < len(good_values_occurencies):
        if good_values_occurencies[i] == bad_values_occurencies[i] == 0:
            good_values_occurencies.pop(i)
            bad_values_occurencies.pop(i)
        else:
            i += 1

    return numpy.array([good_values_occurencies, bad_values_occurencies])


def get_tunes_and_names(path):
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


def get_metrics_data(path):
    """ This method read metrics database,
        returns a matrix with metrics, metrics names, arrays of quality and acquisitions dates. """

    # read qc metrics
    conn = db_connector.create_connection(path)
    database, colnames = db_connector.fetch_table(conn, "qc_metrics")

    metrics = numpy.array(database)
    quality = metrics[:, 3]
    acquisition = metrics[:, 2]

    # remove meta info columns
    metrics = numpy.delete(metrics, range(4), 1)
    metrics_names = colnames[4:]

    # convert to float
    metrics = metrics.astype(numpy.float)

    return metrics, metrics_names, acquisition, quality


def test_tunes_grouped_by_extreme_metrics_values(metrics, quality, acquisition, continuous_tunes, continuous_names, categorical_tunes, categorical_names, inspection_mode=False):
    """ This method groups tunes based on extreme values (outliers) of metrics,
        then performs statistical tests and saves the results in the dict.  """

    all_comparisons = {}  # store results in a dict

    high_score_indices_before_march = (quality == '1') * (acquisition < "2020-03-04")

    for metric in ["resolution_200", "resolution_700", "signal", "s2b", "s2n"]:
        index = metrics_names.index(metric)

        # filter out negative values to estimate lower bound
        non_negative_indices = metrics[:, index] > 0  # missing values are -1s
        values = metrics[non_negative_indices, index]

        # very low values are bad here
        lower_bound = numpy.percentile(values, 25)

        # assign indices
        normal_values_indices = non_negative_indices * (metrics[:, index] >= lower_bound)
        low_extreme_indices = non_negative_indices * (metrics[:, index] < lower_bound)

        group_a_indices = high_score_indices_before_march * normal_values_indices
        group_b_indices = high_score_indices_before_march * low_extreme_indices

        # test tunes grouped by
        all_comparisons[metric] = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_a_indices, group_b_indices, ["normal values", "low extremes"], metric,
                                                                 tunes_type="continuous", inspection_mode=inspection_mode),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_b_indices, ["normal values", "low extremes"], metric,
                                                                  tunes_type="categorical", inspection_mode=inspection_mode)
        }

    for metric in ["average_accuracy", "chemical_dirt", "instrument_noise", "baseline_25_150", "baseline_50_150",
                   "baseline_25_650", "baseline_50_650"]:
        index = metrics_names.index(metric)

        # filter out negative values to estimate lower bound
        non_negative_indices = metrics[:, index] > 0  # missing values are -1s
        values = metrics[non_negative_indices, index]

        # very high values are bad here
        upper_bound = numpy.percentile(values, 75)

        # assign indices
        normal_values_indices = non_negative_indices * (metrics[:, index] <= upper_bound)
        high_extreme_indices = non_negative_indices * (metrics[:, index] > upper_bound)

        group_a_indices = high_score_indices_before_march * normal_values_indices
        group_b_indices = high_score_indices_before_march * high_extreme_indices

        # test tunes grouped by
        all_comparisons[metric] = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_b_indices, group_b_indices, ["normal values", "high extremes"], metric,
                                                                 tunes_type="continuous", inspection_mode=inspection_mode),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_b_indices, ["normal values", "high extremes"], metric,
                                                                  tunes_type="categorical", inspection_mode=inspection_mode)
        }

    for metric in ["isotopic_presence", "transmission", "fragmentation_305", "fragmentation_712"]:
        index = metrics_names.index(metric)

        # filter out negative values to estimate lower bound
        non_negative_indices = metrics[:, index] > 0  # missing values are -1s
        values = metrics[non_negative_indices, index]

        # too high and too low values are bad here
        lower_bound, upper_bound = numpy.percentile(values, [5, 95])

        # assign indices
        normal_values_indices = (metrics[:, index] <= upper_bound) + (metrics[:, index] >= lower_bound)
        extreme_indices = (metrics[:, index] > upper_bound) + (metrics[:, index] < lower_bound)

        group_a_indices = high_score_indices_before_march * normal_values_indices
        group_b_indices = high_score_indices_before_march * extreme_indices

        # test tunes grouped by
        all_comparisons[metric] = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_a_indices, group_b_indices, ["normal values", "extreme values"], metric,
                                                                 tunes_type="continuous", inspection_mode=inspection_mode),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_b_indices, ["normal values", "extreme values"], metric,
                                                                  tunes_type="categorical", inspection_mode=inspection_mode)
        }

    return all_comparisons


if __name__ == "__main__":

    # set full display
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)

    qc_tunes_database_path = "/Users/andreidm/ETH/projects/shiny_qc/data/nas2_qc_tunes_database_may13.sqlite"
    qc_metrics_database_path = "/Users/andreidm/ETH/projects/shiny_qc/data/nas2_qc_metrics_database_may13.sqlite"

    continuous_tunes, continuous_names, categorical_tunes, categorical_names = get_tunes_and_names(qc_tunes_database_path)

    if False:

        # TODO: figure out what's this for?

        # assess how imbalanced the categorical data is
        result_categorical = {}
        for i in range(len(categorical_names)):

            unique_values, counts = numpy.unique(categorical_tunes[:,i], return_counts=True)
            percents = [round(count / categorical_tunes.shape[0], 2) for count in counts]
            # make a dict { unique value: percent of total }
            result_categorical[categorical_names[i]] = dict(zip(unique_values, percents))

        # assess how imbalanced the continuous data is
        description_continuous = numpy.array(pandas.DataFrame(continuous_tunes).describe())  # save quantiles

        result_continuous = {}
        for i in range(len(continuous_names)):
            unique_values, counts = numpy.unique(continuous_tunes[:, i], return_counts=True)
            percents = [round(count / continuous_tunes.shape[0], 2) for count in counts]
            # make a dict { unique value: percent of total }
            result_continuous[continuous_names[i]] = dict(zip(unique_values, percents))

    if False:
        # check cross-correlations in tunes
        assess_cross_correlations(continuous_tunes, continuous_names, type='continuous', level=0.65)
        assess_cross_correlations(categorical_tunes, categorical_names, type='categorical', level=0.65)

    # read qc metrics
    metrics, metrics_names, acquisition, quality = get_metrics_data(qc_metrics_database_path)

    if False:
        # explore general correlations between tunes and metrics
        assess_correlations_between_tunes_and_metrics(metrics, metrics_names, continuous_tunes, continuous_names, tunes_type='continuous', method="spearman", inspection_mode=False)
        assess_correlations_between_tunes_and_metrics(metrics, metrics_names, categorical_tunes, categorical_names, tunes_type='categorical', inspection_mode=False)

        # feed "categorical" tunes to spearman correlation
        assess_correlations_between_tunes_and_metrics(metrics, metrics_names, categorical_tunes, categorical_names, tunes_type='continuous', method="spearman", inspection_mode=False)

    if False:
        # define good or bad based on the score
        high_score_indices = quality == '1'
        low_score_indices = quality == '0'

        # test tunes grouped by quality
        comparisons_for_scores = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, high_score_indices, low_score_indices, ["good quality", "bad quality"], "quality",
                                                                 tunes_type="continuous", inspection_mode=False),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, high_score_indices, low_score_indices, ["good quality", "bad quality"], "quality",
                                                                  tunes_type="categorical", inspection_mode=False)
        }

    if False:
        # test tunes grouped by extreme metrics values
        comparisons_for_metric_outliers = test_tunes_grouped_by_extreme_metrics_values(metrics, quality, acquisition, continuous_tunes, continuous_names, categorical_tunes, categorical_names,
                                                                                       inspection_mode=False)

    if False:
        # test tunes grouped by a recent trend in resolution & baselines
        good_resolution_indices = (quality == '1') * (acquisition < "2020-03-04")
        bad_resolution_indices = (quality == '1') * (acquisition >= "2020-03-04")

        comparisons_for_resolution_trend = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, good_resolution_indices, bad_resolution_indices, ["normal", "decreasing"], "resolution\ trend",
                                                                 tunes_type="continuous", inspection_mode=False),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, good_resolution_indices, bad_resolution_indices,  ["normal", "decreasing"], "resolution\ trend",
                                                                  tunes_type="categorical", inspection_mode=False)
        }

    if False:
        # test tunes grouped by trends in chemical_dirt
        elevated_dirt_indices = (acquisition > "2020-03-05")
        low_dirt_indices = (acquisition > "2019-12-21") * (acquisition < "2020-03-08")
        increasing_dirt_indices = (acquisition > "2019-11-12") * (acquisition < "2020-01-23")

        group_a_indices = (quality == '1') * elevated_dirt_indices
        group_b_indices = (quality == '1') * low_dirt_indices
        group_c_indices = (quality == '1') * increasing_dirt_indices

        comparisons_for_chem_dirt_trends = {

            "elevated vs low": {
                "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_a_indices, group_b_indices, ["elevated level", "low level"], "chemical\ dirt",
                                                                     tunes_type="continuous", inspection_mode=False),
                "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_b_indices, ["elevated level", "low level"], "chemical\ dirt",
                                                                      tunes_type="categorical", inspection_mode=False)
            },

            "elevated vs inscreasing": {
                "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_a_indices, group_c_indices, ["elevated level", "increasing level"], "chemical\ dirt",
                                                                     tunes_type="continuous", inspection_mode=False),
                "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_c_indices, ["elevated level", "increasing level"], "chemical\ dirt",
                                                                      tunes_type="categorical", inspection_mode=False)
            },

            "low vs increasing": {
                "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_b_indices, group_c_indices, ["low level", "increasing level"], "chemical\ dirt",
                                                                     tunes_type="continuous", inspection_mode=False),
                "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_b_indices, group_c_indices,  ["low level", "increasing level"], "chemical\ dirt",
                                                                      tunes_type="categorical", inspection_mode=False)
            }
        }

    if False:
        # test tunes grouped by levels of instrument noise
        higher_noise_indices = (quality == '1') * (acquisition < "2019-12-19")
        lower_noise_indices = (quality == '1') * ((acquisition >= "2020-01-23") * (acquisition < "2020-03-14"))

        comparisons_for_instrument_noise_levels = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, higher_noise_indices, lower_noise_indices, ["higher level", "lower level"], "instrument\ noise",
                                                                 tunes_type="continuous", inspection_mode=False),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, higher_noise_indices, lower_noise_indices, ["higher level", "lower level"], "instrument\ noise",
                                                                  tunes_type="categorical", inspection_mode=False)
        }

    if False:
        # test tunes grouped by variations in signal
        higher_variance_signal_indices = (quality == '1') * ((acquisition >= "2019-09-10") * (acquisition < "2020-01-23"))
        lower_variance_signal_indices = (quality == '1') * (acquisition >= "2020-01-23")

        comparisons_for_signal_variation = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, higher_variance_signal_indices, lower_variance_signal_indices, ["higher variance", "lower variance"], "signal",
                                                                 tunes_type="continuous", inspection_mode=False),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, higher_variance_signal_indices, lower_variance_signal_indices, ["higher variance", "lower variance"], "signal",
                                                                  tunes_type="categorical", inspection_mode=False)
        }

    pass



















