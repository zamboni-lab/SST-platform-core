import numpy, pandas, scipy, seaborn, math

from src.qcmg import db_connector
from src.analysis import features_analysis
from src.constants import user, all_metrics

from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from matplotlib import pyplot
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
        pyplot.tight_layout()
        pyplot.show()

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
        pyplot.tight_layout()
        pyplot.show()

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


def assess_correlations_between_tunes_and_metrics(metrics, metrics_names, tunes, tunes_names, tunes_type, method="", inspection_mode=False, save_to=None):
    """ This method calculates correlations between tunes and metrics. """

    threshold = 0.4

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
                if inspection_mode and abs(correlation) >= threshold:
                    fig, ax = pyplot.subplots(figsize=(10, 5))

                    ax.scatter(tunes[:, j], metrics[:, i])

                    # adds a title and axes labels
                    ax.set_title(df.index[i] + ' vs ' + df.columns[j] + ": r = " + str(correlation))
                    ax.set_xlabel(df.columns[j])
                    ax.set_ylabel(df.index[i])

                    # removing top and right borders
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    pyplot.grid()

                    if save_to is None:
                        pyplot.show()
                    else:
                        pyplot.savefig(save_to + 'r_{}_vs_{}.pdf'.format(df.index[i], df.columns[j]))

    else:
        # use spearman correlation coefficient
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):

                correlation = scipy.stats.spearmanr(metrics[:, i], tunes[:, j])[0]
                df.iloc[i, j] = correlation

                # look into variables closer if correlation is high
                if inspection_mode and abs(correlation) >= threshold:
                    fig, ax = pyplot.subplots(figsize=(10, 5))

                    ax.scatter(tunes[:, j], metrics[:, i])

                    # adds a title and axes labels
                    ax.set_title(df.index[i] + ' vs ' + df.columns[j] + ": r = " + str(correlation))
                    ax.set_xlabel(df.columns[j])
                    ax.set_ylabel(df.index[i])

                    # removing top and right borders
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    if save_to is None:
                        pyplot.show()
                    else:
                        pyplot.savefig(save_to + '{}_vs_{}.pdf'.format(df.index[i], df.columns[j]))

    print("number of pairs with correlation > {}:".format(threshold), numpy.sum(numpy.abs(df.values) > threshold))

    # plot a heatmap
    pyplot.figure(figsize=(10, 6))
    seaborn.heatmap(df, xticklabels=df.columns, yticklabels=df.index, cmap='vlag')
    pyplot.title('Correlations: QC indicators vs machine settings')
    pyplot.tight_layout()

    if save_to is None:
        pyplot.show()
    else:
        pyplot.savefig(save_to + 'corrs_tunes_{}_metrics.pdf'.format(tunes_type))


def plot_tunes_values_by_groups(tunes, index, group_1_indices, group_2_indices, group_names, metric_split_by, testing_result, max_p="", save_plots_to=""):
    """ This method visualises samples of tunes that were statistically different.
        It makes scatter plots for two groups of values. """

    fig, ax = pyplot.subplots(figsize=(8, 5))

    groups = numpy.empty(tunes.shape[0]).astype(str)
    groups[group_1_indices] = group_names[0] + " (N = " + str(numpy.sum(group_1_indices)) + ")"
    groups[group_2_indices] = group_names[1] + " (N = " + str(numpy.sum(group_2_indices)) + ")"

    # add formatted title
    pyplot.title(r"$\bf{" + testing_result.columns[index].replace("_", "\_") + "}$" + ", split by " + r"$\bf{" + metric_split_by.replace("_", "\_") + "}$")

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    data = pandas.DataFrame({"groups": groups[group_1_indices + group_2_indices],
                             testing_result.columns[index]: tunes[group_1_indices + group_2_indices, index]})

    # seaborn.stripplot(x="groups", y=testing_result.columns[index], data=data)
    seaborn.violinplot(x="groups", y=testing_result.columns[index], data=data)
    pyplot.grid()
    pyplot.savefig(save_plots_to + "{}_by_{}_p={}.pdf".format(testing_result.columns[index], metric_split_by, max_p))
    # pyplot.show()


def test_tunes_for_statistical_differences(tunes, tunes_names, group_1_indices, group_2_indices, group_names, metric_split_by, tunes_type="continuous", level=0.05, inspection_mode=False, save_plots_to=""):
    """ This method conducts testing of hypothesis that
        tunes corresponding to "good" and "bad" runs differ statistically. """

    if tunes_type == "continuous":

        # prepare dataframe for statistical tests results
        df = pandas.DataFrame(numpy.empty([3, tunes.shape[1]]), index=["kolmogorov", "wilcoxon", "kruskall"])
        df.columns = tunes_names

        for i in range(tunes.shape[1]):
            # test continuous tunes of group 1 and group 2 runs

            group_1_values = tunes[group_1_indices, i][tunes[group_1_indices, i] != -1]  # filtering out missing values
            group_2_values = tunes[group_2_indices, i][tunes[group_2_indices, i] != -1]  # filtering out missing values

            try:
                p1 = ks_2samp(group_1_values, group_2_values)[1]
                if numpy.isnan(p1):
                    p1 = 1.
            except Exception:
                p1 = 1.
            try:
                p2 = mannwhitneyu(group_1_values, group_2_values)[1]
                if numpy.isnan(p2):
                    p2 = 1.
            except Exception:
                p2 = 1.
            try:
                p3 = kruskal(group_1_values, group_2_values)[1]
                if numpy.isnan(p3):
                    p3 = 1.
            except Exception:
                p3 = 1.

            df.iloc[:, i] = numpy.array([p1, p2, p3])

        for i in range(df.shape[0]):
            # correct p values for multiple testing
            df.iloc[i,:] = multipletests(df.iloc[i,:], method="fdr_bh")[1]

        # just add a row with True / False
        boolean_result = pandas.DataFrame(numpy.array([False for i in range(df.shape[1])])).T
        boolean_result.index = ["different"]
        boolean_result.columns = tunes_names

        for i in range(df.shape[1]):
            if sum(df.iloc[:,i] <= level) >= 3:
                max_p = str(round(numpy.max(df.iloc[:, i]), 4))
                boolean_result.iloc[0,i] = True

                # look into variables closer if they are statistically different
                if inspection_mode:
                    plot_tunes_values_by_groups(tunes, i, group_1_indices, group_2_indices, group_names, metric_split_by, boolean_result, max_p=max_p, save_plots_to=save_plots_to)

        return pandas.concat([df, boolean_result], axis=0)

    elif tunes_type == 'categorical':

        df = pandas.DataFrame(numpy.empty([1, tunes.shape[1]]), index=["chi2"])
        df.columns = tunes_names

        # now go over categorical tunes and perform chi2
        for i in range(tunes.shape[1]):
            continjency_table = get_contingency_table(tunes, i, group_1_indices, group_2_indices)
            print(i, tunes_names[i])
            print(continjency_table)

            try:
                p = chi2_contingency(continjency_table)[1]
                if numpy.isnan(p):
                    p = 1.
            except Exception:
                p = 1.

            df.iloc[0, i] = p

        # correct p values for multiple testing
        df.iloc[0, :] = multipletests(df.iloc[0, :], method="fdr_bh")[1]

        # just add a row with True / False
        boolean_result = pandas.DataFrame(numpy.array([False for i in range(df.shape[1])])).T
        boolean_result.index = ["different"]
        boolean_result.columns = tunes_names

        for i in range(df.shape[1]):
            if round(df.iloc[0, i], 2) <= level:
                max_p = str(round(df.iloc[0, i], 4))
                boolean_result.iloc[0, i] = True

                # look into variables closer if they are statistically different
                if inspection_mode:
                    plot_tunes_values_by_groups(tunes, i, group_1_indices, group_2_indices, group_names, metric_split_by, boolean_result, max_p=max_p, save_plots_to=save_plots_to)

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


def get_tunes_and_names(path, no_filter=False):
    """ This method reads a database with tunes, makes some preprocessing and returns
        categorical and continuous tunes with names. """

    conn = db_connector.create_connection(path)
    database, colnames = db_connector.fetch_table(conn, "qc_tunes")

    tunes = numpy.array(database)
    tunes = tunes[tunes[:, 2].argsort()]

    if no_filter:
        # get a db as is
        return tunes, colnames

    # extract numeric values only
    indices = [10, 13, 16]
    indices.extend([i for i in range(18, 151)])

    # compose arrays
    tunes = numpy.vstack([tunes[:, i].astype(float) for i in indices]).T
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
    continuous_names = numpy.array(continuous_names).T
    categorical_names = numpy.array(categorical_names).T

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
    metrics = metrics.astype(float)

    return metrics, metrics_names, acquisition, quality


def test_tunes_grouped_by_extreme_metrics_values(metrics, quality, acquisition, continuous_tunes, continuous_names, categorical_tunes, categorical_names, inspection_mode=False, save_plots_to="/Users/dmitrav/ETH/projects/monitoring_system/res/analysis/by_outliers_in_metrics/"):
    """ This method groups tunes based on extreme values (outliers) of metrics,
        then performs statistical tests and saves the results in the dict.  """

    all_comparisons = {}  # store results in a dict

    allowed_score_indices = (quality <= '1')  # now all inclusive, but one can filter

    for metric in ["resolution_200", "resolution_700", "signal", "s2b", "s2n"]:
        index = metrics_names.index(metric)

        # filter out negative values to estimate lower bound
        non_negative_indices = metrics[:, index] > 0  # missing values are -1s
        values = metrics[non_negative_indices, index]

        # very low values are bad here
        lower_bound = numpy.percentile(values, 20)

        # assign indices
        normal_values_indices = non_negative_indices * (metrics[:, index] >= lower_bound)
        low_extreme_indices = non_negative_indices * (metrics[:, index] < lower_bound)

        group_a_indices = allowed_score_indices * normal_values_indices
        group_b_indices = allowed_score_indices * low_extreme_indices

        # test tunes grouped by
        all_comparisons[metric] = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_a_indices, group_b_indices, ["above 20%", "below 20%"], metric,
                                                                 tunes_type="continuous", inspection_mode=inspection_mode, save_plots_to=save_plots_to),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_b_indices, ["above 20%", "below 20%"], metric,
                                                                  tunes_type="continuous", inspection_mode=inspection_mode, save_plots_to=save_plots_to)
        }

    for metric in ["average_accuracy", "chemical_dirt", "instrument_noise", "baseline_25_150", "baseline_50_150",
                   "baseline_25_650", "baseline_50_650"]:

        index = metrics_names.index(metric)

        # filter out negative values to estimate lower bound
        non_negative_indices = metrics[:, index] > 0  # missing values are -1s
        values = metrics[non_negative_indices, index]

        # very high values are bad here
        upper_bound = numpy.percentile(values, 80)

        # assign indices
        normal_values_indices = non_negative_indices * (metrics[:, index] <= upper_bound)
        high_extreme_indices = non_negative_indices * (metrics[:, index] > upper_bound)

        group_a_indices = allowed_score_indices * normal_values_indices
        group_b_indices = allowed_score_indices * high_extreme_indices

        # test tunes grouped by
        all_comparisons[metric] = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_b_indices, group_b_indices, ["below 80%", "above 80%"], metric,
                                                                 tunes_type="continuous", inspection_mode=inspection_mode, save_plots_to=save_plots_to),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_b_indices, ["below 80%", "above 80%"], metric,
                                                                  tunes_type="continuous", inspection_mode=inspection_mode, save_plots_to=save_plots_to)
        }

    for metric in ["isotopic_presence", "transmission", "fragmentation_305", "fragmentation_712"]:

        index = metrics_names.index(metric)

        # filter out negative values to estimate lower bound
        non_negative_indices = metrics[:, index] > 0  # missing values are -1s
        values = metrics[non_negative_indices, index]

        # too high and too low values are bad here
        lower_bound, upper_bound = numpy.percentile(values, [10, 90])

        # assign indices
        normal_values_indices = (metrics[:, index] <= upper_bound) + (metrics[:, index] >= lower_bound)
        extreme_indices = (metrics[:, index] > upper_bound) + (metrics[:, index] < lower_bound)

        group_a_indices = allowed_score_indices * normal_values_indices
        group_b_indices = allowed_score_indices * extreme_indices

        # test tunes grouped by
        all_comparisons[metric] = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, group_a_indices, group_b_indices, ["between 10% and 90%", "outlier"], metric,
                                                                 tunes_type="continuous", inspection_mode=inspection_mode, save_plots_to=save_plots_to),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, group_a_indices, group_b_indices, ["between 10% and 90%", "outlier"], metric,
                                                                  tunes_type="continuous", inspection_mode=inspection_mode, save_plots_to=save_plots_to)
        }

    return all_comparisons


if __name__ == "__main__":

    # set full display
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)

    qc_tunes_database_path = "/Users/{}/ETH/projects/monitoring_system/res/nas2/qc_tunes_database.sqlite".format(user)
    qc_metrics_database_path = "/Users/{}/ETH/projects/monitoring_system/res/nas2/qc_metrics_database.sqlite".format(user)

    # read qc metrics
    metrics, metrics_names, acquisition, quality = get_metrics_data(qc_metrics_database_path)
    # read tunes
    continuous_tunes, continuous_names, categorical_tunes, categorical_names = get_tunes_and_names(qc_tunes_database_path)
    # read meta data
    full_meta_data = features_analysis.get_meta_data()

    # FILTER OUT DMSO
    ipa_h20_indices = numpy.where(full_meta_data['buffer_id'] == 'IPA_H2O')[0]

    continuous_tunes = continuous_tunes[ipa_h20_indices, :]
    categorical_tunes = categorical_tunes[ipa_h20_indices, :]
    metrics = metrics[ipa_h20_indices, :]
    acquisition = acquisition[ipa_h20_indices]
    quality = quality[ipa_h20_indices]

    if True:
        """ scan for the trends in QC metrics """

        time_interval_days = 60

        dates = full_meta_data['acquisition_date'][ipa_h20_indices].values
        dates = [date[:10] for date in dates]

        df = pandas.DataFrame(metrics, columns=metrics_names)
        df.insert(0, 'date', dates)
        df = df.sort_values(by=['date'])

        for qc_indicator in all_metrics:

            print('\n{}:\n'.format(qc_indicator.upper()))

            # filter out missing values
            df = df.loc[df[qc_indicator] != -1, :]

            # keep only single QC per date
            u_dates = df['date'].unique()
            for date in u_dates:
                if qc_indicator in ['resolution_200', 'resolution_700', 'signal', 's2b', 's2n']:
                    # for these, pick the min (as the worst)
                    single_date_value = df.loc[df['date'] == date, qc_indicator].min()
                elif qc_indicator in ['average_accuracy', 'chemical_dirt', 'instrument_noise', 'baseline_25_150',
                                      'baseline_50_150', 'baseline_25_650', 'baseline_50_650']:
                    # for these, pick the max (as the worst)
                    single_date_value = df.loc[df['date'] == date, qc_indicator].max()
                elif qc_indicator in ['isotopic_presence', 'transmission', 'fragmentation_305', 'fragmentation_712']:
                    # for these, pick the median
                    single_date_value = df.loc[df['date'] == date, qc_indicator].median()
                else:
                    raise ValueError("Unknown QC indicator")

                idx_to_drop = df[(df['date'] == date) & (df[qc_indicator] != single_date_value)].index
                df = df.drop(idx_to_drop)

            for i in range(df.shape[0]):
                # scan for the trends for a qc indicator
                date_from = str(datetime.fromisoformat(df['date'].values[i]))[:10]
                date_till = str(datetime.fromisoformat(df['date'].values[i]) + timedelta(days=time_interval_days))[:10]
                # select records for specified time period
                subset = df[(df['date'] >= date_from) & (df['date'] <= date_till)]

                if subset.shape[0] < 2:
                    continue
                else:

                    # convert dates to X (= number of days since 0 time point)
                    dates = [datetime.fromisoformat(date) for date in subset['date'].values]
                    x = [0]

                    for i in range(1, len(dates)):
                        delta = dates[i] - dates[i - 1]
                        delta_in_days = delta.days + delta.seconds / 3600 / 24
                        x.append(x[i - 1] + delta_in_days)

                    x = numpy.array(x).reshape(-1,1)

                    y = subset[qc_indicator].values.reshape(-1,1)
                    y = StandardScaler().fit_transform(y)

                    reg = LinearRegression().fit(x, y)

                    score = round(reg.score(x, y), 4)
                    coef = round(reg.coef_[0][0], 4)

                    if score < 0.01:
                        trend_value = "unchanged"
                    else:
                        if abs(coef) < 0.05:
                            trend_value = "unchanged"
                        else:
                            if coef < 0:
                                trend_value = "decreasing"
                            else:
                                trend_value = "increasing"

                    if trend_value != 'unchanged' and score > 0.5 and subset.shape[0] > 2:
                        print('from {} till {}: {}, r2={}, coef={}, size={}'.format(date_from, date_till, trend_value, score, coef, subset.shape[0]))

    if False:
        """ plot all resolution_200 values over time, filtering out missing values """

        dates = full_meta_data['acquisition_date'][ipa_h20_indices].values
        dates = [date[:10] for date in dates]

        df = pandas.DataFrame(metrics, columns=metrics_names)
        df.insert(0, 'date', dates)
        df = df.sort_values(by=['date'])

        # some filtering for better visualisation
        df = df.loc[df['resolution_200'] != -1, :]
        df = df.iloc[20:, :]
        df = df.loc[df['date'] != '2020-02-16', :]

        # keep only single QC per date
        u_dates = df['date'].unique()
        for date in u_dates:
            min_res = df.loc[df['date'] == date, 'resolution_200'].min()
            idx_to_drop = df[(df['date'] == date) & (df['resolution_200'] != min_res)].index
            df = df.drop(idx_to_drop)

        df = df[df['date'] <= '2020-03-14']

        y = df.loc[:, 'resolution_200'].values
        x = [j for j in range(len(y))]
        labels = [df['date'].values[j] for j in range(len(y))]

        pyplot.figure(figsize=(10,6))
        pyplot.plot(x, y, '-o', color='black', label='values')

        y_trend = df.loc[(df['date'] >= '2020-02-22'), 'resolution_200'].values
        x_trend = [j for j in range(len(y)) if y[j] in y_trend]
        pyplot.plot(x_trend, y_trend, '.', color='lightblue', label='negative trend')

        pyplot.xticks(ticks=x, labels=labels, rotation='vertical')
        pyplot.title('resolution_200')
        pyplot.legend()
        pyplot.grid()
        pyplot.tight_layout()
        # pyplot.show()
        pyplot.savefig('/Users/andreidm/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/monitoring_system/v8/resolution_trend.pdf')

    if False:
        """ plot all s2n values over time, highlighting outliers """

        dates = full_meta_data['acquisition_date'][ipa_h20_indices].values
        dates = [date[:10] for date in dates]

        df = pandas.DataFrame(metrics, columns=metrics_names)
        df.insert(0, 'date', dates)
        df = df.sort_values('date')

        all_values = df.loc[:, 's2n'].values

        # some filtering for better visualisation
        df = df.iloc[:-25, :]

        y = df.loc[:, 's2n'].values
        x = [j for j in range(len(y))]
        labels = [df['date'].values[j] for j in range(len(y))]

        pyplot.figure(figsize=(10,6))
        pyplot.plot(x, y, '-o', color='black', label='> 20th percentile')

        lower_bound = numpy.percentile(all_values, 20)
        y_outliers = [value for value in y if value < lower_bound]
        x_outliers = [j for j in range(len(y)) if y[j] < lower_bound]

        pyplot.plot(x_outliers, y_outliers, '.', color='pink', label='< 20th percentile')

        pyplot.xticks(ticks=x, labels=labels, rotation='vertical', fontsize=8)
        pyplot.title('signal-to-noise')
        pyplot.legend()
        pyplot.grid()
        pyplot.tight_layout()
        pyplot.show()
        # pyplot.savefig('/Users/dmitrav/ETH/projects/monitoring_system/res/analysis/trends_in_metrics/s2n.pdf')

    if False:

        """ look for correlation between decreasing resolution_200 trend and the tunes """

        dates = full_meta_data['acquisition_date'][ipa_h20_indices].values
        dates = [date[:10] for date in dates]

        df = pandas.DataFrame(metrics, columns=metrics_names)
        df.insert(0, 'date', dates)
        df = df.sort_values(['date'])

        # keep only single QC per date
        idx_to_drop = []
        u_dates = df['date'].unique()
        for date in u_dates:
            min_res = df.loc[df['date'] == date, 'resolution_200'].min()
            idx_to_drop.extend(list(df[(df['date'] == date) & (df['resolution_200'] != min_res)].index))
        df = df.drop(idx_to_drop)

        picked_dates = (df['date'] >= '2020-02-22') & (df['date'] <= '2020-03-14')

        falling_res200 = df.loc[picked_dates, 'resolution_200'].values

        df = pandas.DataFrame(continuous_tunes, columns=continuous_names)
        df.insert(0, 'date', dates)
        df = df.sort_values(['date'])
        df = df.drop(idx_to_drop)

        pyplot.figure(figsize=(10,6))
        for i in range(len(continuous_names)):

            tune_vals = df.loc[picked_dates, continuous_names[i]].values

            r, p = scipy.stats.pearsonr(falling_res200, tune_vals)
            p = p * (len(continuous_names) - 8)  # bonferroni adjusted to this case

            pyplot.plot(r, -numpy.log10(p), 'ko')
            if abs(r) > 0.5 and p < 0.05:
                pyplot.plot(r, -numpy.log10(p), 'o', label=continuous_names[i])
                print('correlation with {}: {}'.format(continuous_names[i], r))

        pyplot.axvline(x=0.6, ls='--')
        pyplot.axvline(x=-0.6, ls='--')
        pyplot.axhline(y=-numpy.log10(0.05), ls='--')
        pyplot.xlabel('Pearson correlation')
        pyplot.ylabel('-log10(p)')
        pyplot.grid()
        pyplot.legend(bbox_to_anchor=(1.01, 1))
        pyplot.title('resolution_200 vs machine settings')
        pyplot.tight_layout()
        # pyplot.show()
        pyplot.savefig('/Users/andreidm/Library/Mobile Documents/com~apple~CloudDocs/ETHZ/papers_posters/monitoring_system/v8/trend_tunes_corrs_volcano.pdf')

    if False:

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

    if False:
        # plot correlations between metrics (QC indicators) and tunes (machine settings)
        save_plots_to = '/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/metrics_tunes_corrs/'.format(user)

        # explore general correlations between tunes and metrics
        assess_correlations_between_tunes_and_metrics(metrics, metrics_names, continuous_tunes, continuous_names,
                                                      'cont', method='pearson', inspection_mode=True,
                                                      save_to=save_plots_to)

        # assess_correlations_between_tunes_and_metrics(metrics, metrics_names, categorical_tunes, categorical_names,
        #                                               tunes_type='categorical', inspection_mode=False)

        # feed "categorical" tunes to spearman correlation
        assess_correlations_between_tunes_and_metrics(metrics, metrics_names, categorical_tunes, categorical_names,
                                                      'cat', method="pearson", inspection_mode=True,
                                                      save_to=save_plots_to)

    if False:
        # plot MI between metrics (QC indicators) and tunes (machine settings)
        features_analysis.compute_mutual_info_between_tunes_and_features(
            metrics, metrics_names, continuous_tunes, continuous_names, 'cont', features_type='metrics', inspection_mode=True
        )

        features_analysis.compute_mutual_info_between_tunes_and_features(
            metrics, metrics_names, categorical_tunes, categorical_names, 'cat', features_type='metrics', inspection_mode=True
        )

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
                                                                                       inspection_mode=True)

    if False:
        # # test tunes grouped by QC metrics values

        metric_of_interest = 's2b'
        percentile = 20  # split in below and above this percentile

        save_to = '/Users/{}/ETH/projects/monitoring_system/res/analysis/v7_img/statistical_comparisons/'.format(user)

        low_values_indices = metrics[:, metrics_names.index(metric_of_interest)] < numpy.percentile(metrics[:, metrics_names.index(metric_of_interest)], percentile)
        high_values_indices = metrics[:, metrics_names.index(metric_of_interest)] >= numpy.percentile(metrics[:, metrics_names.index(metric_of_interest)], percentile)

        comparisons = {
            "continuous": test_tunes_for_statistical_differences(continuous_tunes, continuous_names, low_values_indices, high_values_indices, ["< {}%".format(percentile), "> {}%".format(percentile)], metric_of_interest,
                                                                 tunes_type="continuous", inspection_mode=True, save_plots_to=save_to),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, low_values_indices, high_values_indices, ["< {}%".format(percentile), "> {}%".format(percentile)], metric_of_interest,
                                                                  tunes_type="categorical", inspection_mode=True, save_plots_to=save_to)
        }

        print(comparisons)

    if False:
        # this code I used to generate plots:
        # - how tunes evolve with time (just examples)
        # - how metrics split by tunes are different
        # all done with hardcode

        # x = [x for x in range(categorical_tunes.shape[0])][21:88][::2]
        # y = categorical_tunes[:, 18][21:88][::2]
        #
        # dates = numpy.sort(full_meta_data.iloc[ipa_h20_indices, :]['acquisition_date'].values)[21:88][::2]
        # dates = [str(date)[:10] for date in dates]
        #
        # pyplot.plot(x, y, 'o-')
        # pyplot.xticks(ticks=x, labels=dates, rotation='vertical')
        # pyplot.title("Ion Focus")
        # pyplot.grid()
        # pyplot.tight_layout()
        # # pyplot.savefig("/Users/dmitrav/ETH/projects/monitoring_system/res/analysis/fig4-1.pdf")
        # pyplot.show()

        first_group_indices = numpy.concatenate([[False for x in range(9)], [True for x in range(36-9)], [False for x in range(100-36)]])
        second_group_indices = numpy.concatenate([[False for x in range(41)], [True for x in range(68-41)], [False for x in range(100-68)]])

        comparisons = {
            "continuous": test_tunes_for_statistical_differences(metrics, metrics_names, first_group_indices, second_group_indices, ["group 1", "group 2"], "Amp_Offset", level=0.05,
                                                                 tunes_type="continuous", inspection_mode=True)
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
                                                                 tunes_type="continuous", inspection_mode=True),
            "categorical": test_tunes_for_statistical_differences(categorical_tunes, categorical_names, higher_noise_indices, lower_noise_indices, ["higher level", "lower level"], "instrument\ noise",
                                                                  tunes_type="categorical", inspection_mode=True)
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

    print()