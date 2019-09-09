
import json, seaborn, pandas, matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def get_normal_distributions(qcm):
    """ This method extracts all values for each qc metric from qcm matrix with normal scans (test1 dataset).
        Metrics names are returned as well (they are the same for all qcm matrices). """

    all_distributions = []

    for qc_metric in qcm['qc_runs'][0]['qc_names']:

        distribution = []
        for qc_run in qcm['qc_runs']:
            distribution.append(qc_run['qc_values'][qc_run['qc_names'].index(qc_metric)])

        all_distributions.append(distribution)

    return all_distributions, qcm['qc_runs'][0]['qc_names']


def get_shifted_distributions(qcm, specific_names):
    """ This method extracts all values for each qc metric from qcm matrix with 'spoiled' scans (test2 dataset).
        Specific names indicate which runs to consider for it. """

    all_shifted_distributions = []

    for qc_metric in qcm['qc_runs'][0]['qc_names']:

        shifted_distribution = []
        for qc_run in qcm['qc_runs']:

            if qc_run['original_filename'].split('.')[0].split('_')[-1] in specific_names:
                shifted_distribution.append(qc_run['qc_values'][qc_run['qc_names'].index(qc_metric)])
            else:
                pass

        all_shifted_distributions.append(shifted_distribution)

    return all_shifted_distributions


def plot_by_type_and_save(normal_values, spoilt_values, metrics_names, plot_title, path_to_save_to, to_show=False):
    """ This method gets the values (normal vs altered)for plotting,
        draws the violin plot to compare two distributions (normal vs altered) for each metric,
        and saves the figure to specified path. """

    normal_df = pandas.DataFrame(normal_values).T
    normal_df.columns = metrics_names
    normal_df['type'] = 'normal'

    spoiled_df = pandas.DataFrame(spoilt_values).T
    spoiled_df.columns = metrics_names
    spoiled_df['type'] = 'spoilt'

    combined_df = pandas.concat([normal_df, spoiled_df])

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(23, 9.5))

    for i in range(len(metrics_names)):

        _, p = ks_2samp(normal_df[metrics_names[i]], spoiled_df[metrics_names[i]])

        seaborn.violinplot(y=metrics_names[i], x='type', palette='muted', data=combined_df, ax=axes[int(i/4), i%4])\
            .set_title("p = " + str(round(p,3)),fontdict={'fontsize': 10,'fontweight': 'medium'})

    plt.suptitle(plot_title, fontweight='bold')

    if to_show:
        plt.show()

    plt.savefig(path_to_save_to, dpi=300)


def plot_by_metric_and_save(normal_values, metrics_names, all_spoilt_values, all_types, path_to_save_to, to_show=False):
    """ This method gets all the values for plotting,
        draws the violin plot to compare all the distributions (normal vs all types of alterations),
        saves the figure to specified path. """

    combined_df = pandas.DataFrame(normal_values).T
    combined_df.columns = metrics_names
    combined_df['type'] = 'normal'

    assert len(all_spoilt_values) == len(all_types)

    for i in range(len(all_types)):

        spoiled_df = pandas.DataFrame(all_spoilt_values[i]).T
        spoiled_df.columns = metrics_names
        spoiled_df['type'] = all_types[i]

        combined_df = pandas.concat([combined_df, spoiled_df])

    for i in range(len(metrics_names)):

        fig, ax = plt.subplots(figsize=(15,8))

        seaborn.set()
        seaborn.violinplot(y=metrics_names[i], x='type', palette='muted', data=combined_df, ax=ax)\
            .set_title(metrics_names[i], fontdict={'fontsize': 12,'fontweight': 'bold'})

        if to_show:
            plt.show()

        plt.savefig(path_to_save_to.replace('.png', '_'+str(i)+'.png'), figsize=(15,8), dpi=300)


if __name__ == '__main__':

    qc_matrix_file_path = '/Users/andreidm/ETH/projects/ms_feature_extractor/res/test1/qc_matrix.json'

    with open(qc_matrix_file_path) as file:
        qcm_test1 = json.load(file)

    normal_metrics_values, metrics_names = get_normal_distributions(qcm_test1)

    qc_matrix_file_path = '/Users/andreidm/ETH/projects/ms_feature_extractor/res/test2/qc_matrix.json'

    with open(qc_matrix_file_path) as file:
        qcm_test2 = json.load(file)

    # almost all ions in the normal region are saturated and HEX is saturated in the background
    saturation_metrics_values_1 = get_shifted_distributions(qcm_test2, ['007', '008', '009', '010', '011', '012', '013',
                                                                        '014', '015', '016', '017', '018', '019'])

    # saturated ions (Fluconazole, Perfluorodecanoic acid, Tricosafluorododecanoic acid and Perfluorotetradecanoic acid)
    # and HEX is saturated in the background
    saturation_metrics_values_2 = get_shifted_distributions(qcm_test2, ['042', '043', '044', '045', '046', '047', '049',
                                                                        '050', '051', '052', '053', '054', '056', '057',
                                                                        '058', '059', '060', '061'])

    # new buffer mix made (contamination after switching the syringe needle)
    contamination_metrics_values_1 = get_shifted_distributions(qcm_test2, ['105', '106', '107', '108', '109', '110',
                                                                           '112', '113', '114'])

    # new buffer mix made (contamination after switching the syringe needle)
    contamination_metrics_values_2 = get_shifted_distributions(qcm_test2, ['115', '116', '117'])

    # small gap between the injector port and PEEK tubing), increase in background
    dead_volume_metrics_values_1 = get_shifted_distributions(qcm_test2, ['119', '120', '121', '122', '123', '124',
                                                                         '126', '127', '128', '129', '130', '131'])

    # small gap between the PEEK tubing and ESI needle), increase in background
    dead_volume_metrics_values_2 = get_shifted_distributions(qcm_test2, ['133', '134', '135', '136', '137', '138',
                                                                         '140', '141', '142', '143', '144', '145'])

    # plot_by_type_and_save(normal_metrics_values, saturation_metrics_values_1, metrics_names,
    #                       "saturation 1: most ions (incl. HEX) are saturated",
    #                       '/Users/andreidm/ETH/projects/qc_metrics/res/test2/images/normal_vs_saturation_1.png')
    #
    # plot_by_type_and_save(normal_metrics_values, saturation_metrics_values_2, metrics_names,
    #                       "saturation 2: some ions (incl. HEX) are saturated",
    #                       '/Users/andreidm/ETH/projects/qc_metrics/res/test2/images/normal_vs_saturation_2.png')
    #
    # plot_by_type_and_save(normal_metrics_values, contamination_metrics_values_1, metrics_names,
    #                       "new buffer mix 1: contamination after switching the syringe needle",
    #                       '/Users/andreidm/ETH/projects/qc_metrics/res/test2/images/normal_vs_new_buffer_1.png')
    #
    # plot_by_type_and_save(normal_metrics_values, contamination_metrics_values_2, metrics_names,
    #                       "new buffer mix 2: contamination after switching the syringe needle",
    #                       '/Users/andreidm/ETH/projects/qc_metrics/res/test2/images/normal_vs_new_buffer_2.png')
    #
    # plot_by_type_and_save(normal_metrics_values, dead_volume_metrics_values_1, metrics_names,
    #                       "dead volume 1: small gap between the injector port and PEEK tubing",
    #                       '/Users/andreidm/ETH/projects/qc_metrics/res/test2/images/normal_vs_dead_volume_1.png')
    #
    # plot_by_type_and_save(normal_metrics_values, dead_volume_metrics_values_2, metrics_names,
    #                       "dead volume 2: small gap between the PEEK tubing and ESI needle",
    #                       '/Users/andreidm/ETH/projects/qc_metrics/res/test2/images/normal_vs_dead_volume_2.png')

    plot_by_metric_and_save(normal_metrics_values, metrics_names,
                            [saturation_metrics_values_1, saturation_metrics_values_2,
                             contamination_metrics_values_1, contamination_metrics_values_2,
                             dead_volume_metrics_values_1, dead_volume_metrics_values_2],
                            ['saturation_1', 'saturation_2', 'contamination_1',
                             'contamination_2', 'dead_volume_1', 'dead_volume_2'],
                            '/Users/andreidm/ETH/projects/qc_metrics/res/test2/images/normal_vs_all.png')



    print()
