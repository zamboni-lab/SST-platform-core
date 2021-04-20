
import numpy, seaborn, pandas
from matplotlib import pyplot

from src.analysis import features_analysis, metrics_tunes_analysis
from src.constants import user


if __name__ == "__main__":

    features_path = '/Users/{}/ETH/projects/monitoring_system/res/nas2/190_injections_15_12_20/qc_features_15_12_20.sqlite'.format(user)
    metrics_path = '/Users/{}/ETH/projects/monitoring_system/res/nas2/190_injections_15_12_20/qc_metrics_15_12_20.sqlite'.format(user)
    tunes_path = '/Users/{}/ETH/projects/monitoring_system/res/nas2/190_injections_15_12_20/qc_tunes_15_12_20.sqlite'.format(user)

    # read qc metrics
    metrics, metrics_names, acquisition, quality = metrics_tunes_analysis.get_metrics_data(metrics_path)
    # read tunes
    continuous_tunes, continuous_names, categorical_tunes, categorical_names = metrics_tunes_analysis.get_tunes_and_names(tunes_path)
    # read features
    meta_info, features, colnames = features_analysis.get_features_data(path=features_path)
    features_cont, features_names_cont, features_cat, features_names_cat = features_analysis.split_features_to_cont_and_cat(features, numpy.array(colnames[4:]))
    # read meta data
    full_meta_data = features_analysis.get_meta_data(path=features_path)

    metrics_vcs = numpy.std(metrics, axis=0) / numpy.mean(metrics, axis=0)
    # print(metrics_names)
    # print(list(metrics_vcs))
    # print(list(numpy.std(metrics, axis=0)))
    print("metrics vcs: median={}, max={}".format(numpy.median(metrics_vcs), numpy.max(metrics_vcs)))

    # data = pandas.DataFrame(metrics, columns=metrics_names)
    # for name in metrics_names:
    #     seaborn.distplot(data[name], kde=True)
    #     pyplot.grid()
    #     pyplot.show()

    features_vcs = numpy.std(features_cont, axis=0) / numpy.mean(features_cont, axis=0)
    print("continuous features vcs: median={}, max={}".format(numpy.median(features_vcs), numpy.max(features_vcs)))
    features_vcs = numpy.std(features_cat, axis=0) / numpy.mean(features_cat, axis=0)
    print("categorical features vcs: median={}, max={}".format(numpy.median(features_vcs), numpy.max(features_vcs)))

