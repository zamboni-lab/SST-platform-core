
class Peak:

    def __init__(self, mz=None):

        self.mz = mz

    def _set_expected_peak_info(self, peak_info):
        """ Extract information about expected peak values from text and set to the class instance. """

        peak_feature_names = peak_info[0]
        peak_feature_values = peak_info[1]

        assert len(peak_feature_names) == len(peak_feature_values)

        for i in range(len(peak_feature_names)):

            if peak_feature_names[i] == 'mz':
                self.expected_mz = peak_feature_values[i]

            elif peak_feature_names == 'intensity':
                self.expected_intensity = peak_feature_values[i]

            else:
                print("Warning: expected feature", peak_feature_names[i], "ignored")

    def _correct_actual_peak_info(self, centroided_mz, profile_mz_region, profile_intensity_region):
        """ This method corrects the centroided m/z index and finds actual peak.
        In fact, this is an additional processing step after CWT application. """

        # TODO  The idea is search for the peak in the vicinity of the centroided m/z and
        # TODO  1) correct for the peak m/z index if needed,
        # TODO  2) create new peaks if they were not identified (think about repeated peacks in the final list)

        pass

    def _extract_actual_peak_features(self, mz_region, intensity_region):

        pass
