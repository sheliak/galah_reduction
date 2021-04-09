import numpy as np
import joblib


# --------------------------------------------------------
# ---------------- Save and read collection dumps --------
# --------------------------------------------------------
def read_pkl_spectra(file_path, read_rows=None, read_cols=None):
    data_read = joblib.load(file_path)
    if read_cols is not None:
        data_read = data_read[:, read_cols]
    if read_rows is not None:
        data_read = data_read[read_rows, :]
    return data_read


def save_pkl_spectra(data, file_path):
	# protocol 2 is compatible with both python versions
    joblib.dump(data, file_path, compress=0, protocol=2)


# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
def spectrum_resample(spectrum, wvl_orig, wvl_target):
    """

    :param spectrum:
    :param wvl_orig:
    :param wvl_target:
    :return:
    """
    idx_finite = np.isfinite(spectrum)
    min_wvl_s = np.nanmin(wvl_orig[idx_finite])
    max_wvl_s = np.nanmax(wvl_orig[idx_finite])
    idx_target = np.logical_and(wvl_target >= min_wvl_s,
                                wvl_target <= max_wvl_s)
    new_flux = np.interp(wvl_target[idx_target], wvl_orig[idx_finite], spectrum[idx_finite])
    return new_flux, idx_target


# --------------------------------------------------------
# ---------------- Collection parameters class -----------
# --------------------------------------------------------
class CollectionParameters:
    def __init__(self, filename):
        self.filename_full = filename
        self.filename_split = '.'.join(filename.split('.')[:-1]).split('_')

    def __get_str_pos__(self, search_str):
        try:
            idx = self.filename_split.index(search_str)
        except ValueError:
            idx = None
        return idx

    def __get_value__(self, search_str):
        idx_value = self.__get_str_pos__(search_str)
        if idx_value is not None:
            value_str = self.filename_split[idx_value + 1]
            try:
                return int(value_str)
            except ValueError:
                return float(value_str)
        else:
            return None

    def get_ccd(self):
        return self.filename_split[2][3]  # the fourth character is the number of ccd

    def get_wvl_range(self):
        wvl_start = float(self.filename_split[3])
        wvl_end = float(self.filename_split[4])
        return wvl_start, wvl_end

    def get_wvl_values(self):
        min, max = self.get_wvl_range()
        return np.arange(min, max, self.get_wvl_step())

    def get_interp_method(self):
        return self.filename_split[5]

    def get_wvl_step(self):
        return self.__get_value__('wvlstep')

    def get_snr_limit(self):
        return self.__get_value__('snr')

    def get_teff_step(self):
        return self.__get_value__('teff')

    def get_logg_step(self):
        return self.__get_value__('logg')

    def get_feh_step(self):
        return self.__get_value__('feh')
