import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from tools import galah_tools as gt
import numpy as np
import joblib
import json

import matplotlib.pyplot as plt


# set Galah tools
gt.setup(root_folder=os.curdir)


def _load_model_complete():
    """
    Procedure that loads data and models needed to evaluate stellar parameters

    :return:
    """

    model_path = 'parameters_model/'

    # load neural network model and its weights
    with open(model_path + 'model_nn.json', 'r') as f:
        model_json = json.load(f)
    model_nn = model_from_json(model_json)
    model_nn.load_weights(model_path + 'model_nn_weights.h5')
    # load wavelength bins at which parameters and abundances are determined
    model_wvl = np.loadtxt(model_path + 'model_wvl.dat')
    # load list of output parameters
    model_out = np.loadtxt(model_path + 'model_out.dat')
    # load scaler of values that are outputted by the neural network
    output_scaler = joblib.load(model_path + 'model_norm.pkl')

    return model_nn, model_wvl, model_out, output_scaler


def _spectrum_resample(spectrum, wvl_orig, wvl_target):
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
    nex_flux_out = np.ndarray(len(wvl_target))
    nex_flux_out.fill(np.nan)
    nex_flux_out[idx_target] = new_flux
    return nex_flux_out, idx_target


def get_parameters_nn(sobjects, logging, processes=1):
    """

    :param sobjects:
    :param logging:
    :param processes:
    :return:
    """

    logging.info('Loading precomputed neural network')
    model_nn, model_wvl, model_out, output_scaler = _load_model_complete()

    # load spectra
    spec_data = np.full((len(sobjects), len(model_wvl)), np.nan)

    logging.info('Preparing spectral data')
    for i_s, s_id in enumerate(sobjects):
        for ccd in [1, 2, 3, 4]:
            # retrieve selected spectrum
            s_id_ccd = str(s_id) + str(ccd)
            gt.spectrum(s_id_ccd, kind='norm', wavelength='object', linearize=False)
            spec_flx = s1.f
            spec_wvl = s1.l

            # intepolate spectrum to predetermined wvl bins
            use_spec_flx, idx_use = _spectrum_resample(spec_flx, spec_wvl, model_wvl)
            spec_data[i_s, idx_use] = use_spec_flx

    # treat possible missing values
    idx_bad_spec = np.where(np.logical_not(np.isfinite(spectral_data)))[0]
    n_bad_spectra = len(idx_bad_spec)
    if n_bad_spectra > 0:
        # replace nan values with theoretical continuum flux value
        spectral_data[idx_bad_spec] = 1.

    # apply neural network to loaded spectra
    nn_results = model_nn.predict(1. - spectral_data)

    # de-normalise outputs
    train_feat_mean, train_feat_std = output_scaler
    for i_f in range(len(model_out)):
        nn_results[:, i_f] = nn_results[:, i_f] * train_feat_std[i_f] + train_feat_mean[i_f]

    # return results
    return model_out, nn_results
