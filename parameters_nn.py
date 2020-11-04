import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import model_from_json
from tools import galah_tools as gt
from astropy.io import fits
import numpy as np
import joblib
import json


# set Galah tools
gt.setup(root_folder=os.curdir+'/reductions/results')


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
    model_wvl = np.loadtxt(model_path + 'model_wvl.dat', dtype=np.float32)
    # load list of output parameters
    model_out = np.loadtxt(model_path + 'model_out.dat', dtype='S20')
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
    return new_flux, idx_target


def _write_out(prt_str, prio=0, logging=None):
    """

    :param prt_str:
    :param prio:
    :param logging:
    :return: None
    """
    if logging is None:
        print prt_str
    else:
        if prio == 0:
            logging.info(prt_str)
        elif prio == 1:
            logging.warning(prt_str)
        elif prio == 2:
            logging.error(prt_str)


def get_parameters_nn(sobjects, logging=None, processes=1):
    """

    :param sobjects:
    :param logging:
    :param processes:
    :return:
    """

    # set environment variable that controls number of threads in keras
    os.environ['OMP_NUM_THREADS'] = str(processes)

    _write_out('Loading precomputed neural network', prio=0, logging=logging)
    model_nn, model_wvl, model_out, output_scaler = _load_model_complete()

    # load spectra
    spec_data = np.full((len(sobjects), len(model_wvl)), np.nan)

    _write_out('Preparing spectral data', prio=0, logging=logging)
    for i_s, s_id in enumerate(sobjects):
        for ccd in [1, 2, 3, 4]:
            # retrieve selected spectrum
            s_id_ccd = str(s_id) + str(ccd)
            try:
                spec_class = gt.spectrum(s_id_ccd, kind='norm', wavelength='object', linearize=False)
            except IOError as err:
                _write_out('Spectrum was not found for ccd sobject_id combination %s' % (s_id_ccd), prio=1, logging=logging)
                continue

            spec_flx = spec_class.f
            spec_wvl = spec_class.l

            # intepolate spectrum to predetermined wvl bins
            use_spec_flx, idx_use = _spectrum_resample(spec_flx, spec_wvl, model_wvl)
            spec_data[i_s, idx_use] = use_spec_flx

    # treat possible missing values
    idx_bad_spec = np.where(np.logical_not(np.isfinite(spec_data)))[0]
    n_bad_spectra = len(idx_bad_spec)
    if n_bad_spectra > 0:
        # replace nan values with theoretical continuum flux value
        _write_out('Replacing %s nan pixels' % (n_bad_spectra), prio=0, logging=logging)
        spec_data[idx_bad_spec] = 1.

    # prepare spectral data for the further use in the Keras library
    # only needed when using deep convolutional network
    spec_data = np.expand_dims(spec_data, axis=2)

    # apply neural network to loaded spectra
    nn_results = model_nn.predict(1. - spec_data)

    # de-normalise outputs
    train_feat_mean, train_feat_std = output_scaler
    for i_f in range(len(model_out)):
        nn_results[:, i_f] = nn_results[:, i_f] * train_feat_std[i_f] + train_feat_mean[i_f]

    # write results into every header of every com spectrum
    _write_out('Writing parameters back to spectra fits files', prio=0, logging=logging)
    for i_s, s_id in enumerate(sobjects):
        date = str(s_id)[:6]
        for ccd in [1, 2, 3, 4]:
            # open fits file and write parameters into headers
            fits_path = 'reductions/results/%s/spectra/com/%s%s.fits' % (date, s_id, ccd)
            if not os.path.isfile(fits_path):
                continue
            hdulist = fits.open(fits_path, mode='update')
            for extension in range(len(hdulist)):
                for i_p, p_name in enumerate(model_out):
                    hdulist[extension].header[p_name.upper()] = nn_results[i_s, i_p]
            hdulist.close()
