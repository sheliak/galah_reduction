import os
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Activation
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from astropy.table import Table, join, unique, vstack
from glob import glob
from copy import deepcopy
from spectra_collection_functions import *

import argparse
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import tensorflow as T
import json
import joblib

# -------------------------------------------------------------
# ----------------- Aditional functions -----------------------
# -------------------------------------------------------------
def custom_error_function(y_true, y_pred):
    bool_finite = T.math.is_finite(y_true)
    mse = K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=-1)
    return K.sum(mse)


def custom_error_function_2(y_true, y_pred):
    # VERSION1 - not sure if indexing and axis value are correct in this way
    # NOTE: boolean_mask reduces the dimensionality of the matrix, therefore the loss is prevailed by parameters with more observations
    # bool_finite = T.math.is_finite(y_true)
    # mse = K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=0)
    # return K.sum(mse)
    # VERSION2 - same thing, but using more understandable, but probably a bit slower for loop
    mse_final = 0
    for i1 in range(K.int_shape(y_pred)[1]):
        v1 = y_pred[:, i1]
        v2 = y_true[:, i1]
        bool_finite = T.math.is_finite(v2)
        mse_final += K.mean(K.square(T.boolean_mask(v1, bool_finite) - T.boolean_mask(v2, bool_finite)))
    return mse_final


def custom_error_function_3(y_true, y_pred):
    bool_finite = T.math.is_finite(y_true)
    mae = K.mean(K.abs(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=0)
    return K.sum(mae)


def custom_error_function_4(y_true, y_pred):
    bool_finite = T.math.is_finite(y_true)
    log_cosh = K.mean(K.log(K.cosh(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite))), axis=0)
    return K.sum(log_cosh)


def custom_error_function_5(y_true, y_pred):
    bool_finite = T.math.is_finite(y_true)
    mse = K.mean(K.square(T.boolean_mask(y_pred, bool_finite) - T.boolean_mask(y_true, bool_finite)), axis=0)
    mse_final = K.sum(mse)
    for i1 in range(K.int_shape(y_pred)[1]):
        for i2 in range(i1+1, K.int_shape(y_pred)[1]):
            v1 = y_pred[:, i1] * y_pred[:, i2]
            v2 = y_true[:, i1] * y_true[:, i2]
            bool_finite = T.math.is_finite(v2)
            mse_final += K.mean(K.square(T.boolean_mask(v1, bool_finite) - T.boolean_mask(v2, bool_finite)))
    return mse_final


def read_spectra(line_list, complete_spectrum=False, get_elements=None, read_wvl_offset=0.2, add_fe_lines=False):  # in A

    read_elements = deepcopy(get_elements)
    read_elements.append('Fe')

    idx_list = np.in1d(line_list['Element'], read_elements, assume_unique=False)
    line_list_read = line_list[idx_list]
    
    wvl_data = list([])
    spectral_data = list([])
    for i_band in range(4):
        spectra_file = 'galah_iraf6_ccd{:1.0f}_{:4.0f}_{:4.0f}_wvlstep_{:01.3f}.pkl'.format(i_band + 1, min_wvl[i_band], max_wvl[i_band], step_wvl[i_band])

        # determine what is to be read from the spectra
        logging.info('Defining cols to be read for ccd{:d}'.format(i_band + 1))
        spectra_file_split = spectra_file[:-4].split('_')
        wvl_values = np.arange(float(spectra_file_split[3]), float(spectra_file_split[4]), float(spectra_file_split[6]))

        if not complete_spectrum:
            abund_cols_read = list([])
            for line in line_list_read:
                idx_wvl = np.logical_and(wvl_values >= line['line_start'] - read_wvl_offset,
                                         wvl_values <= line['line_end'] + read_wvl_offset)
                if np.sum(idx_wvl) > 0:
                    abund_cols_read.append(np.where(idx_wvl)[0])
            abund_cols_read = np.unique(np.hstack(abund_cols_read))  # unique instead of sort to remove duplicated wvl pixels
        else:
            abund_cols_read = np.where(np.logical_and(wvl_values >= min_wvl[i_band],
                                                      wvl_values <= max_wvl[i_band]))[0]
        logging.info('Number of wavelength bins: {:d}'.format(len(abund_cols_read)))
        # do the actual reading oread_wvl_offsetf spectra

        logging.info('Reading spectra file: ' + spectra_file)
        wvl_data.append(wvl_values[abund_cols_read])
        spectral_data.append(read_pkl_spectra(spectra_file, read_rows=None, read_cols=abund_cols_read))

    wvl_data = np.hstack(wvl_data)
    spectral_data = np.hstack(spectral_data)
    logging.info(spectral_data.shape)
    return spectral_data, wvl_data


def bias(f1, f2):
    diff = f1 - f2
    if np.sum(np.isfinite(diff)) == 0:
        return np.nan
    else:
        return np.nanmedian(diff)


def rmse(f1, f2):
    diff = f1 - f2
    n_nonna = np.sum(np.isfinite(diff))
    if n_nonna == 0:
        return np.nan
    else:
        return np.sqrt(np.nansum(diff**2)/n_nonna)


# -------------------------------------------------------------
# ----------------- Preset constants --------------------------
# -------------------------------------------------------------
# algorithm settings - inputs, outputs etc
save_models = True
output_results = True
output_plots = True
save_fits_predictions = True

# data training and handling
abund_positive_offset = 0.

# ann settings
dropout_learning = False
dropout_rate = 0.1
dropout_learning_c = False
dropout_rate_c = 0.
use_regularizer = False
activation_function = None # 'relu'  # dense layers - if set to None defaults to PReLu
activation_function_c = None # 'relu'  # 'relu'  # None  # convolution layers - if set to None defaults to PReLu

# convolution layer 1
C_f_1 = 5  # number of filters
C_k_1 = 7  # size of convolution kernel
C_s_1 = 1  # strides value
P_s_1 = 3  # size of pooling operator
# convolution layer 2
C_f_2 = 5
C_k_2 = 5
C_s_2 = 1
P_s_2 = 3
# convolution layer 3
C_f_3 = 0
C_k_3 = 5
C_s_3 = 1
P_s_3 = 3
n_dense_nodes = [200, 150, 100, 1]  # the last layer is output, its size will be determined on the fly

min_wvl = list([4710, 5640, 6475, 7700])
max_wvl = list([4910, 5880, 6745, 7895])
step_wvl = np.array([0.04, 0.05, 0.06, 0.07])


# -------------------------------------------------------------
# ----------------- Main function -----------------------------
# -------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("reduction", help="Full path to the fits file with the combined reduction database (dr6.0.fits or simillar) from the Iraf reduction pipeline.")
    parser.add_argument("parameters", help="Full path to the fits file with stellar parameters (GALAH+ DR3 or simillar) that will be used for training the CNN architecture.")
    args = parser.parse_args()

    log_file = 'cnn_train_network.log'
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=log_file, level=logging.DEBUG)

    galah_obs_fits = args.reduction
    # read the list of reduced data
    if os.path.isfile(galah_obs_fits):
        galah_param = Table.read(galah_obs_fits)
        logging.info('Number of availible spectra: {:d}'.format(len(galah_param)))
    else:
        logging.error('Unable to read combined reduction table in fits format.')
        raise SystemExit

    # read the parameters file
    if os.path.isfile(args.parameters):
        abund_param = Table.read(args.parameters)
        # TODO: enable traning procedure with multiple sme values for the same sobject_id
        # TODO: temporary solution is to use onyl one the first entry
        abund_param = unique(abund_param, keys=['sobject_id'])
        logging.info('Number of availible parameters: {:d}'.format(len(abund_param)))
    else:
        logging.error('Unable to read file with stellar parameters and abundances.')
        raise SystemExit

    # read line list used during the SME run
    line_list_file = '../aux/' + 'GALAH_DR3_line_info_table.csv'
    if os.path.isfile(line_list_file):
        line_list = Table.read(line_list_file, format='ascii.csv')
        logging.info('Number of read line list entries: {:d}'.format(len(line_list)))
    else:
        logging.error('Unable to read file with stellar parameters and abundances.')
        raise SystemExit

    # create output folder
    output_dir = 'CNN_train'
    os.system('mkdir ' + output_dir)
    os.chdir(output_dir)

    # determine parameters to train on
    sme_abundances_list = [col for col in abund_param.colnames if '_fe' in col and len(col.split('_')) == 2 and len(col.split('_')[0]) <= 2]
    # sme_abundances_list = ['Na_fe', 'Al_fe', 'Si_fe', 'K_fe', 'Ca_fe', 'Ti_fe']
    sme_params = ['teff', 'logg', 'vbroad', 'vmic', 'fe_h', 'alpha_fe']

    # select only abundances with some datapoints
    sme_abundances_list = [col for col in sme_abundances_list if np.sum(np.isfinite(abund_param[col])) > 100]
    sme_abundances_list = [col for col in sme_abundances_list if len(col.split('_')[0]) <= 3]

    # Parameters flagging procedure
    # Only flag_sp and flag_fe_h are taken into account. Individual abundance flags are currentlly not considered for selection
    logging.info('Applying various flags to training set')
    abund_param = abund_param[np.logical_and(abund_param['flag_fe_h'] == 0, 
                                             abund_param['red_flag'] == 0)]
    abund_param = abund_param[np.logical_and(abund_param['vbroad'] <= 50, 
                                             abund_param['flag_sp'] == 0)]
    # TODO: SNR cuts if needed

    logging.info('SME Abundances: ' + ' '.join(sme_abundances_list))
    logging.info('Number of abundances: {:d}'.format(len(sme_abundances_list)))

    # read spectral data for selected abundance absorption lines
    read_elements = [elem.split('_')[0].capitalize() for elem in sme_abundances_list]
    spectral_data, wvl_data = read_spectra(line_list,
                                           complete_spectrum=False,
                                           get_elements=read_elements,
                                           add_fe_lines=True)
    n_wvl_total = spectral_data.shape[1]

    if spectral_data.shape[0] != len(galah_param):
        logging.error('Reduction file and spectral database do not match in the number of entries ({:d} and {:d}).'.format(spectral_data.shape[0], len(galah_param)))
        raise SystemExit

    # export used wavelength bins
    np.savetxt('model_wvl.dat', wvl_data, '%11.5f')

    # determine and filter out spectral entries with nan only rows - no spectral data was added to the database
    # additionaly remove entries that were flagged during reduction precedure
    idx_bad_rows = np.logical_or(np.logical_not(np.isfinite(spectral_data).any(axis=1)),
                                 galah_param['reduction_flags'] > 0)
    logging.info('Removing {:d} spectral lines with no spectral information or reduction flag > 0.'.format(np.sum(idx_bad_rows)))
    spectral_data = spectral_data[np.logical_not(idx_bad_rows), :]
    galah_param = galah_param[np.logical_not(idx_bad_rows)]
    logging.info(spectral_data.shape)

    # somehow handle cols with nan values, delete cols or fill in data
    idx_bad_spectra = np.where(np.logical_not(np.isfinite(spectral_data)))
    n_bad_spectra = len(idx_bad_spectra[0])
    if n_bad_spectra > 0:
        logging.info('Correcting {:d} bad flux values in read spectra.'.format(n_bad_spectra))
        spectral_data[idx_bad_spectra] = 1.  # remove nan values with theoretical continuum flux value

    # normalize data (flux at every wavelength)
    # shift normalized spectra from the 1...0 range to 0...1, where 0 is new flux level
    logging.info('Shifting flux levels into PRelu area')
    spectral_data = 1. - spectral_data

    # prepare spectral data for the further use in the Keras library
    # only needed when using deep convolutional network
    if (C_f_1 + C_f_2 + C_f_3) > 0:    
        spectral_data = np.expand_dims(spectral_data, axis=2)

    # -------------------------------------------------------------
    # ----------------- Determine train set -----------------------
    # -------------------------------------------------------------
    # final set of parameters
    galah_param_complete = join(galah_param['sobject_id', 'ra', 'dec'],
                                abund_param[list(np.hstack(('sobject_id', sme_abundances_list, sme_params)))],
                                keys='sobject_id', join_type='left')
    # replace/fill strange masked (--) values with np.nan
    for c_col in galah_param_complete.colnames[1:]:   # 1: to skis sobject_id column
        if sys.version_info[0] == 2:
            galah_param_complete[c_col] = galah_param_complete[c_col].filled(np.nan)

    # select only rows with valid parameters and spectra data
    logging.info('Size complete: ' + str(len(galah_param_complete)))

    sme_abundance = list(sme_abundances_list)
    additional_train_feat = list(sme_params)

    logging.info('Working on multiple abundances: ' + ' '.join(sme_abundance))
    elements = [sme.split('_')[0] for sme in sme_abundance]
    output_col = [elem + '_abund_ann' for elem in elements]
    # use spectra that have at least one abundance meassurement, bad parameters were already filtered out during data reading
    idx_abund_rows = np.isfinite(abund_param[sme_abundance].to_pandas().values).any(axis=1)

    # determine size of the final node
    n_dense_nodes[-1] = len(sme_abundance) + len(additional_train_feat)  # 3 outputs for stellar physical parameters

    if np.sum(idx_abund_rows) < 10000:
        logging.error('Not enough train ({:d}) data to continue with training procedure.'.format(np.sum(idx_abund_rows)))
        raise SystemExit

    plot_suffix = ''

    param_joined = join(galah_param['sobject_id', 'ra', 'dec'],
                        abund_param[list(np.hstack(('sobject_id', sme_abundance, additional_train_feat)))][idx_abund_rows],
                        keys='sobject_id', join_type='inner')

    # determine final usable spectra and parameter values
    idx_spectra_train = np.in1d(galah_param['sobject_id'], param_joined['sobject_id'])
    abund_values_train = param_joined[list(np.hstack((sme_abundance, additional_train_feat)))].to_pandas().values

    abund_normalizer_file = 'normalizer_abund_'+'_'.join(elements)+'.pkl'
    n_train_feat = abund_values_train.shape[1]
    if os.path.isfile(abund_normalizer_file):
        logging.info('Reading normalization parameters')
        train_feat_mean, train_feat_std = joblib.load(abund_normalizer_file)
    else:
        logging.info('Normalizing input train parameters')
        train_feat_mean = np.zeros(n_train_feat)
        train_feat_std = np.zeros(n_train_feat)
        for i_f in range(n_train_feat):
            train_feat_mean[i_f] = np.nanmedian(abund_values_train[:, i_f])
            train_feat_std[i_f] = np.nanstd(abund_values_train[:, i_f])
        joblib.dump([train_feat_mean, train_feat_std], abund_normalizer_file, protocol=2)

    for i_f in range(n_train_feat):
        abund_values_train[:, i_f] = (abund_values_train[:, i_f] - train_feat_mean[i_f]) / train_feat_std[i_f] + abund_positive_offset
        p_par = list(np.hstack((sme_abundance, additional_train_feat)))
        p_val = abund_values_train[:, i_f]
        p_h_range = np.nanpercentile(p_val, [1, 99])
        plt.hist(p_val, range=p_h_range, bins=50)
        plt.savefig('train_norm_' + p_par[i_f] + '.png', dpi=200)
        plt.close()

    # export the list of parameters that will be predicted
    np.savetxt('model_out.dat', np.hstack((sme_abundance, additional_train_feat)), '%s')

    n_train_sme = np.sum(idx_spectra_train)
    logging.info('Number of train objects: ' + str(n_train_sme))
    spectral_data_train = spectral_data[idx_spectra_train]

    # -------------------------------------------------------------
    # ----------------- Create CNN architecture -------------------
    # -------------------------------------------------------------

    # set up regularizer if needed
    if use_regularizer:
        w_reg = regularizers.l1(1e-5)
        a_reg = regularizers.l1(1e-5)
    else:
        # default values for Conv1D and Dense layers
        w_reg = None
        a_reg = None

    # nn network - fully connected layers
    if (C_f_1 + C_f_2 + C_f_3) > 0:
        ann_input = Input(shape=(spectral_data_train.shape[1], 1), name='Input_'+plot_suffix)
    else:
        ann_input = Input(shape=(spectral_data_train.shape[1],), name='Input_'+plot_suffix)
    ann = ann_input
    # first cnn feature extraction layer
    if C_f_1 > 0:
        ann = Conv1D(C_f_1, C_k_1, activation=activation_function_c, padding='same', name='C_1', strides=C_s_1,
                     kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
        if activation_function_c is None:
            ann = PReLU(name='R_1')(ann)
        ann = MaxPooling1D(P_s_1, padding='same', name='P_1')(ann)
        if dropout_learning_c and dropout_rate_c > 0:
            ann = Dropout(dropout_rate_c, name='D_1')(ann)
    # second cnn feature extraction layer
    if C_f_2 > 0:
        ann = Conv1D(C_f_2, C_k_2, activation=activation_function_c, padding='same', name='C_2', strides=C_s_2,
                     kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
        if activation_function_c is None:
            ann = PReLU(name='R_2')(ann)
        ann = MaxPooling1D(P_s_2, padding='same', name='P_2')(ann)
        if dropout_learning_c and dropout_rate_c > 0:
            ann = Dropout(dropout_rate_c, name='D_2')(ann)
    # third cnn feature extraction layer
    if C_f_3 > 0:
        ann = Conv1D(C_f_3, C_k_3, activation=activation_function_c, padding='same', name='C_3', strides=C_s_3,
                     kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
        if activation_function_c is None:
            ann = PReLU(name='R_3')(ann)
        ann = MaxPooling1D(P_s_3, padding='same', name='P_3')(ann)
        if dropout_learning_c and dropout_rate_c > 0:
            ann = Dropout(dropout_rate_c, name='D_3')(ann)

    if (C_f_1 + C_f_2 + C_f_3) > 0:
        # flatter output from convolutional network to the shape useful for fully-connected dense layers
        ann = Flatten(name='Conv_to_Dense')(ann)

    # fully connected layers
    for n_nodes in n_dense_nodes:
        ann = Dense(n_nodes, activation=activation_function, name='Dense_'+str(n_nodes),
                    kernel_regularizer=w_reg, activity_regularizer=a_reg)(ann)
        # add activation function to the layer
        if n_nodes > 50:
            # internal fully connected layers in ann network
            if dropout_learning and dropout_rate > 0:
                ann = Dropout(dropout_rate, name='Dropout_'+str(n_nodes))(ann)
            if activation_function is None:
                ann = PReLU(name='PReLU_' + str(n_nodes))(ann)
        else:
            # output layer
            ann = Activation('linear')(ann)
            # ann = PReLU(name='PReLU_' + str(n_nodes))(ann)

    abundance_ann = Model(ann_input, ann)
    selected_optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    # selected_optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
    # selected_optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, decay=0.0)
    abundance_ann.compile(optimizer=selected_optimizer, loss=custom_error_function_2)
    abundance_ann.summary()

    metwork_weights_file = 'model_nn_weights_last.h5'
    metwork_weights_file_best = 'model_nn.h5'
    if os.path.isfile(metwork_weights_file_best):
        save_fits = False
        logging.info('Reading NN weighs - CAE')
        abundance_ann.load_weights(metwork_weights_file_best, by_name=True)
    else:
        # define early stopping callback
        earlystop = EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='auto')
        checkpoint = ModelCheckpoint('model_nn_weights_{epoch:03d}-{loss:.3f}-{val_loss:.3f}.h5',
                                     monitor='val_loss', verbose=0, save_best_only=False,
                                     save_weights_only=True, mode='auto', period=1)
        # fit the NN model
        # split into train and validation set
        vals_split_per = 10.  # percent, randomly generate validation set for every run
        idx_val = np.in1d(np.arange(n_train_sme),
                          np.random.choice(np.arange(n_train_sme), int(n_train_sme*vals_split_per/100.), replace=False))
        ann_fit_hist = abundance_ann.fit(spectral_data_train[~idx_val, :], abund_values_train[~idx_val, :],
                                         epochs=250,
                                         batch_size=int(np.ceil(n_train_sme*(100-vals_split_per)/100./12)),  # batches must be keept quite large to reduce overfitting
                                         shuffle=True,
                                         callbacks=[earlystop, checkpoint],
                                         validation_data=(spectral_data_train[idx_val, :], abund_values_train[idx_val, :]),
                                         verbose=2)

        i_best = np.argmin(ann_fit_hist.history['val_loss'])
        plt.plot(ann_fit_hist.history['loss'], label='Train')
        plt.plot(ann_fit_hist.history['val_loss'], label='Validation')
        plt.axvline(i_best, color='black', ls='--', alpha=0.5, label='Best val_loss')
        plt.title('Model accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.ylim(0., np.nanpercentile(ann_fit_hist.history['val_loss'], 90))
        plt.tight_layout()
        plt.legend()
        plt.savefig('model_nn_loss.png', dpi=250)
        plt.close()

        last_loss = ann_fit_hist.history['loss'][-1]
            
        save_model_last = False
        if save_model_last:
            logging.info('Saving NN weighs')
            abundance_ann.save_weights(metwork_weights_file, overwrite=True)

        # recover weights of the best model and compute predictions
        h5_weight_files = glob('model_nn_weight_{:03.0f}-*-*.h5'.format(i_best + 1))
        if len(h5_weight_files) == 1:
            logging.info('Restoring epoch {:.0f} with the lowest validation loss ({:.3f}).'.format(i_best + 1, ann_fit_hist.history['val_loss'][i_best]))
            abundance_ann.load_weights(h5_weight_files[0], by_name=True)
            # delete all other h5 files that were not used and may occupy a lot of hdd space
            for h5_file in glob('model_nn_weights_*-*-*.h5'):
                if h5_file != h5_weight_files[0]:
                    os.system('rm ' + h5_file)
        else:
            logging.info('The last model will be used to compute predictions.')

        logging.info('Saving model for GALAH pipeline')
        abundance_ann.save_weights('model_nn_weights.h5', overwrite=True)

        # save model architecture in HDF5 and JSON fomats - the following can be used in further loading of the architecture
        abundance_ann.save('model_nn.h5', overwrite=True)
        model_json = abundance_ann.to_json()
        with open('model_nn.json', "w") as json_file:
            json.dump(model_json, json_file)

    # evaluate on all spectra
    logging.info('Predicting abundance values from spectra')
    abundance_predicted = abundance_ann.predict(spectral_data)

    logging.info('Denormalizing output values of features')
    # 
    for i_f in range(n_train_feat):
        abundance_predicted[:, i_f] = (abundance_predicted[:, i_f] - abund_positive_offset) * train_feat_std[i_f] + train_feat_mean[i_f]

    # add parameters to the output table
    for s_p in additional_train_feat:
        output_col.append(s_p + '_ann')
    # add abundance results to the final table
    for i_o in range(len(output_col)):
        galah_param_complete[output_col[i_o]] = abundance_predicted[:, i_o]

    # scatter plot of results to the reference cannon and sme values
    sme_abundances_plot = sme_abundance

    # sme_abundances_plot = np.hstack((sme_abundance, additional_train_feat))
    logging.info('Plotting graphs')
    for plot_abund in sme_abundances_plot:
        logging.info(' plotting attribute - ' + plot_abund)
        elem_plot = plot_abund.split('_')[0]
        # determine number of lines used for this element
        n_lines_element = np.sum(line_list['Element'] == elem_plot.capitalize())
        graphs_title = elem_plot.capitalize() + ' - SME train objects: ' + str(np.sum(np.isfinite(abund_param[plot_abund]))) + ' (lines: ' + str(n_lines_element) + ') - BIAS: {:.2f}   RMSE: {:.2f}'.format(bias(galah_param_complete[plot_abund], galah_param_complete[elem_plot+'_abund_ann']), rmse(galah_param_complete[plot_abund], galah_param_complete[elem_plot+'_abund_ann']))
        plot_range = (np.nanpercentile(abund_param[plot_abund], 1), np.nanpercentile(abund_param[plot_abund], 99))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[plot_abund], galah_param_complete[elem_plot+'_abund_ann'],
                    lw=0, s=0.25, alpha=0.20, c='black')
        plt.title(graphs_title)
        plt.xlabel('SME reference value')
        plt.ylabel('CNN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(elem_plot+'_ANN_sme_corr_'+plot_suffix+'.png', dpi=300)
        plt.close()

    for plot_abund in sme_abundances_plot:
        logging.info(' plotting attribute - ' + plot_abund)
        elem_plot = plot_abund.split('_')[0]
        # determine number of lines used for this element
        idx_plot = np.in1d(galah_param_complete['sobject_id'],
                           abund_param['sobject_id'][abund_param['flag_sp'] == 0])
        abund_range = (np.nanpercentile(abund_param[plot_abund], 1), np.nanpercentile(abund_param[plot_abund], 99))
        feh_range = (np.nanpercentile(abund_param['fe_h'], 1), np.nanpercentile(abund_param['fe_h'], 99))
        # first scatter graph - train points
        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        ax[0].scatter(galah_param_complete['fe_h'][idx_plot],
                      galah_param_complete[plot_abund][idx_plot],
                      lw=0, s=0.25, alpha=0.20, c='black')
        ax[1].scatter(galah_param_complete['fe_h_ann'][idx_plot],
                      galah_param_complete[elem_plot+'_abund_ann'][idx_plot],
                      lw=0, s=0.25, alpha=0.20, c='black')
        ax[2].scatter(galah_param_complete['fe_h_ann'],
                      galah_param_complete[elem_plot + '_abund_ann'],
                      lw=0, s=0.25, alpha=0.20, c='black')

        ax[0].set(xlim=feh_range, ylim=abund_range, xlabel='SME [Fe/H]', ylabel='SME '+elem_plot, title='Stars with flag_sp = 0 and valid abundance')
        ax[1].set(xlim=feh_range, ylim=abund_range, xlabel='CNN [Fe/H]', ylabel='CNN '+elem_plot, title='Stars with flag_sp = 0')
        ax[2].set(xlim=feh_range, ylim=abund_range, xlabel='CNN [Fe/H]', ylabel='CNN '+elem_plot, title='All stars, no flagging')

        ax[0].grid(alpha=0.10, color='black', ls='--')
        ax[1].grid(alpha=0.10, color='black', ls='--')
        ax[2].grid(alpha=0.10, color='black', ls='--')

        plt.tight_layout()
        plt.savefig(elem_plot+'_ANN_sme_'+plot_suffix+'.png', dpi=350)
        plt.close(fig)

    for plot_param in sme_params:
        logging.info(' plotting parameter - ' + plot_param)
        param_plot = plot_param  # .split('_')[0]
        # determine number of lines used for this element
        graphs_title = param_plot.capitalize() + ' - SME train objects: ' + str(np.sum(np.isfinite(abund_param[plot_abund]))) + ' - BIAS: {:.2f}   RMSE: {:.2f}'.format(bias(galah_param_complete[param_plot], galah_param_complete[param_plot+'_ann']), rmse(galah_param_complete[param_plot], galah_param_complete[param_plot+'_ann']))
        plot_range = (np.nanpercentile(abund_param[plot_param], 0.1), np.nanpercentile(abund_param[plot_param], 99.9))
        # first scatter graph - train points
        plt.plot([plot_range[0], plot_range[1]], [plot_range[0], plot_range[1]], linestyle='dashed', c='red', alpha=0.5)
        plt.scatter(galah_param_complete[param_plot], galah_param_complete[param_plot+'_ann'],
                    lw=0, s=0.25, alpha=0.20, c='black')
        plt.title(graphs_title)
        plt.xlabel('SME reference value')
        plt.ylabel('CNN computed value')
        plt.xlim(plot_range)
        plt.ylim(plot_range)
        plt.savefig(param_plot+'_ANN_sme_'+plot_suffix+'.png', dpi=300)
        plt.close()

    # Plot both H-R diagrams alongside
    logging.info(' plotting HR diagrams')
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(galah_param_complete['teff'], galah_param_complete['logg'],
                  lw=0, s=0.25, alpha=0.20, c='black')
    ax[1].scatter(galah_param_complete['teff_ann'], galah_param_complete['logg_ann'],
                  lw=0, s=0.25, alpha=0.20, c='black')
    ax[0].set(xlabel='SME Teff', ylabel='SME logg', xlim=[8000, 3000], ylim=[5.5, 0], title='All stars with parameters')
    ax[1].set(xlabel='CNN Teff', ylabel='CNN logg', xlim=[8000, 3000], ylim=[5.5, 0], title='All stars, no limitation')
    ax[0].grid(alpha=0.10, color='black', ls='--')
    ax[1].grid(alpha=0.10, color='black', ls='--')
    plt.tight_layout()
    plt.savefig('HR_ANN_sme_' + plot_suffix + '.png', dpi=350)
    plt.close(fig)

    idx_plot = np.in1d(galah_param_complete['sobject_id'], abund_param['sobject_id'][abund_param['flag_sp'] == 0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(galah_param_complete['teff'][idx_plot], galah_param_complete['logg'][idx_plot],
                  lw=0, s=0.25, alpha=0.20, c='black')
    ax[1].scatter(galah_param_complete['teff_ann'][idx_plot], galah_param_complete['logg_ann'][idx_plot],
                  lw=0, s=0.25, alpha=0.20, c='black')
    ax[0].set(xlabel='SME Teff', ylabel='SME logg', xlim=[8000, 3000], ylim=[5.5, 0], title='Stars with flag_sp = 0')
    ax[1].set(xlabel='CNN Teff', ylabel='CNN logg', xlim=[8000, 3000], ylim=[5.5, 0], title='Stars with flag_sp = 0')
    ax[0].grid(alpha=0.10, color='black', ls='--')
    ax[1].grid(alpha=0.10, color='black', ls='--')
    plt.tight_layout()
    plt.savefig('HR_ANN_sme_flag0_' + plot_suffix + '.png', dpi=350)
    plt.close(fig)

    # also save results (predictions) at the end
    if save_fits_predictions:
        fits_out = 'galah_abund_ANN_DR3.fits'
        galah_param_complete.write(fits_out, overwrite=True)   
