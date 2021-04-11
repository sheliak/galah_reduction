import os
import sys
sys.path.append("..")
from astropy.table import Table
import time, datetime
import numpy as np
import logging
import joblib
import argparse
from tools import galah_tools as gt
from spectra_collection_functions import *


# -------------------------------------------------------------
# ----------------- Preset constants --------------------------
# -------------------------------------------------------------

# Spectra will be resampled to the following ranges and bin steps
min_wvl = list([4710, 5640, 6475, 7700])
max_wvl = list([4910, 5880, 6745, 7895])
step_wvl = list([0.04, 0.05, 0.06, 0.07])  # more or less original values


# -------------------------------------------------------------
# ----------------- Main function--- --------------------------
# -------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the folder with the combined reduction database (dr6.0.fits) from the Iraf_6.0 reduction pipeline.")
    parser.add_argument("band", help="Input a single CCD number in the range 1-4 or a sequence of multiple bands 1,3.", default=1)
    args = parser.parse_args()

    # check and fix input arguments
    bands = [int(bb) for bb in args.band.split(',')]

    # logger
    log_file = 'cnn_prepare_spectra_bands-' + args.band + '.log'
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=log_file, level=logging.DEBUG)

    galah_data_dir = args.path
    if galah_data_dir[-1]!='/': galah_data_dir += '/'

    # read the list of reduced data
    galah_obs_fits = galah_data_dir + 'dr6.0.fits'
    if os.path.isfile(galah_obs_fits):
        general_data = Table.read(galah_obs_fits)
        logging.info('Number of availible spectra: {:d}'.format(len(general_data)))
    else:
        logging.error('Unable to read combined reduction table in fits format.')

    # create output folder
    output_dir = 'CNN_train'
    os.system('mkdir ' + output_dir)
    os.chdir(output_dir)

    # prepare galah tools for spectra reding
    gt.setup(root_folder=galah_data_dir)

    n_spec_ok = 0
    for ccd in bands:
        logging.info('Working on ccd {:d}'.format(ccd))

        target_wvl = min_wvl[ccd - 1] + np.float64(range(0, np.int32(np.ceil((max_wvl[ccd - 1] - min_wvl[ccd - 1]) / step_wvl[ccd - 1])))) * step_wvl[ccd - 1]

        # name of the final pkl file
        out_pkl_file = 'galah_iraf6_ccd{:1.0f}_{:4.0f}_{:4.0f}_wvlstep_{:01.3f}'.format(ccd, min_wvl[ccd - 1], max_wvl[ccd - 1], step_wvl[ccd - 1])

        # empty array that will store all resampled spectra for the selected ccd
        # float16 is used to conserve disk space as higher float accuracy is probably not needed at this point
        spectra_array = np.full((len(general_data), len(target_wvl)), np.nan, dtype='float16')

        time_s = time.time()

        for i_r, row in enumerate(general_data):
            if i_r % 1000 == 0:
                time_now = time.time()
                end_time = (time_now-time_s)/(i_r+1)*(len(general_data)-i_r-1)
                logging.info('Estimated finish in ' + str(datetime.timedelta(seconds=end_time)))

            spectrum_id = str(row['sobject_id']) + str(ccd)
            
            try:
                spectrum = gt.read(spectrum_id, linearize=False, wavelength='object', kind='norm')
            except:
                # logging.warning('Unable to retrieve spectrum {:s}.'.format(spectrum_id))
                continue

            if spectrum.rv == None:
                # do not use spectra without a valid RV value
                continue

            # TODO: resolution equalization if needed

            try:
                # resample read spectrum to the same grid
                new_spec_flx, idx_use = spectrum_resample(spectrum.f, spectrum.l, target_wvl)
            except:
                # logging.warning('Resampling problem for spectrum {:s}.'.format(spectrum_id))
                continue

            n_spec_ok += 1
            spectra_array[i_r, idx_use] = new_spec_flx
            
        logging.info('Number of successfully stored spectra: {:d}'.format(n_spec_ok))
        save_pkl_spectra(spectra_array, out_pkl_file + '.pkl')
