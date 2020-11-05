"""
This script creates an HDF5 spectral database for all entries in the given input fits database. It will be saved alngoside the
original fits file. If one such dababase is already present, only missing spectra will be appended to it.

Before running this script, you should merge databases of individual nights by running merge_db script.

To do:
	Add parallelization to speed up writing if possible.

"""
import os
import logging
import sys
import numpy as np
import argparse
import h5py
from astropy.table import Table
from astropy.io import fits
from time import time
from datetime import timedelta

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("database", help="Full path to the fits database file that contains entries of reduced specctra.")
args = parser.parse_args()

fits_path = args.database

if fits_path[-4:] == 'fits':
	if os.path.isfile(fits_path):
		db_data = Table.read(fits_path)
		db_sobj = db_data['sobject_id']
		db_sobj_str = np.array([str(sid) for sid in db_sobj])
	else:
		logging.error('The given file %s does not exist' % fits_path)
		sys.exit(-1)
else:
	logging.error('Input must be a path to a Galah fits database file.')
	sys.exit(-1)

fits_dir = fits_path[:fits_path.rfind('/')]
spec_dir = fits_dir + '/results/'
logging.info('Combined spectra will be read from the folder %s' % spec_dir)

logging.info('Creating an HDF5 spectral database - this might take some time')
hdf5_targ = fits_dir + '/dr6.0_com.hdf5'
hdf5_f = h5py.File(hdf5_targ, 'a')

in_hdf5 = np.array(hdf5_f.keys())
add_to_hdf5 = db_sobj_str[np.in1d(db_sobj_str, in_hdf5, invert=True)]


def add_spectrum_to_hdf5(sobj_id, ccd_id, root):
	"""
	Add individual ccd spectral information to the final HDF5 spectral database.

	"""
	sobj_id_s = str(sobj_id)
	ccd_id_s = str(ccd_id)
	ccd_name = 'ccd' + ccd_id_s
	date = sobj_id_s[:6]

	# check if spectrum file exists
	fits_spec = root + date + '/spectra/com/' + sobj_id_s + ccd_id_s + '.fits'
	if os.path.isfile(fits_spec):
		fits_data = fits.open(fits_spec)
	else:
		return False

	# create the group for a given sobject_id
	if sobj_id_s not in hdf5_f.keys():
		sobj_data = hdf5_f.create_group(sobj_id_s)
	else:
		sobj_data = hdf5_f[sobj_id_s]

	# create the group for a given ccd
	if ccd_name not in sobj_data.keys():
		ccd_data = sobj_data.create_group(ccd_name)
	else:
		ccd_data = sobj_data[ccd_name]

	# copy spectral informations from all extensions of the fits file
	for ext in fits_data:
		ext_name = ext.name
		ext_data = ext.data
	
		if ext_name not in ccd_data.keys():
			ccd_data.create_dataset(ext_name, 
									shape=(len(ext_data),), 
									dtype=np.float32,
									data=ext_data, 
									compression=None)

	# copy metadata onyl from the extension 0 as they are mostlly repated in every extension
	fits_header = fits_data[0].header
	for head_key in fits_header.keys():
		if head_key not in ccd_data.keys():
			if head_key != 'COMMENT':
				ccd_data.create_dataset(head_key, 
										shape=(1,), 
										# dtype=np.float,
										data=fits_header[head_key])
		else:
			# skip entries with the same name
			pass

	fits_data.close()
	return True

time_s = time()
for i_s, sobj_key in enumerate(add_to_hdf5):
	for ccd in range(1, 5):
		sucess = add_spectrum_to_hdf5(sobj_key, ccd, spec_dir)
		if not sucess:
			logging.warning('Could not add ccd %s of sobject_id %s' % (ccd, sobj_key))

	if i_s % 250 == 0:
		d_time = time() - time_s
		finish_time = d_time / (i_s + 1) * (len(add_to_hdf5) - i_s - 1)
		logging.info('Estimated remaining time {:s} ({:.1f}%)'.format(str(timedelta(seconds=finish_time)), 100.*(i_s+1)/len(add_to_hdf5)))

logging.info('HDF5 creation finished')
hdf5_f.close()
