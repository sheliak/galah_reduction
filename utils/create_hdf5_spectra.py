"""
This script creates an HDF5 spectral database for all given nights. It will be saved in the root folder
where all reductions are located and it is given by the path argument. If HDF5 dababase is already
present, only missing spectra will be appended to it.

To do:
	Add parallelization to speed up writing if possible.

"""
import os
import logging
import sys
import numpy as np
import argparse
import h5py
from glob import glob
from astropy.table import Table
from astropy.io import fits
from time import time
from datetime import timedelta

logging.basicConfig(level=logging.DEBUG)

# read command line arguments
parser=argparse.ArgumentParser()
parser.add_argument("--date", help="Date you want to merge into the global DB (in format yymmdd).", default=None)
parser.add_argument("--all", help="Merge all nightly databases", action="store_true")
parser.add_argument("--path", help="Path to the folder where reducd nights are saved.", default='..')
args=parser.parse_args()

# check path format
if args.path[-1] == '/': 
	path = args.path
else: 
	path = args.path+'/'

dates = []
# create a list of dates that need their databases merged
if args.all:
	# create a list of all databases
	databases = glob(path+'*/db/*.fits')

	# turn this into a list of nights
	for database in databases:
		date = database[-11:-5]
		dates.append(date)

	dates_str = ', '.join(dates)
	logging.info('List of dates that were found to be added: ' + dates_str)

else:
	date = args.date
	if date == None:
		logging.error('Specify --date of use --all to combine all nightly databases. Use -h for help.')
		sys.exit(-1)

	if len(date) == 6 and 10 < int(date[:2]) < 30 and 1 <= int(date[2:4]) <= 12 and 1 <= int(date[4:6]) <= 31:
		pass
	else:
		logging.error('Date must be an intiger in format yymmdd.')
		sys.exit(-1)

	dates.append(date)

os.chdir(path)


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

		if ext_name == 'PRIMARY':
			ext_name = 'fluxed'

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
			if head_key != 'COMMENT' and head_key != '':
				ccd_data.create_dataset(head_key, 
										shape=(1,), 
										# dtype=np.float,
										data=fits_header[head_key])
		else:
			# skip entries with the same name
			pass

	fits_data.close()
	return True


logging.info('Creating an HDF5 spectral database - this might take some time')

os.chdir(path)
hdf5_targ = 'dr6.0_spectra.hdf5'
hdf5_f = h5py.File(hdf5_targ, 'a')

for date in dates:
	logging.info('Adding spectra for the date %s' % date)

	# list of spectra that are already written to the hdf5 structure
	in_hdf5 = np.array(hdf5_f.keys())

	# load the nightly db
	hdul_night = fits.open('%s/db/%s.fits' % (date,date))
	hdu_night = hdul_night[1]
	table_night = Table(hdu_night.data)
	hdul_night.close()

	# determine what still needs to be added to the hdf5 structure
	db_sobj_str = np.array(table_night['sobject_id'])
	add_to_hdf5 = db_sobj_str[np.in1d(db_sobj_str, in_hdf5, invert=True)]

	# add remaining spectra 
	time_s = time()
	for i_s, sobj_key in enumerate(add_to_hdf5):
		for ccd in range(1, 5):
			sucess = add_spectrum_to_hdf5(sobj_key, ccd, path)
			if not sucess:
				logging.warning('Could not add ccd %s of sobject_id %s' % (ccd, str(sobj_key)))

		if i_s % 250 == 0:
			d_time = time() - time_s
			finish_time = d_time / (i_s + 1) * (len(add_to_hdf5) - i_s - 1)
			logging.info('Estimated remaining time {:s} ({:.1f}%) for date {:s}'.format(str(timedelta(seconds=finish_time)), 100.*(i_s+1)/len(add_to_hdf5), date))

logging.info('HDF5 creation finished')
hdf5_f.close()
