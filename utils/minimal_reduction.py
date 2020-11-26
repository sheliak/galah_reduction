import logging
import sys
import os
import argparse
from astropy.io import fits
import shutil
import numpy as np
import glob
import imp
sys.path.append('..')
extract = imp.load_source('extract', 'extract6.0.py')
from pyraf import iraf

def image_change_run(image, d):
	"""
	Change run numbe rof image for d
	"""
	image_root=image[:6]
	image_run=int(image[7:10])
	new_run=image_run+d
	new_image=image_root+str(new_run).zfill(4)+'.fits'
	return new_image

if __name__ == "__main__":
	# logging settings
	consoleLog = logging.StreamHandler()
	consoleLog.setLevel(logging.DEBUG)
	minimal_reduction_log=logging.getLogger('minimal_reduction_log')	
	minimal_reduction_log.setLevel(logging.INFO)
	minimal_reduction_log.addHandler(consoleLog)

	# Set iraf variables that are otherwise defined in local login.cl. Will probably overwrite those settings
	start_folder = os.getcwd()
	iraf.set(min_lenuserarea=128000)
	iraf.set(uparm=start_folder + '/uparm')

	# command line arguments
	parser=argparse.ArgumentParser()
	parser.add_argument("image", help="Path to the image with spectra to be reduced and extracted.")
	parser.add_argument("--all_ccds", help="Reduce and extract spectra from all CCDs, regardless the ccd given in image argument.", action="store_true")
	args=parser.parse_args()

	# set paths and such
	image=args.image.split('/')[-1]
	image_folder='/'.join(args.image.split('/')[:-1])+'/'
	run=int(args.image.split('/')[-1][-9:-5])
	date=args.image.split('/')[-4]
	cob=date+str(run).zfill(4)
	ccd=args.image.split('/')[-1][-10]

	# check if this is a valid image (not MFFLAT, DARK, MFARC, BIAS)
	hdul = fits.open(image_folder+image)
	ndfclass = hdul[0].header['NDFCLASS']
	cfg_file=hdul[0].header['CFG_FILE']
	plate=hdul[0].header['SOURCE']
	hdul.close()

	if ndfclass in ['MFFLAT', 'DARK', 'MFARC', 'BIAS', 'MFFFF']:
		minimal_reduction_log.error('Image %s is not a valid science image. It has NDFCLASS=%s.' % (image, ndfclass))
		sys.exit(0)

	# find flat and arc
	# check if there are flat and arc images made with the same fibre configuration. Start searching with nearest runs
	# if flat and arc are not found, they can also be made with the same plate, but wrong fibre configurations
	flat=None
	arc=None
	flat_backup=None
	arc_backup=None
	for d in range(1,1000):
		if os.path.exists(image_folder+image_change_run(image,-d)):
			hdul = fits.open(image_folder+image_change_run(image,-d))
			ndfclass_i = hdul[0].header['NDFCLASS']
			cfg_file_i=hdul[0].header['CFG_FILE']
			plate_i=hdul[0].header['SOURCE']
			hdul.close()
			if ndfclass_i=='MFFFF' and cfg_file_i==cfg_file and plate_i==plate:
				flat= image_folder+image_change_run(image,-d)
			if ndfclass_i=='MFARC' and cfg_file_i==cfg_file and plate_i==plate:
				arc= image_folder+image_change_run(image,-d)
			if ndfclass_i=='MFFFF' and plate_i==plate:
				flat_backup= image_folder+image_change_run(image,-d)
			if ndfclass_i=='MFARC' and plate_i==plate:
				arc_backup= image_folder+image_change_run(image,-d)

		if os.path.exists(image_folder+image_change_run(image,d)):
			hdul = fits.open(image_folder+image_change_run(image,d))
			ndfclass_i = hdul[0].header['NDFCLASS']
			cfg_file_i=hdul[0].header['CFG_FILE']
			plate_i=hdul[0].header['SOURCE']
			hdul.close()
			if ndfclass_i=='MFFFF' and cfg_file_i==cfg_file and plate_i==plate:
				flat= image_folder+image_change_run(image,d)
			if ndfclass_i=='MFARC' and cfg_file_i==cfg_file and plate_i==plate:
				arc= image_folder+image_change_run(image,d)
			if ndfclass_i=='MFFFF' and plate_i==plate:
				flat_backup= image_folder+image_change_run(image,d)
			if ndfclass_i=='MFARC' and plate_i==plate:
				arc_backup= image_folder+image_change_run(image,d)

		if flat is not None and arc is not None:
			break

	if flat is None:
		flat=flat_backup
	if arc is None:
		arc=arc_backup

	# terminate, if there are no flat or arc
	if flat is None:
		minimal_reduction_log.error('Image %s can not be reduced, because a valid flat field is not found' % (image))
	
	if arc is None:
		minimal_reduction_log.error('Image %s can not be reduced, because a valid arc field is not found' % (image))

	if flat is None or arc is None:
		sys.exit(0)

	# copy files into a reduction folder
	if not os.path.exists('reductions'):
		os.makedirs('reductions')

	if not os.path.exists('reductions/%s' % date):
		os.makedirs('reductions/%s' % date)

	if not os.path.exists('reductions/%s/ccd1' % date):
		os.makedirs('reductions/%s/ccd1' % date)
	if not os.path.exists('reductions/%s/ccd2' % date):
		os.makedirs('reductions/%s/ccd2' % date)
	if not os.path.exists('reductions/%s/ccd3' % date):
		os.makedirs('reductions/%s/ccd3' % date)
	if not os.path.exists('reductions/%s/ccd4' % date):
		os.makedirs('reductions/%s/ccd4' % date)

	if not os.path.exists('reductions/%s/ccd1/%s' % (date,cob)):
		os.makedirs('reductions/%s/ccd1/%s' % (date,cob))
	if not os.path.exists('reductions/%s/ccd2/%s' % (date,cob)):
		os.makedirs('reductions/%s/ccd2/%s' % (date,cob))
	if not os.path.exists('reductions/%s/ccd3/%s' % (date,cob)):
		os.makedirs('reductions/%s/ccd3/%s' % (date,cob))
	if not os.path.exists('reductions/%s/ccd4/%s' % (date,cob)):
		os.makedirs('reductions/%s/ccd4/%s' % (date,cob))

	shutil.copyfile(image_folder+image, 'reductions/%s/ccd%s/%s/%s' % (date,ccd,cob,image))
	shutil.copyfile(flat, 'reductions/%s/ccd%s/%s/%s' % (date,ccd,cob, flat.split('/')[-1]))
	shutil.copyfile(arc, 'reductions/%s/ccd%s/%s/%s' % (date,ccd,cob, arc.split('/')[-1]))

	#create fibre table
	hdul=fits.open('reductions/%s/ccd%s/%s/%s' % (date,ccd,cob,image))
	table_name='reductions/%s/ccd%s/%s/fibre_table_%s' % (date,ccd,cob,image)
	hdul['STRUCT.MORE.FIBRES'].writeto(table_name)
	hdul.close()

	# remove bias, and the regular pipeline will take it from there
	for f in glob.glob("reductions/%s/ccd%s/%s/[01-31]*.fits" % (date,ccd,cob)):
		with fits.open(f, mode='update') as hdu:
			hdu['Primary'].data=hdu['Primary'].data-np.transpose(np.ones([4146,4112])*np.mean(hdu['Primary'].data[:,4096:],axis=1))			
			hdu.flush()

	# do the rest of initial reduction
	extract.fix_gain(date)
	extract.fix_bad_pixels(date)

	# rename flat and arc
	shutil.copyfile('reductions/%s/ccd%s/%s/%s' % (date,ccd,cob, flat.split('/')[-1]), 'reductions/%s/ccd%s/%s/masterflat.fits' % (date,ccd,cob))
	shutil.copyfile('reductions/%s/ccd%s/%s/%s' % (date,ccd,cob, arc.split('/')[-1]), 'reductions/%s/ccd%s/%s/masterarc.fits' % (date,ccd,cob))
	os.remove('reductions/%s/ccd%s/%s/%s' % (date,ccd,cob, flat.split('/')[-1]))
	os.remove('reductions/%s/ccd%s/%s/%s' % (date,ccd,cob, arc.split('/')[-1]))

	# remove cosmics
	#extract.remove_cosmics(date)

	# extract spectra
	extract.find_apertures(date, start_folder)
	extract.extract_spectra(date, start_folder)
	extract.wav_calibration(date, start_folder)

	# plot results
	extract.plot_spectra(date, cob, ccd)