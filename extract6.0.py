import logging
import sys
import glob
from astropy.io import fits
from astropy.time import Time
from astropy.modeling import models, fitting, polynomial
from astropy.table import Table
import numpy as np
import os
import shutil
from pyraf import iraf
from matplotlib import *
from pylab import *
import matplotlib.transforms as transforms
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from collections import defaultdict
import csv
from scipy import optimize, signal, ndimage
from scipy.interpolate import griddata
from astropy.convolution import Gaussian1DKernel, convolve, CustomKernel
from lmfit import Minimizer, Parameters, report_fit
from multiprocessing import Pool
from datetime import datetime
import cosmics
import time
import ephem

# possible NDFCLASS values in human readable form
ndfclass_types={'MFFFF':'fibre flat', 'MFARC':'arc', 'MFOBJECT':'object', 'BIAS':'bias'}

# possible fibre one-letter values in human readable form
fibre_types={'S':'sky fibre', 'P':'positioned fibre', 'U':'parked fibre', 'N':'fibre not in use', 'F':'guiding fibre'}

start_folder=''


def print_cobs(cobs):
	"""
	Print COBs for a given night

	Parameters:
		cobs (array): numpy array of cobs
	
	Returns:
		none

	"""

	print 'The following images and COBs (continous observing blocks) will be reduced.\n'

	print "\033[1m"+'CCD 1:'+"\033[0;0m"+'\n' # weird strings are the beginning and end of the bold text (will not work outside linux/unix)
	for i in cobs[0]:
		for n,j in enumerate(i[1]):
			if n==0:
				print '   ', i[0],
			else:
				print '              ',
			print j[0].split('/')[-1], ndfclass_types[j[1]].ljust(10, ' '), j[3].ljust(34, ' '), j[2]
		print ''

	print '\n'+"\033[1m"+'CCD 2:'+"\033[0;0m"+'\n'
	for i in cobs[1]:
		for n,j in enumerate(i[1]):
			if n==0:
				print '   ', i[0],
			else:
				print '              ',
			print j[0].split('/')[-1], ndfclass_types[j[1]].ljust(10, ' '), j[3].ljust(34, ' '), j[2]
		print ''

	print '\n'+"\033[1m"+'CCD 3:'+"\033[0;0m"+'\n'
	for i in cobs[2]:
		for n,j in enumerate(i[1]):
			if n==0:
				print '   ', i[0],
			else:
				print '              ',
			print j[0].split('/')[-1], ndfclass_types[j[1]].ljust(10, ' '), j[3].ljust(34, ' '), j[2]
		print ''

	print '\n'+"\033[1m"+'CCD 4:'+"\033[0;0m"+'\n'
	for i in cobs[3]:
		for n,j in enumerate(i[1]):
			if n==0:
				print '   ', i[0],
			else:
				print '              ',
			print j[0].split('/')[-1], ndfclass_types[j[1]].ljust(10, ' '), j[3].ljust(34, ' '), j[2]
		print ''


def correct_ndfclass(hdul):
	"""
	NDFCLASS given in the header of the image can be wrong as a result of human error. Calculate the correct one.

	Parameters:
		hdul (object): hdu object of one image
	
	Returns:
		ndfclass (str)

	"""
	h_head = hdul[0].header
	orig_class = h_head['NDFCLASS']

	# check readout speed - all images have to be taken with the same readout speed, FAST is the usual
	if h_head['SPEED'] != 'FAST':
		return None

	# handle uncommon exposure classes
	if orig_class in ['MFFLX', 'LFLAT']:
		return orig_class

	# bias should always be determined correctly
	if orig_class == 'BIAS':
		return orig_class

	# check should always be determined correctly
	if orig_class == 'MFARC':
		return orig_class

	# get essential parameters on which the decision about a class will be based on
	ex_time = h_head['EXPOSED']
	flaps = h_head['FLAPS']
	tracking = h_head['TRACKING']
	t_ra = h_head['MEANRA']
	t_dec = h_head['MEANDEC']
	f_data = Table(hdul['STRUCT.MORE.FIBRES'].data)
	f_ra = np.rad2deg(np.median(f_data['RA']))
	f_dec = np.rad2deg(np.median(f_data['DEC']))
	t_coord = SkyCoord(ra=t_ra*u.deg, dec=t_dec*u.deg)
	f_coord = SkyCoord(ra=f_ra*u.deg, dec=f_dec*u.deg)

	# compute height of the sun above/bellow the horizon
	sun = ephem.Sun()
	obs = ephem.Observer()
	# lat and long must be string if you do not want troubles
	obs.lon, obs.lat, obs.elevation = str(h_head['LONG_OBS']), str(h_head['LAT_OBS']), h_head['ALT_OBS']
	obs.date = h_head['UTDATE'].replace(':', '/') + ' ' + h_head['UTSTART']
	# ephem assumes input time to in UTC
	sun.compute(obs)
	sun_alt = np.rad2deg(float(sun.alt))

	# check other science types - object and flats
	new_class = 'UNKNOWN'

	if flaps == 'OPEN':
		if tracking != 'TRACKING':
			# TODO: delineate between sflat or dflat
			if 0. > sun_alt > -8:
				new_class = 'SFLAT'
			else:
				new_class = 'DFLAT'
		else:
			if t_coord.separation(f_coord) < 1 * u.deg:
				new_class = 'MFOBJECT'
			else:
				if sun_alt > -8:
					new_class = 'SFLAT'
	else:
		# only bias, dark or flat images can be acquired with closed flaps
		if 'LAMPNAME' in h_head.keys():
			new_class = 'MFFFF'
		else:
			new_class = 'DARK'  # or bias or something else

	if new_class == 'UNKNOWN':
		print t_coord, f_coord, sun_alt, t_coord.separation(f_coord)

	return new_class


def prepare_dir(dir, str_range='*'):
	"""
	Open spectra from one night and print exposure types and fields. Create a folder structure in which the reduction will take place

	Parameters:
		date (str): date string (e.g. 190210)
		str_range (str): string with range of runs to be used (e.g.  "1,2,5-7,10")
	
	Returns:
		none

	To do:
		Compose COBs more reliably.
		Test option to only use some range of run_ids (1,2,3,4 or 15- or 12-16)

	"""

	def range_to_list(str_range):
		"""
		Gets string with ranges (e.g.  "1,2,5-7,10") and converts it into a list (e.g. [1,2,5,6,7,10]).
		"""
		temp = [(lambda sub: range(sub[0], sub[-1] + 1))(map(int, ele.split('-'))) for ele in str_range.split(', ')]
		return [b for a in temp for b in a]

	if str_range=='*':
		runs=range(10000)
	else:
		try:
			runs=range_to_list(str_range)
			runs=np.array(runs, dtype=int)
		except:
			logging.error('List of runs (%s) cannot be converted into a list. Check formatting.' % str_range)
			runs=range(10000)

	# function that checks if cob has all three types of science frames
	def _has_cob_all_frames(cob_files, cob_str, ccd_i):
		types_in_cob = list([])
		for cob_exp in cob_files:
			types_in_cob.append(cob_exp[1])
		if len(np.unique(types_in_cob)) == 3:
			return True
		else:
			logging.warning('CCD %d of COB %s is missing some science frames and will be skipped' % (ccd_i, cob_str))
			return False

	logging.info('Checking contents of the given folder.')

	if dir[-1]=='/': dir=dir[:-1]

	cobs_all=[]
	list_of_cob_ids=[]
	biases_all=[]
	comments_dict={}
	for ccd in [1,2,3,4]:
		# read list of all files
		cobs_ccd=[]
		files_all=glob.glob("%s/data/ccd_%s/*.fits" % (dir,ccd))
		# check obstatus
		obstatus_dict={}
		for line in open("%s/comments/comments_%s.txt" % (dir, dir.split('/')[-1])):
			if len(line)>0 and line[0]!='#' and line.split(',')>2 and line.split(',')[0]!='\n':
				l=line.split(',')
				obstatus_dict[int(l[0])]=[int(l[1]),l[-1].replace('\n','')]
		obstatus_dict=defaultdict(lambda:[1, ''], obstatus_dict)
		files=[]

		for file in np.sort(files_all):

			fits_name = file.split('/')[-1]
			try:
				run_num = int(fits_name[6:10])
			except ValueError as err:
				logging.warning('Skipping file %s as it has no valid run number in filename' % fits_name)
				continue

			# accept file if obstatus is >0. Also add comment from comments.txt into the header
			# also check if run is in the list of runs to be reduced
			if obstatus_dict[run_num][0]>0 and int(fits_name[6:10]) in runs:
				files.append(file)
				obs_comment = obstatus_dict[run_num][1].rstrip()  # rstrip will remove trailing whitespaces and \r characters
				comments_dict[run_num]=obs_comment

		files=np.array(files)

		if len(files)==0:
			logging.error('No data for CCD %s exists.' % ccd)
			cobs_all.append([])
		else:
			list_of_fits=[]
			list_of_biases=[]
			for f in files:
				hdul=fits.open(f)
				hdul_ndfclass = correct_ndfclass(hdul)
				# print f.split('/')[-1], hdul[0].header['NDFCLASS'], '->', hdul_ndfclass
				hdul.close()
				if hdul_ndfclass in ['MFFFF', 'MFARC', 'MFOBJECT']: list_of_fits.append([f, hdul_ndfclass, hdul[0].header['UTMJD'], hdul[0].header['CFG_FILE'], hdul[0].header['SOURCE']])
				if hdul_ndfclass == 'BIAS': list_of_biases.append([f, hdul_ndfclass, hdul[0].header['UTMJD'], hdul[0].header['CFG_FILE'], hdul[0].header['SOURCE']])

			# determine date from folder name
			date = dir.split('/')[-1]

			# check if we found any science image
			if len(list_of_fits) == 0:
				error_str = 'No science data found for night ' + date
				logging.error(error_str)
				raise RuntimeError(error_str)

			# order list of files in timely order
			list_of_fits=np.array(list_of_fits)
			list_of_fits=list_of_fits[list_of_fits[:,2].argsort()]

			# iterate over the list of files and break it into cobs
			cob_id=date+list_of_fits[0][0].split('/')[-1][6:10]
			list_of_cob_ids.append(cob_id)
			current_cob=cob_id
			current_field=list_of_fits[0][3]
			current_plate=list_of_fits[0][4]

			hdul = fits.open(list_of_fits[0][0])
			current_t_coord = SkyCoord(ra=hdul[0].header['MEANRA'] * u.deg,
									   dec=hdul[0].header['MEANDEC'] * u.deg)
			hdul.close()

			# t_ra =
			# t_dec =
			# f_data = Table(hdul['STRUCT.MORE.FIBRES'].data)
			# f_ra = np.rad2deg(np.median(f_data['RA']))
			# f_dec = np.rad2deg(np.median(f_data['DEC']))
			# t_coord = SkyCoord(ra=t_ra * u.deg, dec=t_dec * u.deg)
			# f_coord = SkyCoord(ra=f_ra * u.deg, dec=f_dec * u.deg)
			cobs=[]
			cob_content=[]
			for f in list_of_fits:

				hdul = fits.open(f[0])
				new_t_coord = SkyCoord(ra=hdul[0].header['MEANRA'] * u.deg,
									   dec=hdul[0].header['MEANDEC'] * u.deg)
				hdul.close()

				if f[3]!=current_field or f[4]!=current_plate or current_t_coord.separation(new_t_coord) > 1 * u.deg:
					if _has_cob_all_frames(cob_content, current_cob, ccd):
						cobs.append([current_cob,cob_content])
					date=dir.split('/')[-1]
					cob_id=date+f[0].split('/')[-1][6:10]
					list_of_cob_ids.append(cob_id)
					cob_content=[]
					current_cob=cob_id
					current_field=f[3]
					current_plate=f[4]
					current_t_coord = new_t_coord
					cob_content.append(f)
				else:
					cob_content.append(f)

			if _has_cob_all_frames(cob_content, current_cob, ccd):
				cobs.append([current_cob,cob_content])

			cobs_all.append(cobs)
			biases_all.append(list_of_biases)

	#check if the data is the same in all ccds
	cob_lens=[]
	for i in cobs_all:
		cob_lens_tmp=[]
		for j in i:
			cob_lens_tmp.append(len(j[1]))
		cob_lens.append(cob_lens_tmp)

	if cob_lens[0]==cob_lens[1]==cob_lens[2]==cob_lens[3]:
		pass
	else:
		logging.warning('There is a different number of images made with each CCD.')

	#create folder structure

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

	if not os.path.exists('reductions/%s/ccd1/biases' % date):
		os.makedirs('reductions/%s/ccd1/biases' % date)
	if not os.path.exists('reductions/%s/ccd2/biases' % date):
		os.makedirs('reductions/%s/ccd2/biases' % date)
	if not os.path.exists('reductions/%s/ccd3/biases' % date):
		os.makedirs('reductions/%s/ccd3/biases' % date)
	if not os.path.exists('reductions/%s/ccd4/biases' % date):
		os.makedirs('reductions/%s/ccd4/biases' % date)

	for i in set(list_of_cob_ids):
		if not os.path.exists('reductions/%s/ccd1/%s' % (date,i)):
			os.makedirs('reductions/%s/ccd1/%s' % (date,i))
		if not os.path.exists('reductions/%s/ccd2/%s' % (date,i)):
			os.makedirs('reductions/%s/ccd2/%s' % (date,i))
		if not os.path.exists('reductions/%s/ccd3/%s' % (date,i)):
			os.makedirs('reductions/%s/ccd3/%s' % (date,i))
		if not os.path.exists('reductions/%s/ccd4/%s' % (date,i)):
			os.makedirs('reductions/%s/ccd4/%s' % (date,i))

	# print the resulting structure
	print_cobs(cobs_all)

	# copy fits files into correct folders
	logging.info('Copying files into correct folders for the reduction.')

	for biases_all_ccd in biases_all:
		for bias_info in biases_all_ccd:
			bias_ccd = ''.join((bias_info[0].split('/')[-2]).split('_'))
			bias_filename = bias_info[0].split('/')[-1]
			shutil.copyfile(bias_info[0], 'reductions/%s/%s/biases/%s' % (date, bias_ccd, bias_filename))

	for i in cobs_all[0]:
		for j in i[1]:
			shutil.copyfile(j[0], 'reductions/%s/ccd1/%s/%s' % (date, i[0], j[0].split('/')[-1]))
	for i in cobs_all[1]:
		for j in i[1]:
			shutil.copyfile(j[0], 'reductions/%s/ccd2/%s/%s' % (date, i[0], j[0].split('/')[-1]))
	for i in cobs_all[2]:
		for j in i[1]:
			shutil.copyfile(j[0], 'reductions/%s/ccd3/%s/%s' % (date, i[0], j[0].split('/')[-1]))
	for i in cobs_all[3]:
		for j in i[1]:
			shutil.copyfile(j[0], 'reductions/%s/ccd4/%s/%s' % (date, i[0], j[0].split('/')[-1]))

	# add comment from comments file into headers
	files=glob.glob("reductions/%s/*/*/*.fits" % (date))
	comments_dict=defaultdict(lambda:'', comments_dict)
	for file in files:
		run_num=int(file.split('/')[-1][6:10])
		with fits.open(file, mode='update') as hdu_c:
			hdu_c[0].header['COMM_OBS']=(comments_dict[run_num], 'Comments by the observer')
			hdu_c.flush()

	# a list of files in the new destination must be returned
	cobs_out=cobs_all
	for n,i in enumerate(cobs_all[0]):
		for m,j in enumerate(i[1]):
			cobs_out[0][n][1][m][0]='reductions/%s/ccd1/%s/%s' % (date, i[0], j[0].split('/')[-1])
	for n,i in enumerate(cobs_all[1]):
		for m,j in enumerate(i[1]):
			cobs_out[1][n][1][m][0]='reductions/%s/ccd2/%s/%s' % (date, i[0], j[0].split('/')[-1])
	for n,i in enumerate(cobs_all[2]):
		for m,j in enumerate(i[1]):
			cobs_out[2][n][1][m][0]='reductions/%s/ccd3/%s/%s' % (date, i[0], j[0].split('/')[-1])
	for n,i in enumerate(cobs_all[3]):
		for m,j in enumerate(i[1]):
			cobs_out[3][n][1][m][0]='reductions/%s/ccd4/%s/%s' % (date, i[0], j[0].split('/')[-1])

	#create a list of fibres for object files and save them as a new fits file
	for ccd in [1,2,3,4]:
		for i in cobs_out[ccd-1]:
			flats=[]
			arcs=[]
			masterflat=[]
			masterarc=[]
			for j in i[1]:
				if j[1]=='MFOBJECT':
					f=j[0]
					hdul=fits.open(f)
					table_name='fibre_table_'+f.split('/')[-1]
					hdul['STRUCT.MORE.FIBRES'].writeto('/'.join(f.split('/')[:-1])+'/'+table_name)
					hdul.close()

	return date,cobs_out

def remove_bias(date):
	"""
	Bias is removed from all images. If there are not enough biases (less than 3) overscan is used.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none

	ToDo:
		Use overscan if there are no biases (or <3 biases)

	"""
	logging.info('Correcting bias.')
	for ccd in [1,2,3,4]:
		files=glob.glob("reductions/%s/ccd%s/biases/*.fits" % (date,ccd))
		
		if len(files)>=3:
			# create masterbias
			biases=[]
			for f in files:
				hdul=fits.open(f)
				biases.append(hdul[0].data)
				hdul.close()
			masterbias=np.median(np.array(biases), axis=0)

			# remove bias from images
			files=glob.glob("reductions/%s/ccd%s/*/[01-31]*.fits" % (date,ccd))
			for f in files:
				if 'biases' in f: 
					continue
				else:
					with fits.open(f, mode='update') as hdu_c:
						hdu_c[0].data=hdu_c[0].data-masterbias
						hdu_c.flush()
		else: # if there are not enough biases, use overscan
			files=glob.glob("reductions/%s/ccd%s/*/[01-31]*.fits" % (date,ccd))
			for f in files:
				if 'biases' in f: 
					continue
				else:
					with fits.open(f, mode='update') as hdu_c:
						pass
						hdu_c.flush()
						
		
		shutil.rmtree("reductions/%s/ccd%s/biases" % (date,ccd))

def fix_gain(date):
	"""
	Images are transfered through two ADCs with different gains. This function converts flux from ADU to electrons and hence corrects different gain of different ADCs.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none

	"""
	logging.info('Normalizing gain')
	for ccd in [1,2,3,4]:
		files=glob.glob("reductions/%s/ccd%s/*/[01-31]*.fits" % (date,ccd))
		for f in files:
			with fits.open(f, mode='update') as hdu_c:
				gain_top=hdu_c[0].header['RO_GAIN1']
				gain_bottom=hdu_c[0].header['RO_GAIN']
				y=hdu_c[0].header['NAXIS1']
				x=hdu_c[0].header['NAXIS2']
				gain_correction_factor=np.ones((x,y))
				gain_correction_factor[2057:,:]=gain_top
				gain_correction_factor[:2057,:]=gain_bottom
				hdu_c[0].data=hdu_c[0].data*gain_correction_factor
				hdu_c.flush

def fix_bad_pixels(date):
	"""
	CCDs 2 and 4 (green and IR) have bad columns. This is fixed so the bad columns are linearly interpolated from the neighbouring pixels.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none

	"""
	logging.info('Fixing bad columns.')

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.imred(_doprint=0,Stdout="/dev/null")
	iraf.ccdred(_doprint=0,Stdout="/dev/null")
	iraf.ccdred.instrum='aux/blank.txt'

	files=glob.glob("reductions/%s/ccd2/*/[01-31]*.fits" % (date))
	for f in files:
		iraf.ccdproc(images=f, output='', ccdtype='', fixpix='yes', fixfile='aux/badpix2.dat',  oversca='no', trim='no', zerocor='no', darkcor='no', flatcor='no',Stdout="/dev/null")
	files=glob.glob("reductions/%s/ccd4/*/[01-31]*.fits" % (date))
	for f in files:
		iraf.ccdproc(images=f, output='', ccdtype='', fixpix='yes', fixfile='aux/badpix4.dat',  oversca='no', trim='no', zerocor='no', darkcor='no', flatcor='no',Stdout="/dev/null")

def remove_cosmics_proc(args):
	"""
	A wrapper function to enable multiprocessing for cosmics removal.
	"""
	file,ccd=args
	array, header=cosmics.fromfits(file)
	c = cosmics.cosmicsimage(array, gain=1.0, readnoise=10.0, sigclip = 9.5, sigfrac = 0.5, objlim = 3.0)
	if ccd==4:
		c.run(maxiter = 4, xb=3.0, yb=5.0)
	else:
		c.run(maxiter = 3, xb=3.0, yb=5.0)
	cosmics.tofits(file, c.cleanarray, header)

def remove_cosmics(date, ncpu=1):
	"""
	Use LACosmic to remove cosmic rays. This is slow, so multiprocessing is used.

	Parameters:
		date (str): date string (e.g. 190210)
		ncpu (int): number of processes to use for parallelization (default is 1, so no parallelization)
	
	Returns:
		none

	To do:
		Check parameters on a bright field or a field with few exposed fibres.
		Check parameters on an old field with many cosmic rays
		Tweak parameters for IR
	"""

	args=[]
	for ccd in [1,2,3,4]:
		files=glob.glob("reductions/%s/ccd%s/*/[01-31]*.fits" % (date, ccd))
		for file in files:
			args.append([file,ccd])

	pool = Pool(processes=ncpu)
	pool.map(remove_cosmics_proc, args)
	pool.close()


def prepare_flat_arc(date, cobs):
	"""
	There might be multiple flats and arcs for each COB. Combine them into one masterarc and masterflat (one per COB). 

	Parameters:
		date (str): date string (e.g. 190210)
		cobs (array): COB array from function prepare_dir
	
	Returns:
		none

	To do:
		remove the need for cobs parameter

	"""
	# Check if there are multiple flats and arcs for each cob. If yes, combine them into a masterflat and masterarc.
	logging.info('Preparing flats and arcs.')
	for ccd in [1,2,3,4]:
		for i in cobs[ccd-1]:
			flats=[]
			arcs=[]
			masterflat=[]
			masterarc=[]
			for j in i[1]:
				if j[1]=='MFFFF':
					flats.append(j[0])
				if j[1]=='MFARC':
					arcs.append(j[0])
			
			if len(flats)==1:#if there is only one flat, just rename it into masterflat
				shutil.move(flats[0], 'reductions/'+date+'/ccd%s/' % ccd+i[0]+'/masterflat.fits')
			elif len(flats)==0:
				logging.error('Flat missing for COB %s. This COB will be skipped.' % i[0])
				shutil.rmtree("reductions/%s/ccd%s/%s" % (date,ccd,i[0]))
			else:
				iraf.imcombine(input=','.join(flats), output='reductions/'+date+'/ccd%s/' % ccd+i[0]+'/masterflat.fits', combine='average', reject='none', Stdout="/dev/null")

			if len(arcs)==1:#if there is only one arc, just rename it into masterarc
				shutil.move(arcs[0], 'reductions/'+date+'/ccd%s/' % ccd+i[0]+'/masterarc.fits')
			elif len(arcs)==0:
				logging.error('Arc missing for COB %s. This COB will be skipped.' % i[0])
				shutil.rmtree("reductions/%s/ccd%s/%s" % (date,ccd,i[0]))
			else:
				iraf.imcombine(input=','.join(arcs), output='reductions/'+date+'/ccd%s/' % ccd+i[0]+'/masterarc.fits', combine='average', reject='none', Stdout="/dev/null")

			for f in flats:
				if os.path.exists(f):
					os.remove(f)

			for a in arcs:
				if os.path.exists(a):
					os.remove(a)

def shift_ref(date, plate, ccd, cob, flat):
	"""
	To guaranty the correct order of apertures, we use a list of predefined apertures. The image, however, is not necessarly aligned with the list of apertures. List is shifted, so the order of apertures remains the same, but the apertures are at the same place as in the image.

	Parameters:
		date (str): date string (e.g. 190210)
		plate (str): either 'Plate 0' or 'Plate 1'
		ccd (int): CCD number (1=Blue, .., 4=IR)
		cob (str): path to current cob
		flat (str): image name of the flat on which the apertures are to be found
	
	Returns:
		none

	"""

	if int(date)>=140601:
		if plate=='Plate 0':
			ap_ref={1: 'aux/masterflat_blue0.fits', 2: 'aux/masterflat_green0.fits', 3: 'aux/masterflat_red20.fits', 4: 'aux/masterflat_ir20.fits'}
		if plate=='Plate 1':
			ap_ref={1: 'aux/masterflat_blue.fits', 2: 'aux/masterflat_green.fits', 3: 'aux/masterflat_red1.fits', 4: 'aux/masterflat_ir.fits'}
	else:
		if plate=='Plate 0':
			ap_ref={1: 'aux/masterflat_blue0.fits', 2: 'aux/masterflat_green0.fits', 3: 'aux/masterflat_red0.fits', 4: 'aux/masterflat_ir0.fits'}
		if plate=='Plate 1':
			ap_ref={1: 'aux/masterflat_blue.fits', 2: 'aux/masterflat_green.fits', 3: 'aux/masterflat_red.fits', 4: 'aux/masterflat_ir.fits'}

	shift=iraf.xregister(input=ap_ref[ccd]+'[1990:2010,*],'+flat+'[1990:2010,*]', reference=ap_ref[ccd]+'[1990:2010,*]', regions='[*,*]', shifts=cob+"/shifts", output="", databasefmt="no", correlation="fourier", xwindow=3, ywindow=51, xcbox=21, ycbox=21, Stdout=1)
	os.remove(cob+'/shifts')
	shift=float(shift[-2].split()[-2])

	ap_file=ap_ref[ccd][:-5].replace('/','/ap')

	f=open(ap_file, "r")
	o=open('/'.join(flat.split('/')[:-1])+'/apmasterflat', "w")

	for line in f:
		l=line.split()
		if len(l)>0 and l[0]=='begin':
			l[-1]=str(float(l[-1])-shift)
			o.write(l[0]+"\t"+l[1]+"\t"+"masterflat"+"\t"+l[3]+"\t"+l[4]+"\t"+l[5]+"\n")
		elif len(l)>0 and l[0]=='center':
			l[-1]=str(float(l[-1])-shift)
			o.write("\t"+l[0]+"\t"+l[1]+"\t"+l[2]+"\n")
		elif len(l)>0 and l[0]=='image':
			o.write("\t"+"image"+"\t"+"masterflat"+"\n")
		elif len(l)>0 and l[0]=='low':
			o.write("\t"+"low"+"\t"+l[1]+"\t"+"-3."+"\n")
		elif len(l)>0 and l[0]=='high':
			o.write("\t"+"high"+"\t"+l[1]+"\t"+"3."+"\n")
		else:
			o.write(line)

	f.close()
	o.close()
	

def find_apertures(date):
	"""
	Finds apertures on all flats for all COBs inside the date folder. A list of apertures thad can't be traced is saved as a file.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none

	"""

	# load correct IRAF packages
	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.twodspec(_doprint=0,Stdout="/dev/null")
	iraf.apextract(_doprint=0,Stdout="/dev/null")

	# reset apall task and set dispersion orientation
	iraf.unlearn('apall')
	iraf.apextract.dispaxis=1

	for ccd in [1,2,3,4]:
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:
			print cob
			# shift reference flat, so the apertures will be found correctly and will always be in the same order
			try:
				hdul=fits.open(cob+'/masterflat.fits')
			except IOError as err:
				logging.warning('Masterflat was not found in ccd %d of COB %s' % (ccd, cob.split('/')[-1]))
				continue

			plate= hdul[0].header['SOURCE']
			hdul.close()
			shift_ref(date, plate, ccd, cob, cob+'/masterflat.fits')

			# trace apertures on masterflat
			iraf.apextract.database=start_folder+'/'+cob
			os.chdir(cob)
			check=iraf.apall(input='masterflat.fits', format='multispec', referen='masterflat', interac='no', find='no', recenter='yes', resize='no', edit='yes', trace='yes', fittrac='yes', extract='yes', extras='no', review='yes', line=2000, lower=-3, upper=3, llimit=-3, ulimit=3, nfind=392, maxsep=45, minsep=5, width=4.5, radius=3, ylevel='0.3', shift='no', t_order=5, t_niter=5, t_low_r=3, t_high_r=3, t_sampl='1:4095', t_nlost=1, npeaks=392, bkg='no', b_order=7, nsum=-10, background='none', Stdout=1)
			# extract another flat 1 px wide. This is better for measuring fibre throughputs
			#check2=iraf.apall(input='masterflat.fits', output='masterflat1px.ms.fits', format='multispec', referen='masterflat', interac='no', find='no', recenter='yes', resize='yes', edit='yes', trace='yes', fittrac='yes', extract='yes', extras='no', review='yes', line=2000, lower=-0.5, upper=0.5, llimit=-0.5, ulimit=0.5, nfind=392, maxsep=45, minsep=5, width=4.5, radius=3, ylevel='0.3', shift='no', t_order=5, t_niter=5, t_low_r=3, t_high_r=3, t_sampl='1:4095', t_nlost=1, npeaks=392, bkg='no', b_order=7, nsum=-10, background='none', Stdout=1)
			os.chdir('../../../..')

			#find apertures that didn't trace properly
			failed_apertures=[]
			failed_apertures_once=[]
			failed_apertures_twice=[]
			for i in check:
				if float(i.split()[-1])<4090. and i.split()[-4]=='lost':
					failed_apertures_once.append(int(i.split()[-7]))
				if float(i.split()[-1])<4090. and i.split()[-4]=='lost' and int(i.split()[-7]) in failed_apertures_once:
					failed_apertures_twice.append(int(i.split()[-7]))
				if float(i.split()[-1])<4090. and i.split()[-4]=='lost' and int(i.split()[-7]) in failed_apertures_twice:
					failed_apertures.append(int(i.split()[-7]))

			failed_apertures_save=[]
			for i in set(failed_apertures):
				failed_apertures_save.append(i)
			np.save(cob+'/failed_apertures', np.array(failed_apertures_save))

			print 'Failed apertures:', set(failed_apertures)
			print 'Failed once:', set(failed_apertures_once)

			# clear cache, or IRAF will have problems with files with the same name
			iraf.flprcache()

def plot_apertures(date,cob_id,ccd):
	"""
	Plots an image and marks where apertures are in column 2000. Red marks mean that the trace was lost at some point.

	Parameters:
		date (str): date string (e.g. 190210)
		cob_id (str): COB ID (e.g. 1902100040)
		ccd (int): CCD number (1=Blue, .., 4=IR)
	
	Returns:
		none

	To do:
		Mark different fibre types.

	"""
	matplotlib_backend=matplotlib.get_backend()

	fig=figure('Apertures  date=%s, cob_id=%s, ccd=%s' % (date,cob_id,ccd))
	subplots_adjust(left=0.03, right=0.98, top=0.99, bottom=0.045, wspace=0.1, hspace=0.1)
	ax=fig.add_subplot(121, aspect=1)
	
	# display masterflat
	hdul=fits.open('reductions/%s/ccd%s/%s/masterflat.fits' % (date, ccd, cob_id))
	data=hdul[0].data
	hdul.close()
	ax.imshow(data,cmap='binary',origin='lower',vmin=np.percentile(data,0.5),vmax=np.percentile(data,99.5))

	# mark apertures
	a=open('reductions/%s/ccd%s/%s/aplast' % (date, ccd, cob_id))
	failed_apertures=np.load('reductions/%s/ccd%s/%s/failed_apertures.npy' % (date, ccd, cob_id))
	detected_apertures=[]
	for line in a:
		l=line.split()
		if len(l)>0 and l[0]=='begin':
			x=float(l[4])
			y=float(l[5])-1.0
			n=int(l[3])
			if n in failed_apertures:
				ax.plot(x,y, 'rx')
				ax.text(x,y, ' %s' % n, ha='left', va='center', color='r').set_clip_on(True)
			else:
				ax.plot(x,y, 'gx')
				ax.text(x,y, ' %s' % n, ha='left', va='center', color='g').set_clip_on(True)

	ax.set_title('Flat')
	ax.set_xlabel('x / px')
	ax.set_ylabel('y / px')

	ax.set_xlim(0,4096)
	ax.set_ylim(0,4112)

	ax=fig.add_subplot(122, sharex=ax, sharey=ax, aspect=1)
	
	# display an object
	files=glob.glob("reductions/%s/ccd%s/%s/[01-31]*.fits" % (date, ccd, cob_id))
	valid_files=[]
	for f in files:
		if '.ms' not in f: valid_files.append(f)
	hdul=fits.open(valid_files[0])
	data=hdul[0].data
	hdul.close()
	ax.imshow(data,cmap='binary',origin='lower',vmin=np.percentile(data,2),vmax=np.percentile(data,95))

	#create a dict from fibre table
	table_name='fibre_table_'+valid_files[0].split('/')[-1]
	hdul=fits.open('/'.join(valid_files[0].split('/')[:-1])+'/'+table_name)
	fibre_table=hdul[1].data
	hdul.close
	fibre_table_dict={}
	n=0
	for i in fibre_table:
		if i[8]=='F': 
			continue
		else: 
			n+=1
			fibre_table_dict[n]=i

	# mark apertures
	a=open('reductions/%s/ccd%s/%s/aplast' % (date, ccd, cob_id))
	failed_apertures=np.load('reductions/%s/ccd%s/%s/failed_apertures.npy' % (date, ccd, cob_id))
	detected_apertures=[]
	for line in a:
		l=line.split()
		if len(l)>0 and l[0]=='begin':
			x=float(l[4])
			y=float(l[5])-1.0
			n=int(l[3])
			ap_type=fibre_table_dict[n][8]
			if n in failed_apertures:
				ax.plot(x,y, 'rx')
				ax.text(x,y, ' %s (%s)' % (n, ap_type), ha='left', va='center', color='r').set_clip_on(True)
			else:
				ax.plot(x,y, 'gx')
				ax.text(x,y, ' %s (%s)' % (n, ap_type), ha='left', va='center', color='g').set_clip_on(True)

	ax.set_title('Object')
	ax.set_xlabel('x / px')
	ax.set_ylabel('y / px')

	ax.set_xlim(0,4096)
	ax.set_ylim(0,4112)

	#make plot fullscreen
	if 	matplotlib_backend=='TkAgg':
		mng=get_current_fig_manager()
		try:
			mng.window.state('zoomed')
		except:
			try:
				mng.resize(*mng.window.maxsize())
			except:
				logging.error('Cannot display Figure \"Apertures date=%s, cob_id=%s, ccd=%s\" in full screen mode.' % (date,cob_id,ccd))
	elif matplotlib_backend=='wxAgg':
		mng=plt.get_current_fig_manager()
		mng.frame.Maximize(True)
	elif matplotlib_backend=='QT4Agg':
		mng=plt.get_current_fig_manager()
		mng.window.showMaximized()
	else:
		logging.error('Cannot display Figure \"Apertures date=%s, cob_id=%s, ccd=%s\" in full screen mode.' % (date,cob_id,ccd))



	show()

def remove_scattered_proc(arg):
	"""
	Wrapper for remove_scattered to enable multiprocessing
	"""

	cob,file=arg
	os.chdir(cob)
	iraf.apextract.database=start_folder+'/'+cob

	def fill_nans(im):
		"""
		Linearly interpolate nan values in an image
		"""
		nx, ny=im.shape[1], im.shape[0]
		X, Y=np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
		int_im=griddata((X[~np.isnan(im)],Y[~np.isnan(im)]), im[~np.isnan(im)], (X, Y), fill_value=np.nanmedian(im))
		return int_im

	if '.ms' in file: 
		return 0

	# create pixel mask that masks out regions not inside apertures
	iraf.apmask(input=file, output='mask_'+file, apertures='', references='masterflat.fits', interactive='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrace='no', mask='yes', buffer=4.0, Stdout="/dev/null")
	# invert the mask, because we need regions outside of the aperture (and buffer)
	iraf.imarith(operand1='mask_'+file[:-5]+'.pl', op='*', operand2='-1', result='maskinv_'+file[:-5])
	os.remove('mask_'+file[:-5]+'.pl')
	iraf.imarith(operand1='maskinv_'+file, op='+', operand2='1', result='mask_'+file)
	os.remove('maskinv_'+file)
	# because masked regions will have flux 0, add some offset. Hence the pixels that coincidentaly have flux 0, but are valid will not be interpreted as masked
	iraf.imarith(operand1=file, op='+', operand2='1000', result='offset_'+file)
	# multiply image by mask
	iraf.imarith(operand1='offset_'+file, op='*', operand2='mask_'+file, result='scat_'+file)
	os.remove('mask_'+file)
	os.remove('offset_'+file)

	#open image as array
	hdul=fits.open('scat_'+file, mode='update')
	data=hdul[0].data
	#hdul.close()
	data[data==0]=np.nan
	data=data-1000.0

	# 1. method: fit 2d polynomial. This is a bad idea. Scattered light is more complicated than this.
	#xx, yy = np.meshgrid(np.arange(0,data.shape[0],1), np.arange(0,data.shape[1],1), sparse=True)
	#m0=[0.0]*9
	#res=optimize.minimize(eval_diff, m0, args=(data), tol=0.1)
	#print res

	# 2. method: use median of the masked image. Also not great. Looks like it is better to do nothing?
	#scat=np.nanmedian(data)
	#hdul=fits.open(file, mode='update')
	#hdul[0].data=hdul[0].data-scat
	#hdul.close()

	# 3. method: Use gaussian kernel to replace masked regions.
	data_full=fill_nans(data)
	# scattered light is by definition smooth, so lets median filter the scattered light image and then smooth it a little bit. This will get rid of any cosmics and the littrow ghost.
	data_full=ndimage.median_filter(data_full, size=(9,13))
	data_full=ndimage.gaussian_filter(data_full, sigma=(5,101),truncate=3)
	# there cannot be negative scattered light. If there is, something went wrong, perhaps in bias removal. Lets not make it wors and limit scattered light to positive values.
	data_full[data_full<0.0]=0.0
	# supress some very high pixels. They can't be right.
	data_full[data_full>np.percentile(data_full,95)]=np.percentile(data_full,95)
	# if scattered light is very high, display a warning
	if np.max(data_full)>50: logging.warning('Scattered light is larger than 50 e in image %s' % file)
	# write back to fits
	hdul[0].data=data_full
	hdul.close()

	# Subtract scattered light from image
	iraf.imarith(operand1=file, op='-', operand2='scat_'+file, result='noscat_'+file)
	os.remove(file)
	iraf.imcopy(input='noscat_'+file, output=file, verbose='no')
	os.remove('noscat_'+file)

	os.chdir('../../../..')
	iraf.flprcache()


def remove_scattered(date, ncpu=1):
	"""
	Remove sacttered light. Scattered light is measured between slitlets, at least 4 pixels from the edge of any aperture. Elswhere it is interpolated and smoothed.

	Parameters:
		date (str): date string (e.g. 190210)
		ncpu (int): number of processes to use for parallelization (default is 1, so no parallelization)
	
	Returns:
		none

	To do:
		Do some cleaning

	"""

	def polyval2d(x, y, m):
		"""
		Return value of a 2D polynomial with coefficients m
		"""
		order=int(np.sqrt(len(m))) - 1
		z=0
		n=0
		for i in range(order):
			for j in range(order):
				z+=m[n]*x**i*y**j
				n+=1
		return z

	def eval_diff(m, image):
		xx, yy = np.meshgrid(np.arange(0,image.shape[1],1), np.arange(0,image.shape[0],1), sparse=True)
		poly=polyval2d(xx, yy, m)
		return np.nansum(abs(image-poly))

	logging.info('Removing scattered light.')

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.twodspec(_doprint=0,Stdout="/dev/null")
	iraf.apextract(_doprint=0,Stdout="/dev/null")
	iraf.apextract.dispaxis=1

	# crop arc and object images, so the overscan is not in the spectra
	for ccd in [1,2,3,4]:
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:
			os.chdir(cob)
			try:
				iraf.imcopy(input='masterarc.fits[1:4096,*]', output='crop.tmp', verbose='no',Stdout="/dev/null")
				for t in range(20):#wait for max 10 sec for the file to be created
					if os.path.exists('crop.tmp')==False: time.sleep(0.5)
					else: break
				shutil.move('crop.tmp.fits', 'masterarc.fits')
				iraf.flprcache()
			except:
				logging.warning('Masterarc was not found in ccd %d of COB %s' % (ccd, cob.split('/')[-1]))

			files=glob.glob("./[01-31]*.fits")
			for f in files:
				iraf.imcopy(input=f+'[1:4096,*]', output='crop.tmp', verbose='no',Stdout="/dev/null")
				shutil.move('crop.tmp.fits', f)
			os.chdir('../../../..')
			iraf.flprcache()

	cobs=glob.glob("reductions/%s/ccd*/*" % (date))
	args=[]
	for cob in cobs:
		os.chdir(cob)
		files=glob.glob("[01-31]*.fits")
		for file in files:
			args.append([cob,file])
		os.chdir('../../../..')

	pool = Pool(processes=ncpu)
	pool.map(remove_scattered_proc, args)
	pool.close()

	iraf.flprcache()


def measure_cross_on_flat(date):
	"""
	Measure cross talk and use it to correct extracted spectra. We have to improve this, becasue the cross talk is very fibre dependant.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none

	ToDo:
		Everything.
	"""
	pass

def extract_spectra(date):
	"""
	Takes *.fits images and creates *.ms.fits images with 392 spectra in them.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none
	"""
	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.twodspec(_doprint=0,Stdout="/dev/null")
	iraf.apextract(_doprint=0,Stdout="/dev/null")

	iraf.unlearn('apall')
	iraf.apextract.dispaxis=1

	# crop arc and object images, so the overscan is not in the spectra
	for ccd in [1,2,3,4]:
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:
			os.chdir(cob)
			try:
				iraf.imcopy(input='masterarc.fits[1:4096,*]', output='crop.tmp', verbose='no',Stdout="/dev/null")
				for t in range(20):#wait for max 10 sec for the file to be created
					if os.path.exists('crop.tmp')==False: time.sleep(0.5)
					else: break
				shutil.move('crop.tmp.fits', 'masterarc.fits')
				iraf.flprcache()
			except:
				logging.warning('Masterarc was not found in ccd %d of COB %s' % (ccd, cob))

			files=glob.glob("./[01-31]*.fits")
			for f in files:
				iraf.imcopy(input=f+'[1:4096,*]', output='crop.tmp', verbose='no',Stdout="/dev/null")
				shutil.move('crop.tmp.fits', f)
				if os.path.exists('scat_'+f):# at this point we should have a scattered light file somewhere which will also be extracted, so do the same for it
					iraf.imcopy(input='scat_'+f+'[1:4096,*]', output='crop.tmp', verbose='no',Stdout="/dev/null")
					shutil.move('crop.tmp.fits', 'scat_'+f)
				iraf.flprcache()
			os.chdir('../../../..')

	for ccd in [1,2,3,4]:
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:
			iraf.apextract.database=start_folder+'/'+cob
			#extract arc
			os.chdir(cob)
			check=iraf.apall(input='masterarc.fits', format='multispec', referen='masterflat', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='no', lower=-3.0, upper=3.0, nsubaps=1, pfit="fit1d", background='none', Stdout=1)
			#extract spectra
			files=glob.glob("./[01-31]*.fits")
			files_scat=glob.glob("./scat_[01-31]*.fits")
			check=iraf.apall(input=','.join(files), format='multispec', referen='masterflat', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='no', lower=-3.0, upper=3.0, nsubaps=1, pfit="fit1d", background='none', Stdout=1)
			#extract scattered light spectra, so they can be saved in the final output
			if len(files_scat)>0:
				check=iraf.apall(input=','.join(files_scat), format='multispec', referen='masterflat', interac='no', find='no', recenter='no', resize='no', edit='no', trace='no', fittrac='no', extract='yes', extras='no', review='no', lower=-3.0, upper=3.0, nsubaps=1, pfit="fit1d", background='none', Stdout=1)
			os.chdir('../../../..')

			# clear cache, or IRAF will have problems with files with the same name
			iraf.flprcache()

def wav_calibration(date):
	"""
	Calculate wavelength calibration and apply it to spectra.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none

	To do: remove baseline
	       use the same method as in dr5.3. In dr5.3 we use a custom fit, not iraf's
	"""

	# linelists of the arc lamp for each CCD
	cal_linelist={1: 'aux/linelist_art_blue_ar.dat', 2: 'aux/linelist_art_green_ar.dat', 3: 'aux/linelist_art_red.dat', 4: 'aux/linelist_art_ir.dat'}

	# reference images
	cal_arc={'Plate 0': {1: 'aux/master_arc_p_0_ccd_1.ms.fits', 2: 'aux/master_arc_p_0_ccd_2.ms.fits', 3: 'aux/master_arc_p_0_ccd_3.ms.fits', 4: 'aux/master_arc_p_0_ccd_4.ms.fits'}, 'Plate 1':{1: 'aux/master_arc_p_1_ccd_1.ms.fits', 2: 'aux/master_arc_p_1_ccd_2.ms.fits', 3: 'aux/master_arc_p_1_ccd_3.ms.fits', 4: 'aux/master_arc_p_1_ccd_4.ms.fits'}}

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.onedspec(_doprint=0,Stdout="/dev/null")

	iraf.unlearn('reident')

	for ccd in [1,2,3,4]:
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:
			if not os.path.isfile(cob+'/masterarc.ms.fits'):
				logging.warning('Masterarc was not found in ccd %d of COB %s, wavelength calibration will be skipped for this ccd-COB combination' % (ccd, cob.split('/')[-1]))
				continue

			hdul=fits.open(cob+'/masterarc.ms.fits')
			plate= hdul[0].header['SOURCE']
			hdul.close()
			os.chdir(cob)
			shutil.copy('../../../../'+cal_arc[plate][ccd], os.getcwd())
			shutil.copy('../../../../aux/id'+cal_arc[plate][ccd][4:-5], os.getcwd())
			check=iraf.reident(referenc=cal_arc[plate][ccd][4:-5], images='masterarc.ms', interac='no', newaps='no', overrid='yes', refit='yes', trace='no', addfeat='yes', coordlist='../../../../'+cal_linelist[ccd], match=-2, minsep=3, maxfeatures=100, shift='INDEF', step=10, search='INDEF', nlost=1000, cradius=5.0, threshold=0.01, database=start_folder+'/'+cob, answer='yes', verbose='yes', section='middle line', Stdout=1)
			wav_good_dict={}
			for n,i in enumerate(check):
				if len(i.split())>1 and i.split()[0]=='masterarc.ms': 
					#print int(i.split()[3]), i.split()[4], float(i.split()[-1])
					wav_good_dict[int(i.split()[3])]=[i.split()[4], float(i.split()[-1])]
			wav_good_dict=defaultdict(lambda:['0/0', 'None'], wav_good_dict)
			#print wav_good_dict
			#save wavelength solution parameters
			wav_good=[]
			for ap in range(1,393):
				wav_good.append(wav_good_dict[ap])
			np.save('wav_fit', np.array(wav_good))
			files=glob.glob("./[01-31]*.ms.fits")
			files_scat=glob.glob("./scat_[01-31]*.ms.fits")
			#wavelength calibrate object files
			for file in files:
				hdul=fits.open(file, mode='update')
				hdr=hdul[0].header
				hdr['REFSPEC1']='masterarc.ms'
				hdul.close()
				iraf.dispcor(input=file, lineari='no', databas=start_folder+'/'+cob, output="", samedis='no', ignorea='yes',Stdout="/dev/null")
			#wavelength calibrate scattered light files
			for file in files_scat:
				hdul=fits.open(file, mode='update')
				hdr=hdul[0].header
				hdr['REFSPEC1']='masterarc.ms'
				hdul.close()
				iraf.dispcor(input=file, lineari='no', databas=start_folder+'/'+cob, output="", samedis='no', ignorea='yes',Stdout="/dev/null")
			#wavelength calibrate arc spectrum
			hdul=fits.open('masterarc.ms.fits', mode='update')
			hdr=hdul[0].header
			hdr['REFSPEC1']='masterarc.ms'
			hdul.close()
			iraf.dispcor(input='masterarc.ms.fits', lineari='no', databas=start_folder+'/'+cob, output="", samedis='no', ignorea='yes',Stdout="/dev/null")


			os.chdir('../../../..')
			iraf.flprcache()

def plot_wav_coeff(date, cob_id, ccd):
	"""
	Illustrate wavelength calibration, similar to dr5.3

	Parameters:
		date (str): date string (e.g. 190210)
		cob_id (str): COB ID (e.g. 1902100040)
		ccd (int): CCD number (1=Blue, .., 4=IR)
	
	Returns:
		none

	To do:
		Everything

	"""

	pass

def resolution_profile(date):
	"""
	Calculates the resolution profile of each spectrum. Produces an image with ''res_'' prefix with FWHM in Angstroms.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none

	To do:
		Make it pretty
		Do diagnostics plots
		Paralelize
	"""

	def peak_profile(a,x,y):
		return y-a['amp'].value*np.exp(-(abs(2*(x-a['pos'].value)/a['fwhm'].value)**a['b'].value)*0.693147)

	def fit_profile(p0,y,vary_B=True):
		lower=p0[0]-20
		upper=p0[0]+20
		if lower<0: lower=0
		if upper>4096: upper=4096
		y=y[p0[0]-lower:p0[0]+upper]
		x=np.arange(4096)
		x=x[p0[0]-lower:p0[0]+upper]
		params = Parameters()
		params.add('pos', value=p0[0], min=p0[0]-2.0, max=p0[0]+2.0)
		params.add('amp', value=p0[1], min=p0[1]*0.7, max=p0[1]*1.3)
		params.add('fwhm', value=p0[2], min=p0[2]*0.5, max=p0[2]*1.5)
		params.add('b', value=2.5, min=2.0, max=3.8, vary=vary_B)
		minner = Minimizer(peak_profile, params, fcn_args=(x,y))
		result = minner.minimize()
		res=[result.params['pos'].value, result.params['amp'].value, result.params['fwhm'].value, result.params['b'].value]
		return res

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.onedspec(_doprint=0,Stdout="/dev/null")

	# number of lines to be found that give best resolution profile:
	n_of_lines_dict={1:45, 2:45, 3:45, 4:40}

	for ccd in [1,2,3,4]:
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:
			logging.info('Calculating resolution profile for COB %s.' % cob)
			files=glob.glob("%s/[01-31]*.ms.fits" % cob)
			arc=cob+"/masterarc.ms.fits"
			# linearize arc
			iraf.disptrans(input=arc, output='', linearize='yes', units='angstroms', Stdout="/dev/null")
			# remove baseline (we want to measure lines with zero continuum level)
			iraf.continuum(input=arc, output='/'.join(arc.split('/')[:-1])+'/nobaseline_'+arc.split('/')[-1], type='difference', lines='*', bands='*', function='cheb', order='5', low_rej='4.0', high_rej='2.0', niter='8', interac='no', Stdout="/dev/null")
			hdul=fits.open('/'.join(arc.split('/')[:-1])+'/nobaseline_'+arc.split('/')[-1])
			data=hdul[0].data
			hdul.close()

			res_points=[]
			b_values=[]
			for ap in range(1,393):
				line=ap-1 # line in array is different than aperture number
				#fig=figure(0)
				#ax=fig.add_subplot(311)
				#ax.plot(data[line], 'k-')
				# find peaks positions
				peaks,properties=signal.find_peaks(data[line], width=(3.0,8.0), distance=10, height=100.0, prominence=1.0)
				widths=properties['widths']
				heights=properties['peak_heights']
				# accept only some strongest peaks
				sorted_ind=heights.argsort()
				peaks=peaks[sorted_ind]
				widths=widths[sorted_ind]
				heights=heights[sorted_ind]
				peaks=peaks[-n_of_lines_dict[ccd]:]
				widths=widths[-n_of_lines_dict[ccd]:]
				heights=heights[-n_of_lines_dict[ccd]:]
				#ax.plot(peaks, data[line][peaks], 'rx')
				#ax2=fig.add_subplot(312, sharex=ax)
				#ax3=fig.add_subplot(313, sharex=ax)
				#ax2.plot(peaks,widths, 'ko')
				# fit peaks with a modified gaussian function and calculate mean B parameter for each fibre
				bs=[]
				for i,j,k in zip(peaks, widths, heights):
					res=fit_profile([i,k,j], data[line])
					#ax2.plot(res[0],res[2], 'ro')
					#ax.plot(np.arange(len(data[line])), res[1]*np.exp(-(abs(2*(np.arange(len(data[line]))-res[0])/res[2])**res[3])*0.693147), 'r-')
					#ax3.plot(res[0],res[3], 'ro')
					bs.append(res[3])
				b_median=np.nanmedian(bs)
				# in case B cannot be calculated use a typical value
				if np.isnan(b_median) or len(bs)<10:
					if ccd==1: b_median=2.40
					if ccd==2: b_median=2.25
					if ccd==3: b_median=2.15
					if ccd==4: b_median=2.05
				# fit peaks again to get their fwhm, given a fixed B
				widths_fit=[]
				for i,j,k in zip(peaks, widths, heights):
					res=fit_profile([i,k,j], data[line], vary_B=False)
					widths_fit.append(res[2])
				widths=np.array(widths_fit)
				# remember peaks widths and B values
				b_values.append(b_median)
				if len(sorted_ind)>2*n_of_lines_dict[ccd]:# there must be at least
					for i,j in zip(peaks[(peaks>20)&(peaks<4076)],widths[(peaks>20)&(peaks<4076)]):#20 px at each edge are truncated, because width cannot be measured correctly
						res_points.append([ap,i,j])

				#ax2.set_ylim(4,6)
				#ax3.set_ylim(2.0,3.8)
				#show()

			np.save(cob+'/b_values', np.array(b_values))
			res_points=np.array(res_points)
			#fig=figure(0)
			#ax=fig.add_subplot(141)
			cc=res_points[:,2]#-0.0005*res_points[:,1]
			#ax.scatter(res_points[:,1], res_points[:,0], c=cc, vmin=np.percentile(cc,2), vmax=np.percentile(cc,85), s=3, lw=0)

			# fit smooth model
			mask=np.array([True]*len(cc), dtype=bool)
			for n in range(10):
				p_init = polynomial.Chebyshev2D(x_degree=3, y_degree=3)
				fit_p = fitting.LevMarLSQFitter()
				p = fit_p(p_init, res_points[:,1][mask], res_points[:,0][mask], cc[mask])
				std=np.std(cc-p(res_points[:,1], res_points[:,0]))
				mask=np.array([ True]*len(cc), dtype=bool)
				mask[(cc-p(res_points[:,1], res_points[:,0])>1.5*std)|(cc-p(res_points[:,1], res_points[:,0])<-3.*std)]=False

			# fit fibre model
			mask=np.array([ True]*len(cc), dtype=bool)
			for n in range(1):
				f=[]
				profile=cc-p(res_points[:,1], res_points[:,0])
				for ap in range(1,393):
					values=profile[res_points[:,0]==ap]
					if len(values)==0:
						f.append([0.0]*4096)
					else:
						f.append([np.median(values)]*4096)
			f=np.array(f)

			# refit smooth model
			cc_new=[]
			for x,y in zip(res_points[:,1], res_points[:,0]):
				cc_new.append(f[int(y)-1,int(x)])
			cc=cc-np.array(cc_new)
			mask=np.array([True]*len(cc), dtype=bool)
			for n in range(10):
				p_init = polynomial.Chebyshev2D(x_degree=3, y_degree=3)
				fit_p = fitting.LevMarLSQFitter()
				p = fit_p(p_init, res_points[:,1][mask], res_points[:,0][mask], cc[mask])
				std=np.std(cc-p(res_points[:,1], res_points[:,0]))
				mask=np.array([ True]*len(cc), dtype=bool)
				mask[(cc-p(res_points[:,1], res_points[:,0])>1.3*std)|(cc-p(res_points[:,1], res_points[:,0])<-3.*std)]=False

			# refit fibre model
			mask=np.array([ True]*len(cc), dtype=bool)
			for n in range(1):
				f=[]
				profile=cc-p(res_points[:,1], res_points[:,0])+np.array(cc_new)
				for ap in range(1,393):
					values=profile[res_points[:,0]==ap]
					if len(values)==0:
						f.append([0.0]*4096)
					else:
						mask=np.array([True]*len(values), dtype=bool)
						for n in range(5):#do sigma clipping
							model=np.median(values[mask])
							std=np.std(values-model)
							mask=np.array([True]*len(values), dtype=bool)
							mask[(values-model>1.3*std)|(values-model<-3.*std)]=False
						f.append([np.median(values[mask])]*4096)
			f=np.array(f)

			y=np.arange(392)
			x=np.arange(4096)
			xx,yy=np.meshgrid(x,y)

			for file in files:
				disp=[]
				dispersions=iraf.slist(images=file, apertures='', long_header='no', Stdout=1)
				for line in dispersions:
					disp.append(float(line.split()[6]))
				iraf.scopy(input=file, output='/'.join(file.split('/')[:-1])+'/res_'+file.split('/')[-1])
				hdul=fits.open('/'.join(file.split('/')[:-1])+'/res_'+file.split('/')[-1], mode='update')
				hdul[0].data=f+p(xx,yy)#write FWHM in pixels
				hdul[0].data=hdul[0].data*np.array(disp)[:,None]#convert FWHM in pixels to FWHM in A by multyping with dispersion
				hdul.close()

			os.remove('/'.join(arc.split('/')[:-1])+'/nobaseline_'+arc.split('/')[-1])

			

			#y=np.arange(392)
			#x=np.arange(4096)
			#xx,yy=np.meshgrid(x,y)	

			#ax2=fig.add_subplot(142)
			#ax2.pcolor(xx, yy, p(xx,yy))

			#ax3=fig.add_subplot(143)
			#ax3.pcolor(xx,yy,f)

			#ax4=fig.add_subplot(144)
			#ax4.pcolor(xx,yy,p(xx,yy)+f, vmin=4.0,vmax=6.5)

			#ax4.set_title('Combined model')
			#ax3.set_title('Fibre model')
			#ax2.set_title('Smooth model')
			#ax.set_title('Measurements')

			#show()

def v_bary_correction_proc(args):
	"""
	Wrapper for v_bary_correction for quick=False method to enable multiprocessing. 

	"""

	file, fibre_table, obstime, AAT=args
	hdul=fits.open(file, mode='update')
	if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]): 
		hdul_sky=fits.open('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1], mode='update')
	if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]): 
		hdul_tel=fits.open('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], mode='update')
	if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]): 
		hdul_scat=fits.open('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1], mode='update')
	n=0 # n counts apertures (increase whenever fibre type!='F')
	logging.info('Processing (v_bary correction) file %s.' % file)
	for i in fibre_table:
		if i[8]!='F':
			n+=1
		if i[8]=='P':
			sc=SkyCoord(ra=(i[1]/np.pi*180.)*u.deg, dec=(i[2]/np.pi*180.)*u.deg)
			barycorr=sc.radial_velocity_correction(obstime=obstime, location=AAT)
			v_bary=float(barycorr.to(u.km/u.s).value)

			hdul[0].header['BARY_%s' % str(n)]=v_bary
			if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]):
				hdul_sky[0].header['BARY_%s' % str(n)]=v_bary
			if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]):
				hdul_tel[0].header['BARY_%s' % str(n)]=v_bary
			if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]):
				hdul_scat[0].header['BARY_%s' % str(n)]=v_bary
			
			iraf.dopcor(input=file, output='', redshift='-%s' % v_bary, isveloc='yes', add='yes', dispers='yes', apertures=n, flux='no')
			if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]): 
				iraf.dopcor(input='/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1], output='', redshift='-%s' % v_bary, isveloc='yes', add='yes', dispers='yes', apertures=n, flux='no')
			if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]): 
				iraf.dopcor(input='/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], output='', redshift='-%s' % v_bary, isveloc='yes', add='yes', dispers='yes', apertures=n, flux='no')
			if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]): 
				iraf.dopcor(input='/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1], output='', redshift='-%s' % v_bary, isveloc='yes', add='yes', dispers='yes', apertures=n, flux='no')
	hdul.close()
	if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]):
		hdul_sky.close()
	if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]):
		hdul_tel.close()
	if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]): 
		hdul_scat.close()


def v_bary_correction(date, quick=False, ncpu=1):
	"""
	Calculates barycentric velocity correction and applyies it to the wavelength calibration. Slow and quick options are available. With the slow option, the correction must be calculated for every fibre on every image, which takes a while. Quick options calculates the correction for the field centre only. Errors of the quick method are in the order of 0.06 km/s.

	Parameters:
		date (str): date string (e.g. 190210)
		quick (bool): slow and precise (default) or fast and imprecise algorith is used
		ncpu (int): number of processes to use for parallelization (default is 1, so no parallelization)
	
	Returns:
		none

	To do:
		Sort out downloading UT1-UTC tables.
		Make sure the header cleans OK for both (quick=True and False) cases.
		Check with Tomaz whether vbary is the same as his

	"""

	from astropy.utils import iers
	iers.conf.auto_download = False # download UT1-UTC table (slow and URL needs to be updated)

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.onedspec(_doprint=0,Stdout="/dev/null")

	logging.info('Calculating barycentric velocity corrections. This might take some time, if quick=False option is used (default).')
	AAT_lon = 149.0658
	AAT_lat = -31.2769
	AAT_elevation = 1164
	AAT=EarthLocation.from_geodetic(lat=AAT_lat*u.deg, lon=AAT_lon*u.deg, height=AAT_elevation*u.m)

	args=[]

	files=glob.glob("reductions/%s/ccd*/*/[01-31]*.ms.fits" % (date))
	for file in files:
		# we need coordinates for indivisual fibres
		fibre_table_file='/'.join(file.split('/')[:-1])+'/fibre_table_'+file.split('/')[-1].replace('.ms', '')
		hdul=fits.open(fibre_table_file)
		fibre_table=hdul[1].data
		hdul.close()
		# and date of observation for the image
		hdul=fits.open(file)
		mjd_start=hdul[0].header['UTMJD'] # MJD at the beginning of the exposure
		exp=hdul[0].header['EXPOSED']/3600./24. # exposure time in days
		ra=hdul[0].header['MEANRA']
		dec=hdul[0].header['MEANDEC']
		mjd_mid=mjd_start+exp/2. # MJD in the middle of the exposure
		obstime=Time(mjd_mid, format='mjd')
		hdul.close()
		
		if quick:
			#calculate mean v_bary for the quick method
			sc=SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
			barycorr=sc.radial_velocity_correction(obstime=obstime, location=AAT)
			v_bary_mean=float(barycorr.to(u.km/u.s).value)
			hdul=fits.open(file, mode='update')
			hdul[0].header['BARYEFF']=(v_bary_mean, 'Barycentric velocity correction in km/s')
			hdul.close()
			if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]): 
				hdul_sky=fits.open('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1], mode='update')
				hdul_sky[0].header['BARYEFF']=v_bary_mean
				hdul_sky.close()
			if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]): 
				hdul_tel=fits.open('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], mode='update')
				hdul_tel[0].header['BARYEFF']=v_bary_mean
				hdul_tel.close()
			if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]): 
				hdul_scat=fits.open('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1], mode='update')
				hdul_scat[0].header['BARYEFF']=v_bary_mean
				hdul_scat.close()

			iraf.dopcor(input=file, output='', redshift='-%s' % v_bary_mean, isveloc='yes', add='yes', dispers='yes', apertures='', flux='no')
			if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]): 
				iraf.dopcor(input='/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1], output='', redshift='-%s' % v_bary_mean, isveloc='yes', add='yes', dispers='yes', apertures='', flux='no')
			if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]): 
				iraf.dopcor(input='/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], output='', redshift='-%s' % v_bary_mean, isveloc='yes', add='yes', dispers='yes', apertures='', flux='no')
			if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]): 
				iraf.dopcor(input='/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1], output='', redshift='-%s' % v_bary_mean, isveloc='yes', add='yes', dispers='yes', apertures='', flux='no')
		else:
			# Otherwise do it aperture by aperture. Downloading is currently turned off. 
			args.append([file, fibre_table, obstime, AAT])
	
	if quick==False:
		pool = Pool(processes=ncpu)
		pool.map(v_bary_correction_proc, args)
		pool.close()


def plot_spectra(date, cob_id, ccd):
	"""
	Plot normalised spectra for all 392 spectra for one COB.

	Parameters:
		date (str): date string (e.g. 190210)
		cob_id (str): COB ID (e.g. 1902100040)
		ccd (int): CCD number (1=Blue, .., 4=IR)
	
	Returns:
		none

	To do:

	"""
	matplotlib_backend=matplotlib.get_backend()

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.onedspec(_doprint=0,Stdout="/dev/null")

	fig=figure('Apertures date=%s, cob_id=%s, ccd=%s' % (date,cob_id,ccd))
	subplots_adjust(left=0.025, right=0.89, top=0.99, bottom=0.045, wspace=0.0, hspace=0.0)
	ax=fig.add_subplot(111)

	trans=transforms.blended_transform_factory(ax.transAxes, ax.transData)

	wavs_min=[]
	wavs_max=[]
	vel_frame=[]

	files=glob.glob("reductions/%s/ccd%s/%s/[01-31]*.ms.fits" % (date, ccd, cob_id))
	for n_file, file in enumerate(files):
		#create a dict from fibre table
		if n_file==0:
			table_name='fibre_table_'+file.split('/')[-1]
			hdul=fits.open('/'.join(file.split('/')[:-1])+'/'+table_name.replace('.ms', ''))
			fibre_table=hdul[1].data
			hdul.close
			fibre_table_dict={}
			n=0
			for i in fibre_table:
				if i[8]=='F': 
					pass
				else: 
					n+=1
					fibre_table_dict[n]=i
			fibre_table_dict=defaultdict(lambda:('FIBRE NOT IN USE', 0.0, 0.0, 0, 0, 0, 0, 0.0, 'N', 0, 0.0, 0, 'Not in use', '0', 0.0, 0.0, 0.0), fibre_table_dict)
		#create dict of snrs for each fibre
		snr_dict={}
		for ap in range(1,393):
			snr_dict[ap]=[]
		#load failed apertures
		failed_apertures=np.load('reductions/%s/ccd%s/%s/failed_apertures.npy' % (date, ccd, cob_id))

		iraf.continuum(input=file, output='reductions/%s/ccd%s/%s/norm.tmp'  % (date, ccd, cob_id), order=5, interactive='no', low_reject=2.5, high_reject=0.0, niterate=7, naverage=11)
		hdul=fits.open(file)
		data_orig=hdul[0].data
		hdul.close
		for ap in range(1,393):
			iraf.wspectext(input='reductions/%s/ccd%s/%s/norm.tmp[*,%s]'  % (date, ccd, cob_id, ap), output='reductions/%s/ccd%s/%s/wspectext.out'  % (date, ccd, cob_id), header='no')
			data=np.loadtxt('reductions/%s/ccd%s/%s/wspectext.out'  % (date, ccd, cob_id))
			if fibre_table_dict[ap][8]=='P' and ap not in failed_apertures: 
				if np.std(data[:,1])<0.75: ax.plot(data[:,0], data[:,1]+(ap-1)/2.0, 'k-', lw=0.5, alpha=0.7) # plot spectra of all positioned fibres
				else: ax.plot(data[:,0], data[:,1]+(ap-1)/2.0, 'k-', lw=0.5, alpha=0.2) # plot spectra of all positioned fibres (use different shade for crap spectra)
				snr=np.nanmedian(np.sqrt(data_orig[ap-1])) # calculate SNR of this spectrum
				snr_dict[ap].append(snr) # add snr to dict
				if np.std(data[:,1])>0.75: warning='!!!' # add some exclamation marks to bad spectra
				else: warning=''
				if n_file==len(files)-1: ax.text(1.0,1+(ap-1)/2.0,' '+str(ap).ljust(3,' ')+' '+fibre_table_dict[ap][8]+' '+'MAG='+str(round(fibre_table_dict[ap][10],2)).ljust(5, ' ')+' '+'SNR='+str(round(np.nanmean(snr_dict[ap]), 1)).ljust(6,' ')+' '+warning,transform=trans, va='center', family='monospace') # if this is last file to plot, print some more info
			elif not os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]) and fibre_table_dict[ap][8]=='S' and ap not in failed_apertures: 
				ax.plot(data[:,0], data[:,1]+(ap-1)/2.0, 'g-', lw=0.5, alpha=0.5) # plot sky spectra if sky file does not exists (spectra are not sky subtracted)
				snr=np.nanmedian(np.sqrt(data_orig[ap-1])) # calculate SNR of this spectrum
				snr_dict[ap].append(snr) # add snr to dict
				warning=''
				if n_file==len(files)-1: ax.text(1.0,1+(ap-1)/2.0,' '+str(ap).ljust(3,' ')+' '+fibre_table_dict[ap][8]+' '+'MAG='+'N/A'.ljust(5, ' ')+' '+'SNR='+str(round(np.nanmean(snr_dict[ap]), 1)).ljust(6,' ')+' '+warning,transform=trans, va='center', family='monospace', color='g') # if this is last file to plot, print some more info
			elif fibre_table_dict[ap][8]=='S' and ap not in failed_apertures:
				if n_file==len(files)-1: ax.text(1.0,1+(ap-1)/2.0,' '+str(ap).ljust(3,' ')+' '+fibre_table_dict[ap][8],transform=trans, va='center', color='g', family='monospace')
			else:
				if n_file==len(files)-1: ax.text(1.0,1+(ap-1)/2.0,' '+str(ap).ljust(3,' ')+' '+fibre_table_dict[ap][8],transform=trans, va='center', color='r', family='monospace')
			
			wavs_min.append(data[:,0][0])
			wavs_max.append(data[:,0][-1])
			os.remove('reductions/%s/ccd%s/%s/wspectext.out'  % (date, ccd, cob_id))
		#check whether wavelengths are in observer or barycentric velocity frame
		hdul=fits.open(file)
		vel_frame.append(('BARY' in hdul[0].header) or ('BARY_1' in hdul[0].header) or ('BARYEFF' in hdul[0].header))
		hdul.close()
		os.remove('reductions/%s/ccd%s/%s/norm.tmp.fits'  % (date, ccd, cob_id))

	ax.set_xlim(np.percentile(wavs_min,5)-1, np.percentile(wavs_max,95)+1)
	ax.set_ylim(0,392/2.0+1.0)
	if all(vel_frame):
		ax.set_xlabel('Wavelength (barycentric velocity frame) / $\\mathrm{\\AA}$')
	elif any(vel_frame):
		ax.set_xlabel('Wavelength (mixed velocity frames) / $\\mathrm{\\AA}$')
	else:
		ax.set_xlabel('Wavelength (observer velocity frame) / $\\mathrm{\\AA}$')
	ax.set_ylabel('Normalized flux + offset')
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.tick_params(axis='x', which='minor')

	#make plot fullscreen
	if 	matplotlib_backend=='TkAgg':
		mng=get_current_fig_manager()
		try:
			mng.window.state('zoomed')
		except:
			try:
				mng.resize(*mng.window.maxsize())
			except:
				logging.error('Cannot display Figure \"Apertures date=%s, cob_id=%s, ccd=%s\" in full screen mode.' % (date,cob_id,ccd))
	elif matplotlib_backend=='wxAgg':
		mng=plt.get_current_fig_manager()
		mng.frame.Maximize(True)
	elif matplotlib_backend=='QT4Agg':
		mng=plt.get_current_fig_manager()
		mng.window.showMaximized()
	else:
		logging.error('Cannot display Figure \"Apertures date=%s, cob_id=%s, ccd=%s\" in full screen mode.' % (date,cob_id,ccd))

	show()

def remove_sky_nearest(arg):
	"""
	Wrapper for method 'nearest' of remove_sky function. Wrapper is needed so parallelization works.
	"""

	def nearest_sky(ap):
		"""
		Find the nearest sky fibre to aperture ap
		"""
		for diff in range(400):
			if ap+diff<len(fibre_table_dict) and fibre_table_dict[ap+diff]=='S': return ap+diff
			if ap-diff>0 and fibre_table_dict[ap-diff]=='S': return ap-diff
		return -1

	cob,file,fibre_throughputs_dict_use,fibre_table_dict=arg

	logging.info('Processing (removing sky) file %s' % file)

	os.chdir(cob)
	file=file.split('/')[-1]

	sky_ap_old=-1
	for ap in range(1,393):
		sky_ap=nearest_sky(ap)
		if sky_ap==-1:
			logging.error('There are no sky fibres in image %s.' % file)
		if sky_ap_old!=sky_ap:
			if os.path.exists(file+'sky.fits'):	os.remove(file+'sky.fits')
			iraf.scopy(input=file, output=file+'sky.fits', apertures=sky_ap, Stdout="/dev/null")
			sky_ap_old=sky_ap
		iraf.sarith(input1=file+'sky.fits', op='*', input2=fibre_throughputs_dict_use[ap]/fibre_throughputs_dict_use[sky_ap], output=file+'sky_scaled.fits', clobber='no', merge='no', ignorea='yes', Stdout="/dev/null")
		iraf.sarith(input1=file, op='-', input2=file+'sky_scaled.fits', clobber='yes', merge='yes', ignorea='yes', output='nosky_'+file, apertures=ap, Stdout="/dev/null")
		os.remove(file+'sky_scaled.fits')
	# save sky spectrum and processed spectrum
	iraf.sarith(input1=file, op='-', input2='nosky_'+file, output='sky_'+file)
	os.remove(file)
	iraf.scopy(input='nosky_'+file, output=file)
	os.remove('nosky_'+file)
	os.remove(file+'sky.fits')
	iraf.flprcache()
	os.chdir('../../../../')

def remove_sky_nearest3(arg):
	"""
	Wrapper for method 'nearest3' of remove_sky function. Wrapper is needed so parallelization works.
	"""

	def nearest_3_sky(ap):
		"""
		Find the nearest three sky fibres to aperture ap
		"""
		aps=[]
		for diff in range(400):
			if ap+diff<len(fibre_table_dict) and fibre_table_dict[ap+diff]=='S': aps.append(ap+diff)
			if len(aps)==3: break
			if ap-diff>0 and fibre_table_dict[ap-diff]=='S': aps.append(ap-diff)
			if len(aps)==3: break
		if len(aps)==3: 
			return aps
		else:
			return -1

	def sky_diff(a, spec,sky):
		"""
		Calculates total variation of a signal over a kernel. V(y)=\sum_i \sum_n \left| y_{n+i} - y_n \right| w(i)

		Parameters:
			a (array): parameters (scale)
			spec (array): array representing a 1D spectrum (columns are wavelength and flux)
			sky (array): sky spectrum, same shape as spec
	
		Returns:
			total variation (float)

		To Do:
			fine tune the total variation kernel
			Add mask to use only sky line regions.
		"""

		print a['scale'].value
		sky=a['scale'].value*sky
		# Below is a crude implementation of a kernel
		div=spec-sky
		div_roll1=np.roll(div,1)
		tot_var=np.sum(abs(div-div_roll1))*0.4
		div_roll2=np.roll(div,2)
		tot_var+=np.sum(abs(div-div_roll2))*0.3
		div_roll3=np.roll(div,3)
		tot_var+=np.sum(abs(div-div_roll3))*0.2
		div_roll4=np.roll(div,4)
		tot_var+=np.sum(abs(div-div_roll4))*0.1
		return tot_var

	cob,file,fibre_throughputs_dict_use,fibre_table_dict=arg

	logging.info('Processing (removing sky) file %s' % file)

	os.chdir(cob)
	file=file.split('/')[-1]

	for ap in range(1,393):
		sky_aps=nearest_3_sky(ap)
		if sky_aps==-1:
			logging.error('There are no sky fibres in image %s.' % file)
		if os.path.exists(file+'sky1.fits'):	
			os.remove(file+'sky1.fits')
		if os.path.exists(file+'sky2.fits'):	
			os.remove(file+'sky2.fits')
		if os.path.exists(file+'sky3.fits'):	
			os.remove(file+'sky3.fits')
		iraf.scopy(input=file, output=file+'sky1.fits', apertures=sky_aps[0], Stdout="/dev/null")
		iraf.scopy(input=file, output=file+'sky2.fits', apertures=sky_aps[1], Stdout="/dev/null")
		iraf.scopy(input=file, output=file+'sky3.fits', apertures=sky_aps[2], Stdout="/dev/null")
		iraf.sarith(input1=file+'sky1.fits', op='*', input2=fibre_throughputs_dict_use[ap]/fibre_throughputs_dict_use[sky_aps[0]], output=file+'sky1_scaled.fits', clobber='no', merge='no', ignorea='yes', Stdout="/dev/null")
		iraf.sarith(input1=file+'sky2.fits', op='*', input2=fibre_throughputs_dict_use[ap]/fibre_throughputs_dict_use[sky_aps[1]], output=file+'sky2_scaled.fits', clobber='no', merge='no', ignorea='yes', Stdout="/dev/null")
		iraf.sarith(input1=file+'sky3.fits', op='*', input2=fibre_throughputs_dict_use[ap]/fibre_throughputs_dict_use[sky_aps[2]], output=file+'sky3_scaled.fits', clobber='no', merge='no', ignorea='yes', Stdout="/dev/null")
		iraf.scombine(input=file+'sky1.fits, '+file+'sky2.fits, '+file+'sky3.fits', output=file+'sky_scaled.fits', apertures='', group='all', combine='median', Stdout="/dev/null")
		iraf.sarith(input1=file, op='-', input2=file+'sky_scaled.fits', clobber='yes', merge='yes', ignorea='yes', output='nosky_'+file, apertures=ap,  Stdout="/dev/null")
		
		"""
		#test fitting sky the same way telurics are fit
		hdul=fits.open(file)
		spec=hdul[0].data[ap-1]
		hdul.close()
		hdul=fits.open('nosky_'+file)
		if ap==1:
			no_sky_spec=hdul[0].data
		else:
			no_sky_spec=hdul[0].data[ap-1]
		hdul.close()
		params = Parameters()
		params.add('scale', value=1.0, min=0.7, max=1.3)
		print no_sky_spec.shape
		minner = Minimizer(sky_diff, params, fcn_args=(spec[1800:],spec[1800:]-no_sky_spec[1800:]))
		result = minner.minimize(method='brute')
		print file, report_fit(result)
		res=result.params['scale'].value*1.08
		#print res
		#print file, ap, [sky_diff_a(i, spec,spec-no_sky_spec) for i in np.linspace(0.9,1.1,30)]
		#fig=figure(0)
		#ax=fig.add_subplot(111)
		#ax.plot(np.linspace(0.7,1.3,60), [sky_diff_a(i, spec,spec-no_sky_spec) for i in np.linspace(0.7,1.3,60)], 'ko')
		#ax.plot(spec,'k-', lw=0.5)
		#ax.plot(spec-(spec-no_sky_spec),'r-', lw=0.5)
		#ax.plot(spec-0.5*(spec-no_sky_spec),'g-', lw=0.5)
		#ax.plot(spec-1.5*(spec-no_sky_spec),'b-', lw=0.5)
		#ax.plot(spec-no_sky_spec, 'r-')
		#ax.plot(0.5*(spec-no_sky_spec), 'g-')
		#ax.plot(1.5*(spec-no_sky_spec), 'b-')
		#show()
		iraf.sarith(input1=file+'sky_scaled.fits', op='*', input2=res, output=file+'sky_scaled2.fits', clobber='no', merge='no', ignorea='yes', Stdout="/dev/null")
		iraf.sarith(input1=file, op='-', input2=file+'sky_scaled2.fits', clobber='yes', merge='yes', ignorea='yes', output='nosky2_'+file, apertures=ap,  Stdout="/dev/null")
		os.remove(file+'sky_scaled2.fits')
		
		"""

		os.remove(file+'sky_scaled.fits')
		os.remove(file+'sky1_scaled.fits')
		os.remove(file+'sky2_scaled.fits')
		os.remove(file+'sky3_scaled.fits')

	# save sky spectrum and processed spectrum
	iraf.sarith(input1=file, op='-', input2='nosky_'+file, output='sky_'+file)
	os.remove(file)
	iraf.scopy(input='nosky_'+file, output=file)
	os.remove('nosky_'+file)
	os.remove(file+'sky1.fits')
	os.remove(file+'sky2.fits')
	os.remove(file+'sky3.fits')
	iraf.flprcache()
	os.chdir('../../../../')

def remove_sky(date, method='nearest', thr_method='flat', ncpu=1):
	"""
	Removes sky. Replaces *.ms.fits images with reduced images and creates sky_*.ms.fits files.

	Parameters:
		date (str): date string (e.g. 190210)
		method (str): 'nearest' (default) or 'nearest3'. 'nearest' will take the sky from the nearest sky fibre. 'nearest3' will median nearest three sky fibres.
		thr_method (str): 'flat' (default) or 'mags' or 'both'. 'mags' calculates fibre throughputs from the discrepancy between magnitudes and fluxes of observed stars (averaged over the whole night). 'flat' calculates fibre throughputs from the masterflat (individually for each COB). 'both' is the average of first two methods.
		ncpu (int): number of processes to use for parallelization (default is 1, so no parallelization)

	Returns:
		none

	To do:
		More options for different methods of sky removal
		Urgent: separate magnitude based throughput for plate 0 and plate 1

	"""

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.onedspec(_doprint=0,Stdout="/dev/null")

	logging.info('Removing sky.')
	for ccd in [4,3,2,1]:
		# This first part calculates fibre throughputs based on magnitudes
		fibre_throughputs_dict_mags={}
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:
			if not os.path.isfile(cob+'/masterarc.ms.fits'):
				logging.warning('Masterarc was not found in ccd %d of COB %s, fibre throughput will not be estimated for this ccd-COB combination' % (ccd, cob.split('/')[-1]))
				continue

			files=glob.glob("%s/[01-31]*.fits" % cob)
			valid_files=[]
			for f in files:
				if '.ms' not in f: valid_files.append(f)

			#create a dict from fibre table
			table_name='fibre_table_'+valid_files[0].split('/')[-1]
			hdul=fits.open('/'.join(valid_files[0].split('/')[:-1])+'/'+table_name)
			fibre_table=hdul[1].data
			hdul.close
			fibre_table_dict={}
			n=0
			for i in fibre_table:
				if i[8]=='F': 
					pass
				else: 
					n+=1
					fibre_table_dict[n]=i
			fibre_table_dict=defaultdict(lambda:('FIBRE NOT IN USE', 0.0, 0.0, 0, 0, 0, 0, 0.0, 'N', 0, 0.0, 0, 'Not in use', '0', 0.0, 0.0, 0.0), fibre_table_dict)

			# create a dict of fibre throughputs based on magnitudes. This is done for the whole night, because some fibres are positioned on the sky and don't have an associated magnitude. On a different exposure these fibres are used for stars.
			files=glob.glob("%s/[01-31]*.ms.fits" % cob)
			file_data_all=[]
			for file in files:
				hdul=fits.open(file)
				file_data=np.median(hdul[0].data,axis=1)
				file_data_all.append(file_data)
				hdul.close
			file_data_med=np.median(np.array(file_data_all),axis=0)

			for ap in range(1,393):
				mag=fibre_table_dict[ap][10]
				flux=file_data_med[ap-1]
				if 1.0<mag<19.0: 
					if ap in fibre_throughputs_dict_mags:
						fibre_throughputs_dict_mags[ap].append(flux/np.power(10,(mag-19.0)/-2.5))
					else:
						fibre_throughputs_dict_mags[ap]=[(flux/np.power(10,(mag-19.0)/-2.5))]

		for ap in fibre_throughputs_dict_mags:
			fibre_throughputs_dict_mags[ap]=np.median(fibre_throughputs_dict_mags[ap])

		logging.info('Fibre throughputs calculated for ccd %s.' % ccd)

		# The second part removes sky
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		args=[]
		for cob in cobs:	
			files=glob.glob("%s/[01-31]*.fits" % cob)
			valid_files=[]
			for f in files:
				if '.ms' not in f:
					valid_files.append(f)

			if len(valid_files) == 0:
				error_str = 'No spectra files found for sky removal in ccd %d of COB %s' % (ccd, cob.split('/')[-1])
				logging.error(error_str)
				continue

			#create a dict from fibre table
			table_name='fibre_table_'+valid_files[0].split('/')[-1]
			hdul=fits.open('/'.join(valid_files[0].split('/')[:-1])+'/'+table_name)
			fibre_table=hdul[1].data
			hdul.close
			fibre_table_dict={}
			n=0
			for i in fibre_table:
				if i[8]=='F': 
					pass
				else: 
					n+=1
					fibre_table_dict[n]=i
			fibre_table_dict=defaultdict(lambda:('FIBRE NOT IN USE', 0.0, 0.0, 0, 0, 0, 0, 0.0, 'N', 0, 0.0, 0, 'Not in use', '0', 0.0, 0.0, 0.0), fibre_table_dict)

			#create a dict of fibre throughputs
			fibre_throughputs_dict={}
			hdul=fits.open(cob+'/masterflat.ms.fits')
			masterflat_data=np.median(hdul[0].data,axis=1)
			hdul.close
			for ap in range(1,393):
				fibre_throughputs_dict[ap]=masterflat_data[ap-1]/max(masterflat_data)

			# fibre_throughputs_dict_use is the one throughput dictionary that will be used. Decide here which throughput (calculated from magnitudes, flat or combinantion) will go into this dictionary.
			fibre_throughputs_dict_use={}
			fibre_throughputs_mags_max=-100
			for ap in fibre_throughputs_dict_mags:
				if fibre_throughputs_dict_mags[ap]>fibre_throughputs_mags_max: fibre_throughputs_mags_max=fibre_throughputs_dict_mags[ap]
			for ap in fibre_throughputs_dict_mags:
				fibre_throughputs_dict_mags[ap]=fibre_throughputs_dict_mags[ap]/fibre_throughputs_mags_max
			if thr_method=='both':
				for ap in range(1,393):
					if ap in fibre_throughputs_dict_mags:
						fibre_throughputs_dict_use[ap]=(fibre_throughputs_dict_mags[ap]+fibre_throughputs_dict[ap])/2.
					else:
						fibre_throughputs_dict_use[ap]=fibre_throughputs_dict[ap]
			if thr_method=='flat':
				fibre_throughputs_dict_use=fibre_throughputs_dict
			if thr_method=='mags': # in case some aps are never positioned on a star, use the throuhput measured from a masterflat
				for ap in range(1,393):
					if ap in fibre_throughputs_dict_mags:
						fibre_throughputs_dict_use[ap]=fibre_throughputs_dict_mags[ap]
					else:
						fibre_throughputs_dict_use[ap]=fibre_throughputs_dict[ap]

			fibre_throughputs_dict_use=defaultdict(lambda:1.0, fibre_throughputs_dict_use)

			#list of ms files
			files=glob.glob("%s/[01-31]*.ms.fits" % cob)

			#change dicts into arrays, because parallelization fails with dicts. First entry in the array will be unused, because apertures are counted with 1,2,3,4, so element 0 is necessary in the array, but was not in the dict
			fibre_throughputs_arr_use=[fibre_throughputs_dict_use[-1]]
			fibre_table_arr=[fibre_table_dict[-1]]
			for ap in range(1,393):
				fibre_throughputs_arr_use.append(fibre_throughputs_dict_use[ap])
				fibre_table_arr.append(fibre_table_dict[ap][8])

			#save fibre throughputs
			np.save(cob+'/fibre_throughputs', fibre_throughputs_arr_use)

			for file in files:
				args.append([cob,file,fibre_throughputs_arr_use,fibre_table_arr])

		iraf.flprcache()

		if method=='nearest':
			# First method is where we use the nearest sky fibre to remove sky from object spectra. This is good, because the resolutions of the sky and object spectra are as close as possible.
			pool = Pool(processes=ncpu)
			pool.map(remove_sky_nearest, args)
			pool.close()
			#for arg in args:
			#	remove_sky_nearest(arg)

		if method=='nearest3':
			# Second method is where we use three nearest sky spectra. Resolution is still somewhat similar, but we can take a median of three sky spectra and thus remove any remaining cosmics
			pool = Pool(processes=ncpu)
			pool.map(remove_sky_nearest3, args)
			pool.close()
			#for arg in args:
			#	remove_sky_nearest3(arg)


def remove_telurics(date):
	"""
	Removes teluric absorptions.

	Parameters:
		date (str): date string (e.g. 190210)
	
	Returns:
		none
	"""


	"""
	# Use the following part of the code to generate HITRAN atmospheric absorption spectra. You probably don't need this. Spectra are already in folder hitran_data
	def wavenumber2l(w):
		return 1./w/0.00000001

	def l2wavenumber(l):
		return 1./l/0.00000001

	def vac2air(l):
		return l*(1.-0.000275)

	from hapi import *
	db_begin('hitran_data')

	fetch_by_ids('H2O', [1,2,3,4], l2wavenumber(8000), l2wavenumber(4000))
	nu, coef=absorptionCoefficient_Lorentz(SourceTables='H2O', Environment={'p':0.92,'T':286.}, HITRAN_units=False)
	nu1, trans1=transmittanceSpectrum(nu,coef, Environment={'l':9.1e2,'T':296.})
	nu1, trans1, i1_, i2_, slit1_=convolveSpectrum(nu1,trans1, SlitFunction=SLIT_GAUSSIAN, Resolution=0.03)

	fetch_by_ids('O2', [36,37,38], l2wavenumber(8000), l2wavenumber(4000))
	nu, coef=absorptionCoefficient_Lorentz(SourceTables='O2', Environment={'p':0.92,'T':286.}, HITRAN_units=False)
	nu2, trans2=transmittanceSpectrum(nu,coef, Environment={'l':0.95e6,'T':296.})
	nu2, trans2, i3_, i4_, slit2_=convolveSpectrum(nu2,trans2, SlitFunction=SLIT_GAUSSIAN, Resolution=0.03)

	x=np.linspace(7584-10,7888+10,100000)
	trans1=np.interp(x, vac2air(wavenumber2l(nu1))[::-1], trans1[::-1], left=1.0, right=1.0)
	trans2=np.interp(x, vac2air(wavenumber2l(nu2))[::-1], trans2[::-1], left=1.0, right=1.0)

	with open('trans_H2O_CCD4.txt', 'w') as f:
		for i,j in zip(x,trans1):
			f.write(str(i)+'\t'+str(j)+'\n')

	with open('trans_O2_CCD4.txt', 'w') as f:
		for i,j in zip(x,trans2):
			f.write(str(i)+'\t'+str(j)+'\n')
	"""

	def total_variation(a, spec, trans_h2o, trans_o2):
		"""
		Calculates total variation of a signal over a kernel. V(y)=\sum_i \sum_n \left| y_{n+i} - y_n \right| w(i)

		Parameters:
			a (array): parameters (scale, radial velocity, resolution)
			spec (array): array representing a 1D spectrum (columns are wavelength and flux)
			trans (array): atmosphere transmisivity spectrum
	
		Returns:
			total variation (float)

		To Do:
			fine tune the total variation kernel
			fit rv shift properly (as radial velocity, not as a shift of wavelengths)
			make it faster with minimizer convergence criterium. Somehow I can't get it to work.
		"""
		trans_h2o_l=-np.log(trans_h2o[:,1])
		trans_h2o_s=np.exp(-a['scale_h2o'].value*trans_h2o_l)
		trans_o2_l=-np.log(trans_o2[:,1])
		trans_o2_s=np.exp(-a['scale_o2'].value*trans_o2_l)
		x=np.arange(-3*int(a['res'].value),3*int(a['res'].value)+1)
		k=CustomKernel(np.exp(-0.693147*np.abs(2*x/a['res'].value)**a['b'].value))
		trans_h2o_c=convolve(trans_h2o_s,k)
		trans_h2o_interp=np.interp(spec[:,0],trans_h2o[:,0]+a['vr'].value, trans_h2o_c)
		trans_o2_c=convolve(trans_o2_s,k)
		trans_o2_interp=np.interp(spec[:,0],trans_o2[:,0]+a['vr'].value, trans_o2_c)
		# Below is a crude implementation of a kernel
		div=spec[:,1]/trans_h2o_interp/trans_o2_interp
		div_roll1=np.roll(div,1)
		tot_var=np.sum(abs(div-div_roll1))*0.4
		div_roll2=np.roll(div,2)
		tot_var+=np.sum(abs(div-div_roll2))*0.3
		div_roll3=np.roll(div,3)
		tot_var+=np.sum(abs(div-div_roll3))*0.2
		div_roll4=np.roll(div,4)
		tot_var+=np.sum(abs(div-div_roll4))*0.1
		return tot_var

	iraf.noao(_doprint=0,Stdout="/dev/null")
	iraf.onedspec(_doprint=0,Stdout="/dev/null")
	iraf.dataio(_doprint=0,Stdout="/dev/null")

	# create a dictionary of fitted parameters
	trans_params={}

	# we start fitting with ccd4 where telluric lines are strongest. Fitted parameters are then saved and used for telluric correction in other three bands. Parameters in other three bands can either be improved with a new fit of left fixed.
	for ccd in [4,3,2,1]:
		# load telluric spectra
		trans_h2o=np.loadtxt('aux/trans_H2O_CCD%s.txt' % ccd, delimiter='\t', dtype=float)
		trans_o2=np.loadtxt('aux/trans_O2_CCD%s.txt' % ccd, delimiter='\t', dtype=float)
		trans_hitran=np.ones(trans_h2o.shape)
		trans_hitran[:,0]=trans_h2o[:,0]
		trans_hitran[:,1]=trans_h2o[:,1]*trans_o2[:,1]

		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:	
			files=glob.glob("%s/[01-31]*.fits" % cob)
			valid_files=[]
			for f in files:
				if '.ms' not in f: valid_files.append(f)
			
			#create a dict from fibre table
			table_name='fibre_table_'+valid_files[0].split('/')[-1]
			hdul=fits.open('/'.join(valid_files[0].split('/')[:-1])+'/'+table_name)
			fibre_table=hdul[1].data
			hdul.close
			fibre_table_dict={}
			n=0
			for i in fibre_table:
				if i[8]=='F': 
					pass
				else: 
					n+=1
					fibre_table_dict[n]=i
			fibre_table_dict=defaultdict(lambda:('FIBRE NOT IN USE', 0.0, 0.0, 0, 0, 0, 0, 0.0, 'N', 0, 0.0, 0, 'Not in use', '0', 0.0, 0.0, 0.0), fibre_table_dict)

			files=glob.glob("%s/[01-31]*.ms.fits" % cob)
			files=sorted(files)

			# find valid apertures
			failed_wav=np.load('%s/failed_apertures.npy' % cob)

			for file in files:
				aps=[]
				for ap in fibre_table_dict:
					if fibre_table_dict[ap][8]=='P' and ap not in failed_wav: 
						aps.append(str(ap))
				aps=','.join(aps)
				# write them into a file, because a list is too long for iraf
				with open('%s/aps' % cob, 'w') as f:
					f.write(aps)
				# combine spectra from valid apertures, so we only do one tellurics fit per image
				exposure_number=int(file[-12:-8])
				iraf.scombine(input=file, output='/'.join(file.split('/')[:-1])+'/combined_'+file.split('/')[-1], combine='median', apertures='@%s/aps' % cob, group='all', Stdout="/dev/null")
				os.remove('%s/aps' % cob)
				iraf.wspectext(input='/'.join(file.split('/')[:-1])+'/combined_'+file.split('/')[-1], output='/'.join(file.split('/')[:-1])+'/combined_'+file.split('/')[-1][:-5]+'.txt', header='no', Stdout="/dev/null")
				data=np.loadtxt('/'.join(file.split('/')[:-1])+'/combined_'+file.split('/')[-1][:-5]+'.txt')[10:-10] # first and last 10 pixels are truncated, because there are problems with the median due to different wavelength sollutions
				# start fitting with ccd 4 and write the results into the dictionary. If an exposure is missing, ccd 3 will be in the dictionary and so on
				if exposure_number in trans_params:
					# if this exposure was already fit, use the fitted parameters as initial conditions 
					# initial conditions depend on which ccd we are fitting
					if ccd==3:
						params = Parameters()
						params.add('scale_h2o', value=trans_params[exposure_number][0], min=trans_params[exposure_number][0]*0.7, max=trans_params[exposure_number][0]*1.3)
						params.add('scale_o2', value=0.1, min=0.01, max=2.0, vary=False) # there is no O2 in red ccd
						params.add('vr', value=0.0, min=-0.05, max=0.05)
						params.add('res', value=82.0, min=75.0, max=95.0)
						params.add('b', value=2.08, vary=False)
					if ccd==2:
						params = Parameters()
						params.add('scale_h2o', value=trans_params[exposure_number][0], min=trans_params[exposure_number][0]*0.7, max=trans_params[exposure_number][0]*1.3)
						params.add('scale_o2', value=trans_params[exposure_number][1], min=trans_params[exposure_number][1]*0.7, max=trans_params[exposure_number][1]*1.3)
						params.add('vr', value=0.0, min=-0.05, max=0.05)
						params.add('res', value=75.0, min=65.0, max=90.0)
						params.add('b', value=2.15, vary=False)
					if ccd==1:
						params = Parameters()
						params.add('scale_h2o', value=trans_params[exposure_number][0], vary=False) # use already measured values, because tellurics in blue are too weak to fit
						params.add('scale_o2', value=0.1, min=0.01, max=2.0, vary=False) # there is no O2 in blue ccd
						params.add('vr', value=0.0, min=-0.05, max=0.05)
						params.add('res', value=65.0, min=55.0, max=90.0)
						params.add('b', value=2.4, vary=False)

					minner = Minimizer(total_variation, params, fcn_args=(data,trans_h2o,trans_o2))
					result = minner.minimize(method='lbfgsb')
					print file, report_fit(result)
					res=[result.params['scale_h2o'].value, result.params['scale_o2'].value, result.params['vr'].value, result.params['res'].value, result.params['b'].value]
				else:
					# if this is the first time fitting this exposure, start from arbitrary initial conditions
					params = Parameters()
					params.add('scale_h2o', value=0.9, min=0.01, max=2.0)
					params.add('scale_o2', value=0.1, min=0.01, max=2.0)
					params.add('vr', value=0.0, min=-0.05, max=0.05)
					if ccd==4: 
						params.add('res', value=98.0, min=85, max=110)
						params.add('b', value=2.01, vary=False)
					if ccd==3: 
						params.add('res', value=82.0, min=75, max=95)
						params.add('b', value=2.08, vary=False)
					if ccd==2: 
						params.add('res', value=75.0, min=65, max=90)
						params.add('b', value=2.15, vary=False)
					if ccd==1: 
						params.add('res', value=65.0, min=55, max=90)
						params.add('b', value=2.4, vary=False)
					minner = Minimizer(total_variation, params, fcn_args=(data,trans_h2o,trans_o2))
					result = minner.minimize(method='lbfgsb')
					print file, report_fit(result)
					res=[result.params['scale_h2o'].value, result.params['scale_o2'].value, result.params['vr'].value, result.params['res'].value, result.params['b'].value]
					trans_params[exposure_number]=res
				os.remove('/'.join(file.split('/')[:-1])+'/combined_'+file.split('/')[-1])
				os.remove('/'.join(file.split('/')[:-1])+'/combined_'+file.split('/')[-1][:-5]+'.txt')

				try:
					os.remove('/'.join(file.split('/')[:-1])+'/notel_'+file.split('/')[-1])
				except:
					pass
				try:
					os.remove('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1])
				except:
					pass

				# create a fits spectrum from the fitted hitran spectrum
				trans_h2o_l=-np.log(trans_h2o[:,1])
				trans_h2o_s=np.exp(-res[0]*trans_h2o_l)
				trans_o2_l=-np.log(trans_o2[:,1])
				trans_o2_s=np.exp(-res[1]*trans_o2_l)
				x=np.arange(-200,201)
				k=CustomKernel(np.exp(-0.693147*np.abs(2*x/res[3])**res[4]))
				trans_h2o_c=convolve(trans_h2o_s,k)
				trans_o2_c=convolve(trans_o2_s,k)
				trans_hitran_final=trans_h2o_c*trans_o2_c

				#trans_l=-np.log(trans_hitran[:,1])
				#trans_c=np.exp(-res[0]*trans_l)
				#g=Gaussian1DKernel(stddev=res[2])
				#trans_hitran_final=convolve(trans_c,g)
				with open('/'.join(file.split('/')[:-1])+'/telurics.txt', 'w') as f:
					csv.writer(f, delimiter=' ').writerows(zip(trans_hitran[:,0]+res[2], trans_hitran_final))
				iraf.rspectext(input='/'.join(file.split('/')[:-1])+'/telurics.txt', output='/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], dtype='interp')
				os.remove('/'.join(file.split('/')[:-1])+'/telurics.txt')

				# remove teluric spectrum from science spectra
				iraf.sarith(input1=file, op='/', input2='/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], output='/'.join(file.split('/')[:-1])+'/notel_'+file.split('/')[-1], Stdout="/dev/null", ignoreaps='yes')
				# create telurics spectra samplet the same as original spectra
				os.remove('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1])
				iraf.sarith(input1=file, op='/', input2='/'.join(file.split('/')[:-1])+'/notel_'+file.split('/')[-1], output='/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], Stdout="/dev/null")
				# replace original file with reduced file
				os.remove(file)
				iraf.scopy(input='/'.join(file.split('/')[:-1])+'/notel_'+file.split('/')[-1], output=file)		
				os.remove('/'.join(file.split('/')[:-1])+'/notel_'+file.split('/')[-1])
				iraf.flprcache()


				#if ccd<5:
				#	fig=figure('test')
				#	ax=fig.add_subplot(111)
				#	ax.plot(data[:,0], data[:,1]/np.percentile(data[:,1],95), 'k-', label='Input spectrum')
				#	trans_interp=np.interp(data[:,0],trans_hitran[:,0]+res[2], trans_hitran_final)
				#	ax.plot(data[:,0], trans_interp,'r-', label='Model')
				#	ax.plot(data[:,0],data[:,1]/np.percentile(data[:,1],95)/trans_interp, 'g-', label='Corrected spectrum')
				#	ax.set_xlabel('Wavelength / A')
				#	ax.set_ylabel('flux')
				#	ax.legend()
				#	show()

def create_final_spectra_proc(args):
	"""
	Multiprocessing wrapper for create_final_spectra
	"""
	ccd, cob, files=args

	# save individual spectra
	for file in files:
		#create a dict from fibre table
		fibre_table_file='/'.join(file.split('/')[:-1])+'/fibre_table_'+file.split('/')[-1].replace('.ms', '')
		hdul=fits.open(fibre_table_file)
		fibre_table=hdul[1].data
		hdul.close
		fibre_table_dict={}
		n=0
		for fibre,i in enumerate(fibre_table):
			if i[8]=='F': 
				pass
			else: 
				n+=1
				fibre_table_dict[n]=np.append(np.array(i), np.array([fibre+1]))
		fibre_table_dict=defaultdict(lambda:np.array(['FIBRE NOT IN USE', 0.0, 0.0, 0, 0, 0, 0, 0.0, 'N', 0, 0.0, 0, 'Not in use', '0', 0.0, 0.0, 0.0, fibre+1]), fibre_table_dict)

		#read saved b_values
		if os.path.exists(cob+'/b_values.npy'): b_values=np.load(cob+'/b_values.npy')
		else: b_values=np.array([2.0]*392)

		#make a normalized spectrum, because this will be added to the final fits files
		iraf.continuum(input=file, output='/'.join(file.split('/')[:-1])+'/norm_'+file.split('/')[-1], order=5, interactive='no', low_reject=2.0, high_reject=0.0, niterate=5, naverage=11, Stdout="/dev/null")

		#make a sqrt spectrum for adding errors to it
		iraf.sarith(input1=file, op='^', input2='-0.5', output='/'.join(file.split('/')[:-1])+'/err_'+file.split('/')[-1], apertures='', Stdout="/dev/null")

		#make a cross_talk spectrum placeholder
		iraf.sarith(input1=file, op='*', input2='0.0', output='/'.join(file.split('/')[:-1])+'/cross_'+file.split('/')[-1], apertures='', Stdout="/dev/null")

		#linearize all spectra
		iraf.disptrans(input=file, output='', linearize='yes', units='angstroms', Stdout="/dev/null")
		iraf.disptrans(input='/'.join(file.split('/')[:-1])+'/norm_'+file.split('/')[-1], output='', linearize='yes', units='angstroms', Stdout="/dev/null")
		if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]): iraf.disptrans(input='/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1], output='', linearize='yes', units='angstroms', Stdout="/dev/null")
		if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]): iraf.disptrans(input='/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], output='', linearize='yes', units='angstroms', Stdout="/dev/null")
		if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]): iraf.disptrans(input='/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1], output='', linearize='yes', units='angstroms', Stdout="/dev/null")
		if os.path.exists('/'.join(file.split('/')[:-1])+'/cross_'+file.split('/')[-1]): iraf.disptrans(input='/'.join(file.split('/')[:-1])+'/cross_'+file.split('/')[-1], output='', linearize='yes', units='angstroms', Stdout="/dev/null")
		if os.path.exists('/'.join(file.split('/')[:-1])+'/res_'+file.split('/')[-1]): iraf.disptrans(input='/'.join(file.split('/')[:-1])+'/res_'+file.split('/')[-1], output='', linearize='yes', units='angstroms', Stdout="/dev/null")
		

		for ap in range(1,393):
			if fibre_table_dict[ap][8]=='P':
				filename=str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits'
				iraf.scopy(input=file, output='reductions/results/%s/spectra/all/%s' % (date,filename), apertures=ap, Stdout="/dev/null")
				#add normalized spectrum
				iraf.scopy(input='/'.join(file.split('/')[:-1])+'/norm_'+file.split('/')[-1], output='reductions/results/%s/spectra/all/tmp_norm_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
				iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_norm_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
				os.remove('reductions/results/%s/spectra/all/tmp_norm_%s.fits' % (date, filename))
				iraf.hedit(images='reductions/results/%s/spectra/all/%s[1]' % (date,filename), fields="EXTNAME", value='normalized', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				#add error spectrum (add a copy of the spectrum as a placeholder)
				iraf.scopy(input=file, output='reductions/results/%s/spectra/all/tmp_err_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
				iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_err_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
				os.remove('reductions/results/%s/spectra/all/tmp_err_%s.fits' % (date, filename))
				iraf.hedit(images='reductions/results/%s/spectra/all/%s[2]' % (date,filename), fields="EXTNAME", value='relative_error', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				#add sky spectrum
				if os.path.exists('/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1]):
					iraf.scopy(input='/'.join(file.split('/')[:-1])+'/sky_'+file.split('/')[-1], output='reductions/results/%s/spectra/all/tmp_sky_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_sky_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_sky_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[3]' % (date,filename), fields="EXTNAME", value='sky', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				else:
					iraf.sarith(input1='reductions/results/%s/spectra/all/%s[0]' % (date,filename), op='*', input2='0', output='reductions/results/%s/spectra/all/tmp_sky_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_sky_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_sky_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[3]' % (date,filename), fields="EXTNAME", value='sky', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				#add teluric correction
				if os.path.exists('/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1]):
					iraf.scopy(input='/'.join(file.split('/')[:-1])+'/telurics_'+file.split('/')[-1], output='reductions/results/%s/spectra/all/tmp_tel_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")#scopy does not accept [append] option
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_tel_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_tel_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[4]' % (date,filename), fields="EXTNAME", value='teluric', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				else:
					iraf.sarith(input1='reductions/results/%s/spectra/all/%s[0]' % (date,filename), op='*', input2='0', output='reductions/results/%s/spectra/all/tmp_tel_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.sarith(input1='reductions/results/%s/spectra/all/tmp_tel_%s.fits' % (date, filename), op='+', input2='1.0', output='reductions/results/%s/spectra/all/tmp_tel2_%s.fits' % (date, filename), apertures='', Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_tel2_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_tel_%s.fits' % (date, filename))
					os.remove('reductions/results/%s/spectra/all/tmp_tel2_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[4]' % (date,filename), fields="EXTNAME", value='teluric', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				#add scattered light
				if os.path.exists('/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1]):
					iraf.scopy(input='/'.join(file.split('/')[:-1])+'/scat_'+file.split('/')[-1], output='reductions/results/%s/spectra/all/tmp_scat_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_scat_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_scat_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[5]' % (date,filename), fields="EXTNAME", value='scattered', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				else:
					iraf.sarith(input1='reductions/results/%s/spectra/all/%s[0]' % (date,filename), op='*', input2='0', output='reductions/results/%s/spectra/all/tmp_scat_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_scat_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_scat_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[5]' % (date,filename), fields="EXTNAME", value='scattered', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				#add cross talk spectrum
				if os.path.exists('/'.join(file.split('/')[:-1])+'/cross_'+file.split('/')[-1]):
					iraf.scopy(input='/'.join(file.split('/')[:-1])+'/cross_'+file.split('/')[-1], output='reductions/results/%s/spectra/all/tmp_cross_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_cross_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_cross_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[6]' % (date,filename), fields="EXTNAME", value='cross_talk', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				else:
					iraf.sarith(input1='reductions/results/%s/spectra/all/%s[0]' % (date,filename), op='*', input2='0', output='reductions/results/%s/spectra/all/tmp_cross_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_cross_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_cross_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[6]' % (date,filename), fields="EXTNAME", value='cross_talk', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				#add resolution profile
				if os.path.exists('/'.join(file.split('/')[:-1])+'/res_'+file.split('/')[-1]):
					iraf.scopy(input='/'.join(file.split('/')[:-1])+'/res_'+file.split('/')[-1], output='reductions/results/%s/spectra/all/tmp_res_%s.fits' % (date, filename), apertures=ap, Stdout="/dev/null")
					iraf.imcopy(input='reductions/results/%s/spectra/all/tmp_res_%s.fits' % (date, filename), output='reductions/results/%s/spectra/all/%s[append]' % (date,filename), Stdout="/dev/null")
					os.remove('reductions/results/%s/spectra/all/tmp_res_%s.fits' % (date, filename))
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[7]' % (date,filename), fields="EXTNAME", value='resolution_profile', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				else:
					hdul=fits.open('reductions/results/%s/spectra/all/%s' % (date,filename), mode='update')
					hdul.append(fits.ImageHDU([0]))
					hdul.close()
					iraf.hedit(images='reductions/results/%s/spectra/all/%s[7]' % (date,filename), fields="EXTNAME", value='resolution_profile', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
				#add errors to the error spectrum
				hdul=fits.open('reductions/results/%s/spectra/all/%s' % (date,filename), mode='update')
				noise_squared=hdul[0].data+hdul[3].data+hdul[5].data+hdul[6].data+float(hdul[0].header['RO_NOIS1'])**2
				signal=hdul[0].data
				rel_err=np.sqrt(noise_squared)/signal
				hdul[2].data=rel_err
				hdul.close()
				#add data
				hdul=fits.open('reductions/results/%s/spectra/all/%s' % (date,filename), mode='update')
				#read some info drom fibre table
				object_name=fibre_table_dict[ap][0]
				ra=float(fibre_table_dict[ap][1])/np.pi*180.
				dec=float(fibre_table_dict[ap][2])/np.pi*180.
				pivot=fibre_table_dict[ap][9]
				fibre=fibre_table_dict[ap][-1]
				x=fibre_table_dict[ap][3]
				y=fibre_table_dict[ap][4]
				theta=fibre_table_dict[ap][7]
				mag=fibre_table_dict[ap][10]
				#check aperture flags
				aperture_flags_dict={}
				a=open('%s/aplast' % (cob))
				failed_apertures=np.load('%s/failed_apertures.npy' % (cob))
				for line in a:
					l=line.split()
					if len(l)>0 and l[0]=='begin':
						y=float(l[5])
						n=int(l[3])
						if n in failed_apertures:
							aperture_flags_dict[n]=(y,0)
						else:
							aperture_flags_dict[n]=(y,1)
				#load fitting results for wavelength solution
				wav_dict=np.load(cob+'/wav_fit.npy')
				#load fibre throughputs
				fib_thr=np.load(cob+'/fibre_throughputs.npy')
				#write to fits
				for extension in range(8):
					#clean header
					if 'BARY_%s' % ap in hdul[extension].header: hdul[extension].header['BARYEFF']=(hdul[extension].header['BARY_%s' % ap], 'Barycentric velocity correction in km/s')
					elif 'BARYEFF' in hdul[extension].header: pass
					else: hdul[extension].header['BARYEFF']=('None', 'Barycentric velocity correction was not done')
					if hdul[0].header['SOURCE']=='Plate 1': hdul[extension].header['PLATE']=(1, '2dF plate source')
					elif hdul[0].header['SOURCE']=='Plate 0': hdul[extension].header['PLATE']=(0, '2dF plate source')
					else: hdul[extension].header['PLATE']=('None', 'Problem determining plate number')
					del hdul[extension].header['BARY_*']
					del hdul[extension].header['BANDID*']
					del hdul[extension].header['APNUM*']
					del hdul[extension].header['DC-FLAG*']
					del hdul[extension].header['DCLOG*']
					del hdul[extension].header['WAT1*']
					if 'CD1_1' in hdul[extension].header: del hdul[extension].header['CD1_1']
					if 'AAOPRGID' in hdul[extension].header: del hdul[extension].header['AAOPRGID']
					if 'RUN' in hdul[extension].header: del hdul[extension].header['RUN']
					if 'OBSNUM' in hdul[extension].header: del hdul[extension].header['OBSNUM']
					if 'GPRNUM' in hdul[extension].header: del hdul[extension].header['GRPNUM']
					if 'GPRMEM' in hdul[extension].header: del hdul[extension].header['GRPMEM']
					if 'GPRMAX' in hdul[extension].header: del hdul[extension].header['GRPMAX']
					if 'OBSTYPE' in hdul[extension].header: del hdul[extension].header['OBSTYPE']
					if 'NDFCLASS' in hdul[extension].header: del hdul[extension].header['NDFCLASS']
					if 'FILEORIG' in hdul[extension].header: del hdul[extension].header['FILEORIG']
					if 'RO_GAIN1' in hdul[extension].header: del hdul[extension].header['RO_GAIN1']
					if 'RO_NOIS1' in hdul[extension].header: del hdul[extension].header['RO_NOIS1']
					if 'RO_GAIN' in hdul[extension].header: del hdul[extension].header['RO_GAIN']
					if 'RO_NOISE' in hdul[extension].header: del hdul[extension].header['RO_NOISE']
					if 'ELAPSED' in hdul[extension].header: del hdul[extension].header['ELAPSED']
					if 'TOTALEXP' in hdul[extension].header: del hdul[extension].header['TOTALEXP']
					if 'UTSTART' in hdul[extension].header: del hdul[extension].header['UTSTART']
					if 'UTEND' in hdul[extension].header: del hdul[extension].header['UTEND']
					if 'STSTART' in hdul[extension].header: del hdul[extension].header['STSTART']
					if 'STEND' in hdul[extension].header: del hdul[extension].header['STEND']
					if 'APPRA' in hdul[extension].header: del hdul[extension].header['APPRA']
					if 'APPDEC' in hdul[extension].header: del hdul[extension].header['APPDEC']
					if 'COMMENT' in hdul[extension].header: del hdul[extension].header['COMMENT']
					#compatibility
					hdul[extension].header['APNUM1']=('1 1   ', '')
					hdul[extension].header['CD1_1']=(hdul[0].header['CDELT1'], '')
					hdul[extension].header['CTYPE1']=('Wavelength', '')
					hdul[extension].header['CUNIT1']=('angstroms', '')
					#add galah_id (galahic number) if it exists
					if 'galahic' in object_name:
						hdul[extension].header['GALAH_ID']=(object_name.split('_')[1], 'GALAH id number (if exists)')
					else:
						hdul[extension].header['GALAH_ID']='None'
					#add object name (first column in fibre table)
					hdul[extension].header['OBJ_NAME']=(object_name, 'Object name from the .fld file')
					#add average snr and average resolution
					hdul[extension].header['SNR']=(1.0/np.nanmedian(hdul[2].data), 'Average SNR of the final spectrum')
					hdul[extension].header['RES']=(np.nanmean(hdul[7].data), 'Average resolution (FWHM in angstroms)')
					#add ra and dec of object
					hdul[extension].header['RA_OBJ']=(ra, 'RA of object in degrees')
					hdul[extension].header['DEC_OBJ']=(dec, 'dec of object in degrees')
					#fibre and pivot numbers
					hdul[extension].header['APERTURE']=(ap, 'aperture number (1-392, in image)')
					hdul[extension].header['PIVOT']=(int(pivot), 'pivot number (1-400, in 2dF)')
					hdul[extension].header['FIBRE']=(int(fibre), 'fibre number (1-400, in image)')
					hdul[extension].header['X']=(float(x), 'x position of fibre on plate in um')
					hdul[extension].header['Y']=(float(y), 'y position of fibre on plate in um')
					hdul[extension].header['THETA']=(float(theta), 'bend of fibre in degrees')
					#position of aperture on image
					hdul[extension].header['AP_POS']=(aperture_flags_dict[ap][0], 'Position of aperture in image at x=2000')
					hdul[extension].header['TRACE_OK']=(aperture_flags_dict[ap][1], 'Is aperture trace OK? 1=yes, 0=no')
					#magnitude
					hdul[extension].header['MAG']=(float(mag), 'magnitude from the .fld file')
					#origin
					hdul[extension].header['ORIGIN']='IRAF reduction pipeline'
					hdul[extension].header['PIPE_VER']=('6.0', 'IRAF reduction pipeline version')
					#stellar parameters
					hdul[extension].header['RV']=('None', 'radial velocity (one arm) in km/s')
					hdul[extension].header['E_RV']=('None', 'radial velocity uncertainty in km/s')
					hdul[extension].header['RV_OK']=('None', 'Did RV pipeline converge? 1=yes, 0=no')
					hdul[extension].header['RVCOM']=('None', 'Combined radial velocity in km/s')
					hdul[extension].header['E_RVCOM']=('None', 'radial velocity uncertainty in km/s')
					hdul[extension].header['RVCOM_OK']=('None', 'Did RV pipeline converge? 1=yes, 0=no')
					hdul[extension].header['TEFF']=('None', 'T_eff in K')
					hdul[extension].header['LOGG']=('None', 'log g in log cm/s^2')
					hdul[extension].header['MET']=('None', 'Metallicity in dex')#more parameters?
					hdul[extension].header['PAR_OK']=('None', 'Are parameters trustworthy? 1=yes, 0=no')
					#set exposure of the resolution profile to the same value as for spectra
					hdul[extension].header['EXPOSED']=hdul[0].header['EXPOSED']
					#correct times
					hdul[extension].header['UTMJD']=hdul[0].header['UTMJD']+hdul[0].header['EXPOSED']/3600./24./2.#svae MJD in the middle of teh exposure
					tm=Time(hdul[extension].header['UTMJD'], format='mjd')
					hdul[extension].header['UTDATE']=(tm.iso, 'Mean UT date in iso format')#save date and time in iso format in the middle of the exposure
					#convert start and end zd and ha to average ha and zd
					hdul[extension].header['MEAN_ZD']=((hdul[0].header['ZDEND']+hdul[0].header['ZDSTART'])/2., 'Mean zenith distance')
					hdul[extension].header['MEAN_HA']=((hdul[0].header['HAEND']+hdul[0].header['HASTART'])/2., 'Mean hour angle')
					#wavelength solution
					try: wav_rms=float(wav_dict[ap-1][1])
					except: wav_rms='None'
					hdul[extension].header['WAV_RMS']=(wav_rms, 'RMS of the wavelength solution in Angstroms')
					hdul[extension].header['WAVLINES']=(wav_dict[ap-1][0], 'Number of arc lines (used/found)')
					wav_flag=1
					if int(wav_dict[ap-1][0].split('/')[0])<15: wav_flag=0
					if wav_rms=='None' or wav_rms>0.2: wav_flag=0
					hdul[extension].header['WAV_OK']=(wav_flag, 'Is wav. solution OK? 1=yes, 0=no')
					#fibre throughput
					fibre_throughput=fib_thr[ap]
					hdul[extension].header['FIB_THR']=(fibre_throughput, 'Fibre throughput relative to best fibre in field')
					#write down the LSF 
					hdul[extension].header['LSF']=('exp(-0.693147|2x/fwhm|^B)', 'Line spread function')
					hdul[extension].header['LSF_FULL']=('exp(-0.693147|2x/fwhm|^%s)' % round(b_values[ap-1],3), 'Line spread function')
					hdul[extension].header['B']=(round(b_values[ap-1],3), 'Boxiness parameter for LSF')#2.5 is a placeholder
					hdul[extension].header['COMMENT']=('Explanation of the LSF function: function is centred at 0. fwhm is full width at half maximum in angstroms and is given in extension 7, it is wavelength dependent and varies from fibre to fibre as well. It is advised to use the whole resolution profile from extension 7 rather than average resolution in RES keyword. B is a boxiness parameter. It is given in keyword B in the header. LSF_FULL includes B in the function.')
					if extension>0:
						if 'HASTART' in hdul[extension].header: del hdul[extension].header['HASTART']
						if 'HAEND' in hdul[extension].header: del hdul[extension].header['HAEND']
						if 'ZDSTART' in hdul[extension].header: del hdul[extension].header['ZDSTART']
						if 'ZDEND' in hdul[extension].header: del hdul[extension].header['ZDEND']

				del hdul[0].header['HASTART']
				del hdul[0].header['HAEND']
				del hdul[0].header['ZDSTART']
				del hdul[0].header['ZDEND']
				for extension in range(8):
					if 'SOURCE' in hdul[extension].header: del hdul[extension].header['SOURCE']
				#units
				hdul[0].header['BUNIT']=('counts', '')
				hdul[1].header['BUNIT']=('', '')
				hdul[2].header['BUNIT']=('', '')
				hdul[3].header['BUNIT']=('counts', '')
				hdul[4].header['BUNIT']=('', '')
				hdul[5].header['BUNIT']=('counts', '')
				hdul[6].header['BUNIT']=('counts', '')
				hdul[7].header['BUNIT']=('angstroms', '')
				#fix size of extension 7. No need for double precision here
				hdul[7].data=np.array(hdul[7].data, dtype=np.dtype('f4'))
				#fix meanra and meandec for extension 7
				hdul[7].header['MEANRA']=hdul[0].header['MEANRA']
				hdul[7].header['MEANDEC']=hdul[0].header['MEANDEC']

				hdul.close()

				iraf.flprcache()

	#save combined spectra
	for ap in range(1,393):
		if fibre_table_dict[ap][8]=='P':
			filename=str(date)+cob.split('/')[-1][-4:]+'01'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits'
			#sum of all exposures
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[0]')
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/%s' % (date,filename), aperture='', group='all', combine='sum', reject='none', first='no', Stdout="/dev/null")
			#normalised spectrum
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[1]')
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/%s[append]' % (date,filename), aperture='', group='all', combine='average', reject='avsigclip', first='no', scale='none', weight='!EXPOSED', lsigma=3.0, hsigma=3.0, Stdout="/dev/null")
			#error spectrum
			filenames=[]
			filenames_del=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'_tmp2.fits')
				filenames_del.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'_tmp.fits')
				filenames_del.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'_tmp2.fits')
				iraf.sarith(input1='reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[2]', op='*', input2='reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[0]', output='reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'_tmp.fits')#convert relative error to absolute error
				iraf.sarith(input1='reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'_tmp.fits', op='^', input2='2', output='reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'_tmp2.fits')#square absolute errors, because we will add them in quadrature
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/errors_tmp_%s' % (date, filename), aperture='', group='all', combine='sum', reject='none', first='no', scale='none', Stdout="/dev/null")
			for file in filenames_del:
				os.remove(file)
			iraf.sarith(input1='reductions/results/%s/spectra/com/errors_tmp_%s' % (date, filename), op='sqrt', input2='', output='reductions/results/%s/spectra/com/errors_abs_tmp_%s' % (date, filename))
			iraf.sarith(input1='reductions/results/%s/spectra/com/errors_abs_tmp_%s' % (date, filename), op='/', input2='reductions/results/%s/spectra/com/%s[0]' % (date,filename), output='reductions/results/%s/spectra/com/errors_rel_tmp_%s' % (date, filename))
			iraf.imcopy(input='reductions/results/%s/spectra/com/errors_rel_tmp_%s' % (date, filename), output='reductions/results/%s/spectra/com/%s[append]' % (date,filename), Stdout="/dev/null")
			os.remove('reductions/results/%s/spectra/com/errors_abs_tmp_%s' % (date, filename))
			os.remove('reductions/results/%s/spectra/com/errors_rel_tmp_%s' % (date, filename))
			os.remove('reductions/results/%s/spectra/com/errors_tmp_%s' % (date, filename))
			iraf.hedit(images='reductions/results/%s/spectra/com/%s[2]' % (date,filename), fields="EXTNAME", value='relative_error', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")
			#sky
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[3]')
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/%s[append]' % (date,filename), aperture='', group='all', combine='sum', reject='none', first='no', scale='none', Stdout="/dev/null")
			#teluric
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[4]')
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/%s[append]' % (date,filename), aperture='', group='all', combine='average', reject='none', first='no', scale='none', weight='!EXPOSED', Stdout="/dev/null")
			#scattered light
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[5]')
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/%s[append]' % (date,filename), aperture='', group='all', combine='sum', reject='none', first='no', scale='none', Stdout="/dev/null")
			#cross_talk
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[6]')
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/%s[append]' % (date,filename), aperture='', group='all', combine='sum', reject='none', first='no', scale='none', Stdout="/dev/null")
			#resolution profile
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits[7]')
			to_combine=','.join(filenames)
			iraf.scombine(input=to_combine, output='reductions/results/%s/spectra/com/%s[append]' % (date,filename), aperture='', group='all', combine='average', reject='none', first='no', scale='none', weight='!EXPOSED', Stdout="/dev/null")
			#add data
			filenames=[]
			for file in files:
				filenames.append('reductions/results/'+str(date)+'/spectra/all/'+str(date)+file.split('/')[-1][6:10]+'00'+str(fibre_table_dict[ap][9]).zfill(3)+str(ccd)+'.fits')
			#open combined fits file
			hdul=fits.open('reductions/results/%s/spectra/com/%s' % (date,filename), mode='update')
			#open individual headers read only
			headers_dict={}
			for file in filenames:
				hdul_comb=fits.open(file)
				headers_dict[file]=hdul_comb[0].header
				hdul_comb.close()
			#
			#write to fits
			for extension in range(8):
				#write into header which spectra were combined
				for n,file in enumerate(filenames):
					hdul[extension].header['COMB%s' % n]=(file.split('/')[-1][:15], 'Combined spectra')
				#combine UTMJD
				mjd=0
				for file in filenames:
					mjd+=float(headers_dict[file]['UTMJD'])
				hdul[extension].header['UTMJD']=mjd/len(filenames)
				#combine epoch
				epoch=0
				for file in filenames:
					epoch+=float(headers_dict[file]['EPOCH'])
				hdul[extension].header['EPOCH']=epoch/len(filenames)
				#combine times
				tm=Time(hdul[extension].header['UTMJD'], format='mjd')
				hdul[extension].header['UTDATE']=(tm.iso, 'Mean UT date of all exposures in ISO format')
				#add exposure times
				exposed=0
				for file in filenames:
					exposed+=float(headers_dict[file]['EXPOSED'])
				hdul[extension].header['EXPOSED']=(exposed, 'Total exposure time of all combined spectra')
				#trace_ok flag
				traceok=[]
				for file in filenames:
					traceok.append(int(headers_dict[file]['TRACE_OK']))
				if 0 in traceok: hdul[extension].header['TRACE_OK']=0
				else: hdul[extension].header['TRACE_OK']=1
				#combine BARYEFF
				try:
					baryeff=0
					for file in filenames:
						baryeff+=float(headers_dict[file]['BARYEFF'])
					hdul[extension].header['BARYEFF']=baryeff/len(filenames)
				except:
					hdul[extension].header['BARYEFF']=('None', 'Barycentric velocity correction was not done.')
				#correct HA, ZD
				ha=[]
				zd=[]
				for file in filenames:
					ha.append(float(headers_dict[file]['MEAN_HA']))
					zd.append(float(headers_dict[file]['MEAN_ZD']))
				hdul[extension].header['MEAN_HA']=(np.average(ha), 'Mean hour angle')
				hdul[extension].header['MEAN_ZD']=(np.average(zd), 'Mean zenith distance')
				#combine wavelength solution
				wav_flag=[]
				for file in filenames:
					wav_flag.append(int(headers_dict[file]['WAV_OK']))
				hdul[extension].header['WAV_OK']=(min(wav_flag), 'Is wav. solution OK? 1=yes, 0=no')
				#combine resolution maps and B parameter
				# There is only one resolution map for the whole COB. There is also only one wavelength sollution for the whole COB, so the resolution after combining should remain the same.
				#fix origin
				hdul[extension].header['ORIGIN']='IRAF reduction pipeline'
				#edit combined SNR and resolution
				hdul[extension].header['SNR']=(1.0/np.nanmedian(hdul[2].data), 'Average SNR of the final spectrum')
				hdul[extension].header['RES']=(np.nanmean(hdul[7].data), 'Average resolution (FWHM in Angstroms)')

			hdul.close()
			iraf.flprcache()




def create_final_spectra(date, ncpu=1):
	"""
	Turn reduced spectra into final output and combine consecutive exposures.

	Parameters:
		date (str): date string (e.g. 190210)

	Returns:
		none

	To do:
		CHeck if headers are in accordance with Datacentral requirements

	"""

	logging.info('Preparing folders to save final spectra')

	# Create folder structure for final data
	if not os.path.exists('reductions/results'):
		os.makedirs('reductions/results')

	if not os.path.exists('reductions/results/%s' % date):
		os.makedirs('reductions/results/%s' % date)

	if not os.path.exists('reductions/results/%s/spectra' % date):
		os.makedirs('reductions/results/%s/spectra' % date)

	if not os.path.exists('reductions/results/%s/db' % date):
		os.makedirs('reductions/results/%s/db' % date)

	if not os.path.exists('reductions/results/%s/spectra/all' % date):
		os.makedirs('reductions/results/%s/spectra/all' % date)
	if not os.path.exists('reductions/results/%s/spectra/com' % date):
		os.makedirs('reductions/results/%s/spectra/com' % date)
	
	args=[]

	for ccd in [1,2,3,4]:
		cobs=glob.glob("reductions/%s/ccd%s/*" % (date,ccd))
		for cob in cobs:	
			files=glob.glob("%s/[01-31]*.ms.fits" % cob)
			args.append([ccd,cob,files])

	pool = Pool(processes=ncpu)
	pool.map(create_final_spectra_proc, args)
	pool.close()


def create_database(date):
	"""
	Create databse

	Parameters:
		date (str): date string (e.g. 190210)

	Returns:
		none

	To do:
		Add fibre throughput into the db and add flags for poor throughput

	"""

	# Create empty table for this night
	cols=[]
	cols.append(fits.Column(name='sobject_id', format='K'))
	cols.append(fits.Column(name='ra', format='E', unit='deg'))
	cols.append(fits.Column(name='dec', format='E', unit='deg'))
	cols.append(fits.Column(name='mjd', format='D'))
	cols.append(fits.Column(name='utdate', format='A23'))
	cols.append(fits.Column(name='epoch', format='D'))
	cols.append(fits.Column(name='aperture', format='I'))
	cols.append(fits.Column(name='pivot', format='I'))
	cols.append(fits.Column(name='fibre', format='I'))
	cols.append(fits.Column(name='fibre_x', format='E', unit='um'))
	cols.append(fits.Column(name='fibre_y', format='E', unit='um'))
	cols.append(fits.Column(name='fibre_theta', format='E', unit='deg'))
	cols.append(fits.Column(name='plate', format='I', null=None))
	cols.append(fits.Column(name='aperture_position', format='4E'))
	cols.append(fits.Column(name='mean_ra', format='E', unit='deg'))
	cols.append(fits.Column(name='mean_dec', format='E', unit='deg'))
	cols.append(fits.Column(name='mean_zd', format='E', unit='deg'))
	cols.append(fits.Column(name='mean_ha', format='E', unit='deg'))
	cols.append(fits.Column(name='cfg_file', format='A48', null=None))
	cols.append(fits.Column(name='cfg_field_name', format='A56', null=None))
	cols.append(fits.Column(name='obj_name', format='A48', null=None))
	cols.append(fits.Column(name='galah_id', format='K', null=None))
	cols.append(fits.Column(name='snr', format='4E', null=None))
	cols.append(fits.Column(name='fibre_throughput', format='4E', null=None))
	cols.append(fits.Column(name='res', format='4E', null=None, unit='A'))
	cols.append(fits.Column(name='b_par', format='4E', null=None))
	cols.append(fits.Column(name='v_bary_eff', format='E', unit='km/s'))
	cols.append(fits.Column(name='exposed', format='E', unit='s'))
	cols.append(fits.Column(name='mag', format='E'))
	cols.append(fits.Column(name='wav_rms', format='4E', unit='km/s', null=None))
	cols.append(fits.Column(name='wav_n_lines', format='28A7'))
	cols.append(fits.Column(name='rv', format='4E', unit='km/s', null=None))
	cols.append(fits.Column(name='e_rv', format='4E', unit='km/s', null=None))
	cols.append(fits.Column(name='rv_com', format='E', unit='km/s', null=None))
	cols.append(fits.Column(name='e_rv_com', format='E', unit='km/s', null=None))
	cols.append(fits.Column(name='teff', format='E', unit='K', null=None))
	cols.append(fits.Column(name='logg', format='E', unit='cm / s^-2', null=None))
	cols.append(fits.Column(name='met', format='E', null=None))
	cols.append(fits.Column(name='obs_comment', format='A56'))
	cols.append(fits.Column(name='pipeline_version', format='A5'))
	cols.append(fits.Column(name='reduction_flags', format='I'))
	hdu=fits.BinTableHDU.from_columns(cols)
	table=Table.read(hdu)	

	# list of all files
	files=glob.glob("reductions/results/%s/spectra/com/*.fits" % (date))
	# drop last numeral to get a unique list of sobject_ids
	sobjects=[int(file.split('/')[-1][:-6]) for file in files]
	sobjects=set(sobjects)

	for sobject in sobjects:
		#open all four (for four ccds) headers.
		file1="reductions/results/%s/spectra/com/%s1.fits" % (date, sobject)
		hdul1=fits.open(file1)
		header1=hdul1[0].header
		hdul1.close()
		file2="reductions/results/%s/spectra/com/%s2.fits" % (date, sobject)
		hdul2=fits.open(file2)
		header2=hdul2[0].header
		hdul2.close()
		file3="reductions/results/%s/spectra/com/%s3.fits" % (date, sobject)
		hdul3=fits.open(file3)
		header3=hdul3[0].header
		hdul3.close()
		file4="reductions/results/%s/spectra/com/%s4.fits" % (date, sobject)
		hdul4=fits.open(file4)
		header4=hdul4[0].header
		hdul4.close()

		#read parameters from headers
		ra=header1['RA_OBJ']
		dec=header1['DEC_OBJ']
		mean_ra=header1['MEANRA']
		mean_dec=header1['MEANDEC']
		mean_zd=header1['MEAN_ZD']
		mean_ha=header1['MEAN_HA']
		mjd=header1['UTMJD']
		utdate=header1['UTDATE']
		epoch=header1['EPOCH']
		aperture=header1['APERTURE']
		pivot=header1['pivot']
		fibre=header1['fibre']
		fibre_x=header1['X']
		fibre_y=header1['Y']
		fibre_theta=header1['THETA']
		plate=header1['PLATE']
		if plate=='None': plate=None
		aperture_position=[header1['AP_POS'], header2['AP_POS'], header3['AP_POS'], header4['AP_POS']]
		cfg_file=header1['CFG_FILE'].replace(',', ';')#replace commas, so the don't interfere with a csv version of the database
		if cfg_file=='None': cfg_file=None
		cfg_field_name=header1['OBJECT'].replace(',', ';')
		if cfg_field_name=='None': cfg_field_name=None
		obj_name=header1['OBJ_NAME'].replace(',', ';')
		if obj_name=='None': obj_name=None
		galah_id=header1['GALAH_ID']
		if galah_id=='None': galah_id=-1#not sure why None is not working here
		mag=header1['mag']
		wav_rms=[header1['WAV_RMS'], header2['WAV_RMS'], header3['WAV_RMS'], header4['WAV_RMS']]
		wav_rms=[None if i=='None' else i for i in wav_rms]
		wav_n_lines=[header1['WAVLINES'], header2['WAVLINES'], header3['WAVLINES'], header4['WAVLINES']]
		wav_n_lines_arr=[None if i=='None' else i for i in wav_n_lines]
		wav_n_lines=''.join([i.ljust(7, ' ') for i in wav_n_lines_arr])#weird formating because of astropy bugs
		snr=[header1['SNR'], header2['SNR'], header3['SNR'], header4['SNR']]
		snr=[None if i=='None' else i for i in snr]
		fibre_throughput=[header1['FIB_THR'], header2['FIB_THR'], header3['FIB_THR'], header4['FIB_THR']]
		fibre_throughput=[None if i=='None' else i for i in fibre_throughput]
		res=[header1['RES'], header2['RES'], header3['RES'], header4['RES']]
		res=[None if i=='None' else i for i in res]
		b=[header1['B'], header2['B'], header3['B'], header4['B']]
		b=[None if i=='None' else i for i in b]
		v_bary_eff=header1['BARYEFF']
		if v_bary_eff=='None': v_bary_eff=None
		exposed=header1['EXPOSED']
		if exposed=='None': exposed=None
		rv=[header1['rv'], header2['rv'], header3['rv'], header4['rv']]
		rv=[None if i=='None' else i for i in rv]
		e_rv=[header1['E_RV'], header2['E_RV'], header3['E_RV'], header4['E_RV']]
		e_rv=[None if i=='None' else i for i in e_rv]
		rv_com=header1['rvcom']
		if rv_com=='None': rv_com=None
		e_rv_com=header1['e_rvcom']
		if e_rv_com=='None': e_rv_com=None
		teff=header1['TEFF']
		if teff=='None': teff=None
		logg=header1['LOGG']
		if logg=='None': logg=None
		met=header1['MET']
		if met=='None': met=None
		pipeline_version=header1['PIPE_VER']
		obs_comment=header1['COMM_OBS'].replace(',', ';')
		#bin mask reduction flags
		flag=0
		if header1['TRACE_OK']==0: flag+=1
		if header2['TRACE_OK']==0: flag+=2
		if header3['TRACE_OK']==0: flag+=4
		if header4['TRACE_OK']==0: flag+=8
		if header1['WAV_OK']==0: flag+=16
		if header2['WAV_OK']==0: flag+=32
		if header3['WAV_OK']==0: flag+=64
		if header4['WAV_OK']==0: flag+=128
		if header1['RV_OK']==0: flag+=256
		if header2['RV_OK']==0: flag+=512
		if header3['RV_OK']==0: flag+=1024
		if header4['RV_OK']==0: flag+=2048
		if header1['RVCOM_OK']==0: flag+=4096#this one is the same in all 4 ccds
		if header1['PAR_OK']==0: flag+=8192#parameters are calculated over all arms, so this flag is the same for all 4 ccds

		#add parameters into the table
		table.add_row([sobject, ra, dec, mjd, utdate, epoch, aperture, pivot, fibre, fibre_x, fibre_y, fibre_theta, plate, aperture_position, mean_ra, mean_dec, mean_zd, mean_ha, cfg_file, cfg_field_name, obj_name, galah_id, snr, fibre_throughput, res, b, v_bary_eff, exposed, mag, wav_rms, wav_n_lines, rv, e_rv, rv_com, e_rv_com, teff, logg, met, obs_comment, pipeline_version, flag])

	#write table to hdu
	hdu=fits.BinTableHDU(table)

	#write hdu to file
	hdu.writeto('reductions/results/%s/db/%s.fits' % (date, date))

	#open table again and add comments to all columns. MAKE SURE THESE ARE IN THE RIGHT ORDER
	hdul=fits.open('reductions/results/%s/db/%s.fits' % (date, date), mode='update')
	header=hdul[1].header
	header['TTYPE1']=(header['TTYPE1'], 'sobject_id is unique for each GALAH observation')
	header['TTYPE2']=(header['TTYPE2'], 'RA of object in deg')
	header['TTYPE3']=(header['TTYPE3'], 'dec of object in deg')
	header['TTYPE4']=(header['TTYPE4'], 'modified UTC julian date, exposures weighted')
	header['TTYPE5']=(header['TTYPE5'], 'UTC date in ISO 8601 format')
	header['TTYPE6']=(header['TTYPE6'], 'epoch')
	header['TTYPE7']=(header['TTYPE7'], 'aperture number (1-392)')
	header['TTYPE8']=(header['TTYPE8'], 'pivot number (1-400)')
	header['TTYPE9']=(header['TTYPE9'], 'fibre number (1-400)')
	header['TTYPE10']=(header['TTYPE10'], 'x position on 2dF plate')
	header['TTYPE11']=(header['TTYPE11'], 'y position on 2dF plate')
	header['TTYPE12']=(header['TTYPE12'], 'fibre bent on 2dF plate')
	header['TTYPE13']=(header['TTYPE13'], '2dF plate number (0-1)')
	header['TTYPE14']=(header['TTYPE14'], 'aperture position in image (at x=2000)')
	header['TTYPE15']=(header['TTYPE15'], 'RA of telescope position')
	header['TTYPE16']=(header['TTYPE16'], 'dec of telescope position')
	header['TTYPE17']=(header['TTYPE17'], 'zenith distance of telescope position')
	header['TTYPE18']=(header['TTYPE18'], 'hour angle of telescope position')
	header['TTYPE19']=(header['TTYPE19'], 'name of the fibre configuration file')
	header['TTYPE20']=(header['TTYPE20'], 'name of the field in cfg_file')
	header['TTYPE21']=(header['TTYPE21'], 'object name')
	header['TTYPE22']=(header['TTYPE22'], 'galahic id')
	header['TTYPE23']=(header['TTYPE23'], 'mean snr in 4 CCDs')
	header['TTYPE24']=(header['TTYPE24'], 'fibre throughput in 4 CCDs')
	header['TTYPE25']=(header['TTYPE25'], 'mean resolution (FWHM) in 4 CCDs')
	header['TTYPE26']=(header['TTYPE26'], 'LSF B parameter in 4 CCDs')
	header['TTYPE27']=(header['TTYPE27'], 'mean barycentric velocity (already corrected)')
	header['TTYPE28']=(header['TTYPE28'], 'total exposure time')
	header['TTYPE29']=(header['TTYPE29'], 'magnitude as given in cfg_file')
	header['TTYPE30']=(header['TTYPE30'], 'RMS of wavlength calibr. in 4 CCDs')
	header['TTYPE31']=(header['TTYPE31'], 'number of lines found/used for wav. cal.')
	header['TTYPE32']=(header['TTYPE32'], 'radial velocity in 4 CCDs')
	header['TTYPE33']=(header['TTYPE33'], 'rv uncertainty in 4 CCDs')
	header['TTYPE34']=(header['TTYPE34'], 'rv combined from all arms')
	header['TTYPE35']=(header['TTYPE35'], 'combined rv uncertainty')
	header['TTYPE36']=(header['TTYPE36'], 'effective tmperature')
	header['TTYPE37']=(header['TTYPE37'], 'log of surface gravitational acceleration')
	header['TTYPE38']=(header['TTYPE38'], 'metallicity ([M/H])')
	header['TTYPE39']=(header['TTYPE39'], 'comments by observer')
	header['TTYPE40']=(header['TTYPE40'], 'pipeline evrsion')
	header['TTYPE41']=(header['TTYPE41'], 'reduction flags given as a binary mask')

	#fix bug in astropy (formating of string arrays. Last 7 is dropped when data is inserted)
	header['TFORM31']='28A7'#this is wav_n_lines column
	hdul.close()

	#set table name
	iraf.hedit(images='reductions/results/%s/db/%s.fits[1]' % (date, date), fields="EXTNAME", value='table', add='yes', addonly='no', delete='no', verify='no', show='no', update='yes', Stdout="/dev/null")

	# convert table to ascii. add _1, _2, _3, _4 to fields in arrays (always representing four arms)
	csv_db=open('reductions/results/%s/db/%s.csv' % (date, date), 'w')
	hdul=fits.open('reductions/results/%s/db/%s.fits' % (date, date))
	hdu=hdul[1]
	t=Table(hdu.data)
	data=np.array(hdu.data)
	#print line with column names
	str_to_write=[]
	for n,colname in enumerate(t.colnames):
		if data[0][n].shape==(4,):
			for i in range(1,5):
				str_to_write.append(colname+'_'+str(i))
		else:
			str_to_write.append(colname)

	csv_db.write(','.join(str_to_write))
	csv_db.write('\n')
	#add data
	for i in data:
		str_to_write=[]
		for j in i:
			if j.shape==(4,):
				for k in j:
					str_to_write.append(str(k))
			else:
				str_to_write.append(str(j))
		csv_db.write(','.join(str_to_write))
		csv_db.write('\n')
	csv_db.close()
					

def analyze_rv(args):
	"""
	Calculate radial velocity and write it into headers
	"""

	def residual(pars, x, data=None):
		model=pars['amp']*np.exp(-(x-pars['shift'])**2/(2*pars['sigma']**2))+pars['offset']
		#model=gaussian(x, pars['amp'], pars['shift'], pars['sigma'])
		if data is None:
			return model
		return model-data

	#hipass filter in km/s
	filter_limit={3500:250.0, 4000:250.0, 4750:280.0, 5000:300.0, 6000:450.0, 7000:550.0, 9000:800.0, 12000:1500.0, 25000:2500.0, 40000:2500.0}

	#rvs=np.linspace(-3000,3000,3001)
	#rvs=np.linspace(-1500,1500,1501)
	rvs=np.linspace(-1010,1010,506)
	dd=2.0*1010/506.0#step in rv

	sobject, templates=args

	ccf_global=np.zeros(len(rvs))
	ccfs=[]
	specs=[]
	best_template=[]
	for ccd in [1,2,3,4]:
		# open spectrum
		hdulist = fits.open('reductions/results/%s/spectra/com/%s%s.fits' % (date, sobject,ccd))
		crval=hdulist[1].header['CRVAL1']
		crdel=hdulist[1].header['CDELT1']
		f=hdulist[1].data
		hdulist.close()
		l=np.linspace(crval, crval+crdel*(len(f)-1), len(f))
		# ignore first 10 and last 10 pixels, just in case there are problems with normalisation. In CCD 4 ignore more because of telurics.
		if ccd==4:
			f=f[1500:-10]
			l=l[1500:-10]
		else:
			f=f[10:-10]
			l=l[10:-10]
		spec=np.array(zip(l,f), dtype=[('l', 'f8'), ('f', 'f8')])

		# Cross correlate
		# Following S. Zucker, MNRAS, Volume 342, Issue 4, July 2003
		max_ccv=0
		max_ccf=[]
		best_template_tmp=0
		for t,template in enumerate(templates[ccd-1]):
			ccf=[]
			for rv in rvs:
				# convert shift in rv to shift in wavelength
				l=spec['l']*(1-rv/299792.458)
				# normalize spectrum and template so their average is zero
				f_spec=spec['f']-np.average(spec['f'])
				f_temp=np.interp(l,template[4]['l'],template[4]['f'])-np.average(np.interp(l,template[4]['l'],template[4]['f']))
				# calculate cross covariance
				if np.all(f_spec==0): f_spec=f_spec+1
				R=sum(f_temp*f_spec)/len(l)
				# calculate cross corelation
				sf2=np.sum(f_spec*f_spec)/len(f_spec)
				st2=np.sum(f_temp*f_temp)/len(f_temp)
				if sf2==0: sf2=1e-16
				if st2==0: st2=1e-16
				C=R/(np.sqrt(sf2)*np.sqrt(st2))
				ccf.append(C)

			# filter ccf of large trends
			# there should be a separate filter limit for each template.
			# watch out for units if you change rvs array.
			b,a = signal.butter(3,1./filter_limit[template[2]]/2.,'hp', fs=0.5*2.)
			ccf_filtered=signal.filtfilt(b, a, ccf)
			ccf=ccf_filtered+np.average(ccf)-np.average(ccf_filtered)

			#fig=figure(0)
			#ax=fig.add_subplot(111)
			#ax.plot(rvs,ccf,'r-')
			#ax.plot(rvs,ccf_filtered,'g-')
			#show()

			#check if this template produces a better correlation peak
			if max(ccf[50:-50])>max_ccv:
				max_ccv=max(ccf[50:-50])
				max_ccf=ccf
				best_template_tmp=t

		# only use CCF of the template with best correlation
		ccfs.append(max_ccf)
		best_template.append(best_template_tmp)

		#fig=figure(0)
		#ax=fig.add_subplot(211)
		#ax.plot(spec['l'], spec['f'], 'k-')
		#ax.plot(templates[ccd-1][best_template_tmp][4]['l'],templates[ccd-1][best_template_tmp][4]['f'], 'r-')
		#ax=fig.add_subplot(212)
		#ax.plot(rvs, max_ccf, 'k-')
		#show()

	# combine ccfs from all ccds into one ccf
	ccf_global2=1-np.power(np.product(1-np.square(ccfs), axis=0),1.0/4.0)
	ccf_global=np.sqrt(ccf_global2)
	ccf_sgn=np.sign(np.sum(ccfs,axis=0))#FIX THIS. sign first, sum later

	# find peaks
	peaks,other_info=signal.find_peaks(ccf_global, width=5, distance=10, height=0.15, prominence=0.1)

	# filter out valid peaks
	peaks_ok=[]
	if len(peaks)>0:
		for peak,width,power,sgn in zip(peaks, other_info['widths'], other_info['peak_heights'],ccf_sgn[peaks]):
			# valid peak has sgn>=0 (at least two CCDs produce correlation peak)
			# valid peak is significantly stronger than any other peaks
			if sgn>=0 and power/max(ccf_global)>0.5:
				peaks_ok.append([peak,width,power])
	
	# fit only the main peak to get radial velocity
	rv_ind_ar=[]
	sigma_ind_ar=[]
	flag_ind_ar=[]
	if len(peaks_ok)>0:
		# sort peaks by height
		peaks_ok=np.array(peaks_ok)
		peaks_ok=peaks_ok[peaks_ok[:,2].argsort()]
		# fit
		peak_prime=int(peaks_ok[0][0]) # index of peak
		peak_max=peaks_ok[0][2] # height of peak
		peak_width=int(peaks_ok[0][1]) # width of peak
		if len(peaks_ok)>1: min_dist=min(np.diff(np.sort(peaks_ok[:,0]))) # region to fit (if there are two peaks close by, only one must be in the region)
		else: min_dist=3000
		if min_dist<peak_width*0.33: dist=min_dist*0.5
		else: dist=peak_width*0.33
		if dist<3: dist=3
		fit_params = Parameters()
		fit_params.add('amp', value=peak_max-np.median(ccf_global), max=(peak_max-np.median(ccf_global))*1.3, min=(peak_max-np.median(ccf_global))*0.7)
		fit_params.add('sigma', value=peak_width*0.75)
		fit_params.add('shift', value=rvs[peak_prime], max=rvs[peak_prime]+3.0, min=rvs[peak_prime]-3.0)
		fit_params.add('offset', value=np.median(ccf_global))
		out = minimize(residual, fit_params, args=(rvs[peak_prime-int(dist):peak_prime+int(dist)],), kws={'data': ccf_global[peak_prime-int(dist):peak_prime+int(dist)]})
		rv_com=out.params['shift'].value
		# calculate error around the main peak (following S. Zucker, MNRAS, Volume 342, Issue 4, July 2003)
		#second derivative of the fitted gaussian
		fit_second_derivative=np.diff(residual(out.params, rvs),2)
		#ccf_second_derivative=np.diff(ccf_global, 2)
		# factor MN (number of bins) is 4 times 4096 minus edge (10) we cut out minus 1500 for the IR arm = 14814
		sigma2=-1.0/(14814*(fit_second_derivative[peak_prime-1]/dd/dd/dd)/ccf_global[peak_prime]*(ccf_global2[peak_prime])/(1.0-ccf_global2[peak_prime]))# watch derivative units if you change rvs
		sigma_com=np.sqrt(sigma2)
		if sigma_com>25: flag_com=0
		else: flag_com=1
		#report_fit(out)
		#print 'com', rv_com, sigma_com
		fit = residual(out.params, rvs[peak_prime-int(dist):peak_prime+int(dist)])

		# fit individual CCDs
		for ccd in [1,2,3,4]:
			peak_max=ccfs[ccd-1][peak_prime] # height of peak
			fit_params = Parameters()
			fit_params.add('amp', value=peak_max-np.median(ccfs[ccd-1]), max=(peak_max-np.median(ccfs[ccd-1]))*2.0, min=0.0)
			fit_params.add('sigma', value=peak_width*0.75)
			fit_params.add('shift', value=rvs[peak_prime], max=rvs[peak_prime]+12.0, min=rvs[peak_prime]-12.0)
			fit_params.add('offset', value=np.median(ccfs[ccd-1]))
			out = minimize(residual, fit_params, args=(rvs[peak_prime-int(dist):peak_prime+int(dist)],), kws={'data': ccfs[ccd-1][peak_prime-int(dist):peak_prime+int(dist)]})
			rv_ind=out.params['shift'].value
			## calculate error
			fit_second_derivative=np.diff(residual(out.params, rvs),2)
			#ccf_second_derivative=np.diff(ccf_global, 2)
			if ccd<4: sigma2=-1.0/(4076*(fit_second_derivative[peak_prime-1]/dd/dd/dd)/ccfs[ccd-1][peak_prime]*(ccfs[ccd-1][peak_prime])/(1.0-ccfs[ccd-1][peak_prime]))# watch derivative units if you change rvs
			else: sigma2=-1.0/(2586*(fit_second_derivative[peak_prime-1]/dd/dd/dd)/ccfs[ccd-1][peak_prime]*(ccfs[ccd-1][peak_prime])/(1.0-ccfs[ccd-1][peak_prime]))# watch derivative units if you change rvs
			sigma_ind=np.sqrt(sigma2)
			if np.isnan(sigma_ind): sigma_ind='None'
			#report_fit(out)
			rv_ind_ar.append(rv_ind)
			sigma_ind_ar.append(sigma_ind)
			if ccfs[ccd-1][peak_prime]<0.15 or sigma_ind=='None' or np.isnan(sigma_ind) or sigma_ind>25: flag_ind_ar.append(0)
			else: flag_ind_ar.append(1)

	else:#if there are no peaks in the ccf
		rv_com='None'
		sigma_com='None'
		flag_com=0
		rv_ind_ar=['None', 'None', 'None', 'None']
		sigma_ind_ar=['None', 'None', 'None', 'None']
		flag_ind_ar=[0, 0, 0, 0]

	#print rv_com, sigma_com, flag_com
	#print rv_ind_ar, sigma_ind_ar, flag_ind_ar

	for ccd in [1,2,3,4]:
		# open fits file and write rvs into header
		hdulist = fits.open('reductions/results/%s/spectra/com/%s%s.fits' % (date, sobject,ccd), mode='update')
		for extension in range(8):
			hdulist[extension].header['RV']=rv_ind_ar[ccd-1]
			hdulist[extension].header['E_RV']=sigma_ind_ar[ccd-1]
			hdulist[extension].header['RV_OK']=flag_ind_ar[ccd-1]
			hdulist[extension].header['RVCOM']=rv_com
			hdulist[extension].header['E_RVCOM']=sigma_com
			hdulist[extension].header['RVCOM_OK']=flag_com
		hdulist.close()
	



	"""
	print rv_com, sigma_com
	print rv_ind_ar, sigma_ind_ar
	fig=figure(0)
	ax=fig.add_subplot(111)
	ax.plot(rvs, ccfs[0], 'b-')
	ax.plot(rvs, ccfs[1], 'g-')
	ax.plot(rvs, ccfs[2], 'r-')
	ax.plot(rvs, ccfs[3], 'k-')
	ax.plot(rvs, ccf_global, 'y')
	#ax.plot(rvs, ccf_sgn)
	if len(peaks_ok)>0:
		ax.plot(rvs[peak_prime-int(dist):peak_prime+int(dist)], fit, 'm-')
	#if rv_com!='None': ax.set_xlim(rv_com-100,rv_com+100)
	show()
	"""
	
def analyze(date):
	"""
	Calculate radial velocity and atmospheric parameters. Should be done after the data is combined and before the database is generated. Caculated parameters can be written into headers.

	Parameters:
		date (str): date string (e.g. 190210)

	Returns:
		none

	To do:
		Make rv calculations faster
		Detect multiple rv peaks and create flag

	"""

	# calculate radial velocity

	logging.info('Loading radial velocity template spectra.')
	#load file with wavelengths for templates
	w_file='rv_templates/LAMBDA_R20.DAT'
	w=np.loadtxt(w_file, dtype=float)

	#GALAH wavelength ranges (as line numbers in w_file)
	ranges={1:[30814,34458], 2:[39876,43524], 3:[46293,50326], 4:[56018,57630]}

	#templates
	templates=[]

	#create templates
	for ccd in [1, 2, 3, 4]:
		templates_tmp=[]
		for t,template_file in enumerate(['rv_templates/T03500G00P00V000K2ANWNVR20N.ASC','rv_templates/T03500G45P00V000K2ANWNVR20N.ASC','rv_templates/T04000G15P00V000K2ANWNVR20N.ASC','rv_templates/T04000G45P00V000K2ANWNVR20N.ASC','rv_templates/T04750G25P00V000K2ANWNVR20N.ASC','rv_templates/T05000G35P00V000K2ANWNVR20N.ASC','rv_templates/T05000G45P00V000K2ANWNVR20N.ASC','rv_templates/T06000G40P00V000K2ANWNVR20N.ASC','rv_templates/T07000G40P00V000K2ANWNVR20N.ASC','rv_templates/T09000G40P00V000K2ANWNVR20N.ASC','rv_templates/T12000G35P00V000K2SODNVR20N.ASC','rv_templates/T25000G35P00V000K2SODNVR20N.ASC','rv_templates/T40000G45P00V000K2SODNVR20N.ASC']):
			template_raw=np.loadtxt(template_file, dtype=float)
			template_ar=np.array(zip(w[ranges[ccd][0]:ranges[ccd][1]],template_raw[ranges[ccd][0]:ranges[ccd][1]]), dtype=[('l', 'f8'), ('f', 'f8')])
			template_teff=float(template_file[14:19])
			template_logg=float(template_file[20:22])/10.0
			templates_tmp.append([t, template_file, template_teff, template_logg, template_ar])

		templates.append(templates_tmp)

	files=glob.glob('reductions/results/%s/spectra/com/*.fits' % date)
	sobjects=[file.split('/')[-1][:15] for file in files]
	sobjects=set(sobjects)

	logging.info('Calculating radial velocities.')

	args=[]
	for sobject in sobjects:
		args.append([sobject, templates])

	pool = Pool(processes=12)
	pool.map(analyze_rv, args)
	pool.close()

	# calculate stellar parameters


if __name__ == "__main__":
	"""
	Below you can give a list of all procedures to be run in the pipeline. The given order is not always necessary but is recomended. All procedures are as independent as possible.
	From the command line you can run the pipeline with $ python extract6.0.py /path/to/files/. The pipeline was tested with python 2.7. 

	The reduction pipeline is very read/write intensive. Run it from a folder on an SSD disk or create a RAM filesystem and run it from there. The latter option is also the fastest, but you need a lot of RAM for it. Running the pipeline on HDDs is not advised. 
	When testing, always create a copy of your data in case it gets damaged. First function in the pipeline (prepare_dir) will copy all the data into a local folder (./reductions), but it is best to be careful anyways.
	
	Possible errors:
	- Not enough space in image header: Find login.cl file and raise the min_lenuserarea parameter (also uncomment it, if commented before).
	- Your "iraf" and "IRAFARCH" environment variables are not defined: define them as instructed. Do it in bash.rc permanently.
	- Image xxxxxx already exists: You are trying to override previous reductions. Delete ./reductions folder (or part thereof) and try again.
	- Memory error: You don't have enough memory to run the code with current settings. Change ncpu parameters to a lower value and try again. 
	"""

	global start_folder
	start_folder = os.getcwd()

	# Set iraf variables that are otherwise defined in local login.cl. Will probably overwrite those settings
	iraf.set(min_lenuserarea=128000)
	iraf.set(uparm=start_folder + '/uparm')

	logging.basicConfig(level=logging.DEBUG)

	if len(sys.argv) == 2:
		# date='170711'
		date, cobs = prepare_dir(sys.argv[1])
		#remove_bias(date)
		#fix_gain(date)
		#fix_bad_pixels(date)
		#prepare_flat_arc(date, cobs)
		#remove_cosmics(date, ncpu=8)
		#find_apertures(date)
		#plot_apertures(190210, 1902100045, 3)
		#remove_scattered(date, ncpu=6)
		#measure_cross_on_flat(date)
		#extract_spectra(date)
		#wav_calibration(date)
		#os.system('cp -r reductions-test reductions')
		#remove_sky(date, method='nearest3', thr_method='flat', ncpu=8)
		#plot_spectra(190210, 1902100045, 3)
		#remove_telurics(date)
		#v_bary_correction(date, quick=False, ncpu=8)
		#os.system('cp -r reductions_bary reductions')
		#resolution_profile(date)
		#os.system('cp -r reductions-test reductions')
		#create_final_spectra(date)
		#analyze(date)
		#create_database(date)
		
	else:
		logging.critical('Wrong number of command line arguments.')