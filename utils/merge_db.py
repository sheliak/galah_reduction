"""
This script merges a database for one night with a global database. If global database does not exist yet, it is created.

TO DO:
	Add command line arguments
	Remove .dblock when procedure fails
"""
import os
import shutil
import logging
import sys
from astropy.io import fits
from astropy.table import vstack, Table
import time
import numpy as np
import argparse
import glob

logging.basicConfig(level=logging.DEBUG)

# read command line arguments
parser=argparse.ArgumentParser()
parser.add_argument("--date", help="Date you want to merge into the global DB (in format yymmdd).", default=None)
parser.add_argument("--all", help="Merge all nightly databases", action="store_true")
parser.add_argument("--path", help="Path to the folder where reducd nights are saved.", default='..')
args=parser.parse_args()

# check path format
if args.path[-1]=='/': path=args.path
else: path=args.path+'/'

dates=[]
# create a list of dates that need their databases merged
if args.all:
	# create a list of all databases
	databases=glob.glob(path+'*/db/*.fits')

	# turn this into a list of nights
	for database in databases:
		date=database[-11:-5]
		dates.append(date)
else:
	date = args.date
	if date==None:
		logging.error('Specify --date of use --all to combine all nightly databases. Use -h for help.')
		sys.exit(-1)

	if len(date) == 6 and 10 < int(date[:2]) < 30 and 1 <= int(date[2:4]) <= 12 and 1 <= int(date[4:6]) <= 31:
		pass
	else:
		logging.error('Date must be an intiger in format yymmdd.')
		sys.exit(-1)

	dates.append(date)

os.chdir(path)

dates=np.sort(dates)

logging.info('Databases will be combined for dates %s.' % ",".join(dates))

for date in dates:
	logging.info('Adding database for date %s.' % date)

	if os.path.isfile('dr6.0.fits'):
		# check if global database exists and open it if it does
		# but first create a lockfile, so only one process can edit the database at a time
		t_sleep=0
		while os.path.isfile('.dblock'):
			if t_sleep>=60: 
				sys.exit(0)
			if t_sleep==0: logging.warning('Database dr6.0.fits is being used by another program. Will wait 1 minute for accsess and then terminate.')
			time.sleep(1)
			t_sleep+=1

		open('.dblock', 'a').close()
		hdul=fits.open('dr6.0.fits', mode='update')
		hdu=hdul[1]
		table_global=Table(hdu.data)

		#create a list of sobject_ids already in the database:
		sobjects=np.array(table_global['sobject_id'])

		# load the nightly db
		hdul_night=fits.open('%s/db/%s.fits' % (date,date))
		hdu_night=hdul_night[1]
		table_night=Table(hdu_night.data)
		hdul_night.close()

		# only use sobject_ids not already in the database
		mask=[i in sobjects for i in np.array(table_night['sobject_id'])]
		mask=np.array(mask, dtype=bool)
		table_night=table_night[~mask]

		#merge tables
		table_stacked=vstack([table_global, table_night])
		hdul[1]=fits.BinTableHDU(table_stacked)
		hdul.flush()
		hdul.close()

		if date==dates[-1]:# if this is the last database to process create csv and hdf databases too
			# convert table to ascii. add _1, _2, _3, _4 to fields in arrays (always representing four arms)
			csv_db=open('dr6.0.csv', 'a')
			hdul=fits.open('dr6.0.fits')
			hdu=hdul[1]

			data=np.array(hdu.data)
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
			hdul.close()

			# convert table to hdf5. astropy write function has append option, but it doesn't work for hdf5, aparently
			hdul=fits.open('dr6.0.fits', mode='update')
			hdu=hdul[1]
			t=Table(hdu.data)
			hdul.close()
			t.write('dr6.0.hdf5', format='hdf5', path='data', overwrite=True)

		#save results
		os.remove('.dblock')

	else:
		# if global database does not exist, copy the nightly database and hence make it global. After this we are done.
		t_sleep=0
		while os.path.isfile('.dblock'):
			if t_sleep>=60: 
				sys.exit(0)
			logging.warning('Database dr6.0.fits is being used by another process. Will wait 1 minute for accsess and then terminate.')
			time.sleep(1)
			t_sleep+=1
		open('.dblock', 'a').close()
		lock_created=True
		shutil.copyfile('%s/db/%s.fits' % (date,date), 'dr6.0.fits')

		if date==dates[-1]:# if this is the last database to process create csv and hdf databases too
			# convert table to ascii. add _1, _2, _3, _4 to fields in arrays (always representing four arms)
			csv_db=open('dr6.0.csv', 'w')
			hdul=fits.open('%s/db/%s.fits' % (date,date))
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
			hdul.close()

			# convert table to hdf5
			t.write('dr6.0.hdf5', format='hdf5', path='data')

		os.remove('.dblock')
