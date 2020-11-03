"""
This script merges a database for one night with a global database. If global database does not exist yet, it is created.

TO DO:
	Add command line arguments
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

logging.basicConfig(level=logging.DEBUG)

parser=argparse.ArgumentParser()
parser.add_argument("date", help="Date you want to merge into the global DB (in format yymmdd).")
args=parser.parse_args()

date = args.date

if len(date) == 6 and 10 < int(date[:2]) < 30 and 1 <= int(date[2:4]) <= 12 and 1 <= int(date[4:6]) <= 31:
	pass
else:
	logging.error('Date must be an intiger in format yymmdd.')
	sys.exit(-1)

os.chdir('..')

if os.path.isfile('reductions/dr6.0.fits'):
	# check if global database exists and open it if it does
	# but first create a lockfile, so only one process can edit the database at a time
	t_sleep=0
	while os.path.isfile('reductions/.dblock'):
		if t_sleep>=60: 
			sys.exit(0)
		if t_sleep==0: logging.warning('Database reductions/dr6.0.fits is being used by another program. Will wait 1 minute for accsess and then terminate.')
		time.sleep(1)
		t_sleep+=1

	open('reductions/.dblock', 'a').close()
	hdul=fits.open('reductions/dr6.0.fits', mode='update')
	hdu=hdul[1]
	table_global=Table(hdu.data)

	#create a list of sobject_ids already in the database:
	sobjects=np.array(table_global['sobject_id'])

	# load the nightly db
	hdul_night=fits.open('reductions/results/%s/db/%s.fits' % (date,date))
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

	# convert table to ascii. add _1, _2, _3, _4 to fields in arrays (always representing four arms)
	csv_db=open('reductions/dr6.0.csv', 'a')
	hdul=fits.open('reductions/results/%s/db/%s.fits' % (date,date))
	hdu=hdul[1]
	data=np.array(hdu.data)
	#add data
	for i in data:
		if i[0] not in sobjects:
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

	#save results
	os.remove('reductions/.dblock')

else:
	# if global database does not exist, copy the nightly database and hence make it global. After this we are done.
	t_sleep=0
	while os.path.isfile('reductions/.dblock'):
		if t_sleep>=60: 
			sys.exit(0)
		logging.warning('Database reductions/dr6.0.fits is being used by another process. Will wait 1 minute for accsess and then terminate.')
		time.sleep(1)
		t_sleep+=1
	open('reductions/.dblock', 'a').close()
	lock_created=True
	shutil.copyfile('reductions/results/%s/db/%s.fits' % (date,date), 'reductions/dr6.0.fits')

	# convert table to ascii. add _1, _2, _3, _4 to fields in arrays (always representing four arms)
	csv_db=open('reductions/dr6.0.csv', 'w')
	hdul=fits.open('reductions/results/%s/db/%s.fits' % (date,date))
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

	os.remove('reductions/.dblock')
