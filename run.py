"""
This script runs the reduction pipeline. It is a bit weird in a way that it is spawning processes by system calls, but this is the most reliable way of making the parellization to work.
If some steps fail, the script will recover from a backup and try once again.

TODO:
	Figure out what is not working with multiprocessing in run_one_night. It is working with multiprocess module for now.
	Add optional parameters for sky and bary
"""
from os import system
from time import sleep
import collections
import argparse
from multiprocess import Pool

def run_one_night(args):
	st,date=args

	data_folder=st.data_folder
	if data_folder[-1]!='/': data_fodler=data_folder+'/'
	# Check whether this is a reduction of one night or multiple nights and set ncpu accordingly (one nmight is reduced with ncpu threads, multiple nights are not)
	ncpu=st.ncpu
	if isinstance(st.nights, (int, long)):
		# if nights is intiger, keep ncpu as is
		pass
	elif isinstance(st.nights, collections.Iterable) and len(st.nights)==1:
		# if nights is list or tuple of len 1, keep ncpu as is
		pass
	elif isinstance(st.nights, collections.Iterable) and len(st.nights)>1:
		# if nights is a list or tuple and there are multiple nights to reduce, set ncpu to 1
		ncpu=1
	else:
		logging.error('Nights parameter must be int or iterable.')
		return -1

	try:
		# Start first batch of reduction steps
		if st.initial: system('python extract6.0.py %s%s --initial --n_cpu %s --runs \'%s\'' % (data_folder, date, ncpu, st.runs))
		if st.cosmics: system('python extract6.0.py %s%s --cosmic --n_cpu %s' % (data_folder, date, ncpu))
		if st.trace: system('python extract6.0.py %s%s --trace --n_cpu %s' % (data_folder, date, ncpu))
		if st.scattered: system('python extract6.0.py %s%s --scattered --n_cpu %s' % (data_folder, date, ncpu))
		if st.cross: system('python extract6.0.py %s%s --xtalk --n_cpu %s' % (data_folder, date, ncpu))
		if st.extract: system('python extract6.0.py %s%s --extract --n_cpu %s' % (data_folder, date, ncpu))
		if st.wav_cal: system('python extract6.0.py %s%s --wav --n_cpu %s' % (data_folder, date, ncpu))
		if st.sky: system('python extract6.0.py %s%s --sky --n_cpu %s' % (data_folder, date, ncpu))
		if st.teluric: system('python extract6.0.py %s%s --telurics --n_cpu %s' % (data_folder, date, ncpu))
		if st.v_bary: system('python extract6.0.py %s%s --bary --n_cpu %s' % (data_folder, date, ncpu))
		if st.resolution: system('python extract6.0.py %s%s --resolution --n_cpu %s' % (data_folder, date, ncpu))
	except:
		# If they fail, delete folder and try again
		system('rm -r reductions/%s' % date)
		if st.initial: system('python extract6.0.py %s%s --initial --n_cpu %s --runs \'%s\'' % (data_folder, date, ncpu, st.runs))
		if st.cosmics: system('python extract6.0.py %s%s --cosmic --n_cpu %s' % (data_folder, date, ncpu))
		if st.trace: system('python extract6.0.py %s%s --trace --n_cpu %s' % (data_folder, date, ncpu))
		if st.scattered: system('python extract6.0.py %s%s --scattered --n_cpu %s' % (data_folder, date, ncpu))
		if st.cross: system('python extract6.0.py %s%s --xtalk --n_cpu %s' % (data_folder, date, ncpu))
		if st.extract: system('python extract6.0.py %s%s --extract --n_cpu %s' % (data_folder, date, ncpu))
		if st.wav_cal: system('python extract6.0.py %s%s --wav --n_cpu %s' % (data_folder, date, ncpu))
		if st.sky: system('python extract6.0.py %s%s --sky --n_cpu %s' % (data_folder, date, ncpu))
		if st.teluric: system('python extract6.0.py %s%s --telurics --n_cpu %s' % (data_folder, date, ncpu))
		if st.v_bary: system('python extract6.0.py %s%s --bary --n_cpu %s' % (data_folder, date, ncpu))
		if st.resolution: system('python extract6.0.py %s%s --resolution --n_cpu %s' % (data_folder, date, ncpu))

	# Create backup of the first batch of steps
	system('cp -r reductions/%s reductions/.backup_%s' % (date, date))
	sleep(3)

	try:
		# Start second batch of reduction steps
		if st.final: system('python extract6.0.py %s%s --final --n_cpu %s' % (data_folder, date, ncpu))
		if st.analyse: system('python extract6.0.py %s%s --analyze --n_cpu %s' % (data_folder, date, ncpu))
		if st.database: system('python extract6.0.py %s%s --database --n_cpu %s' % (data_folder, date, ncpu))
	except:
		# If they fail, recover from backup and try again
		system('rm -r reductions/%s' % date)
		sleep(3)
		system('cp -r reductions/.backup_%s reductions/%s' % date, date)
		sleep(3)
		if st.final: system('python extract6.0.py %s%s --final --n_cpu %s' % (data_folder, date, ncpu))
		if st.analyse: system('python extract6.0.py %s%s --analyze --n_cpu %s' % (data_folder, date, ncpu))
		if st.database: system('python extract6.0.py %s%s --database --n_cpu %s' % (data_folder, date, ncpu))

	# Clear the reduction folder. Only keep reduced data.
	system('rm -r reductions/%s' % date)
	system('rm -r reductions/.backup_%s' % date)
	sleep(3)

if __name__=="__main__":

	parser=argparse.ArgumentParser()
	parser.add_argument("settings", help="Path to the settings file (settings.py).")
	args=parser.parse_args()

	settings_file=args.settings
	if settings_file[-3:]=='.py': settings_file=settings_file[:-3]

	settings= __import__(settings_file)

	if isinstance(settings.nights, collections.Iterable) and len(settings.nights)>1:
		# if there are multiple nights to reduce 
		dates=settings.nights
		args=[]
		for date in dates:
			args.append([settings, date])
		pool = Pool(processes=settings.ncpu)
		pool.map(run_one_night, args)
		pool.close()
	elif isinstance(settings.nights, (int, long)):
		# if there is only one night to reduce
		run_one_night([settings, settings.nights])
	elif isinstance(settings.nights, collections.Iterable) and len(settings.nights)==1:
		# if there is only one night to reduce
		run_one_night([settings, settings.nights[0]])
	else:
		logging.error('Nights parameter must be int or iterable.')


	