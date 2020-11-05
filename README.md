# GALAH reduction pipeline

# Installation

**Dependencies**

 - Python 2.7
 - iraf
 - pyraf
 - scipy
 - numpy
 - astropy
 - matplotlib 
 - pylab
 - lmfit
 - numdifftools
 - ephem
 - json
 - joblib
 - h5py
 - galah_tools (included in this repository)
 - tesndorflow 2.1.0 (the latest version working on python2.7)
 - keras 2.3.1 (this is the latest version that works with tensorflow 2.1.0)

**Iraf installation**

Under Ubuntu >= 18.04 or Debian >= 10 iraf and pyraf can be installed directly from the command line using the following command:
```
$ apt-get install iraf
$ pip install pyraf
```

**Iraf installation with conda**

To install iraf under conda, run the following commands:
```
$ conda config --add channels http://ssb.stsci.edu/astroconda
$ conda install iraf-all pyraf-all
```

# Before running - optional

Iraf needs to know which configuration file to use. If you want to set some custom parameters, create a login.cl file in the folder from which you will run the reduction procedure. To create an ifraf login.cl file with default setting run the following command:
```
$ mkiraf
```
This step is optional as iraf will instead use settings from the default login.cl file. Which login.cl file is used is displayed when pyraf is run. 

# How to run the pipeline

**1. RUNNING COMPLETE REDUCTION PIPELINE**

All parameters, including which nights to reduce, are set in the settings.py file. 

To run the reduction do
```
python run.py /path/to/settings.py
```
where `/path/to/settings.py` is a path to the settings file you are using.

**2. RUNNING MINIMAL REDUCTION**

The pipeline will make a minimal reduction to get 1D spectra. This preview is useful for quick checks during observations. Comming soon.

**3. PARTIAL REDUCTION**

Reduction steps that will be performed are listed at the end of `extract6.0.py`. If only some steps are performed, the reduction can be run again with only the remaining steps. This is most useful for testing. That way, you don't have to run the whole pipeline every time.

Warning: **DO NOT** run the same reduction step multiple times on the same data as they are not overwritten. The pipeline also does not check if the selected step was already performed. Therefore data could get manipulated multiple times, providing wrong final results. This warning also holds true for the step called `prepare_dir`.

When calling a part of the pipeline, you will have to manually define the `date` and `cobs` variables. They are currently determined as part of a longer procedure that prepares and copies original observed data.

# Parameter estimation

During the run, the pipeline will estimate rough stellar parameters and abundances, but won't provide any warning flags or uncertainties. Use the data with caution, especially the abundances.

The estimation procedure is a convolutional neural network that was trained on spectra from previous extraction run and their unflagged Galah DR3 parameters. After the spectra are completely re-reduce, the network will be re-trained. 

# Where are my reduced data

During the run, the pipeline will, by default create several folders with intermediate results in your working directory. That is the folder you were in when you started the python pipeline. As the process is very read/write intensive select as fast memory as possible. After the pipeline has finished working, spectra combined from multiple exposures will be located in the directory `../reductions/results/yymmdd/spectra/com/`.

Spectral information and physical parameters of spectra in an individual night will be gathered in a single database that can be found in the directory `../reductions/results/yymmdd/db/`. It is available as a fits and csv file. To combine results from multiple nights run the procedure `merge_db.py` that is located in the utils folder.

# After the pipeline has finished

**1. MERGE DATABASES**

To create a final combined database of all spectra, a procedure `merge_db.py` has to be run for every individual night. It will create a `dr6.0.fits` and `dr6.0.csv` files in the directory `../reductions/`

**2. HDF5 SPECTRAL DATABASE (optional)**

If you prefer to have all spectra combined into a single hdf5 file, the procedure `create_hdf5_database.py` that is located in the utils folder. It will read the final merged database and save all mentioned combined spectra into the `dr6.0_com.hdf5` file. This procedure takes a long time if the number of spectra is large.

# Known errors and issues
 - **Not enough space in image header**: Find login.cl file and raise the `min_lenuserarea` parameter (also uncomment it, if commented before).
 - **Your "iraf" and "IRAFARCH" environment variables are not defined**: define them as instructed. Do it in bash.rc permanently.
 - **Image xxxxxx already exists**: You are trying to override previous reductions. Delete `../reductions` folder (or part thereof) and try again.
 - **Memory error: You don't have enough memory to run the code with current settings**. Change `ncpu` parameters to a lower value and try again.
