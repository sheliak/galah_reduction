# GALAH reduction pipeline

# Installation

**Dependencies**

 - Python 2.7
 - iraf
 - scipy
 - numpy
 - pyfits
 - astropy
 - matplotlib
 - pylab
 - lmfit
 - ephem

**Iraf installation**

Under Ubuntu >= 18.04 or Debian >= 10 iraf can be installed directly from the command line using the following command:
```
$ apt-get install iraf
```

**Iraf installation with conda**

To install iraf under conda, run the following commands:
```
$ conda config --add channels http://ssb.stsci.edu/astroconda
$ conda install iraf-all pyraf-all
```

# Before running - optional

Move to the folder from which you will run the reduction procedure. To create an ifraf login.cl file with settings run the following command:
```
$ mkiraf
```
This step is optional as iraf will instead use settings from the default login.cl file.

# How to run the pipeline

**1. RUNNING COMPLETE REDUCTION PIPELINE**

To reduce one night run
```
python extract6.0.py /path/to/spectra/night
```
where `/path/to/spectra/night` is a path to standard GALAH folders for one night. Inside this folder should be folders: comments, data, fld, sds, etc. When testing, make a copy of data in case they get corrupted. The reduction pipeline should make a local copy and not write anything into `/path/to/spectra/night`.

**2. RUNNING MINIMAL REDUCTION**

The pipeline will make a minimal reduction to get 1D spectra. This preview is useful for quick checks during observations. The steps required for this reduction are already defined in the function `min_red`.

**3. PARTIAL REDUCTION**

Reduction steps that will be performed are listed at the end of `extract6.0.py`. If only some steps are performed, the reduction can be run again with only the remaining steps. This is most useful for testing. That way, you don't have to run the whole pipeline every time.

Warning: **DO NOT** run the same reduction step multiple times on the same data as they are not overwritten. The pipeline also does not check if the selected step was already performed. Therefore data could get manipulated multiple times, providing wrong final results. This warning also holds true for the step called `prepare_dir`.

When calling a part of the pipeline, you will have to manually define the `date` and `cobs` variables. They are currently determined as part of a longer procedure that prepares and copies original observed data.

# Where are my reduced data

During the run, the pipeline will, by default create several folders with intermediate results in your working directory. That is the folder you were in when you started the python pipeline. As the process is very read/write intensive select as fast memory as possible.

Spectral and physical parameters of spectra in an individual night will be gathered in a single database that can be found in the directory `?`. To combine results from multiple nights run the procedure `merge_db.py` that is located in the utils folder.

# Known errors and issues
 - **Not enough space in image header**: Find login.cl file and raise the `min_lenuserarea` parameter (also uncomment it, if commented before).
 - **Your "iraf" and "IRAFARCH" environment variables are not defined**: define them as instructed. Do it in bash.rc permanently.
 - **Image xxxxxx already exists**: You are trying to override previous reductions. Delete `../reductions` folder (or part thereof) and try again.
 - **Memory error: You don't have enough memory to run the code with current settings**. Change `ncpu` parameters to a lower value and try again. 

