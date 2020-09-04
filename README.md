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

**Iraf installation**

Under Ubuntu >= 18.04 or Debian >10.0 iraf can be installed directlly from the command line using the following command:
```
$ apt-get install iraf
```

**Iraf installation with conda**

To istall iraf under conda, run the following commands:
```
$ conda config --add channels http://ssb.stsci.edu/astroconda
$ conda install iraf-all pyraf-all
```

# Before running - optional

Move to the folder from which you will run the reduction procedure. To create an ifraf login.cl file with settings run the following command:
```
$ mkiraf
```
This step is optional as iraf will istead use settings from the default login.cl file.

# How to run the pipeline

**1. RUNNING COMPLETE REDUCTION PIPELINE**

To reduce one night run
```
python extract6.0.py /path/to/spectra/night
```
where `/path/to/spectra/night` is a path to standard GALAH folders for one night. Inside this folder should be folders comments, data, fld, sds, etc. When testing, make a copy of data in case it gets corrupted. The reduction pipeline should make a local copy and not write anything into /path/to/spectra/night.

**2. PARTIAL REDUCTION**

Reduction steps that will be performed are listed at the end of extract6.0.py. If only some steps are performed, the reduction can be run again with only the remaining steps. This is most useful for testing. so you don't have to run the whole pipeline every time.

# Where are my reduced data

During the run, the pipeline will by default create several folders with intermediate results in your working directory. That is the folder you were in when you started the python pipeline. As the proces is very read/write intensieve select as fast memory as possible.

# Possible errors

