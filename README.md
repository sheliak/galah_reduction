# GALAH reduction pipeline

This is a package of programs and tools used to reduce spectra obtained by the HERMES instrument at the AAO. They are being developed to reduce data obtained by the GALAH survey and related programs. It is not guranateed to produce trustworthy results or work at all for data outside the GALAH ecosystem. 

# Installation

GALAH reduction pipeline does not need to be installed. It depends on the following software that is needed to run the reduction pipeline.

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
 - tensorflow 2.1.0 (the latest version working on python2.7)
 - keras 2.3.1 (this is the latest version that works with tensorflow 2.1.0)
 - dustmaps (follow installation instructions at https://dustmaps.readthedocs.io/en/latest/installation.html and download Planck maps)
 - astroquery (version 0.3.9 works with python 2.7)

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

# How to run the pipeline

**1. RUNNING COMPLETE REDUCTION PIPELINE USING run.py SCRIPT**

All parameters, including which nights to reduce, are set in the `settings.py` file. 

To run the reduction do
```
python run.py /path/to/settings.py
```
where `/path/to/settings.py` is a path to the settings file you are using.

File `settings.py` defines which reductions steps will be performed. Those marked with `(optional)` can be skipped and the pipeline should perform the rest with no issues. 

**2. RUNNING COMPLETE REDUCTION PIPELINE WITH A CUSTOM SET OF INSTRUCTIONS**

Coming soon

**3. CHECKING DATA PARTITIONING INTO COBs**

First step of the reduction process is to partition the nightly data into consecutive observing blocks (COBs). Each COB should include consecutive observations of one field and associated flat field and arc frame. To list the inferred COBs, run the reduction with `--list_cobs` flag or `list_cobs = True` set in `settings.py`. In this case the reduction pipeline will only list the COBs, but won't proceede with reduction.

**4. RUNNING MINIMAL REDUCTION**

Coming soon.

**5. PARTIAL REDUCTION**

If only part of the reduction is performed, the intermediate results can be saved and the reduction pipeline can resume from where it previously finished. This can not be done if any reduction step returned errors or the reduction was interrupted while performing a step.

**6. REDUCING DATA NOT IN GALAH FORMAT**

GALAH stores data in a specific order (uses specific filenames, folder structure, comment files etc.). Data with a different structure can be reduced, if the initial reduction steps are bypassed. In any case, the reduction will only work with data taken with the HERMES instrument.

1. You will have to remove bias, compensate for gain, fix bad pixels and rename files.
1. Create the following structure in the `reductions` folder. Data from different nights must be collected in separate folders (named 200811 for data collected August 11 2020, for example). Fits files for each observed field must be in respective folders (named 2008110041 for filed starting with run number 0041, for example). Each of these files must contain exactly one flat field image and one arc image named masterflat.fits and masterarc.fits, respectively. There can be any number of science images (named 11avg10046.fits for run number 0046 taken with ccd 1, for example).
```
reductions
|-- yymmdd (6 digit date number)
|   |-- ccd1
|   |   |-- yymmddrrrr (10 digit COB number)
|   |   |   |-- masterflat.fits
|   |   |   |-- masterarc.fits
|   |   |   |-- ddmmm1rrrr.fits
|   |   |   |-- ...
|   |   |-- ...
|   |
|   |-- ccd2
|   |   |-- yymmddrrrr (10 digit COB number)
|   |   |   |-- masterflat.fits
|   |   |   |-- masterarc.fits
|   |   |   |-- ddmmm2rrrr.fits
|   |   |   |-- ...
|   |   |-- ...
|   |   
|   |-- ccd3
|   |   |-- yymmddrrrr (10 digit COB number)
|   |   |   |-- masterflat.fits
|   |   |   |-- masterarc.fits
|   |   |   |-- ddmmm3rrrr.fits
|   |   |   |-- ...
|   |   |-- ...
|   |   
|   |-- ccd4
|   |   |-- yymmddrrrr (10 digit COB number)
|   |   |   |-- masterflat.fits
|   |   |   |-- masterarc.fits
|   |   |   |-- ddmmm4rrrr.fits
|   |   |   |-- ...
|   |   |-- ...
```
3. Run the reduction pipeline as usual, skipping initial part of the reduction. You can set 
```
list_cobs = False
initial = False
```
in the `settings.py` file.

# Parameter estimation

During the run, the pipeline will estimate rough stellar parameters and abundances, but won't provide any warning flags or uncertainties. Use the data with caution, especially the abundances.

The estimation procedure is a convolutional neural network that was trained on spectra from previous extraction run and their unflagged Galah DR3 parameters. After the spectra are completely re-reduce, the network will be re-trained. 

# Where are my reduced data

**1. REDUCED SPECTRA**

During the reduction, two folders will be created in the `reductions` folder. `reductions/yymmdd/` will hold intermediate results, as well as all the files produced during the reduction. If the reduction is run all the way through (using `final = True` or `--final` option), the results will be saved in the `reductions/results/yymmdd/` folder. The latter is the intended way of producing final reduced spectra. Folder structure is as follows:
```
results
|-- yymmdd (6 digit date number)
|   |-- spectra
|   |   |-- all
|   |   |   |-- yymmddrrrr00ppp1.fits
|   |   |   |-- yymmddrrrr00ppp2.fits
|   |   |   |-- yymmddrrrr00ppp3.fits
|   |   |   |-- yymmddrrrr00ppp4.fits
|   |   |   |-- ...
|   |   |-- com
|   |   |   |-- yymmddrrrr01ppp1.fits
|   |   |   |-- yymmddrrrr01ppp2.fits
|   |   |   |-- yymmddrrrr01ppp3.fits
|   |   |   |-- yymmddrrrr01ppp4.fits
|   |   |   |-- ...
|   |-- db
|   |   |-- yymmdd.fits
|   |   |-- yymmdd.csv
|   |   |-- yymmdd.hdf5
|   |-- diagnostics
|   |   |-- various .png and .pdf files
```

**2. DATABASE**

To produce the final table (with all the parameters produced by the reduction pipeline) run the reduction with a setting `database = True` or `--database` flag. Table for each night is saved in `/reductions/results/yymmdd/db/`. It is available as a .fits, .csv, or .hdf5 file.

**3. INTERMEDIATE SPECTRA**

From the reduced spectra, one can produce intermediate spectra, as they would appear at varius steps of the reduction process. The starting point is the spuctrum in extension 0. Note that the normalised spectrum cannot be used in this process (with the exception of undoing the teluric correction). The following steps can be done in this exact order to undo reduction steps done in the official GALAH reduction:

- Remove barycentric velocity correction (add velocity given in BARYEFF keyword in the header to all wavelength solutions).
- Divide spectrum in extension 0 by the teluric correction (extension 4). 
- Add sky spectrum (extension 3)
- Add cross talk (extension 6)
- Add scattered light (extension 5)

# After the pipeline has finished

**1. MERGE DATABASES**

After one night is reduced, the reduction database is created for that night only. If you are happy with the results, you can merge it into a global database that includes results for multiple nights. Script `merge_db.py` in `utils` does that for you. It will merge the database for one night into `dr6.0.fits` and `dr6.0.csv` files (or create them, if they don't exist yet) in the `reductions/` directory.

**2. HDF5 SPECTRAL DATABASE (optional)**

If you prefer to have all spectra combined into a single hdf5 file, the procedure `create_hdf5_database.py` that is located in the `utils` folder. It will read the final merged database and save all combined spectra into a `dr6.0_com.hdf5` file. This procedure takes a long time if the number of spectra is large.

# Known errors and issues
 - **Not enough space in image header**: Find login.cl file and raise the `min_lenuserarea` parameter (also uncomment it, if commented before).
 - **Your "iraf" and "IRAFARCH" environment variables are not defined**: define them as instructed. Do it in bash.rc permanently.
 - **Image xxxxxx already exists**: You are trying to override previous reductions. Delete `../reductions` folder (or part thereof) and try again.
 - **Memory error: You don't have enough memory to run the code with current settings**. Change `ncpu` parameters to a lower value and try again.
 - **ERROR (827, "Cannot open image (masterflat.ms.fits)")**. You failed to do some mandatory reduction steps. Check your settings file.

