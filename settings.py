# For running the reduction, you only need to set parameters in this file. Then run the reduction with
# >python run.py path/to_settings/file


# PARAMETERS

# Number of processes to use
# If reducing one night, this many threads will be used for the reduction. If reducing multiple night this many nights will be reduced in parallel.
ncpu = 16

# Data folder
data_folder = "/media/storage/HERMES_DATA/site-ftp.aao.gov.au/survey/"

# Night to be reduced
# If this is an intiger (e.g. 190210) only this night will be reduced using ncpu threads.
# If array (e.g. [190210, 180211, 200112]) nights will be reduced in parallel. One thread will be used for each night.
nights = [140610, 141101, 150410, 150424, 160530, 160919, 161213, 170804, 171229, 181220]

# Runs to be reduced 
# Runs with obtatus=0 in the comments file will not be reduced, even if they are included in this list.
# Use runs = '*' to reduce all runs
# runs = "0-23"
runs = "*"


# REDUCTION STEPS

# Only list COBs. Nothing will be reduced.
list_cobs = True

# Folder preparation and initial reduction
initial = True

# Cosmic rays removal (optional)
cosmics = True

# Trace apertures
trace = True

# Measure and remove scattered light (optional)
scattered = True

# Measure and remove cross-talk (optional)
cross = True

# Extract 1D spectra
extract = True

# Wavelength calibration
wav_cal = True

# Remove sky spectrum (optional)
# options method (nearest|nearest3) and throughput_method (flat|mags|both) are available
sky = True
method = "nearest3"
throughput_method = "flat"

# Remove teluric absorptions (optional)
teluric = True

# Correct for barycentric velocity (optional)
# option quick (True|False) is available
v_bary = True
quick = False

# Calculate the resolution profile (optional)
resolution = True

# Create final spectra
final = True

# Analyse final spectra (optional)
analyse = True

# Create database for reduced nights
database = True


# DISPLAYED PLOTS

# Plot apertures
plot_apertures = False

# Plot spectra
plot_spectra = False

# Plot diagnostic files
plot_diagnostics = False
