# For running the reduction, you only need to set parameters in this file. Then run the reduction with
# >python run.py path/to_settings/file


# PARAMETERS

# Number of processes to use
# If reducing one night, this many threads will be used for the reduction. If reducing multiple night this many nights will be reduced in parallel.
ncpu = 3

# Data folder
data_folder = "/home/kevin/Documents/Reduction_pipe/"

# Night to be reduced
# If this is an intiger (e.g. 190210) only this night will be reduced using ncpu threads.
# If array (e.g. [190210, 180211, 200112]) nights will be reduced in parallel. One thread will be used for each night.
nights = [190223]

# Runs to be reduced 
# Runs with obtatus=0 in the comments file will not be reduced, even if they are included in this list.
# Use runs = '*' to reduce all runs
runs = "*"


#Logging level. 
#The logging levels has been split into console and file levels 
console_level='DEBUG'
file_level='DEBUG'


# REDUCTION STEPS

# Only list COBs. Nothing will be reduced.
list_cobs = True

# Folder preparation and initial reduction
initial = True

# Cosmic rays removal (optional)
cosmics = False

# Trace apertures
trace = False

# Measure and remove scattered light (optional)
scattered = False

# Measure and remove cross-talk (optional)
cross = False

# Extract 1D spectra
extract = False

# Wavelength calibration
wav_cal = False

# Remove sky spectrum (optional)
# options method (nearest|nearest3) and throughput_method (flat|mags|both) are available
sky = False
method = "nearest3"
throughput_method = "flat"

# Remove teluric absorptions (optional)
teluric = False

# Correct for barycentric velocity (optional)
# option quick (False|False) is available
v_bary = False
quick = False

# Calculate the resolution profile (optional)
resolution = False

# Create final spectra
final = False

# Analyse final spectra (optional)
analyse = False

# Create database for reduced nights
database = False


# DISPLAYED PLOTS

# Plot apertures

plot_apertures = False

# Plot spectra

plot_spectra = False

