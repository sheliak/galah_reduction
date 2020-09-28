"""
This script takes all the spectra in a database aresamples them and serializes them. 
TO DO:
	Write galah_tools and use that for opening spectra
	Add option for the velocity frame
"""

from astropy.io import fits
import numpy as np

#load database
hdul=fits.open('reductions/dr6.0.fits')
hdu=hdul[1]
t=Table(hdu.data)
hdul.close()

#create list of filenames
sobjects=np.array(t['sobject_id'], dtype=str)
for ccd in [1,2,3,4]:
	pass
