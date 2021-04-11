# GALAH utils



## Minimal reduction for a night



## Merge databases from multiple nights



## Combine spectra of the same star from multple nights



## Create HDF5 spectral database in a single file



## Retrain convolutional neural network for determination of stellar parameters and abundances

Neural network for the determination of parameters and abundances in reduction pipeline can be trained using the following steps and scripts:

### Collect and resample spectra

First, we need to collect and resample all spectra to the same wavelength bins. For this, we use the script `cnn_prepare_spectra.py` that takes two arguments. The first is the root path to the latest Iraf6.0 reductions, whose reading is handled by the `galah_tools` in the pipeline's `tools` subfolder. The second argument determines the CCD bands (1-4) for which the procedure will be run. The argument can specify only one band or multiple bands by giving multiple comma-separated values with no whitspaces. The procedure will output one pkl file per band.

Example of use:
```
python cnn_prepare_spectra.py /media/hdd/home2/janez/storage/OWNCLOUD/GALAH/obs/reductions/Iraf_6.0 1
```

### Run training procedure

After all bands were resampled and prepared, the script `cnn_train_network.py` handles the training procedure. It takes two arguments: the path to the reduction database fits file and the path to the fits file containing parameters and abundances on which the network will be trained. 

Example of use:
```
python cnn_train_network.py /media/hdd/home2/janez/storage/OWNCLOUD/GALAH/obs/reductions/Iraf_6.0/dr6.0.fits /shared/mari/cotar/GALAH_DR3_main_allspec_v2.fits
```

### Replace existing model files

After the training is complete, copy all model files (model_nn.json, model_nn_weights.h5, model_norm.pkl, model_out.dat, model_wvl.dat) from the newly created subfolder `CNN_train` to the folder `parameters_model` that is located in the mail reduction pipeline folder.

