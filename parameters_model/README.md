# Parameter estimation

In this folder we have all files that are need for running a predetermined (convolutional) neural network that will determine stellar parameters and abundances from the normalised input spectra. Currentlly no quality flags are supplied allong the determined values.

**Files description**

Files that are needed by the procedure:

 - `model_nn.json` - Keras neural network model that contains architecture of the model
 - `model_nn_weights.h5` - Weights of the trained model
 - `model_wvl.dat` - Wavelength bins that are used in the model_nn
 - `model_norm.pkl` - Values of means and sigmas used for the normalisation of output parameters
 - `model_out.dat` - Ordered list of parameters produced by the network
