## @package stella_net_predict
# File contains StellaNet Prediction class definition

# local imports
from . import stella_net_utilities

# other imports
import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
from keras.models import load_model


## Methods used for prediction with the neural network
class Prediction:
    ## Gets predictions for effective temperature, log g, and metallicity ([M/H])
    # @param a StellaNet spectrum.Spectrum object that the predictions will be generated for
    # @return a tuple of teff (K), log g (dex), [M/H] (dex)
    @staticmethod
    def getPredictions(spectrum):
        logg_mh_model = load_model('best_model.06-1213.08-0.20-0.09.h5') # this model is good only for log g and [M/H]
        teff_model = load_model('best_model.476-137.77-0.02-0.01.h5') # this model is good only for teff
        spec_to_predict = spectrum
        spec_to_predict.CutAndInterpolateFluxesToGrid(27000, replace_nan=True) # the default neural network requires a shape of 1, 27000, 1
        flux_values_to_predict = np.array(spec_to_predict.fluxes)
        flux_values_to_predict = flux_values_to_predict.reshape(1,27000,1)
        drop, logg, mh = logg_mh_model.predict(flux_values_to_predict) # dropped values are not used
        teff, drop2, drop3 = teff_model.predict(flux_values_to_predict)
        print('Effective Temperature: ' + str(teff) + '\n' + 
        'Log (g): ' + str(logg) + '\n' + 
        '[M/H]: ' + str(mh))
        return teff, logg, mh

