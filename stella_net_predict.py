from numpy import mean
from numpy import std
from numpy import dstack
import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
import utilities
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalAveragePooling1D, AveragePooling1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from keras.models import load_model
from astropy.convolution import convolve, Box1DKernel


#x_train, y_train = utilities.FileOperations.build_dataset_from_grid_folder('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/grid', 0)
done_model = load_model('best_model.476-137.77-0.02-0.01.h5')
#spec_to_predict = utilities.FileOperations.read_fits_spectrum('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/grid/8000_3.50_-2.00_0.00_2.00_0.00_0.00_0.00.fits.gz',0,'0','0','0')
spec_to_predict = utilities.FileOperations.read_fits_spectrum('/Volumes/Storage/gamcru_ADP.2016-09-27T11_56_48.190.fits',1,'wave','flux','err', read_range=[400,670], is_feros=True)

#spec_to_predict.write_column_spectrum('/Volumes/Storage/', filename='hd221756_test.tsv')
#spec_to_predict = utilities.FileOperations.read_tsv_spectrum('/Volumes/Storage/hd221756_test.nspec')
#spec_to_predict = utilities.FileOperations.read_tsv_spectrum('/Volumes/Storage/vega_test_good_norm.nspec')
#spec_to_predict.fluxes = convolve(spec_to_predict.fluxes, Box1DKernel(20))
#spec_to_predict.fluxes = convolve(spec_to_predict.fluxes, Box1DKernel(100))
spec_to_predict.write_column_spectrum('/Volumes/Storage',use_opt_params=True, filename='gam_cru_feros_1.tsv')
#spec_to_predict.SplineNormalize(4)
#spec_to_predict.write_column_spectrum('/Volumes/Storage/', filename='vega_test_quick_norm.nspec')
#spec_to_predict.apply_rad_vel_shift(7.9)
#spec_to_predict.PlotSpectrum()
spec_to_predict.CutAndInterpolateFluxesToGrid(27000, replace_nan=True)
#spec_to_predict.MaxNormalize()

print(spec_to_predict.wavelengths)

spec_to_predict.PlotSpectrum()
flux_values_to_predict = np.array(spec_to_predict.fluxes)
flux_values_to_predict = flux_values_to_predict.reshape(1,27000,1)
pred1, pred2, pred3 = done_model.predict(flux_values_to_predict)
print(pred1, pred2, pred3)

