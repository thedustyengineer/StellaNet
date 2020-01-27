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


#x_train, y_train = utilities.FileOperations.build_dataset_from_grid_folder('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/grid', 0)
done_model = load_model('best_model.02-393734.68.h5')
spec_to_predict = utilities.FileOperations.read_fits_spectrum('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/grid/9000_3.00_-1.25_0.00_2.00_0.00_0.00_0.00.fits.gz',0,'0','0','0')
flux_values_to_predict = np.array(spec_to_predict.fluxes)
flux_values_to_predict = flux_values_to_predict.reshape(1,40000,1)
pred = done_model.predict(flux_values_to_predict)
print(pred)