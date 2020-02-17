## @package stella_net_trainer
# File contains StellaNet trainer logic

# local imports
import stella_net_exceptions
import stella_net_utilities

# numpy and matplotlib
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
# os
import os; os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

# keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalAveragePooling1D, AveragePooling1D, concatenate
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from keras.models import load_model, Model


## Shuffles the input data such that the labels and data are shuffled in the same way
# must be run before training begins so that training and validation sets are randomized
# @param arr1 - first array
# @param arr2 - second array
# @return the shuffled arrays
# @exception stella_net_exceptions.ArrayLengthMismatchError
def shuffle_arrays(arr1, arr2):
    if (len(arr1) != len(arr2)):
        raise stella_net_exceptions.ArrayLengthMismatchError
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]

## Callback function to get batch info to print for each batch
# @param batch - the batch
# @param logs - the log data
def batchOutput(batch, logs):
    print("Finished batch: " + str(batch))
    print(logs)

batchLogCallback = LambdaCallback(on_batch_end=batchOutput)

# some needed info to tell keras what the shape of the data is
num_points = 12500
num_columns = 1 
input_shape = num_points * num_columns


BATCH_SIZE = 64 # if you reduce batch size to 32 or 16 make sure to adjust epochs as necessary
EPOCHS = 50  # ~10 is enough for log g, [M/H] (might even be overfit depending on how you tune hyperparameters)
             # for Teff usually < 500 epochs is plenty to achieve convergence without overfitting
             # you can train once by running for 500 epochs and then use an later model for the teff
             # and an earlier model for log g and [M/H]

# the teff branch
spectra_input = Input(shape=(num_points,1))
output_1 = Conv1D(100, 3, activation='relu')(spectra_input)
output_1 = Flatten()(output_1)
output_1 = Dropout(0.5)(output_1)
pred_teff = Dense(1, activation='relu', name='predicted_teff')(output_1)

# the [M/H] branch
output_2 = Conv1D(50, 5, strides=9, activation='relu')(spectra_input)
output_2  = Flatten()(output_2)
output_2  = Dropout(0.5)(output_2)
pred_mh = Dense(1, activation='linear', name='predicted_mh')(output_2)

# the log g branch
output_3 = Conv1D(100, 21, strides=21, activation='relu')(spectra_input)
output_3 = Flatten()(output_3)
outout_3 = Dropout(0.5)(output_3)
pred_logg = Dense(1, activation='linear', name='predicted_logg')(output_3)

# the vsini branch
output_4 = Conv1D(50, 9, strides=9, activation='relu')(spectra_input)
output_4 = Flatten()(output_4)
outout_4 = Dropout(0.5)(output_4)
pred_vsini = Dense(1, activation='linear', name='predicted_vsini')(output_4)


model = Model(inputs=spectra_input, outputs=[pred_teff, pred_logg, pred_mh, pred_vsini])

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.compile(loss=['mean_absolute_error','mean_squared_error','mean_squared_error','mean_squared_error'],
                optimizer='adam', metrics=['mean_absolute_error'])

# the min_lr is set to 0.0001 to get the log g and [M/H] branches to train
# if just training effective temperature 0.001 works fine to get it started
# you can tweak the patience values as needed
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_predicted_teff_loss:.2f}-{val_predicted_logg_loss:.2f}-{val_predicted_mh_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True, mode='min'), 
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001),
    keras.callbacks.EarlyStopping(monitor='val_predicted_logg_loss', patience=20),
    batchLogCallback
]

# the first time you run the neural network on a new grid you should run ths to build
# the dataset from the grid directory and then save it to a numpy bindary file (.npy)
# which will dramatically improve performance on future runs
#
#x_train, y_train = stella_net_utilities.FileOperations.build_dataset_from_grid_folder('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_400-525nm_normalized', 0, save_npy_binary_file=True)

# on subsequent runs of the neural network (i.e. multiple runs to tweak hyperparameters)
# you should use the saved .npy binaries from the last step.
x_train, y_train = stella_net_utilities.FileOperations.build_dataset_from_npy_binaries('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_400-525nm_normalized/x_train.npy', '/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_400-525nm_normalized/y_train.npy')
x_train, y_train = shuffle_arrays(x_train, y_train) # randomize the data set so that the validation set is not just the last x% of the data
x_train = np.expand_dims(x_train, axis=2) # expand dims to what model.fit expects
y_train = np.split(y_train, 4, axis=1) # split the training labels
print(x_train.shape) # check the shape

history = model.fit(x_train, # x-train contains data, y-train contains labels
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.25,
                      verbose=2)
###
# Plot training & validation accuracy values
###

plt.ioff()

figure_teff = plt.figure()
plt.plot(history.history['predicted_teff_loss'])
plt.plot(history.history['val_predicted_teff_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['Teff (Train)', 'Teff (Test)'], loc='upper left')
plt.savefig('teff')
plt.close(figure_teff)

figure_mh = plt.figure()
plt.plot(history.history['predicted_mh_loss'])
plt.plot(history.history['val_predicted_mh_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['[M/H] (Train)', '[M/H] (Test)'], loc='upper left')
plt.savefig('mh')
plt.close(figure_mh)

figure_logg = plt.figure()
plt.plot(history.history['predicted_logg_loss'])
plt.plot(history.history['val_predicted_logg_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['log(g) (Train)', 'log(g) (Test)'], loc='upper left')
plt.savefig('logg')
plt.close(figure_logg)

figure_vsini = plt.figure()
plt.plot(history.history['predicted_vsini_loss'])
plt.plot(history.history['val_predicted_vsini_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['vsini (Train)', 'vsini (Test)'], loc='upper left')
plt.savefig('vsini')
plt.close(figure_vsini)

# Plot training & validation loss values
figure_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss')
plt.close(figure_loss)
