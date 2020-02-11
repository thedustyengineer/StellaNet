from numpy import mean
from numpy import std
from numpy import dstack
from numpy import random
import numpy as np
import os; os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
import utilities
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalAveragePooling1D, AveragePooling1D, concatenate
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
from keras.utils import plot_model
from keras.models import load_model, Model
import matplotlib.pyplot as plt


def batchOutput(batch, logs):
    print("Finished batch: " + str(batch))
    print(logs)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

batchLogCallback = LambdaCallback(on_batch_end=batchOutput)

num_points = 27000
num_columns = 1 #(wave, flux) # same as number of features, only using flux to start...
input_shape = num_points * num_columns


BATCH_SIZE = 64
EPOCHS = 1000

#model_m = Sequential()
#model_m.add(Conv1D(100, 15, activation='relu', input_shape=(num_points, num_columns)))
#model_m.add(Flatten())
#model_m.add(Dropout(0.5))
#model_m.add(Dense(3, activation='relu'))
#print(model_m.summary())

spectra_input = Input(shape=(num_points,1))
output_1 = Conv1D(100, 3, activation='relu')(spectra_input)
output_1 = Flatten()(output_1)
output_1 = Dropout(0.5)(output_1)
pred_teff = Dense(1, activation='relu', name='predicted_teff')(output_1)

output_2 = Conv1D(50, 5, strides=9, activation='relu')(spectra_input)
output_2  = Flatten()(output_2)
output_2  = Dropout(0.5)(output_2)
pred_mh = Dense(1, activation='linear', name='predicted_mh')(output_2)

output_3 = Conv1D(100, 21, strides=21, activation='relu')(spectra_input)
output_3 = Flatten()(output_3)
outout_3 = Dropout(0.5)(output_3)
pred_logg = Dense(1, activation='linear', name='predicted_logg')(output_3)



model = Model(inputs=spectra_input, outputs=[pred_teff, pred_logg, pred_mh])

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.compile(loss=['mean_absolute_error','mean_squared_error','mean_squared_error'],
                optimizer='adam', metrics=['mean_absolute_error'])




callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_predicted_teff_loss:.2f}-{val_predicted_logg_loss:.2f}-{val_predicted_mh_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True, mode='min'), 
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001),
    keras.callbacks.EarlyStopping(monitor='val_predicted_teff_loss', patience=100),
    batchLogCallback
]






#x_train, y_train = utilities.FileOperations.build_dataset_from_grid_folder('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_grid_400-670nm', 0, save_npy_binary_file=True)

x_train, y_train = utilities.FileOperations.build_dataset_from_npy_binaries('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_grid_400-670nm/x_train_all_params.npy', '/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_grid_400-670nm/y_train_all_params.npy')
x_train, y_train = unison_shuffled_copies(x_train, y_train) # randomize the data set so that the validation set is not just the last x% of the data
x_train = np.expand_dims(x_train, axis=2)
y_train = np.split(y_train,3, axis=1)
print(x_train.shape)

history = model.fit(x_train, # x-train contains data, y-train contains labels
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.25,
                      verbose=2)

# Plot training & validation accuracy values
plt.plot(history.history['predicted_teff_loss'])
plt.plot(history.history['val_predicted_teff_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['Teff (Train)', 'Teff (Test)'], loc='upper left')
plt.show()

plt.plot(history.history['predicted_mh_loss'])
plt.plot(history.history['val_predicted_mh_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['[M/H] (Train)', '[M/H] (Test)'], loc='upper left')
plt.show()

plt.plot(history.history['predicted_logg_loss'])
plt.plot(history.history['val_predicted_logg_loss'])
plt.title('Model Loss')
plt.ylabel('Model Loss')
plt.xlabel('Epoch')
plt.legend(['log(g) (Train)', 'log(g) (Test)'], loc='upper left')
plt.show()


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

