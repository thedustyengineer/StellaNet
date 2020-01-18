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
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical




num_points = 400000
num_columns = 1 #(wave, flux) # same as number of features, only using flux to start...
input_shape = num_points * num_columns

labels = [3500,3750,4000,4250,4500,4750]

model_m = Sequential()
#model_m.add(Reshape((num_points, num_columns), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 50, activation='relu', input_shape=(num_points, num_columns)))
model_m.add(Conv1D(100, 50, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 50, activation='relu'))
model_m.add(Conv1D(160, 50, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(1, activation='relu'))
print(model_m.summary())


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='mean_squared_error',
                optimizer='adam', metrics=['mean_squared_error'])

BATCH_SIZE = 12
EPOCHS = 50


x_train, y_train = utilities.FileOperations.build_dataset_from_ispec_grid_folder('/Users/dustin/iSpec/nn_grid_spectrum/grid', 0)
x_train = np.asarray(x_train, dtype=np.float32)
x_train = np.expand_dims(x_train, axis=2)
print(x_train.shape)

history = model_m.fit(x_train, # x-train contains data, y-train contains labels
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=2)


