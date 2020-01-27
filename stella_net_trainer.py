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


def batchOutput(batch, logs):
    print("Finished batch: " + str(batch))
    print(logs)


batchLogCallback = LambdaCallback(on_batch_end=batchOutput)

num_points = 40000
num_columns = 1 #(wave, flux) # same as number of features, only using flux to start...
input_shape = num_points * num_columns

labels = [3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8250,8500,8750,9000, \
    9250,9500,9750,10000,10250,10500,10750,11000,11250,11500,11750,12000,12250,12500,12750,13000]

BATCH_SIZE = 128
EPOCHS = 1000

model_m = Sequential()
#model_m.add(Reshape((num_points, num_columns), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 3, activation='relu', input_shape=(num_points, num_columns)))
model_m.add(Conv1D(100, 3, activation='relu'))
#model_m.add(MaxPooling1D(3))
#model_m.add(Conv1D(100, 3, activation='relu'))
#model_m.add(Conv1D(100, 3, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(1, activation='linear'))
print(model_m.summary())


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    #keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=100),
    batchLogCallback
]

model_m.compile(loss='mean_squared_error',
                optimizer='adam', metrics=['mean_squared_error'])




x_train, y_train = utilities.FileOperations.build_dataset_from_grid_folder('/Volumes/Storage/nn_R55kA_FG42kA_grid_spectrum/perturbed_grid', 0)
x2_train = np.asarray(x_train, dtype=np.float16)
x2_train = np.expand_dims(x2_train, axis=2)
print(x2_train.shape)

history = model_m.fit(x2_train, # x-train contains data, y-train contains labels
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=2)


