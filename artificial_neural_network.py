import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

#linear stack of layers
model = Sequential([
    #Dense layer, number of neurons(16)
    #needs to know what kind of input (1,) tuple integers
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.summary()