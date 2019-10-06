'''
Supervised Learning: when your deep learning model learns and makes inferences
from data that has already been labeled.

Unsupervised Learning: when the model learns and makes inferences from unlabeled
data.

ex)classify images of cats and dogs

Models in Deep Learning: Artificial Neural Networks
->input layers, hidden layers, output layers
'''

#Keras Sequential Model (the Sequential model is a linear stack of layers)
'''
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    #Dense is a type of a layer(there are many kinds, they all process different things)
    #connects each input layer to each output layer
    #first paramenter: number of neurons(Nodes)
    #second paramenter: shape of our data passing into our model
    #third parameter: non-linear function that follows a Dense layer

    #Define Hiddel layer(along with implicit specification of input layer)
    Dense(32, input_shape=(3,), activation='relu'),
    Dense(2, activation='softmax'), #Define Output layer
    #in the next layer, we're not putting in any more input_shape
    #   because only the first layer requires an input shape
    #No need to Specify Input layer in other layers that will come in between.
])
'''
#Different format of adding layers to your model:
'''
model = Sequential()
model.add(Dense(5, input_shape=(3,)))
model.add(Activation('relu'))
'''
#Activation Function of a neuron defines the output of that neuron given a set of inputs
#transform the inputs into certain numbers between 0 and 1.
#closer to 1, the more activated the neuron is, and the less activated otherwise(closer to 0)
#Biologically inspired by activity in our brains, where different neurons fire,
#or are ACTIVATED, by different stimuli.

#'relu' rectified linear unit, transforms input into either 0 or input itself.
#the more positive the neuron is, the more activated it is.


#Train an Artificial Neural Network
#solve an otimization problem.
#During training, the arbitrary wight values will constantly be updated until
# they reach optimized level
#SGD(minimize loss function, making loss as close to 0 as possible)
#During training, we supply our model labels and data to that data
#(i.e. images of cats and dogs)
#epoch: a single pass through the data model
#First the model will be set with arbitrary weights.
#At the end of the model the output layer will "spit out" the output for given input
#compute the loss or the error of that specific output by looking at what the model predicted
#for that input VS what the input truly is.
#Update the weight with new value

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])
#Optimizer Adam, pass learning grade 0.0001
#Loss(there are many other types)
#what will be the output = 'accuracy'
model.compile(Adam(lr=.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#(training data, train labels of this training data,
#how many pieces of data sent to the model at once
#20 individual passes through our model
#shuffle->literally..
#verbose-> how many outputs?
#model.fit(scaled_train_samples, train_labels, batch_size=10, epochs=20, shuffle='True', verbose=2)