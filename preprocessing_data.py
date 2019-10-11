import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

#datasets(these have to be in the format of numpy arrays.
train_labels = []  #corresponding labels for the samples
train_samples = [] #economic data, image, text, all kinds of samples, etc.

'''
Example data: An experimental drug was tested on individuals from ages 13 to 65. 
The trial had 2100 participants. 
Half were under 65 years old, half were over 65 years old. 
95% of patients over 65 or older experienced side effects. 
95% of patients under 65 experienced no side effects.

We want out neural network to detect whether or not 
an individual is likely to experience side effect or not.
'''
for i in range(1000):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(50):
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

#print raw data
for i in train_samples:
    print(i)

for i in train_labels:
    print(i)
    #0 stands for no experience of side effects


#converting labels and samples into numpy array so that keras can interpret them
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

#However, in this example, our neural network may not learn much from such small amount of data.
#so we use scaler to get more accurate predictions
scaler = MinMaxScaler(feature_range=(0, 1))#scale input 0 to 1
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))#technical formality for fuction

#print scaled data
for i in scaled_train_samples:
    print(i)

#create model.
model = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

#each time we build a model, before we can train it, we must compile it
model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit the model with our training data set(train samples and labels)
#Train Samples and Train labels as the first two parameters in our 'fit' function
#the train samples are the entire training set.
# and keras expects these training data set to be in the format of either a numpy array
# or a list of numpy arrays.

#Validation Set: there is no need to explicitly make such set
#parameter validation_split will split out the training set (20%)
# and use it as a validation set. (So we are implicitly making validation set)
model.fit(scaled_train_samples, train_labels,
          validation_split=0.20, batch_size=10, epochs=20, shuffle='True', verbose=2)

#another way to Explicitly pass on the validation set:
#make a validation set in the format of tuple with sample and label
# valid_set = [(sample, label), ... , (sample, label)]

#PREDICTIONS (passing our unlabeled test data and have our model to make predictions)
#model has no access to the labels
predictions = model.predict(scaled_train_samples, batch_size=10, verbose=0)

for i in predictions:
    print(i)

#probabilities of each categories
#0.95 -> 95% of falling into the first category, 0.04->4% of falling into the second one


#OverFitting? -> occurs when our model becomes really good at being able to classify our predict on data in the training set
# but not so good at classifying data that it wasn't trained on.
# we can tell by observing the metrics
# if the val_metrics are considerably worse than the training metrics, then that's an
# indication that our model is overfitting.
# Model is unable to generalize well.
# if the model is given test data that is slightly deviated from the train data set,
#then it won't be able to generalize and accurately predict the output.

#How to reduce it?
#add more data. the more it will be able to learn from the training set.
#Data Augmentation- modify data
#Dropout-randomly ignore, drop out nodes from layer, preventing dropout nodes from participating

#Underfitting? -> opposite of overfitting.
#when metrics given for the data is poor.
#Increase complexity. (increase number of neurons, layers, type of layers, etc.
#add more features to samples. (help model to classify datat better)
#Reduce Dropout. (regularization technique.