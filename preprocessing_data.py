import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

#datasets(these have to be in the format of numpy arrays.
train_labels = []  #corresponding labels for the samples
train_samples = [] #train on economic data, image, text, etc.

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
