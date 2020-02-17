import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading Data
data = pd.read_csv('Boston.csv')
X1 = data.iloc[:,1:5]
X2 = data.iloc[:,6:14]
X = pd.concat([X1,X2],axis=1)
y = pd.DataFrame(data.iloc[:,14])

#Data Preprocessing
from sklearn import preprocessing
X = preprocessing.normalize(X)

#Splitting Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Neural Network
classifier = Sequential()

classifier.add(Dense(output_dim = 512, init = 'normal', activation = 'relu', input_dim = 12))
classifier.add(Dense(output_dim = 128, init = 'normal', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))

classifier.compile(optimizer = 'adam', loss = 'mse') 

classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 100)

y_pred1 = classifier.predict(X_test)

from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred1)
metrics.explained_variance_score(y_test, y_pred1)
