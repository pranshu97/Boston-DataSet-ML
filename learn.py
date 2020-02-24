import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Reading the dataset
data = pd.read_csv('Boston.csv')
X1 = data.iloc[:,1:5]
X2 = data.iloc[:,6:14]
X = pd.concat([X1,X2],axis=1)
y = pd.DataFrame(data.iloc[:,14])

# Preprocessing data
X = preprocessing.normalize(X)

# Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)


# Initialising the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

'''
model = pickle.load(open('model.sav', 'rb'))
'''

#Skip this step if model.sav is already trained and loaded
#Building and training the neural network
model = Sequential()
model.add(Dense(output_dim = 1024, init = 'normal', activation = 'relu', input_dim = 12))
model.add(Dense(output_dim = 512, init = 'normal', activation = 'relu'))
model.add(Dense(output_dim = 1, init = 'normal', activation = 'linear'))
model.compile(optimizer = 'adam', loss = 'mse') 
model.fit(X_train, y_train, batch_size = 5, nb_epoch = 200)

#Saving the model
import pickle
pickle.dump(model, open('model.sav','wb'))

# Predicting
y_pred = model.predict(X_test)

#Model Evaluation
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)
metrics.explained_variance_score(y_test, y_pred)
