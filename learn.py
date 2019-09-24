import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#for nox prediction
dataset = pd.read_csv('Boston.csv')
X1= dataset.iloc[:,1:5].values
X2=  dataset.iloc[:,6:].values
X = np.concatenate((X1,X2),axis=1)
y = dataset.iloc[:,5].values
#for price prediction
'''
dataset = pd.read_csv('Boston.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))