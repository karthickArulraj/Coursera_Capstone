# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 01:58:16 2020

@author: karulraj
"""

import pandas as pd
a =pd.read_csv("sr2.csv")
a.head()

X=a.iloc[:,:-1]
Y=a.iloc[:,-1]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =     \
               train_test_split(X,Y,test_size=0.3,random_state=1234)
              
from sklearn.linear_model import LinearRegression

sreg=LinearRegression()

sreg.fit(xtrain,ytrain)

# predict data

ypredict=sreg.predict(xtest)

# r square determinatrion
n =sreg.score(xtest,ytest)

# finding coeff and intercept

corff = sreg.coef_
inter = sreg.intercept_

from sklearn.metrics import mean_squared_error
import math

m=math.sqrt(mean_squared_error(ytest,ypredict))
