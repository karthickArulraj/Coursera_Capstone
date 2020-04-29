# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:34:13 2020

@author: karulraj
"""
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# read dataset

bikes = pd.read_csv("hour.csv")

#preliminary analysis of the data & dropp obvious features


bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['instant','dteday','casual','registered'],axis=1)

# find missing values

bikes_prep.isnull().sum()

#visualisation of data using histogram
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

# visualising data in different form
#visualisation of continuos feature with demand
#subplot 

plt.subplot(2,2,1)
plt.title("temp vs demand")
plt.scatter(bikes_prep["temp"],bikes_prep["cnt"],s=0.1,c='g')


plt.subplot(2,2,2)
plt.title("atemp vs demand")
plt.scatter(bikes_prep["atemp"],bikes_prep["cnt"],s=0.1,c='b')


plt.subplot(2,2,3)
plt.title("humidity vs demand")
plt.scatter(bikes_prep["hum"],bikes_prep["cnt"],s=0.1,c='r')


plt.subplot(2,2,4)
plt.title("windspeed vs demand")
plt.scatter(bikes_prep["windspeed"],bikes_prep["cnt"],s=0.1,c='m')
plt.tight_layout()

# visualisation of categorical feature with deamnd
color=["r","b","g","m"]
plt.subplot(3,3,1)
plt.title("average demand per season")
cat_list = bikes_prep["season"].unique()
cat_average = bikes_prep.groupby("season").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)

plt.subplot(3,3,2)
plt.title("average demand per yr")
cat_list = bikes_prep["yr"].unique()
cat_average = bikes_prep.groupby("yr").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)


plt.subplot(3,3,3)
plt.title("average demand per mnth")
cat_list = bikes_prep["mnth"].unique()
cat_average = bikes_prep.groupby("mnth").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)

plt.subplot(3,3,4)
plt.title("average demand per hr")
cat_list = bikes_prep["hr"].unique()
cat_average = bikes_prep.groupby("hr").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)



plt.subplot(3,3,5)
plt.title("average demand per holiday")
cat_list = bikes_prep["holiday"].unique()
cat_average = bikes_prep.groupby("holiday").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)

plt.subplot(3,3,6)
plt.title("average demand per weekday")
cat_list = bikes_prep["weekday"].unique()
cat_average = bikes_prep.groupby("weekday").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)




plt.subplot(3,3,7)
plt.title("average demand per workingday")
cat_list = bikes_prep["workingday"].unique()
cat_average = bikes_prep.groupby("workingday").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)

plt.subplot(3,3,8)
plt.title("average demand per weathersit")
cat_list = bikes_prep["weathersit"].unique()
cat_average = bikes_prep.groupby("weathersit").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)
plt.tight_layout()

plt.title("average demand per hr")
cat_list = bikes_prep["hr"].unique()
cat_average = bikes_prep.groupby("hr").mean()["cnt"]
plt.bar(cat_list,cat_average,color=color)

#finding outlayers
bikes_prep["cnt"].describe()
bikes_prep["cnt"].quantile([0.005,0.10,0.15,0.90,0.950,0.99])

#checking assumoption
#coreelationn

correl=bikes_prep[["temp","atemp","hum","windspeed"]].corr()
bikes_prep=bikes_prep.drop(["atemp","windspeed","weekday","workingday","yr"],axis=1)

#autocollinerity
df1=pd.to_numeric(bikes_prep["cnt"],downcast="float")
plt.acorr(df1,maxlags=12)
#normalise the demand column

df1=bikes_prep["cnt"]
df2=np.log(df1)

plt.figure()
df1.hist(rwidth=0.8,bins=20)

plt.figure()
df2.hist(rwidth=0.8,bins=20)
bikes_prep["cnt"]=np.log(bikes_prep["cnt"])

#create new depanded columns for demand for auto collinearity

t_1=bikes_prep["cnt"].shift(+1).to_frame()
t_1.columns=["t-1"]

t_2=bikes_prep["cnt"].shift(+2).to_frame()
t_2.columns=["t-2"]
t_3=bikes_prep["cnt"].shift(+3).to_frame()
t_3.columns=["t-3"]

bikes_prep_lag=pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)

bikes_prep_lag=bikes_prep_lag.dropna()



#dummy variables

# junk dummy_df=pd.get_dummies(bikes_prep_lag,drop_first=True)

bikes_prep_lag.dtypes

bikes_prep_lag["season"]=bikes_prep_lag["season"].astype("category")
bikes_prep_lag["holiday"]=bikes_prep_lag["holiday"].astype("category")
bikes_prep_lag["weathersit"]=bikes_prep_lag["weathersit"].astype("category")
bikes_prep_lag["mnth"]=bikes_prep_lag["mnth"].astype("category")
bikes_prep_lag["hr"]=bikes_prep_lag["hr"].astype("category")

bikes_prep_lag=pd.get_dummies(bikes_prep_lag,drop_first=True)
X=bikes_prep_lag.drop(["cnt"],axis=1)
Y=bikes_prep_lag[["cnt"]]
#create train set
tr_size=0.7*len(X)
tr_size=int(tr_size)
#test and train data 
Xtrain=X.values[0:tr_size]
Xtest=X.values[tr_size:len(X)]

Ytrain=Y.values[0:tr_size]
Ytest=Y.values[tr_size:len(Y)]


#linear regressiion

from sklearn.linear_model import LinearRegression

bikes_reg=LinearRegression()

bikes_reg.fit(Xtrain,Ytrain)

# r square determinatrion
r1 =bikes_reg.score(Xtrain,Ytrain)
r2 =bikes_reg.score(Xtest,Ytest)

# predict data

Ypredict=bikes_reg.predict(Xtest)



# finding coeff and intercept

corff = bikes_reg.coef_
inter = bikes_reg.intercept_

from sklearn.metrics import mean_squared_error
import math
#rmse 
rmse=math.sqrt(mean_squared_error(Ytest,Ypredict))
#rmsle 
Ytest_e=[]
Ypredict_e=[]

for i in range(0,len(Ytest)):
    Ytest_e.append(math.exp(Ytest[i]))
    Ypredict_e.append(math.exp(Ypredict[i]))
    
#calculate rmse
log_sq_sum=0.0
for i in range(0,len(Ytest_e)):
    log_a=math.log(Ytest_e[i]+1)
    log_p=math.log(Ypredict_e[i]+1)
    log_diff=(log_p-log_a)**2
    log_sq_sum=log_sq_sum+log_diff
    
rmsle=math.sqrt(log_sq_sum/len(Ytest))

print("the rmsle value is {}".format(rmsle))
    
    
    


