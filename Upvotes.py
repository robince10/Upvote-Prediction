# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Dataset
train = pd.read_csv('Train_Set.csv')
test = pd.read_csv('Test_Set.csv')

train['Tag'].value_counts()
train.drop(['ID','Username'],axis = 1, inplace = True)
test.drop(['ID','Username'],axis = 1, inplace = True)

train.isnull().sum()

df = [train,test]
df = pd.concat(df,axis = 0)

train.corr()

X = train['Views'].values.reshape(-1,1)
Y = train['Upvotes']
X = pd.DataFrame(X)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(X,Y,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_cv_pred = regressor.predict(X_cv)

rmse = sqrt(mean_squared_error(Y_cv_pred,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

regressor_OLS.summary()

df_2 = [X,train['Reputation']]
df_2 = pd.concat(df_2,axis = 1)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(df_2,Y,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor_2 = LinearRegression()
regressor_2.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_pred_cv_2 = regressor_2.predict(X_cv)

rmse_2 = sqrt(mean_squared_error(Y_pred_cv_2,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

df_3 = [df_2,train['Answers']]
df_3 = pd.concat(df_3,axis = 1)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(df_3,Y,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor_3 = LinearRegression()
regressor_3.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_pred_cv_3 = regressor_3.predict(X_cv)

rmse_3 = sqrt(mean_squared_error(Y_pred_cv_3,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

regressor_OLS.summary()

df_4 = [train,test]
df_4 = pd.concat(df_4,axis = 0)

df_4 = pd.get_dummies(df_4)
df_4.drop(['Tag_a'],axis = 1, inplace = True)

x_train = df_4[:330045]
x_test = df_4[330045:]

x_test.drop(['Upvotes'],axis = 1, inplace = True)
y_train = x_train['Upvotes']
x_train.drop(['Upvotes'],axis = 1, inplace = True)

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(x_train,y_train,test_size = 0.2, random_state = 9)

from sklearn.linear_model import LinearRegression
regressor_4 = LinearRegression()
regressor_4.fit(X_train,Y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt

Y_pred_cv_4 = regressor_4.predict(X_cv)

rmse_4 = sqrt(mean_squared_error(Y_pred_cv_4,Y_cv))

import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = Y_train,exog = X_train).fit()

regressor_OLS.summary()

y_pred = regressor_4.predict(x_test)

y_pred = pd.DataFrame(y_pred)

y_pred.to_csv('Sample.csv')



