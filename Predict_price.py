# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:18:17 2019

@author: Kirti Sodhi
"""
#####importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn import metrics
from sklearn import preprocessing


#####importing file
df=pd.read_excel(r'C:\Users\Kirti Sodhi\Desktop\kc_house_data.xlsx')

##### Exploratory Data analysis
df.describe()
df=df.drop(['id'],axis=1) 
df['date'] = pd.DatetimeIndex(df['date']).year
df['date']=df['date'].astype('object')
df['bedrooms']=df['bedrooms'].astype('object')
df['bathrooms']=df['bathrooms'].astype('object')
df['yr_built']=df['yr_built'].astype('object')
df['yr_renovated']=df['yr_renovated'].astype('object')
df.columns.to_series().groupby(df.dtypes).groups
df.isnull().sum(axis=0)

######Converting date to category
le=preprocessing.LabelEncoder()
categorical_feature_mask = df.dtypes==object
categorical_cols = df.columns[categorical_feature_mask].tolist()
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

#######Checking for outliers in the data
def reject_outliers(data):
 u = np.mean(data["price"])
 s = np.std(data["price"])
 data_filtered = data[(data["price"]>(u-2*s)) & (data["price"]<(u+2*s))]
 return data_filtered
df1=reject_outliers(df)


#plt.bar(df1['bedrooms'],df1['price'])
#plt.scatter(df1['sqft_living'],df1['price'],color='g')
#plt.xlabel("Feature")
#plt.ylabel("Response")
#plt.show()

#####Diving features and response variables
Y=df1.iloc[:,1]

#####Dropped based on multicollinearity
#X=df.drop(['price'],axis=1)
X=df1.drop(['price','sqft_living','bathrooms','sqft_living15','sqft_lot15','grade','sqft_above','zipcode','yr_built'],axis=1)

#####Checking for multicollinearity in the data
#corr = X.corr()
#sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
#heat_map=plt.gcf()
#heat_map.set_size_inches(15,15)
#plt.xticks(fontsize=10)
#plt.yticks(fontsize=10)

#######Building Model
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred=lin_reg.predict(X_test)

#######RMSE
print( np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(lin_reg.intercept_)
print(f'R^2 score: {lin_reg.score(X_train, y_train)}')
#
######calling the summary of linear regression model
X_constant = sm.add_constant(X_train)
lin_reg = sm.OLS(y_train,X_constant).fit()
print(lin_reg.summary())
##
#######Assumption: Homoscedasticity
lr=LinearRegression()
visualizer=ResidualsPlot(lr)
visualizer.fit(X_train,y_train)
visualizer.score(X_test,y_test)
visualizer.poof()
#
#########Assumption:Normality of Error
mod_fit=sm.OLS(y_train,X_train).fit()
res=mod_fit.resid
fig=sm.qqplot(res,fit=True,line='45')
plt.show()