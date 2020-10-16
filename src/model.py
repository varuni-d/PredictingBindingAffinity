#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import pickle
import warnings
warnings.filterwarnings("ignore")


# In[3]:


models = ['LinearRegression','DecisionTreeRegressor','RandomForestRegressor']
score_table = pd.DataFrame(index = models, columns= ['rmse_train','rmse_test','mae_train','mae_test'])


# In[4]:


#Plot the performance for both training and test datasets
def plot_result(model_name,y_train, train_pred, y_test, test_pred):
    
    # compute the performance
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)

    mae_train = MAE(y_train, train_pred)
    mse_train = MSE(y_train, train_pred)
    rmse_train=np.sqrt(mse_train)

    mae_test = MAE(y_test, test_pred)
    mse_test = MSE(y_test, test_pred)   
    rmse_test=np.sqrt(mse_test)

#    score_table.loc[model_name,:] = r2_train, r2_test, mae_train, mae_test
    score_table.loc[model_name,:] = rmse_train, rmse_test, mae_train, mae_test


# In[7]:


#linear Regression model

def linear_regression(X_train, y_train, X_test, y_test):

  #  lr_model = Pipeline([('scaler', MinMaxScaler()),('lr_model',LinearRegression())])
  
    lr_model = Pipeline([('scaler', StandardScaler()),('lr_model',LinearRegression())])
    lr_model.fit(X_train, y_train)

    #predict on training set    
    train_pred = lr_model.predict(X_train)
    #repeat on test set 
    test_pred = lr_model.predict(X_test)
    
    plot_result('LinearRegression', y_train, train_pred, y_test, test_pred)  
    return test_pred


# In[8]:


def decisiontree(X_train, y_train, X_test, y_test):

    dt_model = Pipeline([('scaler', StandardScaler()),('dt_model',DecisionTreeRegressor(random_state=1))])
    #Hyper-parameter tuning and 5-fold cross-validation
    params_dt = {'dt_model__max_leaf_nodes': [150, 250, 300],'dt_model__max_features': ['log2', 'auto', 'sqrt'],'dt_model__min_samples_leaf': [10, 30, 50]}
    grid_dt=GridSearchCV(estimator=dt_model, param_grid=params_dt, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=-1)

    grid_dt.fit(X_train, y_train)

    #Predict and convert back to real
    train_pred = grid_dt.predict(X_train)

    #repeat on test set 
    test_pred = grid_dt.predict(X_test)

    plot_result('DecisionTreeRegressor', y_train, train_pred, y_test, test_pred)

    #Extract the best estimators
    best_hyperparameters=grid_dt.best_params_
    print('Best hyperparameters:',best_hyperparameters)
    best_model = grid_dt.best_estimator_
    print('Best model:',best_model)
    print('Corresponding score:', grid_dt.best_score_)
    
    return test_pred   


# In[9]:


def randomforest(X_train, y_train, X_test, y_test):
    #Instantiate model
    rf_model = Pipeline([('scaler', StandardScaler()),('rf_model',RandomForestRegressor(random_state=1))])
#    rf_model = RandomForestRegressor(random_state=1)
    #Hyper-parameter tuning and 5-fold cross-validation
    params_rf = {'rf_model__n_estimators': [100, 200, 300],'rf_model__max_features': ['log2', 'auto', 'sqrt'],'rf_model__min_samples_leaf': [10, 30, 50]}
    grid_rf=GridSearchCV(estimator=rf_model, param_grid=params_rf, scoring='neg_mean_absolute_error', cv=3, verbose=1, n_jobs=-1)

    #Fit
    grid_rf.fit(X_train, y_train)

    #Predict and convert back to real
    train_pred = grid_rf.predict(X_train)

    #repeat on test set 
    test_pred = grid_rf.predict(X_test)

    plot_result('RandomForestRegressor', y_train, train_pred, y_test, test_pred)

    #Extract the best estimators
    best_hyperparameters=grid_rf.best_params_
    print('Best hyperparameters:',best_hyperparameters)
    best_model = grid_rf.best_estimator_
    print('Best model:',best_model)
    print('Corresponding score:', grid_rf.best_score_)
    
    filename = 'randomforest_model.sav'
    pickle.dump(grid_rf, open(filename, 'wb'))
    
    return test_pred


# In[ ]:


def gradientboosting(X_train, y_train, X_test, y_test):
    
    # Instantiate gb
    gb_model = Pipeline([('scaler', StandardScaler()),('gb_model',GradientBoostingRegressor(random_state=1))])

    #Hyper-parameter tuning and 5-fold cross-validation
    params_gb = {'gb_model__n_estimators': [100, 500, 1000],'gb_model__max_depth': [5,10,20],'gb_model__learning_rate': [0.01,0.1,1.0]}
    grid_gb=GridSearchCV(estimator=gb_model, param_grid=params_gb, scoring='neg_mean_absolute_error', cv=3, verbose=1, n_jobs=-1)

    grid_gb.fit(X_train, y_train)

    #Predict and convert back to real
    train_pred = grid_gb.predict(X_train)
    
    #repeat on test set 
    test_pred = grid_gb.predict(X_test)

    plot_result('GradientBoostingRegressor', y_train, train_pred, y_test, test_pred)

    #Extract the best estimators
    best_hyperparameters=grid_gb.best_params_
    print('Best hyperparameters:',best_hyperparameters)
    best_model = grid_gb.best_estimator_
    print('Best model:',best_model)
    print('Corresponding score:', grid_gb.best_score_)
    
    filename = 'gradientboost_model.sav'
    pickle.dump(grid_gb, open(filename, 'wb'))
    
    return test_pred

