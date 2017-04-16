#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Place to hide functions referenced in notebook ###

import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
import feature_engineering as fe


# Load Kaggle train and test datasets
def load_data():
    train_df = pd.read_csv('data/train.csv', index_col=0, infer_datetime_format=True)
    train_df.index.name=None # Remove index name to remove confusing datetime column
    train_df.index = pd.to_datetime(train_df.index) # Convert index to datetime
    test_df = pd.read_csv('data/test.csv', index_col=0, infer_datetime_format=True)
    test_df.index.name=None # Remove index name to remove confusing datetime column
    test_df.index = pd.to_datetime(test_df.index) # Convert index to datetime


# Define some time based variables
def eda_transform(df):
    df['hour'] = df.index.hour
    df['weekday name'] = df.index.weekday_name
    df['weekday'] = df.index.weekday
    df['weekofyear'] = df.index.weekofyear
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['atempsq'] = df['atemp']**2
    df['tempsq'] = df['temp']**2
    temp_df = df
    temp_df['temp'] = ('0' + df['month'].astype(str))
    df['year_month']=df.year.astype(str) + temp_df.temp.str[-2:]
    return df


# Create boxplots for registered/casual rides by hour
def boxplot_by_hour(df):
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    df.boxplot(column='registered',by='hour',ax=ax1)
    df.boxplot(column='casual',by='hour',ax=ax2)
    ax1.set_ylabel('rides')
    ax2.set_ylabel('rides')
    ax1.set_ylim(-10,900)
    ax2.set_ylim(-10,250)
    plt.show()


# Create boxplots for registered/casual rides throughout time
def boxplot_by_yearmonth(df):
    plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    df.boxplot(column='registered',by='year_month',ax=ax1)
    df.boxplot(column='casual',by='year_month',ax=ax2)
    ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(),rotation=90)
    ax2.set_xticklabels(ax2.xaxis.get_majorticklabels(),rotation=90)
    ax1.set_ylabel('rides')
    ax2.set_ylabel('rides')
    ax1.set_ylim(-10,800)
    ax2.set_ylim(-10,250)
    plt.show()


# Define pipeline steps
def define_pipeline():
    categorical = ('season', 'holiday', 'workingday', )
    numerical = ('datetime', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',) # Datetime isn't numerical, but needs to be in the numeric branch
    pipeline = Pipeline([
        # Process cat & num separately, then join back together
        ('union', FeatureUnion([ 
            ('categorical', Pipeline([
                ('select_cat', fe.SelectCols(cols = categorical)),
                ('onehot', OneHotEncoder()),    
            ])),    
            ('numerical', Pipeline([
                ('select_num', fe.SelectCols(cols = numerical)),
                ('date', fe.DateFormatter()),
                #('drop_datetime', fe.SelectCols(cols = ('datetime'), invert = True)),
                ('temp', fe.ProcessNumerical(cols_to_square = ('temp', 'atemp', 'humidity'),)),
                # ('bad_weather', fe.BinarySplitter(col = 'weather', threshold = 2)),
                # ('filter', fe.PassFilter(col='atemp', lb = 15, replacement_style = 'mean'))
                ('scale', StandardScaler()),    
            ])),    
        ])),
        ('to_dense', preprocessing.FunctionTransformer(lambda x: x.todense(), accept_sparse=True)), 
        ('clf', GradientBoostingRegressor(n_estimators=100,random_state=2)),
    ])
    return pipeline

def param_tuning_graphs(train_data,dev_data,train_label,pipeline,parameter,param_values):

    categorical = ('season', 'holiday', 'workingday', )
    numerical = ('datetime', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',) # Datetime isn't numerical, but needs to be in the numeric branch
    pipeline = Pipeline([
        # Process cat & num separately, then join back together
        ('union', FeatureUnion([ 
            ('categorical', Pipeline([
                ('select_cat', fe.SelectCols(cols = categorical)),
                ('onehot', OneHotEncoder()),    
            ])),    
            ('numerical', Pipeline([
                ('select_num', fe.SelectCols(cols = numerical)),
                ('date', fe.DateFormatter()),
                #('drop_datetime', fe.SelectCols(cols = ('datetime'), invert = True)),
                ('temp', fe.ProcessNumerical(cols_to_square = ('temp', 'atemp', 'humidity'),)),
                # ('bad_weather', fe.BinarySplitter(col = 'weather', threshold = 2)),
                # ('filter', fe.PassFilter(col='atemp', lb = 15, replacement_style = 'mean'))
                ('scale', StandardScaler()),    
            ])),    
        ])),
        ('to_dense', preprocessing.FunctionTransformer(lambda x: x.todense(), accept_sparse=True)), 
        #('clf', GradientBoostingRegressor(n_estimators=100,random_state=2)),
    ])

    # Run train and dev data through pipeline for feature engineering
    features = [c for c in train_data.columns if c not in ['count', 'casual', 'registered', 'log_casual', 'log_registered']]
    fe_train_data = pipeline.fit_transform(train_data[features])
    fe_dev_data = pipeline.transform(dev_data[features])

    row_format = "{:>10}" *(6)
    rmse_list=[]
    for i in param_values:
        t0 = time()
        if parameter == 'n_estimators':
            gb = GradientBoostingRegressor(n_estimators=i,learning_rate=0.05,max_depth=10, min_samples_leaf=20,random_state=2)
        if parameter == 'learning_rate': 
            gb = GradientBoostingRegressor(n_estimators=115,learning_rate=i,max_depth=10, min_samples_leaf=20,random_state=2)
        if parameter == 'max_depth': 
            gb = GradientBoostingRegressor(n_estimators=115,learning_rate=0.05,max_depth=i, min_samples_leaf=20,random_state=2)
        if parameter == 'min_samples_leaf': 
            gb = GradientBoostingRegressor(n_estimators=115,learning_rate=0.05,max_depth=10, min_samples_leaf=i,random_state=2)
        gb.fit(fe_train_data, train_data[train_label])
        predicted_y = gb.predict(fe_dev_data)
        rmse = get_RMSE(actual_values = dev_data[train_label], predicted_values = predicted_y)
        rmse_list.append(round(rmse,3))
        print row_format.format(parameter+":", i, "RMSE:", round(rmse,3),
                                "Runtime:", round((time() - t0),3))
    plt.plot(param_values,rmse_list)
    plt.show()

    
def train_dev_model_search(pipeline, train_data, dev_data, train_label, parameters, RMSE_scorer):

    print("Performing grid search...")
    t0 = time()
    gs = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring=RMSE_scorer)
    features = [c for c in train_data.columns if c not in ['count', 'casual', 'registered', 'log_casual', 'log_registered']]
    gs.fit(train_data[features], train_data[train_label])
    print("Best parameters set:")
    best_param = gs.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_param[param_name]))
    predicted_y = gs.predict(dev_data[features])
    print "GridSearch RMSE " + str(gs.best_score_)
    rmse = get_RMSE(actual_values = dev_data[train_label], predicted_values = predicted_y)
    print "RMSE: ", str(rmse)
    print("Done in %0.3fs" % (time() - t0))
    print ""


def get_RMSE(actual_values, predicted_values):
    n = len(actual_values)
    RMSE = np.sqrt(np.sum(((np.log(predicted_values + 1) - np.log(actual_values + 1)) ** 2) / n))
    return RMSE


#### I THINK WE CAN LEAVE THE BELOW IN THE NOTEBOOK
# RMSE_scorer = make_scorer(get_RMSE, greater_is_better = False)


