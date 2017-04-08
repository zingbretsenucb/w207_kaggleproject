# W207 Final Project
# Bike Sharing Demand Kaggle Competition
# Team Members: Zach Ingbretsen, Nicholas Chen, Keri Wheatley, and Rob Mulla
# Kaggle Link: https://www.kaggle.com/c/bike-sharing-demand




##############################################
##############################################
##############################################
# BUSINESS UNDERSTANDING

# What problem are we trying solve?
# What are the relevant metrics? How much do we plan to improve them?
# What will we deliver?

# A public bicycle-sharing system is a service in which bicycles are made available for a shared use to individuals on a very short-term basis. A bike-sharing system is comprised of a network of kiosks throughout a city which allows a participant to check-out a bike at one location and return it to a different location. Participants of a bike-sharing system can rent bikes on an as-needed basis and are charged for the duration of rental. Most programs require participants to register as users prior to usage. As of December 2016, roughly 1000 cities worldwide have bike-sharing systems.
 
# Bike-sharing kiosks act as sensor networks for recording customer demand and usage patterns. For each bike rental, data is recorded for departure location, arrival location, duration of travel, and time elapsed. This data has valuable potential to researchers for studying mobility within a city. For this project, we explore customer mobility in relationship to these factors:
 
# 1.     Time of day
# 2.     Day type (workday, weekend, holiday, etc.)
# 3.     Season (Spring, Summer, Fall, Winter)
# 4.     Weather (clear, cloudy, rain, fog, snowfall, etc.)
# 5.     Temperature (actual, “feels like”)
# 6.     Humidity
# 7.     Windspeed
 
# This project explores changes in demand given changes in weather and day. Our project delivers an exploratory data analysis as well as a machine-learning model to forecast bike rental demand. Bike rental demand is measured by total rental count which is further broken down into two rental types: rentals by registered users and rentals by non-registered users.  
 
# ***Does it also predict casual and registered?***
# ***Should we include information about the RMSLE?***




##############################################
##############################################
##############################################
# DATA UNDERSTANDING

# What are the raw data sources?
# The data sources for this project are provided by kaggle. A train and test set and a example solution submission. https://www.kaggle.com/c/bike-sharing-demand

# What does each 'unit' (e.g. row) of data represent?

# What are the fields (columns)?
# Feature	Description
# datetime	hourlydate + timestamp
# season	1 = spring, 2 = summer, 3 = fall, 4 = winter
# holiday	whether the day is considered a holiday
# workingday	whether the day is neither a weekend nor holiday
# weather	1: Clear, Few clouds, Partly cloudy, Partly cloudy 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 4: A Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# temp	temperature in Celsius
# atemp	"feels like" temperature in Celsius
# humidity	relative humidity
# windspeed	wind speed
# casual	number of non-registered user rentals initiated
# registered	number of registered user rentals initiated
# count	number of total rentals

# EDA
	# Distribution of each feature
	# Missing values
	# Distribution of target
	# Relationships between features
	# Other idiosyncracies?

##############################################
# IMPORT THE REQUIRED MODULES
%matplotlib inline
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from time import time
import logging
from sklearn.model_selection import train_test_split
# SK-learn libraries for learning.
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#SK-learn libraries for transformation and pre-processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Custom classes for this assignment
import feature_engineering as fe


##############################################
# LOAD THE DATASETS
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

targets = ['count', 'casual', 'registered']
predictors = [c for c in train_df.columns if c not in targets]

y_count = train_df[['count']]
y_casual = train_df[['casual']]
y_registered = train_df[['registered']]

X_train, X_dev,y_count_train, y_count_dev,\
y_casual_train, y_casual_dev,y_registered_train, y_registered_dev =\
train_test_split(train_df, y_count, y_casual, y_registered, random_state=2)


##############################################
##############################################
##############################################
# DATA PREPARATION

# What steps are taken to prepare the data for modeling?
# feature transformations? engineering?
# table joins? aggregation?
# Precise description of modeling base tables.
# What are the rows/columns of X (the predictors)?
# What is y (the target)?




##############################################
##############################################
##############################################
# MODELING
# What model are we using? Why?
# Assumptions?
# Regularization?

##############################################
# Define pipeline

categorical = ('season', 'holiday', 'workingday', )
# datetime isn't numerical, but needs to be in the numeric branch
numerical = ('datetime', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',)
pipeline = Pipeline([
    # process cat & num separately, then join back together
    ('union', FeatureUnion([ 
        ('categorical', Pipeline([
            ('select_cat', fe.SelectCols(cols = categorical)),
            ('onehot', OneHotEncoder()),    
        ])),    
        ('numerical', Pipeline([
            ('select_num', fe.SelectCols(cols = numerical)),
            ('date', fe.DateFormatter()),
            ('drop_datetime', fe.SelectCols(cols = ('datetime'), invert = True)),
            ('temp', fe.ProcessNumerical(cols_to_square = ('temp', 'atemp', 'humidity'))),
            # ('bad_weather', fe.BinarySplitter(col = 'weather', threshold = 2)),
            # ('filter', fe.PassFilter(col='atemp', lb = 15, replacement_style = 'mean'))
            ('scale', StandardScaler()),    
        ])),    
    ])),
    ('clf', RandomForestRegressor(n_estimators = 100)),
])

def gs(y_train):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(train_df[predictors].copy(), y_train)
    print("done in %0.3fs" % (time() - t0))
    print()


    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    return grid_search

preds_count = gs(y_count).predict(test_df)
preds_casual = gs(y_casual).predict(test_df)
preds_registered = gs(y_registered).predict(test_df)

test_df.set_index(pd.DatetimeIndex(test_df['datetime']), inplace=True)
test_df['count'] = preds_count
test_df[['count']].to_csv('data/zi_count_preds.csv')

test_df['count'] = preds_casual + preds_registered
test_df[['count']].to_csv('data/zi_combined_preds.csv')

##############################################
##############################################
##############################################
# EVALUATION
# How well does the model perform?
# Accuracy
# ROC curves
# Cross-validation
# other metrics? performance?
# AB test results (if any)




##############################################
##############################################
##############################################
# DEPLOYMENT
# How is the model deployed?
# prediction service?
# serialized model?
# regression coefficients?
# What support is provided after initial deployment?



