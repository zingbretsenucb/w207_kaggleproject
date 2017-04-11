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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing

#SK-learn libraries for transformation and pre-processing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer

# Custom classes for this assignment
import feature_engineering as fe
import paramTuning_GradientBoost as ptgb
##############################################
# LOAD THE DATASETS
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

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
# Base model before feature engineering


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
    ('to_dense', preprocessing.FunctionTransformer(lambda x: x.todense(), accept_sparse=True)), 
    ('clf', GradientBoostingRegressor(n_estimators=100,random_state=2)),
])

#Helper function to calculate root mean squared error
def get_RMSE(actual_values, predicted_values):
    n = len(actual_values)
    RMSE = np.sqrt(np.sum(((np.log(predicted_values + 1) - np.log(actual_values + 1)) ** 2) / n))
    return RMSE

#create custom scorer
RMSE_scorer = make_scorer(get_RMSE, greater_is_better = False)

##############################################
# Split into Dev and Train data and find best parameters
features = [c for c in train_df.columns if c not in ['count', 'casual', 'registered']]    
def train_dev_model_search(registered_or_casual,parameters):
    print("Performing grid search...")
    t0 = time()
    gs = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1, scoring=RMSE_scorer)
    gs.fit(train_data[features], train_data[registered_or_casual])
    print("Best parameters set:")
    best_param = gs.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_param[param_name]))
    predicted_y = gs.predict(dev_data[features])
    print "GridSearch RMSE " + str(gs.best_score_)
    rmse = get_RMSE(actual_values = dev_data[registered_or_casual], predicted_values = predicted_y)
    print "RMSE: ", str(rmse)
    print("Done in %0.3fs" % (time() - t0))
    print ""

# Split the data into train data and a dev data based on day of the month.
# This makes sense since the test data is days 19-30 of the month.
train_data = train_df[pd.DatetimeIndex(train_df['datetime']).day <= 16]
dev_data = train_df[pd.DatetimeIndex(train_df['datetime']).day > 16]
	
# Test for casual and registered separately
parameters = {
    'clf__n_estimators': (80,),
    'clf__learning_rate': (0.05,),
    'clf__max_depth': (10,),
    'clf__min_samples_leaf': (20,),
}

print "Casual rides"
train_dev_model_search('casual',parameters)

print "Registered rides"
train_dev_model_search('registered',parameters)

##############################################
# Further parameter tuning
n_estimators=[70,75,80,85,90,95,100,105,110,115,120,125,130,135,140]
learning_rate=[.01,.02,.03,.04,.045,.05,.055,.06,.07,.08,.09,.1,.2,.3,.4]
max_depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
min_samples_leaf=[1,5,10,15,20,25,30,35]

tune_nEstimators('casual',n_estimators)
tune_learningRate('casual',learning_rate)
tune_maxDepth('casual',max_depth)
tune_minSamplesLeaf('casual',min_samples_leaf)

tune_nEstimators('registered',n_estimators)
tune_learningRate('registered',learning_rate)
tune_maxDepth('registered',max_depth)
tune_minSamplesLeaf('registered',min_samples_leaf)

##############################################
# Create full model using all train data

casual_best_param = {
    'clf__n_estimators': (80,),
    'clf__learning_rate': (0.05,),
    'clf__max_depth': (10,),
    'clf__min_samples_leaf': (20,),
}

registered_best_param = {
    'clf__n_estimators': (80,),
    'clf__learning_rate': (0.05,),
    'clf__max_depth': (10,),
    'clf__min_samples_leaf': (20,),
}

full_casual_gs = GridSearchCV(pipeline, casual_best_param, n_jobs=1, verbose=1, scoring='neg_mean_squared_error')
full_casual_gs.fit(train_df[features], train_df['casual'])
full_casual_predicted_y = full_casual_gs.predict(test_df[features])

full_registered_gs = GridSearchCV(pipeline, registered_best_param, n_jobs=1, verbose=1, scoring='neg_mean_squared_error')
full_registered_gs.fit(train_df[features], train_df['registered'])
full_registered_predicted_y = full_registered_gs.predict(test_df[features])

##############################################
# Create CSV for submission

test_df.set_index(pd.DatetimeIndex(test_df['datetime']), inplace=True)
test_df['count'] = (np.array([full_casual_predicted_y>0]*full_casual_predicted_y)).T+\
    (np.array([full_registered_predicted_y>0]*full_registered_predicted_y)).T
test_df[['count']].to_csv('combined_preds.csv')

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

