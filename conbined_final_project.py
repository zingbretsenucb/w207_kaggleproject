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
from kaggle import feature_engineering as fe


##############################################
# LOAD THE DATASETS
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

targets = ['count', 'casual', 'registered']
predictors = [c for c in train_df.columns if c not in targets]




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



