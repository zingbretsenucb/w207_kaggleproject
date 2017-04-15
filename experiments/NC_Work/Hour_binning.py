# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
#Change system directory to import feature engineering
os.chdir('/Users/nwchen24/Desktop/UC_Berkeley/machine_learning/final_project_github_repo/w207_kaggleproject/')

from kaggle import feature_engineering_NC as fe
import pandas as pd

#reload fe to get concurrent changes
reload(fe)

# LOAD THEDATASETS
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')


#create a copy of the train data
train_df_test = train_df.copy()

#create date formatter object
dateformatter = fe.DateFormatter()

#get hour variable
#note, in feature_engineering_NC modified hour to be numeric
train_df_test = dateformatter.transform(train_df_test)

#*****************************
#This section of code works to manually get hour bins 
#create list of bin borders
hour_bin_edges = [-.5, 6.5, 7.5, 8.5, 16.5, 18.5, 24]
hour_bin_labels = ['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6']

#instantiate new column based on hour variable
train_df_test['hour_bin'] = pd.cut(train_df_test['hour'], bins = hour_bin_edges, labels = hour_bin_labels)



#*****************************
#Test BinSeparator class of feature_engineering

#instantiate the binner
hour_binner = fe.BinSeparator(col = 'hour', bin_edges = hour_bin_edges, bin_labels = hour_bin_labels)

train_df_test = hour_binner.transform(train_df_test)



