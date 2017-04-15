# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import os
sys.path.insert(0, '/kaggle/')

os.chdir('/Users/nwchen24/Desktop/UC_Berkeley/machine_learning/final_project_github_repo/w207_kaggleproject/kaggle/')

import feature_engineering_NC as fe
import pandas as pd


#create a copy of the train data
train_df_test = train_df.copy()

#create date formatter object
df = fe.DateFormatter()

#get hour variable
train_df_test = df.transform(train_df_test)

#create list of bin borders
bin_borders = [-.5, 6.5, 7.5, 8.5, 16.5, 18.5, 24]
bin_labels = ['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6']

#convert hour to numeric
train_df_test['hour'] = train_df_test['hour'].astype('int64')

#instantiate new column based on hour variable
train_df_test['hour_bin'] = pd.cut(train_df_test['hour'], bins = bin_borders, labels = bin_labels)


os.getcwd()



