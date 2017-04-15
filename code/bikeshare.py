#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Place to hide functions referenced in notebook ###


from time import time

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



# Define some time based variables
def eda_transform(df):
    df['hour'] = df.index.hour
    df['weekday name'] = df.index.weekday_name
    df['weekday'] = df.index.weekday
    df['weekofyear'] = df.index.weekofyear
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['year_month'] = df.index.month
    df['atempsq'] = df['atemp']**2
    df['tempsq'] = df['temp']**2
    df['year_month']=df['year'].astype(str) + df['month'].astype(str)
return df

def train_dev_model_search(registered_or_casual,parameters, pipeline, RMSE_scorer):
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


def get_RMSE(actual_values, predicted_values):
    n = len(actual_values)
    RMSE = np.sqrt(np.sum(((np.log(predicted_values + 1) - np.log(actual_values + 1)) ** 2) / n))
    return RMSE


#### I THINK WE CAN LEAVE THE BELOW IN THE NOTEBOOK
# RMSE_scorer = make_scorer(get_RMSE, greater_is_better = False)


