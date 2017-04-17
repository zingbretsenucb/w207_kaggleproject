#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class PipelineEstimator(BaseEstimator, TransformerMixin):
    """Define the necessary methods"""

    def __init__(self):
        pass


    def fit(self, X, y = None):
        return self


    def transform(self, X, y = None):
        return X



class Duplicate(PipelineEstimator):
    def __init__(self, cols, suffix = "_2"):
        self.cols = cols
        self.suffix = suffix
        
    def transform(self, X, y = None):
        for col in self.cols:
            X[col + self.suffix] = X[col]
        return X

class Square(PipelineEstimator):
    def transform(self, X):
        return np.square(X)
    


class ProcessNumerical(PipelineEstimator):

    def __init__(self, cols_to_square = (), cols_to_log = ()):
        self.cols_to_square = cols_to_square
        self.cols_to_log = cols_to_log


    def square(self, X):
        for col in self.cols_to_square:
            X[col + '_sq'] = np.square(X[col])
        return X


    def log(self, X):
        for col in self.cols_to_log:
            X[col + '_log'] = np.log(X[col])
        return X


    def transform(self, X, y = None):
        """Square or log given numerical columns"""
        X = self.square(X)
        X = self.log(X)

        return X


class RollingWindow(PipelineEstimator):
    def __init__(self, cols = (), window = 6, style = 'mean', shift = True):
        self.cols = cols
        self.window = window
        self.style = style
        self.shift = shift
        self.shift_val = 1 if window > 0 else -1
        
    def transform(self, X, y = None):
        if self.window == 0:
            return X
        for col in self.cols:
            if self.shift:
                rolling = X[col].shift(self.shift_val)
            else:
                rolling = X[col]

            rolling = rolling.rolling(self.window)

            if self.style == 'mean':
                rolling = rolling.mean()
            elif self.style == 'sum':
                rolling = rolling.sum()

            X["{}_roll_{}".format(col, self.window)] = rolling.fillna(1)
        return X


class DailyGroup(PipelineEstimator):
    def __init__(self, func = np.mean, cols = None, rsuffix = '_dailyavg'):
        self.func = func
        self.cols = cols
        self.rsuffix = rsuffix


    def group(self, X):
        return X.groupby('date').apply(self.func)


    def transform(self, X, y = None):
        if self.cols is None:
            return X

        X['date'] = X.index.date
        grouped = self.group(X[['date'] + self.cols])
        X = X.join(grouped, on = 'date', how = 'left', rsuffix = self.rsuffix)
        try: 
            X = X.drop('date', axis = 1)
            X = X.drop('date' + self.rsuffix, axis = 1)
        finally:
            return X


class WeatherForecast(PipelineEstimator):
    def __init__(self, use = True):
        self.use = use
        
        
    def transform(self, X, y = None):
        if not self.use:
            return X

        # X = X.copy()
        X['weather_was_better'] = X['weather'].shift() - X['weather']
        X['weather_getting_better'] = X['weather'].shift(-1) - X['weather']
        X[['weather_was_better', 'weather_getting_better']] = X[['weather_was_better', 'weather_getting_better']].fillna(0)
        return X


class DateFormatter(PipelineEstimator):
    """Parse datetime into its component parts"""

    def __init__(self):
        self.earliest_date = None


    def fit(self, X, y = None):
        self.earliest_date = np.min(X.index.date)
        return self
        


    def transform(self, X, y = None):
        """Split the datetime into its component parts."""

        X['hour'] = X.index.hour
        #X['weekday name'] = X.index.weekday_name
        X['weekday'] = X.index.weekday
        X['weekofyear'] = X.index.weekofyear
        X['month'] = X.index.month
        X['quarter'] = X.index.quarter
        X['year'] = X.index.year
        X['dom'] = X.index.day
        X['weekend'] = np.where(X['weekday']>5, 1, 0)

        # X['days_since_start'] = X.index.date - self.earliest_date
        # X['days_since_start'] = X['days_since_start'].apply(lambda x: x.days)

        return X

class SelectCols(PipelineEstimator):
    """Return only specified columnss"""

    def __init__(self, cols = (), invert = False):
        self.cols = cols
        self.invert = invert


    def transform(self, X, y = None):
        mask = np.array([True if col in self.cols else False for col in X.columns])
        if self.invert:
            mask = np.invert(mask)
        return X.loc[:, mask]


class BinSeparator(PipelineEstimator):
    """Separate specified column into bins and return a new column whose values are these bins"""
    
    def __init__(self, col = (), bin_edges = (), bin_labels = ()):
        self.col = col
        self.bin_edges = bin_edges
        self.bin_labels = bin_labels
        
    def transform(self, X, y = None):
        X[self.col] = X[self.col].astype('int64')
        X[self.col + '_bin'] = pd.cut(X[self.col], bins = self.bin_edges, labels = self.bin_labels)
        return X

class BinarySplitter(PipelineEstimator):
    """Binarize a feature and add that as a new feature"""

    def __init__(self, col, threshold, new_name = None):
        """Split col based on threshold"""
        self.col = col
        self.threshold = threshold
        self.new_name = new_name or col + "_bin"


    def transform(self, X, y = None):
        X[self.new_name] = X[self.col] >= self.threshold
        return X


class PassFilter(PipelineEstimator):
    """Low, hi, or bandpass filter

    If you supply:
    a lowerbound (lb): you will get a high pass filter.
    an upperbound (ub): you will get a low pass filter.
    both lb and ub: you will get a band pass filter.
    
    You may specify the values you want to replace values that do not
    pass. If neither the replacements nor the replacement style are
    supplied, then the low values will be replaced with the lb, and
    high values with the ub.

    If you specify the replacement style to be:
    'med': Values will be replaced by the median of the values that do
    not pass on that side
    'mean': Ditto, but with the mean
    'minmax': The minimum value out of the low values, and the max of
    the high values
    """

    def __init__(self, col, lb = None, ub = None, new_name = None,
                 lb_replacement = None, ub_replacement = None,
                 replacement_style = None):
        self.col = col
        self.lb  = lb
        self.ub  = ub
        self.new_name = new_name or col + "_filt"
        self.lb_replacement = lb_replacement
        self.ub_replacement = ub_replacement
        self.replacement_style = replacement_style


    def fit(self, X, y = None):
        """Determine replacement values if none given"""
        if self.lb and self.lb_replacement is None:

            if self.replacement_style == 'min_max':
                self.lb_replacement = np.min(X[self.col])

            elif self.replacement_style == 'med':
                med = np.nanmedian(
                    np.where(X[self.col] < self.lb, X[self.col], np.nan))
                self.lb_replacement = med

            elif self.replacement_style == 'mean':
                mean = np.nanmean(
                    np.where(X[self.col] < self.lb, X[self.col], np.nan))
                self.lb_replacement = mean

            else:
                self.lb_replacement = self.lb


        if self.ub and self.ub_replacement is None:

            if self.replacement_style == 'min_max':
                self.ub_replacement = np.max(X[self.col])

            elif self.replacement_style == 'med':
                med = np.nanmedian(
                    np.where(X[self.col] > self.ub, X[self.col], np.nan))
                self.ub_replacement = med

            elif self.replacement_style == 'mean':
                mean = np.nanmean(
                    np.where(X[self.col] > self.ub, X[self.col], np.nan))
                self.ub_replacement = mean

            else:
                self.ub_replacement = self.ub

        return self
        
    def transform(self, X, y = None):
        X[self.new_name] = X[self.col]

        if self.lb:
            X[self.new_name] = np.where(
                X[self.new_name] < self.lb, self.lb_replacement, X[self.new_name]
            )

        if self.ub:
            X[self.new_name] = np.where(
                X[self.new_name] > self.ub, self.ub_replacement, X[self.new_name]
            )

        return X
        
        
        
