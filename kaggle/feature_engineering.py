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


class DateFormatter(PipelineEstimator):
    """Parse datetime into its component parts"""

    def __init__(self):
        pass


    def transform(self, X, y = None):
        """Split the datetime into its component parts."""
        # Change the index to datetime
        X.set_index(pd.DatetimeIndex(X['datetime']), inplace=True)
        # Create new day and time related features
        X['date'] = pd.DatetimeIndex(X['datetime']).strftime("%Y%m%d")
        X['day'] = pd.DatetimeIndex(X['datetime']).strftime("%j")
        X['month1'] = pd.DatetimeIndex(X['datetime']).strftime("%m")
        # X['month2'] = pd.DatetimeIndex(X['datetime']).strftime("%B")
        X['year'] = pd.DatetimeIndex(X['datetime']).strftime("%Y")
        X['hour'] = pd.DatetimeIndex(X['datetime']).strftime("%H")
        X['dow1'] = pd.DatetimeIndex(X['datetime']).strftime("%w")
        # X['dow2'] = pd.DatetimeIndex(X['datetime']).strftime("%A")
        X['woy'] = pd.DatetimeIndex(X['datetime']).strftime("%W")
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
        
        
        
