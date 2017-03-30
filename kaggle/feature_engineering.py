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
        X = X.drop('datetime', axis = 1)
        return X



class SelectCols(PipelineEstimator):
    """Return only specified columnss"""

    def __init__(self, cols = ()):
        self.cols = cols


    def transform(self, X, y = None):
        mask = np.array([True if col in self.cols else False for col in X.columns])
        return X.loc[:, mask]
