#!/usr/bin/env python

import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils import is_pandas

logger = logging.getLogger(__name__)


class StormSubsetter(BaseEstimator, TransformerMixin):
    def __init__(self, times_path):
        self.storm_times = pd.read_csv(times_path, index_col=0)

    @property
    def storms(self):
        return self.storm_times.index

    @property
    def n_storms(self):
        return len(self.storms)

    def _storm_iter(self):
        return self.storm_times.iterrows()

    def _subset_data(self, X, row):
        # Subset X by one storm

        storm_num, storm = row

        if isinstance(X, pd.Series):
            X = X.to_frame()

        start, end = storm["start_time"], storm["end_time"]
        X_ = X[start:end]
        X_["storm"] = storm_num

        return X_

    def _storm_label(self, X, row):
        # There's probably a more memory efficient way to do this
        return self._subset_data(X, row)["storm"]

    def get_storm_labels(self, x):

        if self._storms_subsetted(x):
            return x.index.to_frame().set_index(["times"])
        else:
            storm_labels_iter = (
                self._storm_label(x, row) for row in self._storm_iter()
            )
            return pd.concat(storm_labels_iter)

    def _min_y_storms(self, y, storm_labels):
        # TODO: Get min y for each storm
        # QUESTION: Should I just assume y has already been subsetted for storms?

        def _get_min_y(y, storm_labels, i):
            idx = storm_labels == i
            return np.amin(y[idx], where=~np.isnan(y[idx]), initial=0)

        unique_storms = np.unique(storm_labels)

        min_y_iter = (_get_min_y(y, storm_labels, i) for i in unique_storms)
        min_y = np.fromiter(min_y_iter, dtype=np.float32, count=unique_storms.shape[0])

        return min_y

    def _threshold_storms(self, y, n_storms, threshold, threshold_less_than):

        min_y = self._min_y_storms(y)

        if threshold_less_than:
            idx = np.where(min_y < threshold)[0]
        else:
            idx = np.where(min_y > threshold)[0]

        test_idx = np.random.choice(idx, size=n_storms)

        return self.storms[test_idx]

    def _storms_subsetted(self, x):
        # Check if data has been subsetted for storms
        subsetted = False

        if isinstance(x.index, pd.MultiIndex):
            # Has MultiIndex

            if "storm" in x.index.names and "times" in x.index.names:
                # MultiIndex has level 'storm' and "times"
                subsetted = True

        return subsetted

    def train_test_split(
        self,
        X,
        y,
        test_size=0.2,
        test_storms=None,
        threshold=None,
        threshold_less_than=True,
    ):

        # If data doesn't have storm index yet, subset it using subset_data
        # if not self._storms_subsetted(X):
        #     X = self.subset_data(X)
        # if not self._storms_subsetted(y):
        #     y = self.subset_data(y)

        storm_labels_X = self.get_storm_labels(X)
        storm_labels_y = self.get_storm_labels(y)

        if test_storms is None:
            # If threshold, is given, choose storms that are < threshold
            # for testing. Else choose randomly

            assert isinstance(test_size, (float, int))

            if test_size >= 1:
                n_test = test_size
            elif test_size < 1:
                n_test = test_size * self.n_storms

            if threshold is None:
                test_storms = np.random.choice(self.storms, size=n_test)
            else:
                test_storms = self._threshold_storms(
                    y, n_test, threshold, threshold_less_than
                )

        # TODO: Get test storm labels for X and y

    def subset_data(self, X):

        X_storms_iter = (self._subset_data(X, row) for row in self._storm_iter())

        X_storms = pd.concat(X_storms_iter)
        X_storms.set_index([X_storms["storm"], X_storms.index], inplace=True)
        X_storms.drop(columns=["storm"], inplace=True)

        return X_storms


def get_min_y_storms(y, storm_labels=None):
    def _get_min_y(y, storm_labels, i):
        idx = storm_labels == i
        return np.amin(y[idx], where=~np.isnan(y[idx]), initial=0)

    unique_storms = np.unique(storm_labels)

    min_y_iter = (_get_min_y(y, storm_labels, i) for i in unique_storms)
    min_y = np.fromiter(min_y_iter, dtype=np.float32, count=unique_storms.shape[0])

    return min_y


def split_data_storms(
    X, y, stormtimes_path, test_size=0.2, test_storms=None, threshold=None
):

    subsetter = StormSubsetter(stormtimes_path)
    # X_storms = subsetter.subset_data(X)
    # y_storms = subsetter.subset_data(y)

    # TODO: Remove storms with too many NAs

    if test_storms is not None:
        pass
    elif threshold is not None:
        pass
        # TODO: Get test_storms
    else:
        # TODO: Get test_storms
        n_test = int(round(test_size * subsetter.n_storms))

        test_storms = subsetter.get_storm_labels(X)

    return test_storms


def split_data_ts(X, y, test_size=0.2):

    test_start_idx = round(y.shape[0] * (1 - test_size))
    test_start = y.index[test_start_idx]

    def _split(x):
        if is_pandas(x):
            x_train, x_test = x.loc[:test_start], x.loc[test_start:]

            # HACK: Remove overlaps if there are any
            overlap = x_train.index.intersection(x_test.index)
            x_test.drop(index=overlap, inplace=True)
        else:
            raise TypeError("x must be a pandas DataFrame or series.")

        # FIXME: Data is split before processed so it looks like there is time
        # overlap if times are resampled

        return x_train, x_test

    X_train, X_test = _split(X)
    y_train, y_test = _split(y)

    Train = namedtuple("Train", ["X", "y"])
    Test = namedtuple("Test", ["X", "y"])

    return Train(X_train, y_train), Test(X_test, y_test)
