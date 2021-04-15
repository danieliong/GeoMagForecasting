#!/usr/bin/env python

import itertools
import logging
import numpy as np
import pandas as pd

from functools import partial
from sklearn.base import TransformerMixin
from pandas.tseries.frequencies import to_offset

from src.utils import get_freq
from src.storm_utils import (
    iterate_storms_method,
    apply_storms,
    StormAccessor,
    StormIndexAccessor,
    has_storm_index,
)

logger = logging.getLogger(__name__)


class LaggedFeaturesProcessor:
    """
    NOTE: X, y don't necessarily have the same freq so we can't just pass one
    combined dataframe.

    XXX: Not a proper scikit learn transformer. fit and transform take X and y.

    """

    def __init__(
        self,
        lag="0T",
        exog_lag="H",
        lead="0T",
        unit="minutes",
        transformer_y=None,
        njobs=1,
        return_pandas=False,
        verbose=False,
        **transformer_y_kwargs,
    ):

        self.unit = unit
        self.lag = self._process_params(lag)
        self.exog_lag = self._process_params(exog_lag)
        self.lead = self._process_params(lead)

        self.njobs = njobs
        self.return_pandas = return_pandas
        self.verbose = verbose

        # NOTE: transformer_y must keep input as pd DataFrame
        # NOTE: Pass transformer that was used for X here.
        # (use PandasTransformer if required)
        self.transformer_y = transformer_y
        self.transformer_y_kwargs = transformer_y_kwargs

    def _process_params(self, param):

        assert isinstance(param, (str, int))

        if isinstance(param, str):
            param = to_offset(param)
        elif isinstance(param, int):
            param = to_offset(pd.Timedelta(**{self.unit: param}))

        return param

    def _check_data(self, X, y):
        # TODO: Input validation
        # - pd Dataframe
        pass

    # TODO: Use storm accessor wherever possible
    def _compute_feature(self, target_index, X, y):
        """Computes ARX features to predict target at a specified time
           `target_time`.

        This method ravels subsets of `target` and `self.solar_wind` that depend on
        `self.lag` and `self.exog_lag` to obtain features to predict the target
        time series at a specified `target_time`.

        Parameters
        ----------

        target_time : datetime-like
            Time to predict.
        X : pd.DataFrame
            Exogeneous features
        y : pd.DataFrame or pd.Series
            Target time series to use as features to predict at target_time

        Returns
        -------
        np.ndarray
            Array containing features to use to predict target at target_time.
        """

        # HACK: Assume time is the second element if target_index is MultiIndex tuple
        if isinstance(target_index, tuple):
            target_storm, target_time = target_index
        else:
            # TODO: Change to elif time index
            target_time = target_index

        # FIXME: When self.lead, self.lag = 0, self.n_cols_ is wrong

        # Get start and end times
        end = target_time - self.lead
        start = end - self.lag
        start_exog = end - self.exog_lag

        # HACK: Subset storm
        if has_storm_index(X):
            X = X.xs(target_storm, level="storm")
        if has_storm_index(y):
            y = y.xs(target_storm, level="storm")

        # Ravel target and solar wind between start and end time
        if start == end:
            lagged = np.array([])
        else:
            lagged = np.ravel(y[start:end][::-1].to_numpy())

        if start_exog == end:
            exog = np.array([])
        else:
            exog = np.ravel(X[start_exog:end][::-1].to_numpy())

        feature = np.concatenate((lagged, exog))

        error_msg = f"Length of feature ({len(feature)}) at {target_index} != self.n_cols_ ({self.n_cols_})"
        assert len(feature) == self.n_cols_, error_msg

        return feature

    def fit(self, X, y):

        # TODO: Replace with function from utils
        self.freq_X_ = get_freq(X)
        self.freq_y_ = get_freq(y)

        # Add ones to get number of features inclusive
        if self.lag == to_offset("0T"):
            n_lag = 0
        else:
            n_lag = int(self.lag / self.freq_y_) + 1
        logger.debug("# of lagged features: %s", n_lag)

        if self.exog_lag == to_offset("0T"):
            n_exog_each_col = 0
        else:
            n_exog_each_col = int((self.exog_lag / self.freq_X_)) + 1

        n_exog = n_exog_each_col * X.shape[1]
        logger.debug("# of exogeneous features: %s", n_exog)

        self.n_cols_ = n_lag + n_exog

        self.feature_names_ = self.get_feature_names(X, y)

        # NOTE: Don't need resampler. Target is already resampled in process_data
        # pipeline_list = [
        #     # ("resampler", Resampler(freq=self.freq_y_)),
        #     ("interpolator", Interpolator())
        # ]

        # if self.transformer_y is not None:
        #     # NOTE: Pass in clone of feature pipeline for transformer_y
        #     # transformer_y = _get_callable(self.transformer_y)
        #     pipeline_list.append(
        #         ("transformer", self.transformer_y))
        # self.pipeline_y_ = Pipeline(pipeline_list)

        if self.transformer_y is not None:
            self.transformer_y.set_params(**self.transformer_y_kwargs)
            self.transformer_y.fit(y)

        return self

    def get_feature_names(self, X, y=None):
        exog_feature_names = self._get_feature_names(
            self.exog_lag, X.columns, self.freq_X_
        )

        if y is None:
            return exog_feature_names

        lag_feature_names = self._get_feature_names(self.lag, ["y"], self.freq_y_)

        # Lagged y goes first
        return lag_feature_names + exog_feature_names

    @staticmethod
    def _get_feature_names(lag, columns, freq):
        lags_timedelta = pd.timedelta_range(start="0 days", end=lag, freq=freq)
        # Minutes in reverse order
        lags = (int(t.seconds / 60) for t in lags_timedelta)

        # Order: Iterate columns first
        # e.g. [density0, density5, ..., temperature0, temperature5, ....]
        feature_names = [f"{col}{t}" for t, col in itertools.product(lags, columns)]

        return feature_names

    @iterate_storms_method(drop_storms=True)
    def _get_target(self, X, y):
        # TODO: Handle MultiIndex case

        y = y.dropna()

        max_time = max(
            to_offset(self.lag) + y.index[0], to_offset(self.exog_lag) + X.index[0]
        )
        cutoff = max_time + self.lead

        return y[y.index > cutoff]

    @iterate_storms_method(["target_index"], concat="numpy", drop_storms=True)
    def _transform(self, X, y, target_index):
        # TODO: Implement parallel

        n_obs = len(target_index)

        compute_feature_ = partial(self._compute_feature, X=X, y=y)
        features_map = map(compute_feature_, target_index)

        features_iter = itertools.chain.from_iterable(features_map)
        features = np.fromiter(
            features_iter, dtype=np.float32, count=n_obs * self.n_cols_
        ).reshape(n_obs, self.n_cols_)

        return features

    def transform(self, X, y):
        # NOTE: Include interpolator in transformer_y if want to interpolate
        # TODO: Write tests

        if self.transformer_y is not None:
            y_feature = self.transformer_y.transform(y)
        else:
            y_feature = y

        logger.debug("Getting targets...")
        y_target = self._get_target(X, y)

        logger.debug("Computing lagged features...")
        features = self._transform(X, y_feature, target_index=y_target.index)

        assert features.shape[0] == y_target.shape[0]

        if self.return_pandas:
            features = pd.DataFrame(
                features, index=y_target.index, columns=self.feature_names_
            )

        return features, y_target

    def fit_transform(self, X, y, **fit_params):
        # fit_transform from TransformerMixin doesn't allow y in transform
        return self.fit(X, y, **fit_params).transform(X, y)
