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
    _has_storm_index,
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
        transformer_y=None,
        njobs=1,
        verbose=False,
        **transformer_y_kwargs,
    ):
        self.lag = lag
        self.exog_lag = exog_lag
        self.lead = lead
        self.njobs = njobs
        self.verbose = verbose

        # NOTE: transformer_y must keep input as pd DataFrame
        # NOTE: Pass transformer that was used for X here.
        # (use PandasTransformer if required)
        self.transformer_y = transformer_y
        self.transformer_y.set_params(**transformer_y_kwargs)

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

        # Get start and end times
        end = target_time - self.lead
        start = (end - self.lag)
        start_exog = (end - self.exog_lag)

        # Ravel target and solar wind between start and end time
        lagged = np.ravel(y[start:end].to_numpy())
        exog = np.ravel(X[start_exog:end].to_numpy())
        feature = np.concatenate((lagged, exog))

        return feature


    def fit(self, X, y):
        self.lag = to_offset(self.lag)
        self.exog_lag = to_offset(self.exog_lag)
        self.lead = to_offset(self.lead)

        # TODO: Replace with function from utils
        self.freq_X_ = get_freq(X)
        self.freq_y_ = get_freq(y)

        n_lag = int(self.lag / self.freq_y_)
        n_exog = int((self.exog_lag / self.freq_X_) * X.shape[1])
        self.n_cols_ = n_lag + n_exog

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

        # self.transformer_y.fit(y)

        return self

    def _get_target(self, X, y):

        y = y.dropna()

        max_time = max(to_offset(self.lag) + y.index[0],
                       to_offset(self.exog_lag) + X.index[0])
        cutoff = max_time + self.lead

        return y[y.index > cutoff]

    def transform(self, X, y):
        # NOTE: Include interpolator in transformer_y if want to interpolate
        # TODO: Write tests
        # TODO: Take out missing values in target times

        y_feature = self.transformer_y.fit_transform(y)
        y_target = self._get_target(X, y)
        n_obs = y_target.shape[0]

        compute_feature_ = partial(self._compute_feature, X=X, y=y_feature)
        features_map = map(compute_feature_, y_target.index)

        # TODO: Implement parallel

        features_iter = itertools.chain.from_iterable(features_map)
        features = np.fromiter(features_iter,
                               dtype=np.float32,
                               count=n_obs * self.n_cols_).reshape(n_obs, self.n_cols_)

        return features, y_target
