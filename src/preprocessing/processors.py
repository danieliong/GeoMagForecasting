import itertools
import pandas as pd
import numpy as np
import logging

# from loguru import logger
from functools import partial
from pandas.tseries.frequencies import to_offset
from importlib import import_module
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)

class Resampler(TransformerMixin):
    def __init__(self, freq="T", label="right", func="mean", verbose=True,
                 **kwargs):
        """Scikit-learn Wrapper for Pandas resample method


        Parameters
        ----------
        freq : DateOffset, Timedelta or str, default="T"
            The offset string or object representing target conversion
        label : {'right','left'}, default="right"
            Which bin edge label to label bucket with. The default is ‘left’ for
            all frequency offsets except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’,
            and ‘W’ which all have a default of ‘right’.
        func : function or str, default="max"
            Function to apply to time-aggregated data.
        kwargs : Keyword arguments for pd.DataFrame.resample

        """

        self.freq = freq
        self.label = label
        self.func = func
        self.verbose = verbose
        self.kwargs = kwargs

    @staticmethod
    def _get_freq(X):
        return to_offset(pd.infer_freq(X.index))

    def fit(self, X, y=None):

        if self.func is None:
            self.func = "mean"

        self.freq = to_offset(self.freq)
        self.X_freq_ = self._get_freq(X)

        return self


    def transform(self, X, y=None):

        X_freq = self._get_freq(X)
        assert X_freq == self.X_freq_, "X does not have the correct frequency."

        if X_freq is None:
            X = X.resample(self.freq, label=self.label, **self.kwargs).apply(self.func)
        elif self.freq > X_freq:
            X = X.resample(self.freq, label=self.label, **self.kwargs).apply(self.func)
        else:
            if self.verbose:
                logger.debug(
                    f"Specified frequency ({self.freq}) is <= data "
                    + f"frequency ({X_freq}). Resampling is ignored.")

        return X

    def inverse_transform(self, X):
        return X


class Interpolator(TransformerMixin):
    def __init__(self,
                 method="linear",
                 axis=0,
                 limit_direction="both",
                 limit=15,
                 **kwargs):
        """Scikit-learn wrapper for Pandas interpolate method
        """

        self.method = method
        self.axis = axis
        self.limit_direction = limit_direction
        self.limit = limit
        self.kwargs = kwargs

    def fit(self, X, y=None):
        # For compatibility only
        return self

    def transform(self, X, y=None):

        X = X.interpolate(method=self.method,
                          axis=self.axis,
                          limit_direction=self.limit_direction,
                          **self.kwargs)

        return X

    # NOTE: This is for inverse transforming the pipeline when computing metrics later.
    # The only thing that needs to be inversed is the scaler.
    # Find better way to do this?
    def inverse_transform(self, X):
        return X


class PandasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None, **transformer_params):
        self.transformer = transformer
        self.transformer_params = transformer_params


    def fit(self, X, y=None, **fit_params):

        self.type_ = type(X)

        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        elif isinstance(X, pd.Series):
            self.name_ = X.name
        else:
            logger.warning("X is not a pandas object.")

        self.transformer.fit(X, **fit_params)

        return self

    def transform(self, X):
        check_is_fitted(self)

        assert isinstance(X, self.type_)

        if isinstance(X, pd.Series):
            assert X.name == self.name_
            X_ = self.transformer.transform(X.to_numpy().reshape(-1, 1))
            X_pd = pd.Series(X_.flatten(), name=self.name_, index=X.index)
        elif isinstance(X, pd.DataFrame):
            assert X.columns == self.columns_
            X_ = self.transformer.transform(X)
            X_pd = pd.DataFrame(X_, columns=self.columns_, index=X.index)

        return X_pd

    def inverse_transform(self, X):
        check_is_fitted(self)
        assert isinstance(X, self.type_)

        if isinstance(X, pd.Series):
            assert X.name == self.name_
            X_ = self.transformer.inverse_transform(X.to_numpy().reshape(-1, 1))
            X_pd = pd.Series(X_.flatten(), name=self.name_, index=X.index)
        elif isinstance(X, pd.DataFrame):
            assert X.columns == self.columns_
            X_ = self.transformer.inverse_transform(X)
            X_pd = pd.DataFrame(X_, columns=self.columns_, index=X.index)

        return X_pd




# TODO: Move to fit models section
class LaggedFeaturesProcessor(BaseEstimator, TransformerMixin):
    """
    (Not exactly a sklearn transformer)
    NOTE: X, y don't necessarily have the same freq so we can't just pass one
    combined dataframe.

    HACK: Not a proper scikit learn transformer. fit and transform take X and y.

    """
    def __init__(self,
                 lag="0T",
                 exog_lag="H",
                 lead="0T",
                 transformer_y=None,
                 njobs=1,
                 verbose=False,
                 **transformer_y_kwargs):
        self.lag = lag
        self.exog_lag = exog_lag
        self.lead = lead
        self.njobs = njobs
        self.verbose = verbose

        # NOTE: transformer_y must keep input as pd DataFrame
        # (use PandasTransformer if required)
        self.transformer_y = transformer_y
        self.transformer_y_kwargs = transformer_y_kwargs

    def _compute_feature(self, target_time, X, y):
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

        # TODO: Input validation
        # - pd dataframe

        # TODO: Replace with function from utils
        self.freq_X_ = to_offset(pd.infer_freq(X.index))
        self.freq_y_ = to_offset(pd.infer_freq(y.index))

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

        self.transformer_y.transform(y)

        return self


    def transform(self, X, y):
        # TODO: Check if freqs is same as in fit
        # TODO: Write tests
        # NOTE: Include interpolator in transformer_y if want to interpolate

        y = self.transformer_y.transform(y)

        max_time = max(to_offset(self.lag) + y.index[0],
                       to_offset(self.exog_lag) + X.index[0])
        cutoff = max_time + self.lead
        target_times = y.index[y.index > cutoff]
        n_obs = len(target_times)

        compute_feature_ = partial(self._compute_feature, X=X, y=y)
        features_map = map(compute_feature_, target_times)

        # TODO: Implement parallel

        features_iter = itertools.chain.from_iterable(features_map)
        features = np.fromiter(features_iter,
                               dtype=np.float32,
                               count=n_obs * self.n_cols_).reshape(n_obs, self.n_cols_)

        return features, y.loc[target_times]


def _get_callable(obj_str):
    # TODO: Modify to allow scaler_str to be more general
    # TODO: Validation

    obj = eval(obj_str)

    return obj


def create_pipeline(
        interpolate=True,
        resample_func="mean",
        resample_freq="T",
        scaler=None,
        func=None,
        inverse_func=None,
        **kwargs
):
    # TODO: Allow user to specify a function that gets put either in front or
    # last in pipeline

    pipeline_list = []

    # NOTE: Resampler does nothing when freq < data freq
    # and fills gaps with NA when freq = data freq

    if resample_freq is not None:
        pipeline_list.append(
            ("resampler", Resampler(freq=resample_freq, func=resample_func)))

    if interpolate:
        pipeline_list.append(("interpolator", Interpolator()))

    # NOTE: scaler ignored when func is specified
    if func is not None:
        func = _get_callable(func)
        inverse_func = _get_callable(inverse_func)
        func_transformer = FunctionTransformer(
            func=func, inverse_func=inverse_func, **kwargs)
        pipeline_list.append(
            ("func", PandasTransformer(transformer=func_transformer)))
    elif scaler is not None:
        scaler_callable = _get_callable(scaler)
        pipeline_list.append(
            ("scaler", PandasTransformer(transformer=scaler_callable(), **kwargs)))
    else:
        logger.debug("scaler or func was not specified.")

    if len(pipeline_list) == 0:
        return None

    pipeline = Pipeline(pipeline_list)

    return pipeline
