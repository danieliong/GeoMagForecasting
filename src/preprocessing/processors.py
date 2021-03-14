import itertools
import warnings
import pandas as pd
import numpy as np

from loguru import logger
from pandas.tseries.frequencies import to_offset
from importlib import import_module
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline


class Resampler(TransformerMixin):
    def __init__(self, freq="T", label="right", func="max", verbose=True,
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

        self.freq = to_offset(self.freq)
        self.X_freq_ = self._get_freq(X)

        return self


    def transform(self, X, y=None):

        X_freq = self._get_freq(X)
        assert X_freq == self.X_freq_, "X does not have the correct frequency."

        if self.freq > X_freq or X_freq is None:
            X = X.resample(self.freq, label=self.label, **self.kwargs).apply(self.func)
        else:
            if self.verbose:
                logger.debug(
                    f"Specified frequency ({self.freq}) is <= data "
                    + f"frequency ({X_freq}). Resampling is ignored.")

        return X


class Interpolator(TransformerMixin):
    def __init__(self,
                 method="linear",
                 axis=1,
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


class PandasTransformer(TransformerMixin):
    def __init__(self,
                 transformer: str = 'StandardScaler',
                 module: str = 'sklearn.preprocessing',
                 **transformer_params):
        """Scikit-learn transformer for transforming Pandas DataFrames


        Parameters
        ----------
        transformer : str, default="StandardScaler"
            Name of scikit-learn transformer class
        module : str, default="sklearn.preprocessing"
            Name of scikit-learn module that transformer belongs to
        transformer_params : Keyword arguments for transformer

        Note: Arguments are strings instead of functions so we can easily pickle
              them.

        """

        # NOTE: Could've just used dill. Maybe think about reverting it to take
        # in actual transformer objects?

        self.transformer = transformer
        self.module = module
        self.transformer_params = transformer_params

    def get_transformer_(self, vars=None):
        # Get transformer object

        module_ = import_module(self.module)
        transf_call = getattr(module_, self.transformer)
        transformer_ = transf_call(**self.transformer_params)

        if vars is not None:
            for param, value in vars.items():
                setattr(transformer_, param, value)

        return transformer_

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns

        # Get transformer object
        transformer_ = self.get_transformer_(vars=None)
        transformer_.fit(X)

        self.transformer_vars_ = vars(transformer_)

        # Add transformer attributes to self
        for param, value in self.transformer_vars_.items():
            setattr(self, param, value)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)

        transformer_ = self.get_transformer_(vars=self.transformer_vars_)

        if isinstance(X, pd.Series):
            X_ = transformer_.transform(X.to_numpy().reshape(-1, 1))
            X_pd = pd.Series(X_, index=X.index)
        elif isinstance(X, pd.DataFrame):
            X_ = transformer_.transform(X)
            X_pd = pd.DataFrame(X_, columns=self.columns_, index=X.index)

        return X_pd


class LaggedFeaturesProcessor(BaseEstimator):
    """
    (Not exactly a sklearn transformer)
    # NOTE: X, y don't necessarily have the same freq so we can't just pass one
    combined dataframe.
    """
    def __init__(self,
                 lag="0T",
                 exog_lag="H",
                 lead="0T",
                 transformer_y=None,
                 njobs=1,
                 verbose=False):
        self.lag = lag
        self.exog_lag = exog_lag
        self.lead = lead
        self.njobs = njobs
        self.verbose = verbose

        # NOTE: Must keep input as pd DataFrame
        # (use PandasTransformer if required)
        self.transformer_y = transformer_y

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
        end_time = target_time - self.lead
        start = (end_time - self.lag)
        start_exog = (end_time - self.exog_lag)
        end = end_time - to_offset('S')

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

        self.pipeline_y_ = Pipeline(
            [
                ("resampler", Resampler(freq=self.y_freq_))
                ("interpolator", Interpolator()),

            ]
        )

        # Transform y
        # Fit transformer_y
        self.transformer_y.fit(y)

        return self


    def transform(self, X, y):
        # TODO: Check if freqs is same as in fit

        # NOTE: Include interpolator in transformer_y if want to interpolate


        y = self.transformer_y.transform(y)

        max_lag = max(to_offset(self.lag), to_offset(self.exog_lag))
        cutoff = X.index[0] + max_lag + self.lead
        target_times = y.index[y.index > cutoff]
        n_obs = len(target_times)

        compute_feature_ = partial(self._compute_feature, X=X, y=y)
        features_map = map(compute_feature_, target_times)

        # TODO: Implement parallel

        features_iter = itertools.chain.from_iterable(features_map)
        features = np.fromiter(features_iter,
                               dtype=np.float32,
                               count=n_obs * self.n_cols_).reshape(n_obs, self.n_cols_)

        return features



def create_pipeline(
        interpolate=True,
        resample_func="mean",
        resample_freq="T",
        log=False,
        scaler=None,
        module="sklearn.preprocessing",
        **kwargs
):

    # TODO: Allow user to specify a function that gets put either in front or
    # last in pipeline

    pipeline_list = []

    # NOTE: Resampler does nothing when freq < data freq
    # and fills gaps with NA when freq = data freq
    pipeline_list.append(
        ("resampler", Resampler(freq=resample_freq, func=resample_func)))

    # TODO: Add log option

    # if log:
    #     log_transformer = PandasTransformer(transformer="FunctionTransformer",
    #                                         func=)

    if interpolate:
        pipeline_list.append(("interpolator", Interpolator()))

    if scaler is not None:
        pipeline_list.append(
            ("transformer", PandasTransformer(transformer=scaler,
                                              module=module)))

    pipeline = Pipeline(pipeline_list)

    return pipeline
