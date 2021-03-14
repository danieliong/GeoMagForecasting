#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
from importlib import import_module


# def get_features_list(path="parameters/features.txt"):

#     with open(path, "r") as features_file:
#         features = features_file.read().splitlines()

#     return features


def load_solar_wind(path, features=None, time_col="time", **kwargs):
    """ Load solar wind data

    Parameters
    ----------
    path: str
        Path to solar wind data
    features: list-like or None, default=None
        List of features to load. If None, load all features in csv
    time_col: str
        Name of time column
    kwargs: Keyword arguments for pd.read_csv

    Returns
    -------
    pd.DataFrame
        Dataframe containing solar wind data

    """

    # TODO: Add years arg to be consistent with load_supermag

    if features is not None:
        usecols = [time_col] + features
    else:
        usecols = None

    data = pd.read_csv(path,
                       index_col=time_col,
                       parse_dates=True,
                       dtype=np.float32,
                       usecols=usecols,
                       **kwargs)

    return data


def load_supermag(station, years, data_dir="data/raw", **read_kwargs):
    """Load SuperMag data


    Parameters
    ----------
    station : str
        Magnetometer station to load data from
    years : list of int
        List of years to load data from
    data_dir : str, default="data/raw"
        Path to SuperMag data directory


    Returns
    -------
    DataFrame
        SuperMag data from specified station and years

    """

    # Create mapper for data
    def _read_csv_oneyear(year):
        return pd.read_csv(f"{data_dir}/{year}/{station}.csv.gz",
                           **read_kwargs)

    data_mapper = map(_read_csv_oneyear, years)

    # Concatenate data from specified years
    data = pd.concat(data_mapper)

    # Set time index
    data.index = pd.DatetimeIndex(data['Date_UTC'])
    data.drop(columns=['Date_UTC'], inplace=True)

    return data


class Resampler(TransformerMixin):
    def __init__(self, freq="T", label="right", func="max", **kwargs):
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
        self.kwargs = kwargs

    def fit(X, y=None):
        # For compatibility only
        pass

    def transform(X, y=None):
        if self.func == "max":
            X = X.resample(self.freq, label=self.label, **self.kwargs).max()
        # TODO: Add other functions

        return X


class Interpolator(TransformerMixin):
    def __init__(self,
                 method="time",
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

    def fit(X, y=None):
        # For compatibility only
        pass

    def transform(X, y=None):

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
