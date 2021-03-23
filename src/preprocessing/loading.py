#!/usr/bin/env python

import os
import pandas as pd
import numpy as np

from omegaconf import OmegaConf


def load_solar_wind(path,
                    start="2010-01-01",
                    end="2019-12-31",
                    features=None,
                    time_col="times",
                    working_dir=None,
                    **kwargs):
    """ Load solar wind data

    Parameters
    ----------
    path: str
        Path to solar wind data
    features: list-like or None, default=None
        List of features to load. If None, load all features in csv
    time_col: str
        Name of time column
    working_dir : str or None
        Absolute path to working directory that path is relative to.
        (For use with Hydra because Hydra changes cwd to output dir.)
    kwargs: Keyword arguments for pd.read_csv

    Returns
    -------
    pd.DataFrame
        Dataframe containing solar wind data

    """

    # QUESTION: What is this for?
    if working_dir is not None:
        path = os.path.join(working_dir, path)

    # TODO: Add years arg to be consistent with load_supermag

    if features is not None:
        if OmegaConf.is_config(features):
            features = OmegaConf.to_container(features)

        usecols = [time_col] + features
    else:
        usecols = None

    data = pd.read_csv(path,
                       index_col=time_col,
                       parse_dates=True,
                       dtype=np.float32,
                       usecols=usecols,
                       **kwargs)

    # Subset times [start, end)
    # NOTE: Not sure if this works if target and features don't have same freq
    data = data.truncate(before=start, after=end)[:-1]

    return data


def load_supermag(station,
                  start="2010-01-01",
                  end="2019-12-31",
                  horizontal=True,
                  data_dir="data/raw",
                  working_dir=None,
                  **read_kwargs):
    """Load SuperMag data


    Parameters
    ----------
    station : str
        Magnetometer station to load data from
    years : list of int
        List of years to load data from
    data_dir : str, default="data/raw"
        Path to SuperMag data directory
    working_dir : str or None
        Absolute path to working directory that path is relative to.
        (For use with Hydra because Hydra changes cwd to output dir.)

    Returns
    -------
    DataFrame
        SuperMag data from specified station and years

    """

    if working_dir is not None:
        data_dir = os.path.join(working_dir, data_dir)

    daterange = pd.date_range(start=start, end=end, freq="Y")
    years = list(daterange.year)

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

    data = data.loc[start:end]

    if horizontal:
        data.eval("db_h = ((dbn_nez**2)+(dbe_nez**2))**(1/2)", inplace=True)
        return data['db_h']
    else:
        return data


def load_symh(path,
              start="2010-01-01",
              end="2019-12-31",
              time_col="times",
              working_dir=None,
              **kwargs):

    if working_dir is not None:
        path = os.path.join(working_dir, path)

    data = pd.read_csv(path,
                       index_col=time_col,
                       parse_dates=True,
                       squeeze=True,
                       dtype=np.float32)
    data = data.truncate(before=start, after=end)[:-1]

    return data
