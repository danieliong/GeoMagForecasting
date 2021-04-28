#!/usr/bin/env python

import logging
import pandas as pd
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from src import utils

logger = logging.getLogger(__name__)

OMNI_FEATURES = [
    "times",
    "b",
    "bx",
    "by_gse",
    "bz_gse",
    "by",
    "bz",
    "v",
    "vx_gse",
    "vy_gse",
    "vz_gse",
    "density",
    "temperature",
    "pressure",
    "e",
    "beta",
    "alfven_mach",
    "x_gse",
    "y_gse",
    "z_gse",
    # QUESTION: What are bsn_*_gse?
    "bsn_x_gse",
    "bsn_y_gse",
    "bsn_z_gse",
    "mach",
]

ACE_FEATURES = [
    "density",
    "speed",
    "temperature",
    "bx",
    "by",
    "bz",
    "bt",
    "status_swepam",
    "status_mag",
]


def _load_solar_wind(
    start="2010-01-01",
    end="2019-12-31",
    path="data/omni_2010-2019.csv.gz",
    features=None,
    time_col="times",
    **kwargs,
):
    # ACE and OMNI are loaded the same way except with different paths and features

    path = to_absolute_path(path)

    if features is not None:
        if OmegaConf.is_config(features):
            features = OmegaConf.to_container(features)

        usecols = [time_col] + features
    else:
        usecols = None

    data = pd.read_csv(
        path, index_col=time_col, parse_dates=True, usecols=usecols, **kwargs,
    )

    # Subset times [start, end)
    # NOTE: Not sure if this works if target and features don't have same freq
    data = data.truncate(before=start, after=end)[:-1]

    return data


def load_features_omni(
    path="data/omni_2010-2019.csv.gz",
    features=OMNI_FEATURES,
    dtype=np.float32,
    **kwargs,
):
    return _load_solar_wind(path=path, features=features, dtype=dtype, **kwargs)


def load_features_ace(
    path="data/ace_2010-2019.csv",
    positions_path="data/ace_pos_2010-2019.csv",
    features=ACE_FEATURES,
    dtype=np.float32,
    time_col="times",
    x_coord_col="x",
    fillna_method="ffill",
    **kwargs,
):
    dtypes = {"status_swepam": np.uint8, "status_mag": np.uint8, "temperature": int}
    df = _load_solar_wind(path=path, features=features, dtype=dtype, **kwargs)

    if positions_path is not None:
        positions_path = to_absolute_path(positions_path)

        positions = pd.read_csv(
            positions_path,
            index_col=time_col,
            usecols=[time_col, x_coord_col],
            parse_dates=True,
        )
        df = df.merge(positions, how="left", on=time_col)
        df[x_coord_col].fillna(method=fillna_method, inplace=True)

        for col in dtypes.copy().keys():
            if col not in df.columns:
                _ = dtypes.pop(col)

    return df.astype(dtypes)


def load_positions_ace(path="data/ace_pos_2010-2019.csv", **kwargs):
    df = _load_solar_wind(path=path, **kwargs)
    return df


def load_target_supermag(
    start="2010-01-01",
    end="2019-12-31",
    station="OTT",
    horizontal=True,
    data_dir="data/raw",
    **read_kwargs,
):
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

    data_dir = to_absolute_path(data_dir)

    daterange = pd.date_range(start=start, end=end, freq="Y")
    years = list(daterange.year)

    # Create mapper for data
    def _read_csv_oneyear(year):
        return pd.read_csv(f"{data_dir}/{year}/{station}.csv.gz", **read_kwargs)

    data_mapper = map(_read_csv_oneyear, years)

    # Concatenate data from specified years
    data = pd.concat(data_mapper)

    # Set time index
    data.index = pd.DatetimeIndex(data["Date_UTC"])
    data.drop(columns=["Date_UTC"], inplace=True)

    data = data.loc[start:end]

    if horizontal:
        data.eval("db_h = ((dbn_nez**2)+(dbe_nez**2))**(1/2)", inplace=True)
        return data["db_h"]
    else:
        return data


def load_target_symh(
    start="2010-01-01",
    end="2019-12-31",
    path="data/symh.csv",
    time_col="times",
    **kwargs,
):

    path = to_absolute_path(path)
    # if working_dir is not None:
    #     path = os.path.join(working_dir, path)

    data = pd.read_csv(
        path, index_col=time_col, parse_dates=True, squeeze=True, dtype=np.float32
    )
    data = data.truncate(before=start, after=end)[:-1]

    return data


def load_processed_data(
    name, paths: dict, inputs_dir=None, absolute=True, must_exist=True
):

    if inputs_dir is None:
        inputs_dir = Path(".")
    else:
        inputs_dir = Path(inputs_dir)

    if absolute:
        rel_path = inputs_dir / paths[name]
        path = Path(to_absolute_path(rel_path))
    else:
        path = inputs_dir / paths[name]

    ext = path.suffix

    if not path.exists():
        if must_exist:
            raise ValueError(f"{path} does not exist.")
        # Mainly used for when cv=timeseries in fit_models
        return None

    if ext == ".npy":
        processed_data = np.load(path)
    elif ext == ".pkl":
        processed_data = pd.read_pickle(path)
    else:
        raise utils.NotSupportedError(ext, name_type="Extension")

    return processed_data


def load_processor(
    name_or_path, paths=None, inputs_dir=None, absolute=True, must_exist=True
):
    # paths is either dict or None.If None, then name should be a path

    if paths is None:
        path = name_or_path
    else:
        path = paths[name_or_path]

    if inputs_dir is None:
        inputs_dir = Path(".")
    else:
        inputs_dir = Path(inputs_dir)

    if absolute:
        rel_path = inputs_dir / path
        path = Path(to_absolute_path(rel_path))
    else:
        path = inputs_dir / path

    ext = path.suffix

    if not path.exists():
        if must_exist:
            raise ValueError(f"{path} does not exist.")
        return None

    with open(path, "rb") as f:
        if ext == ".pkl":
            import dill

            processor = dill.load(f)
        elif ext == ".joblib":
            import joblib

            processor = joblib.load(f)
        else:
            raise utils.NotSupportedError(ext, name_type="Extension")

    return processor
