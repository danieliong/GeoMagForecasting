#!/usr/bin/env python

import os
import pickle
import joblib
import dill
import sklearn
import pandas as pd
import numpy as np
import logging

from pathlib import Path

# from loguru import logger
from hydra.utils import to_absolute_path, get_original_cwd
from omegaconf import OmegaConf
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)


class NotSupportedError(Exception):
    def __init__(self, name, name_type=None, add_message=None):
        self.name = name
        self.name_type = name_type
        self.add_message = add_message
        self.message = f"{self.name_type} {self.name} is not currently supported. {self.add_message}"
        super().__init__(self.message)


def is_pandas(x):
    return isinstance(x, (pd.DataFrame, pd.Series))


def is_numpy(x):
    return isinstance(x, np.ndarray)


def get_freq_multi_idx(X, time_col=1):
    def _get_freq_one_storm(x):
        # Infer freq for one storm
        times = x.index.get_level_values(time_col)
        return to_offset(pd.infer_freq(times))

    # Infer frequency within each storm and get unique frequencies
    freqs = X.groupby(level=0).apply(_get_freq_one_storm).unique()

    # If there is only one unique frequency
    if len(freqs) == 1:
        return freqs[0]
    else:
        return None


def get_freq(X):
    if isinstance(X.index, pd.MultiIndex):
        freq = get_freq_multi_idx(X)
    else:
        freq = to_offset(pd.infer_freq(X.index))

    return freq


def has_storm_index(x):
    # Returns True if x has a pd MultiIndex with level named 'storm'

    if not isinstance(x, (pd.Series, pd.DataFrame)):
        return False

    if isinstance(x.index, pd.MultiIndex):
        if "storm" in x.index.names:
            return True

    return False


def _apply_storm(func, x_tuple, y_tuple=None, keep_storm_index=False, **kwargs):
    # Helper function for apply_storms
    # Apply func to one storm

    x_storm = x_tuple[0]
    if not keep_storm_index:
        x = x_tuple[1].droplevel("storm")
    else:
        x = x_tuple[1]

    if y_tuple is None:
        return func(x, **kwargs)
    else:
        y_storm = y_tuple[0]
        assert x_storm == y_storm
        if not keep_storm_index:
            y = y_tuple[1].droplevel("storm")
        else:
            y = y_tuple[1]

        return func(x, y, **kwargs)


def apply_storms(func, X, y=None, check=True, keep_storm_index=False, **kwargs):
    # Apply func to each storm and combine results

    # QUESTION: What if only one has MultiIndex?
    # NOTE: Assuming both or neither have MultiIndex for now

    if check:
        assert has_storm_index(X) or has_storm_index(
            y
        ), "Neither X or y have storm index"

    X_groupby = X.groupby(level="storm")
    if y is None:
        apply_iter = (
            _apply_storm(func, x_tuple, keep_storm_index=keep_storm_index, **kwargs)
            for x_tuple in X_groupby
        )
    else:
        y_groupby = y.groupby(level="storm")

        apply_iter = (
            _apply_storm(
                func, x_tuple, y_tuple, keep_storm_index=keep_storm_index, **kwargs
            )
            for x_tuple, y_tuple in zip(X_groupby, y_groupby)
        )

    storms = X.index.unique(level="storm")
    res = pd.concat(apply_iter, keys=storms)

    return res


def save_output(obj, path, symlink=False):
    """Save output object to path.

    Parameters
    ----------
    obj: pickle-able
        Any object that can be pickled.
    path: path-like
        Path to save object to
    symlink: bool, default=True
        Save object to path's filename and sym-link path to it.
        Used with Hydra because Hydra saves files to the outputs/ dir.

    """

    # TODO: symlink should be dir name to symlink from
    # TODO: Check that file extensions match object types
    # i.e. .npy -> numpy arrays, etc

    # Do nothing if obj is None
    # e.g. groups if split=timeseries
    if obj is None:
        return None

    orig_cwd = get_original_cwd()
    logger.debug(f"Original working directory is {orig_cwd}")

    # Example: /home/danieliong/geomag-forecasting/filename
    orig_path = Path(to_absolute_path(path))

    # Example: /home/danieliong/geomag-forecasting/outputs/{...}/filename
    output_path = Path(orig_path.name).resolve()

    # if symlink:
    #     save_path = output_path
    # else:
    #     save_path = orig_path

    if is_pandas(obj):
        obj.to_pickle(output_path)
    elif is_numpy(obj):
        np.save(output_path, obj)
    # elif isinstance(obj, sklearn.pipeline.Pipeline):
    #     with open(output_path, "wb") as f:
    #         try:
    #             joblib.dump(obj, f)
    #         except:
    #             dill.dump(obj, f)
    # try:
    #     joblib.dump(obj, f)
    # except (pickle.PickleError, pickle.PicklingError):
    #     dill.dump(obj, f)
    #     # FIXME: ModuleNotFound error when unpickling
    else:
        with open(output_path, "wb") as f:
            dill.dump(obj, f)
            # pickle.dump(obj, f)

    rel_orig_path = orig_path.relative_to(orig_cwd)
    rel_output_path = output_path.relative_to(orig_cwd)

    logger.debug(f"Saved output to {rel_output_path}.")

    if symlink:
        if orig_path.exists() or orig_path.is_symlink():
            logger.debug(f"Deleting {rel_orig_path} since it already exists...")
            logger.debug(f"{rel_orig_path} was a symlink to {os.readlink(orig_path)}.")

            orig_path.unlink()

        logger.info(f"Sym-linking {rel_output_path} to {rel_orig_path}...")
        orig_path.symlink_to(output_path)


# Might not be needed if using MLFlow
class Results:
    def __init__(self, model, time, root_dir="~/geomag-forecasting"):
        self.model = model
        self.time = time
        self.root_dir = root_dir

    def get_multirun_dir(self):

        time = pd.to_datetime(self.time)
        time_dir = time.strftime("%Y-%m-%d_%H-%M")
        dir_name = f"{self.root_dir}/multirun/{self.model}/{time_dir}"
        dir_path = Path(dir_name).expanduser()

        return dir_path

    def get_multirun_results(self):

        results_dir = self.get_multirun_dir()
        results = OmegaConf.load(f"{results_dir}/optimization_results.yaml")

        return results

    def get_best_dir(self):

        results_dir = self.get_multirun_dir()
        results = self.get_multirun_results()

        best_params = results.best_evaluated_params
        best_params_list = [f"{key}={val}" for key, val in best_params.items()]

        dir_iter = results_dir.iterdir()

        def _best_dir(dir_path):
            overrides_path = dir_path / ".hydra/overrides.yaml"

            best = False
            if overrides_path.exists():
                overrides = OmegaConf.load(dir_path / overrides_path)
                best = set(overrides) == set(best_params_list)

            return best

        best_dir = next(path for path in dir_iter if _best_dir(path))
        return best_dir

    def load_best_pred(self):

        best_dir = self.get_best_dir()
        ypred_path = best_dir / "ypred.pkl"

        ypred = pd.read_pickle(ypred_path)
        return ypred
