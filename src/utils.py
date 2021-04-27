#!/usr/bin/env python

import os
import pickle
import joblib
import dill
import pandas as pd
import numpy as np
import logging

from pathlib import Path

# from loguru import logger
from hydra.experimental import compose
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


def parse_override(node_cfg, node_name=None):

    # Return empty list if node_cfg is empty
    if not bool(node_cfg):
        return []

    def _parse(key, val, node=None):
        if isinstance(val, dict):
            result = []
            for key2, val2 in val.items():
                if node is not None:
                    new_node = f"{node}."
                else:
                    new_node = ""

                new_node += key
                result.append(_parse(key2, val2, new_node))
        else:
            if node is not None:
                result = f"{node}."
            else:
                result = ""

            result += "=".join([str(key), str(val)])

        return result

    override_list = []
    for key, val in node_cfg.items():
        override = _parse(key, val, node=node_name)
        if isinstance(override, list):
            override_list.extend(np.hstack(override))
        else:
            override_list.append(override)

    return override_list


def parse_data_overrides(cfg, override_nodes=["features", "target", "split"]):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    overrides = []

    # If dictionary is not empty
    if bool(cfg["data"]):
        overrides.extend(parse_override(cfg["data"]))

    for node in override_nodes:

        # HACK: Name could either be called name or method (for split)
        name = cfg[node].pop("name", None)
        method = cfg[node].pop("method", None)
        name = method if name is None else name
        overrides.append("=".join([node, name]))

        overrides.extend(parse_override(cfg[node], node_name=node))

    return overrides


def parse_processed_data_overrides(cfg):

    cfg = OmegaConf.to_container(cfg, resolve=True)
    overrides = []

    if bool(cfg["data"]):
        overrides.extend(parse_override(cfg["data"], node_name="+data"))

    for node in ["target", "features", "split"]:
        name = cfg[node].pop("name", None)
        method = cfg[node].pop("method", None)

        if name is not None:
            overrides.append("=".join([f"{node}.name", name]))
        elif method is not None:
            overrides.append("=".join([f"{node}.method", method]))

        overrides.extend(parse_override(cfg[node], node_name=f"+{node}"))

    return overrides


def get_data_cfg(cfg):

    # Get data configs/overrides
    data_overrides = parse_data_overrides(cfg)
    data_cfg = compose(
        config_name="process_data", return_hydra_config=True, overrides=data_overrides,
    )

    return data_cfg


def get_features_cfg(cfg):
    # Get features configs/overrides
    features_overrides = parse_processed_data_overrides(cfg)
    features_overrides.extend(parse_override(cfg.lagged_features))
    features_cfg = compose(
        config_name="compute_lagged_features",
        return_hydra_config=True,
        overrides=features_overrides,
    )

    return features_cfg


def save_output(obj, path):
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

    # TODO: Check that file extensions match object types
    # i.e. .npy -> numpy arrays, etc

    # Do nothing if obj is None
    # e.g. groups if split=timeseries
    if obj is None:
        return None

    # orig_cwd = get_original_cwd()
    # logger.debug(f"Original working directory is {orig_cwd}")

    # Example: /home/danieliong/geomag-forecasting/filename
    # orig_path = Path(to_absolute_path(path))

    # Example: /home/danieliong/geomag-forecasting/outputs/{...}/filename

    # output_path = Path(orig_path.name).resolve()
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix

    if is_pandas(obj):
        if ext == ".pkl":
            obj.to_pickle(output_path)
        elif ext == ".csv":
            obj.to_csv(output_path)
        else:
            raise ValueError(
                f"Unsupported file extension {ext} for saving pandas object."
            )
    elif is_numpy(obj):
        if ext == ".npy":
            np.save(output_path, obj)
        else:
            np.savetxt(output_path, obj)
    elif ext == ".joblib":
        joblib.dump(obj, output_path)
    elif ext == ".pkl":
        with open(output_path, "wb") as f:
            try:
                pickle.dump(obj, f)
            except (pickle.PickleError, pickle.PicklingError):
                dill.dump(obj, f)
    else:
        raise ValueError(f"Cannot save {type(obj)} object to {path}.")

    logger.debug(f"Saved output to {path}.")


# # Might not be needed if using MLFlow
# class Results:
#     def __init__(self, model, time, root_dir="~/geomag-forecasting"):
#         self.model = model
#         self.time = time
#         self.root_dir = root_dir

#     def get_multirun_dir(self):

#         time = pd.to_datetime(self.time)
#         time_dir = time.strftime("%Y-%m-%d_%H-%M")
#         dir_name = f"{self.root_dir}/multirun/{self.model}/{time_dir}"
#         dir_path = Path(dir_name).expanduser()

#         return dir_path

#     def get_multirun_results(self):

#         results_dir = self.get_multirun_dir()
#         results = OmegaConf.load(f"{results_dir}/optimization_results.yaml")

#         return results

#     def get_best_dir(self):

#         results_dir = self.get_multirun_dir()
#         results = self.get_multirun_results()

#         best_params = results.best_evaluated_params
#         best_params_list = [f"{key}={val}" for key, val in best_params.items()]

#         dir_iter = results_dir.iterdir()

#         def _best_dir(dir_path):
#             overrides_path = dir_path / ".hydra/overrides.yaml"

#             best = False
#             if overrides_path.exists():
#                 overrides = OmegaConf.load(dir_path / overrides_path)
#                 best = set(overrides) == set(best_params_list)

#             return best

#         best_dir = next(path for path in dir_iter if _best_dir(path))
#         return best_dir

#     def load_best_pred(self):

#         best_dir = self.get_best_dir()
#         ypred_path = best_dir / "ypred.pkl"

#         ypred = pd.read_pickle(ypred_path)
#         return ypred
