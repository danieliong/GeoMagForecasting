#!/usr/bin/env python

import hydra
import pandas as pd
import numpy as np
import logging

from src.models import *
from hydra.utils import to_absolute_path
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def load_inputs(cfg, train=True):

    params = cfg.inputs
    dir_path = Path(params.dir)

    def _load(name):
        rel_path = dir_path / params[name]
        abs_path = to_absolute_path(rel_path)
        return np.load(abs_path)

    if train:
        X = _load("train_features")
        y = _load("train_target")
    else:
        X = _load("test_features")
        y = _load("test_target")

    return X, y


def get_cv(y, cfg):

    split_params = OmegaConf.to_container(cfg.split)
    method = split_params.pop("method")

    if method == "timeseries":
        splitter = TimeSeriesSplit(**split_params)

    # split = splitter.split(y)

    return splitter



def compute_metric(y, ypred, cfg):
    # QUESTION: Should we inverse transform y and ypred before
    # computing metrics?

    metric = OmegaConf.select(cfg, "metric")
    if metric is None:
        # Default to rmse
        metric = "rmse"

    logger.debug(f"Computing {metric}...")

    if metric == "rmse":
        metric_val = mean_squared_error(y, ypred, squared=False)
    elif metric == "mse":
        metric_val = mean_squared_error(y, ypred, squared=True)


    return float(metric_val)


# NOTE: Make this return RMSE to use Nevergrad
@hydra.main(config_path="../configs/models", config_name="config")
def main(cfg):

    model_name = cfg.model

    logger.debug("Loading training data...")
    X_train, y_train = load_inputs(cfg, train=True)

    logger.debug("Getting CV split...")
    cv = get_cv(y_train, cfg)

    logger.debug(f"Fitting model {model_name}...")
    model = get_model(model_name, cfg)
    model.fit(X_train, y_train, cv=cv)

    logger.debug("Loading testing data...")
    X_test, y_test = load_inputs(cfg, train=False)

    logger.debug(f"Computing predictions...")
    ypred = model.predict(X_test)
    metric_val = compute_metric(y_test, ypred, cfg)

    return metric_val

if __name__ == '__main__':
    main()
