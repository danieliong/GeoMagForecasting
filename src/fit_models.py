#!/usr/bin/env python

import hydra
import pandas as pd
import numpy as np
import logging

from src.models import get_model
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

from src.preprocessing.processors import LaggedFeaturesProcessor

logger = logging.getLogger(__name__)


def load_inputs(name, cfg, type="data"):

    params = cfg.inputs
    dir_path = Path(params.dir)
    rel_path = dir_path / params[name]

    abs_path = Path(to_absolute_path(rel_path))
    ext = abs_path.suffix

    if type == "data":
        if ext == ".npy":
            inputs = np.load(abs_path)
        elif ext == ".pkl":
            inputs = pd.read_pickle(abs_path)
        else:
            raise ValueError("Data path must have extension .npy or .pkl.")
    elif type == "processor":
        with open(abs_path, "rb") as f:
            if ext == ".pkl":
                import dill
                inputs = dill.load(f)
            elif ext == ".joblib":
                import joblib
                inputs = joblib.load(f)

    return inputs


def compute_lagged_features(cfg,
                            train=True,
                            processor=None):

    if not train and processor is None:
        raise ValueError("processor must be specified if train=False")

    # Load processed data
    if train:
        X = load_inputs("train_features", cfg, type="data")
        y = load_inputs("train_target", cfg, type="data")
    else:
        X = load_inputs("test_features", cfg, type="data")
        y = load_inputs("test_target", cfg, type="data")


    # Load features pipeline
    # QUESTION: What if this is really big?
    # HACK: Passing in clone of features_pipeline might be problem if
    # Resampler is in the pipeline. Fortunately, Resampler doesn't do anything if
    # freq is < data's freq. Find a better way to handle this.
    # IDEA: Remove Resampler?
    transformer_y = clone(load_inputs("features_pipeline", cfg, type="processor"))
    # TODO: Delete Resampler in pipeline
    # It is okay for now because feature freq is probably < target freq.

    if processor is None:
        # Transform lagged y same way as other solar wind features
        processor = LaggedFeaturesProcessor(transformer_y=transformer_y,
                                            lag=cfg.lag,
                                            exog_lag=cfg.exog_lag,
                                            lead=cfg.lead)
        processor.fit(X, y)

    # NOTE: fitted transformer is an attribute in processor
    X_lagged, y_target = processor.transform(X, y)

    return X_lagged, y_target, processor


def get_cv(y, cfg):


    split_params = OmegaConf.to_container(cfg.split)
    method = split_params.pop("method")

    if method == "timeseries":
        splitter = TimeSeriesSplit(**split_params)

    # split = splitter.split(y)

    return splitter


def convert_pred_to_pd(ypred, y):

    if isinstance(ypred, np.ndarray):
        # NOTE: This should be okay but check to make sure later.
        if isinstance(y, pd.DataFrame):
            ypred = pd.DataFrame(ypred, columns=y.columns, index=y.index)
        elif isinstance(y, pd.Series):
            ypred = pd.Series(ypred, name=y.name, index=y.index)

    return ypred


def inv_transform_targets(y, ypred, cfg):

    inverse_transform = OmegaConf.select(cfg, "inverse_transform")

    if inverse_transform:
        logger.debug("Inverse transforming y and predictions...")
        target_pipeline = load_inputs("target_pipeline", cfg, type="processor")
        y = target_pipeline.inverse_transform(y)
        ypred = target_pipeline.inverse_transform(ypred)

    return y, ypred


def compute_metric(y, ypred, cfg):
    # QUESTION: Should we inverse transform y and ypred before
    # computing metrics?

    metric = OmegaConf.select(cfg, "metric")
    if metric is None:
        # Default to rmse
        metric = "rmse"

    # NOTE: We switched to inverse transforming targets before computing metric
    # y, ypred = inv_transform_targets(y, ypred, cfg)

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
    pred_path = cfg.outputs.predictions

    logger.info("Loading training data and computing lagged features...")
    X_train, y_train, processor = compute_lagged_features(cfg, train=True)

    logger.info("Getting CV split...")
    cv = get_cv(y_train, cfg)

    logger.info(f"Fitting model {model_name}...")
    model = get_model(model_name, cfg)
    model.fit(X_train, y_train, cv=cv)

    logger.info("Saving model outputs...")
    model.save_output()

    logger.info("Loading testing data and computing lagged features...")
    X_test, y_test, _ = compute_lagged_features(
        cfg, train=False, processor=processor)

    logger.info("Computing predictions...")
    ypred = model.predict(X_test)
    ypred = convert_pred_to_pd(ypred, y_test)
    ypred = inv_transform_targets(y_test, ypred, cfg)

    metric_val = compute_metric(y_test, ypred, cfg)

    logger.info("Saving predictions...")
    ypred.to_pickle(pred_path)

    return metric_val


if __name__ == '__main__':
    main()
