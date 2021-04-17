#!/usr/bin/env python

import hydra
import pandas as pd
import numpy as np
import logging
import yaml
import matplotlib.pyplot as plt

from hydra.utils import to_absolute_path, get_original_cwd
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

from src import STORM_LEVEL
from src import utils
from src.models import get_model
from src.preprocessing.lag_processor import LaggedFeaturesProcessor
from src.preprocessing.load import load_processed_data, load_processor
from src.storm_utils import has_storm_index, StormIndexAccessor, StormAccessor
from src._train import (
    get_cv_split,
    convert_pred_to_pd,
    inv_transform_targets,
    compute_metrics,
    plot_predictions,
    compute_lagged_features,
)

logger = logging.getLogger(__name__)

DATA_CONFIGS_TO_LOG = {
    "start": "start",
    "end": "end",
    "target": "target.name",
    "features_source": "features.name",
    "features": "features.load.features",
    "target_processing": "target.pipeline.order",
    "features_processing": "features.pipeline.order",
}


def setup_mlflow(cfg):
    import mlflow

    processed_data_dir = Path(to_absolute_path(cfg.processed_data_dir))
    experiment_id = OmegaConf.select(cfg, "experiment_id")

    if experiment_id is None and cfg.experiment_name is not None:
        mlflow.set_experiment(cfg.experiment_name)
        experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
        if cfg.experiment_name is not None:
            logger.debug(f"MLFlow Experiment: {cfg.experiment_name}")

        experiment_id = experiment.experiment_id

    orig_cwd = get_original_cwd()
    tracking_uri = f"file://{orig_cwd}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    logger.debug(f"MLFlow Tracking URI: {tracking_uri}")

    if cfg.model == "xgboost":
        import mlflow.xgboost

        logger.debug("Turning on xgboost MLFlow autologging...")
        mlflow.xgboost.autolog()

    mlflow.start_run(experiment_id=experiment_id)

    data_hydra_dir = processed_data_dir / ".hydra"
    model_hydra_dir = Path(".hydra")
    mlflow.log_artifacts(data_hydra_dir, artifact_path="processed_data_configs")
    mlflow.log_artifacts(model_hydra_dir, artifact_path="model_configs")

    data_cfg = OmegaConf.load(data_hydra_dir / "config.yaml")
    for name, param_name in DATA_CONFIGS_TO_LOG.items():
        param = OmegaConf.select(data_cfg, param_name)
        if param is not None:
            if isinstance(param, list):
                param = ", ".join(param)
            mlflow.log_param(name, param)

    mlflow.log_params(
        {
            "model": cfg.model,
            "lag": cfg.lag,
            "exog_lag": cfg.exog_lag,
            "lead": cfg.lead,
            "cv_method": cfg.cv.method,
        }
    )
    mlflow.log_params(cfg.cv.params)

    return


# NOTE: Make this return RMSE to use Nevergrad
@hydra.main(config_path="../configs", config_name="train")
def train(cfg):
    from src.storm_utils import StormIndexAccessor, StormAccessor

    # TODO: Change using mlflow to be optional
    use_mlflow = OmegaConf.select(cfg, "mlflow")
    if use_mlflow is None:
        use_mlflow = False

    # General parameters
    load_kwargs = cfg.load
    processed_data_dir = Path(to_absolute_path(cfg.processed_data_dir))
    target_pipeline_path = cfg.target_pipeline
    inverse_transform = cfg.inverse_transform
    cv_method = cfg.cv.method
    cv_init_params = cfg.cv.params

    lag = cfg.lag
    exog_lag = cfg.exog_lag
    lead = cfg.lead

    # Model specific parameters
    model_name = cfg.model
    pred_path = OmegaConf.select(cfg.outputs, "predictions")
    metrics = cfg.metrics
    # seed = cfg.seed

    # TODO: Separate this into another function
    if use_mlflow:
        import mlflow

        setup_mlflow(cfg)

    logger.info("Loading training data and computing lagged features...")

    # HACK: Compute lagged features if they haven't been computed yet.
    inputs_dir = Path(to_absolute_path(load_kwargs.inputs_dir))
    paths = [inputs_dir / path for path in load_kwargs.paths.values()]
    if any(not path.exists() for path in paths):
        # if not inputs_dir.exists():
        compute_lagged_features(lag, exog_lag, lead, inputs_dir)

    X_train = load_processed_data("X_train", **load_kwargs)
    y_train = load_processed_data("y_train", **load_kwargs)
    X_test = load_processed_data("X_test", **load_kwargs)
    y_test = load_processed_data("y_test", **load_kwargs)
    feature_names = load_processed_data("feature_names", **load_kwargs)

    if use_mlflow:
        n_train_obs, n_features = X_train.shape
        n_test_obs, _ = y_test.shape
        mlflow.log_params(
            {
                "n_train_obs": n_train_obs,
                "n_test_obs": n_test_obs,
                "n_features": n_features,
            }
        )

    logger.info(f"Getting CV split for '{cv_method}' method...")
    cv = get_cv_split(y_train, cv_method, cv_init_params, **load_kwargs)
    # QUESTION: What if model fit method doesn't need CV?

    ###########################################################################
    # Fit and evaluate model
    ###########################################################################

    logger.info(f"Fitting model {model_name}...")
    model = get_model(model_name)(cfg, cv=cv, mlflow=use_mlflow)
    model.fit(X_train, y_train, feature_names=feature_names)

    # TODO: Make this more general. It currently only applies to xgboost
    # QUESTION: compute CV score in score method?
    score = model.cv_score(X_train, y_train)

    # logger.info("Saving model outputs...")
    # model.save_output()

    ###########################################################################
    # Compute/plot/save predictions on test set
    ###########################################################################

    logger.info("Computing predictions...")
    ypred = model.predict(X_test)
    ypred = convert_pred_to_pd(ypred, y_test)
    if inverse_transform:
        y_test, ypred = inv_transform_targets(
            y_test, ypred, path=target_pipeline_path, processor_dir=processed_data_dir,
        )

    logger.info("Saving predictions...")

    if use_mlflow:
        mlflow.log_artifact(pred_path)
    else:
        utils.save_output(ypred, pred_path)

    # Plot predictions
    plot_predictions(y_test, ypred, metrics=metrics, use_mlflow=use_mlflow)

    ###########################################################################
    # Compute and log test metrics
    ###########################################################################

    test_score = compute_metrics(y_test, ypred, metrics=metrics)

    if use_mlflow:
        if isinstance(metrics, (list, tuple)):
            if len(metrics) > 1:
                for metric in metrics:
                    mlflow.log_metrics({metric: test_score[metric]})
        else:
            mlflow.log_metrics({metrics: test_score})

    if use_mlflow:
        mlflow.end_run()

    return score


if __name__ == "__main__":
    train()
