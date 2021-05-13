#!/usr/bin/env python

import os
import hydra
import pandas as pd
import numpy as np
import logging
import yaml
import mlflow
import matplotlib.pyplot as plt

from hydra.utils import to_absolute_path, get_original_cwd
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from hydra.experimental import compose

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
    setup_mlflow,
    # compute_lagged_features,
    # parse_overrides,
)
from src.compute_lagged_features import compute_lagged_features
from src.plot import plot_predictions

logger = logging.getLogger(__name__)

OmegaConf.register_resolver("range", lambda x, y: list(range(int(x), int(y) + 1)))


def update_tuned_hyperparams(cfg, features_cfg):

    tune_cfg = OmegaConf.select(cfg, "tune", default=None)
    if tune_cfg is not None:
        exp_name = OmegaConf.select(tune_cfg, "experiment_name", default=None)
        exp_id = OmegaConf.select(tune_cfg, "experiment_id", default=None)
        params = OmegaConf.select(tune_cfg, "params", default=None)
        kwargs = OmegaConf.select(tune_cfg, "kwargs", default=None)
        metric = OmegaConf.select(tune_cfg, "metric", default="rmse")
        filter_string = OmegaConf.select(tune_cfg, "filter_string", default="")

        if exp_id is None and exp_name is not None:
            exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id

        if exp_id is not None:
            best_run = mlflow.search_runs(
                str(exp_id),
                order_by=[f"metric.{metric}"],
                max_results=1,
                filter_string=filter_string,
            )
            best_params = best_run.filter(regex="^params.")
            best_params.rename(columns=lambda x: x.replace("params.", ""), inplace=True)

            # Update hyperparams
            if params is not None:
                best_hyperparams = best_params[params].iloc[0].to_dict()
                OmegaConf.update(cfg, "param", best_hyperparams, merge=True)
                hyperparams_str = ", ".join(
                    ["=".join(x) for x in best_hyperparams.items()]
                )
                logger.info(f"Updated hyperparameters: {hyperparams_str}")

            # Update kwargs
            if kwargs is not None:
                best_kwargs = best_params[kwargs].iloc[0].to_dict()
                OmegaConf.update(cfg, "kwargs", best_kwargs, merge=True)
                kwargs_str = ", ".join(["=".join(x) for x in best_kwargs.items()])
                logger.info(f"Updated kwargs: {kwargs_str}")

            if tune_cfg.lags:
                OmegaConf.update(features_cfg, "lag", int(best_params["lag"]))
                OmegaConf.update(features_cfg, "exog_lag", int(best_params["exog_lag"]))
                logger.info(f"Updated lag: {best_params['lag']}")
                logger.info(f"Updated exog_lag: {best_params['exog_lag']}")


# NOTE: Make this return RMSE to use Nevergrad
@hydra.main(config_path="../configs", config_name="train")
def train(cfg):
    from src.storm_utils import StormIndexAccessor, StormAccessor

    # # Get data configs/overrides
    # data_overrides = utils.parse_data_overrides(cfg)
    # data_cfg = compose(
    #     config_name="process_data", return_hydra_config=True, overrides=data_overrides,
    # )

    # # Get features configs/overrides
    # features_overrides = utils.parse_processed_data_overrides(cfg)
    # features_overrides.extend(utils.parse_override(cfg.lagged_features))
    # features_cfg = compose(
    #     config_name="compute_lagged_features",
    #     return_hydra_config=True,
    #     overrides=features_overrides,
    # )

    # Model specific parameters
    model_name = cfg.model
    metrics = cfg.metrics
    pred_path = OmegaConf.select(cfg.outputs, "predictions", default="ypred.pkl")
    # seed = cfg.seed

    data_cfg = utils.get_data_cfg(cfg)
    features_cfg = utils.get_features_cfg(cfg)
    processed_data_dir = Path(to_absolute_path(data_cfg.hydra.run.dir))
    inputs_dir = Path(to_absolute_path(features_cfg.hydra.run.dir))
    paths = features_cfg.outputs

    # Setup mlflow
    use_mlflow = OmegaConf.select(cfg, "mlflow", default=False)
    if use_mlflow:
        import mlflow

        run = setup_mlflow(cfg, features_cfg=features_cfg, data_cfg=data_cfg)

    update_tuned_hyperparams(cfg, features_cfg)

    # Compute lagged features if they haven't been computed yet
    if any(not (inputs_dir / path).exists() for path in paths.values()):
        inputs_dir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(inputs_dir)
        lagged_features_cfg = features_cfg.copy()
        with open_dict(lagged_features_cfg):
            _ = lagged_features_cfg.pop("hydra")
        compute_lagged_features(lagged_features_cfg)
        os.chdir(cwd)

    # General parameters
    # load_kwargs = cfg.load
    # processed_data_dir = Path(to_absolute_path(cfg.processed_data_dir))
    target_pipeline_path = cfg.target_pipeline
    inverse_transform = cfg.inverse_transform
    cv_method = cfg.cv.method
    cv_init_params = cfg.cv.params
    # lag = cfg.lag
    # exog_lag = cfg.exog_lag
    # lead = cfg.lead

    logger.info("Loading training data and computing lagged features...")

    # # HACK: Compute lagged features if they haven't been computed yet.
    # inputs_dir = Path(to_absolute_path(load_kwargs.inputs_dir))
    # paths = [inputs_dir / path for path in load_kwargs.paths.values()]
    # if any(not path.exists() for path in paths):
    #     # if not inputs_dir.exists():
    #     compute_lagged_features(lag, exog_lag, lead, inputs_dir)

    X_train = load_processed_data("X_train", inputs_dir=inputs_dir, paths=paths)
    y_train = load_processed_data("y_train", inputs_dir=inputs_dir, paths=paths)
    X_test = load_processed_data("X_test", inputs_dir=inputs_dir, paths=paths)
    y_test = load_processed_data("y_test", inputs_dir=inputs_dir, paths=paths)
    feature_names = load_processed_data(
        "features_names", inputs_dir=inputs_dir, paths=paths
    )

    # QUESTION: Log everything at end cleaner?
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
    cv = get_cv_split(y_train, cv_method, **cv_init_params)
    # QUESTION: Do we still need CV here?

    ###########################################################################
    # Fit and evaluate model
    ###########################################################################

    logger.info(f"Fitting model {model_name}...")
    model = get_model(model_name)(cfg, cv=cv, metrics=metrics, mlflow=use_mlflow)
    model.fit(X_train, y_train, feature_names=feature_names)

    # TODO: Make this more general. It currently only applies to xgboost
    # QUESTION: compute CV score in score method?
    # score = model.cv_score(X_train, y_train)

    model.save_output()

    ###########################################################################
    # Compute/save predictions on test set
    ###########################################################################

    logger.info("Computing predictions...")
    ypred = model.predict(X_test)
    ypred = convert_pred_to_pd(ypred, y_test)
    if inverse_transform:
        y_test, ypred = inv_transform_targets(
            y_test, ypred, path=target_pipeline_path, processor_dir=processed_data_dir,
        )

    logger.info("Saving predictions...")

    utils.save_output(ypred, pred_path)
    if use_mlflow:
        mlflow.log_artifact(pred_path)

    ###########################################################################
    # Compute and log test metrics
    ###########################################################################

    if use_mlflow:
        test_score = compute_metrics(y_test, ypred, metrics=metrics)
        if isinstance(metrics, (list, tuple)):
            if len(metrics) > 1:
                for metric in metrics:
                    mlflow.log_metrics({metric: test_score[metric]})
        else:
            mlflow.log_metrics({metrics: test_score})

    ##########################################################################
    # Plot predictions on test set
    ##########################################################################

    # Plot predictions
    plot_kwargs = OmegaConf.to_container(cfg.plot, resolve=True)
    fig, ax = model.plot(
        X_test,
        y_test,
        lead=features_cfg.lead,
        unit=features_cfg.lag_processor.unit,
        **plot_kwargs,
    )
    plt.close()
    # if isinstance(fig, list):
    #     for f in fig:
    #         f.close()
    # elif isinstance(fig, dict):
    #     for f in fig.values():
    #         f.close()
    # else:
    #     fig.close()

    if use_mlflow:
        mlflow.end_run()

    # plot_predictions(
    #     y_test,
    #     ypred,
    #     metrics=metrics,
    #     use_mlflow=use_mlflow,
    #     persistence=cfg.plot.persistence,
    #     lead=features_cfg.lead,
    # )

    # return score


if __name__ == "__main__":
    train()
