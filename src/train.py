#!/usr/bin/env python

import hydra
import pandas as pd
import numpy as np
import logging

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

# from src.storm_utils import apply_storms

logger = logging.getLogger(__name__)


# DELETE
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


# DELETE
def compute_lagged_features(
    lag, exog_lag, lead, train=True, processor=None, **load_kwargs
):

    # QUESTION: What if there is no processor?
    if not train and processor is None:
        raise ValueError("processor must be specified if train=False")

    # Load processed data
    if train:
        X = load_processed_data(name="train_features", **load_kwargs)
        X = load_processed_data(name="train_features", **load_kwargs)
        y = load_processed_data(name="train_target", **load_kwargs)
    else:
        X = load_processed_data(name="test_features", **load_kwargs)
        y = load_processed_data(name="test_target", **load_kwargs)

    # Load features pipeline
    # QUESTION: What if this is really big?
    # HACK: Passing in clone of features_pipeline might be problem if
    # Resampler is in the pipeline. Fortunately, Resampler doesn't do anything if
    # freq is < data's freq. Find a better way to handle this.
    # IDEA: Remove Resampler?
    transformer_y = clone(load_processor(name="features_pipeline", **load_kwargs))
    # TODO: Delete Resampler in pipeline
    # It is okay for now because feature freq is probably < target freq.

    if processor is None:
        # Transform lagged y same way as other solar wind features
        processor = LaggedFeaturesProcessor(
            transformer_y=transformer_y, lag=lag, exog_lag=exog_lag, lead=lead,
        )
        processor.fit(X, y)

    # NOTE: fitted transformer is an attribute in processor
    X_lagged, y_target = processor.transform(X, y)

    return X_lagged, y_target, processor


def get_cv_split(y, method, init_params, **load_kwargs):

    # Returns None if group_labels doesn't exist (e.g. for cv=timeseries)
    # groups = load_processed_data(name="group_labels", **load_kwargs)

    if method == "timeseries":
        splitter = TimeSeriesSplit(**init_params)
        groups = None
    elif method == "storms":
        splitter = GroupKFold(**init_params)
        # groups = load_processed_data(
        #     name="group_labels", must_exist=True, **load_kwargs
        # )

        # NOTE: y should be a pandas object with storm index so set groups to be storm index
        groups = y.storms.index

        # Reindex times within each storm
        # Cannot just reindex groups with y index directly because there are
        # overlapping storms
        # groups = pd.concat(
        #     (
        #         groups[groups[STORM_LEVEL] == storm].reindex(y.storms.get(storm).index)
        #         for storm in y.storms.level
        #     )
        # )

        # assert len(groups) == len(
        #     y
        # ), f"Length of groups ({len(groups)}) does not match length of y ({len(y)})"

    split = splitter.split(y, groups=groups)

    return list(split)


# TODO: Add more general version to utils later.
def convert_pred_to_pd(ypred, y):

    if isinstance(ypred, (pd.Series, pd.DataFrame)):
        logger.debug(
            "Predictions are already pandas objects and will not be converted."
        )
        return ypred

    if isinstance(ypred, np.ndarray):
        # NOTE: This should be okay but check to make sure later.
        if isinstance(y, pd.DataFrame):
            ypred = pd.DataFrame(ypred, columns=y.columns, index=y.index)
        elif isinstance(y, pd.Series):
            ypred = pd.Series(ypred, name=y.name, index=y.index)

    return ypred


def inv_transform_targets(y, ypred, **load_kwargs):

    logger.debug("Inverse transforming y and predictions...")
    target_pipeline = load_processor(name="target_pipeline", **load_kwargs)
    y = target_pipeline.inverse_transform(y)
    ypred = target_pipeline.inverse_transform(ypred)

    return y, ypred


def compute_metric(y, ypred, metric):
    # QUESTION: Should we inverse transform y and ypred before
    # computing metrics?

    if metric is None:
        logger.debug("Metric not given. Defaulting to rmse.")
        # Default to rmse
        metric = "rmse"

    logger.debug(f"Computing {metric}...")

    if metric == "rmse":
        metric_val = mean_squared_error(y, ypred, squared=False)
    elif metric == "mse":
        metric_val = mean_squared_error(y, ypred, squared=True)
    else:
        raise utils.NotSupportedError(metric, name_type="Metric")

    logger.info(f"{metric}: {metric_val}")

    return float(metric_val)


# NOTE: Make this return RMSE to use Nevergrad
@hydra.main(config_path="../configs/models", config_name="config")
def train(cfg):

    # XXX: Setting URI doesn't work unless I import inside main
    import mlflow

    load_kwargs = cfg.load
    inputs_dir = load_kwargs.inputs_dir

    cv_method = cfg.cv.method
    cv_init_params = cfg.cv.init_params

    model_name = cfg.model
    pred_path = cfg.outputs.predictions
    lag = cfg.lag
    exog_lag = cfg.exog_lag
    lead = cfg.lead
    inverse_transform = cfg.inverse_transform
    metric = cfg.metric
    # seed = cfg.seed

    mlflow.set_experiment(cfg.experiment_name)
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if cfg.experiment_name is not None:
        logger.debug(f"MLFlow Experiment: {cfg.experiment_name}")

    orig_cwd = get_original_cwd()
    tracking_uri = f"file://{orig_cwd}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    logger.debug(f"MLFlow Tracking URI: {tracking_uri}")

    if model_name == "xgboost":
        import mlflow.xgboost

        logger.debug("Turning on xgboost MLFlow autologging...")
        mlflow.xgboost.autolog()

    # # Nest runs if train was run within another run
    # if mlflow.active_run() is None:
    #     nested = True
    #     logger.debug("Nesting MLFlow runs...")
    # else:
    #     nested = False

    active_run = mlflow.active_run()
    if active_run is None:
        run_id = None
    else:
        run_id = active_run.info.run_id
        logger.debug(f"Active run_id: {run_id}")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment.experiment_id):

        inputs_hydra_dir = Path(to_absolute_path(inputs_dir)) / ".hydra"
        mlflow.log_artifacts(inputs_hydra_dir, artifact_path="inputs_configs")
        mlflow.log_artifacts(".hydra", artifact_path="model_configs")

        mlflow.log_param("model", model_name)

        logger.info("Loading training data and computing lagged features...")
        mlflow.log_params({"lag": lag, "exog_lag": exog_lag, "lead": lead})

        X_train = load_processed_data("X_train", **load_kwargs)
        y_train = load_processed_data("y_train", **load_kwargs)
        X_test = load_processed_data("X_test", **load_kwargs)
        y_test = load_processed_data("y_test", **load_kwargs)

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
        mlflow.log_param("cv_method", cv_method)
        mlflow.log_params(cv_init_params)

        logger.info(f"Fitting model {model_name}...")
        model = get_model(model_name, cfg)
        model.fit(X_train, y_train, cv=cv)

        # logger.info("Saving model outputs...")
        # model.save_output()

        logger.info("Computing predictions...")
        ypred = model.predict(X_test)
        ypred = convert_pred_to_pd(ypred, y_test)
        if inverse_transform:
            y_test, ypred = inv_transform_targets(y_test, ypred, **load_kwargs)

        metric_val = compute_metric(y_test, ypred, metric=metric)
        mlflow.log_metrics({metric: metric_val})

        logger.info("Saving predictions...")
        utils.save_output(ypred, pred_path)
        mlflow.log_artifact(pred_path)

    return metric_val


if __name__ == "__main__":
    train()
