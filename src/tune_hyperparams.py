#!/usr/bin/env python

import os
import logging
import hydra

from pathlib import Path
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, open_dict
from hydra.experimental import compose

from src import utils
from src.preprocessing.load import load_processed_data, load_processor

from src.models import get_model
from src._train import get_cv_split, setup_mlflow
from src.compute_lagged_features import compute_lagged_features

logger = logging.getLogger(__name__)

OmegaConf.register_resolver("range", lambda x, y: list(range(int(x), int(y) + 1)))


@hydra.main(config_path="../configs", config_name="tune_hyperparams")
def tune_hyperparams(cfg):
    from src.storm_utils import StormIndexAccessor, StormAccessor

    data_cfg = utils.get_data_cfg(cfg)
    features_cfg = utils.get_features_cfg(cfg)
    inputs_dir = Path(to_absolute_path(features_cfg.hydra.run.dir))
    paths = features_cfg.outputs

    use_mlflow = OmegaConf.select(cfg, "mlflow", default=False)
    if use_mlflow:
        import mlflow

        run = setup_mlflow(cfg, features_cfg=features_cfg, data_cfg=data_cfg)

    cv_method = cfg.cv.method
    cv_init_params = cfg.cv.params

    fit_model = cfg.fit_model

    # Model specific parameters
    model_name = cfg.model
    metrics = cfg.metrics
    val_storms = cfg.val_storms
    # seed = cfg.seed

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

    X_train = load_processed_data("X_train", inputs_dir=inputs_dir, paths=paths)
    y_train = load_processed_data("y_train", inputs_dir=inputs_dir, paths=paths)

    if val_storms is not None:
        model = get_model(model_name)(
            cfg, val_storms=val_storms, metrics=metrics, mlflow=use_mlflow
        )

        score = model.val_score(X_train, y_train)
        if use_mlflow:
            mlflow.log_metric(model.cv_metric, score)
    else:
        logger.info(f"Getting CV split for '{cv_method}' method...")
        cv = get_cv_split(y_train, cv_method, **cv_init_params)
        model = get_model(model_name)(cfg, cv=cv, metrics=metrics, mlflow=use_mlflow)

        score = model.cv_score(X_train, y_train, fit_model=fit_model)
        logger.info(f"CV score: {score}")
        if use_mlflow:
            mlflow.log_metric(model.cv_metric, score)

    mlflow.log_params(model.params)

    model.save_output()
    # utils.save_output(model, cfg.outputs.hydra_model)

    if use_mlflow:
        mlflow.end_run()

    return score


if __name__ == "__main__":
    tune_hyperparams()
