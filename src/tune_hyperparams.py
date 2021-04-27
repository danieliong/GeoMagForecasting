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
from src._train import compute_lagged_features, get_cv_split

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="tune_hyperparams")
def tune_hyperparams(cfg):
    from src.storm_utils import StormIndexAccessor, StormAccessor

    data_cfg = utils.get_data_cfg(cfg)
    features_cfg = utils.get_features_cfg(cfg)
    inputs_dir = Path(to_absolute_path(features_cfg.hydra.run.dir))
    paths = features_cfg.outputs

    # load_kwargs = OmegaConf.select(cfg, "load")
    cv_method = cfg.cv.method
    cv_init_params = cfg.cv.params

    # Model specific parameters
    model_name = cfg.model
    metrics = cfg.metrics
    # seed = cfg.seed

    overrides = utils.parse_processed_data_overrides(cfg)

    # features_cfg = compose(
    #     config_name="compute_lagged_features",
    #     return_hydra_config=True,
    #     overrides=overrides,
    # )
    # lag = OmegaConf.select(features_cfg, "lag", default=60)
    # exog_lag = OmegaConf.select(features_cfg, "exog_lag", default=60)
    # lead = OmegaConf.select(features_cfg, "lead", default=60)
    # inputs_dir = Path(to_absolute_path(features_cfg.hydra.run.dir))
    # paths = features_cfg.outputs

    # # TODO: Modify code after removing features configs
    # if any(not (inputs_dir / path).exists() for path in paths.values()):
    #     # if not inputs_dir.exists():
    #     compute_lagged_features(lag, exog_lag, lead, inputs_dir)

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

    logger.info(f"Getting CV split for '{cv_method}' method...")
    cv = get_cv_split(y_train, cv_method, **cv_init_params)

    model = get_model(model_name)(cfg, cv=cv, metrics=metrics, mlflow=False)

    cv_score = model.cv_score(X_train, y_train)
    logger.info(f"CV score: {cv_score}")

    model.save_output()
    # utils.save_output(model, cfg.outputs.hydra_model)

    return cv_score


if __name__ == "__main__":
    tune_hyperparams()
