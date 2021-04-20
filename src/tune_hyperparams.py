#!/usr/bin/env python

import logging
import hydra

from pathlib import Path
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from hydra.experimental import compose

from src import utils
from src.preprocessing.load import load_processed_data, load_processor

from src.models import get_model
from src._train import compute_lagged_features, get_cv_split

logger = logging.getLogger(__name__)


def _parse_overrides(cfg, override_nodes=["data", "target", "features", "split"]):

    cfg = OmegaConf.to_container(cfg)
    overrides = []

    if bool(cfg["data"]):
        overrides.extend(utils.parse_override(cfg["data"], node_name="+data"))

    for node in ["target", "features", "split"]:
        name = cfg[node].pop("name", None)
        method = cfg[node].pop("method", None)

        if name is not None:
            overrides.append("=".join([f"{node}.name", name]))
        elif method is not None:
            overrides.append("=".join([f"{node}.method", method]))

        overrides.extend(utils.parse_override(cfg[node], node_name=f"+{node}"))

    return overrides


@hydra.main(config_path="../configs", config_name="tune_hyperparams")
def tune_hyperparams(cfg):
    from src.storm_utils import StormIndexAccessor, StormAccessor

    # load_kwargs = OmegaConf.select(cfg, "load")
    cv_method = cfg.cv.method
    cv_init_params = cfg.cv.params

    # Model specific parameters
    model_name = cfg.model
    metrics = cfg.metrics
    # seed = cfg.seed

    overrides = _parse_overrides(cfg)

    features_cfg = compose(
        config_name="compute_lagged_features",
        return_hydra_config=True,
        overrides=overrides,
    )
    lag = OmegaConf.select(features_cfg, "lag", default=60)
    exog_lag = OmegaConf.select(features_cfg, "exog_lag", default=60)
    lead = OmegaConf.select(features_cfg, "lead", default=60)
    inputs_dir = Path(to_absolute_path(features_cfg.hydra.run.dir))
    paths = features_cfg.outputs

    # TODO: Modify code after removing features configs
    if any(not (inputs_dir / path).exists() for path in paths.values()):
        # if not inputs_dir.exists():
        compute_lagged_features(lag, exog_lag, lead, inputs_dir)

    X_train = load_processed_data("X_train", inputs_dir=inputs_dir, paths=paths)
    y_train = load_processed_data("y_train", inputs_dir=inputs_dir, paths=paths)

    logger.info(f"Getting CV split for '{cv_method}' method...")
    cv = get_cv_split(y_train, cv_method, **cv_init_params)

    model = get_model(model_name)(cfg, cv=cv, metrics=metrics, mlflow=False)

    cv_score = model.cv_score(X_train, y_train)
    logger.info(f"CV score: {cv_score}")

    return cv_score


if __name__ == "__main__":
    tune_hyperparams()
