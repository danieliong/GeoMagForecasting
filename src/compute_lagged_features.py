#!/usr/bin/env python

import logging
import hydra
import numpy as np

from sklearn.base import clone
from pathlib import Path
from hydra.experimental import compose
from omegaconf import OmegaConf
from src import STORM_LEVEL
from src import utils
from src.preprocessing.load import load_processed_data, load_processor
from src.preprocessing.lag_processor import LaggedFeaturesProcessor

# TODO: Separate compute lagged features from fitting models

logger = logging.getLogger(__name__)


def _compute_lagged_features(
    lag,
    exog_lag,
    lead,
    history_freq=None,
    train=True,
    processor=None,
    inputs_dir=None,
    paths=None,
    # load_kwargs={},
    **processor_kwargs,
):

    # QUESTION: What if there is no processor?
    if not train and processor is None:
        raise ValueError("processor must be specified if train=False")

    # Load processed data
    if train:
        X = load_processed_data("train_features", inputs_dir=inputs_dir, paths=paths)
        X = load_processed_data("train_features", inputs_dir=inputs_dir, paths=paths)
        y = load_processed_data("train_target", inputs_dir=inputs_dir, paths=paths)
    else:
        X = load_processed_data("test_features", inputs_dir=inputs_dir, paths=paths)
        y = load_processed_data("test_target", inputs_dir=inputs_dir, paths=paths)

    # Load features pipeline
    # QUESTION: What if this is really big?
    # HACK: Passing in clone of features_pipeline might be problem if
    # Resampler is in the pipeline. Fortunately, Resampler doesn't do anything if
    # freq is < data's freq. Find a better way to handle this.
    # IDEA: Remove Resampler?
    transformer_y = clone(
        load_processor("features_pipeline", inputs_dir=inputs_dir, paths=paths)
    )
    # TODO: Delete Resampler in pipeline
    # It is okay for now because feature freq is probably < target freq.

    if processor is None:
        # Transform lagged y same way as other solar wind features
        logger.info("Computing lagged features...")
        processor = LaggedFeaturesProcessor(
            lag=lag,
            exog_lag=exog_lag,
            lead=lead,
            history_freq=history_freq,
            transformer_y=transformer_y,
            **processor_kwargs,
        )
        processor.fit(X, y)

    # NOTE: fitted transformer is an attribute in processor
    X_lagged, y_target = processor.transform(X, y)

    return X_lagged, y_target, processor


def _parse_data_overrides(cfg, override_nodes=["features", "target", "split"]):

    cfg = OmegaConf.to_container(cfg)
    overrides = []

    # If dictionary is not empty
    if bool(cfg["data"]):
        overrides.extend(utils.parse_override(cfg["data"]))

    for node in override_nodes:

        # HACK: Name could either be called name or method (for split)
        name = cfg[node].pop("name", None)
        method = cfg[node].pop("method", None)
        name = method if name is None else name
        overrides.append("=".join([node, name]))

        overrides.extend(utils.parse_override(cfg[node], node_name=node))

    return overrides


@hydra.main(config_path="../configs", config_name="compute_lagged_features")
def compute_lagged_features(cfg):

    # load_kwargs = cfg.inputs
    lag = cfg.lag
    exog_lag = cfg.exog_lag
    lead = cfg.lead
    history_freq = cfg.history_freq
    processor_kwargs = cfg.lag_processor
    outputs = cfg.outputs
    # inputs_dir = cfg.inputs_dir
    # output_dir = Path(outputs.output_dir)

    overrides = _parse_data_overrides(
        cfg, override_nodes=["features", "target", "split"]
    )

    data_cfg = compose(
        config_name="process_data", return_hydra_config=True, overrides=overrides
    )
    inputs_dir = data_cfg.hydra.run.dir
    paths = data_cfg.output

    X_train, y_train, processor = _compute_lagged_features(
        lag=lag,
        exog_lag=exog_lag,
        lead=lead,
        history_freq=history_freq,
        train=True,
        inputs_dir=inputs_dir,
        paths=paths,
        **processor_kwargs,
    )
    utils.save_output(X_train, outputs.X_train)
    utils.save_output(y_train, outputs.y_train)
    utils.save_output(processor, outputs.lag_processor)
    utils.save_output(processor.feature_names_, outputs.features_name)

    logger.info("Loading testing data and computing lagged features...")
    X_test, y_test, _ = _compute_lagged_features(
        lag=lag,
        exog_lag=exog_lag,
        lead=lead,
        train=False,
        inputs_dir=inputs_dir,
        processor=processor,
        paths=paths,
        **processor_kwargs,
    )

    utils.save_output(X_test, outputs.X_test)
    utils.save_output(y_test, outputs.y_test)


if __name__ == "__main__":
    compute_lagged_features()
