#!/usr/bin/env python

import logging
import hydra
from sklearn.base import clone

from pathlib import Path
from src import STORM_LEVEL
from src import utils
from src.preprocessing.load import load_processed_data, load_processor
from src.preprocessing.lag_processor import LaggedFeaturesProcessor

# TODO: Separate compute lagged features from fitting models

logger = logging.getLogger(__name__)


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


@hydra.main(config_path="../configs/preprocessing", config_name="lagged_features")
def main(cfg):

    load_kwargs = cfg.inputs
    lag = cfg.lag
    exog_lag = cfg.exog_lag
    lead = cfg.lead
    outputs = cfg.outputs
    output_dir = Path(outputs.output_dir)

    X_train, y_train, processor = compute_lagged_features(
        lag=lag, exog_lag=exog_lag, lead=lead, train=True, **load_kwargs
    )
    utils.save_output(X_train, output_dir / outputs.X_train)
    utils.save_output(y_train, output_dir / outputs.y_train)
    utils.save_output(processor, output_dir / "lag_processor.pkl")

    logger.info("Loading testing data and computing lagged features...")
    X_test, y_test, _ = compute_lagged_features(
        lag=lag,
        exog_lag=exog_lag,
        lead=lead,
        train=False,
        processor=processor,
        **load_kwargs,
    )

    utils.save_output(X_test, output_dir / outputs.X_test)
    utils.save_output(y_test, output_dir / outputs.y_test)


if __name__ == "__main__":
    main()
