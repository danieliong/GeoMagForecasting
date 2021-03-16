#!/usr/bin/env python

import hydra
import pandas as pd
import numpy as np
import logging

from collections import namedtuple
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pandas.tseries.frequencies import to_offset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
# from loguru import logger

# NOTE: Had to install src as package first
from src.utils import is_pandas, is_numpy, save_output
from src.preprocessing.loading import load_solar_wind, load_supermag

# TODO: Move this to model fitting section
from src.preprocessing.processors import create_pipeline, LaggedFeaturesProcessor

logger = logging.getLogger(__name__)

LOAD_PARAMS_NAME = "loading"


def load_data(cfg, start, end):
    def _get_kwargs(name):

        kwargs = OmegaConf.select(cfg, f"{name}.{LOAD_PARAMS_NAME}")
        kwargs = OmegaConf.to_container(kwargs)
        kwargs['start'] = start
        kwargs['end'] = end

        return kwargs

    original_cwd = get_original_cwd()

    if 'solar_wind' in cfg:
        features_df = load_solar_wind(working_dir=original_cwd,
                                      **_get_kwargs("solar_wind"))

    # assert len(cfg.target) == 1, "More than one target is specified."

    if cfg.target.name == "supermag":
        target_df = load_supermag(working_dir=original_cwd,
                                  **_get_kwargs("target"))

    return features_df, target_df


def split_data(X, y, cfg):
    # QUESTION: Should I use lead time in splitting?

    params = OmegaConf.select(cfg, "split")

    if OmegaConf.is_none(params.test_size):
        logger.debug(
            "Test size not provided in configs. It will be set to .2.")
        test_size = .2
    else:
        test_size = params.test_size

    test_start_idx = round(y.shape[0] * (1 - test_size))
    test_start = y.index[test_start_idx]
    one_sec = to_offset('S')

    def _split(x):
        if is_pandas(x):
            x_train, x_test = x.loc[:test_start], x.loc[test_start:]

            # HACK: Remove overlaps if there are any
            overlap = x_train.index.intersection(x_test.index)
            x_test.drop(index=overlap, inplace=True)
        else:
            raise TypeError("x must be a pandas DataFrame or series.")

        # FIXME: Data is split before processed so it looks like there is time
        # overlap if times are resampled

        return x_train, x_test

    X_train, X_test = _split(X)
    y_train, y_test = _split(y)

    Train = namedtuple('Train', ['X', 'y'])
    Test = namedtuple('Test', ['X', 'y'])

    return Train(X_train, y_train), Test(X_test, y_test)


# QUESTION: Should I compute lagged features here?
# QUESTION: Pass in clone of feature processor for transformer_y?
# def compute_lagged_features(X, y, cfg, transformer_y=None, processor=None):

#     if processor is None:
#         # Transform lagged y same way as other solar wind features
#         processor = LaggedFeaturesProcessor(transformer_y=transformer_y,
#                                             lag=cfg.lag,
#                                             exog_lag=cfg.exog_lag,
#                                             lead=cfg.lead)
#         processor.fit(X, y)

#     # NOTE: fitted transformer is an attribute in processor
#     X_lagged, y_target = processor.transform(X, y)

#     return X_lagged, y_target, processor


@hydra.main(config_path="../configs/preprocessing", config_name="config")
def main(cfg: DictConfig) -> None:
    """Loads data from data/raw and do initial pre-processing before extracting
    features. Pre-processed data is saved in data/interim
    """

    logger.info("Loading data...")
    features_df, target_df = load_data(cfg,
                                       start=cfg.start,
                                       end=cfg.end)

    logger.info("Splitting data...")
    train, test = split_data(features_df, target_df, cfg)
    # NOTE: train, test are namedtuples with attributes X, y

    logger.info("Transforming features...")
    features_pipeline = create_pipeline(**cfg.solar_wind.pipeline)

    X_train = features_pipeline.fit_transform(train.X)
    X_test = features_pipeline.transform(test.X)

    logger.info("Transforming target...")
    target_pipeline = create_pipeline(**cfg.target.pipeline)

    y_train = target_pipeline.fit_transform(train.y)
    y_test = target_pipeline.transform(test.y)

    # HACK: Delete overlap in y times.
    # Overlap occurs when we resample.
    y_overlap_idx = y_train.index.intersection(y_test.index)
    n_overlap = len(y_overlap_idx)
    if n_overlap > 0:
        logger.debug(f"Dropping {n_overlap} overlap(s) in target times "
                     + "between train and test.")
        y_test.drop(index=y_overlap_idx, inplace=True)

    # logger.info("Computing lagged features...")

    # NOTE: Moving this section to fit_models
    # HACK: Passing in clone of features_pipeline might be problem if
    # Resampler is in the pipeline. Fortunately, Resampler doesn't do anything if
    # freq is < data's freq. Find a better way to handle this.
    # IDEA: Remove Resampler?
    # train_features, train_target, processor = compute_lagged_features(
    #     X_train, y_train, cfg, transformer_y=clone(features_pipeline))

    # NOTE: Passing in processor here because it contains a transformer that
    # should only be fitted with training data
    # test_features, test_target, _ = compute_lagged_features(
    #     X_test, y_test, cfg, processor=processor)

    logger.info("Saving outputs...")
    # FIXME: symlink needs to be changed to dir name since cfg.outline contains
    # relative paths now.
    save_output(features_pipeline,
                cfg.output.features_pipeline,
                symlink=cfg.symlink)
    save_output(target_pipeline,
                cfg.output.target_pipeline,
                symlink=cfg.symlink)
    save_output(X_train, cfg.output.train_features, symlink=cfg.symlink)
    save_output(X_test, cfg.output.test_features, symlink=cfg.symlink)
    save_output(y_train, cfg.output.train_target, symlink=cfg.symlink)
    save_output(y_test, cfg.output.test_target, symlink=cfg.symlink)


if __name__ == '__main__':
    main()
