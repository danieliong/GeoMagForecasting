#!/usr/bin/env python

import hydra
import pandas as pd
import numpy as np
import logging

from collections import namedtuple
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pandas.tseries.frequencies import to_offset
# from loguru import logger

# NOTE: Had to install src as package first
from src.utils import is_pandas, is_numpy, save_output
from src.preprocessing.loading import load_solar_wind, load_supermag, load_symh
from src.preprocessing.processors import HydraPipeline

logger = logging.getLogger(__name__)


def load_data(cfg, start, end):
    def _get_kwargs(name):

        kwargs = OmegaConf.select(cfg, f"{name}.loading")
        kwargs = OmegaConf.to_container(kwargs)
        kwargs['start'] = start
        kwargs['end'] = end

        return kwargs

    original_cwd = get_original_cwd()

    if cfg.features.name == "solar_wind":
        features_df = load_solar_wind(
            working_dir=original_cwd, **_get_kwargs("features")
        )

    # assert len(cfg.target) == 1, "More than one target is specified."

    if cfg.target.name == "supermag":
        target_df = load_supermag(working_dir=original_cwd,
                                  **_get_kwargs("target"))
    elif cfg.target.name == "symh":
        target_df = load_symh(working_dir=original_cwd, **_get_kwargs("target"))

    return features_df, target_df

def split_data_storms(X, y, test_size=.2):
    pass

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
    features_pipeline = HydraPipeline(cfg=cfg.features.pipeline)

    X_train = features_pipeline.fit_transform(train.X)
    X_test = features_pipeline.transform(test.X)

    # Note used after I created HydraPipeline
    # if features_pipeline is None:
    #     X_train = train.X
    #     X_test = test.X
    # else:
    #     X_train = features_pipeline.fit_transform(train.X)
    #     X_test = features_pipeline.transform(test.X)

    logger.info("Transforming target...")
    target_pipeline = HydraPipeline(cfg=cfg.target.pipeline)

    y_train = target_pipeline.fit_transform(train.y)
    y_test = target_pipeline.transform(test.y)

    # Note used after I created HydraPipeline
    # if target_pipeline is None:
    #     y_train = train.y
    #     y_test = test.y
    # else:
    #     y_train = target_pipeline.fit_transform(train.y)
    #     y_test = target_pipeline.transform(test.y)

    # HACK: Delete overlap in y times.
    # Overlap occurs when we resample.
    y_overlap_idx = y_train.index.intersection(y_test.index)
    n_overlap = len(y_overlap_idx)
    if n_overlap > 0:
        logger.debug(f"Dropping {n_overlap} overlap(s) in target times "
                     + "between train and test.")
        y_test.drop(index=y_overlap_idx, inplace=True)


    logger.info("Saving outputs...")
    # FIXME: symlink needs to be changed to dir name since cfg.outline contains
    # relative paths now.
    # (Probably will delete symlink anyways)
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
