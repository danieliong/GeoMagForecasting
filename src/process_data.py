#!/usr/bin/env python

import hydra
import pandas as pd
import numpy as np

from collections import namedtuple
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import TimeSeriesSplit
from loguru import logger

# TODO: Find better way to do this
try:
    from utils import is_pandas, is_numpy, save_output
    from preprocessing.loading import load_solar_wind, load_supermag
    from preprocessing.processors import create_pipeline
except ImportError:
    from src.utils import is_pandas, is_numpy, save_output
    from src.processing.loading import load_solar_wind, load_supermag
    from src.preprocessing.processors import create_pipeline


def load_data(cfg):
    original_cwd = get_original_cwd()

    if 'solar_wind' in cfg:
        features_df = load_solar_wind(working_dir=original_cwd,
                                      **cfg.solar_wind.loading)

    if 'supermag' in cfg:
        target_df = load_supermag(working_dir=original_cwd,
                                  **cfg.supermag.loading)

    return features_df, target_df


# TODO: Move this to model training part
def get_train_val_split(y, method, val_size, **kwargs):

    if method == "timeseries":
        splitter = TimeSeriesSplit(**kwargs)

    split = splitter.split(y)

    return split


def split_data(X, y, cfg):

    # params = OmegaConf.select(cfg, "split").to_container()
    params_dict = OmegaConf.to_container(cfg)

    # Defaults to .2 if none specified
    test_size = params_dict.pop('test_size', .2)

    test_start = round(y.shape[0] * (1 - test_size))

    def _split(x):
        if is_pandas(x):
            x_train, x_test = x.iloc[:test_start], x[test_start:]
        elif is_numpy(x):
            x_train, x_test = x[:test_start], x[test_start:]
        else:
            raise TypeError("x must be either a pd.DataFrame or np.ndarray.")

        return x_train, x_test

    X_train, X_test = _split(X)
    y_train, y_test = _split(y)

    Train = namedtuple('Train', ['X', 'y'])
    Test = namedtuple('Test', ['X', 'y'])

    return Train(X_train, y_train), Test(X_test, y_test)


def compute_lagged_features(X, y, cfg):
    # TODO
    pass


@hydra.main(config_path="../configs/preprocessing", config_name="config")
def main(cfg: DictConfig) -> None:
    """Loads data from data/raw and do initial pre-processing before extracting
    features. Pre-processed data is saved in data/interim
    """

    logger.info("Loading data...")
    features_df, target_df = load_data(cfg)

    logger.info("Splitting data...")
    train, test = split_data(features_df, target_df, cfg)

    logger.info("Making features pipeline...")
    features_pipeline = create_pipeline(**cfg.solar_wind.pipeline)

    logger.info("Transforming train features...")
    X_train = features_pipeline.fit_transform(train.X)
    logger.info("Transforming test features...")
    X_test = features_pipeline.transform(test.X)

    logger.info("Making target pipeline...")
    target_pipeline = create_pipeline(**cfg.target.pipeline)

    logger.info("Transforming train target...")
    X_train = target_pipeline.fit_transform(train.X)
    logger.info("Transforming test target...")
    X_test = target_pipeline.transform(test.X)



    # TODO: Figure out Hydra logging
    # TODO: Transform train and test target


    # Save outputs
    logger.info("Saving outputs...")
    save_output(X_train, cfg.output.train_features, symlink=True)
    save_output(X_test, cfg.output.test_features, symlink=True)


if __name__ == '__main__':
    main()
