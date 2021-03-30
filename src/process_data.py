#!/usr/bin/env python

import hydra
import logging

from omegaconf import DictConfig

# Load functions from src
# NOTE: Have to install src as package first
from src.preprocessing.load import load_features, load_target
from src.preprocessing.processors import HydraPipeline
from src.preprocessing.split import split_data
from src.utils import save_output
from src.storm_utils import _has_storm_index, StormIndexAccessor, StormAccessor

logger = logging.getLogger(__name__)


def _delete_overlap_times(train, test):
    # HACK: Delete overlap in times between train, test.
    # Overlap occurs when we resample.
    overlap_idx = train.index.intersection(test.index)
    n_overlap = len(overlap_idx)

    if n_overlap > 0:
        logger.debug(
            f"Dropping {n_overlap} overlap(s) in target times "
            + "between train and test."
        )
        test.drop(index=overlap_idx, inplace=True)

    return train, test


@hydra.main(config_path="../configs/preprocessing", config_name="config")
def main(cfg: DictConfig) -> None:
    """Loads data from data/raw and do initial pre-processing before extracting
    features. Pre-processed data is saved in data/interim
    """

    # Get needed parameters from Hydra
    start = cfg.start
    end = cfg.end
    features_kwargs = cfg.features.load
    target_kwargs = cfg.target.load
    features_pipeline_cfg = cfg.features.pipeline
    target_pipeline_cfg = cfg.target.pipeline
    split_kwargs = cfg.split
    output_paths = cfg.output

    #######################################################################
    logger.info("Loading data...")

    features = load_features(cfg.features.name, start=start, end=end, **features_kwargs)
    target = load_target(cfg.target.name, start=start, end=end, **target_kwargs)

    #######################################################################
    logger.info("Splitting data...")
    train, test, groups = split_data(X=features, y=target, **split_kwargs)
    # NOTE: train, test are namedtuples with attributes X, y

    #######################################################################
    logger.info("Transforming features...")
    features_pipeline = HydraPipeline(cfg=features_pipeline_cfg)

    X_train = features_pipeline.fit_transform(train.X)
    X_test = features_pipeline.transform(test.X)

    logger.info("Transforming target...")
    target_pipeline = HydraPipeline(cfg=target_pipeline_cfg)

    y_train = target_pipeline.fit_transform(train.y)
    y_test = target_pipeline.transform(test.y)

    y_train, y_test = _delete_overlap_times(y_train, y_test)

    #######################################################################
    logger.info("Saving outputs...")
    save_output(features_pipeline, output_paths.features_pipeline)
    save_output(target_pipeline, output_paths.target_pipeline)
    save_output(X_train, output_paths.train_features)
    save_output(X_test, output_paths.test_features)
    save_output(y_train, output_paths.train_target)
    save_output(y_test, output_paths.test_target)
    save_output(groups, output_paths.group_labels)


if __name__ == "__main__":
    main()
