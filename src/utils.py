#!/usr/bin/env python

import os
# import pickle
import dill
import pandas as pd
import numpy as np

from pathlib import Path
from loguru import logger
from hydra.utils import to_absolute_path, get_original_cwd


def is_pandas(x):
    return isinstance(x, (pd.DataFrame, pd.Series))


def is_numpy(x):
    return isinstance(x, np.ndarray)


def save_output(obj, path, symlink=True):
    """Save output object to path.

    Parameters
    ----------
    obj: pickle-able
        Any object that can be pickled.
    path: path-like
        Path to save object to
    symlink: bool, default=True
        Save object to path's filename and sym-link path to it.
        Used with Hydra because Hydra saves files to the outputs/ dir.

    """

    orig_cwd = get_original_cwd()
    logger.trace(f"Original working directory is {orig_cwd}")

    # Example: /home/danieliong/geomag-forecasting/filename
    orig_path = Path(to_absolute_path(path))

    # Example: /home/danieliong/geomag-forecasting/outputs/{...}/filename
    output_path = Path(orig_path.name).resolve()

    if symlink:
        save_path = output_path
    else:
        save_path = orig_path

    if is_pandas(obj):
        obj.to_pickle(save_path)
    elif is_numpy(obj):
        np.save(save_path, obj)
    else:
        with open(save_path, "wb") as f:
            dill.dump(obj, f)
            # pickle.dump(obj, f)

    rel_save_path = save_path.relative_to(orig_cwd)
    rel_orig_path = orig_path.relative_to(orig_cwd)
    rel_output_path = output_path.relative_to(orig_cwd)

    logger.info(f"Saved output to {rel_save_path}.")

    if symlink:
        if orig_path.exists():
            logger.debug(f"Deleting {rel_orig_path} since it already exists...")
            logger.debug(
                f"{orig_path} was a symlink to {os.readlink(rel_orig_path)}.")
            if not orig_path.is_symlink():
                logger.warning(f"{rel_orig_path} is not a symlink.")

            orig_path.unlink()

        orig_path.symlink_to(rel_output_path)
        logger.info(f"Sym-linked {rel_output_path} to {rel_orig_path}.")
