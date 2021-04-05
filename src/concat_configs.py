#!/usr/bin/env python

from dotenv import dotenv_values
from pathlib import Path
from omegaconf import OmegaConf
from hydra.experimental import initialize_config_dir, compose

STAGES = ["process_data", "compute_lagged_features", "train"]
CONFIG_DIR = dotenv_values().get("CONFIG_DIR")

# cfg = OmegaConf.create()

with initialize_config_dir(CONFIG_DIR):
    for stage in STAGES:
        cfg = compose(config_name=stage, return_hydra_config=True)
        OmegaConf.save(cfg, f"params/{stage}.yaml")
