#!/usr/bin/env python

import mlflow
import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf
from hydra.experimental import initialize
from src.utils import get_features_cfg
from src.preprocessing.load import load_processor
from src.models.ebm import HydraEBM


exp_id = mlflow.get_experiment_by_name("ebm").experiment_id
ebm_runs = mlflow.search_runs(exp_id)

for run_id in ebm_runs["run_id"].values:
    print(f"run_id: {run_id}")
    with mlflow.start_run(run_id=run_id):
        artifacts_uri = mlflow.get_artifact_uri()
        artifacts_dir = Path(artifacts_uri.replace("file://", ""))

        model_configs_dir = artifacts_dir / "model_configs"
        model_cfg = OmegaConf.load(model_configs_dir / "config.yaml")
        hydra_cfg = OmegaConf.load(model_configs_dir / "hydra.yaml")
        cfg = OmegaConf.merge(model_cfg, hydra_cfg)
        with initialize(config_path="../configs"):
            features_cfg = get_features_cfg(cfg)

        features_dir = Path(features_cfg.hydra.run.dir)
        X_test = pd.read_pickle(features_dir / "X_test.pkl")
        y_test = pd.read_pickle(features_dir / "y_test.pkl")

        pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        model_path = pyfunc_model._model_impl.context.artifacts["model"]

        hydra_model = HydraEBM(cfg, mlflow=True)
        hydra_model.model = load_processor(model_path)

        hydra_model.plot(
            X_test,
            y_test,
            lead=features_cfg.lead,
            unit=features_cfg.lag_processor.unit,
            **cfg.plot,
        )
