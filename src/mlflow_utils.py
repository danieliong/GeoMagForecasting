#!/usr/bin/env python

import mlflow
import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf
from hydra.experimental import initialize
from src.utils import get_features_cfg
from src.preprocessing.load import load_processor
from src.models.ebm import HydraEBM
from src.models import get_model


class MLFlowRun:
    def __init__(self, run=None, run_id=None):
        assert run is not None or run_id is not None
        self.run = run
        self.run_id = run_id

        if self.run_id is None:
            self.run_id = self.run.info.run_id
        elif self.run is None:
            self.run = mlflow.get_run(self.run_id)

    @property
    def artifacts_dir(self):
        return Path(self.run.info.artifact_uri.replace("file://", ""))

    @property
    def configs(self):
        model_configs_dir = self.artifacts_dir / "model_configs"
        model_cfg = OmegaConf.load(model_configs_dir / "config.yaml")
        hydra_cfg = OmegaConf.load(model_configs_dir / "hydra.yaml")
        return OmegaConf.merge(model_cfg, hydra_cfg)

    @property
    def features_configs(self):
        with initialize(config_path="../configs"):
            features_cfg = get_features_cfg(self.configs)

        return features_cfg

    def get_pyfunc_model(self):
        return mlflow.pyfunc.load_model(f"runs:/{self.run_id}/model")

    def get_hydra_model(self):
        pyfunc_model = self.get_pyfunc_model()
        model_path = pyfunc_model._model_impl.context.artifacts["model"]
        model_name = self.run.data.params["model"]

        hydra_model = get_model(model_name)(self.configs)
        hydra_model.model = load_processor(model_path)

        return hydra_model

    def plot_predictions(self, **kwargs):
        hydra_model = self.get_hydra_model()
        X_test, y_test = self.get_test_data(**kwargs)
        fig, ax = hydra_model.plot(
            X=X_test,
            y=y_test,
            lead=self.features_configs.lead,
            unit=self.features_configs.lag_processor.unit,
            **self.configs.plot,
        )

        return fig, ax

    def get_test_data(self, X_path="X_test.pkl", y_path="y_test.pkl"):
        with initialize(config_path="../configs"):
            features_cfg = get_features_cfg(self.configs)

        features_dir = Path(features_cfg.hydra.run.dir).resolve()
        X_test = pd.read_pickle(features_dir / X_path)
        y_test = pd.read_pickle(features_dir / y_path).squeeze()
        return X_test, y_test
