#!/usr/bin/env python

import os
import logging
import mlflow

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from src.utils import save_output

logger = logging.getLogger(__name__)


class HydraModel(ABC):
    def __init__(self, cfg, metrics="rmse", cv=None, mlflow=False):
        # cfg is from entire yaml file for a specific model (e.g. xgboost.yaml)

        # Required
        params = OmegaConf.select(cfg, "param")
        assert params is not None, "param must be provided in Hydra."
        logger.debug("\n" + OmegaConf.to_yaml(params))
        self.params = OmegaConf.to_container(params)
        # Keyword arguments in model class for sklearn or param dict for XGB

        # Keyword arguments used in fit.
        # Pop keys when required
        # Can be None
        kwargs = OmegaConf.select(cfg, "kwargs")
        if kwargs is not None:
            logger.debug("\n" + OmegaConf.to_yaml(kwargs))
            # Convert to dict
            self.kwargs = OmegaConf.to_container(kwargs)
        else:
            logger.debug("No kwargs were passed.")
            self.kwargs = {}

        # Required
        outputs = OmegaConf.select(cfg, "outputs")
        assert outputs is not None
        self.outputs = OmegaConf.to_container(outputs)

        # self.metrics = OmegaConf.select(cfg, "metrics")
        self.metrics = metrics

        self.mlflow = mlflow
        # Initiate model
        self.cv = cv
        self.model = None
        self.python_model = None

        self.model_artifacts_path = "model"
        self.model_artifacts = {"model": self.model_path}
        self.conda_env = None
        self.mlflow_kwargs = {}
        if self.mlflow:
            self._setup_mlflow()

    @property
    def cv_metric(self):
        return self.metrics[-1] if isinstance(self.metrics, list) else self.metrics

    @property
    def model_path(self):
        return self.outputs.get("model", "model.pkl")

    @property
    def mlflow_path(self):
        return "model"

    @abstractmethod
    def _setup_mlflow(self):
        # - Logging params
        # - Set up autologging
        pass

    @abstractmethod
    def _save_output(self):
        pass

    def save_output(self):
        self._save_output()
        if self.mlflow and self.python_model is not None:
            # NOTE: If autologging, don't specify python_model
            mlflow.pyfunc.log_model(
                artifact_path=self.model_artifacts_path,
                python_model=self.python_model,
                artifacts=self.model_artifacts,
                conda_env=self.conda_env,
                **self.mlflow_kwargs,
            )

    # @property
    # @abstractmethod
    # def model(self):
    #     pass

    @abstractmethod
    def fit(self, X, y, cv=None, feature_names=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def cv_score(self, X, y):
        pass

    # TODO: Implement general CV
