#!/usr/bin/env python

import logging
import json
import mlflow

import numpy as np
import pandas as pd
import xgboost as xgb

from abc import ABC, abstractmethod
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class HydraModel(ABC):
    def __init__(self, cfg, mlflow=False):

        params = OmegaConf.select(cfg, "param")
        assert params is not None, "param must be provided in Hydra."
        logger.debug(OmegaConf.to_yaml(params))
        self.params = OmegaConf.to_container(params)

        # Keyword arguments used in fit.
        # Pop keys when required
        kwargs = OmegaConf.select(cfg, "kwargs")
        logger.debug(OmegaConf.to_yaml(kwargs))
        self.kwargs = OmegaConf.to_container(kwargs)

        outputs = OmegaConf.select(cfg, "outputs")
        self.outputs = OmegaConf.to_container(outputs)

        self.mlflow = mlflow

        self.model = None

    @abstractmethod
    def fit(self, cfg):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def save_output(self):
        pass


class MLFlowXGBCallback(xgb.callback.TrainingCallback):
    def __init__(self, cv=True):
        self.cv = cv
        self.run = mlflow.active_run()

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False

        if self.run is not None:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    if isinstance(log[-1], tuple):
                        score = log[-1][0]
                    else:
                        score = log[-1]
                    if self.cv:
                        key = f"cv-{data}-{metric_name}"
                    else:
                        key = f"{data}-{metric_name}"
                    mlflow.log_metric(key=key, value=score, step=epoch)
        return False


class HydraXGB(HydraModel):
    def __init__(self, cfg, mlflow=True):
        super().__init__(cfg, mlflow=mlflow)

        self.metrics = self.kwargs.pop("metrics", "rmse")

    def fit(self, X, y, cv=None):

        num_boost_round = self.kwargs.pop("num_boost_round", 100)
        early_stopping_rounds = self.kwargs.pop("early_stopping_rounds", 30)
        # metrics = self.kwargs.pop("metrics", "rmse")

        if self.mlflow:
            callbacks = [MLFlowXGBCallback()]
        else:
            callbacks = None

        dtrain = xgb.DMatrix(X, label=y)

        if cv is not None:
            self.cv_res_ = xgb.cv(
                params=self.params,
                dtrain=dtrain,
                folds=cv,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                metrics=self.metrics,
                callbacks=callbacks,
            )

            metric = (
                self.metrics[-1] if isinstance(self.metrics, list) else self.metrics
            )
            num_boost_round = np.argmin(self.cv_res_[f"test-{metric}-mean"]) + 1

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            **self.kwargs,
        )

        return self

    def predict(self, X):
        # Check fit was called successfully
        assert self.model is not None

        dtest = xgb.DMatrix(X)

        ypred = self.model.predict(dtest)
        return ypred

    def save_output(self):
        self.model.save_model(self.outputs["model"])

        if "cv_results" in self.outputs.keys():
            if self.outputs["cv_results"] is not None:
                cv_res = pd.DataFrame(self.cv_res_)
                cv_res.to_csv(self.outputs["cv_results"])

                if self.mlflow:
                    mlflow.log_artifact(self.outputs["cv_results"])


# Dictionary with model name as keys and HydraModel class as value
MODELS_DICT = {"xgboost": HydraXGB}


def get_model(name, cfg):
    model = MODELS_DICT[name](cfg)
    return model
