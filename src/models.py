#!/usr/bin/env python

import logging
import json
import mlflow
import interpret

import numpy as np
import pandas as pd
import xgboost as xgb

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from interpret.glassbox import ExplainableBoostingRegressor

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
            self.kwargs = None

        # Required
        outputs = OmegaConf.select(cfg, "outputs")
        assert outputs is not None
        self.outputs = OmegaConf.to_container(outputs)

        # self.metrics = OmegaConf.select(cfg, "metrics")
        self.metrics = metrics

        self.mlflow = mlflow
        # Initiate model
        self.model = None
        self.cv = cv

    @abstractmethod
    def fit(self, X, y, cv=None, feature_names=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def cv_score(self, X, y):
        pass

    @abstractmethod
    def save_output(self):
        pass

    # TODO: Implement general CV


class HydraEBM(HydraModel):
    def __init__(self, cfg, cv=None):
        super().__init__(cfg, cv=cv)
        self.model = ExplainableBoostingRegressor(**self.params)

    def fit(self, X, y, feature_names=None):
        # TODO: Look into how interpret does CV
        self.feature_names_ = feature_names
        self.model.set_params(feature_names=feature_names)

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_output(self):
        model_path = getattr(self.outputs, "model", None)
        if model_path is not None:
            save_output(self.model, self.outputs["model"])


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
    def __init__(self, cfg, cv=None, mlflow=True):
        super().__init__(cfg, cv=cv, mlflow=mlflow)
        # self.metrics = self.kwargs.pop("metrics", "rmse")

    def fit(self, X, y, feature_names=None):

        self.feature_names_ = feature_names

        num_boost_round = self.kwargs.pop("num_boost_round", 100)
        early_stopping_rounds = self.kwargs.pop("early_stopping_rounds", 30)
        # metrics = self.kwargs.pop("metrics", "rmse")

        if self.mlflow:
            callbacks = [MLFlowXGBCallback()]
        else:
            callbacks = None

        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names_)

        if self.cv is not None:
            self.cv_res_ = xgb.cv(
                params=self.params,
                dtrain=dtrain,
                folds=self.cv,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                metrics=self.metrics,
                callbacks=callbacks,
            )

            self.metric_ = (
                self.metrics[-1] if isinstance(self.metrics, list) else self.metrics
            )
            num_boost_round = np.argmin(self.cv_res_[f"test-{self.metric_}-mean"]) + 1

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            **self.kwargs,
        )

        return self

    def cv_score(self, X=None, y=None):
        # For XGB, CV was done in fit to choose number of trees
        # so just return results from that
        return float(min(self.cv_res_[f"test-{self.metric_}-mean"]))

    def predict(self, X):
        # Check fit was called successfully
        assert self.model is not None

        dtest = xgb.DMatrix(X, feature_names=self.feature_names_)

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
MODELS_DICT = {"xgboost": HydraXGB, "ebm": HydraEBM}


def get_model(name):
    return MODELS_DICT[name]
    # model = MODELS_DICT[name](cfg, **kwargs)
    # return model
