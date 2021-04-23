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
from src.preprocessing.load import load_processor

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
        self.model = None
        self.cv = cv

    @property
    def cv_metric(self):
        return self.metrics[-1] if isinstance(self.metrics, list) else self.metrics

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
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.model = ExplainableBoostingRegressor(**self.params)

    def fit(self, X, y, feature_names=None):
        # TODO: Look into how interpret does CV
        self.feature_names_ = feature_names
        self.model.set_params(feature_names=feature_names)

        validation_indices = self.cv[-1][1]
        self.model.fit(X, y, validation_indices=validation_indices, **self.kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def save_output(self):
        model_path = self.outputs.get("model", None)
        if model_path is not None:
            save_output(self.model, self.outputs["model"])

    def load_model(self, path):
        return load_processor(path)

    def plot_interpret(self):
        pass


class HydraXGB(HydraModel):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # self.metrics = self.kwargs.pop("metrics", "rmse")

    def cross_validate(self, dtrain):
        num_boost_round = getattr(self.kwargs, "num_boost_round", 100)
        early_stopping_rounds = getattr(self.kwargs, "early_stopping_rounds", 30)

        if self.mlflow:
            callbacks = [MLFlowXGBCallback()]
        else:
            callbacks = None

        cv_res = xgb.cv(
            params=self.params,
            dtrain=dtrain,
            folds=self.cv,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            metrics=self.metrics,
            callbacks=callbacks,
        )

        return cv_res

    def _get_optimal_num_trees(self):
        assert hasattr(self, "cv_res_")
        return np.argmin(self.cv_res_[f"test-{self.cv_metric}-mean"]) + 1

    def fit(self, X, y, feature_names=None):

        self.feature_names_ = feature_names

        # metrics = self.kwargs.pop("metrics", "rmse")

        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names_)

        if self.cv is not None and not hasattr(self, "cv_res_"):
            self.cv_res_ = self.cross_validate(dtrain)

        kwargs = self.kwargs.copy()
        num_boost_round = kwargs.pop("num_boost_round")
        _ = kwargs.pop("early_stopping_rounds")

        if hasattr(self, "cv_res_"):
            num_boost_round = self._get_optimal_num_trees()

        # if self.cv is not None:
        #     if not hasattr(self, "cv_res_"):
        #         self.cv_res_ = self.cross_validate(dtrain)

        #     num_boost_round = np.argmin(self.cv_res_[f"test-{self.cv_metric}-mean"]) + 1

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            **kwargs,
        )

        return self

    def cv_score(self, X, y):
        # assert self.cv is not None, "cv must be specified."
        assert self.cv is not None or hasattr(self, "cv_res_")

        if not hasattr(self, "cv_res_"):
            dtrain = xgb.DMatrix(X, label=y)
            self.cv_res_ = self.cross_validate(dtrain)

        return float(min(self.cv_res_[f"test-{self.cv_metric}-mean"]))

    def predict(self, X):
        # Check fit was called successfully
        assert self.model is not None

        dtest = xgb.DMatrix(X, feature_names=self.feature_names_)

        ypred = self.model.predict(dtest)
        return ypred

    def save_output(self):
        if self.model is not None:
            self.model.save_model(self.outputs["model"])

        if "cv_table" in self.outputs.keys():
            if self.outputs["cv_table"] is not None:
                cv_tbl = pd.DataFrame(self.cv_res_)
                cv_tbl.to_csv(self.outputs["cv_table"])

                if self.mlflow:
                    mlflow.log_artifact(self.outputs["cv_table"])

        if "cv_results" in self.outputs.keys():
            cv_res = OmegaConf.create(
                {"num_boost_rounds": int(self._get_optimal_num_trees())}
            )
            OmegaConf.save(cv_res, self.outputs["cv_results"])


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
