#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from abc import ABC, abstractmethod
from omegaconf import OmegaConf



class HydraModel(ABC):

    def __init__(self, cfg):

        params = OmegaConf.select(cfg, "param")
        assert params is not None, "param must be provided in Hydra."
        self.params = OmegaConf.to_container(params)

        # Keyword arguments used in fit.
        # Pop keys when required
        kwargs = OmegaConf.select(cfg, "kwargs")
        self.kwargs = OmegaConf.to_container(kwargs)

        outputs = OmegaConf.select(cfg, "outputs")
        self.outputs = OmegaConf.to_container(outputs)

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


class HydraXGB(HydraModel):

    def __init__(self, cfg):
        super().__init__(cfg)


    def fit(self, X, y, cv=None):

        num_boost_round = self.kwargs.pop('num_boost_round', 100)
        early_stopping_rounds = self.kwargs.pop('early_stopping_rounds', 30)
        metrics = self.kwargs.pop('metrics', "rmse")

        dtrain = xgb.DMatrix(X, label=y)

        if cv is not None:
            self.cv_res_ = xgb.cv(
                params=self.params,
                dtrain=dtrain,
                folds=cv,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                metrics=metrics)

            metric = metrics[-1] if isinstance(metrics, list) else metrics
            num_boost_round = np.argmin(self.cv_res_[f'test-{metric}-mean']) + 1

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            **self.kwargs
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


# Dictionary with model name as keys and HydraModel class as value
MODELS_DICT = {"xgboost": HydraXGB}


def get_model(name, cfg):
    model = MODELS_DICT[name](cfg)
    return model
