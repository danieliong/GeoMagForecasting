#!/usr/bin/env python

import mlflow
from interpret.glassbox import ExplainableBoostingRegressor
from src.utils import save_output
from src.preprocessing.load import load_processor
from ._hydra_model import HydraModel


class EBMWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from interpret.glassbox import ExplainableBoostingRegressor

        self.model = load_processor(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)


class HydraEBM(HydraModel):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._setup_mlflow()
        self.model = ExplainableBoostingRegressor(**self.params)
        self.python_model = EBMWrapper()
        self._fitted = False

    def _setup_mlflow(self):
        # TODO
        import mlflow

        mlflow.log_params(self.params)

    def fit(self, X, y, feature_names=None):
        self.feature_names_ = feature_names
        self.model.set_params(feature_names=feature_names)

        # NOTE: Use my fork of interpret library. I modified fit to take validation indices
        validation_indices = self.cv[-1][1]
        self.model.fit(X, y, validation_indices=validation_indices, **self.kwargs)

        self._fitted = True

    def predict(self, X):
        return self.model.predict(X)

    def _save_output(self):
        assert self._fitted
        save_output(self.model, self.model_path)

        impt_plot_path = "importance_plot.html"
        ebm_global = self.model.explain_global()
        ebm_global.visualize().write_html(impt_plot_path)
        mlflow.log_artifact(impt_plot_path)

        # if self.mlflow:
        #     # TODO: Add conda_env
        #     artifacts = {"model": self.model_path}
        #     mlflow.pyfunc.save_model(
        #         path="model", python_model=EBMWrapper(), artifacts=artifacts
        #     )

    def load_model(self, path):
        # INCOMPLETE
        return load_processor(path)

    def plot_interpret(self):
        pass
