#!/usr/bin/env python

import pickle
import mlflow
import src
import matplotlib.pyplot as plt

from mlflow.models import Model
from pathlib import Path
from interpret.glassbox import ExplainableBoostingRegressor

from src.utils import save_output
from src.preprocessing.load import load_processor
from ._hydra_model import HydraModel
from src._train import compute_metrics
from src.plot import plot_prediction
from src.storm_utils import has_storm_index

MLFLOW_FLAVOR_NAME = "ebm"


# INCOMPLETE
# Do later
def save_model(
    ebm_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature=None,
    input_example=None,
):
    import interpret

    path = Path(path).resolve()
    if path.exists():
        raise mlflow.exceptions.MlFlowException(f"Path {path} already exists.")
    path.mkdir(parents=True)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        # TODO
        pass

    model_data_subpath = "model.pkl"
    model_data_path = path / model_data_subpath

    with open(model_data_path, "wb") as f:
        pickle.dump(f)

    # TODO: Conda env

    mlflow.pyfunc.add_to_model(
        mlflow_model, loader_module="src.models.ebm", data=model_data_subpath
    )
    mlflow_model.add_flavor(MLFLOW_FLAVOR_NAME, data=model_data_subpath)
    mlflow_model.save(path / "MLmodel")


# INCOMPLETE
# Do later
def log_model(
    ebm_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature=None,
    input_example=None,
    **kwargs,
):
    Model.log(
        artifact_path=artifact_path,
        flavor=src.models.ebm,
        registered_model_name=registered_model_name,
    )


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

    # def score(self, X, y):
    #     ypred = self.predict(X)
    #     score = compute_metrics(y, ypred, metric=self.metrics)

    #     # if self.mlflow:
    #     #     ebm_global = self.model.explain_global()
    #     #     ebm_local = self.model.explain_local(X, y)

    #     return score

    def _plot(self, X, y):
        # TODO

        # explanation_path = "ebm_local.pkl"
        # ebm_local = self.model.explain_local(X, y)
        # with open(explanation_path, "wb") as f:
        #     pickle.dump(ebm_local, f)

        return None, None

    def _save_output(self):
        assert self._fitted
        save_output(self.model, self.model_path)

        impt_plot_path = "importance_plot.html"
        explanation_path = "ebm_global.pkl"
        ebm_global = self.model.explain_global()

        ebm_global.visualize().write_html(impt_plot_path)
        with open(explanation_path, "wb") as f:
            pickle.dump(ebm_global, f)

        mlflow.log_artifact(impt_plot_path)
        mlflow.log_artifact(explanation_path)

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
