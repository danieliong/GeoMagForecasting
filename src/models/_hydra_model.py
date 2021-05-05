#!/usr/bin/env python

import os
import logging
import mlflow
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from src.utils import save_output
from src.plot import plot_prediction
from src.storm_utils import has_storm_index

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
            import mlflow

            run = mlflow.active_run()
            if run is not None:
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

    # @abstractmethod
    def _plot(self, X, y, **kwargs):
        # Return fig and ax you want to plot predictions on
        # Should only plot for one storm
        return None, None

    def plot(
        self,
        X,
        y,
        pdf_path="prediction_plots.pdf",
        persistence=False,
        lead=None,
        unit="minutes",
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf = PdfPages(pdf_path)

        # plot_in_pdf = not self.mlflow and pdf_path is not None
        # if plot_in_pdf:
        #     from matplotlib.backends.backend_pdf import PdfPages

        #     pdf = PdfPages(pdf_path)

        # Use last metric if there are more than one
        if isinstance(self.metrics, (list, tuple)):
            if len(self.metrics) > 1:
                metric = self.metrics[-1]
            else:
                metric = self.metrics[0]
        else:
            metric = self.metrics

        y = y.squeeze()
        ypred = self.predict(X).squeeze()

        if isinstance(ypred, np.ndarray):
            ypred = pd.Series(ypred, index=y.index, name=y.name)

        if has_storm_index(y):
            fig = []
            ax = []
            for storm in y.storms.level:
                fig_storm, ax_storm = self._plot(
                    X.storms.get(storm), y.storms.get(storm), lead=lead, **kwargs
                )
                fig_storm_, ax_storm_ = plot_prediction(
                    y,
                    ypred,
                    metric=metric,
                    storm=storm,
                    persistence=persistence,
                    lead=lead,
                    unit=unit,
                    ax=ax_storm,
                )

                if fig_storm is None:
                    fig_storm = fig_storm_
                    ax_storm = ax_storm_

                fig.append(fig_storm)
                ax.append(ax_storm)
                pdf.savefig(fig_storm, bbox_inches="tight")

                # if self.mlflow:
                #     mlflow.log_figure(fig_storm, f"prediction_plots/storm_{storm}.png")
                # elif plot_in_pdf:
                #     pdf.savefig(fig_storm)
        else:
            fig, ax = self._plot(X, y, **kwargs)
            _ = plot_prediction(
                y,
                ypred,
                metric=metric,
                persistence=persistence,
                lead=lead,
                unit=unit,
                ax=ax,
            )

            pdf.savefig(fig, bbox_inches="tight")

            # if self.mlflow:
            #     mlflow.log_figure(fig, "prediction_plot.png")
            # elif plot_in_pdf:
            #     pdf.savefig(fig, bbox_inches="tight")

        pdf.close()
        if self.mlflow:
            mlflow.log_artifact(pdf_path)

        return fig, ax
