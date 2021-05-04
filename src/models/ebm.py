#!/usr/bin/env python

import logging
import pickle
import mlflow
import re
import src
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlflow.models import Model
from pathlib import Path
from interpret.glassbox import ExplainableBoostingRegressor

from src.utils import save_output, get_offset, infer_freq
from src.preprocessing.load import load_processor
from ._hydra_model import HydraModel
from src._train import compute_metrics
from src.plot import plot_prediction
from src.storm_utils import has_storm_index

MLFLOW_FLAVOR_NAME = "ebm"

logger = logging.getLogger(__name__)

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

    def _compute_aggregated_scores(self, scores):

        re_features_names = np.unique(
            [re.sub("_[0-9]+", "_[0-9]+", col) for col in scores.columns]
        )

        contrib_df = pd.concat(
            (scores.filter(regex=f"^{r}$").sum(axis=1) for r in re_features_names),
            axis=1,
        )
        contrib_df.rename(
            columns={
                i: x.replace("_[0-9]+", "") for i, x in enumerate(re_features_names)
            },
            inplace=True,
        )

        return contrib_df

    def compute_scores(self, X, y, aggregate=True):
        ebm_local = self.model.explain_local(X, y)

        scores_df = pd.DataFrame(
            [ebm_local.data(i)["scores"] for i in range(y.shape[0])],
            columns=ebm_local.data(0)["names"],
            index=y.index,
        )
        scores_df["intercept"] = ebm_local.data(0)["extra"]["scores"][0]

        if aggregate:
            scores_df = self._compute_aggregated_scores(scores_df)

        return scores_df

    def _plot(
        self,
        X,
        y,
        plot_features=None,
        score_threshold=0.9,
        lead=None,
        unit="minutes",
        figsize=(17, 12),
        linewidth=0.7,
        gridspec_kw={},
        sharex=True,
        **kwargs,
    ):

        agg_scores = self.compute_scores(X, y, aggregate=True)

        if plot_features is not None:
            assert "height_ratios" in gridspec_kw.keys()

        if plot_features is None:
            mean_abs_scores = agg_scores.abs().mean().sort_values(ascending=False)
            # mean_abs_scores.drop(["intercept", "y"], errors="ignore", inplace=True)
            mean_abs_scores.drop(["intercept"], errors="ignore", inplace=True)

            contrib_plot_features = mean_abs_scores.index[:5].tolist()
            subplot_features = contrib_plot_features.copy()
            try:
                subplot_features.remove("y")
            except ValueError:
                pass
            # plot_features = mean_abs_scores.drop(["intercept"], errors="ignore").index[
            #     :5
            # ]

            perc_scores = mean_abs_scores / mean_abs_scores.sum()
            # plot_features = []
            # # Get features that have cumulative scores > threshold
            # for feature in perc_scores.index:
            #     plot_features.append(feature)
            #     if perc_scores[feature] > 0.9:
            #         break

        if "height_ratios" not in gridspec_kw.keys():
            gridspec_kw["height_ratios"] = [1, 1] + [1] * len(subplot_features)
        elif len(gridspec_kw["height_ratios"]) != len(subplot_features):
            height_ratios = [1, 1] + [1] * len(subplot_features)
            logger.debug(
                f"Height_ratios not equal to number of features to plot. Defaulting to {height_ratios}"
            )
            gridspec_kw["height_ratios"] = height_ratios

        fig, ax = plt.subplots(
            nrows=2 + len(subplot_features),
            figsize=figsize,
            gridspec_kw=gridspec_kw,
            sharex=sharex,
            **kwargs,
        )
        plt.rcParams["font.size"] = "9"
        colors = [plt.cm.Set1(i) for i in range(9)]
        # locator = mdates.AutoDateLocator(minticks=5)
        # formatter = mdates.ConciseDateFormatter(locator)
        # for x in ax:
        #     x.xaxis.set_major_locator(locator)
        #     x.xaxis.set_major_formatter(formatter)

        # Shift X back to original times
        # Times in X are with respect to times to predict
        lead = get_offset(lead)
        features = X.filter(regex="._0$").shift(periods=-1, freq=lead)
        features.rename(columns=lambda col: re.sub("_0", "", col), inplace=True)

        idx = features.index.intersection(agg_scores.index)
        features, agg_scores = features.loc[idx], agg_scores.loc[idx]

        # HACK: Resample to get consistent frequency
        # This is because the formatting for dates gets mixed up if frequency is inconsistent
        freq = infer_freq(features)
        features = features.resample(freq).interpolate()
        agg_scores = agg_scores.resample(freq).interpolate()

        # Plot contributions on one plot
        for i, feature in enumerate(contrib_plot_features):
            agg_scores[feature].plot(
                ax=ax[1], label=feature, linewidth=linewidth, color=colors[i]
            )
        ax[1].legend(ncol=7, prop={"size": 7})
        perc_contrib = round(perc_scores[contrib_plot_features].sum(), 2)
        y_label = f"Contributions ({perc_contrib*100}%)"
        ax[1].set_ylabel(y_label)
        # ax[1].xaxis.set_major_locator(locator)
        # ax[1].xaxis.set_major_formatter(formatter)

        # Plot individual features and their contributions in subplots
        for i, feature in enumerate(subplot_features, start=2):
            interactions = re.split(" x ", feature)
            if len(interactions) > 1:
                feat_to_plot = features[interactions].product(axis=1)
                ax_i = ax[i]
                ax_i.yaxis.tick_right()
            elif feature != "y":
                feat_to_plot = features[feature]

                feat_to_plot.plot(
                    ax=ax[i], color="black", label=feature, linewidth=linewidth,
                )
                ax_i = ax[i].twinx()
            else:
                # Don't plot y in subplot
                continue

            perc_contrib_i = round(perc_scores[feature], 2)
            y_label_i = f"{feature} ({perc_contrib_i*100}%)"
            ax[i].set_ylabel(y_label_i)
            color = colors[i - 1] if "y" in contrib_plot_features else colors[i - 2]
            agg_scores[feature].plot(
                ax=ax_i, color=color, label="Contrib.", linewidth=linewidth
            )
            ax_i.axhline(linestyle="dashed", linewidth=linewidth, color="grey")

            # ax[i].xaxis.set_major_locator(locator)
            # ax[i].xaxis.set_major_formatter(formatter)

            # ax[i].legend()

        # ax[-1].xaxis.set_major_locator(locator)
        # ax[-1].xaxis.set_major_formatter(formatter)
        ax[-1].set_xlabel("")
        ax[-1].set_xlim(left=idx[0], right=idx[-1])

        fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)
        # fig.tight_layout()

        # Return first axis to plot predictions
        return fig, ax[0]

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
