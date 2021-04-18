#!/usr/bin/env python

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.metrics import mean_squared_error

from src import utils
from src.preprocessing.load import load_processed_data, load_processor
from src.storm_utils import has_storm_index, StormIndexAccessor, StormAccessor


logger = logging.getLogger(__name__)


def get_cv_split(y, method, init_params, **load_kwargs):

    # Returns None if group_labels doesn't exist (e.g. for cv=timeseries)
    # groups = load_processed_data(name="group_labels", **load_kwargs)

    if method == "timeseries":
        splitter = TimeSeriesSplit(**init_params)
        groups = None
    elif method == "storms":
        splitter = GroupKFold(**init_params)
        # groups = load_processed_data(
        #     name="group_labels", must_exist=True, **load_kwargs
        # )

        # NOTE: y should be a pandas object with storm index so set groups to be storm index
        groups = y.storms.index

        # Reindex times within each storm
        # Cannot just reindex groups with y index directly because there are
        # overlapping storms
        # groups = pd.concat(
        #     (
        #         groups[groups[STORM_LEVEL] == storm].reindex(y.storms.get(storm).index)
        #         for storm in y.storms.level
        #     )
        # )

        # assert len(groups) == len(
        #     y
        # ), f"Length of groups ({len(groups)}) does not match length of y ({len(y)})"

    split = splitter.split(y, groups=groups)

    return list(split)


# TODO: Add more general version to utils later.
def convert_pred_to_pd(ypred, y):

    if isinstance(ypred, (pd.Series, pd.DataFrame)):
        logger.debug(
            "Predictions are already pandas objects and will not be converted."
        )
        return ypred

    if isinstance(ypred, np.ndarray):
        # NOTE: This should be okay but check to make sure later.
        if isinstance(y, pd.DataFrame):
            ypred = pd.DataFrame(ypred, columns=y.columns, index=y.index)
        elif isinstance(y, pd.Series):
            ypred = pd.Series(ypred, name=y.name, index=y.index)

    return ypred


def inv_transform_targets(y, ypred, path, processor_dir):

    logger.debug("Inverse transforming y and predictions...")

    target_pipeline = load_processor(path, inputs_dir=processor_dir)
    y = target_pipeline.inverse_transform(y)
    ypred = target_pipeline.inverse_transform(ypred)

    return y, ypred


def compute_metric(y, ypred, metric, storm=None):
    # QUESTION: Should we inverse transform y and ypred before
    # computing metrics?

    if metric is None:
        logger.debug("Metric not given. Defaulting to rmse.")
        # Default to rmse
        metric = "rmse"

    if storm is None:
        logger.debug(f"Computing {metric}...")
    else:
        logger.debug(f"Computing {metric} for storm {storm}.")

    if metric == "rmse":
        metric_val = mean_squared_error(y, ypred, squared=False)
    elif metric == "mse":
        metric_val = mean_squared_error(y, ypred, squared=True)
    else:
        raise utils.NotSupportedError(metric, name_type="Metric")

    if storm is None:
        logger.info(f"{metric}: {metric_val}")
    else:
        logger.info(f"[Storm {storm}] {metric}: {metric_val}")

    return float(metric_val)


def compute_metrics(y, ypred, metrics, storm=None):

    if isinstance(metrics, (list, tuple)):
        if len(metrics) > 1:
            return {
                metric: compute_metric(y, ypred, metric, storm=storm)
                for metric in metrics
            }

    return compute_metric(y, ypred, metrics, storm=storm)


def plot_predictions(
    y, ypred, metrics, use_mlflow=True, pdf_path="prediction_plots.pdf"
):
    if use_mlflow:
        import mlflow
    else:
        # Save plots into pdf instead
        from matplotlib.backends.backend_pdf import PdfPages

        pdf = PdfPages(pdf_path)

    # Use last metric if there are more than one
    if isinstance(metrics, (list, tuple)):
        if len(metrics) > 1:
            metric = metrics[-1]
        else:
            metric = metrics[0]
    else:
        metric = metrics

    y = y.squeeze()
    ypred = ypred.squeeze()

    if has_storm_index(y):
        # Plot predictions for each storm individually
        for storm in y.storms.level:
            fig, ax = _plot_prediction(y, ypred, metric, storm=storm)

            if use_mlflow:
                mlflow.log_figure(fig, f"prediction_plots/storm_{storm}.png")
            else:
                pdf.savefig(fig)
    else:
        fig, ax = _plot_prediction(y, ypred, metric)
        if use_mlflow:
            mlflow.log_figure(fig, "prediction_plot.png")
        else:
            pdf.savefig(fig)

    if not use_mlflow:
        pdf.close()

    return fig, ax


def _plot_prediction(y, ypred, metric, storm=None):
    if storm is None:
        metric_val = compute_metric(y, ypred, metric, storm=None)
    else:
        metric_val = compute_metric(y.loc[storm], ypred.loc[storm], metric, storm=storm)

    metric_val = round(metric_val, ndigits=3)

    fig, ax = plt.subplots(figsize=(15, 10))
    y.plot(ax=ax, color="black", linewidth=0.7)
    ypred.plot(ax=ax, color="red", linewidth=0.7)

    ax.legend(["Truth", "Prediction"])
    if storm is not None:
        ax.set_title(f"Storm: {storm} [{metric}: {metric_val}]")
    ax.set_xlabel("")

    return fig, ax


def compute_lagged_features(lag, exog_lag, lead, save_dir):
    import src.compute_lagged_features as lf

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = to_absolute_path("configs/compute_lagged_features.yaml")
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "lag", lag, merge=False)
    OmegaConf.update(cfg, "exog_lag", exog_lag, merge=False)
    OmegaConf.update(cfg, "lead", lead, merge=False)

    for name, path in cfg.outputs.items():
        OmegaConf.update(cfg.outputs, name, str(save_dir / path))

    lf.compute_lagged_features(cfg)
