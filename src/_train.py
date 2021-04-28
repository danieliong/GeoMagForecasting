#!/usr/bin/env python

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from pandas.tseries.frequencies import to_offset

# from sklearn.metrics import mean_squared_error

from src import STORM_LEVEL
from src import utils
from src.preprocessing.load import load_processed_data, load_processor
from src.storm_utils import has_storm_index, StormIndexAccessor, StormAccessor


logger = logging.getLogger(__name__)

DATA_CONFIGS_TO_LOG = {
    "start": "start",
    "end": "end",
    "target": "target.name",
    "features_source": "features.name",
    "features": "features.load.features",
    "target_processing": "target.pipeline.order",
    "features_processing": "features.pipeline.order",
}


def get_cv_split(y, method, **init_params):

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


def mean_squared_error(y, ypred, *, squared=False):
    mse = np.mean((y - ypred) ** 2)
    if not squared:
        return np.sqrt(mse)

    return mse


def compute_metric(y, ypred, metric, storm=None):
    # QUESTION: Should we inverse transform y and ypred before
    # computing metrics?

    if storm is None:
        y_, ypred_ = y, ypred
    else:
        y_, ypred_ = y.storms.get(storm), ypred.storms.get(storm)

    if metric is None:
        logger.debug("Metric not given. Defaulting to rmse.")
        # Default to rmse
        metric = "rmse"

    if storm is None:
        logger.debug(f"Computing {metric}...")
    else:
        logger.debug(f"Computing {metric} for storm {storm}.")

    if metric == "rmse":
        metric_val = mean_squared_error(y_, ypred_, squared=False)
    elif metric == "mse":
        metric_val = mean_squared_error(y_, ypred_, squared=True)
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


# def plot_predictions(
#     y,
#     ypred,
#     metrics,
#     use_mlflow=False,
#     pdf_path="prediction_plots.pdf",
#     persistence=False,
#     lead=None,
#     unit="minutes",
# ):
#     if use_mlflow:
#         import mlflow
#     elif pdf_path is not None:
#         # Save plots into pdf instead
#         from matplotlib.backends.backend_pdf import PdfPages

#         pdf = PdfPages(pdf_path)

#     # Use last metric if there are more than one
#     if isinstance(metrics, (list, tuple)):
#         if len(metrics) > 1:
#             metric = metrics[-1]
#         else:
#             metric = metrics[0]
#     else:
#         metric = metrics

#     y = y.squeeze()
#     ypred = ypred.squeeze()

#     if has_storm_index(y):
#         # Plot predictions for each storm individually
#         for storm in y.storms.level:
#             fig, ax = _plot_prediction(
#                 y,
#                 ypred,
#                 metric,
#                 storm=storm,
#                 persistence=persistence,
#                 lead=lead,
#                 unit=unit,
#             )

#             if use_mlflow:
#                 mlflow.log_figure(fig, f"prediction_plots/storm_{storm}.png")
#             elif pdf_path is not None:
#                 pdf.savefig(fig)
#     else:
#         fig, ax = _plot_prediction(
#             y, ypred, metric, persistence=persistence, lead=lead, unit=unit
#         )
#         if use_mlflow:
#             mlflow.log_figure(fig, "prediction_plot.png")
#         elif pdf_path is not None:
#             pdf.savefig(fig)

#     if not use_mlflow:
#         pdf.close()

#     return fig, ax


# def _plot_prediction(y, ypred, metric, storm, persistence, lead, unit):
#     # TODO: Plot persistence predictions

#     if storm is None:
#         metric_val = compute_metric(y, ypred, metric, storm=None)
#         y_, ypred_ = y, ypred
#     else:
#         metric_val = compute_metric(y, ypred, metric, storm=storm)
#         y_, ypred_ = y.storms.get(storm), ypred.storms.get(storm)

#     if isinstance(lead, str):
#         lead = to_offset(lead)
#     elif isinstance(lead, int):
#         lead = to_offset(pd.Timedelta(**{unit: lead}))

#     metric_val = round(metric_val, ndigits=3)

#     fig, ax = plt.subplots(figsize=(15, 10))
#     y_.plot(ax=ax, color="black", linewidth=0.7)
#     ypred_.plot(ax=ax, color="red", linewidth=0.7)
#     legend = ["Truth", f"Prediction [{metric}: {metric_val}]"]

#     if persistence:
#         assert lead is not None, "Lead must be specified if persistence=True"
#         ypers = y_.shift(periods=1, freq=lead)
#         pers_metric = compute_metric(y_, ypers, metric, storm=None)
#         pers_metric = round(pers_metric, ndigits=3)
#         ypers.plot(ax=ax, color="blue", linewidth=0.5, linestyle="dashed")
#         legend.append(f"Persistence [{metric}: {pers_metric}]")

#     ax.legend(legend)
#     if storm is not None:
#         ax.set_title(f"Storm #{storm}")
#     ax.set_xlabel("")

#     return fig, ax


def _compute_lagged_features(lag, exog_lag, lead, save_dir, overrides=None):
    import src.compute_lagged_features as lf

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # FIXME: Doesn't account for overrides
    config_path = to_absolute_path("configs/compute_lagged_features.yaml")
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "lag", lag, merge=False)
    OmegaConf.update(cfg, "exog_lag", exog_lag, merge=False)
    OmegaConf.update(cfg, "lead", lead, merge=False)

    for name, path in cfg.outputs.items():
        OmegaConf.update(cfg.outputs, name, str(save_dir / path))

    lf.compute_lagged_features(cfg)


def compute_lagged_features(lag, exog_lag, lead, save_dir, overrides=None):
    import src.compute_lagged_features as lf

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # FIXME: Doesn't account for overrides
    config_path = to_absolute_path("configs/compute_lagged_features.yaml")
    cfg = OmegaConf.load(config_path)
    OmegaConf.update(cfg, "lag", lag, merge=False)
    OmegaConf.update(cfg, "exog_lag", exog_lag, merge=False)
    OmegaConf.update(cfg, "lead", lead, merge=False)

    for name, path in cfg.outputs.items():
        OmegaConf.update(cfg.outputs, name, str(save_dir / path))

    lf.compute_lagged_features(cfg)


def predict_persistence(y, lead, unit="minutes"):

    if isinstance(lead, str):
        lead = to_offset(lead)
    elif isinstance(lead, int):
        lead = to_offset(pd.Timedelta(**{unit: lead}))

    if has_storm_index(y):
        return y.storms.apply(
            lambda x: x.droplevel(STORM_LEVEL).shift(periods=1, freq=lead)
        )
    else:
        return y.shift(periods=1, freq=lead)


def setup_mlflow(cfg, features_cfg, data_cfg):
    import mlflow

    experiment_id = OmegaConf.select(cfg, "experiment_id")

    if experiment_id is None and cfg.experiment_name is not None:
        mlflow.set_experiment(cfg.experiment_name)
        experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
        if cfg.experiment_name is not None:
            logger.debug(f"MLFlow Experiment: {cfg.experiment_name}")

        experiment_id = experiment.experiment_id

    orig_cwd = get_original_cwd()
    tracking_uri = f"file://{orig_cwd}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLFlow Tracking URI: {tracking_uri}")

    # if cfg.model == "xgboost":
    #     import mlflow.xgboost

    #     logger.debug("Turning on MLFlow autologging for XGBoost...")
    #     mlflow.xgboost.autolog()

    run = mlflow.start_run(experiment_id=experiment_id)

    # tracking_uri = mlflow.get_tracking_uri()
    # logger.info(f"MLFlow Tracking URI: {tracking_uri}")

    processed_data_dir = Path(to_absolute_path(data_cfg.hydra.run.dir))
    if processed_data_dir is not None:
        data_hydra_dir = processed_data_dir / ".hydra"
        mlflow.log_artifacts(data_hydra_dir, artifact_path="processed_data_configs")
        data_cfg = OmegaConf.load(data_hydra_dir / "config.yaml")
        for name, param_name in DATA_CONFIGS_TO_LOG.items():
            param = OmegaConf.select(data_cfg, param_name)
            if param is not None:
                if isinstance(param, list):
                    param = ", ".join(param)
                mlflow.log_param(name, param)

    model_hydra_dir = Path(".hydra")
    mlflow.log_artifacts(model_hydra_dir, artifact_path="model_configs")

    mlflow.log_params(
        {
            "model": cfg.model,
            "lag": features_cfg.lag,
            "exog_lag": features_cfg.exog_lag,
            "lead": features_cfg.lead,
            "cv_method": cfg.cv.method,
        }
    )
    mlflow.log_params(cfg.cv.params)

    tags = OmegaConf.select(cfg, "tags", default={})
    if bool(tags):
        mlflow.set_tags(tags)

    return run
