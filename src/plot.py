#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.tseries.frequencies import to_offset
from ._train import compute_metric, predict_persistence
from src.storm_utils import has_storm_index

FIGSIZE = (15, 10)


def plot_prediction(y, ypred, metric, storm, persistence, lead, unit, ax=None):

    # locator = mdates.AutoDateLocator(minticks=5)
    # formatter = mdates.ConciseDateFormatter(locator)
    # ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(formatter)

    if storm is None:
        metric_val = compute_metric(y, ypred, metric, storm=None)
        y_, ypred_ = y, ypred
    else:
        metric_val = compute_metric(y, ypred, metric, storm=storm)
        y_, ypred_ = y.storms.get(storm), ypred.storms.get(storm)

    metric_val = round(metric_val, ndigits=3)

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)
    else:
        fig = None

    y_.plot(ax=ax, color="black", linewidth=0.7, label="Truth")
    ypred_.plot(
        ax=ax, color="red", linewidth=0.7, label=f"Prediction [{metric}: {metric_val}]"
    )

    if persistence:
        _ = _plot_persistence(y_, lead=lead, metric=metric, unit=unit, ax=ax)

    ax.legend(ncol=3, prop={"size": 8})
    if storm is not None:
        ax.set_title(f"Storm #{storm}")
    ax.set_xlabel("")

    return fig, ax


def _plot_persistence(y, lead, metric, unit="minutes", ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE)

    ypers = predict_persistence(y, lead, unit=unit)
    metric_val = round(compute_metric(y, ypers, metric), ndigits=3)
    ypers.plot(
        ax=ax,
        color="blue",
        linewidth=0.5,
        linestyle="dashed",
        label=f"Persistence [{metric}: {metric_val}]",
    )

    ax.legend()
    if ax is None:
        return fig, ax
    else:
        return None, ax


def plot_predictions(
    y,
    ypred,
    metrics,
    use_mlflow=False,
    pdf_path="prediction_plots.pdf",
    persistence=False,
    lead=None,
    unit="minutes",
):
    if use_mlflow:
        import mlflow
    elif pdf_path is not None:
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
        fig = []
        ax = []
        # Plot predictions for each storm individually
        for storm in y.storms.level:
            storm_fig, storm_ax = plot_prediction(
                y,
                ypred,
                metric,
                storm=storm,
                persistence=persistence,
                lead=lead,
                unit=unit,
            )

            fig.append(storm_fig)
            ax.append(storm_ax)

            if use_mlflow:
                mlflow.log_figure(fig, f"prediction_plots/storm_{storm}.png")
            elif pdf_path is not None:
                pdf.savefig(fig)
    else:
        fig, ax = plot_prediction(
            y, ypred, metric, persistence=persistence, lead=lead, unit=unit
        )
        if use_mlflow:
            mlflow.log_figure(fig, "prediction_plot.png")
        elif pdf_path is not None:
            pdf.savefig(fig)

    if not use_mlflow:
        pdf.close()

    return fig, ax
