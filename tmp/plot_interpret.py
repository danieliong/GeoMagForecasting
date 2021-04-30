#!/usr/bin/env python


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.plot import plot_prediction


def plot_interpret(
    storm,
    y_test,
    ypred,
    X_test,
    ebm_local,
    lead=180,
    figsize=(15, 10),
    gridspec_kw={"height_ratios": [2, 1, 1]},
    sharex=True,
):
    names = ebm_local.data(0)["names"]
    intercept = ebm_local.data(0)["extra"]["scores"][0]

    scores_df = pd.DataFrame(
        [ebm_local.data(i)["scores"] for i in range(y_test.shape[0])],
        columns=names,
        index=y_test.index,
    )

    re_features_names = np.unique(
        [re.sub("_[0-9]+", "_[0-9]+", col) for col in scores_df.columns]
    )

    contrib_df = pd.concat(
        (scores_df.filter(regex=f"^{r}$").sum(axis=1) for r in re_features_names),
        axis=1,
    )
    contrib_df.rename(
        columns={i: x.replace("_[0-9]+", "") for i, x in enumerate(re_features_names)},
        inplace=True,
    )

    fig, ax = plt.subplots(
        nrows=3, figsize=figsize, gridspec_kw=gridspec_kw, sharex=sharex,
    )
    plot_prediction(y_test, ypred, "rmse", storm, True, lead, "minutes", ax=ax[0])

    contrib_df["bz"].loc[storm].plot(ax=ax[1], color="red", label="Bz", linewidth=0.7)
    contrib_df["y"].loc[storm].plot(ax=ax[1], color="blue", label="y", linewidth=0.7)
    ax[1].legend()
    ax[1].set_ylabel("Contributions")

    X_test["bz"].reindex(contrib_df.index).loc[storm].plot(
        ax=ax[2], color="black", label="Bz", linewidth=0.7, linestyle="dashed"
    )
    contrib_df["bz"].loc[storm].plot(
        ax=ax[2], color="red", label="Contrib.", linewidth=0.7
    )
    ax[2].set_ylabel("Bz")
    ax[2].legend()

    return fig, ax
