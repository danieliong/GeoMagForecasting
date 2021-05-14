#!/usr/bin/env python

import mlflow
import shutil
import click

from pathlib import Path
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf
from hydra.experimental import initialize
from src.mlflow_utils import MLFlowRun
from src.utils import get_features_cfg


def _copy_data(
    artifacts_path, output_dir, X_test_path="X_test.pkl", y_test_path="y_test.pkl"
):
    model_configs_dir = Path(artifacts_path) / "model_configs"
    model_cfg = OmegaConf.load(model_configs_dir / "config.yaml")
    hydra_cfg = OmegaConf.load(model_configs_dir / "hydra.yaml")
    cfg = OmegaConf.merge(model_cfg, hydra_cfg)
    with initialize(config_path="../configs"):
        features_cfg = get_features_cfg(cfg)

        features_dir = Path(features_cfg.hydra.run.dir)
        # X_test = pd.read_pickle(features_dir / "X_test.pkl")
        # y_test = pd.read_pickle(features_dir / "y_test.pkl")

    shutil.copy(features_dir / X_test_path, output_dir)
    shutil.copy(features_dir / y_test_path, output_dir)
    print(f"Copied test data into {output_dir}.")


@click.command()
@click.option(
    "--experiments",
    "-e",
    multiple=True,
    default=["5", "8"],
    help="Experiment IDs (Can take multiple).",
)
@click.option("--model", "-m", default="xgboost", type=str, help="Model to query.")
@click.option("--lead", "-l", default="60", type=str, help="Lead time to query.")
@click.option(
    "--features-source",
    "-f",
    default="ace_cdaweb",
    type=str,
    help="Feature source to query.",
)
@click.option(
    "--filter-string",
    "-s",
    default="",
    type=str,
    help="filter_string argument in mlflow.search_runs.",
)
@click.option("--output-dir", "-o", default=None, help="Dir. to copy results to.")
def retrieve_results_from_mlflow(
    experiments=["5", "8"],
    model="xgboost",
    lead="60",
    features_source="ace_cdaweb",
    filter_string="",
    output_dir=None,
):

    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
    else:
        output_dir = Path(f"paper_results/{model}_{lead}_{features_source}").resolve()

    client = MlflowClient("mlruns")

    filter_string_all = f"params.model='{model}' and params.lead='{lead}' and params.features_source='{features_source}'"
    if len(filter_string) != 0:
        filter_string_all = f"{filter_string_all} and {filter_string}"
    runs = client.search_runs(experiments, filter_string_all)

    if len(runs) == 1:
        artifact_path = runs[0].info.artifact_uri.replace("file://", "")
        shutil.copytree(artifact_path, output_dir, dirs_exist_ok=True)
        print(f"Saved contents of {artifact_path} to {output_dir}.")

        _copy_data(artifact_path, output_dir)

    elif len(runs) > 1:
        print("More than one run matches the query.")
        for run in runs:
            artifact_path = run.info.artifact_uri.replace("file://", "")
            output_subdir = output_dir / run.info.run_id

            shutil.copytree(artifact_path, output_subdir, dirs_exist_ok=True)
            print(f"Saved contents of {artifact_path} to {output_dir}.")

            _copy_data(artifact_path, output_subdir)
    else:
        raise Exception("No runs match the query!")


if __name__ == "__main__":
    retrieve_results_from_mlflow()
