# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
try:
    from .utils import load_solar_wind, load_supermag
except ImportError:
    from src.data.utils import get_features_list, load_solar_wind, load_supermag


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath="data/raw",
         output_filepath="data/interim",
         params_path="parameters/",
         station="OTT",
         years=list(range(2010:2020)),
         ):
    """Loads data from data/raw and do initial pre-processing before extracting features.
    Pre-processed data is saved in data/interim
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data and performing initial pre-processing...")

    # Load data
    sw_path = input_filepath / "solar_wind.csv"
    features = get_features_list(params_path)

    solar_wind = load_solar_wind(sw_path, features=features)
    supermag = load_supermag(station, years, data_dir=input_filepath)

    # Save interim data
    solar_wind.to_pickle(output_filepath / "solar_wind.pkl")
    supermag.to_pickle(output_filepath / "supermag.pkl")

    return solar_wind, supermag


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
