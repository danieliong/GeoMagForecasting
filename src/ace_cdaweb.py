#!/usr/bin/env python3

import click
import warnings
import pandas as pd
import datetime as dt

from spacepy import pycdf
from pathlib import Path
from dateutil import rrule
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

SWEPAM_STR_FMT = "%Y/ac_h0_swe_%Y%m%d_*.cdf"
SWEPAM_DATA_KEYS = {
    "V_GSM": ["vx", "vy", "vz"],
    "Np": ["density"],
    "Tpr": ["temperature"],
    "SC_pos_GSM": ["x", "y", "z"],
}

MAG_STR_FMT = "%Y/ac_h3_mfi_%Y%m%d.cdf"
MAG_DATA_KEYS = {"BGSM": ["bx", "by", "bz"]}


def _generate_data_ace_cdaweb(
    stormtimes, data_dir, str_format, data_keys, index_key, freq, **kwargs
):
    data_dir = Path(data_dir)

    def _generate_df():
        for data_key, label in data_keys.items():
            if isinstance(label, str):
                columns = cdf[label][...]
            elif isinstance(label, list):
                columns = label

            df = pd.DataFrame(
                cdf[data_key][...],
                index=cdf[index_key][...],
                columns=columns,
                **kwargs,
            )

            if freq is not None:
                # Fill NA values
                meta = cdf[data_key].meta
                df.where(df != meta["FILLVAL"], inplace=True)
                df.where(df >= meta["VALIDMIN"], inplace=True)
                df.where(df <= meta["VALIDMAX"], inplace=True)

                df = df.resample(freq).mean()
                # df = df.resample(freq, label="right").mean()

            # Resample to 1-minute resolution
            # yield df.resample("T").mean()
            yield df

    for i, (start, end) in tqdm(
        list(stormtimes[["start_time", "end_time"]].iterrows())
    ):
        # print(f"Storm #{i}")
        dtstart = dt.datetime.fromisoformat(start)
        until = dt.datetime.fromisoformat(end)
        for date in rrule.rrule(rrule.DAILY, dtstart=dtstart, until=until):
            # Find file that matches pattern specified in str_format
            paths = list(data_dir.glob(date.strftime(str_format)))
            assert len(paths) == 1
            path = paths[0]
            # path = data_dir / date.strftime(str_format)

            with pycdf.CDF(str(path)) as cdf:
                # times = cdf[index_key][...]
                # columns = cdf[cols_keys][...]

                df_iter = _generate_df()
                df = pd.concat(df_iter, axis=1)

                # df_iter = (
                #     pd.DataFrame(
                #         cdf[data_key][...],
                #         index=times,
                #         columns=cdf[cols_key][...],
                #         **kwargs,
                #     )
                #     .resample("T")
                #     .mean()
                #     for data_key, cols_key in zip(data_keys, cols_keys)
                # )

                # df = (
                #     pd.DataFrame(
                #         cdf[data_key][...], index=times, columns=columns, **kwargs,
                #     )
                #     .resample("T")
                #     .mean()
                # )
                index = pd.MultiIndex.from_product(
                    [[i], df.index], names=["storm", "times"]
                )
                df.set_index(index, inplace=True)
            yield df


def load_features_ace_cdaweb(
    data_dir="data/ace_cdaweb",
    # start="2010-01-01",
    # end="2019-12-31",
    stormtimes_path=None,
    str_format="%Y/ac_h3_mfi_%Y%m%d.cdf",
    data_keys={"BGSM": "label_bgsm"},
    index_key="Epoch",
    freq=None,
    **kwargs,
):
    # XXX: This function takes a long time since the data is in 1 second resolution
    if stormtimes_path is not None:
        # TODO: Only get data from storm times
        stormtimes = pd.read_csv(stormtimes_path, index_col=0)
        data_iter = _generate_data_ace_cdaweb(
            stormtimes=stormtimes,
            data_dir=data_dir,
            str_format=str_format,
            data_keys=data_keys,
            index_key=index_key,
            # cols_keys=cols_keys,
            freq=freq,
            **kwargs,
        )
        data = pd.concat(data_iter)
        data.index.rename(["storm", "times"], inplace=True)
    else:
        raise ValueError("Loading data within a time period is not supported yet.")
        # TODO: Add loading all data in a time period
        pass

    return data


@click.command()
@click.option("--stormtimes", "-t", help="Path to stormtimes.")
@click.option("--output-path", "-o", help="Path to output combined SWEPAM/MAG data.")
@click.option("--swepam/--no-swepam", default=True, help="Save SWEPAM data.")
@click.option("--mag/--no-mag", default=True, help="Save MAG data.")
@click.option(
    "--swepam-dir",
    default="data/ace_cdaweb/swepam",
    help="Path to SWEPAM data directory.",
)
@click.option(
    "--mag-dir", default="data/ace_cdaweb/mag", help="Path to MAG data directory."
)
@click.option("--swepam-path", default=None, help="Path to output SWEPAM data.")
@click.option("--mag-path", default=None, help="Path to output MAG data.")
@click.option(
    "--swepam-freq", default="5T", help="Frequency to resample SWEPAM data to."
)
@click.option("--mag-freq", default="5T", help="Frequency to resample MAG data to.")
def main(
    stormtimes,
    output_path,
    swepam=True,
    mag=True,
    swepam_dir="data/ace_cdaweb/swepam",
    mag_dir="data/ace_cdaweb/mag",
    swepam_path=None,
    mag_path=None,
    swepam_freq="5T",
    mag_freq="5T",
):

    swepam_freq = None if swepam_freq == "None" else swepam_freq
    mag_freq = None if mag_freq == "None" else mag_freq

    print(f"Reading storms from {stormtimes}")

    data_list = []

    if swepam:
        print("Loading SWEPAM data...")
        swepam_data = load_features_ace_cdaweb(
            data_dir=swepam_dir,
            stormtimes_path=stormtimes,
            str_format=SWEPAM_STR_FMT,
            data_keys=SWEPAM_DATA_KEYS,
            freq=swepam_freq,
        )
        data_list.append(swepam_data)
        if swepam_path is not None:
            print(f"Saving SWEPAM data to {swepam_path}")
            swepam_data.to_pickle(swepam_path)

    if mag:
        print("Loading MAG data...")
        mag_data = load_features_ace_cdaweb(
            data_dir=mag_dir,
            stormtimes_path=stormtimes,
            str_format=MAG_STR_FMT,
            data_keys=MAG_DATA_KEYS,
            freq=mag_freq,
        )
        data_list.append(mag_data)
        if mag_path is not None:
            print(f"Saving MAG data to {mag_path}")
            mag_data.to_pickle(mag_path)

    data = pd.concat(data_list, axis=1)
    if output_path is not None:
        print(f"Saving combined data to {output_path}")
        data.to_pickle(output_path)

    return data


if __name__ == "__main__":
    main()
