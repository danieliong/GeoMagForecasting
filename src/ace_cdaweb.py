#!/usr/bin/env jupyter

import pandas as pd
import datetime as dt

from spacepy import pycdf
from pathlib import Path
from dateutil import rrule


def _generate_data_ace_cdaweb(
    stormtimes, data_dir, str_format, data_keys, index_key, **kwargs
):
    data_dir = Path(data_dir)

    def _generate_df():
        for data_key, label in data_keys.items():
            if isinstance(label, str):
                columns = cdf[label][...]
            elif isinstance(label, list):
                columns = label

            yield pd.DataFrame(
                cdf[data_key][...],
                index=cdf[index_key][...],
                columns=columns,
                **kwargs,
            ).resample("T").mean()

    for i, (start, end) in stormtimes[["start_time", "end_time"]].iterrows():
        print(f"Storm #{i}")
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
    start="2010-01-01",
    end="2019-12-31",
    stormtimes_path=None,
    str_format="%Y/ac_h3_mfi_%Y%m%d.cdf",
    data_keys={"BGSM": "label_bgsm"},
    index_key="Epoch",
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
            **kwargs,
        )
        data = pd.concat(data_iter)
        data.index.rename(["storm", "times"], inplace=True)
    else:
        raise ValueError("Loading data within a time period is not supported yet.")
        # TODO: Add loading all data in a time period
        pass

    return data


if __name__ == "__main__":
    # TODO: Create CLI
    stormtimes_path = "data/stormtimes_combined.csv"
    data_path = "data/ace_cdaweb_combined.pkl"

    swepam_data_dir = "data/ace_cdaweb/swepam"
    swepam_str_fmt = "%Y/ac_h0_swe_%Y%m%d_*.cdf"
    swepam_data_keys = {
        "V_GSM": ["vx", "vy", "vz"],
        "Np": ["density"],
        "Tpr": ["temperature"],
        "SC_pos_GSM": ["x", "y", "z"],
    }
    # swepam_data_keys = ["V_GSM", "Np", "Tpr"]
    # swepam_cols_keys = ["label_V_GSM", ["density"], ["temperature"]]

    print("Loading SWEPAM data...")
    swepam_data = load_features_ace_cdaweb(
        data_dir=swepam_data_dir,
        stormtimes_path=stormtimes_path,
        str_format=swepam_str_fmt,
        data_keys=swepam_data_keys,
    )

    mag_data_dir = "data/ace_cdaweb/mag"
    mag_str_fmt = "%Y/ac_h3_mfi_%Y%m%d.cdf"
    mag_data_keys = {"BGSM": ["bx", "by", "bz"]}

    print("Loading MAG data...")
    mag_data = load_features_ace_cdaweb(
        data_dir=mag_data_dir,
        stormtimes_path=stormtimes_path,
        str_format=mag_str_fmt,
        data_keys=mag_data_keys,
    )

    data = pd.concat([swepam_data, mag_data], axis=1)
    print(f"Saving data to {data_path}")
    data.to_pickle(data_path)
