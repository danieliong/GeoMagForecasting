#!/usr/bin/env jupyter

import pandas as pd
import datetime as dt

from spacepy import pycdf
from pathlib import Path
from dateutil import rrule


def _generate_data_ace_cdaweb(
    stormtimes, data_dir, str_format, data_keys, index_key, cols_keys, **kwargs
):
    data_dir = Path(data_dir)

    for i, (start, end) in stormtimes.iterrows():
        print(f"Storm #{i}")
        dtstart = dt.date.fromisoformat(start)
        until = dt.date.fromisoformat(end)
        for date in rrule.rrule(rrule.DAILY, dtstart=dtstart, until=until):
            # Find file that matches pattern specified in str_format
            paths = list(data_dir.glob(date.strftime(str_format)))
            assert len(paths) == 1
            path = paths[0]
            # path = data_dir / date.strftime(str_format)

            with pycdf.CDF(str(path)) as cdf:
                times = cdf[index_key][...]
                # columns = cdf[cols_keys][...]

                df_iter = (
                    pd.DataFrame(
                        cdf[data_key][...],
                        index=times,
                        columns=cdf[cols_key][...],
                        **kwargs,
                    )
                    .resample("T")
                    .mean()
                    for data_key, cols_key in zip(data_keys, cols_keys)
                )
                df = pd.concat(df_iter, axis=1)

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
    data_keys=["BGSM"],
    cols_keys=["label_bgsm"],
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
            cols_keys=cols_keys,
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
    data = load_features_ace_cdaweb(stormtimes_path="data/stormtimes_siciliano.csv")
    data.to_pickle("data/ace_cdaweb_siciliano.pkl")
