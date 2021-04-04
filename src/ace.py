#!/usr/bin/env python

import string
import requests
import re
import datetime as dt
import pandas as pd
import numpy as np

from functools import partial
from pathlib import Path
from operator import itemgetter
from dateutil import rrule
from collections import defaultdict

# SWEPAM_COLS = ["Density", "Speed", "Temperature"]
# MAG_COLS = ["Bx", "By", "Bz", "Lat.", "Long."]
FEATURE_COLS = {
    "swepam": ["Density", "Speed", "Temperature"],
    "mag": ["Bx", "By", "Bz", "Lat.", "Long."],
    "loc": ["X", "Y", "Z"],
}

PREFIX = "http://sohoftp.nascom.nasa.gov/sdb/goes/ace/"
TIME_COLS = ["YR", "MO", "DA", "HHMM"]
STATUS_COL = "S"
COLS_TO_EXCLUDE = ["Day"]
ACCEPTED_RESOLUTIONS = ["1m", "5m", "1h"]
HEADER_END_REGEX = re.compile("^#--------------")

# FIXME
class ACEDownloader:
    def __init__(
        self,
        start_time="2010-01-01",
        end_time="2019-12-31",
        resolution="1m",
        prefix=PREFIX,
        time_cols=TIME_COLS,
        cols_to_exclude=COLS_TO_EXCLUDE,
    ):
        self.start_time = dt.datetime.fromisoformat(start_time)
        self.end_time = dt.datetime.fromisoformat(end_time)
        self.resolution = resolution
        self.prefix = prefix
        self.time_cols = time_cols
        self.cols_to_exclude = cols_to_exclude

        assert (
            self.resolution in ACCEPTED_RESOLUTIONS
        ), "resolution must be 1m, 5m, or 1h."

        if resolution in ["1m", "5m"]:
            self.prefix += "daily"
        elif resolution in ["1h"]:
            self.prefix += "monthly"

    def _get_obs_from_line(self, line, headers):

        line = line.decode("ascii")
        values = list(filter(str.strip, line.split(" ")))
        obs = dict(zip(headers, values))

        # Add time
        time = self._get_time(obs)

        # Remove time and exclude columns
        [obs.pop(cols) for cols in self.time_cols + self.cols_to_exclude]

        # Add time
        obs["times"] = time

        return obs

    def download_url(self, url):
        # Return dict
        r = requests.get(url, stream=True)
        lines_iter = r.iter_lines()

        # Get header row
        header_row = None
        prev_line = next(lines_iter)
        for line in lines_iter:
            if HEADER_END_REGEX.match(line.decode("ascii")):
                header_row = prev_line.decode("ascii").replace("#", "")
                break
            else:
                prev_line = line

        assert header_row is not None, "There is not header row."

        headers = list(filter(str.strip, header_row.split(" ")))
        data = defaultdict(list)

        # Iterate through lines in file
        for line in lines_iter:
            obs = self._get_obs_from_line(line, headers)

            if obs["times"] >= self.start_time and obs["times"] <= self.end_time:
                # Append obs to data
                for col, val in obs.items():
                    data[col].append(val)

            if obs["times"] > self.end_time:
                # Stop iterating if already passed end_time
                break

        return data

    def _get_url(self, date_str, feature_type):
        return f"{self.prefix}/{date_str}_ace_{feature_type}_{self.resolution}.txt"

    def _generate_urls(self, feature_type):

        if self.resolution in ["1m", "5m"]:
            dstart = self.start_time.date()
            until = self.end_time.date()
            str_fmt = "%Y%m%d"
        elif self.resolution in ["1h"]:
            dstart = dt.date(self.start_time.year, self.start_time.month)
            until = dt.date(self.end_time.year, self.end_time.month)
            str_fmt = "%Y%m"

        for date in rrule.rrule(rrule.DAILY, dtstart=dstart, until=until):
            date_str = date.strftime(str_fmt)
            yield self._get_url(date_str, feature_type)

    def _get_time(self, obs):
        time_str = "".join(itemgetter(*self.time_cols)(obs))
        time = dt.datetime.strptime(time_str, "%Y%m%d%H%M")
        return time

    def download_data(self, features=None, pandas=True, progressbar=False):

        data = defaultdict(list)

        for feature_type in FEATURE_COLS.keys():
            if features is not None:
                if all(col not in FEATURE_COLS[feature_type] for col in features):
                    continue

            times = []

            if progressbar:
                from tqdm import tqdm

                urls = tqdm(list(self._generate_urls(feature_type)), desc=feature_type)
            else:
                urls = self._generate_urls(feature_type)

            for url in urls:
                url_data = self.download_url(url)

                times.extend(url_data.pop("times"))
                data[_fmt_col(STATUS_COL, feature_type)].extend(
                    url_data.pop(STATUS_COL)
                )

                for col, val in url_data.items():
                    data[_fmt_col(col)].extend(url_data[col])

            if "times" not in data.keys():
                data["times"].extend(times)
            else:
                assert times == data["times"]

        if pandas:
            df = pd.DataFrame(data, dtype=np.float32)
            df.set_index(["times"], inplace=True)
            return df

        return data


def _fmt_col(x, append_txt=None):
    # Format column names
    # Lowercase and remove punctuations

    x_fmt = x.translate(
        str.maketrans("", "", string.punctuation.replace("_", ""))
    ).lower()
    if append_txt is not None:
        x_fmt = x_fmt + "_" + append_txt
    return x_fmt


def _ace_paths(
    feature_types, start, end, resolution="1m", root_dir="data/ace/",
):

    # TODO: Raise exceptions for wrong input
    # - loc cannot be in feature_types if swepam or mag is in it and resolution is not "1h"
    # - resolution cannot be "1m" or "5m" if loc is in feature_types

    root_dir = Path(root_dir)

    if resolution in ["1m", "5m"]:
        parent_dir = root_dir / "daily"
        date_fmt = "%Y%m%d"
        rrule_freq = rrule.DAILY
    elif resolution == "1h":
        parent_dir = root_dir / "hourly"
        date_fmt = "%Y%m"
        rrule_freq = rrule.MONTHLY

    def _is_datetime_object(x):
        return isinstance(x, dt.datetime) or isinstance(x, pd.Timestamp)

    if not _is_datetime_object(start):
        start = dt.datetime.fromisoformat(start)

    if not _is_datetime_object(end):
        end = dt.datetime.fromisoformat(end)

    for date in rrule.rrule(rrule_freq, dtstart=start, until=end):
        date_str = date.strftime(date_fmt)
        yield {
            feat_type: parent_dir / f"{date_str}_ace_{feat_type}_{resolution}.txt"
            for feat_type in feature_types
        }


def _format_df(df, start, end, feat_type):

    if STATUS_COL in df.columns:
        df.rename({STATUS_COL: f"status_{feat_type}"}, axis=1, inplace=True)

    df.rename(_fmt_col, axis=1, inplace=True)

    return df[start:end]


def _read_path(
    path,
    feat_type,
    features,
    start,
    end,
    time_cols=TIME_COLS,
    exclude_cols=COLS_TO_EXCLUDE,
    delim_whitespace=True,
    **kwargs,
):
    # Read ACE data for one path

    # Mapper for usecols argument in read_csv
    def _usecols(col):
        exclude = any(re.match(exclude_col, col) for exclude_col in exclude_cols)
        # Needed because there are duplicate column names (Day)

        if features is None:
            return not exclude
        else:
            return not exclude and col in time_cols + features

    df = pd.read_csv(
        path,
        index_col="times",
        parse_dates={"times": time_cols},
        delim_whitespace=delim_whitespace,
        usecols=_usecols,
        **kwargs,
    )

    df = _format_df(df, start, end, feat_type)

    return df


def _generate_dfs(paths_dicts, **kwargs):
    # Generate dataframes for each path

    for path_dict in paths_dicts:
        df_iter = (
            _read_path(path, feat_type, **kwargs)
            for feat_type, path in path_dict.items()
        )
        df = pd.concat(df_iter, axis="columns", join="inner")
        yield df


def concat_ace_data(
    start="2010-01-01",
    end="2019-12-31",
    features=None,
    feature_types=None,
    root_dir="data/ace",
    resolution="1m",
    time_cols=TIME_COLS,
    **read_kwargs,
):
    # Read from already downloaded files

    if feature_types is None:
        # If feature types is not specified, infer it from features
        feature_types = []
        if features is None:
            feature_types.extend(FEATURE_COLS.keys())
        else:
            for feature_type, cols in FEATURE_COLS.items():
                if any(feature in cols for feature in features):
                    feature_types.append(feature_type)
    else:
        # If feature_types is specified, just get all features from those types
        features = None

    paths_dicts = _ace_paths(
        feature_types, start=start, end=end, root_dir=root_dir, resolution=resolution,
    )

    data = pd.concat(
        _generate_dfs(
            paths_dicts, features=features, start=start, end=end, **read_kwargs,
        )
    )

    return data


concat_ace_positions = partial(concat_ace_data, feature_types=["loc"], resolution="1h")


if __name__ == "__main__":
    data_path = Path("data/ace_2010-2019.csv")
    pos_path = Path("data/ace_pos_2010-2019.csv")

    if not data_path.exists():
        data = concat_ace_data(feature_types=["swepam", "mag"])
        data.to_csv(data_path)
    else:
        print(f"{data_path} already exists.")

    if not pos_path.exists():
        print(f"Loading satellite positions from {pos_path}")
        positions = concat_ace_positions()
        positions.to_csv(pos_path)
    else:
        print(f"{pos_path} already exists.")

    # # Download data
    # downloader = ACEDownloader()
    # data = downloader.download_data(pandas=True, progressbar=True)
    # path = "data/ace_2010-2019.csv"
    # data.to_csv(path)
    # print(f"Downloaded data to {path}")
    # # FIXME: Keep getting connection error
