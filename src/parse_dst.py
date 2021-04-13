#!/usr/bin/env python

import re
import pandas as pd


def _parse_one_day(line):

    # Parse first string in each line that contains info
    r_info = re.compile(
        r"DST([0-9]{2})([0-9]{2})\*([0-9]{2})(RR|PP|  )X([0-9]{1}| )([0-9]{2}| )"
    )

    # Match numbers (negative or positive)
    r_data = re.compile(r"(-*[0-9]+)")

    # Get data for day
    # # This didn't work because there could be spaces in info
    # info, nums = line.split(maxsplit=1)
    info = line[:16]
    nums = line[16:]
    data = r_data.findall(nums)[1:-1]

    # Get times for day
    # QUESTION: Should I include version (real-time, provisional, etc) in data?
    yr_end, month, day, _, vers, yr_start = r_info.match(info).groups()

    # top two digits of year might be space
    yr_start = "19" if yr_start == " " else yr_start

    date = f"{yr_start}{yr_end}-{month}-{day}"
    times = pd.date_range(date, periods=24, freq="H")

    # Make sure there are 24 observations
    assert len(data) == 24

    return pd.DataFrame({"times": times, "dst": data, "version": vers}).set_index(
        "times"
    )


def parse_dst(path):
    with open(path) as f:
        dst = pd.concat((_parse_one_day(line) for line in f))

    # Replace 9999 with NA
    dst.mask(dst == 9999, inplace=True)

    return dst


if __name__ == "__main__":
    path = "data/dst_201001-202103.dat"
    dst = parse_dst(path)
    dst.to_csv("data/dst_201001-202103.csv")
