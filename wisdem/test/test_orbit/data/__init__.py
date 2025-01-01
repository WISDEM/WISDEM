__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent
_fp = DIR / "test_weather.csv"
test_weather = (
    pd.read_csv(_fp, parse_dates=["datetime"])
    .set_index("datetime")
    .to_records()
)
