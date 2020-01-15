"""Tests for scour protection installation tasks"""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import copy

import numpy as np
import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.vessels import tasks
from wisdem.orbit.vessels.tasks._defaults import defaults


def test_load_rocks():
    load_rocks_time = {"load_rocks_time": 100}
    res = tasks.load_rocks(**load_rocks_time)
    assert res == 100

    load_rocks_time = {"load_rocks_time": 100.0}
    res = tasks.load_rocks(**load_rocks_time)
    assert res == pytest.approx(load_rocks_time["load_rocks_time"], rel=1e-10)

    load_rocks_time = {"load_rocks_times": 100}
    res = tasks.load_rocks(**load_rocks_time)
    assert res == defaults["load_rocks_time"]


def test_drop_rocks():
    drop_rocks_time = {"drop_rocks_time": 100}
    res = tasks.drop_rocks(**drop_rocks_time)
    assert res == 100

    drop_rocks_time = {"drop_rocks_time": 100.0}
    res = tasks.drop_rocks(**drop_rocks_time)
    assert res == pytest.approx(drop_rocks_time["drop_rocks_time"], rel=1e-10)

    drop_rocks_time = {"drop_rocks_times": 100}
    res = tasks.drop_rocks(**drop_rocks_time)
    assert res == defaults["drop_rocks_time"]
