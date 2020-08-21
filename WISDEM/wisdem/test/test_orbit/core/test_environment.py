"""Tests for the `Vessel` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

import pandas as pd
import pytest
from marmot import le

from wisdem.orbit.core import Environment
from wisdem.test.test_orbit.data import test_weather as _weather

# Weather data with 'windspeed' column
simple_weather = _weather.copy()

# Weather data with windspeed heights and complex varied column name precision
weather = pd.DataFrame(_weather)
weather["windspeed_10m"] = weather["windspeed"].copy()
weather = weather.drop("windspeed", axis=1)
weather["windspeed_100.0m"] = weather["windspeed_10m"].copy() * 1.3
weather["windspeed_1.2m"] = weather["windspeed_10m"].copy() * 0.7
weather = weather.to_records()


def test_simple_inputs():
    env = Environment(state=simple_weather)

    assert "waveheight" in env.state.dtype.names
    assert "windspeed" in env.state.dtype.names


def test_inputs():
    env = Environment(state=weather)

    assert "waveheight" in env.state.dtype.names
    assert "windspeed" not in env.state.dtype.names
    assert "windspeed_10m" in env.state.dtype.names
    assert "windspeed_100m" in env.state.dtype.names
    assert "windspeed_1.2m" in env.state.dtype.names

    env2 = Environment(state=weather, ws_alpha=0.12, ws_default_height=20)

    assert env2.alpha == 0.12
    assert env2.default_height == 20


def test_simple_constraint_application():
    env = Environment(state=simple_weather)

    constraints = {"windspeed": le(10)}
    valid = env._find_valid_constraints(**constraints)
    assert valid == constraints

    with_height = {"windspeed_100m": le(10)}
    valid = env._find_valid_constraints(**with_height)
    assert "windspeed" in list(valid.keys())

    with_mult_heights = {"windspeed_100m": le(10), "windspeed_10m": le(8)}
    with pytest.raises(ValueError):
        valid = env._find_valid_constraints(**with_mult_heights)


def test_constraint_application():
    env = Environment(state=weather)

    constraints = {"waveheight": le(2), "windspeed": le(10)}
    valid = env._find_valid_constraints(**constraints)
    assert "windspeed_10m" in list(valid.keys())

    constraints = {"waveheight": le(2), "windspeed_10m": le(10)}
    valid = env._find_valid_constraints(**constraints)
    assert "waveheight" in list(valid.keys())
    assert "windspeed_10m" in list(valid.keys())

    assert "windspeed_20m" not in env.state.dtype.names
    constraints = {"waveheight": le(2), "windspeed_20m": le(10)}
    valid = env._find_valid_constraints(**constraints)
    assert "windspeed_20m" in list(valid.keys())
    assert "windspeed_20m" in env.state.dtype.names

    assert "windspeed_120m" not in env.state.dtype.names
    constraints = {"waveheight": le(2), "windspeed_120m": le(10)}
    valid = env._find_valid_constraints(**constraints)
    assert "windspeed_120m" in list(valid.keys())
    assert "windspeed_120m" in env.state.dtype.names


def test_interp():
    env = Environment(state=weather)

    assert "windspeed_20m" not in env.state.dtype.names
    constraints = {"waveheight": le(2), "windspeed_20m": le(10)}
    valid = env._find_valid_constraints(**constraints)
    assert "windspeed_20m" in env.state.dtype.names
    assert (env.state["windspeed_10m"] < env.state["windspeed_20m"]).all()
    assert (env.state["windspeed_20m"] < env.state["windspeed_100m"]).all()


def test_extrap():
    env = Environment(state=weather)

    assert "windspeed_120m" not in env.state.dtype.names
    constraints = {"waveheight": le(2), "windspeed_120m": le(10)}
    valid = env._find_valid_constraints(**constraints)
    assert "windspeed_120m" in env.state.dtype.names
    assert (env.state["windspeed_120m"] > env.state["windspeed_100m"]).all()

    env2 = Environment(state=weather, ws_alpha=0.12)

    assert "windspeed_120m" not in env2.state.dtype.names
    constraints = {"waveheight": le(2), "windspeed_120m": le(10)}
    valid = env2._find_valid_constraints(**constraints)
    assert (env.state["windspeed_100m"] == env2.state["windspeed_100m"]).all()
    assert (env.state["windspeed_120m"] < env2.state["windspeed_120m"]).all()
