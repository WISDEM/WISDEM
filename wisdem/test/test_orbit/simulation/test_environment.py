"""Tests for the `Environment` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import logging

import pandas as pd
import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.simulation import Environment
from wisdem.orbit.simulation.exceptions import (
    WeatherWindowNotFound,
    WeatherProfileExhausted,
)

logger = logging.Logger("TestLogger", level=logging.INFO)
_shared = {"agent": "TestSetup", "action": "Test"}


def test_weather_default():
    """
    Tests Environment creation and weather assignment.
    """

    env = Environment()
    assert isinstance(env._weather, type(None))


def test_weather_assignment():
    """
    Tests Environment creation and weather assignment.
    """

    env = Environment(weather=test_weather)
    assert isinstance(env._weather, pd.DataFrame)


def test_weather_windows():
    """
    Tests the finding of weather windows. Currently only has functionality for
    'windspeed' and 'waveheight'.
    """

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {**_shared, "duration": 10}
    delay, action = list(env._schedule_task(task))
    assert delay._delay == 0
    assert action._delay == 10

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {**_shared, "duration": 10, "max_windspeed": 5}
    delay, action = list(env._schedule_task(task))
    assert delay._delay == 330
    assert action._delay == 10

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {**_shared, "duration": 10, "max_waveheight": 0.5}
    delay, action = list(env._schedule_task(task))
    assert delay._delay == 4
    assert action._delay == 10

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {
        **_shared,
        "duration": 10,
        "max_windspeed": 6,
        "max_waveheight": 0.7,
    }
    delay, action = list(env._schedule_task(task))
    assert delay._delay == 337
    assert action._delay == 10

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {
        **_shared,
        "duration": 24,
        "max_windspeed": 6,
        "max_waveheight": 0.7,
    }
    delay, action = list(env._schedule_task(task))
    assert delay._delay == 337
    assert action._delay == 24

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {
        **_shared,
        "duration": 24,
        "max_windspeed": 6,
        "max_waveheight": 0.5,
    }
    delay, action = list(env._schedule_task(task))
    assert delay._delay == 4168
    assert action._delay == 24

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {**_shared, "duration": 1, "max_windspeed": 0}
    with pytest.raises(WeatherWindowNotFound):
        delay, action = list(env._schedule_task(task))

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {**_shared, "duration": 1, "max_waveheight": 0}
    with pytest.raises(WeatherWindowNotFound):
        delay, action = list(env._schedule_task(task))


def test_multiple_tasks():
    """
    Tests task_handler behavior that occurs when multiple tasks are passed in.
    """

    tasks = [
        {**_shared, "duration": 8},
        {**_shared, "duration": 12},
        {**_shared, "duration": 25},
    ]

    env = Environment()
    task_list = list(env.task_handler(tasks))
    assert len(task_list) == 3

    env = Environment()
    env.logger = logger
    env.process(env.task_handler(tasks))
    env.run()
    assert int(env.now) == 45


def test_weather_data_exhaustion():
    """
    Tests behavior that occurs at the end of a weather file.
    Should raise a custom exception when the end of the DataFrame is reached.
    """

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {**_shared, "duration": 1e6}
    with pytest.raises(WeatherProfileExhausted):
        delay, action = list(env._schedule_task(task))

    env = Environment(weather=test_weather)
    env.logger = logger
    task = {**_shared, "duration": 36750}
    delay, action = list(env._schedule_task(task))
    env.run()
    assert int(env.now) == 36750

    task = {**_shared, "duration": 50}
    with pytest.raises(WeatherProfileExhausted):
        delay, action = list(env._schedule_task(task))
