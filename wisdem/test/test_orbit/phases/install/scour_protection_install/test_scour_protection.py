"""
Provides a testing framework for the `ScourProtectionInstallation`.
"""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__credits__ = ["Jake Nunemaker"]
__version__ = "0.0.1"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"
__status__ = "Development"


import copy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.library import initialize_library, extract_library_specs
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import ScourProtectionInstallation

initialize_library(pytest.library)
config = extract_library_specs("config", "scour_protection_install")


def test_simulation_creation():
    sim = ScourProtectionInstallation(config, print_logs=False)

    assert sim.config == config
    assert sim.env
    assert sim.env.logger
    assert sim.port
    assert sim.scour_vessel
    assert sim.num_turbines
    assert sim.scour_protection_tonnes_to_install


def test_port_creation():
    sim = ScourProtectionInstallation(config, print_logs=False)

    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]


def test_scour_protection_creation():
    sim = ScourProtectionInstallation(config, print_logs=False)

    assert sim.num_turbines == config["plant"]["num_turbines"]

    _items = [
        item for item in sim.port.items if item["type"] == "Scour Protection"
    ]
    assert len(_items) == sim.num_turbines

    for _item in _items:
        assert _item["weight"] == pytest.approx(
            sim.scour_protection_tonnes_to_install, rel=1e-6
        )


@pytest.mark.parametrize(
    "level,expected", (("INFO", 20), ("DEBUG", 10)), ids=["info", "debug"]
)
def test_logger_creation(level, expected):
    sim = ScourProtectionInstallation(config, log_level=level)
    assert sim.env.logger.level == expected


@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_full_run_completes(weather):
    sim = ScourProtectionInstallation(
        config, weather=weather, log_level="DEBUG"
    )
    sim.run()

    complete = float(sim.logs[sim.logs.action == "Complete"]["time"])
    assert complete > 0


@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_full_run_is_valid(weather):
    sim = ScourProtectionInstallation(
        config, weather=weather, log_level="INFO"
    )
    sim.run()

    df = sim.phase_dataframe[
        ~sim.phase_dataframe.agent.isin(["Director", "Port"])
    ]
    for action in ("FastenItem", "DropRocks"):
        assert df[df.action == action].shape[0] == sim.num_turbines


@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_full_run_logging(weather):
    sim = ScourProtectionInstallation(
        config, weather=weather, log_level="INFO"
    )
    sim.run()

    df = sim.phase_dataframe[
        ~sim.phase_dataframe.agent.isin(["Director", "Port"])
    ]
    df = df.assign(shift=(df.time - df.time.shift(1)))
    assert (df.duration - df["shift"]).max() == pytest.approx(0, abs=1e-9)


def test_kwargs():

    sim = ScourProtectionInstallation(
        config, log_level="INFO", print_logs=False
    )
    sim.run()
    baseline = sim.total_phase_time

    keywords = ["drop_rocks_time"]

    failed = []

    for kw in keywords:

        default = defaults[kw]
        kwargs = {kw: default + 2}

        new_sim = ScourProtectionInstallation(
            config, log_level="INFO", print_logs=False, **kwargs
        )
        new_sim.run()
        new_time = new_sim.total_phase_time

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"'{failed}' not affecting results.")

    else:
        assert True
