"""Tests for the `InstallPhase` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import pandas as pd
import pytest

from wisdem.orbit.simulation import Environment
from wisdem.orbit.phases.install import InstallPhase


class BadInstallPhase(InstallPhase):
    """Subclass for testing InstallPhase."""

    def __init__(self, config, **kwargs):
        """Creates an instance of BadInstallPhase."""

        self.config = config
        self.env = Environment(weather=None)
        self.init_logger(**kwargs)

        self.agent_costs["test_vessel"] = 10.0


class SampleInstallPhase(InstallPhase):
    """Subclass for testing InstallPhase."""

    def __init__(self, config, **kwargs):
        """Creates an instance of SampleInstallPhase."""

        self.config = config
        self.env = Environment(weather=None)
        self.init_logger(**kwargs)

    def detailed_output(self):
        pass

    def setup_simulation(self):
        pass


base_config = {"port": {"num_cranes": 1, "name": "TEST_PORT"}}


def test_abstract_methods():

    with pytest.raises(TypeError):
        install = BadInstallPhase(base_config)

    install = SampleInstallPhase(base_config)


def test_logger_creation():
    director = SampleInstallPhase(base_config, log_level="INFO")
    assert director.logger.level == 20

    director = SampleInstallPhase(base_config, log_level="DEBUG")
    assert director.logger.level == 10

    director.config = director.initialize_library(base_config)
    director.extract_defaults()

    director.run(until=10)
    assert isinstance(director.logs, pd.DataFrame)


def test_run():
    director = SampleInstallPhase(base_config, log_level="INFO")

    director.config = director.initialize_library(base_config)
    director.extract_defaults()

    director.run(until=10)

    assert director.env.now == 10


def test_logger():
    director = SampleInstallPhase(base_config, log_level="INFO")

    director.config = director.initialize_library(base_config)
    director.extract_defaults()

    director.logger.info(
        "Test Info Log", extra={"time": 0, "agent": "test_vessel"}
    )
    director.logger.debug("Test Debug Log")
    director.generate_log_df()

    df = director.logs
    assert "Test Info Log" in list(df["message"])
    assert "Test Debug Log" not in list(df["message"])


def test_comma_escape():
    director = SampleInstallPhase(base_config, log_level="INFO")

    director.config = director.initialize_library(base_config)
    director.extract_defaults()

    director.logger.info(
        "Test, Comma",
        extra={"time": 0, "location": "City, State", "agent": "test_vessel"},
    )
    director.generate_log_df()

    df = director.logs
    assert "Test, Comma" in list(df["message"])
    assert "City, State" in list(df["location"])
