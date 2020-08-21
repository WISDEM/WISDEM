"""Tests for the `InstallPhase` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import pandas as pd
import pytest
from marmot import Environment

from wisdem.orbit.phases.install import InstallPhase


class BadInstallPhase(InstallPhase):
    """Subclass for testing InstallPhase."""

    def __init__(self, config, **kwargs):
        """Creates an instance of BadInstallPhase."""

        self.config = config
        self.env = Environment()


class SampleInstallPhase(InstallPhase):
    """Subclass for testing InstallPhase."""

    phase = "SampleInstallPhase"

    def __init__(self, config, **kwargs):
        """Creates an instance of SampleInstallPhase."""

        self.config = config
        self.env = Environment()

    def detailed_output(self):
        pass

    def setup_simulation(self):
        pass


base_config = {"port": {"num_cranes": 1, "name": "TEST_PORT"}}


def test_abstract_methods():

    with pytest.raises(TypeError):
        install = BadInstallPhase(base_config)

    install = SampleInstallPhase(base_config)


def test_run():

    sim = SampleInstallPhase(base_config)
    sim.run(until=10)

    assert sim.env.now == 10
