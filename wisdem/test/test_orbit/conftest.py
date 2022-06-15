"""Shared pytest settings and fixtures."""


import os

import pytest
from marmot import Environment

from wisdem.orbit.core import Vessel
from wisdem.orbit.core.library import initialize_library, extract_library_specs
from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.phases.install.cable_install import SimpleCable


def pytest_configure():
    """Creates the default library for pytest testing suite and initializes it
    when required.
    """

    test_dir = os.path.split(os.path.abspath(__file__))[0]
    pytest.library = os.path.join(test_dir, "data", "library")
    initialize_library(pytest.library)


@pytest.fixture()
def env():

    return Environment("Test Environment", state=test_weather)


@pytest.fixture()
def wtiv():

    specs = extract_library_specs("wtiv", "test_wtiv")
    return Vessel("Test WTIV", specs)


@pytest.fixture()
def feeder():

    specs = extract_library_specs("feeder", "test_feeder")
    return Vessel("Test Feeder", specs)


@pytest.fixture()
def cable_vessel():

    specs = extract_library_specs("array_cable_install_vessel", "test_cable_lay_vessel")
    return Vessel("Test Cable Vessel", specs)


@pytest.fixture()
def heavy_lift():

    specs = extract_library_specs("oss_install_vessel", "test_heavy_lift_vessel")
    return Vessel("Test Heavy Vessel", specs)


@pytest.fixture()
def spi_vessel():

    specs = extract_library_specs("spi_vessel", "test_scour_protection_vessel")
    return Vessel("Test SPI Vessel", specs)


@pytest.fixture()
def simple_cable():

    return SimpleCable(linear_density=50.0)


@pytest.fixture(scope="function")
def tmp_yaml_del():

    yield
    os.remove("tmp.yaml")
