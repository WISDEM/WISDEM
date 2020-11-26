"""Test suite for the library module."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"

import os
from copy import deepcopy

import pytest

from wisdem.orbit import ProjectManager
from wisdem.orbit.core import library
from wisdem.orbit.core.exceptions import LibraryItemNotFoundError

config = {
    "site": {"distance": 100, "depth": 15},
    "plant": {"num_turbines": 10},
    "turbine": {
        "hub_height": 100,
        "tower": {"type": "Tower", "deck_space": 100, "mass": 400},
        "nacelle": {"type": "Nacelle", "deck_space": 200, "mass": 400},
        "blade": {"type": "Blade", "deck_space": 100, "mass": 100},
    },
    "MonopileInstallation": {"wtiv": "test_wtiv"},
    "TurbineInstallation": {"wtiv": "test_wtiv"},
    "port": {"num_cranes": 1, "monthly_rate": 10000},
    "monopile": {
        "type": "Monopile",
        "length": 50,
        "diameter": 10,
        "deck_space": 500,
        "mass": 350,
    },
    "transition_piece": {
        "type": "Transition Piece",
        "deck_space": 250,
        "mass": 350,
    },
    "install_phases": ["MonopileInstallation", "TurbineInstallation"],
}


def test_initialize_library():
    library.initialize_library(None)
    assert os.environ["DATA_LIBRARY"]

    _ = os.environ.pop("DATA_LIBRARY")
    library.initialize_library(pytest.library)
    assert os.environ["DATA_LIBRARY"] == pytest.library


def test_extract_library_specs_fail():
    library.initialize_library(pytest.library)
    with pytest.raises(LibraryItemNotFoundError):
        library.extract_library_specs("turbine", "unknown.yaml")


def test_phase_specific_file_extraction():

    project = ProjectManager(config)
    turbine_config = project.create_config_for_phase("TurbineInstallation")
    monopile_config = project.create_config_for_phase("MonopileInstallation")

    assert isinstance(turbine_config["wtiv"], dict)
    assert isinstance(monopile_config["wtiv"], dict)

    bad_config = deepcopy(config)
    _ = bad_config.pop("TurbineInstallation")
    bad_config["wtiv"] = "example_wtiv"
    bad_config["MonopileInstallation"]["wtiv"] = "missing_vessel"

    with pytest.raises(LibraryItemNotFoundError):
        bad_project = ProjectManager(bad_config)
