"""Tests for the `ScourProtectionDesign` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


from copy import deepcopy

import numpy as np
import pytest

from wisdem.orbit.phases.design import ScourProtectionDesign

config_min_defined = {
    "monopile": {"diameter": 9},
    "plant": {"num_turbines": 50},
    "scour_protection_design": {"cost_per_tonne": 400},
}

config_fully_defined = {
    "monopile": {"diameter": 10},
    "plant": {"num_turbines": 50},
    "scour_protection_design": {
        "rock_density": 2300,
        "cost_per_tonne": 400,
        "design_time": 500,
        "soil_friction_angle": 33.0,
        "scour_depth_equilibrium": 1.2,
        "scour_protection_depth": 0.3,
    },
}


def test_default_setup():
    scour = ScourProtectionDesign(config_min_defined)
    scour.run()

    assert scour.phi == 33.5
    assert scour.equilibrium == 1.3
    assert scour.rock_density == 2600


def test_fully_defined_setup():
    scour = ScourProtectionDesign(config_fully_defined)
    scour.run()

    design = config_fully_defined["scour_protection_design"]

    assert scour.phi == design["soil_friction_angle"]
    assert scour.equilibrium == design["scour_depth_equilibrium"]
    assert scour.rock_density == design["rock_density"]


@pytest.mark.parametrize(
    "config,expected",
    ((config_fully_defined, 1141), (config_min_defined, 3852)),
    ids=["fully_defined", "minimally_defined"],
)
def test_tonnes_per_substructure(config, expected):
    scour = ScourProtectionDesign(config)
    scour.run()

    assert scour.scour_protection_tonnes == expected


@pytest.mark.parametrize(
    "config",
    (config_fully_defined, config_min_defined),
    ids=["fully_defined", "minimally_defined"],
)
def test_total_cost(config):
    scour = ScourProtectionDesign(config_min_defined)
    scour.run()

    cost = (
        config["plant"]["num_turbines"]
        * config["scour_protection_design"]["cost_per_tonne"]
        * scour.scour_protection_tonnes
    )
    assert scour.total_cost == pytest.approx(cost, rel=1e-8)
