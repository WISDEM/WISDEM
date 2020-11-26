"""Provides a testing framework for the `Cable` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import copy
import itertools

import numpy as np
import pytest

from wisdem.orbit.phases.design._cables import Cable, Plant

cables = {
    "empty": {},
    "passes": {
        "conductor_size": 400,
        "current_capacity": 610,
        "rated_voltage": 33,
        "ac_resistance": 0.06,
        "inductance": 0.375,
        "capacitance": 225,
        "linear_density": 35,
        "cost_per_km": 300000,
        "name": "passes",
    },
}

plant_config_calculate_all_ring = {
    "site": {"depth": 20},
    "plant": {
        "layout": "ring",
        "row_spacing": 7,
        "turbine_spacing": 5,
        "num_turbines": 40,
    },
    "turbine": {"rotor_diameter": 154, "turbine_rating": 10},
}

plant_config_calculate_all_grid = {
    "site": {"depth": 20},
    "plant": {
        "layout": "grid",
        "row_spacing": 7,
        "turbine_spacing": 5,
        "num_turbines": 40,
    },
    "turbine": {"rotor_diameter": 154, "turbine_rating": 10},
}

plant_config_distance_provided_ring = {
    "site": {"depth": 20},
    "plant": {
        "layout": "ring",
        "row_distance": 0.4,
        "turbine_distance": 0.1,
        "num_turbines": 40,
        "substation_distance": 0.2,
    },
    "turbine": {"rotor_diameter": 154, "turbine_rating": 10},
}

plant_config_distance_provided_grid = {
    "site": {"depth": 20},
    "plant": {
        "layout": "grid",
        "row_distance": 0.4,
        "turbine_distance": 0.1,
        "num_turbines": 40,
        "substation_distance": 0.2,
    },
    "turbine": {"rotor_diameter": 154, "turbine_rating": 10},
}

plant_config_custom = {
    "site": {"depth": 20},
    "plant": {
        "layout": "custom",
        "row_distance": 0.4,
        "turbine_distance": 0.1,
        "num_turbines": 40,
        "substation_distance": 0.2,
    },
    "turbine": {"rotor_diameter": 154, "turbine_rating": 10},
}


def test_cable_creation():
    cable = Cable(cables["passes"])

    assert cable
    for r in cable.required:
        assert getattr(cable, r, None) == cables["passes"][r]


def test_cable_required_inputs():
    with pytest.raises(ValueError):
        Cable(cables["empty"])


def test_power_factor():
    c = copy.deepcopy(cables["passes"])

    results = []
    for i in itertools.product(
        range(100, 1001, 150),  # conductor size
        np.arange(0.01, 0.91, 0.1),  # ac_resistance
        np.arange(0, 1, 0.15),  # inductance
        range(100, 1001, 150),  # capacitance
    ):

        c["conductor_size"] = i[0]
        c["ac_resistance"] = i[1]
        c["inductance"] = i[2]
        c["capacitance"] = i[3]

        cable = Cable(c)
        results.append(cable.power_factor)

    if any((a < 0) | (a > 1) for a in results):
        raise Exception("Invalid Power Factor.")


@pytest.mark.parametrize(
    "config",
    (
        plant_config_calculate_all_ring,
        plant_config_calculate_all_grid,
        plant_config_distance_provided_ring,
        plant_config_distance_provided_grid,
    ),
    ids=["calculate_ring", "calculate_grid", "provided_ring", "provided_grid"],
)
def test_plant_creation(config):
    plant = Plant(config)

    assert plant.turbine_rating == config["turbine"]["turbine_rating"]
    assert plant.site_depth == config["site"]["depth"] / 1000.0
    assert plant.layout == config["plant"]["layout"]
    assert plant.num_turbines == config["plant"]["num_turbines"]

    if "turbine_spacing" in config["plant"]:
        td = config["turbine"]["rotor_diameter"] * config["plant"]["turbine_spacing"] / 1000.0
    else:
        td = config["plant"]["turbine_distance"]
    assert plant.turbine_distance == td

    if "row_spacing" in config["plant"]:
        if config["plant"]["layout"] == "grid":
            rd = config["turbine"]["rotor_diameter"] * config["plant"]["row_spacing"] / 1000.0
        if config["plant"]["layout"] == "ring":
            rd = td
    else:
        rd = config["plant"]["row_distance"]
    assert plant.row_distance == rd

    if "substation_distance" in config["plant"]:
        sd = config["plant"]["substation_distance"]
    else:
        sd = td
    assert plant.substation_distance == sd


def test_custom_plant_creation():
    plant = Plant(plant_config_custom)

    for attr in ("row_distance", "turbine_distance", "substation_distance"):
        assert getattr(plant, attr, None) is None
