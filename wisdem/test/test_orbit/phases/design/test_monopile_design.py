"""Tests for the `MonopileDesign` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy
from itertools import product

import pytest

from wisdem.orbit.phases.design import MonopileDesign

base = {
    "site": {"depth": 50, "mean_windspeed": 9},
    "plant": {"num_turbines": 20},
    "turbine": {
        "rotor_diameter": 150,
        "hub_height": 110,
        "rated_windspeed": 11,
    },
}

turbines = [
    {"rotor_diameter": 155, "hub_height": 100, "rated_windspeed": 12},
    {"rotor_diameter": 180, "hub_height": 112, "rated_windspeed": 12},
    {"rotor_diameter": 205, "hub_height": 125, "rated_windspeed": 12},
    {"rotor_diameter": 222, "hub_height": 136, "rated_windspeed": 12},
    {"rotor_diameter": 248, "hub_height": 149, "rated_windspeed": 12},
]


@pytest.mark.parametrize(
    "depth,mean_ws,turbine",
    product(range(10, 51, 10), range(8, 13, 1), turbines),
)
def test_paramater_sweep(depth, mean_ws, turbine):

    config = {
        "site": {"depth": depth, "mean_windspeed": mean_ws},
        "plant": {"num_turbines": 20},
        "turbine": turbine,
    }

    m = MonopileDesign(config)
    m.run()

    # Check valid monopile length
    assert 10 < m._outputs["monopile"]["length"] < 130

    # Check valid monopile diameter
    assert 4 < m._outputs["monopile"]["diameter"] < 13

    # Check valid monopile mass
    assert 200 < m._outputs["monopile"]["mass"] < 2500

    # Check valid transition piece diameter
    assert 4 < m._outputs["transition_piece"]["diameter"] < 14


def test_monopile_kwargs():

    test_kwargs = {
        "yield_stress": 400000000,
        "load_factor": 1.25,
        "material_factor": 1.2,
        "monopile_density": 9000,
        "monopile_modulus": 220e9,
        "soil_coefficient": 4500000,
        "air_density": 1.125,
        "weibull_scale_factor": 10,
        "weibull_shape_factor": 1.87,
        "turb_length_scale": 350,
    }

    m = MonopileDesign(base)
    m.run()
    base_results = m._outputs["monopile"]

    for k, v in test_kwargs.items():

        config = deepcopy(base)
        config["monopile_design"] = {}
        config["monopile_design"][k] = v

        m = MonopileDesign(config)
        m.run()
        results = m._outputs["monopile"]

        assert results != base_results


def test_transition_piece_kwargs():

    test_kwargs = {
        # Transition piece specific
        "monopile_tp_connection_thickness": 0.005,
        "transition_piece_density": 8200,
        "transition_piece_thickness": 0.12,
        "transition_piece_length": 30,
    }

    m = MonopileDesign(base)
    m.run()
    base_results = m._outputs["transition_piece"]

    for k, v in test_kwargs.items():

        config = deepcopy(base)
        config["monopile_design"] = {}
        config["monopile_design"][k] = v

        m = MonopileDesign(config)
        m.run()
        results = m._outputs["transition_piece"]

        assert results != base_results
