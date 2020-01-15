"""Tests for the `MonopileDesign` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import itertools
from copy import deepcopy

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
    {"rotor_diam": 155, "hub_height": 100, "rated_ws": 12},
    {"rotor_diam": 180, "hub_height": 112, "rated_ws": 12},
    {"rotor_diam": 205, "hub_height": 125, "rated_ws": 12},
    {"rotor_diam": 222, "hub_height": 136, "rated_ws": 12},
    {"rotor_diam": 248, "hub_height": 149, "rated_ws": 12},
]


def test_paramater_sweep():

    config = deepcopy(base)

    inputs = []
    lengths = []
    diameters = []
    weights = []

    for i in itertools.product(
        range(10, 51, 10),
        range(8, 13, 1),
        turbines,  # Depth  # Mean windspeed, site
    ):

        config["site"]["depth"] = i[0]
        config["site"]["mean_windspeed"] = i[1]

        _turb = i[2]
        config["turbine"]["rotor_diameter"] = _turb["rotor_diam"]
        config["turbine"]["hub_height"] = _turb["hub_height"]
        config["turbine"]["rated_windspeed"] = _turb["rated_ws"]

        m = MonopileDesign(config)
        m.run()

        # Monopile
        if not 10 < m._outputs["monopile"]["length"] < 130:
            print(
                f"Invalid monopile length encountered: {m._outputs['monopile']['length']}"
            )
            print(
                f"\tInputs: Depth: {i[0]}m, Windspeed: {i[1]}, Turbine: {i[2]}"
            )
            assert False

        if not 4 < m._outputs["monopile"]["diameter"] < 12:
            print(
                f"Invalid monopile diameter encountered: {m._outputs['monopile']['diameter']}"
            )
            print(
                f"\tInputs: Depth: {i[0]}m, Windspeed: {i[1]}, Turbine: {i[2]}"
            )
            assert False

        if not 200 < m._outputs["monopile"]["weight"] < 2000:
            print(
                f"Invalid monopile weight encountered: {m._outputs['monopile']['weight']}"
            )
            print(
                f"\tInputs: Depth: {i[0]}m, Windspeed: {i[1]}, Turbine: {i[2]}"
            )
            assert False

        # Transition Piece
        if not 4 < m._outputs["transition_piece"]["diameter"] < 12:
            print(
                f"Invalid transition piece diameter encountered: {m._outputs['transition_piece']['diameter']}"
            )
            print(
                f"\tInputs: Depth: {i[0]}m, Windspeed: {i[1]}, Turbine: {i[2]}"
            )
            assert False

        if not 100 < m._outputs["transition_piece"]["weight"] < 800:
            print(
                f"Invalid transition piece weight encountered: {m._outputs['transition_piece']['weight']}"
            )
            print(
                f"\tInputs: Depth: {i[0]}m, Windspeed: {i[1]}, Turbine: {i[2]}"
            )
            assert False


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
