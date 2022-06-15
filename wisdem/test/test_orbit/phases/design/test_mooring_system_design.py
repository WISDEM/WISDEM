"""Tests for the `MooringSystemDesign` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.orbit.phases.design import MooringSystemDesign

base = {
    "site": {"depth": 200},
    "turbine": {"turbine_rating": 6},
    "plant": {"num_turbines": 50},
}


@pytest.mark.parametrize("depth", range(10, 1000, 100))
def test_depth_sweep(depth):

    config = deepcopy(base)
    config["site"]["depth"] = depth

    m = MooringSystemDesign(config)
    m.run()

    assert m.design_result
    assert m.total_cost


@pytest.mark.parametrize("rating", range(3, 15, 1))
def test_rating_sweeip(rating):

    config = deepcopy(base)
    config["turbine"]["turbine_rating"] = rating

    m = MooringSystemDesign(config)
    m.run()

    assert m.design_result
    assert m.total_cost


def test_drag_embedment_fixed_length():

    m = MooringSystemDesign(base)
    m.run()

    baseline = m.line_length

    default = deepcopy(base)
    default["mooring_system_design"] = {"anchor_type": "Drag Embedment"}

    m = MooringSystemDesign(default)
    m.run()

    with_default = m.line_length
    assert with_default > baseline

    custom = deepcopy(base)
    custom["mooring_system_design"] = {
        "anchor_type": "Drag Embedment",
        "drag_embedment_fixed_length": 1000,
    }

    m = MooringSystemDesign(custom)
    m.run()

    assert m.line_length > with_default
    assert m.line_length > baseline


def test_custom_num_lines():

    config = deepcopy(base)
    config["mooring_system_design"] = {"num_lines": 5}

    m = MooringSystemDesign(config)
    m.run()

    assert m.design_result["mooring_system"]["num_lines"] == 5
