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
    "mooring_system_design": {},
}


@pytest.mark.parametrize("depth", range(10, 1000, 100))
def test_depth_sweep(depth):

    config = deepcopy(base)
    config["site"]["depth"] = depth

    moor = MooringSystemDesign(config)
    moor.run()

    assert moor.design_result
    assert moor.total_cost


@pytest.mark.parametrize("rating", range(3, 15, 1))
def test_rating_sweep(rating):

    config = deepcopy(base)
    config["turbine"]["turbine_rating"] = rating

    moor = MooringSystemDesign(config)
    moor.run()

    assert moor.design_result
    assert moor.total_cost


def test_mooring_system_defaults():

    moor_base = MooringSystemDesign(base)
    moor_base.run()

    base_cost = moor_base.detailed_output["system_cost"]

    config_defs = deepcopy(base)
    config_defs["mooring_system_design"] = {}
    config_defs["mooring_system_design"]["mooring_type"] = "Catenary"
    config_defs["mooring_system_design"]["anchor_type"] = "Suction Pile"

    moor_defs = MooringSystemDesign(config_defs)
    moor_defs.run()

    assert moor_defs.detailed_output["system_cost"] == base_cost


def test_catenary_mooring_system_kwargs():

    test_kwargs = {
        "num_lines": 6,
        "anchor_type": "Drag Embedment",
        "mooring_line_cost_rate": 2500,
    }

    moor = MooringSystemDesign(base)
    moor.run()

    base_cost = moor.detailed_output["system_cost"]

    assert base_cost == pytest.approx(76173891, abs=1e0)

    for k, v in test_kwargs.items():
        config = deepcopy(base)
        config["mooring_system_design"] = {}
        config["mooring_system_design"][k] = v

        moor = MooringSystemDesign(config)
        moor.run()

        assert moor.detailed_output["system_cost"] != base_cost


def test_semitaut_mooring_system_kwargs():

    semi_base = deepcopy(base)
    semi_base["mooring_system_design"]["mooring_type"] = "SemiTaut"

    test_kwargs = {
        "num_lines": 6,
        "anchor_type": "Drag Embedment",
        "chain_density": 10000,
        "rope_density": 1000,
    }

    moor = MooringSystemDesign(semi_base)
    moor.run()

    base_cost = moor.detailed_output["system_cost"]

    assert base_cost == pytest.approx(102227311, abs=1e0)

    for k, v in test_kwargs.items():
        config = deepcopy(semi_base)
        config["mooring_system_design"] = {}
        config["mooring_system_design"][k] = v

        moor = MooringSystemDesign(config)
        moor.run()

        assert moor.detailed_output["system_cost"] != base_cost


def test_tlp_mooring_system_kwargs():

    tlp_base = deepcopy(base)
    tlp_base["mooring_system_design"]["mooring_type"] = "TLP"

    test_kwargs = {
        "num_lines": 6,
        "anchor_type": "Drag Embedment",
        "mooring_line_cost_rate": 2500,
        "draft_depth": 10,
    }

    moor = MooringSystemDesign(tlp_base)
    moor.run()

    base_cost = moor.detailed_output["system_cost"]

    assert base_cost == pytest.approx(57633231, abs=1e0)

    for k, v in test_kwargs.items():
        config = deepcopy(tlp_base)
        config["mooring_system_design"] = {}
        config["mooring_system_design"][k] = v

        moor = MooringSystemDesign(config)
        moor.run()

        assert moor.detailed_output["system_cost"] != base_cost


def test_drag_embedment_fixed_length():

    moor = MooringSystemDesign(base)
    moor.run()

    baseline = moor.line_length

    default = deepcopy(base)
    default["mooring_system_design"] = {"anchor_type": "Drag Embedment"}

    moor = MooringSystemDesign(default)
    moor.run()

    with_default = moor.line_length
    assert with_default > baseline

    custom = deepcopy(base)
    custom["mooring_system_design"] = {
        "anchor_type": "Drag Embedment",
        "drag_embedment_fixed_length": 1000,
    }

    moor = MooringSystemDesign(custom)
    moor.run()

    assert moor.line_length > with_default
    assert moor.line_length > baseline


def test_custom_num_lines():

    config = deepcopy(base)
    config["mooring_system_design"] = {"num_lines": 5}

    moor = MooringSystemDesign(config)
    moor.run()

    assert moor.design_result["mooring_system"]["num_lines"] == 5
