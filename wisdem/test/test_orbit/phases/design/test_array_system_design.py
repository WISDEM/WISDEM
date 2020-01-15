"""Tests for the `ArraySystemDesign` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import os
from copy import deepcopy

import numpy as np
import pytest

from wisdem.orbit.phases.design import ArraySystemDesign, CustomArraySystemDesign
from wisdem.orbit.simulation.exceptions import LibraryItemNotFoundError

_ = os.environ.pop("DATA_LIBRARY", None)


ROOT = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
TEST_LIBRARY = os.path.join(ROOT, "data", "library")

config_full_ring = {
    "site": {"depth": 20},
    "plant": {
        "layout": "ring",
        "row_spacing": 7,
        "turbine_spacing": 5,
        "num_turbines": 40,
        "substation_distance": 1,
    },
    "turbine": {"rotor_diameter": 154, "turbine_rating": 10},
    "array_system_design": {"cables": ["XLPE_400mm_33kV", "XLPE_630mm_33kV"]},
}

config_partial_ring = deepcopy(config_full_ring)
config_partial_ring["plant"]["num_turbines"] = 49

config_full_grid = deepcopy(config_full_ring)
config_full_grid["plant"]["layout"] = "grid"

config_partial_grid = deepcopy(config_full_grid)
config_partial_grid["plant"]["num_turbines"] = 49

config_cables_from_file_fail = deepcopy(config_full_grid)
config_cables_from_file_fail["array_system_design"]["cables"] = "Cable1"

config_custom_base = deepcopy(config_full_grid)
config_custom_base["plant"]["num_turbines"] = 8
config_custom_base["plant"]["layout"] = "custom"

config_incorrect_turbines = deepcopy(config_custom_base)
config_incorrect_turbines["plant"]["num_turbines"] = 11
config_incorrect_turbines["array_system_design"]["location_data"] = "passes"

config_missing_col = deepcopy(config_custom_base)
config_missing_col["array_system_design"]["location_data"] = "missing_columns"

config_incomplete_optional = deepcopy(config_custom_base)
config_incomplete_optional["array_system_design"][
    "location_data"
] = "incomplete_optional"

config_incomplete_required = deepcopy(config_custom_base)
config_incomplete_required["array_system_design"][
    "location_data"
] = "incomplete_required"

config_duplicate_coordinates = deepcopy(config_custom_base)
config_duplicate_coordinates["array_system_design"][
    "location_data"
] = "duplicate_coordinates"


def test_array_system_creation():
    array = ArraySystemDesign(config_full_grid, library_path=TEST_LIBRARY)
    array.run()
    assert array


def test_cable_not_found():
    array = ArraySystemDesign(
        config_cables_from_file_fail, library_path=TEST_LIBRARY
    )
    with pytest.raises(LibraryItemNotFoundError):
        array.run()


@pytest.mark.parametrize(
    "config,num_full_strings,num_partial_strings,num_turbines_full_string,num_turbines_partial_string",
    (
        (config_full_ring, 10, 0, 4, 0),
        (config_partial_ring, 12, 1, 4, 1),
        (config_full_grid, 10, 0, 4, 0),
        (config_partial_grid, 12, 1, 4, 1),
    ),
)
def test_string_creation(
    config,
    num_full_strings,
    num_partial_strings,
    num_turbines_full_string,
    num_turbines_partial_string,
):
    array = ArraySystemDesign(config, library_path=TEST_LIBRARY)
    array.run()

    assert array

    string_properties = (
        (num_full_strings, "num_full_strings"),
        (num_partial_strings, "num_partial_strings"),
        (num_turbines_full_string, "num_turbines_full_string"),
        (num_turbines_partial_string, "num_turbines_partial_string"),
    )
    for val, name in string_properties:
        assert val == array.__getattribute__(name)


@pytest.mark.parametrize(
    "turbine_rating,expected",
    ((1, 40), (2, 20), (4.5, 8), (6, 6), (8.5, 4), (12, 3)),
)
def test_max_turbines_per_cable(turbine_rating, expected):
    config = deepcopy(config_full_grid)
    config["array_system_design"]["cables"] = "XLPE_630mm_33kV"
    config["turbine"]["turbine_rating"] = turbine_rating
    array = ArraySystemDesign(config, library_path=TEST_LIBRARY)
    array.run()
    assert array.cables["XLPE_630mm_33kV"].max_turbines == expected


@pytest.mark.parametrize(
    "config,shape,num_null",
    (
        (config_full_ring, (10, 4), 0),
        (config_partial_ring, (13, 4), 3),
        (config_full_grid, (10, 4), 0),
        (config_partial_grid, (13, 4), 3),
    ),
)
def test_grid_creation(config, shape, num_null):
    array = ArraySystemDesign(config, library_path=TEST_LIBRARY)
    array.run()

    assert array.turbines_x.shape == shape
    assert array.turbines_y.shape == shape
    assert np.isnan(array.turbines_x).sum() == num_null
    assert np.isnan(array.turbines_y).sum() == num_null

    assert isinstance(array.oss_x, float)
    assert isinstance(array.oss_y, float)

    shape = (shape[0], shape[1] + 1, 2)
    assert array.coordinates.shape == shape
    assert np.isnan(array.coordinates).sum() == 2 * num_null


@pytest.mark.parametrize(
    "config,total_length",
    (
        (config_full_ring, 34.7),
        (config_partial_ring, 42.68),
        (config_full_grid, 53.29),
        (config_partial_grid, 77.0),
    ),
)
def test_total_cable_length(config, total_length):
    array = ArraySystemDesign(config, library_path=TEST_LIBRARY)
    array.run()

    val = round(
        sum(val.sum() for val in array.cable_lengths_by_type.values()), 2
    )
    assert total_length == val

    val = round(sum(array.total_cable_length_by_type.values()), 2)
    assert total_length == val

    val = round(
        sum(
            length * n
            for val in array.design_result["array_system"]["cables"].values()
            for length, n in val["cable_sections"]
        ),
        2,
    )
    assert total_length == val


def test_missing_columns():
    os.environ.pop("DATA_LIBRARY")
    array = CustomArraySystemDesign(
        config_missing_col, library_path=TEST_LIBRARY
    )

    with pytest.raises(ValueError):
        array.run()


def test_duplicate_turbine_coordinates():
    _ = os.environ.pop("DATA_LIBRARY", None)
    array = CustomArraySystemDesign(
        config_duplicate_coordinates, library_path=TEST_LIBRARY
    )

    with pytest.raises(ValueError):
        array.run()


def test_incomplete_required_columns():
    _ = os.environ.pop("DATA_LIBRARY", None)
    array = CustomArraySystemDesign(
        config_incomplete_required, library_path=TEST_LIBRARY
    )

    with pytest.raises(ValueError):
        array.run()


def test_incomplete_optional_columns():
    _ = os.environ.pop("DATA_LIBRARY", None)
    array = CustomArraySystemDesign(
        config_incomplete_optional, library_path=TEST_LIBRARY
    )

    with pytest.warns(UserWarning):
        array.run()


def test_correct_turbines():
    _ = os.environ.pop("DATA_LIBRARY", None)
    array = CustomArraySystemDesign(
        config_incorrect_turbines, library_path=TEST_LIBRARY
    )

    with pytest.raises(ValueError):
        array.run()
