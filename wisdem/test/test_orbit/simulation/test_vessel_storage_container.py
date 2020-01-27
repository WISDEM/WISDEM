"""Tests for the `VesselStorageContainer` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import pytest

from wisdem.orbit.simulation import Environment, VesselStorageContainer
from wisdem.orbit.simulation.exceptions import (
    DeckSpaceExceeded,
    InsufficientAmount,
    CargoWeightExceeded,
    ItemPropertyNotDefined,
)

env = Environment()
storage_args = {"max_cargo": 1000, "max_deck_load": 1}

sample_items = {
    "typical": {
        "name": "Item1",
        "type": "Carousel",
        "weight": 500,
        "deck_space": 1,
    },
    "typical2": {
        "name": "Item2",
        "type": "Carousel",
        "weight": 500,
        "deck_space": 1,
    },
    "overweight": {
        "name": "Item3",
        "type": "Carousel",
        "weight": 2000,
        "deck_space": 1,
    },
    "ill-defined": {"name": "Item4", "type": "Carousel"},
}


def test_creation():
    storage = VesselStorageContainer(env, **storage_args, capacity=1)

    assert storage.capacity == storage_args["max_cargo"]
    assert storage.max_cargo_weight == storage_args["max_cargo"]
    assert storage.max_deck_space == 1
    assert storage.max_deck_load == 1

    assert storage.current_cargo_weight == 0
    assert storage.current_deck_space == 0


def test_put_item():
    storage = VesselStorageContainer(env, **storage_args)

    storage.put_item(sample_items["typical"])

    assert storage.current_cargo_weight == sample_items["typical"]["weight"]
    assert storage.current_deck_space == sample_items["typical"]["deck_space"]


def test_get_item():
    storage = VesselStorageContainer(env, **storage_args)

    storage.put_item(sample_items["typical"])
    storage.get_item("cable", 400)

    assert (
        storage.current_cargo_weight == sample_items["typical"]["weight"] - 400
    )
    assert storage.current_deck_space == sample_items["typical"]["deck_space"]


def test_max_cargo_weight():
    storage = VesselStorageContainer(env, **storage_args)

    with pytest.raises(CargoWeightExceeded):
        storage.put_item(sample_items["overweight"])


def test_max_deck_space():
    storage = VesselStorageContainer(env, **storage_args)

    storage.put_item(sample_items["typical"])
    with pytest.raises(DeckSpaceExceeded):
        storage.put_item(sample_items["typical2"])


def test_bad_items():
    storage = VesselStorageContainer(env, **storage_args)

    with pytest.raises(ItemPropertyNotDefined):
        storage.put_item(sample_items["ill-defined"])
