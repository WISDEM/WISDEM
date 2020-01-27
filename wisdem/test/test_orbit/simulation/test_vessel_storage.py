"""Tests for the `VesselStorage` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import pytest

from wisdem.orbit.simulation import Environment, VesselStorage
from wisdem.orbit.simulation.exceptions import (
    ItemNotFound,
    DeckSpaceExceeded,
    CargoWeightExceeded,
    ItemPropertyNotDefined,
)

env = Environment()
storage_args = {"max_cargo": 1000, "max_deck_space": 1000, "max_deck_load": 10}

sample_items = [
    {"name": "Item1", "type": "TestItem", "weight": 500, "deck_space": 500},
    {"name": "Item2", "type": "TestItem", "weight": 499, "deck_space": 499},
    {"name": "Item3", "type": "TestItem", "weight": 10, "deck_space": 10},
]


no_space = {"name": "Item4", "type": "TestItem", "weight": 10, "deck_space": 0}


bad_item = {"name": "Item4", "type": "TestItem"}


def test_creation():

    storage = VesselStorage(env, **storage_args, capacity=1)

    assert storage.capacity == 1
    assert storage.max_cargo_weight == 1000
    assert storage.max_deck_space == 1000
    assert storage.max_deck_load == 10


def test_max_cargo_weight():

    storage = VesselStorage(env, **storage_args)

    for item in sample_items[:2]:
        storage.put_item(item)

    with pytest.raises(CargoWeightExceeded):
        storage.put_item(no_space)


def test_max_deck_space():

    storage = VesselStorage(env, **storage_args)

    for item in sample_items[:2]:
        storage.put_item(item)

    storage = VesselStorage(env, **storage_args)
    with pytest.raises(DeckSpaceExceeded):
        for item in sample_items[:3]:
            storage.put_item(item)


def test_current_cargo_weight():

    storage = VesselStorage(env, **storage_args)

    for item in sample_items[:2]:
        storage.put_item(item)

    assert storage.current_cargo_weight == 999


def test_current_deck_space():

    storage = VesselStorage(env, **storage_args)

    for item in sample_items[:2]:
        storage.put_item(item)

    assert storage.current_deck_space == 999


def test_bad_items():

    storage = VesselStorage(env, **storage_args)

    with pytest.raises(ItemPropertyNotDefined):
        storage.put_item(bad_item)


def test_get_item():

    storage = VesselStorage(env, **storage_args)

    for item in sample_items[:2]:
        storage.put_item(item)

    returned = storage.get_item(("type", "TestItem"))
    assert returned.value["type"] == "TestItem"
    assert len(storage.items) == 1

    with pytest.raises(ItemNotFound):
        _ = storage.get_item(("type", "WrongItem"))

    _ = storage.get_item(("type", "TestItem"))
    with pytest.raises(ItemNotFound):
        _ = storage.get_item(("type", "TestItem"))


def test_any_remaining():

    storage = VesselStorage(env, **storage_args)

    for item in sample_items[:2]:
        storage.put_item(item)

    assert storage.any_remaining(("type", "TestItem"))

    _ = storage.get_item(("type", "TestItem"))
    _ = storage.get_item(("type", "TestItem"))

    assert storage.any_remaining(("type", "TestItem")) is False
