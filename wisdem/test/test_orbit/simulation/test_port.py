"""Tests for the `Port` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import pytest

from wisdem.orbit.simulation import Port, Environment
from wisdem.orbit.simulation.port import ItemNotFound


def test_port_creation():

    env = Environment()
    test_item = {"type": "TestItem"}

    port = Port(env)
    port.put(test_item)
    port.put(test_item)
    items = [item for item in port.items if item["type"] == "TestItem"]
    assert len(items) == 2


def test_get_item():

    env = Environment()
    test_item = {"type": "TestItem"}

    port = Port(env)
    port.put(test_item)
    port.put(test_item)

    returned = port.get_item(("type", "TestItem"))
    assert returned.value["type"] == "TestItem"
    assert len(port.items) == 1

    with pytest.raises(ItemNotFound):
        _ = port.get_item(("type", "WrongItem"))

    _ = port.get_item(("type", "TestItem"))
    with pytest.raises(ItemNotFound):
        _ = port.get_item(("type", "TestItem"))
