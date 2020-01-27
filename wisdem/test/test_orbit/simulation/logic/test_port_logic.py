"""Tests for port simulation logic."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy
import pytest

from wisdem.orbit.simulation.exceptions import FastenTimeNotFound
from wisdem.orbit.simulation.logic.port_logic import vessel_fasten_time

item_list = [
    "Blade",
    "Nacelle",
    "Tower Section",
    "Monopile",
    "Transition Piece",
    "Scour Protection",
    "Topside",
    "Carousel",
    "Non Existent",
]
item_list = [{"type": item} for item in item_list]


@pytest.mark.parametrize("item", item_list)
def test_vessel_fasten_time(item):
    if item["type"] == "Non Existent":
        with pytest.raises(FastenTimeNotFound):
            vessel_fasten_time(item)
    else:
        assert vessel_fasten_time(item) > 0
