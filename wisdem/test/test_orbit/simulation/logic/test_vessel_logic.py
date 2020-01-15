"""Tests for the vessel simulation logic."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.vessels import Vessel
from wisdem.test.test_orbit.vessels import WTIV_SPECS, FEEDER_SPECS
from wisdem.orbit.simulation import Environment, VesselStorage
from wisdem.orbit.simulation.logic import (
    get_item_from_storage,
    shuttle_items_to_queue,
    prep_for_site_operations,
)
