"""Provides the `Vessel` class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

from collections import Counter, namedtuple

import numpy as np

from wisdem.orbit.vessels.tasks import defaults

from .components import Crane, JackingSys

Trip = namedtuple("Trip", "cargo_weight deck_space items")


class Vessel:
    """Base Vessel Class"""

    name = None

    def __init__(self, name, vessel_specs):
        """
        Creates an instance of Vessel.

        Parameters
        ----------
        vessel_specs : dict
            Nested dictionary containing vessel specifications.
        """

        self.name = name
        self.extract_vessel_specs(vessel_specs)

    def extract_vessel_specs(self, vessel_specs):
        """
        Extracts and defines vessel specifications.

        Parameters
        ----------
        vessel_specs : dict
            Nested dictionary containing vessel specifications.
        """

        transport_specs = vessel_specs.get("transport_specs", None)
        if transport_specs:
            self._transport_specs = transport_specs
            self.extract_transport_specs(transport_specs)

        jacksys_specs = vessel_specs.get("jacksys_specs", None)
        if jacksys_specs:
            self._jacksys_specs = jacksys_specs
            self.jacksys = JackingSys(jacksys_specs)

        crane_specs = vessel_specs.get("crane_specs", None)
        if crane_specs:
            self._crane_specs = crane_specs
            self.crane = Crane(crane_specs)

        storage_specs = vessel_specs.get("storage_specs", None)
        if storage_specs:
            self.trip_data = []
            self._storage_specs = storage_specs
            self.extract_storage_specs(storage_specs)

        cable_lay_specs = vessel_specs.get("cable_lay_specs", None)
        if cable_lay_specs:
            self.extract_cable_lay_specs(cable_lay_specs)

        scour_protection_specs = vessel_specs.get(
            "scour_protection_install_specs", None
        )
        if scour_protection_specs:
            self.extract_scour_protection_specs(scour_protection_specs)

    def extract_transport_specs(self, transport_specs):
        """
        Extracts and defines transport related specifications.

        Parameters
        ----------
        transport_specs : dict
            Dictionary containing transport related specifications.
        """

        self.transit_speed = transport_specs.get("transit_speed", None)

    def extract_storage_specs(self, storage_specs):
        """
        Extracts and defines storage system specifications.

        Parameters
        ----------
        storage_specs : dict
            Dictionary containing storage system specifications.
        """

        self.max_deck_space = storage_specs.get("max_deck_space", None)
        self.max_deck_load = storage_specs.get("max_deck_load", None)
        self.max_cargo = storage_specs.get("max_cargo", None)

    def extract_cable_lay_specs(self, cable_lay_specs):
        """
        Extracts and defines cable lay system specifications.

        Parameters
        ----------
        cable_lay_specs : dict
            Dictionary containing cable lay system specifications.
        """

        self.cable_lay_speed = cable_lay_specs.get(
            "cable_lay_speed", defaults["cable_lay_speed"]
        )
        self.max_cable_diameter = cable_lay_specs.get(
            "max_cable_diameter", None
        )

    def extract_scour_protection_specs(self, scour_protection_specs):
        """
        Extracts and defines scour protection installation specifications.

        Parameters
        ----------
        scour_protection_specs : dict
            Dictionary containing scour protection installation specifications.
        """

        self.scour_protection_install_speed = scour_protection_specs.get(
            "scour_protection_install_speed", 10
        )

    def transit_time(self, distance):
        """
        Calculates transit time for a given distance.

        Parameters
        ----------
        distance : int | float
            Distance to travel (km).

        Returns
        -------
        transit_time : float
            Time required to travel 'distance' (h).
        """

        transit_time = distance / self.transit_speed

        return transit_time

    @property
    def transit_limits(self):
        """
        Returns dictionary with 'max_windspeed' and 'max_waveheight'
        for transit.
        """

        _dict = {
            "max_windspeed": self._transport_specs["max_windspeed"],
            "max_waveheight": self._transport_specs["max_waveheight"],
        }

        return _dict

    @property
    def operational_limits(self):
        """
        Returns dictionary with 'max_windspeed' and 'max_waveheight'
        for operations.
        """

        crane = getattr(self, "crane", None)
        if crane is None:
            max_windspeed = self._transport_specs["max_windspeed"]

        else:
            max_windspeed = self._crane_specs["max_windspeed"]

        _dict = {
            "max_windspeed": max_windspeed,
            "max_waveheight": self._transport_specs["max_waveheight"],
        }

        return _dict

    def update_trip_data(self, cargo=True, deck=True, items=True):
        """
        Appends the current cargo utilization to the `self._cargo_utlization`.
        Used to collect cargo utilization statistics throughout a simulation.

        Parameters
        ----------
        items : bool
            Toggles optional item list collection.
        """

        storage = getattr(self, "storage", None)
        if storage is None:
            raise Exception("Vessel does not have storage capacity.")

        _cargo = storage.current_cargo_weight if cargo else np.NaN
        _deck = storage.current_deck_space if deck else np.NaN
        _items = (
            dict(Counter(i["type"] for i in storage.items))
            if items
            else np.NaN
        )

        trip = Trip(cargo_weight=_cargo, deck_space=_deck, items=_items)

        self.trip_data.append(trip)

    @property
    def cargo_weight_list(self):
        """Returns cargo weights trips in self.trip_data."""

        return [trip.cargo_weight for trip in self.trip_data]

    @property
    def cargo_weight_utilizations(self):
        """Returns cargo weight utilizations for list of trips."""

        return np.array(self.cargo_weight_list) / self.max_cargo

    @property
    def deck_space_list(self):
        """Returns deck space used for trips in self.trip_data."""

        return [trip.deck_space for trip in self.trip_data]

    @property
    def deck_space_utilizations(self):
        """Returns deck space utilizations for list of trips."""

        return np.array(self.deck_space_list) / self.max_deck_space

    @property
    def max_cargo_weight_utilization(self):
        """Returns maximum cargo weight utilization."""

        if not self.trip_data:
            return np.NaN

        return np.max(self.cargo_weight_utilizations)

    @property
    def min_cargo_weight_utilization(self):
        """Returns minimum cargo weight utilization."""

        if not self.trip_data:
            return np.NaN

        return np.min(self.cargo_weight_utilizations)

    @property
    def mean_cargo_weight_utilization(self):
        """Returns mean cargo weight utilization."""

        if not self.trip_data:
            return np.NaN

        return np.mean(self.cargo_weight_utilizations)

    @property
    def median_cargo_weight_utilization(self):
        """Returns median cargo weight utilization."""

        if not self.trip_data:
            return np.NaN

        return np.median(self.cargo_weight_utilizations)

    @property
    def max_deck_space_utilization(self):
        """Returns maximum deck_space utilization."""

        if not self.trip_data:
            return np.NaN

        return np.max(self.deck_space_utilizations)

    @property
    def min_deck_space_utilization(self):
        """Returns minimum deck_space utilization."""

        if not self.trip_data:
            return np.NaN

        return np.min(self.deck_space_utilizations)

    @property
    def mean_deck_space_utilization(self):
        """Returns mean deck space utilization."""

        if not self.trip_data:
            return np.NaN

        return np.mean(self.deck_space_utilizations)

    @property
    def median_deck_space_utilization(self):
        """Returns median deck space utilization."""

        if not self.trip_data:
            return np.NaN

        return np.median(self.deck_space_utilizations)

    @property
    def max_items_by_weight(self):
        """Returns items corresponding to `self.max_cargo_weight`."""

        if not self.trip_data:
            return np.NaN

        i = np.argmax(self.cargo_weight_list)
        return self.trip_data[i].items

    @property
    def min_items_by_weight(self):
        """Returns items corresponding to `self.min_cargo_weight`."""

        if not self.trip_data:
            return np.NaN

        i = np.argmin(self.cargo_weight_list)
        return self.trip_data[i].items

    @property
    def max_items_by_space(self):
        """Returns items corresponding to `self.max_deck_space`."""

        if not self.trip_data:
            return np.NaN

        i = np.argmax(self.deck_space_list)
        return self.trip_data[i].items

    @property
    def min_items_by_space(self):
        """Returns items corresponding to `self.min_deck_space`."""

        if not self.trip_data:
            return np.NaN

        i = np.argmin(self.deck_space_list)
        return self.trip_data[i].items
