"""Provides the `VesselStorage` and `VesselStorageContainer` classes."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy

from wisdem.orbit.simulation.exceptions import (
    ItemNotFound,
    DeckSpaceExceeded,
    InsufficientAmount,
    CargoWeightExceeded,
    ItemPropertyNotDefined,
)


class VesselStorage(simpy.FilterStore):
    """Vessel Storage Class"""

    required_keys = ["type", "weight", "deck_space"]

    def __init__(
        self, env, max_cargo, max_deck_space, max_deck_load, **kwargs
    ):
        """
        Creates an instance of VesselStorage.

        Parameters
        ----------
        env : simpy.Environment
            SimPy environment that simulation runs on.
        max_cargo : int | float
            Maximum weight the storage system can carry (t).
        max_deck_space : int | float
            Maximum deck space the storage system can use (m2).
        max_deck_load : int | float
            Maximum deck load that the storage system can apply (t/m2).
        """

        capacity = kwargs.get("capacity", float("inf"))
        super().__init__(env, capacity)

        self.max_cargo_weight = max_cargo
        self.max_deck_space = max_deck_space
        self.max_deck_load = max_deck_load

    @property
    def current_cargo_weight(self):
        """Returns current cargo weight in tons."""

        return sum([item["weight"] for item in self.items])

    @property
    def current_deck_space(self):
        """Returns current deck space used in m2."""

        return sum([item["deck_space"] for item in self.items])

    def put_item(self, item):
        """
        Checks VesselStorage specific constraints and triggers self.put()
        if successful.

        Items put into the instance should be a dictionary with the following
        attributes:
        - name
        - weight (t)
        - deck_space (m2)

        Parameters
        ----------
        item : dict
            Dictionary of item properties.
        """

        if any(x not in item.keys() for x in self.required_keys):
            raise ItemPropertyNotDefined(item, self.required_keys)

        if self.current_deck_space + item["deck_space"] > self.max_deck_space:
            raise DeckSpaceExceeded(
                self.max_deck_space, self.current_deck_space, item
            )

        if self.current_cargo_weight + item["weight"] > self.max_cargo_weight:
            raise CargoWeightExceeded(
                self.max_cargo_weight, self.current_cargo_weight, item
            )

        self.put(item)

    def get_item(self, rule):
        """
        Checks self.items for an item satisfying 'rule'. Returns item if found,
        otherwise returns an error.

        Parameters
        ----------
        rule : tuple
            Tuple defining the rule to filter items by.
            - ('key': 'value')

        Yields
        ------
        response :
        """

        _key, _value = rule

        target = None
        for item in self.items:
            try:
                if item[_key] == _value:
                    target = item

            except KeyError:
                pass

        if not target:
            raise ItemNotFound(rule)

        else:
            res = self.get(lambda x: x == target)
            return res

    def any_remaining(self, rule):
        """
        Checks self.items for any items satisfying 'rule'. Returns True/False.
        Used to trigger vessel release if empty without having to wait for next
        self.get_item() iteration.

        Parameters
        ----------
        rule : tuple
            Tuple defining the rule to filter items by.
            - ('key': 'value')

        Returns
        -------
        resp : bool
            Indicates if any items in self.items satisfy 'rule'.
        """

        _key, _value = rule

        target = None
        for item in self.items:
            try:
                if item[_key] == _value:
                    target = item

            except KeyError:
                pass

        if target:
            return True

        else:
            return False


class VesselStorageContainer(simpy.Container):
    """Vessel Storage Class"""

    required_keys = ["weight", "deck_space"]

    def __init__(self, env, max_cargo, max_deck_load, **kwargs):
        """
        Creates an instance of VesselStorage.

        Parameters
        ----------
        env : simpy.Environment
            SimPy environment that simulation runs on.
        max_cargo : int | float
            Maximum weight the storage system can carry (t).
        """

        self.max_cargo_weight = max_cargo
        super().__init__(env, self.max_cargo_weight)
        self.deck_space = 0

        # Only needed for port interactions
        self.max_deck_space = 1
        self.max_deck_load = max_deck_load

    @property
    def current_cargo_weight(self):
        """
        Returns current cargo weight in tonnes.
        NOTE: Only necessary to interact with port.
        """

        return self.level

    @property
    def current_deck_space(self):
        """
        Returns current deck space used in m2.
        NOTE: Only necessary to interact with port.
        """

        return self.deck_space

    def put_item(self, item):
        """
        A wrapper for simpy.Container.put that checks VesselStorageContainer
        constraints and triggers self.put() if successful.

        Items put into the instance should be a dictionary with the following
        attributes:
         - name
         - weight (t)
         - length (km)

        Parameters
        ----------
        item : dict
            Dictionary of item properties.
        """

        if any(x not in item.keys() for x in self.required_keys):
            raise ItemPropertyNotDefined(item, self.required_keys)

        if self.current_deck_space + item["deck_space"] > self.max_deck_space:
            raise DeckSpaceExceeded(
                self.max_deck_space, self.current_deck_space, item
            )

        if self.current_cargo_weight + item["weight"] > self.max_cargo_weight:
            raise CargoWeightExceeded(
                self.max_cargo_weight, self.current_cargo_weight, item
            )

        self.deck_space += item["deck_space"]
        self.put(item["weight"])

    def get_item(self, item_type, item_amount):
        """
        Checks if there is enough of item, otherwise returns an error.

        Parameters
        ----------
        item_type : str
            Short, descriptive name of the item being accessed.
        item_amount : int or float
            Amount of the item to be loaded into storage.
        """

        if self.current_cargo_weight < item_amount:
            raise InsufficientAmount(
                self.current_cargo_weight, item_type, item_amount
            )

        return self.get(item_amount)
