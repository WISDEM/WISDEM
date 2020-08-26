"""Provides the `Port` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy

from wisdem.orbit.core.exceptions import ItemNotFound


class Port(simpy.FilterStore):
    """Port Class"""

    def __init__(self, env, **kwargs):
        """
        Creates an instance of Port.

        Parameters
        ----------
        env : simpy.Environment
            SimPy environment that simulation runs on.
        """

        capacity = kwargs.get("capacity", float("inf"))
        super().__init__(env, capacity)

    def get_item(self, _type):
        """
        Checks self.items for an item satisfying `item.type = _type`, otherwise
        returns `ItemNotFound`.

        Parameters
        ----------
        _type : str
            Type of item to match. Checks `item.type`.

        Returns
        -------
        res.value : FilterStoreGet.value
            Returned item.

        Raises
        ------
        ItemNotFound
        """

        target = None
        for i in self.items:
            try:
                if i.type == _type:
                    target = i
                    break

            except AttributeError:
                continue

        if not target:
            raise ItemNotFound(_type)

        else:
            res = self.get(lambda x: x == target)
            return res.value


class WetStorage(simpy.Store):
    """Storage infrastructure for floating substructures."""

    def __init__(self, env, capacity):
        """
        Creates an instance of WetStorage.

        Parameters
        ----------
        capacity : int
            Number of substructures or assemblies that can be stored.
        """

        super().__init__(env, capacity)
