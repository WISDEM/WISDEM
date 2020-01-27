"""Provides the `Port` class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy

from .exceptions import ItemNotFound


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

    def get_item(self, rule):
        """
        Checks self.items for an item satisfying 'rule'. Returns item if found,
        otherwise returns an error.

        Parameters
        ----------
        rule : tuple
            Tuple defining the rule to filter items by.
            - ('key': 'value')

        Returns
        -------
        res : FilterStoreGet
            Response from underlying FilterStore. Call 'res.value' for the
            underlying dictionary.
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
