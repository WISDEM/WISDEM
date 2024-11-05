"""Provides the base `DesignPhase` class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from abc import abstractmethod

from wisdem.orbit.phases import BasePhase
from wisdem.orbit.core.defaults import common_costs


class DesignPhase(BasePhase):
    """BasePhase subclass for design modules."""

    expected_config = None
    output_config = None

    @property
    @abstractmethod
    def design_result(self):
        """
        Returns result of DesignPhase to be passed into config and consumed by
        InstallPhase.

        Returns
        -------
        dict
            Dictionary of design results.
        """

        return {}

    def get_default_cost(self, design_name, key, subkey=None):
        """Return the cost value for a key in a design
        dictionary read from common_cost.yaml.
        """

        if (design_dict := common_costs.get(design_name, None)) is None:
            raise KeyError(f"No {design_name} in common_cost.yaml.")

        if (cost_value := design_dict.get(key, None)) is None:
            raise KeyError(f"{key} not found in [{design_name}] common_costs.")

        if isinstance(cost_value, dict):
            if subkey is None:
                raise ValueError(
                    f"{key} is a dictionary and requires a 'subkey' input."
                )

            if (sub_cost_value := cost_value.get(subkey, None)) is None:
                raise KeyError(
                    f"{subkey} not found in [{design_name}][{cost_value}]"
                    " common_costs."
                )

            return sub_cost_value

        else:
            return cost_value
