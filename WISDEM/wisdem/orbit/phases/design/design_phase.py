"""Provides the base `DesignPhase` class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from abc import abstractmethod

from wisdem.orbit.phases import BasePhase


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
