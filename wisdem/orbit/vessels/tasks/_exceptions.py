"""
Jake Nunemaker
National Renewable Energy Lab
07/11/2019

This module contains custom exceptions for vessel processes.
"""


class MissingComponent(Exception):
    """Error for a missing component on a vessel."""

    def __init__(self, vessel, component, action):
        """
        Creates an instance of MissingComponent.

        Parameters
        ----------
        vessel : Vessel
        component : str | list
            Missing required component.
        action : str
            Action that requires component.
        """

        self.vessel = vessel
        self.component = component
        self.action = action

        self.msg = (
            f"{vessel} is missing required component(s) '{component}' to"
            f" complete action '{action}'."
        )

        super().__init__(self.msg)
