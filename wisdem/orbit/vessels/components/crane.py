"""Provides the `Crane` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.vessels.tasks._defaults import defaults


class Crane:
    """Base Crane Class"""

    def __init__(self, crane_specs):
        """
        Creates an instance of Crane.

        Parameters
        ----------
        crane_specs : dict
            Dictionary containing crane system specifications.
        """

        self.extract_crane_specs(crane_specs)

    def extract_crane_specs(self, crane_specs):
        """
        Extracts and defines crane specifications.

        Parameters
        ----------
        crane_specs : dict
            Dictionary of crane specifications.
        """

        # Physical Dimensions
        self.boom_length = crane_specs.get("boom_length", None)
        self.radius = crane_specs.get("radius", None)

        # Operational Parameters
        self.max_lift = crane_specs.get("max_lift", None)
        self.max_hook_height = crane_specs.get("max_hook_height", None)
        self.max_windspeed = crane_specs.get("max_windspeed", 99)

    @staticmethod
    def crane_rate(**kwargs):
        """
        Calculates minimum crane rate based on current wave height equation
        from DNV standards for offshore lifts.

        Parameters
        ----------
        wave_height : int | float
            Significant wave height (m).

        Returns
        -------
        crane_rate : float
            Hoist speed of crane (m/hr).
        """

        wave_height = kwargs.get("wave_height", 2)
        return 0.6 * wave_height * 3600

    @staticmethod
    def reequip(**kwargs):
        """
        Calculates time taken to change crane equipment.

        Parameters
        ----------
        crane_reequip_time : int | float
            Time required to change crane equipment (h).

        Returns
        -------
        reequip_time : float
            Time required to change crane equipment (h).
        """

        _key = "crane_reequip_time"
        duration = kwargs.get(_key, defaults[_key])

        return duration
