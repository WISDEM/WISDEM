"""Provides the `JackingSys` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


class JackingSys:
    """Base Jacking System Class"""

    def __init__(self, jacksys_specs):
        """
        Creates an instance of JackingSys.

        Parameters
        ----------
        jacksys_specs : dict
            Dictionary containing jacking system specifications.
        """

        self.extract_jacksys_specs(jacksys_specs)

    def extract_jacksys_specs(self, jacksys_specs):
        """
        Extracts and defines jacking system specifications.

        Parameters
        ----------
        jacksys_specs : dict
            Dictionary containing jacking system specifications.
        """

        # Physical Dimensions
        self.num_legs = jacksys_specs.get("num_legs", None)
        self.leg_length = jacksys_specs.get("leg_length", None)
        self.air_gap = jacksys_specs.get("air_gap", None)
        self.leg_pen = jacksys_specs.get("leg_pen", None)

        # Operational Parameters
        self.max_depth = jacksys_specs.get("max_depth", None)
        self.max_extension = jacksys_specs.get("max_extension", None)
        self.speed_below_depth = jacksys_specs.get("speed_below_depth", None)
        self.speed_above_depth = jacksys_specs.get("speed_above_depth", None)

    def jacking_time(self, extension, depth):
        """
        Calculates jacking time for a given depth.

        Parameters
        ----------
        extension : int | float
            Height to jack-up to or jack-down from (m).
        depth : int | float
            Depth at jack-up location (m).

        Returns
        -------
        extension_time : float
            Time required to jack-up to given extension (h).
        """

        if extension > self.max_extension:
            raise Exception(
                "{} extension is greater than {} maximum"
                "".format(extension, self.max_extension)
            )

        elif depth > self.max_depth:
            raise Exception(
                "{} is beyond the operating depth {}"
                "".format(depth, self.max_depth)
            )

        elif depth > extension:
            raise Exception("Extension must be greater than depth")

        else:
            return (
                depth / self.speed_below_depth
                + (extension - depth) / self.speed_above_depth
            ) / 60
