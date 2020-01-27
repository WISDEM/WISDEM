"""Provides the `ExportSystemDesign` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"

import numpy as np

from wisdem.orbit.phases.design._cables import CableSystem


class ExportSystemDesign(CableSystem):
    """
    Design phase for the export cabling system.

    Attributes
    ----------
    num_cables : int
        Total number of cables required for transmitting power.
    length : float
        Length of a single cable connecting the OSS to the interconnection in km.
    mass : float
        Mass of `length` in tonnes.
    cable : `Cable`
        Instance of `ORBIT.phases.design.Cable`. An export system will
        only require a single type of cable.
    total_length : float
        Total length of cable required to trasmit power.
    total_mass : float
        Total mass of cable required to transmit power.
    sections_cables : np.ndarray, shape: (`num_cables, )
        An array of `cable`.
    sections_lengths : np.ndarray, shape: (`num_cables, )
        An array of `length`.
    """

    expected_config = {
        "site": {
            "distance_to_landfall": "int | float",
            "distance_to_interconnection": "int | float",
            "depth": "int | float",
        },
        "plant": {"num_turbines": "int"},
        "turbine": {"turbine_rating": "int | float"},
        "export_system_design": {
            "cables": "str",
            "num_redundant": "int (optional)",
            "percent_added_length": "float (optional)",
        },
    }

    output_config = {"export_system": {"cables": "dict"}}

    def __init__(self, config, **kwargs):
        """
        Defines the cables and sections required to install an offwhore wind
        farm.

        Parameters
        ----------
        config : dict
            Configuration dictionary for the array cabling system. See
            `expected_config` for details on what is required.
        """

        super().__init__(config, "export", **kwargs)
        # For export cables there should only be one cable type due to the
        # custom nature of their design

        for name in self.expected_config["site"]:
            setattr(self, "".join(("_", name)), config["site"][name])
        self._depth = config["site"]["depth"]
        self._plant_capacity = (
            self.config["plant"]["num_turbines"]
            * self.config["turbine"]["turbine_rating"]
        )
        self._distance_to_landfall = config["site"]["distance_to_landfall"]
        self._distance_to_interconnection = config["site"][
            "distance_to_interconnection"
        ]

    def run(self):
        """
        Instantiates the export cable system and runs all the required methods.
        """

        self._initialize_cables()
        self.cable = self.cables[[*self.cables][0]]
        self.compute_number_cables()
        self.compute_cable_length()
        self.compute_cable_mass()
        self.compute_total_cable()

    def compute_number_cables(self):
        """
        Calculate the total number of required and redundant cables to
        transmit power to the onshore interconnection.
        """

        num_required = np.ceil(self._plant_capacity / self.cable.cable_power)
        num_redundant = self._design.get("num_redundant", 0)

        self.num_cables = int(num_required + num_redundant)

    def compute_cable_length(self):
        """
        Calculates the total distance an export cable must travel.
        """

        added_length = 1.0 + self._design.get("percent_added_length", 0.0)
        self.length = round(
            (
                (self._depth / 1000.0)  # convert to km
                + self._distance_to_landfall
                + self._distance_to_interconnection
            )
            * added_length,
            10,
        )

    def compute_cable_mass(self):
        """
        Calculates the total mass of a single length of export cable.
        """

        self.mass = round(self.length * self.cable.linear_density, 10)

    def compute_total_cable(self):
        """
        Calculates the total length and mass of cables required to fully
        connect the OSS to the interconnection point.
        """

        self.total_length = round(self.num_cables * self.length, 10)
        self.total_mass = round(self.num_cables * self.mass, 10)

    @property
    def sections_cable_lengths(self):
        """
        Creates an array of section lengths to work with `CableSystem`

        Returns
        -------
        np.ndarray
            Array of `length` with shape (`num_cables`, ).
        """
        return np.full(self.num_cables, self.length)

    @property
    def sections_cables(self):
        """
        Creates an array of cable names to work with `CableSystem`.

        Returns
        -------
        np.ndarray
            Array of `cable.name` with shape (`num_cables`, ).
        """

        return np.full(self.num_cables, self.cable.name)
