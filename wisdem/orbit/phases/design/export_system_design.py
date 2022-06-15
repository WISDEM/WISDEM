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
        "site": {"distance_to_landfall": "km", "depth": "m"},
        "landfall": {"interconnection_distance": "km (optional)"},
        "plant": {"capacity": "MW"},
        "export_system_design": {
            "cables": "str",
            "num_redundant": "int (optional)",
            "touchdown_distance": "m (optional, default: 0)",
            "percent_added_length": "float (optional)",
        },
    }

    output_config = {
        "export_system": {
            "cable": {
                "linear_density": "t/km",
                "number": "int",
                "sections": "list",
                "cable_power": "MW",
            }
        }
    }

    def __init__(self, config, **kwargs):
        """
        Defines the cables and sections required to install an offshore wind
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
        self._plant_capacity = self.config["plant"]["capacity"]
        self._distance_to_landfall = config["site"]["distance_to_landfall"]
        self._get_touchdown_distance()
        try:
            self._distance_to_interconnection = config["landfall"]["interconnection_distance"]
        except KeyError:
            self._distance_to_interconnection = 3

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

    @property
    def total_cable_cost(self):
        """Returns total array system cable cost."""

        return sum(self.cost_by_type.values())

    @property
    def detailed_output(self):
        """Returns export system design outputs."""

        _output = {
            **self.design_result,
            "export_system_total_mass": self.total_mass,
            "export_system_total_length": self.total_length,
            "export_system_total_cost": self.total_cable_cost,
            "export_system_cable_power": self.cable.cable_power,
        }

        return _output

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
                self.free_cable_length
                + (self._distance_to_landfall - self.touchdown / 1000)
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

    @property
    def design_result(self):
        """
        A dictionary of cables types and number of different cable lengths and
        linear density.

        Returns
        -------
        output : dict
            Dictionary containing the output export system. Contains:
            - 'linear_density': 't/km'
            - 'sections': 'list [self.length]'
            - 'number': 'int'
        """

        if self.cables is None:
            raise Exception(f"Has {self.__class__.__name__} been ran?")

        output = {
            "export_system": {
                "interconnection_distance": self._distance_to_interconnection,
                "system_cost": self.total_cost,
            }
        }

        for name, cable in self.cables.items():

            output["export_system"]["cable"] = {
                "linear_density": cable.linear_density,
                "sections": [self.length],
                "number": self.num_cables,
                "cable_power": cable.cable_power,
            }

        return output
