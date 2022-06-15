"""`MooringSystemDesign` and related functionality."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from math import sqrt

from wisdem.orbit.phases.design import DesignPhase


class MooringSystemDesign(DesignPhase):
    """Mooring System and Anchor Design."""

    expected_config = {
        "site": {"depth": "float"},
        "turbine": {"turbine_rating": "int | float"},
        "plant": {"num_turbines": "int"},
        "mooring_system_design": {
            "num_lines": "int | float (optional, default: 4)",
            "anchor_type": "str (optional, default: 'Suction Pile')",
            "mooring_line_cost_rate": "int | float (optional)",
            "drag_embedment_fixed_length": "int | float (optional, default: .5km)",
        },
    }

    output_config = {
        "mooring_system": {
            "num_lines": "int",
            "line_diam": "m, float",
            "line_mass": "t",
            "line_cost": "USD",
            "line_length": "m",
            "anchor_mass": "t",
            "anchor_type": "str",
            "anchor_cost": "USD",
        }
    }

    def __init__(self, config, **kwargs):
        """
        Creates an instance of MooringSystemDesign.

        Parameters
        ----------
        config : dict
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self.num_turbines = self.config["plant"]["num_turbines"]

        self._design = self.config.get("mooring_system_design", {})
        self.num_lines = self._design.get("num_lines", 4)
        self.anchor_type = self._design.get("anchor_type", "Suction Pile")

        self._outputs = {}

    def run(self):
        """
        Main run function.
        """

        self.determine_mooring_line()
        self.calculate_breaking_load()
        self.calculate_line_length_mass()
        self.calculate_anchor_mass_cost()

        self._outputs["mooring_system"] = {**self.design_result}

    def determine_mooring_line(self):
        """
        Returns the diameter of the mooring lines based on the turbine rating.
        """

        tr = self.config["turbine"]["turbine_rating"]
        fit = -0.0004 * (tr**2) + 0.0132 * tr + 0.0536

        if fit <= 0.09:
            self.line_diam = 0.09
            self.line_mass_per_m = 0.161
            self.line_cost_rate = 399.0

        elif fit <= 0.12:
            self.line_diam = 0.12
            self.line_mass_per_m = 0.288
            self.line_cost_rate = 721.0

        else:
            self.line_diam = 0.15
            self.line_mass_per_m = 0.450
            self.line_cost_rate = 1088.0

    def calculate_breaking_load(self):
        """
        Returns the mooring line breaking load.
        """

        self.breaking_load = 419449 * (self.line_diam**2) + 93415 * self.line_diam - 3577.9

    def calculate_line_length_mass(self):
        """
        Returns the mooring line length and mass.
        """

        if self.anchor_type == "Drag Embedment":
            fixed = self._design.get("drag_embedment_fixed_length", 500)

        else:
            fixed = 0

        depth = self.config["site"]["depth"]
        self.line_length = 0.0002 * (depth**2) + 1.264 * depth + 47.776 + fixed

        self.line_mass = self.line_length * self.line_mass_per_m

    def calculate_anchor_mass_cost(self):
        """
        Returns the mass and cost of anchors.

        TODO: Anchor masses are rough estimates based on initial literature
        review. Should be revised when this module is overhauled in the future.
        """

        if self.anchor_type == "Drag Embedment":
            self.anchor_mass = 20
            self.anchor_cost = self.breaking_load / 9.81 / 20.0 * 2000.0

        else:
            self.anchor_mass = 50
            self.anchor_cost = sqrt(self.breaking_load / 9.81 / 1250) * 150000

    @property
    def line_cost(self):
        """Returns cost of one line mooring line."""

        return self.line_length * self.line_cost_rate

    @property
    def total_cost(self):
        """Returns the total cost of the mooring system."""

        return self.num_lines * self.num_turbines * (self.anchor_cost + self.line_length * self.line_cost_rate)

    @property
    def detailed_output(self):
        """Returns detailed phase information."""

        return {
            "num_lines": self.num_lines,
            "line_diam": self.line_diam,
            "line_mass": self.line_mass,
            "line_length": self.line_length,
            "line_cost": self.line_cost,
            "anchor_type": self.anchor_type,
            "anchor_mass": self.anchor_mass,
            "anchor_cost": self.anchor_cost,
            "system_cost": self.total_cost,
        }

    @property
    def design_result(self):
        """Returns the results of the design phase."""

        return {"mooring_system": self.detailed_output}
