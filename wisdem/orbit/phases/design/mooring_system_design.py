"""`MooringSystemDesign` and related functionality."""

__author__ = "Jake Nunemaker, Becca Fuchs"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Nicholas Riccobono"
__email__ = (
    "jake.nunemaker@nrel.gov, rebecca.fuchs@nrel.gov,"
    "nicholas.riccobono@nrel.gov"
)

from math import sqrt

from scipy.interpolate import interp1d

from wisdem.orbit.phases.design import DesignPhase

"""
[1] Maness et al. 2017, NREL Offshore Balance-of-System Model.
https://www.nrel.gov/docs/fy17osti/66874.pdf

[2] Cooperman et al. (2022), Assessment of Offshore Wind Energy Leasing Areas
for Humboldt and Morry Bay. https://www.nrel.gov/docs/fy22osti/82341.pdf
"""


class MooringSystemDesign(DesignPhase):
    """Mooring System and Anchor Design."""

    expected_config = {
        "site": {"depth": "float"},
        "turbine": {"turbine_rating": "int | float"},
        "plant": {"num_turbines": "int"},
        "mooring_system_design": {
            "num_lines": "int | float (optional, default: 4)",
            "anchor_type": "str (optional, default: 'Suction Pile')",
            "mooring_type": "str (optional, default: 'Catenary')",
            "mooring_line_cost_rate": "int | float (optional)",
            "drag_embedment_fixed_length": "int (optional, default: 500m)",
            "draft_depth": "int (optional, default: 20m)",
            "chain_density": "int | float (optional, default: 19900 kg/m**3)",
            "rope_density": "int | float (optional, default: 797.8 kg/m**3)",
        },
    }

    output_config = {
        "mooring_system": {
            "num_lines": "int",
            "line_diam": "m, float",
            "line_mass": "t",
            "line_cost": "USD",
            "line_length": "m",
            "mooring_type": "str",
            "anchor_mass": "t",
            "anchor_type": "str",
            "anchor_cost": "USD",
            "system_cost": "USD",
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
        self.depth = self.config["site"]["depth"]

        self._design = self.config.get("mooring_system_design", {})
        self.num_lines = self._design.get("num_lines", 4)
        self.anchor_type = self._design.get(
            "anchor_type", "Suction Pile"
        ).title()
        self.mooring_type = self._design.get(
            "mooring_type", "Catenary"
        ).title()

        # Semi-Taut mooring system design parameters based on depth [2].
        self._semitaut_params = {
            "depths": [500.0, 750.0, 1000.0, 1250.0, 1500.0],
            "rope_lengths": [478.41, 830.34, 1229.98, 1183.93, 1079.62],
            "rope_diameter": 0.2,
            "chain_lengths": [917.11, 800.36, 609.07, 896.42, 1280.57],
            "chain_diameters": [0.13, 0.17, 0.22, 0.22, 0.22],
            "anchor_costs": [112766.0, 125511.0, 148703.0, 204988.0, 246655.0],
            "total_line_costs": [
                826598.0,
                1221471.0,
                1682208.0,
                2380035.0,
                3229700.0,
            ],
        }

        self._outputs = {}

    def run(self):
        """Runs the design model."""

        self.determine_mooring_line()
        self.calculate_breaking_load()
        self.calculate_line_length_mass()
        self.calculate_anchor_mass_cost()

        self._outputs["mooring_system"] = {**self.design_result}

    def determine_mooring_line(self):
        """
        Returns the diameter of the mooring lines based on the turbine rating.

        TODO: Add TLP option and consider merging SemiTaut interp here
        """

        tr = self.config["turbine"]["turbine_rating"]
        fit = -0.0004 * (tr**2) + 0.0132 * tr + 0.0536

        _key = "mooring_line_cost_rate"

        mooring_line_cost_rate = self._design.get(
            _key,
            self.get_default_cost(
                "mooring_system_design",
                _key,
            ),
        )
        if isinstance(mooring_line_cost_rate, (int, float)):
            mooring_line_cost_rate = [mooring_line_cost_rate] * 3

        if fit <= 0.09:
            self.line_diam = 0.09
            self.line_mass_per_m = 0.161
            self.line_cost_rate = mooring_line_cost_rate[0]

        elif fit <= 0.12:
            self.line_diam = 0.12
            self.line_mass_per_m = 0.288
            self.line_cost_rate = mooring_line_cost_rate[1]

        else:
            self.line_diam = 0.15
            self.line_mass_per_m = 0.450
            self.line_cost_rate = mooring_line_cost_rate[2]

    def calculate_breaking_load(self):
        """Returns the mooring line breaking load."""

        self.breaking_load = (
            419449 * (self.line_diam**2) + 93415 * self.line_diam - 3577.9
        )

    def calculate_line_length_mass(self):
        """
        Returns the mooring line length and mass.

        SemiTaut model based on:
        https://github.com/NREL/MoorPy/blob/dev/moorpy/MoorProps_default.yaml

        TODO: Improve TLP line length and mass

        """

        # Add extra fixed line length for drag embedments
        if self.anchor_type == "Drag Embedment":
            fixed = self._design.get("drag_embedment_fixed_length", 500)

        else:
            fixed = 0

        draft = self._design.get("draft_depth", 20)

        if self.mooring_type == "Semitaut":

            # Interpolation of rope and chain length at project depth
            self.chain_length = interp1d(
                self._semitaut_params["depths"],
                self._semitaut_params["chain_lengths"],
                fill_value="extrapolate",
            )(self.depth).item()
            self.rope_length = interp1d(
                self._semitaut_params["depths"],
                self._semitaut_params["rope_lengths"],
                fill_value="extrapolate",
            )(self.depth).item()

            # Rope and interpolated chain diameter at project depth
            rope_diameter = self._semitaut_params["rope_diameter"]
            chain_diameter = interp1d(
                self._semitaut_params["depths"],
                self._semitaut_params["chain_diameters"],
                fill_value="extrapolate",
            )(self.depth).item()

            fixed = self._design.get("drag_embedment_fixed_length", 0)
            self.line_length = self.rope_length + self.chain_length + fixed

            # line characteristics based on MoorPy defaults,
            chain_mass_per_m = (
                self._design.get("mooring_chain_density", 19900)
                * chain_diameter**2
            )  # kg/m
            rope_mass_per_m = (
                self._design.get("mooring_rope_density", 797.8)
                * rope_diameter**2
            )  # kg/m

            self.line_mass = (
                self.chain_length * chain_mass_per_m
                + self.rope_length * rope_mass_per_m
            ) / 1e3  # tonnes

        elif self.mooring_type == "Tlp":

            self.line_length = self.depth - draft

            self.line_mass = self.line_length * self.line_mass_per_m

        else:

            self.line_length = (
                0.0002 * (self.depth**2) + 1.264 * self.depth + 47.776 + fixed
            )

            self.line_mass = self.line_length * self.line_mass_per_m

    def calculate_anchor_mass_cost(self):
        """
        Returns the mass and cost of anchors.

        TODO: Anchor masses are rough estimates based on [1]. Should be
        revised when this module is overhauled in the future.
        TODO: Mooring types for Catenary, TLP, SemiTaut will likely have
        different anchors.
        """

        if self.mooring_type == "Semitaut":

            if self.anchor_type == "Drag Embedment":
                self.anchor_mass = 20

                # Interpolation of anchor cost at project depth
                self.anchor_cost = interp1d(
                    self._semitaut_params["depths"],
                    self._semitaut_params["anchor_costs"],
                    fill_value="extrapolate",
                )(self.depth).item()

            else:
                self.anchor_mass = 50
                self.anchor_cost = (
                    sqrt(self.breaking_load / 9.81 / 1250) * 150000
                )

        else:

            if self.anchor_type == "Drag Embedment":
                self.anchor_mass = 20
                self.anchor_cost = self.breaking_load / 9.81 / 20.0 * 2000.0

            else:
                self.anchor_mass = 50
                self.anchor_cost = (
                    sqrt(self.breaking_load / 9.81 / 1250) * 150000
                )

    @property
    def line_cost(self):
        """Returns cost of one line mooring line."""

        if self.mooring_type == "Semitaut":
            # Interpolation of line cost at project depth
            line_cost = interp1d(
                self._semitaut_params["depths"],
                self._semitaut_params["total_line_costs"],
                fill_value="extrapolate",
            )(self.depth).item()

        else:

            line_cost = self.line_length * self.line_cost_rate

        return line_cost

    @property
    def total_cost(self):
        """Returns the total cost of the mooring system."""

        return (
            self.num_lines
            * self.num_turbines
            * (self.anchor_cost + self.line_cost)
        )

    @property
    def detailed_output(self):
        """Returns detailed phase information."""

        return {
            "num_lines": self.num_lines,
            "line_diam": self.line_diam,
            "line_mass": self.line_mass,
            "line_length": self.line_length,
            "line_cost": self.line_cost,
            "mooring_type": self.mooring_type,
            "anchor_type": self.anchor_type,
            "anchor_mass": self.anchor_mass,
            "anchor_cost": self.anchor_cost,
            "system_cost": self.total_cost,
        }

    @property
    def design_result(self):
        """Returns the results of the design phase."""

        return {"mooring_system": self.detailed_output}
