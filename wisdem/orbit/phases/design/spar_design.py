"""Provides the `SparDesign` class (from OffshoreBOS)."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from numpy import exp, log

from wisdem.orbit.phases.design import DesignPhase


class SparDesign(DesignPhase):
    """Spar Substructure Design"""

    expected_config = {
        "site": {"depth": "m"},
        "plant": {"num_turbines": "int"},
        "turbine": {"turbine_rating": "MW"},
        "spar_design": {
            "stiffened_column_CR": "$/t (optional, default: 3120)",
            "tapered_column_CR": "$/t (optional, default: 4220)",
            "ballast_material_CR": "$/t (optional, default: 100)",
            "secondary_steel_CR": "$/t (optional, default: 7250)",
            "towing_speed": "km/h (optional, default: 6)",
        },
    }

    output_config = {
        "substructure": {
            "mass": "t",
            "ballasted_mass": "t",
            "unit_cost": "USD",
            "towing_speed": "km/h",
        }
    }

    def __init__(self, config, **kwargs):
        """
        Creates an instance of `SparDesign`.

        Parameters
        ----------
        config : dict
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self._design = self.config.get("spar_design", {})

        self._outputs = {}

    def run(self):
        """Main run function."""

        substructure = {
            "mass": self.unballasted_mass,
            "ballasted_mass": self.ballasted_mass,
            "unit_cost": self.substructure_cost,
            "towing_speed": self._design.get("towing_speed", 6),
        }

        self._outputs["substructure"] = substructure

    @property
    def stiffened_column_mass(self):
        """
        Calculates the mass of the stiffened column for a single spar in tonnes. From original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]
        depth = self.config["site"]["depth"]

        mass = 535.93 + 17.664 * rating**2 + 0.02328 * depth * log(depth)

        return mass

    @property
    def tapered_column_mass(self):
        """
        Calculates the mass of the atpered column for a single spar in tonnes. From original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]

        mass = 125.81 * log(rating) + 58.712

        return mass

    @property
    def stiffened_column_cost(self):
        """
        Calculates the cost of the stiffened column for a single spar. From original OffshoreBOS model.
        """

        cr = self._design.get("stiffened_column_CR", 3120)
        return self.stiffened_column_mass * cr

    @property
    def tapered_column_cost(self):
        """
        Calculates the cost of the tapered column for a single spar. From original OffshoreBOS model.
        """

        cr = self._design.get("tapered_column_CR", 4220)
        return self.tapered_column_mass * cr

    @property
    def ballast_mass(self):
        """
        Calculates the ballast mass of a single spar. From original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]
        mass = -16.536 * rating**2 + 1261.8 * rating - 1554.6

        return mass

    @property
    def ballast_cost(self):
        """
        Calculates the cost of ballast material for a single spar. From original OffshoreBOS model.
        """

        cr = self._design.get("ballast_material_CR", 100)
        return self.ballast_mass * cr

    @property
    def secondary_steel_mass(self):
        """
        Calculates the mass of the required secondary steel for a single
        spar. From original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]
        depth = self.config["site"]["depth"]

        mass = exp(3.58 + 0.196 * (rating**0.5) * log(rating) + 0.00001 * depth * log(depth))

        return mass

    @property
    def secondary_steel_cost(self):
        """
        Calculates the cost of the required secondary steel for a single
        spar. For original OffshoreBOS model.
        """

        cr = self._design.get("secondary_steel_CR", 7250)
        return self.secondary_steel_mass * cr

    @property
    def unballasted_mass(self):
        """Returns the unballasted mass of the spar substructure."""

        return self.stiffened_column_mass + self.tapered_column_mass + self.secondary_steel_mass

    @property
    def ballasted_mass(self):
        """Returns the ballasted mass of the spar substructure."""

        return self.unballasted_mass + self.ballast_mass

    @property
    def substructure_cost(self):
        """Returns the total cost (including ballast) of the spar substructure."""

        return self.stiffened_column_cost + self.tapered_column_cost + self.secondary_steel_cost + self.ballast_cost

    @property
    def detailed_output(self):
        """Returns detailed phase information."""

        _outputs = {
            "stiffened_column_mass": self.stiffened_column_mass,
            "stiffened_column_cost": self.stiffened_column_cost,
            "tapered_column_mass": self.tapered_column_mass,
            "tapered_column_cost": self.tapered_column_cost,
            "ballast_mass": self.ballast_mass,
            "ballast_cost": self.ballast_cost,
            "secondary_steel_mass": self.secondary_steel_mass,
            "secondary_steel_cost": self.secondary_steel_cost,
        }

        return _outputs

    @property
    def total_cost(self):
        """Returns total phase cost in $USD."""

        num = self.config["plant"]["num_turbines"]
        return num * self.substructure_cost

    @property
    def design_result(self):
        """Returns the result of `self.run()`"""

        if not self._outputs:
            raise Exception("Has `SparDesign` been ran yet?")

        return self._outputs
