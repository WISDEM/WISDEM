"""Provides the `SemiSubmersibleDesign` class (from OffshoreBOS)."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.phases.design import DesignPhase


class SemiSubmersibleDesign(DesignPhase):
    """Semi-Submersible Substructure Design"""

    expected_config = {
        "site": {"depth": "m"},
        "plant": {"num_turbines": "int"},
        "turbine": {"turbine_rating": "MW"},
        "semisubmersible_design": {
            "stiffened_column_CR": "$/t (optional, default: 3120)",
            "truss_CR": "$/t (optional, default: 6250)",
            "heave_plate_CR": "$/t (optional, default: 6250)",
            "secondary_steel_CR": "$/t (optional, default: 7250)",
            "towing_speed": "km/h (optional, default: 6)",
        },
    }

    output_config = {
        "substructure": {
            "mass": "t",
            "unit_cost": "USD",
            "towing_speed": "km/h",
        }
    }

    def __init__(self, config, **kwargs):
        """
        Creates an instance of `SemiSubmersibleDesign`.

        Parameters
        ----------
        config : dict
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self._design = self.config.get("semisubmersible_design", {})

        self._outputs = {}

    def run(self):
        """Main run function."""

        substructure = {
            "mass": self.substructure_mass,
            "unit_cost": self.substructure_cost,
            "towing_speed": self._design.get("towing_speed", 6),
        }

        self._outputs["substructure"] = substructure

    @property
    def stiffened_column_mass(self):
        """
        Calculates the mass of the stiffened column for a single
        semi-submersible in tonnes. From original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]
        mass = -0.9581 * rating**2 + 40.89 * rating + 802.09

        return mass

    @property
    def stiffened_column_cost(self):
        """
        Calculates the cost of the stiffened column for a single
        semi-submersible. From original OffshoreBOS model.
        """

        cr = self._design.get("stiffened_column_CR", 3120)
        return self.stiffened_column_mass * cr

    @property
    def truss_mass(self):
        """
        Calculates the truss mass for a single semi-submersible in tonnes. From
        original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]
        mass = 2.7894 * rating**2 + 15.591 * rating + 266.03

        return mass

    @property
    def truss_cost(self):
        """
        Calculates the cost of the truss for a signle semi-submerisble. From
        original OffshoreBOS model.
        """

        cr = self._design.get("truss_CR", 6250)
        return self.truss_mass * cr

    @property
    def heave_plate_mass(self):
        """
        Calculates the heave plate mass for a single semi-submersible in tonnes.
        From original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]
        mass = -0.4397 * rating**2 + 21.545 * rating + 177.42

        return mass

    @property
    def heave_plate_cost(self):
        """
        Calculates the heave plate cost for a single semi-submersible. From
        original OffshoreBOS model.
        """

        cr = self._design.get("heave_plate_CR", 6250)
        return self.heave_plate_mass * cr

    @property
    def secondary_steel_mass(self):
        """
        Calculates the mass of the required secondary steel for a single
        semi-submersible. From original OffshoreBOS model.
        """

        rating = self.config["turbine"]["turbine_rating"]
        mass = -0.153 * rating**2 + 6.54 * rating + 128.34

        return mass

    @property
    def secondary_steel_cost(self):
        """
        Calculates the cost of the required secondary steel for a single
        semi-submersible. For original OffshoreBOS model.
        """

        cr = self._design.get("secondary_steel_CR", 7250)
        return self.secondary_steel_mass * cr

    @property
    def substructure_mass(self):
        """Returns single substructure mass."""

        return self.stiffened_column_mass + self.truss_mass + self.heave_plate_mass + self.secondary_steel_mass

    @property
    def substructure_cost(self):
        """Returns single substructure cost."""

        return self.stiffened_column_cost + self.truss_cost + self.heave_plate_cost + self.secondary_steel_cost

    @property
    def total_substructure_mass(self):
        """Returns mass of all substructures."""

        num = self.config["plant"]["num_turbines"]
        return num * self.substructure_mass

    @property
    def design_result(self):
        """Returns the result of `self.run()`"""

        if not self._outputs:
            raise Exception("Has `SemiSubmersibleDesign` been ran yet?")

        return self._outputs

    @property
    def total_cost(self):
        """Returns total phase cost in $USD."""

        num = self.config["plant"]["num_turbines"]
        return num * self.substructure_cost

    @property
    def detailed_output(self):
        """Returns detailed phase information."""

        _outputs = {
            "stiffened_column_mass": self.stiffened_column_mass,
            "stiffened_column_cost": self.stiffened_column_cost,
            "truss_mass": self.truss_mass,
            "truss_cost": self.truss_cost,
            "heave_plate_mass": self.heave_plate_mass,
            "heave_plate_cost": self.heave_plate_cost,
            "secondary_steel_mass": self.secondary_steel_mass,
            "secondary_steel_cost": self.secondary_steel_cost,
        }

        return _outputs
