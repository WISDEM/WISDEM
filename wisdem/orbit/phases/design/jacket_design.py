__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.phases.design import DesignPhase


class JacketDesign(DesignPhase):
    """Jacket Design Module"""

    expected_config = {
        "site": {"depth": "m"},
        "plant": {"num_turbines": "int"},
        "turbine": {"turbine_rating": "MW"},
        "jacket_design": {},
    }

    output_config = {
        "jacket": {  # TODO: Does this need to be 'substructure' to work with downstream modules?
            "height": "m",
            "deck_space": "m2",
            "mass": "t",
            "unit_cost": "USD",
        },
        "transition_piece": {
            "deck_space": "m2",
            "mass": "t",
            "unit_cost": "USD",
        },
    }

    def __init__(self, config, **kwargs):
        """Creates an instance of `JacketDesign`."""

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self.jacket_design = self.config.get("jacket_design", {})

        self._outputs = {}

    def run(self):
        """Main run function used by ProjectManager."""

        self.design_jacket()

    def design_jacket(self):
        """Main jacket design script."""

        raise NotImplementedError("'JacketDesign' module has not been fullyimplemented yet.")

    @property
    def detailed_output(self):
        """Returns detailed output dictionary."""

        return {}

    @property
    def total_cost(self):
        """Returns total cost of subcomponent(s)."""

        num_turbines = self.config["plant"]["num_turbines"]
        return (self._outputs["jacket"]["unit_cost"] + self._outputs["transition_piece"]["unit_cost"]) * num_turbines

    @property
    def design_result(self):
        """Must match `self.output_config` structure."""

        return self._outputs
