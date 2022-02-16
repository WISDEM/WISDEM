__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.phases.design import DesignPhase


class GbfDesign(DesignPhase):
    """Gravity Based Foundation Design Module"""

    expected_config = {
        "site": {"depth": "m"},
        "plant": {"num_turbines": "int"},
        "turbine": {"turbine_rating": "MW"},
        "gbf_design": {},
    }

    output_config = {
        "gbf": {  # TODO: Does this need to be 'substructure' to work with downstream modules?
            "height": "m",
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
        """Creates an instance of `GbfDesign`."""

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self.gbf_design = self.config.get("gbf_design", {})

        self._outputs = {}

    def run(self):
        """Main run function used by ProjectManager."""

        self.design_gbf()

    def design_gbf(self):
        """Main GBF design script."""

        raise NotImplementedError("'GbfDesign' module has not been fullyimplemented yet.")

    @property
    def detailed_output(self):
        """Returns detailed output dictionary."""

        return {}

    @property
    def total_cost(self):
        """Returns total cost of subcomponent(s)."""

        num_turbines = self.config["plant"]["num_turbines"]
        return (self._outputs["gbf"]["unit_cost"] + self._outputs["transition_piece"]["unit_cost"]) * num_turbines

    @property
    def design_result(self):
        """Must match `self.output_config` structure."""

        return self._outputs
