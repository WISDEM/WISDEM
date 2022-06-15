"""Provides the 'OffshoreSubstationDesign` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


import numpy as np

from wisdem.orbit.phases.design import DesignPhase


class OffshoreSubstationDesign(DesignPhase):
    """Offshore Substation Design Class."""

    expected_config = {
        "site": {"depth": "m"},
        "plant": {"num_turbines": "int"},
        "turbine": {"turbine_rating": "MW"},
        "substation_design": {
            "mpt_cost_rate": "USD/MW (optional)",
            "topside_fab_cost_rate": "USD/t (optional)",
            "topside_design_cost": "USD (optional)",
            "shunt_cost_rate": "USD/MW (optional)",
            "switchgear_cost": "USD (optional)",
            "backup_gen_cost": "USD (optional)",
            "workspace_cost": "USD (optional)",
            "other_ancillary_cost": "USD (optional)",
            "topside_assembly_factor": "float (optional)",
            "oss_substructure_cost_rate": "USD/t (optional)",
            "oss_pile_cost_rate": "USD/t (optional)",
            "num_substations": "int (optional)",
        },
    }

    output_config = {
        "num_substations": "int",
        "offshore_substation_topside": "dict",
        "offshore_substation_substructure": "dict",
    }

    def __init__(self, config, **kwargs):
        """
        Creates an instance of OffshoreSubstationDesign.

        Parameters
        ----------
        config : dict
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self._outputs = {}

    def run(self):
        """Main run function."""

        self.calc_substructure_length()
        self.calc_substructure_deck_space()
        self.calc_topside_deck_space()

        self.calc_num_mpt_and_rating()
        self.calc_mpt_cost()
        self.calc_topside_mass_and_cost()
        self.calc_shunt_reactor_cost()
        self.calc_switchgear_cost()
        self.calc_ancillary_system_cost()
        self.calc_assembly_cost()
        self.calc_substructure_mass_and_cost()

        self._outputs["offshore_substation_substructure"] = {
            "type": "Monopile",  # Substation install only supports monopiles
            "deck_space": self.substructure_deck_space,
            "mass": self.substructure_mass,
            "length": self.substructure_length,
            "unit_cost": self.substructure_cost,
        }

        self._outputs["offshore_substation_topside"] = {
            "deck_space": self.topside_deck_space,
            "mass": self.topside_mass,
            "unit_cost": self.substation_cost,
        }

        self._outputs["num_substations"] = self.num_substations

    @property
    def substation_cost(self):
        """Returns total procuremet cost of the topside."""

        return (
            self.mpt_cost
            + self.topside_cost
            + self.shunt_reactor_cost
            + self.switchgear_costs
            + self.ancillary_system_costs
            + self.land_assembly_cost
        )

    @property
    def total_cost(self):
        """Returns total procurement cost of the substation(s)."""

        if not self._outputs:
            raise Exception("Has OffshoreSubstationDesign been ran yet?")

        return (self.substructure_cost + self.substation_cost) * self.num_substations

    def calc_substructure_length(self):
        """
        Calculates substructure length as the site depth + 10m
        """

        self.substructure_length = self.config["site"]["depth"] + 10

    def calc_substructure_deck_space(self):
        """
        Calculates required deck space for the substation substructure.

        Coming soon!
        """

        self.substructure_deck_space = 1

    def calc_topside_deck_space(self):
        """
        Calculates required deck space for the substation topside.

        Coming soon!
        """

        self.topside_deck_space = 1

    def calc_num_mpt_and_rating(self):
        """
        Calculates the number of main power transformers (MPTs) and their rating.

        Parameters
        ----------
        num_turbines : int
        turbine_rating : float
        """

        _design = self.config.get("substation_design", {})

        num_turbines = self.config["plant"]["num_turbines"]
        turbine_rating = self.config["turbine"]["turbine_rating"]
        capacity = num_turbines * turbine_rating

        self.num_substations = _design.get("num_substations", int(np.ceil(capacity / 500)))
        self.num_mpt = np.ceil(num_turbines * turbine_rating / (250 * self.num_substations))
        self.mpt_rating = (
            round(((num_turbines * turbine_rating * 1.15) / (self.num_mpt * self.num_substations)) / 10.0) * 10.0
        )

    def calc_mpt_cost(self):
        """
        Calculates the total cost for all MPTs.

        Parameters
        ----------
        mpt_cost_rate : float
        """

        _design = self.config.get("substation_design", {})
        mpt_cost_rate = _design.get("mpt_cost_rate", 12500)

        self.mpt_cost = self.mpt_rating * self.num_mpt * mpt_cost_rate

    def calc_topside_mass_and_cost(self):
        """
        Calculates the mass and cost of the substation topsides.

        Parameters
        ----------
        topside_fab_cost_rate : int | float
        topside_design_cost: int | float
        """

        _design = self.config.get("substation_design", {})
        topside_fab_cost_rate = _design.get("topside_fab_cost_rate", 14500)
        topside_design_cost = _design.get("topside_design_cost", 4.5e6)

        self.topside_mass = 3.85 * self.mpt_rating * self.num_mpt + 285
        self.topside_cost = self.topside_mass * topside_fab_cost_rate + topside_design_cost

    def calc_shunt_reactor_cost(self):
        """
        Calculates the cost of the shunt reactor.

        Parameters
        ----------
        shunt_cost_rate : int | float
        """

        _design = self.config.get("substation_design", {})
        shunt_cost_rate = _design.get("shunt_cost_rate", 35000)

        self.shunt_reactor_cost = self.mpt_rating * self.num_mpt * shunt_cost_rate * 0.5

    def calc_switchgear_cost(self):
        """
        Calculates the cost of the switchgear.

        Parameters
        ----------
        switchgear_cost : int | float
        """

        _design = self.config.get("substation_design", {})
        switchgear_cost = _design.get("switchgear_cost", 14.5e5)

        self.switchgear_costs = self.num_mpt * switchgear_cost

    def calc_ancillary_system_cost(self):
        """
        Calculates cost of ancillary systems.

        Parameters
        ----------
        backup_gen_cost : int | float
        workspace_cost : int | float
        other_ancillary_cost : int | float
        """

        _design = self.config.get("substation_design", {})
        backup_gen_cost = _design.get("backup_gen_cost", 1e6)
        workspace_cost = _design.get("workspace_cost", 2e6)
        other_ancillary_cost = _design.get("other_ancillary_cost", 3e6)

        self.ancillary_system_costs = backup_gen_cost + workspace_cost + other_ancillary_cost

    def calc_assembly_cost(self):
        """
        Calculates the cost of assembly on land.

        Parameters
        ----------
        topside_assembly_factor : int | float
        """

        _design = self.config.get("substation_design", {})
        topside_assembly_factor = _design.get("topside_assembly_factor", 0.075)
        self.land_assembly_cost = (
            self.switchgear_costs + self.shunt_reactor_cost + self.ancillary_system_costs
        ) * topside_assembly_factor

    def calc_substructure_mass_and_cost(self):
        """
        Calculates the mass and associated cost of the substation substructure.

        Parameters
        ----------
        oss_substructure_cost_rate : int | float
        oss_pile_cost_rate : int | float
        """

        _design = self.config.get("substation_design", {})
        oss_substructure_cost_rate = _design.get("oss_substructure_cost_rate", 3000)
        oss_pile_cost_rate = _design.get("oss_pile_cost_rate", 0)

        substructure_mass = 0.4 * self.topside_mass
        substructure_pile_mass = 8 * substructure_mass**0.5574
        self.substructure_cost = (
            substructure_mass * oss_substructure_cost_rate + substructure_pile_mass * oss_pile_cost_rate
        )

        self.substructure_mass = substructure_mass + substructure_pile_mass

    @property
    def design_result(self):
        """
        Returns the results of self.run().
        """

        if not self._outputs:
            raise Exception("Has OffshoreSubstationDesign been ran yet?")

        return self._outputs

    @property
    def detailed_output(self):
        """Returns detailed phase information."""

        _outputs = {
            "num_substations": self.num_substations,
            "substation_mpt_rating": self.mpt_rating,
            "substation_topside_mass": self.topside_mass,
            "substation_topside_cost": self.topside_cost,
            "substation_substructure_mass": self.substructure_mass,
            "substation_substructure_cost": self.substructure_cost,
        }

        return _outputs
