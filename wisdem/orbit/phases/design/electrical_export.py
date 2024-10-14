"""Provides the `ElectricalDesign` class."""

__author__ = ["Sophie Bredenkamp"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ""
__email__ = []

from warnings import warn

import numpy as np

from wisdem.orbit.phases.design._cables import CableSystem

"""
[1] Maness et al. 2017, NREL Offshore Balance-of-System Model.
https://www.nrel.gov/docs/fy17osti/66874.pdf
"""


class ElectricalDesign(CableSystem):
    """
    Design phase for export cabling and offshore substation systems.

    Attributes
    ----------
    num_cables : int
        Total number of cables required for transmitting power.
    length : float
        Length of a single cable connecting the OSS to the interconnection
        in km.
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

    #:
    expected_config = {
        "site": {"distance_to_landfall": "km", "depth": "m"},
        "landfall": {"interconnection_distance": "km (optional)"},
        "plant": {"capacity": "MW"},
        "export_system_design": {
            "cables": "str",
            "num_redundant": "int (optional)",
            "touchdown_distance": "m (optional, default: 0)",
            "percent_added_length": "float (optional)",
            "interconnection_distance": "km (optional)",
            "cable_crossings": {
                "crossing_number": "int (optional)",
                "crossing_unit_cost": "float (optional)",
            },
        },
        "substation_design": {
            "substation_capacity": "MW (optional)",
            "num_substations": "int (optional)",
            "mpt_unit_cost": "USD/cable (optional)",
            "topside_design_cost": "USD (optional)",
            "shunt_unit_cost": "USD/cable (optional)",
            "switchgear_cost": "USD (optional)",
            "dc_breaker_cost": "USD (optional)",
            "backup_gen_cost": "USD (optional)",
            "workspace_cost": "USD (optional)",
            "other_ancillary_cost": "USD (optional)",
            "converter_cost": "USD (optional)",
            "onshore_converter_cost": "USD (optional)",
            "topside_assembly_factor": "float (optional)",
            "oss_substructure_type": "str (optional, default: Monopile)",
            "oss_substructure_cost_rate": "USD/t (optional)",
            "oss_pile_cost_rate": "USD/t (optional)",
        },
        "onshore_substation_design": {
            "shunt_unit_cost": "USD/cable (optional)",
            "onshore_converter_cost": "USD (optional)",
        },
    }

    output_config = {
        "num_substations": "int",
        "offshore_substation_topside": "dict",
        "offshore_substation_substructure": "dict",
        "export_system": {
            "system_cost": "USD",
            "cable": {
                "linear_density": "t/km",
                "sections": [("length, km", "speed, km/h (optional)")],
                "number": "int (optional)",
                "diameter": "int",
                "cable_type": "str",
            },
        },
        "offshore_substation": "dict, (optional)",
    }

    def __init__(self, config, **kwargs):
        """Creates an instance of ElectricalDesign."""

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        # CABLES
        super().__init__(config, "export", **kwargs)

        for name in self.expected_config["site"]:
            setattr(self, "".join(("_", name)), config["site"][name])

        self._depth = config["site"]["depth"]
        self._distance_to_landfall = config["site"]["distance_to_landfall"]
        self._plant_capacity = self.config["plant"]["capacity"]
        self._get_touchdown_distance()

        self._design = self.config["export_system_design"]

        _landfall = self.config.get("landfall", {})
        if _landfall:
            warn(
                "landfall dictionary will be deprecated and moved"
                " into [export_system_design][landfall].",
                DeprecationWarning,
                stacklevel=2,
            )

        else:
            _landfall = self._design.get("landfall", {})

        self._distance_to_interconnection = _landfall.get(
            "interconnection_distance", 3
        )

        self._oss_design = self.config.get("substation_design", {})

        self.substructure_type = self._oss_design.get(
            "oss_substructure_type", "Monopile"
        ).title()

        self._outputs = {}

    def run(self):
        """Main run function."""

        # CABLES
        self._initialize_cables()
        self.cable = self.cables[[*self.cables][0]]
        self.compute_number_cables()
        self.compute_cable_length()
        self.compute_cable_mass()
        self.compute_total_cable()
        self.calc_crossing_cost()

        self._outputs["export_system"] = {
            "landfall": {
                "interconnection_distance": (self._distance_to_interconnection)
            },
            "system_cost": self.total_cable_cost,
        }

        for cable in self.cables.values():
            self._outputs["export_system"]["cable"] = {
                "linear_density": cable.linear_density,
                "sections": [self.length],
                "number": self.num_cables,
                "cable_power": cable.cable_power,
                "cable_type": cable.cable_type,
            }

        # SUBSTATION
        self.calc_num_substations()
        self.calc_substructure_length()
        self.calc_substructure_deck_space()
        self.calc_topside_deck_space()

        self.calc_mpt_cost()
        self.calc_topside_mass_and_cost()
        self.calc_shunt_reactor_cost()
        self.calc_switchgear_costs()
        self.calc_ancillary_system_cost()
        self.calc_assembly_cost()
        self.calc_substructure_mass_and_cost()
        self.calc_converter_cost()
        self.calc_dc_breaker_cost()
        self.calc_onshore_cost()

        self._outputs["offshore_substation"] = {
            "substation_mpt_cost": self.mpt_cost,
            "substation_shunt_cost": self.shunt_reactor_cost,
            "substation_switchgear_cost": self.switchgear_cost,
            "substation_converter_cost": self.converter_cost,
            "substation_breaker_cost": self.dc_breaker_cost,
            "substation_ancillary_cost": self.ancillary_system_costs,
            "substation_land_assembly_cost": self.land_assembly_cost,
        }

        self._outputs["offshore_substation_substructure"] = {
            "type": self.substructure_type,
            "deck_space": self.substructure_deck_space,
            "mass": self.substructure_mass,
            "length": self.substructure_length,
            "unit_cost": self.substructure_cost,
        }

        # TODO: cheap fix for topside unit_cost bug #168
        self._outputs["offshore_substation_topside"] = {
            "deck_space": self.topside_deck_space,
            "mass": self.topside_mass,
            "unit_cost": self.substation_cost + self.topside_cost,
        }

        self._outputs["num_substations"] = self.num_substations
        self._outputs["total_substation_cost"] = self.total_substation_cost

    @property
    def detailed_output(self):
        """Returns export system design outputs."""

        _output = {
            **self.design_result,
            "export_system_total_mass": self.total_mass,
            "export_system_total_length": self.total_length,
            "export_system_total_cost": self.total_cable_cost,
            "export_system_cable_power": self.cable.cable_power,
            "num_substations": self.num_substations,
            "substation_mpt_rating": self.mpt_rating,
            "substation_topside_mass": self.topside_mass,
            "substation_topside_cost": self.topside_cost,
            "substation_substructure_mass": self.substructure_mass,
            "substation_substructure_cost": self.substructure_cost,
            "total_substation_cost": self.total_substation_cost,
            "substation_mpt_cost": self.mpt_cost,
            "substation_shunt_cost": self.shunt_reactor_cost,
            "substation_switchgear_cost": self.switchgear_cost,
            "substation_converter_cost": self.converter_cost,
            "substation_breaker_cost": self.dc_breaker_cost,
            "substation_ancillary_cost": self.ancillary_system_costs,
            "substation_land_assembly_cost": self.land_assembly_cost,
            "onshore_shunt_cost": self.onshore_shunt_reactor_cost,
            "onshore_converter_cost": self.onshore_converter_cost,
            "onshore_switchgear_cost": self.onshore_switchgear_cost,
            "onshore_construction_cost": self.onshore_construction,
            "onshore_compensation_cost": self.onshore_compensation_cost,
            "onshore_mpt_cost": self.mpt_cost,
        }

        return _output

    @property
    def design_result(self):
        """Returns the results of self.run()."""
        return self._outputs

        # CABLES

    @property
    def total_cable_cost(self):
        """Returns total export system cable cost."""

        return sum(self.cost_by_type.values()) + self.crossing_cost

    def compute_number_cables(self):
        """
        Calculate the total number of required and redundant cables to
        transmit power to the onshore interconnection.

        """

        num_required = np.ceil(self._plant_capacity / self.cable.cable_power)
        num_redundant = self._design.get("num_redundant", 0)

        if "HVDC" in self.cable.cable_type:
            num_required *= 2
            num_redundant *= 2

        self.num_cables = int(num_required + num_redundant)

    def compute_cable_length(self):
        """Calculates the total distance an export cable must travel."""

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
        """Calculates the total mass of a single length of export cable."""

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
        Creates an array of section lengths to work with ``CableSystem``.

        Returns
        -------
        np.ndarray
            Array of `length` with shape (``num_cables``, ).
        """
        return np.full(self.num_cables, self.length)

    @property
    def sections_cables(self):
        """
        Creates an array of cable names to work with ``CableSystem``.

        Returns
        -------
        np.ndarray
            Array of ``cable.name`` with shape (``num_cables``, ).
        """

        return np.full(self.num_cables, self.cable.name)

    def calc_crossing_cost(self):
        """Compute cable crossing costs."""
        _crossing_design = self._design.get("cable_crossings", {})

        _key = "crossing_unit_cost"
        crossing_cost = _crossing_design.get(
            _key,
            self.get_default_cost(
                "export_system_design", "cable_crossings", subkey=_key
            ),
        )

        self.crossing_cost = crossing_cost * _crossing_design.get(
            "crossing_number", 0
        )

        """SUBSTATION"""

    @property
    def total_substation_cost(self):
        """Returns the total substation cost."""

        return (
            self.topside_cost + self.substructure_cost + self.substation_cost
        )

    def calc_num_substations(self):
        """Computes number of substations based on HVDC or HVAC
        export cables.
        """

        # HVAC substation capacity
        _substation_capacity = self._oss_design.get(
            "substation_capacity", 1200
        )  # MW

        if "HVDC" in self.cable.cable_type:
            self.num_substations = self._oss_design.get(
                "num_substations", int(self.num_cables / 2)
            )
        else:
            self.num_substations = self._oss_design.get(
                "num_substations",
                int(np.ceil(self._plant_capacity / _substation_capacity)),
            )

    @property
    def substation_cost(self):
        """Returns total procuremet cost of the topside."""

        return (
            self.mpt_cost
            + self.shunt_reactor_cost
            + self.switchgear_cost
            + self.converter_cost
            + self.dc_breaker_cost
            + self.ancillary_system_costs
            + self.land_assembly_cost
        ) / self.num_substations

    def calc_mpt_cost(self):
        """Computes HVAC main power transformer (MPT). MPT cost is 0 for
        HVDC.
        """

        _key = "mpt_unit_cost"
        _mpt_cost = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        self.num_mpt = self.num_cables

        self.mpt_cost = (
            0 if "HVDC" in self.cable.cable_type else self.num_mpt * _mpt_cost
        )

        self.mpt_rating = (
            round((self._plant_capacity * 1.15 / self.num_mpt) / 10.0) * 10.0
        )

    def calc_shunt_reactor_cost(self):
        """Computes HVAC shunt reactor cost. Shunt reactor cost is 0 for
        HVDC.
        """

        touchdown = self.config["site"]["distance_to_landfall"]

        _key = "shunt_unit_cost"

        shunt_unit_cost = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        if "HVDC" in self.cable.cable_type:
            self.compensation = 0
        else:
            for cable in self.cables.values():
                self.compensation = touchdown * cable.compensation_factor  # MW

        self.shunt_reactor_cost = (
            self.compensation * shunt_unit_cost * self.num_cables
        )

    def calc_switchgear_costs(self):
        """Computes HVAC switchgear cost. Switchgear cost is 0 for HVDC."""

        _key = "switchgear_cost"
        switchgear_cost = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        self.num_switchgear = (
            0 if "HVDC" in self.cable.cable_type else self.num_cables
        )

        self.switchgear_cost = self.num_switchgear * switchgear_cost

    def calc_dc_breaker_cost(self):
        """Computes HVDC circuit breaker cost. Breaker cost is 0 for HVAC."""

        _key = "dc_breaker_cost"
        dc_breaker_cost = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        num_dc_breakers = (
            self.num_cables if "HVDC" in self.cable.cable_type else 0
        )

        self.dc_breaker_cost = num_dc_breakers * dc_breaker_cost

    def calc_ancillary_system_cost(self):
        """Calculates cost of ancillary systems."""

        _key = "backup_gen_cost"
        backup_gen_cost = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        _key = "workspace_cost"
        workspace_cost = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        _key = "other_ancillary_cost"
        other_ancillary_cost = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        self.ancillary_system_costs = (
            backup_gen_cost + workspace_cost + other_ancillary_cost
        ) * self.num_substations

    def calc_assembly_cost(self):
        """Calculates the cost of assembly on land."""

        _key = "topside_assembly_factor"
        topside_assembly_factor = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        if topside_assembly_factor > 1.0:
            topside_assembly_factor /= 100

        self.land_assembly_cost = (
            self.switchgear_cost
            + self.shunt_reactor_cost
            + self.ancillary_system_costs
        ) * topside_assembly_factor

    def calc_converter_cost(self):
        """Computes converter cost."""

        _key = "converter_cost"
        converter_cost = self._oss_design.get(
            _key,
            self.get_default_cost(
                "substation_design", _key, subkey=self.cable.cable_type
            ),
        )

        self.converter_cost = converter_cost

    def calc_substructure_mass_and_cost(self):
        """
        Calculates the mass and associated cost of the substation substructure
        based on equations 81-84 [1].
        """

        _key = "oss_substructure_cost_rate"
        oss_substructure_cost_rate = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        _key = "oss_pile_cost_rate"
        oss_pile_cost_rate = self._oss_design.get(
            _key, self.get_default_cost("substation_design", _key)
        )

        # Substructure mass components
        # TODO: Determine a better method to calculate substructure mass
        #       for different substructure types
        substructure_mass = 0.4 * self.topside_mass

        substructure_pile_mass = (
            0
            if self.substructure_type == "Floating"
            else 8 * substructure_mass**0.5574
        )

        self.substructure_cost = (
            substructure_mass * oss_substructure_cost_rate
            + substructure_pile_mass * oss_pile_cost_rate
        )

        self.substructure_mass = substructure_mass + substructure_pile_mass

    def calc_substructure_length(self):
        """Calculates substructure length as the site depth + 10m."""

        if self.substructure_type == "Floating":
            self.substructure_length = 0

        else:
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

    def calc_topside_mass_and_cost(self):
        """Calculates the mass and cost of the substation topsides."""

        self.topside_mass = (
            3.85 * (self.mpt_rating * self.num_mpt) / self.num_substations
            + 285
        )

        _key = "topside_design_cost"
        topside_design_cost = self._oss_design.get(
            _key,
            self.get_default_cost(
                "substation_design", _key, subkey=self.cable.cable_type
            ),
        )

        self.topside_cost = topside_design_cost

    def calc_onshore_cost(self):
        """Minimum Cost of Onshore Substation Connection."""

        _design = self.config.get("onshore_substation_design", {})

        _key = "onshore_converter_cost"
        _converter_cost = _design.get(
            _key,
            self.get_default_cost(
                "onshore_substation_design", _key, subkey=self.cable.cable_type
            ),
        )

        self.onshore_converter_cost = self.num_substations * _converter_cost

        _key = "switchgear_cost"
        _switchgear_cost = _design.get(
            _key, self.get_default_cost("onshore_substation_design", _key)
        )

        self.onshore_switchgear_cost = self.num_switchgear * _switchgear_cost

        _key = "onshore_construction_rate"
        _construction_rate = _design.get(
            _key,
            self.get_default_cost(
                "onshore_substation_design", _key, subkey=self.cable.cable_type
            ),
        )

        self.onshore_construction = self.num_substations * _construction_rate

        _key = "shunt_unit_cost"
        _shunt_unit_cost = _design.get(
            _key, self.get_default_cost("onshore_substation_design", _key)
        )

        self.onshore_shunt_reactor_cost = (
            self.compensation * self.num_cables * _shunt_unit_cost
        )

        _key = "compensation_rate"
        _compensation_rate = _design.get(
            _key,
            self.get_default_cost(
                "onshore_substation_design", _key, subkey=self.cable.cable_type
            ),
        )

        self.onshore_compensation_cost = (
            self.num_cables * _compensation_rate
            + self.onshore_shunt_reactor_cost
        )

        self.onshore_cost = (
            self.onshore_converter_cost
            + self.onshore_switchgear_cost
            + self.onshore_construction
            + self.onshore_compensation_cost
            + self.mpt_cost
        )

        self._outputs["export_system"][
            "onshore_substation_costs"
        ] = self.onshore_cost
