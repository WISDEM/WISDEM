"""Provides the base `Cable`, `Plant`, and `CableSystem` classes."""

__author__ = ["Matt Shields", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import math
from collections import Counter, OrderedDict

import numpy as np
from scipy.optimize import fsolve
from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.phases.design import DesignPhase


class Cable:
    """
    Base cable class

    Attributes
    ----------
    conductor_size : float
        Cable cross section in :math:`mm^2`.
    current_capacity : float
        Cable current rating at 1m burial depth, :math:`A`.
    rated_voltage : float
        Cable rated line-to-line voltage, :math:`kV`.
    ac_resistance : float
        Cable resistance for AC current, (ohms/km).
    inductance : float
        Cable inductance, :math:`\\frac{MHz}{km}`.
    capacitance : float
        Cable capacitance, :math:`\\frac{nF}{km}`.
    linear_density : float
        Dry mass per kilometer, :math:`\\frac{tonnes}{km}`.
    cost_per_km : int
        Cable cost per kilometer, :math:`\\frac{USD}{km}`.
    char_impedance : float
        Characteristic impedance of equivalent cable circuit, (ohms).
    power_factor : float
        Power factor of AC current in cable, no units.
    cable_power : float
        Maximum 3-phase power dissipated in cable in :math:`MW`.
    line_frequency: int, default: 60
        Frequency of the AC current, :math:`Hz`.
    """

    required = (
        "conductor_size",
        "current_capacity",
        "rated_voltage",
        "ac_resistance",
        "inductance",
        "capacitance",
        "linear_density",
        "cost_per_km",
        "name",
    )

    def __init__(self, cable_specs, **kwargs):
        """
        Create an instance of Cable (either array or export)

        Parameters
        ---------
        cable_specs : dict
            Dictionary containing cable specifications.
        kwargs : dict
            Additional user inputs.
        """

        needs_value = []
        for r in self.required:
            x = cable_specs.get(r, None)
            if x is not None:
                setattr(self, r, x)
            else:
                needs_value.append(r)

        if needs_value:
            raise ValueError(f"{needs_value} must be defined in cable_specs")

        self.line_frequency = cable_specs.get("line_frequency", 60)

        # Calc additional cable specs
        self.calc_char_impedance(**kwargs)
        self.calc_power_factor()
        self.calc_cable_power()

    def calc_char_impedance(self):
        """
        Calculate characteristic impedance of cable.
        """

        conductance = 1 / self.ac_resistance

        num = complex(
            self.ac_resistance,
            2 * math.pi * self.line_frequency * self.inductance,
        )
        den = complex(conductance, 2 * math.pi * self.line_frequency * self.capacitance)
        self.char_impedance = np.sqrt(num / den)

    def calc_power_factor(self):
        """
        Calculate power factor.
        """

        phase_angle = math.atan(np.imag(self.char_impedance) / np.real(self.char_impedance))
        self.power_factor = math.cos(phase_angle)

    def calc_cable_power(self):
        """
        Calculate maximum power transfer through 3-phase cable in :math:`MW`.
        """

        self.cable_power = np.sqrt(3) * self.rated_voltage * self.current_capacity * self.power_factor / 1000


class Plant:
    """
    A "data class" to create the windfarm specifications for
    `ArraySystemDesign`.

    Attributes
    ----------
    layout : str
        The layout of the windfarm. Can only be "grid", "ring", or "custom".
        ..note:: custom is not implemented at this time.
    num_turbines : int
        Number of turbines contained in the windfarm.
    site_depth : float
        Average depth at the site in km.
    turbine_rating : float
        Capacity of an individual turbine in MW.
    row_distance : float
        Distance between any two strings in a grid layout in km. This is not
        used for ring layouts or custom layouts.
    turbine_distance : float
        Distance between any two turbines in a string in km. This is not used
        for custom layouts.
    substation_distance : float
        The shortest distance between the offshore substation and the first
        turbine of each string in km. In the ring layout this distance is
        uniform across all strings. In grid layout this represents the
        perpendicular distance to the first row of turbines. This is not used
        in custom layouts.
    """

    expected_config = {
        "plant": {
            "num_turbines": "int",
            "layout": "str",
            "turbine_distance": "km (optional)",
            "turbine_spacing": "rotor diameters",
            "row_spacing": "rotor diameters (optional)",
            "row_distance": "km (optional)",
            "substation_distance": "km",
        },
        "site": {"depth": "m"},
        "turbine": {"turbine_rating": "MW", "rotor_diameter": "m"},
    }

    def __init__(self, config):
        """
        Creates the object.

        Parameters
        ----------
        config : dict
            Dictionary of configuration settings.
        """

        self.num_turbines = config["plant"]["num_turbines"]
        self.site_depth = config["site"]["depth"] / 1000.0
        self.turbine_rating = config["turbine"]["turbine_rating"]

        self.layout = config["plant"]["layout"].lower()
        if self.layout not in ("custom", "grid", "ring"):
            raise ValueError("config: site: layout should be one of " "'custom', 'ring', or 'grid'.")
        if self.layout != "custom":
            self._initialize_distances(config)

    def _initialize_distances(self, config):
        """
        Initializes the distance attributes that are created for a grid and
        ring layout.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """

        rotor_diameter = config["turbine"]["rotor_diameter"]

        self.turbine_distance = config["plant"].get("turbine_distance", None)
        if self.turbine_distance is None:
            self.turbine_distance = rotor_diameter * config["plant"]["turbine_spacing"] / 1000.0

        self.substation_distance = config["plant"].get("substation_distance", None)
        if self.substation_distance is None:
            self.substation_distance = self.turbine_distance

        self.row_distance = config["plant"].get("row_distance", None)
        if self.row_distance is None:
            if self.layout == "grid":
                self.row_distance = rotor_diameter * config["plant"]["row_spacing"] / 1000.0
            else:
                self.row_distance = self.turbine_distance


class CableSystem(DesignPhase):
    """
    Base cabling system class. This is the parent class to
    `ArraySystemDesign` and `ExportSystemDesign`.

    Attributes
    ----------
    cable_type : str
        An input of "array" or "export" to signify which cabling system is
        being designed.
    cables : dict
        Dictionary of cables being used with items as {"cable_name": `Cable`}.

    Raises
    ------
    NotImplementedError
        The property `detailed_output` is not yet defined.
    Exception
        `cables` must be created in order to get outputs.
    """

    cables = None

    def __init__(self, config, cable_type, **kwargs):
        """
        Initializes the configuration.

        Parameters
        ----------
        cable_type : str
            Type of cabling system. Only "array" or "export" should be passed.
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self.cable_type = cable_type.lower()
        if self.cable_type not in ("array", "export"):
            raise ValueError(
                f"{self.cable_type} is invalid input to `cable_type`. '",
                "Input must be `array` or `export`.",
            )

        self._design = config["_".join((cable_type, "system_design"))]

    def _initialize_cables(self):
        """
        Creates the base cable objects for each type of array cable being used.
        """

        if isinstance(self._design["cables"], str):
            self._design["cables"] = [self._design["cables"]]
        if isinstance(self._design["cables"], list):
            _cables = {}
            for name in self._design["cables"]:
                _cables[name] = extract_library_specs("cables", name)
            self._design["cables"] = _cables

        _cables = {}
        for name, cable in self._design["cables"].items():
            cable.setdefault("name", name)
            _cables[name] = cable
        self._design["cables"] = _cables

        # Cables are ordered from smallest to largest
        cables = OrderedDict(
            sorted(
                self._design["cables"].items(),
                key=lambda item: item[1]["current_capacity"],
            )
        )

        # Instantiate cables as Cable objects. Use "name" property from file
        # instead of filename.
        self.cables = OrderedDict()
        for specs in cables.values():
            name = specs["name"]
            self.cables[name] = Cable(specs)

    def _get_touchdown_distance(self):
        """
        Returns the cable touchdown distance measured from the centerpoint of
        the substructure.

        If depth <= 60, default is 0km (straight down assumed for fixed bottom).
        If depth > 60, default is 0.3 * depth.
        """

        _design = f"{self.cable_type}_system_design"
        depth = self.config["site"]["depth"]
        touchdown = self.config[_design].get("touchdown_distance", None)

        if touchdown is not None:
            self.touchdown = touchdown

        else:
            if depth <= 60:
                self.touchdown = 0

            else:
                self.touchdown = depth * 0.3
                # TODO: Update this scaling function - should be closer to cable bend radius.  Unrealistic for deep water

    @staticmethod
    def _catenary(a, *data):
        """Simple catenary equation."""

        d, h = data
        res = a * np.cosh(h / a) - (d + a)
        return res

    def _get_catenary_length(self, d, h):
        """
        Returns the catenary length of a cable that touches down at depth `d`
        and horizontal distance `h`.

        Returns
        -------
        float
            Catenary length.
        """

        a = fsolve(self._catenary, 8, (d, h))

        x = np.linspace(0, h)
        y = a * np.cosh(x / a) - a

        if not np.isclose(y[-1], d):
            print("Warning: Catenary calculation failed. Reverting to simple vertical profile.")
            return d

        return np.trapz(np.sqrt(1 + np.gradient(y, x) ** 2), x)

    @property
    def free_cable_length(self):
        """Returns the length of the vertical portion of a cable section in km."""

        _design = f"{self.cable_type}_system_design"
        depth = self.config["site"]["depth"]
        _cable_depth = self.config[_design].get("floating_cable_depth", depth)

        # Select prescribed cable depth if it is less than or equal to overall water dpeth
        if _cable_depth > depth:
            cable_depth = depth
        else:
            cable_depth = _cable_depth

        if not self.touchdown:
            return cable_depth / 1000

        return self._get_catenary_length(cable_depth, self.touchdown) / 1000

    @property
    def cable_lengths_by_type(self):
        """
        Creates dictionary of lists of cable sections for each type of cable

        Returns
        -------
        lengths : dict
            A dictionary of the section lengths required for each type of cable
            to fully connect the array cabling system.
            E.g.: {`Cable`.`name`: np.ndarray(float)}
        """

        lengths = {name: self.sections_cable_lengths[np.where(self.sections_cables == name)] for name in self.cables}
        return lengths

    @property
    def total_cable_length_by_type(self):
        """
        Calculates the total cable length for each type of cable.

        Returns
        -------
        total : dict
            A dictionary of the total cable length for each type of cable.
            E.g.: {`Cable.name`: list(section_lengths)}
        """

        total = {name: sections.sum() for name, sections in self.cable_lengths_by_type.items()}
        return total

    @property
    def cost_by_type(self):
        """
        Calculates the cost of each array cable type.

        Returns
        -------
        cost : dict
            A dictionary of the total cost of each type of array cable.
            E.g.: {`Cable.name`: cost}
        """

        cost = {
            name: length * self.cables[name].cost_per_km for name, length in self.total_cable_length_by_type.items()
        }
        return cost

    @property
    def total_cost(self):
        """
        Calculates the cost of the array cabling system.

        Returns
        -------
        float
            Total cost of the array cabling system.
        """

        return sum(self.cost_by_type.values())

    @property
    def detailed_output(self):
        """Returns detailed design outputs."""

        _output = {
            "length": self.total_cable_length_by_type,
            "cost": self.cost_by_type,
        }

        return _output

    @property
    def design_result(self):
        """
        A dictionary of cables types and number of different cable lengths and
        linear density.

        Returns
        -------
        output : dict
            Dictionary of the number of section lengths and the linear density
            of each cable type.
             - <`cable_type`>_system: dict
                - cables: dict
                    - `Cable.name`: dict
                        - sections: [
                            (length of unique section, number of sections)
                          ],
                        - linear_density: `Cable.linear_density`
        """

        if self.cables is None:
            raise Exception(f"Has {self.__class__.__name__} been ran?")

        system = "_".join((self.cable_type, "system"))
        output = {system: {"cables": {}, "system_cost": self.total_cost}}
        _temp = output[system]["cables"]

        for name, cable in self.cables.items():
            try:
                sections = self.cable_lengths_by_type_speed[name]
            except AttributeError:
                sections = self.cable_lengths_by_type[name]

            try:
                sections = [(data[0], count, *data[1:]) for data, count in Counter(sections).items()]
            except IndexError:
                sections = [(*data[:-1], data[-1]) for data in Counter(sections).items()]

            if sections:
                _temp[name] = {
                    "cable_sections": sections,
                    "linear_density": cable.linear_density,
                }

        return output
