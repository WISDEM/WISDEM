"""`ArrayCableInstallation` class and related processes."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import numpy as np
from marmot import process

from wisdem.orbit.core import Vessel
from wisdem.orbit.core.logic import position_onsite
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.core.exceptions import InsufficientCable

from .common import SimpleCable as Cable
from .common import (
    lay_cable,
    bury_cable,
    dig_trench,
    prep_cable,
    lower_cable,
    pull_in_cable,
    lay_bury_cable,
    terminate_cable,
    load_cable_on_vessel,
)


class ArrayCableInstallation(InstallPhase):
    """Array Cable Installation Phase"""

    phase = "Array Cable Installation"
    capex_category = "Array System"

    #:
    expected_config = {
        "array_cable_install_vessel": "str",
        "array_cable_bury_vessel": "str (optional)",
        "array_cable_trench_vessel": "str (optional)",
        "site": {"distance": "km", "depth": "m"},
        "array_system": {
            "system_cost": "USD",
            "num_strings": "int (optional, default: 10)",
            "free_cable_length": "km (optional, default: 'depth')",
            "cables": {
                "name (variable)": {
                    "linear_density": "t/km",
                    "cable_sections": [("length, km", "int", "speed, km/h (optional)")],
                }
            },
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of ArrayCableInstallation.

        Parameters
        ----------
        config : dict
            Simulation specific configuration.
        weather : np.ndarray
            Weather profile at site.
        """

        super().__init__(weather, **kwargs)

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self.initialize_port()
        self.setup_simulation(**kwargs)

    def setup_simulation(self, **kwargs):
        """
        Setup method for ArrayCableInstallation phase.
        - Extracts key inputs
        -
        """

        depth = self.config["site"]["depth"]
        system = self.config["array_system"]
        self.free_cable_length = system.get("free_cable_length", depth / 1000)

        self.initialize_installation_vessel()
        self.initialize_burial_vessel()
        self.initialize_trench_vessel()

        self.num_strings = system.get("num_strings", 10)
        self.cable_data = [
            (Cable(data["linear_density"]), deepcopy(data["cable_sections"])) for _, data in system["cables"].items()
        ]

        # Perform cable installation
        install_array_cables(
            self.install_vessel,
            distance=self.config["site"]["distance"],
            cable_data=self.cable_data,
            num_strings=self.num_strings,
            burial_vessel=self.bury_vessel,
            trench_vessel=self.trench_vessel,
            free_cable_length=self.free_cable_length,
            **kwargs,
        )

    @property
    def system_capex(self):
        """Returns total procurement cost of the array system."""

        return self.config["array_system"]["system_cost"]

    def initialize_installation_vessel(self):
        """Creates the array cable installation vessel."""

        # Vessel name and costs
        vessel_specs = self.config.get("array_cable_install_vessel", None)
        name = vessel_specs.get("name", "Array Cable Installation Vessel")

        vessel = self.initialize_vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        vessel.at_port = True
        vessel.at_site = False
        self.install_vessel = vessel

    def initialize_burial_vessel(self):
        """Creates the array cable burial vessel."""

        # Vessel name and costs
        vessel_specs = self.config.get("array_cable_bury_vessel", None)
        if vessel_specs is None:
            self.bury_vessel = None
            return
        name = vessel_specs.get("name", "Array Cable Burial Vessel")

        vessel = self.initialize_vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        vessel.at_port = True
        vessel.at_site = False
        self.bury_vessel = vessel

    def initialize_trench_vessel(self):
        """Creates the array cable trenching vessel."""

        # Vessel name and costs
        vessel_specs = self.config.get("array_cable_trench_vessel", None)
        if vessel_specs is None:
            self.trench_vessel = None
            return
        name = vessel_specs.get("name", "Array Cable Trench Vessel")

        vessel = self.initialize_vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        vessel.at_port = True
        vessel.at_site = False
        self.trench_vessel = vessel

    @property
    def detailed_output(self):
        """Detailed outputs of the array system installation."""

        outputs = {self.phase: {**self.agent_efficiencies}}

        return outputs


@process
def install_array_cables(
    vessel,
    distance,
    cable_data,
    num_strings,
    burial_vessel=None,
    trench_vessel=None,
    free_cable_length=None,
    **kwargs,
):
    """
    Simulation of the installation of array cables.

    Parameters
    ----------
    vessel : Vessel
        Cable installation vessel.
    cable_data : list
        List of tuples containing `Cable` instances and sections.
    num_strings : int
        Number of array system strings. Used for partial generation assumptions
        for the post-processed cash flow model.
    burial_vessel : Vessel
        Optional configuration for burial vessel. If configured, the
        installation vessel only lays the cable on the seafloor and this
        vessel will bury them at the end of the simulation.
    trench_vessel: Vessel
        Optional configuration for trenching vessel.  If configured, the
        trenching vessel travels along the cable route prior to arrival of
        the cable lay vessel and digs a trench.
    """

    breakpoints = list(np.linspace(1 / num_strings, 1, num_strings))
    trench_sections = []
    total_cable_length = 0
    installed = 0

    for cable, sections in cable_data:
        for s in sections:
            l, num_i, *_ = s
            total_cable_length += l * num_i

            _trench_length = max(0, l - 2 * free_cable_length)
            if _trench_length:
                trench_sections.extend([_trench_length] * num_i)

    ## Trenching Process
    # Conduct trenching along cable routes before laying cable
    if trench_vessel is None:
        pass

    else:
        # Conduct trenching operations
        while True:
            if trench_vessel.at_port:
                trench_vessel.at_port = False
                yield trench_vessel.transit(distance, **kwargs)
                trench_vessel.at_site = True

            elif trench_vessel.at_site:

                try:
                    # Dig trench along each cable section distance
                    trench_distance = trench_sections.pop(0)
                    yield dig_array_cables_trench(trench_vessel, trench_distance, **kwargs)

                except IndexError:
                    trench_vessel.at_site = False
                    yield trench_vessel.transit(distance, **kwargs)
                    trench_vessel.at_port = True
                    break

        vessel.submit_debug_log(message="Array cable trench digging process completed!")

    ## Cable Lay Process
    to_bury = []
    for cable, sections in cable_data:
        vessel.cable_storage.reset()

        while True:
            if vessel.at_port:
                yield load_cable_on_vessel(vessel, cable, **kwargs)

                vessel.at_port = False
                yield vessel.transit(distance, **kwargs)
                vessel.at_site = True

            elif vessel.at_site:

                try:
                    length, num_sections, *extra = sections.pop(0)
                    if extra:
                        speed = extra[0]

                        if burial_vessel is None:
                            specs = {**kwargs, "cable_lay_bury_speed": speed}

                        else:
                            specs = {**kwargs, "cable_lay_speed": speed}

                    else:
                        specs = deepcopy(kwargs)

                except IndexError:
                    vessel.at_site = False
                    yield vessel.transit(distance, **kwargs)
                    vessel.at_port = True
                    break

                for _ in range(num_sections):

                    try:
                        section = vessel.cable_storage.get_cable(length)

                    except InsufficientCable:

                        yield vessel.transit(distance, **kwargs)
                        yield load_cable_on_vessel(vessel, cable, **kwargs)
                        yield vessel.transit(distance, **kwargs)
                        section = vessel.cable_storage.get_cable(length)

                    # Prep for cable laying procedure (at substructure 1)
                    yield position_onsite(vessel, **kwargs)
                    yield prep_cable(vessel, **kwargs)
                    yield pull_in_cable(vessel, **kwargs)
                    yield terminate_cable(vessel, **kwargs)
                    yield lower_cable(vessel, **kwargs)

                    # Cable laying procedure
                    if burial_vessel is None:
                        yield lay_bury_cable(vessel, section, **specs)
                        installed += section

                    else:
                        yield lay_cable(vessel, section, **specs)
                        _bury = max(0, (section - 2 * free_cable_length))
                        if _bury:
                            to_bury.append(_bury)

                    # Post cable laying procedure (at substructure 2)
                    yield prep_cable(vessel, **kwargs)
                    yield pull_in_cable(vessel, **kwargs)
                    yield terminate_cable(vessel, **kwargs)

                    if burial_vessel is None:
                        breakpoints = check_for_completed_string(vessel, installed, total_cable_length, breakpoints)

        # Transit back to port
        vessel.at_site = False
        yield vessel.transit(distance, **kwargs)
        vessel.at_port = True

    ## Burial Process
    if burial_vessel is None:
        vessel.submit_debug_log(message="Array cable lay/burial process completed!")

    else:
        vessel.submit_debug_log(message="Array cable lay process completed!")
        bury_array_cables(burial_vessel, to_bury, breakpoints, **kwargs)


@process
def bury_array_cables(vessel, sections, breakpoints, **kwargs):
    """
    Simulation for the burial of array cables if configured.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel.
    sections : list
        List of cable sections that need to be buried at site.
    breakpoints : list
        TODO
        List of string breakpoints.
    """

    installed = 0
    total_length = sum(sections)

    for length in sections:
        yield position_onsite(vessel, site_position_time=2)
        yield bury_cable(vessel, length, **kwargs)
        installed += length

        breakpoints = check_for_completed_string(vessel, installed, total_length, breakpoints)

    vessel.submit_debug_log(message="Array cable burial process completed!")


@process
def dig_array_cables_trench(vessel, distance, **kwargs):
    """
    Simulation for digging a trench for the array cables (if configured).

    Parameters
    ----------
    vessel : Vessel
        Performing vessel.
    distance : int | float
        Distance between turbines to dig trench for array cable
    """

    yield position_onsite(vessel, site_position_time=2)
    yield dig_trench(vessel, distance, **kwargs)


def check_for_completed_string(vessel, installed, total, breakpoints):
    """
    TODO:
    """

    if (installed / total) >= breakpoints[0]:
        vessel.submit_debug_log(progress="Array String")
        _ = breakpoints.pop(0)

    return breakpoints
