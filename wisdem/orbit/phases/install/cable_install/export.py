"""`ExportCableInstallation` and related processes."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy
from math import ceil
from warnings import warn

from marmot import process

from wisdem.orbit.core.logic import position_onsite
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.core.exceptions import InsufficientCable

from .common import SimpleCable as Cable
from .common import (
    lay_cable,
    bury_cable,
    dig_trench,
    pull_in_cable,
    landfall_tasks,
    lay_bury_cable,
    splice_process,
    terminate_cable,
    load_cable_on_vessel,
)


class ExportCableInstallation(InstallPhase):
    """Export Cable Installation Phase."""

    phase = "Export Cable Installation"
    capex_category = "Export System"

    #:
    expected_config = {
        "landfall": {"trench_length": "km (optional)"},
        "export_cable_install_vessel": "str | dict",
        "export_cable_bury_vessel": "str | dict (optional)",
        "export_cable_trench_vessel": "str (optional)",
        "site": {"distance": "km"},
        "plant": {"capacity": "MW"},
        "export_system": {
            "system_cost": "USD",
            "cable": {
                "linear_density": "t/km",
                "sections": [("length, km", "speed, km/h (optional)")],
                "number": "int (optional)",
                "cable_type": "str(optional, defualt: 'HVAC')",
                "landfall": {
                    "trench_length": "km (optional)",
                    "interconnection_distance": "km (optional); default: 3km",
                },
            },
            "interconnection_distance": "km (optional)",
            "interconnection_voltage": "kV (optional); default: 345kV",
            "onshore_construction_cost": "$, (optional)",
            "onshore_construction_time": "h, (optional)",
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of ExportCableInstallation.

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
        self.extract_distances()
        self.setup_simulation(**kwargs)

    def setup_simulation(self, **kwargs):
        """
        Setup method for the ExportCableInstallation phase.
        - Extracts key inputs
        - Performs onshore infrastructure construction
        - Routes to specific setup scripts based on configured strategy.
        """

        depth = self.config["site"]["depth"]
        system = self.config["export_system"]
        self.free_cable_length = system.get("free_cable_length", depth / 1000)

        self.cable = Cable(system["cable"]["linear_density"])
        self.cable_type = system["cable"].get("cable_type", "HVAC")
        self.sections = system["cable"]["sections"]

        if self.cable_type == "HVDC-monopole":
            self.number = int(system["cable"].get("number", 2) / 2)
        else:
            self.number = system["cable"].get("number", 1)

        self.initialize_installation_vessel()
        self.initialize_burial_vessel()
        self.initialize_trench_vessel()

        # Perform onshore construction
        onshore = kwargs.get("include_onshore_construction", True)
        if onshore:
            self.onshore_construction(**kwargs)

        # Perform cable installation
        install_export_cables(
            self.install_vessel,
            sections=self.sections,
            cable=self.cable,
            number=self.number,
            distances=self.distances,
            burial_vessel=self.bury_vessel,
            trench_vessel=self.trench_vessel,
            free_cable_length=self.free_cable_length,
            **kwargs,
        )

    @property
    def system_capex(self):
        """Returns total procurement cost of the array system."""

        return self.config["export_system"]["system_cost"]

    def extract_distances(self):
        """Extracts distances from input configuration or default values."""

        site = self.config["site"]["distance"]

        _landfall = self.config.get("landfall", {})
        if _landfall:
            warn(
                "landfall dictionary will be deprecated and moved"
                " into [export_system][landfall].",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            _landfall = self.config["export_system"].get("landfall", {})

        trench = _landfall.get("trench_length", 1)

        self.distances = {"site": site, "trench": trench}

    def onshore_construction(self, **kwargs):
        """
        Performs onshore construction prior to the installation of the export
        cable system.

        Parameters
        ----------
        construction_time : int | float
            Amount of time onshore construction takes.
            Default: 0h
        construction_rate : int | float
            Day rate of onshore construction.
            Default: 50000 USD/day
        """

        construction_time = self.config["export_system"].get(
            "onshore_construction_time", 0.0
        )
        construction_cost = self.config["export_system"].get(
            "onshore_construction_cost", None
        )
        if construction_cost is None:
            construction_cost = self.calculate_onshore_transmission_cost(
                **kwargs
            )

        if construction_time:
            _ = self.env.timeout(construction_time)
            self.env.run()

        self.env._submit_log(
            {
                "action": "Onshore Construction",
                "agent": "Onshore Construction",
                "duration": construction_time,
                "cost": construction_cost,
                "location": "Landfall",
            },
            level="ACTION",
        )

    def calculate_onshore_transmission_cost(self, **kwargs):
        """
        Calculates the cost of onshore transmission costs. From legacy
        OffshoreBOS model.
        """

        # capacity = self.config["plant"]["capacity"]
        system = self.config["export_system"]
        voltage = system.get("interconnection_voltage", 345)
        distance = system.get("interconnection_distance", None)

        if distance:
            warn(
                "[export_system][interconnection_distance] will be deprecated"
                " and moved to"
                " [export_system][landfall][interconnection_distance].",
                DeprecationWarning,
                stacklevel=2,
            )

        landfall = system.get("landfall", {})
        distance = landfall.get("interconnection_distance", 3)

        # switchyard_cost = 18115 * voltage + 165944
        # onshore_substation_cost = (
        #     0.165 * 1e6
        # ) * capacity  # From BNEF Tomorrow's Cost of Offshore Wind
        # onshore_misc_cost = 11795 * capacity**0.3549 + 350000
        transmission_line_cost = (1176 * voltage + 218257) * (
            distance ** (1 - 0.1063)
        )

        self.onshore_transmission_cost = (
            transmission_line_cost
            #                        + switchyard_cost
            #                        + onshore_substation_cost
            #                        + onshore_misc_cost
        )

        return self.onshore_transmission_cost

    def initialize_installation_vessel(self):
        """Creates the export cable installation vessel."""

        # Vessel name and costs
        vessel_specs = self.config.get("export_cable_install_vessel", None)
        name = vessel_specs.get("name", "Export Cable Installation Vessel")

        vessel = self.initialize_vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        self.install_vessel = vessel

    def initialize_burial_vessel(self):
        """Creates the export cable burial vessel."""

        # Vessel name and costs
        vessel_specs = self.config.get("export_cable_bury_vessel", None)
        if vessel_specs is None:
            self.bury_vessel = None
            return

        name = vessel_specs.get("name", "Export Cable Burial Vessel")

        vessel = self.initialize_vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        self.bury_vessel = vessel

    def initialize_trench_vessel(self):
        """Creates the export cable trenching vessel."""

        # Vessel name and costs
        vessel_specs = self.config.get("export_cable_trench_vessel", None)
        if vessel_specs is None:
            self.trench_vessel = None
            return
        name = vessel_specs.get("name", "Export Cable Trench Vessel")

        vessel = self.initialize_vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        vessel.at_port = True
        vessel.at_site = False
        self.trench_vessel = vessel

    @property
    def detailed_output(self):
        """Detailed outputs of the export system installation."""

        outputs = {self.phase: {**self.agent_efficiencies}}

        return outputs


@process
def install_export_cables(
    vessel,
    sections,
    cable,
    number,
    distances,
    burial_vessel=None,
    trench_vessel=None,
    free_cable_length=None,
    **kwargs,
):
    """
    Simulation of the installation of export cables.

    Parameters
    ----------
    vessel : Vessel
        Cable installation vessel.
    sections : float
        Section lengths of the export cable.
    cable : SimpleCable | Cable
        Cable type to use.
    number : int
        Number of export cables.
    distances : dict
        Distances required for export cable installation simulation:
        site : int | float
            Distance between from the offshore substation and port. For
            simplicity, the cable landfall point is assumed to be at port.
        trench : int | float
            Trench length at landfall. Determines time required to tow the plow
            and pull-in cable (km).
    burial_vessel : Vessel
        Optional configuration for burial vessel. If configured, the
        installation vessel only lays the cable on the seafloor and this
        vessel will bury them at the end of the simulation.
    trench_vessel: Vessel
        Optional configuration for trenching vessel.  If configured, the
        trenching vessel travels along the cable route prior to arrival of
        the cable lay vessel and digs a trench.
    """

    ground_distance = -free_cable_length
    for s in sections:
        try:
            length, speed = s

        except TypeError:
            length = s

        ground_distance += length

    # Conduct trenching operations
    if trench_vessel is None:
        pass

    else:
        for _ in range(number):
            # Trench vessel can dig a trench during inbound or outbound journey
            if trench_vessel.at_port:
                trench_vessel.at_port = False
                yield dig_export_cables_trench(
                    trench_vessel,
                    ground_distance,
                    **kwargs,
                )
                trench_vessel.at_site = True
            elif trench_vessel.at_site:
                trench_vessel.at_site = False
                yield dig_export_cables_trench(
                    trench_vessel,
                    ground_distance,
                    **kwargs,
                )
                trench_vessel.at_port = True

        # If the vessel finishes trenching at site, return to shore
        # TODO: replace with demobilization method
        if trench_vessel.at_site:
            trench_vessel.at_site = False
            yield trench_vessel.transit(ground_distance, **kwargs)
        trench_vessel.at_port = True

    for _ in range(number):
        vessel.cable_storage.reset()
        yield load_cable_on_vessel(vessel, cable, **kwargs)

        # At Landfall
        yield landfall_tasks(vessel, distances["trench"], **kwargs)

        for s in sections:
            splice_required = False
            try:
                length, speed = s
                if burial_vessel is None:
                    specs = {**kwargs, "cable_lay_bury_speed": speed}

                else:
                    specs = {**kwargs, "cable_lay_speed": speed}

            except TypeError:
                length = s
                specs = deepcopy(kwargs)

            remaining = length
            while remaining > 0:
                if splice_required:
                    yield splice_process(vessel, **kwargs)

                try:
                    section = vessel.cable_storage.get_cable(remaining)

                except InsufficientCable as e:
                    section = vessel.cable_storage.get_cable(e.current)

                if burial_vessel is None:
                    yield lay_bury_cable(vessel, section, **specs)

                else:
                    yield lay_cable(vessel, section, **specs)

                remaining -= ceil(section)
                if remaining > 0:
                    splice_required = True

                    yield vessel.transit(distances["site"])
                    vessel.cable_storage.reset()
                    yield load_cable_on_vessel(vessel, cable, **kwargs)
                    yield vessel.transit(distances["site"])

        # At Site
        yield position_onsite(vessel, **kwargs)
        yield pull_in_cable(vessel, **kwargs)
        yield terminate_cable(vessel, **kwargs)

        # Transit back to port
        yield vessel.transit(distances["site"])

    if burial_vessel is None:
        vessel.submit_debug_log(
            message="Export cable lay/burial process completed!",
            progress="Export System",
        )

    else:
        vessel.submit_debug_log(message="Export cable lay process completed!")
        bury_export_cables(burial_vessel, ground_distance, number, **kwargs)


@process
def bury_export_cables(vessel, length, number, **kwargs):
    """
    Simulation for the burial of export cables if configured.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel.
    length : float
        Full length of an export cable.
    number : int
        Number of export cables.
    """

    for _ in range(number):
        yield bury_cable(vessel, length, **kwargs)

    vessel.submit_debug_log(
        message="Export cable burial process completed!",
        progress="Export System",
    )


@process
def dig_export_cables_trench(vessel, distance, **kwargs):
    """
    Simulation for digging a trench for the export cables (if configured).

    Parameters
    ----------
    vessel : Vessel
        Performing vessel.
    distance : int | float
        Distance along export cable route to dig trench for cable
    """

    yield position_onsite(vessel, site_position_time=2)
    yield dig_trench(vessel, distance, **kwargs)

    vessel.submit_debug_log(
        message="Export cable trench digging process completed!"
    )
