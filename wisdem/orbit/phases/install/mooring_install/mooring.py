"""Installation strategies for mooring systems."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import process

from wisdem.orbit.core import Cargo, Vessel
from wisdem.orbit.core.logic import position_onsite, get_list_of_items_from_port
from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.core.exceptions import ItemNotFound


class MooringSystemInstallation(InstallPhase):
    """Module to model the installation of mooring systems at sea."""

    phase = "Mooring System Installation"
    capex_category = "Mooring System"

    #:
    expected_config = {
        "mooring_install_vessel": "dict | str",
        "site": {"depth": "m", "distance": "km"},
        "plant": {"num_turbines": "int"},
        "mooring_system": {
            "num_lines": "int",
            "line_mass": "t",
            "line_cost": "USD",
            "anchor_mass": "t",
            "anchor_cost": "USD",
            "anchor_type": "str (optional, default: 'Suction Pile')",
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of `MooringSystemInstallation`.

        Parameters
        ----------
        config : dict
            Simulation specific configuration.
        weather : np.array
            Weather data at site.
        """

        super().__init__(weather, **kwargs)

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self.setup_simulation(**kwargs)

    def setup_simulation(self, **kwargs):
        """
        Sets up the required simulation infrastructure:
            - initializes port
            - initializes installation vessel
            - initializes mooring systems at port.
        """

        self.initialize_port()
        self.initialize_installation_vessel()
        self.initialize_components()

        depth = self.config["site"]["depth"]
        distance = self.config["site"]["distance"]

        self.num_lines = self.config["mooring_system"]["num_lines"]
        self.line_cost = self.config["mooring_system"]["line_cost"]
        self.anchor_cost = self.config["mooring_system"]["anchor_cost"]

        install_mooring_systems(self.vessel, self.port, distance, depth, self.num_systems, **kwargs)

    @property
    def system_capex(self):
        """Returns total procurement cost of all mooring systems."""

        return self.num_systems * self.num_lines * (self.line_cost + self.anchor_cost)

    def initialize_installation_vessel(self):
        """Initializes the mooring system installation vessel."""

        vessel_specs = self.config.get("mooring_install_vessel", None)
        name = vessel_specs.get("name", "Mooring System Installation Vessel")

        vessel = self.initialize_vessel(name, vessel_specs)
        self.env.register(vessel)

        vessel.initialize()
        vessel.at_port = True
        vessel.at_site = False
        self.vessel = vessel

    def initialize_components(self):
        """Initializes the Cargo components at port."""

        system = MooringSystem(**self.config["mooring_system"])
        self.num_systems = self.config["plant"]["num_turbines"]

        for _ in range(self.num_systems):
            self.port.put(system)

    @property
    def detailed_output(self):
        """Detailed outputs of the scour protection installation."""

        outputs = {self.phase: {**self.agent_efficiencies}}

        return outputs


@process
def install_mooring_systems(vessel, port, distance, depth, systems, **kwargs):
    """
    Logic for the Mooring System Installation Vessel.

    Parameters
    ----------
    vessel : Vessel
        Mooring System Installation Vessel
    port : Port
    distance : int | float
        Distance between port and site (km).
    systems : int
        Total systems to install.
    """

    n = 0
    while n < systems:
        if vessel.at_port:
            try:
                # Get mooring systems from port.
                yield get_list_of_items_from_port(vessel, port, ["MooringSystem"], **kwargs)

            except ItemNotFound:
                # If no items are at port and vessel.storage.items is empty,
                # the job is done
                if not vessel.storage.items:
                    vessel.submit_debug_log(message="Item not found. Shutting down.")
                    break

            # Transit to site
            vessel.update_trip_data()
            vessel.at_port = False
            yield vessel.transit(distance)
            vessel.at_site = True

        if vessel.at_site:

            if vessel.storage.items:

                system = yield vessel.get_item_from_storage("MooringSystem", **kwargs)
                for _ in range(system.num_lines):
                    yield position_onsite(vessel, **kwargs)
                    yield perform_mooring_site_survey(vessel, **kwargs)
                    yield install_mooring_anchor(vessel, depth, system.anchor_type, **kwargs)
                    yield install_mooring_line(vessel, depth, **kwargs)

                n += 1

            else:
                # Transit to port
                vessel.at_site = False
                yield vessel.transit(distance)
                vessel.at_port = True

    vessel.submit_debug_log(message="Mooring systems installation complete!")


@process
def perform_mooring_site_survey(vessel, **kwargs):
    """
    Calculates time required to perform a mooring system survey.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.

    Yields
    ------
    vessel.task representing time to "Perform Mooring Site Survey".
    """

    key = "mooring_site_survey_time"
    survey_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Perform Mooring Site Survey",
        survey_time,
        constraints=vessel.transit_limits,
        **kwargs,
    )


@process
def install_mooring_anchor(vessel, depth, _type, **kwargs):
    """
    Calculates time required to install a mooring system anchor.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    depth : int | float
        Depth at site (m).
    _type : str
        Anchor type. 'Suction Pile' or 'Drag Embedment'.

    Yields
    ------
    vessel.task representing time to install mooring anchor.
    """

    if _type == "Suction Pile":
        key = "suction_pile_install_time"
        task = "Install Suction Pile Anchor"
        fixed = kwargs.get(key, pt[key])

    elif _type == "Drag Embedment":
        key = "drag_embed_install_time"
        task = "Install Drag Embedment Anchor"
        fixed = kwargs.get(key, pt[key])

    else:
        raise ValueError(f"Mooring System Anchor Type: {_type} not recognized.")

    install_time = fixed + 0.005 * depth
    yield vessel.task_wrapper(task, install_time, constraints=vessel.transit_limits, **kwargs)


@process
def install_mooring_line(vessel, depth, **kwargs):
    """
    Calculates time required to install a mooring system line.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    depth : int | float
        Depth at site (m).

    Yields
    ------
    vessel.task representing time to install mooring line.
    """

    install_time = 0.005 * depth

    yield vessel.task_wrapper(
        "Install Mooring Line",
        install_time,
        constraints=vessel.transit_limits,
        **kwargs,
    )


class MooringSystem(Cargo):
    """Mooring System Cargo"""

    def __init__(
        self,
        num_lines=None,
        line_mass=None,
        anchor_mass=None,
        anchor_type="Suction Pile",
        **kwargs,
    ):
        """Creates an instance of MooringSystem"""

        self.num_lines = num_lines
        self.line_mass = line_mass
        self.anchor_mass = anchor_mass
        self.anchor_type = anchor_type

        self.deck_space = 0

    @property
    def mass(self):
        """Returns total system mass in t."""

        return self.num_lines * (self.line_mass + self.anchor_mass)

    @staticmethod
    def fasten(**kwargs):
        """Dummy method to work with `get_list_of_items_from_port`."""

        key = "mooring_system_load_time"
        time = kwargs.get(key, pt[key])

        return "Load Mooring System", time

    @staticmethod
    def release(**kwargs):
        """Dummy method to work with `get_list_of_items_from_port`."""

        return "", 0

    def anchor_install_time(self, depth):
        """
        Returns time to install anchor. Varies by depth.

        Parameters
        ----------
        depth : int | float
            Depth at site (m).
        """

        if self.anchor_type == "Suction Pile":
            fixed = 11

        elif self.anchor_type == "Drag Embedment":
            fixed = 5

        else:
            raise ValueError(f"Mooring System Anchor Type: {self.anchor_type} not recognized.")

        return fixed + 0.005 * depth
