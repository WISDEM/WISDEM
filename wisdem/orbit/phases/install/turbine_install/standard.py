"""`TurbineInstallation` class and related processes."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy
from math import ceil

import numpy as np
import simpy
from marmot import process

from wisdem.orbit.core import Vessel
from wisdem.orbit.core.logic import (
    jackdown_if_required,
    shuttle_items_to_queue,
    prep_for_site_operations,
    get_list_of_items_from_port,
)
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.core.exceptions import ItemNotFound

from .common import Blade, Nacelle, TowerSection, install_nacelle, install_tower_section, install_turbine_blade


class TurbineInstallation(InstallPhase):
    """
    Standard turbine installation module using a Wind Turbine Installation
    Vessel (WTIV). If input `feeder` and `num_feeders` are not supplied, the
    WTIV will perform all transport and installation tasks. If the above inputs
    are defined, feeder barges will transport turbine components from port to
    site.
    """

    phase = "Turbine Installation"
    capex_category = "Turbine"

    #:
    expected_config = {
        "wtiv": "dict | str",
        "feeder": "dict | str (optional)",
        "num_feeders": "int (optional)",
        "site": {"depth": "m", "distance": "km"},
        "plant": {"num_turbines": "int"},
        "port": {
            "num_cranes": "int (optional, default: 1)",
            "monthly_rate": "USD/mo (optional)",
            "name": "str (optional)",
        },
        "turbine": {
            "hub_height": "m",
            "tower": {
                "deck_space": "m2",
                "mass": "t",
                "length": "m",
                "sections": "int (optional)",
            },
            "nacelle": {"deck_space": "m2", "mass": "t"},
            "blade": {"deck_space": "m2", "mass": "t"},
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of TurbineInstallation.

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
        self.initialize_wtiv()
        self.initialize_turbines()
        self.setup_simulation(**kwargs)

    @property
    def system_capex(self):
        """Returns 0 as turbine capex is handled at in ProjectManager."""

        return 0

    def setup_simulation(self, **kwargs):
        """
        Sets up simulation infrastructure, routing to specific methods dependent
        on number of feeders.
        """

        if self.config.get("num_feeders", None):
            self.initialize_feeders()
            self.initialize_queue()
            self.setup_simulation_with_feeders(**kwargs)

        else:
            self.feeders = None
            self.setup_simulation_without_feeders(**kwargs)

    def setup_simulation_without_feeders(self, **kwargs):
        """
        Sets up infrastructure for turbine installation without feeder barges.
        """

        site_distance = self.config["site"]["distance"]
        site_depth = self.config["site"]["depth"]
        hub_height = self.config["turbine"]["hub_height"]

        solo_install_turbines(
            self.wtiv,
            port=self.port,
            distance=site_distance,
            turbines=self.num_turbines,
            tower_sections=self.num_sections,
            num_blades=3,
            site_depth=site_depth,
            hub_height=hub_height,
            **kwargs,
        )

    def setup_simulation_with_feeders(self, **kwargs):
        """
        Sets up infrastructure for turbine installation using feeder barges.
        """

        site_distance = self.config["site"]["distance"]
        site_depth = self.config["site"]["depth"]
        hub_height = self.config["turbine"]["hub_height"]

        install_turbine_components_from_queue(
            self.wtiv,
            queue=self.active_feeder,
            distance=site_distance,
            turbines=self.num_turbines,
            tower_sections=self.num_sections,
            num_blades=3,
            site_depth=site_depth,
            hub_height=hub_height,
            **kwargs,
        )

        for feeder in self.feeders:
            shuttle_items_to_queue(
                feeder,
                port=self.port,
                queue=self.active_feeder,
                distance=site_distance,
                items=self.component_list,
                **kwargs,
            )

    def initialize_wtiv(self):
        """
        Initializes the WTIV simulation object and the onboard vessel storage.
        """

        wtiv_specs = self.config.get("wtiv", None)
        name = wtiv_specs.get("name", "WTIV")

        wtiv = self.initialize_vessel(name, wtiv_specs)
        self.env.register(wtiv)

        wtiv.initialize()
        wtiv.at_port = True
        wtiv.at_site = False
        self.wtiv = wtiv

    def initialize_feeders(self):
        """
        Initializes feeder barge objects.
        """

        number = self.config.get("num_feeders", None)
        feeder_specs = self.config.get("feeder", None)

        self.feeders = []
        for n in range(number):
            # TODO: Add in option for named feeders.
            name = "Feeder {}".format(n)

            feeder = self.initialize_vessel(name, feeder_specs)
            self.env.register(feeder)

            feeder.initialize()
            feeder.at_port = True
            feeder.at_site = False
            self.feeders.append(feeder)

    def initialize_turbines(self):
        """
        Initializes turbine components at port.
        """

        tower = deepcopy(self.config["turbine"]["tower"])
        self.num_sections = tower.get("sections", 1)

        _section = {}
        for k in ["length", "deck_space", "mass"]:
            try:
                _section[k] = ceil(tower.get(k) / self.num_sections)

            except TypeError:
                pass

        section = TowerSection(**_section)
        nacelle = Nacelle(**self.config["turbine"]["nacelle"])
        blade = Blade(**self.config["turbine"]["blade"])

        component_list = [
            *np.repeat(section, self.num_sections),
            nacelle,
            # TODO: Add in configuration for number of blades.
            *np.repeat(blade, 3),
        ]

        self.num_turbines = self.config["plant"]["num_turbines"]

        for _ in range(self.num_turbines):
            for item in component_list:
                self.port.put(item)

        self.component_list = [a.type for a in component_list]

    def initialize_queue(self):
        """
        Initializes the queue, modeled as a ``SimPy.Resource`` that feeders
        join at site. This limits the simulation to one active feeder at a time.
        """

        self.active_feeder = simpy.Resource(self.env, capacity=1)
        self.active_feeder.vessel = None
        self.active_feeder.activate = self.env.event()

    @property
    def detailed_output(self):
        """Returns detailed outputs of the turbine installation."""

        if self.feeders:
            transport_vessels = [*self.feeders]

        else:
            transport_vessels = [self.wtiv]

        outputs = {
            self.phase: {
                **self.agent_efficiencies,
                **self.get_max_cargo_mass_utilzations(transport_vessels),
                **self.get_max_deck_space_utilzations(transport_vessels),
            }
        }

        return outputs


@process
def solo_install_turbines(vessel, port, distance, turbines, tower_sections, num_blades, **kwargs):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses during a single
    turbine installation process.

    Parameters
    ----------
    vessel : vessels.Vessel
        Vessel object that represents the WTIV.
    distance : int | float
        Distance between port and site (km).
    component_list : dict
        Turbine components to retrieve and install.
    number : int
        Total turbine component sets to install.
    """

    reequip_time = vessel.crane.reequip(**kwargs)

    component_list = [
        *np.repeat("TowerSection", tower_sections),
        "Nacelle",
        *np.repeat("Blade", num_blades),
    ]

    n = 0
    while n < turbines:
        if vessel.at_port:
            try:
                # Get turbine components
                yield get_list_of_items_from_port(vessel, port, component_list, **kwargs)

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
                yield prep_for_site_operations(vessel, **kwargs)

                for i in range(tower_sections):
                    # Get tower section
                    section = yield vessel.get_item_from_storage("TowerSection", **kwargs)

                    # Install tower section
                    height = section.length * (i + 1)
                    yield install_tower_section(vessel, section, height, **kwargs)

                # Get turbine nacelle
                nacelle = yield vessel.get_item_from_storage("Nacelle", **kwargs)

                # Install nacelle
                yield vessel.task_wrapper("Reequip", reequip_time, constraints=vessel.transit_limits)
                yield install_nacelle(vessel, nacelle, **kwargs)

                # Install turbine blades
                yield vessel.task_wrapper("Reequip", reequip_time, constraints=vessel.transit_limits)
                for _ in range(num_blades):
                    blade = yield vessel.get_item_from_storage("Blade", **kwargs)

                    yield install_turbine_blade(vessel, blade, **kwargs)

                yield jackdown_if_required(vessel, **kwargs)
                vessel.submit_debug_log(progress="Turbine")
                n += 1

            else:
                # Transit to port
                vessel.at_site = False
                yield vessel.transit(distance)
                vessel.at_port = True

    vessel.submit_debug_log(message="Turbine installation complete!")


@process
def install_turbine_components_from_queue(wtiv, queue, distance, turbines, tower_sections, num_blades, **kwargs):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses to install
    turbine componenets from a queue of feeder barges.

    Parameters
    ----------
    env : simulation.Environment
        SimPy environment that the simulation runs in.
    wtiv : vessels.Vessel
        Vessel object that represents the WTIV.
    queue : simpy.Resource
        Queue object to interact with active feeder barge.
    component_list : dict
        Turbine components to retrieve and install.
    number : int
        Total turbine component sets to install.
    distance : int | float
        Distance from site to port (km).
    """

    reequip_time = wtiv.crane.reequip(**kwargs)

    n = 0
    while n < turbines:
        if wtiv.at_port:
            # Transit to site
            wtiv.at_port = False
            yield wtiv.transit(distance)
            wtiv.at_site = True

        if wtiv.at_site:

            if queue.vessel:

                # Prep for turbine install
                yield prep_for_site_operations(wtiv, **kwargs)

                for i in range(tower_sections):
                    # Get tower section
                    section = yield wtiv.get_item_from_storage("TowerSection", vessel=queue.vessel, **kwargs)

                    # Install tower section
                    height = section.length * (i + 1)
                    yield install_tower_section(wtiv, section, height, **kwargs)

                # Get turbine nacelle
                nacelle = yield wtiv.get_item_from_storage("Nacelle", vessel=queue.vessel, **kwargs)

                # Install nacelle
                yield wtiv.task_wrapper("Reequip", reequip_time, constraints=wtiv.transit_limits)
                yield install_nacelle(wtiv, nacelle, **kwargs)

                # Install turbine blades
                yield wtiv.task_wrapper("Reequip", reequip_time, constraints=wtiv.transit_limits)

                for i in range(num_blades):
                    release = True if i + 1 == num_blades else False

                    blade = yield wtiv.get_item_from_storage("Blade", vessel=queue.vessel, release=release, **kwargs)

                    yield install_turbine_blade(wtiv, blade, **kwargs)

                yield jackdown_if_required(wtiv, **kwargs)
                wtiv.submit_debug_log(progress="Turbine")
                n += 1

            else:
                start = wtiv.env.now
                yield queue.activate
                delay_time = wtiv.env.now - start
                wtiv.submit_action_log("Delay", delay_time, location="Site")

    # Transit to port
    wtiv.at_site = False
    yield wtiv.transit(distance)
    wtiv.at_port = True

    wtiv.submit_debug_log(message="Turbine installation complete!")
