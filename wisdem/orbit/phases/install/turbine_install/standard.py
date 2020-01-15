"""Provides the `TurbineInstallation` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import numpy as np
import simpy

from wisdem.orbit.vessels import Vessel
from wisdem.orbit.simulation import Environment, VesselStorage
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.simulation.logic import shuttle_items_to_queue
from wisdem.orbit.phases.install.turbine_install._single_wtiv import (
    solo_install_turbines,
)
from wisdem.orbit.phases.install.turbine_install._wtiv_with_feeders import (
    install_turbine_components_from_queue,
)


class TurbineInstallation(InstallPhase):
    """
    Standard turbine installation module using a Wind Turbine Installation
    Vessel (WTIV). If input `feeder` and `num_feeders` are not supplied, the
    WTIV will perform all transport and installation tasks. If the above inputs
    are defined, feeder barges will transport turbine components from port to
    site.
    """

    phase = "Turbine Installation"

    #:
    expected_config = {
        "wtiv": "dict | str",
        "feeder": "dict | str (optional)",
        "num_feeders": "int (optional)",
        "site": {"depth": "float", "distance": "float"},
        "plant": {"num_turbines": "int"},
        "port": {
            "num_cranes": "int",
            "monthly_rate": "float (optional)",
            "name": "str (optional)",
        },
        "turbine": {
            "hub_height": "float",
            "tower": {
                "type": "Tower",
                "deck_space": "float",
                "weight": "float",
            },
            "nacelle": {
                "type": "Nacelle",
                "deck_space": "float",
                "weight": "float",
            },
            "blade": {
                "type": "Blade",
                "deck_space": "float",
                "weight": "float",
            },
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of TurbineInstallation.

        Parameters
        ----------
        config : dict
            Simulation specific configuration.
        weather : pd.DataFrame (optional)
            Weather profile at site.
            Expects columns 'max_waveheight' and 'max_windspeed'.
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self.env = Environment(weather)
        self.init_logger(**kwargs)
        self.extract_phase_kwargs(**kwargs)
        self.extract_defaults()

        self.initialize_port()
        self.initialize_wtiv()

        self.num_turbines = self.config["plant"]["num_turbines"]
        self.initialize_turbines()

        self.setup_simulation(**kwargs)

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

        self.env.process(
            solo_install_turbines(
                env=self.env,
                vessel=self.wtiv,
                port=self.port,
                distance=site_distance,
                number=self.num_turbines,
                site_depth=site_depth,
                hub_height=hub_height,
                **kwargs,
            )
        )

    def setup_simulation_with_feeders(self, **kwargs):
        """
        Sets up infrastructure for turbine installation using feeder barges.
        """

        site_distance = self.config["site"]["distance"]
        site_depth = self.config["site"]["depth"]
        hub_height = self.config["turbine"]["hub_height"]

        self.env.process(
            install_turbine_components_from_queue(
                env=self.env,
                wtiv=self.wtiv,
                queue=self.active_feeder,
                site_depth=site_depth,
                distance=site_distance,
                number=self.num_turbines,
                hub_height=hub_height,
                **kwargs,
            )
        )

        component_list = [
            ("type", "Tower"),
            ("type", "Nacelle"),
            ("type", "Blade"),
            ("type", "Blade"),
            ("type", "Blade"),
        ]

        for feeder in self.feeders:
            self.env.process(
                shuttle_items_to_queue(
                    env=self.env,
                    vessel=feeder,
                    port=self.port,
                    queue=self.active_feeder,
                    distance=site_distance,
                    items=component_list,
                    **kwargs,
                )
            )

    def initialize_wtiv(self):
        """
        Initializes the WTIV simulation object and the onboard vessel storage.
        """

        wtiv_specs = self.config.get("wtiv", None)

        if wtiv_specs is None:
            raise Exception("WTIV is not defined.")

        name = wtiv_specs.get("name", "WTIV")
        cost = wtiv_specs["vessel_specs"].get(
            "day_rate", self.defaults["wtiv_day_rate"]
        )

        self.wtiv = Vessel(name, wtiv_specs)

        _storage_specs = wtiv_specs.get("storage_specs", None)
        if _storage_specs is None:
            raise Exception("Storage specifications must be set for WTIV.")

        self.wtiv.storage = VesselStorage(self.env, **_storage_specs)

        self.wtiv.at_port = True
        self.wtiv.at_site = False

        self.agent_costs[name] = cost

    def initialize_feeders(self):
        """
        Initializes feeder barge objects.
        """

        number = self.config.get("num_feeders", None)
        feeder_specs = self.config.get("feeder", None)

        if feeder_specs is None:
            raise Exception("Feeder Barge is not defined.")

        cost = feeder_specs["vessel_specs"].get(
            "day_rate", self.defaults["feeder_day_rate"]
        )

        _storage_specs = feeder_specs.get("storage_specs", None)
        if _storage_specs is None:
            raise Exception(
                "Storage specifications must be set in feeder_specs."
            )

        self.feeders = []
        for n in range(number):
            name = "Feeder {}".format(n)
            feeder = Vessel(name, feeder_specs)
            feeder.storage = VesselStorage(self.env, **_storage_specs)

            feeder.at_port = True
            feeder.at_site = False

            self.feeders.append(feeder)

            self.agent_costs[name] = cost

    def initialize_turbines(self):
        """
        Initializes turbine components at port.
        """

        component_list = [
            self.config["turbine"]["tower"],
            self.config["turbine"]["nacelle"],
            *np.repeat(self.config["turbine"]["blade"], 3),
        ]

        for _ in range(self.num_turbines):
            for item in component_list:
                self.port.put(item)

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
        """
        Returns detailed outputs in a dictionary, including:

        - Agent operational efficiencies, ``operations time / total time``
        - Cargo weight efficiencies, ``highest weight used / maximum weight``
        - Deck space efficiencies, ``highest space used / maximum space``
        """

        if self.feeders:
            transport_vessels = [*self.feeders]

        else:
            transport_vessels = [self.wtiv]

        outputs = {
            **self.agent_efficiencies,
            **self.get_max_cargo_weight_utilzations(transport_vessels),
            **self.get_max_deck_space_utilzations(transport_vessels),
        }

        return outputs
