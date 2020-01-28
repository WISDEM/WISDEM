"""Provides the `MonopileInstallation` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy

from wisdem.orbit.vessels import Vessel
from wisdem.orbit.simulation import Environment, VesselStorage
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.simulation.logic import shuttle_items_to_queue
from wisdem.orbit.phases.install.monopile_install._single_wtiv import (
    solo_install_monopiles,
)
from wisdem.orbit.phases.install.monopile_install._wtiv_with_feeders import (
    install_monopiles_from_queue,
)


class MonopileInstallation(InstallPhase):
    """
    Standard monopile installation module using a Wind Turbine Installation
    Vessel (WTIV). If input `feeder` and `num_feeders` are not supplied, the
    WTIV will perform all transport and installation tasks. If the above inputs
    are defined, feeder barges will transport monopile components from port to
    site.
    """

    phase = "Monopile Installation"

    #:
    expected_config = {
        "wtiv": "dict | str",
        "feeder": "dict | str (optional)",
        "num_feeders": "int (optional)",
        "site": {"depth": "float", "distance": "float"},
        "plant": {"num_turbines": "int"},
        "turbine": {"hub_height": "float"},
        "port": {
            "num_cranes": "int",
            "monthly_rate": "float (optional)",
            "name": "str (optional)",
        },
        "monopile": {
            "type": "Monopile",
            "length": "float",
            "diameter": "float",
            "deck_space": "float",
            "weight": "float",
        },
        "transition_piece": {
            "type": "Transition Piece",
            "deck_space": "float",
            "weight": "float",
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of MonopileInstallation.

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

        self.num_monopiles = self.config["plant"]["num_turbines"]
        self.initialize_monopiles()

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
            solo_install_monopiles(
                env=self.env,
                vessel=self.wtiv,
                port=self.port,
                distance=site_distance,
                number=self.num_monopiles,
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

        self.env.process(
            install_monopiles_from_queue(
                env=self.env,
                wtiv=self.wtiv,
                queue=self.active_feeder,
                site_depth=site_depth,
                distance=site_distance,
                number=self.num_monopiles,
                **kwargs,
            )
        )

        component_list = [("type", "Monopile"), ("type", "Transition Piece")]

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

    def initialize_monopiles(self):
        """
        Initializes monopile and transition piece objects at port.
        """

        monopile = self.config.get("monopile", None)
        transition_piece = self.config.get("transition_piece", None)

        for _ in range(self.num_monopiles):
            self.port.put(monopile)
            self.port.put(transition_piece)

        self.logger.debug(
            "SUBSTRUCTURES INITIALIZED",
            extra={"time": self.env.now, "agent": "Director"},
        )

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
