"""`MonopileInstallation` class and related processes."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy
from marmot import process
from wisdem.orbit.core import Vessel
from wisdem.orbit.core.logic import shuttle_items_to_queue, prep_for_site_operations, get_list_of_items_from_port
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.core.exceptions import ItemNotFound

from .common import Monopile, TransitionPiece, upend_monopile, install_monopile, install_transition_piece


class MonopileInstallation(InstallPhase):
    """
    Standard monopile installation module using a Wind Turbine Installation
    Vessel (WTIV). If input `feeder` and `num_feeders` are not supplied, the
    WTIV will perform all transport and installation tasks. If the above inputs
    are defined, feeder barges will transport monopile components from port to
    site.
    """

    phase = "Monopile Installation"
    capex_category = "Substructure"

    #:
    expected_config = {
        "wtiv": "dict | str",
        "feeder": "dict | str (optional)",
        "num_feeders": "int (optional)",
        "site": {"depth": "m", "distance": "km"},
        "plant": {"num_turbines": "int"},
        "turbine": {"hub_height": "m"},
        "port": {
            "num_cranes": "int (optional, default: 1)",
            "monthly_rate": "USD/mo (optional)",
            "name": "str (optional)",
        },
        "monopile": {
            "length": "m",
            "diameter": "m",
            "deck_space": "m2",
            "mass": "t",
            "unit_cost": "USD",
        },
        "transition_piece": {
            "deck_space": "m2",
            "mass": "t",
            "unit_cost": "USD",
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
            Expects columns 'waveheight' and 'windspeed'.
        """

        super().__init__(weather, **kwargs)

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)

        self.initialize_port()
        self.initialize_wtiv()
        self.initialize_monopiles()
        self.setup_simulation(**kwargs)

    @property
    def system_capex(self):
        """Returns procurement cost of the substructures."""

        return (self.config["monopile"]["unit_cost"] + self.config["transition_piece"]["unit_cost"]) * self.config[
            "plant"
        ]["num_turbines"]

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

        solo_install_monopiles(
            self.wtiv,
            port=self.port,
            distance=site_distance,
            monopiles=self.num_monopiles,
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
        component_list = ["Monopile", "TransitionPiece"]

        install_monopiles_from_queue(
            self.wtiv,
            queue=self.active_feeder,
            monopiles=self.num_monopiles,
            distance=site_distance,
            site_depth=site_depth,
            **kwargs,
        )

        for feeder in self.feeders:
            shuttle_items_to_queue(
                feeder,
                port=self.port,
                queue=self.active_feeder,
                distance=site_distance,
                items=component_list,
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
            name = "Feeder {}".format(n)

            feeder = self.initialize_vessel(name, feeder_specs)
            self.env.register(feeder)

            feeder.initialize()
            feeder.at_port = True
            feeder.at_site = False
            self.feeders.append(feeder)

    def initialize_monopiles(self):
        """
        Initializes monopile and transition piece objects at port.
        """

        monopile = Monopile(**self.config["monopile"])
        tp = TransitionPiece(**self.config["transition_piece"])
        self.num_monopiles = self.config["plant"]["num_turbines"]

        for _ in range(self.num_monopiles):
            self.port.put(monopile)
            self.port.put(tp)

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
        """Returns detailed outputs of the monopile installation."""

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
def solo_install_monopiles(vessel, port, distance, monopiles, **kwargs):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses during a single
    monopile installation process.

    Parameters
    ----------
    vessel : vessels.Vessel
        Vessel object that represents the WTIV.
    port : Port
    distance : int | float
        Distance between port and site (km).
    monopiles : int
        Total monopiles to install.
    """

    component_list = ["Monopile", "TransitionPiece"]

    n = 0
    while n < monopiles:
        if vessel.at_port:
            try:
                # Get substructure + transition piece from port
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
                # Prep for monopile install
                yield prep_for_site_operations(vessel, survey_required=True, **kwargs)

                # Get monopile from internal storage
                monopile = yield vessel.get_item_from_storage("Monopile", **kwargs)

                yield upend_monopile(vessel, monopile.length, **kwargs)
                yield install_monopile(vessel, monopile, **kwargs)

                # Get transition piece from internal storage
                tp = yield vessel.get_item_from_storage("TransitionPiece", **kwargs)

                yield install_transition_piece(vessel, tp, **kwargs)
                vessel.submit_debug_log(progress="Substructure")
                n += 1

            else:
                # Transit to port
                vessel.at_site = False
                yield vessel.transit(distance)
                vessel.at_port = True

    vessel.submit_debug_log(message="Monopile installation complete!")


@process
def install_monopiles_from_queue(wtiv, queue, monopiles, distance, **kwargs):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses to install
    monopiles and transition pieces from queue of feeder barges.

    Parameters
    ----------
    env : simulation.Environment
        SimPy environment that the simulation runs in.
    wtiv : vessels.Vessel
        Vessel object that represents the WTIV.
    queue : simpy.Resource
        Queue object to interact with active feeder barge.
    number : int
        Total monopiles to install.
    distance : int | float
        Distance from site to port (km).
    """

    n = 0
    while n < monopiles:
        if wtiv.at_port:
            # Transit to site
            wtiv.at_port = False
            yield wtiv.transit(distance)
            wtiv.at_site = True

        if wtiv.at_site:

            if queue.vessel:

                # Prep for monopile install
                yield prep_for_site_operations(wtiv, survey_required=True, **kwargs)

                # Get monopile
                monopile = yield wtiv.get_item_from_storage("Monopile", vessel=queue.vessel, **kwargs)

                yield upend_monopile(wtiv, monopile.length, **kwargs)
                yield install_monopile(wtiv, monopile, **kwargs)

                # Get transition piece from active feeder
                tp = yield wtiv.get_item_from_storage(
                    "TransitionPiece",
                    vessel=queue.vessel,
                    release=True,
                    **kwargs,
                )

                # Install transition piece
                yield install_transition_piece(wtiv, tp, **kwargs)
                wtiv.submit_debug_log(progress="Substructure")
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

    wtiv.submit_debug_log(message="Monopile installation complete!")
