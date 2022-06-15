"""`OffshoreSubstationInstallation` and related processes."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy
from marmot import process

from wisdem.orbit.core import Vessel
from wisdem.orbit.core.logic import shuttle_items_to_queue, prep_for_site_operations
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.phases.install.monopile_install.common import Monopile, upend_monopile, install_monopile

from .common import Topside, install_topside


class OffshoreSubstationInstallation(InstallPhase):
    """
    Offshore Substation (OSS) installation process using a single heavy lift
    vessel and feeder barge.
    """

    phase = "Offshore Substation Installation"
    capex_category = "Offshore Substation"

    #:
    expected_config = {
        "num_substations": "int",
        "oss_install_vessel": "dict | str",
        "num_feeders": "int (optional, default: 1)",
        "feeder": "dict | str",
        "site": {"distance": "km", "depth": "m"},
        "port": {
            "num_cranes": "int (optional, default: 1)",
            "monthly_rate": "USD/mo (optional)",
            "name": "str (optional)",
        },
        "offshore_substation_topside": {
            "deck_space": "m2",
            "mass": "t",
            "unit_cost": "USD",
        },
        "offshore_substation_substructure": {
            "type": "Monopile",
            "deck_space": "m2",
            "mass": "t",
            "length": "m",
            "unit_cost": "USD",
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of OffshoreSubstationInstallation.

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

    @property
    def system_capex(self):
        """Returns procurement CapEx of the offshore substations."""

        return self.config["num_substations"] * (
            self.config["offshore_substation_topside"]["unit_cost"]
            + self.config["offshore_substation_substructure"]["unit_cost"]
        )

    def setup_simulation(self, **kwargs):
        """
        Initializes required objects for simulation.
        - Creates port + crane
        - Creates monopile and topside
        - Creates heavy lift vessel and feeder
        """

        self.initialize_topsides_and_substructures()
        self.initialize_oss_install_vessel()
        self.initialize_feeders()
        self.initialize_queue()

        site_distance = self.config["site"]["distance"]
        site_depth = self.config["site"]["depth"]
        num_subsations = self.config["num_substations"]

        install_oss_from_queue(
            self.oss_vessel,
            queue=self.active_feeder,
            site_depth=site_depth,
            distance=site_distance,
            substations=num_subsations,
            **kwargs,
        )

        component_list = ["Monopile", "Topside"]

        for feeder in self.feeders:
            shuttle_items_to_queue(
                feeder,
                port=self.port,
                queue=self.active_feeder,
                distance=site_distance,
                items=component_list,
                **kwargs,
            )

    def initialize_topsides_and_substructures(self):
        """
        Creates offshore substation objects at port.
        """

        top = Topside(**self.config["offshore_substation_topside"])
        sub = Monopile(**self.config["offshore_substation_substructure"])
        self.num_substations = self.config["num_substations"]

        for _ in range(self.num_substations):
            self.port.put(sub)
            self.port.put(top)

    def initialize_oss_install_vessel(self):
        """
        Creates the offshore substation installation vessel object.
        """

        oss_vessel_specs = self.config.get("oss_install_vessel", None)
        name = oss_vessel_specs.get("name", "Heavy Lift Vessel")

        oss_vessel = self.initialize_vessel(name, oss_vessel_specs)
        self.env.register(oss_vessel)

        oss_vessel.initialize()
        oss_vessel.at_port = True
        oss_vessel.at_site = False
        self.oss_vessel = oss_vessel

    def initialize_feeders(self):
        """
        Initializes feeder barge objects.
        """

        number = self.config.get("num_feeders", 1)
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

    def initialize_queue(self):
        """
        Creates the queue that feeders will join at site. Limited to one active
        feeder at a time.
        """

        self.active_feeder = simpy.Resource(self.env, capacity=1)
        self.active_feeder.vessel = None
        self.active_feeder.activate = self.env.event()

    @property
    def detailed_output(self):
        """Returns detailed outputs of the oss installation."""

        if self.feeders:
            transport_vessels = [*self.feeders]

        else:
            transport_vessels = [self.oss_vessel]

        outputs = {
            self.phase: {
                **self.agent_efficiencies,
                **self.get_max_cargo_mass_utilzations(transport_vessels),
                **self.get_max_deck_space_utilzations(transport_vessels),
            }
        }

        return outputs


@process
def install_oss_from_queue(vessel, queue, substations, distance, **kwargs):
    """
    Installs offshore subsations and substructures from queue of feeder barges.

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    queue : simpy.Resource
        Queue object to shuttle to.
    number : int
        Number of substructures to install.
    distance : int | float
        Distance from site to port (km).
    """

    n = 0
    while n < substations:
        if vessel.at_port:
            # Transit to site
            vessel.at_port = False
            yield vessel.transit(distance)
            vessel.at_site = True

        if vessel.at_site:

            if queue.vessel:

                # Prep for monopile install
                yield prep_for_site_operations(vessel, survey_required=True, **kwargs)

                # Get monopile
                monopile = yield vessel.get_item_from_storage("Monopile", vessel=queue.vessel, **kwargs)

                yield upend_monopile(vessel, monopile.length, **kwargs)
                yield install_monopile(vessel, monopile, **kwargs)

                # Get topside
                topside = yield vessel.get_item_from_storage("Topside", vessel=queue.vessel, release=True, **kwargs)
                yield install_topside(vessel, topside, **kwargs)
                n += 1

            else:
                start = vessel.env.now
                yield queue.activate
                delay_time = vessel.env.now - start
                vessel.submit_action_log("Delay", delay_time, location="Site")

    # Transit to port
    vessel.at_site = False
    yield vessel.transit(distance)
    vessel.at_port = True
    vessel.submit_debug_log(
        message="Monopile installation complete!",
        progress="Offshore Substation",
    )
