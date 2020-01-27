"""Provides the `OffshoreSubstationInstallation` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy

from wisdem.orbit.vessels import Vessel, tasks
from wisdem.orbit.simulation import Environment, VesselStorage
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.simulation.logic import (
    get_item_from_storage,
    shuttle_items_to_queue,
    prep_for_site_operations,
)
from wisdem.orbit.phases.install.monopile_install._common import install_monopile


class OffshoreSubstationInstallation(InstallPhase):
    """
    Offshore Substation (OSS) installation process using a single heavy lift
    vessel and feeder barge.
    """

    #:
    expected_config = {
        "num_substations": "int",
        "oss_install_vessel": "dict | str",
        "num_feeders": "int",
        "feeder": "dict | str",
        "site": {"distance": "float", "depth": "int"},
        "port": {
            "num_cranes": "int",
            "monthly_rate": "float (optional)",
            "name": "str (optional)",
        },
        "offshore_substation_topside": {
            "type": "Topside",
            "deck_space": "float",
            "weight": "float",
        },
        "offshore_substation_substructure": {
            "type": "Monopile",
            "deck_space": "float",
            "weight": "float",
            "length": "float",
        },
    }

    phase = "Offshore Substation Installation"

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of OffshoreSubstationInstallation.

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
        self.extract_phase_kwargs(**kwargs)
        self.extract_defaults()

        self.env = Environment(weather)
        self.init_logger(**kwargs)
        self.setup_simulation(**kwargs)

    def setup_simulation(self, **kwargs):
        """
        Initializes required objects for simulation.
        - Creates port + crane
        - Creates monopile and topside
        - Creates heavy lift vessel and feeder
        """

        self.initialize_port()
        self.initialize_oss()
        self.initialize_oss_install_vessel()
        self.initialize_feeders()
        self.initialize_queue()

        site_distance = self.config["site"]["distance"]
        site_depth = self.config["site"]["depth"]
        num_subsations = self.config["num_substations"]

        self.env.process(
            install_oss_from_queue(
                env=self.env,
                oss_vessel=self.oss_vessel,
                queue=self.active_feeder,
                site_depth=site_depth,
                distance=site_distance,
                number=num_subsations,
                **kwargs,
            )
        )

        component_list = [("type", "Monopile"), ("type", "Topside")]

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

    def initialize_oss(self):
        """
        Creates offshore substation objects at port.
        """

        oss_topside = self.config["offshore_substation_topside"]
        oss_substructure = self.config["offshore_substation_substructure"]
        self.num_substations = self.config["num_substations"]

        for _ in range(self.num_substations):
            self.port.put(oss_topside)
            self.port.put(oss_substructure)

        self.logger.debug(
            "OFFSHORE SUBSTATIONS INITIALIZED",
            extra={"time": self.env.now, "agent": "Director"},
        )

    def initialize_oss_install_vessel(self):
        """
        Creates the offshore substation installation vessel object.
        """

        oss_vessel_specs = self.config.get("oss_install_vessel", None)
        if oss_vessel_specs is None:
            raise Exception("OSS installation vessel is not defined.")

        name = oss_vessel_specs.get("name", "Heavy Lift Vessel")
        cost = oss_vessel_specs["vessel_specs"].get(
            "day_rate", self.defaults["heavy_lift_vessel_day_rate"]
        )

        self.oss_vessel = Vessel(name, oss_vessel_specs)

        _storage_specs = oss_vessel_specs.get("storage_specs", None)
        if _storage_specs is None:
            raise Exception(
                "Storage specifications must be set for OSS Vessel."
            )

        self.oss_vessel.storage = VesselStorage(self.env, **_storage_specs)

        self.oss_vessel.at_port = True
        self.oss_vessel.at_site = False

        self.agent_costs[name] = cost

    def initialize_feeders(self):
        """
        Creates the feeder vessel object.
        """

        number = self.config.get("num_feeders", None)
        feeder_specs = self.config.get("feeder", None)
        if feeder_specs is None:
            raise Exception("Feeder vessel is not defined.")

        name = feeder_specs.get("name", "OSS Feeder")
        cost = feeder_specs["vessel_specs"].get(
            "day_rate", self.defaults["feeder_day_rate"]
        )

        _storage_specs = feeder_specs.get("storage_specs", None)
        if _storage_specs is None:
            raise Exception("Storage specifications must be set for Feeder.")

        self.feeders = []
        for n in range(number):
            name = "Feeder {}".format(n)
            feeder = Vessel(name, feeder_specs)
            feeder.storage = VesselStorage(self.env, **_storage_specs)

            feeder.at_port = True
            feeder.at_site = False

            self.feeders.append(feeder)

            self.agent_costs[name] = cost

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
        """
        Returns detailed outputs in a dictionary.
        """

        outputs = {
            **self.agent_efficiencies,
            **self.get_max_cargo_weight_utilzations([*self.feeders]),
            **self.get_max_deck_space_utilzations([*self.feeders]),
        }

        return outputs


def install_oss_from_queue(env, oss_vessel, queue, number, distance, **kwargs):
    """
    Installs offshore subsations and substructures from queue of feeder barges.

    Parameters
    ----------
    env : Environment
    oss_vessel : Vessel
    queue : simpy.Resource
        Queue object to shuttle to.
    number : int
        Number of substructures to install.
    distance : int | float
        Distance from site to port (km).
    """

    transit_time = oss_vessel.transit_time(distance)

    transit = {
        "agent": oss_vessel.name,
        "location": "At Sea",
        "type": "Operations",
        "action": "Transit",
        "duration": transit_time,
        **oss_vessel.transit_limits,
    }

    n = 0
    while n < number:
        if oss_vessel.at_port:
            # Transit to site
            oss_vessel.at_port = False
            yield env.process(env.task_handler(transit))
            oss_vessel.at_site = True

        if oss_vessel.at_site:

            if queue.vessel:

                # Prep for monopile install
                yield env.process(
                    prep_for_site_operations(
                        env, oss_vessel, survey_required=True, **kwargs
                    )
                )

                # Get monopile
                monopile = yield env.process(
                    get_item_from_storage(
                        env=env,
                        vessel=queue.vessel,
                        item_type="Monopile",
                        action_vessel=oss_vessel,
                        release=False,
                        **kwargs,
                    )
                )

                upend_time = tasks.upend_monopile(
                    oss_vessel, monopile["length"], **kwargs
                )

                upend = {
                    "action": "UpendMonopile",
                    "duration": upend_time,
                    "agent": oss_vessel.name,
                    "location": "Site",
                    "type": "Operations",
                    **oss_vessel.operational_limits,
                }

                yield env.process(env.task_handler(upend))

                # Install monopile
                yield env.process(
                    install_monopile(env, oss_vessel, monopile, **kwargs)
                )

                # Get topside
                topside = yield env.process(
                    get_item_from_storage(
                        env=env,
                        vessel=queue.vessel,
                        item_type="Topside",
                        action_vessel=oss_vessel,
                        release=True,
                        **kwargs,
                    )
                )

                yield env.process(
                    install_topside(env, oss_vessel, topside, **kwargs)
                )

                n += 1

            else:
                start = env.now
                yield queue.activate
                delay_time = env.now - start
                env.logger.info(
                    "",
                    extra={
                        "agent": oss_vessel.name,
                        "time": env.now,
                        "type": "Delay",
                        "action": "WaitForFeeder",
                        "duration": delay_time,
                        "location": "Site",
                    },
                )
    # Transit to port
    oss_vessel.at_site = False
    yield env.process(env.task_handler(transit))
    oss_vessel.at_port = True

    env.logger.debug(
        "Offshore substation installation complete!",
        extra={
            "agent": oss_vessel.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )


def install_topside(env, vessel, topside, **kwargs):
    """
    Substation topside installation process.
    Subprocesses:
    - Crane reequip
    - Lift topside
    - Attach topside to substructure
    - Pump grout
    - Cure grout

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    topsdie : dict
    """

    reequip_time = vessel.crane.reequip()
    lift_time = tasks.lift_topside(vessel, **kwargs)
    attach_time = tasks.attach_topside(vessel, **kwargs)
    grout_pump_time = tasks.pump_transition_piece_grout(**kwargs)
    grout_cure_time = tasks.cure_transition_piece_grout(**kwargs)

    _shared = {
        "agent": vessel.name,
        "type": "Operations",
        "location": "Site",
        **vessel.operational_limits,
    }

    task_list = [
        {"action": "CraneReequip", "duration": reequip_time, **_shared},
        {"action": "LiftTopside", "duration": lift_time, **_shared},
        {"action": "AttachTopside", "duration": attach_time, **_shared},
        {
            "action": "PumpGrout",
            "duration": grout_pump_time,
            **_shared,
            **vessel.transit_limits,
        },
        {
            "action": "CureGrout",
            "duration": grout_cure_time,
            **_shared,
            **vessel.transit_limits,
        },
    ]

    yield env.process(env.task_handler(task_list))
