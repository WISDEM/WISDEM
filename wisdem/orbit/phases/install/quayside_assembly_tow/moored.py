"""Installation strategies for moored floating systems."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from warnings import warn

import simpy
from marmot import le, process

from wisdem.orbit.core import WetStorage
from wisdem.orbit.phases.install import InstallPhase

from .common import TowingGroup, TurbineAssemblyLine, SubstructureAssemblyLine


class MooredSubInstallation(InstallPhase):
    """
    Installation module to model the quayside assembly, tow-out and
    installation at sea of moored substructures.
    """

    phase = "Moored Substructure Installation"
    capex_category = "Substructure"

    #:
    expected_config = {
        "support_vessel": "str, (optional)",
        "ahts_vessel": "str",
        "towing_vessel": "str",
        "towing_vessel_groups": {
            "towing_vessels": "int",
            "station_keeping_vessels": "int (optional)",
            "ahts_vessels": "int (optional, default: 1)",
            "num_groups": "int (optional)",
        },
        "substructure": {
            "takt_time": "int | float (optional, default: 0)",
            "towing_speed": "int | float (optional, default: 6 km/h)",
            "unit_cost": "USD",
        },
        "site": {"depth": "m", "distance": "km"},
        "plant": {"num_turbines": "int"},
        "turbine": "dict",
        "port": {
            "sub_assembly_lines": "int (optional, default: 1)",
            "sub_storage": "int (optional, default: inf)",
            "turbine_assembly_cranes": "int (optional, default: 1)",
            "assembly_storage": "int (optional, default: inf)",
            "monthly_rate": "USD/mo (optional)",
            "name": "str (optional)",
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of MooredSubInstallation.

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

        self.setup_simulation()

    def setup_simulation(self):
        """
        Sets up simulation infrastructure.

        - Initializes substructure production
        - Initializes turbine assembly processes
        - Initializes towing groups
        """

        self.distance = self.config["site"]["distance"]
        self.num_turbines = self.config["plant"]["num_turbines"]

        self.initialize_port()
        self.initialize_substructure_production()
        self.initialize_turbine_assembly()
        self.initialize_queue()
        self.initialize_towing_groups()
        self.initialize_support_vessel()

    @property
    def system_capex(self):
        """Returns total procurement cost of the substructures."""

        return self.num_turbines * self.config["substructure"]["unit_cost"]

    def initialize_substructure_production(self):
        """
        Initializes the production of substructures at port.

        The number of independent assembly lines and production time associated
        with a substructure can be configured with the following parameters:

        - self.config["substructure"]["takt_time"]
        - self.config["port"]["sub_assembly_lines"]
        """

        try:
            storage = self.config["port"]["sub_storage"]

        except KeyError:
            storage = float("inf")

        self.wet_storage = WetStorage(self.env, storage)

        try:
            time = self.config["substructure"]["takt_time"]

        except KeyError:
            time = 0

        try:
            lines = self.config["port"]["sub_assembly_lines"]

        except KeyError:
            lines = 1

        to_assemble = [1] * self.num_turbines

        self.sub_assembly_lines = []
        for i in range(lines):
            a = SubstructureAssemblyLine(
                to_assemble,
                time,
                self.wet_storage,
                i + 1,
            )

            self.env.register(a)
            a.start()
            self.sub_assembly_lines.append(a)

    def initialize_turbine_assembly(self):
        """
        Initializes turbine assembly lines.

        The number of independent lines
        can be configured with the following parameters:

        - self.config["port"]["turb_assembly_lines"]
        """

        try:
            storage = self.config["port"]["assembly_storage"]

        except KeyError:
            storage = float("inf")

        self.assembly_storage = WetStorage(self.env, storage)

        try:
            lines = self.config["port"]["turbine_assembly_cranes"]

        except KeyError:
            lines = 1

        turbine = self.config["turbine"]
        self.turbine_assembly_lines = []
        for i in range(lines):
            a = TurbineAssemblyLine(
                self.wet_storage,
                self.assembly_storage,
                turbine,
                i + 1,
            )

            self.env.register(a)
            a.start()
            self.turbine_assembly_lines.append(a)

    def initialize_towing_groups(self, **kwargs):
        """
        Initializes towing groups to bring completed assemblies to site and
        stabilize the assembly during final installation.
        """

        self.installation_groups = []

        towing_vessel = self.config["towing_vessel"]
        num_groups = self.config["towing_vessel_groups"].get("num_groups", 1)
        num_towing = self.config["towing_vessel_groups"]["towing_vessels"]
        towing_speed = self.config["substructure"].get("towing_speed", 6)

        ahts_vessel = self.config.get("ahts_vessel", None)
        num_ahts = self.config["towing_vessel_groups"].get("ahts_vessels", 1)

        if ahts_vessel is None:
            warn(
                "No ['ahts_vessel'] specified. num_ahts set to 0."
                " ahts_vessel will be required in future releases.\n",
                stacklevel=1,
            )
            num_ahts = 0

        remaining_substructures = [1] * self.num_turbines

        for i in range(num_groups):
            g = TowingGroup(towing_vessel, ahts_vessel, i + 1)
            self.env.register(g)
            g.initialize()
            self.installation_groups.append(g)

            transfer_install_moored_substructures_from_storage(
                g,
                self.assembly_storage,
                self.distance,
                num_towing,
                num_ahts,
                towing_speed,
                remaining_substructures,
                **kwargs,
            )

    def initialize_queue(self):
        """
        Initializes the queue, modeled as a ``SimPy.Resource`` that towing
        groups join at site.
        """

        self.active_group = simpy.Resource(self.env, capacity=1)
        self.active_group.vessel = None
        self.active_group.activate = self.env.event()

    def initialize_support_vessel(self):
        """
        ** DEPRECATED ** The support vessel is deprecated and an AHTS
        vessel will perform the installation with the towing group.

        Initializes Multi-Purpose Support Vessel to perform installation
        processes at site.
        """

        specs = self.config.get("support_vessel", None)

        if specs is not None:
            warn(
                "support_vessel will be deprecated and replaced with"
                " towing_vessels and ahts_vessel in the towing groups.\n",
                DeprecationWarning,
                stacklevel=2,
            )

        # vessel = self.initialize_vessel("Multi-Purpose Support Vessel",
        # specs)

        # self.env.register(vessel)
        # vessel.initialize(mobilize=False)
        # self.support_vessel = vessel

        station_keeping_vessels = self.config["towing_vessel_groups"].get(
            "station_keeping_vessels", None
        )

        if station_keeping_vessels is not None:
            warn(
                "['towing_vessl_groups]['station_keeping_vessels']"
                " will be deprecated and replaced with"
                " ['towing_vessl_groups]['ahts_vessels'].\n",
                DeprecationWarning,
                stacklevel=2,
            )

        # install_moored_substructures(
        #    self.support_vessel,
        #    self.active_group,
        #    self.distance,
        #    self.num_turbines,
        #    station_keeping_vessels,
        #    **kwargs,
        # )

    @property
    def detailed_output(self):
        """Return detailed outputs."""

        return {
            "operational_delays": {
                **{
                    k: self.operational_delay(str(k))
                    for k in self.sub_assembly_lines
                },
                **{
                    k: self.operational_delay(str(k))
                    for k in self.turbine_assembly_lines
                },
                **{
                    k: self.operational_delay(str(k))
                    for k in self.installation_groups
                },
                # self.support_vessel: self.operational_delay(
                #    str(self.support_vessel)
                # ),
            },
        }

    def operational_delay(self, name):
        """Return operational delays."""

        actions = [a for a in self.env.actions if a["agent"] == name]
        delay = sum(a["duration"] for a in actions if "Delay" in a["action"])

        return delay


@process
def transfer_install_moored_substructures_from_storage(
    group,
    feed,
    distance,
    towing_vessels,
    ahts_vessels,
    towing_speed,
    remaining_substructures,
    **kwargs,
):
    """
    Trigger the substructure installtions. Shuts down after
    self.remaining_substructures is empty.
    """

    while True:
        try:
            _ = remaining_substructures.pop(0)
            yield towing_group_actions(
                group,
                feed,
                distance,
                towing_vessels,
                ahts_vessels,
                towing_speed,
                **kwargs,
            )

        except IndexError:
            break


@process
def towing_group_actions(
    group,
    feed,
    distance,
    towing_vessels,
    ahts_vessels,
    towing_speed,
):
    """
    Process logic for the towing vessel group. Assumes there is an
    anchor tug boat with each group.

    Parameters
    ----------
    group : Vessel
        Towing group.
    feed : simpy.Store
        Completed assembly storage.
    distance : int | float
        Distance from port to site.
    towing_vessels : int
        Number of vessels to use for towing to site.
    ahts_vessels : int
        Number of anchor handling tug vessels.
    towing_speed : int | float
        Configured towing speed (km/h).
    """

    towing_time = distance / towing_speed
    transit_time = distance / group.transit_speed

    start = group.env.now
    _ = yield feed.get()
    delay = group.env.now - start

    if delay > 0:
        group.submit_action_log(
            "Delay: No Completed Turbine Assemblies",
            delay,
            num_vessels=towing_vessels,
            num_ahts_vessels=ahts_vessels,
        )

    yield group.group_task(
        "Ballast to Towing Draft",
        6,
        num_vessels=towing_vessels,
        num_ahts_vessels=ahts_vessels,
        constraints={
            "windspeed": le(group.max_windspeed),
            "waveheight": le(group.max_waveheight),
        },
    )

    yield group.group_task(
        "Tow Substructure",
        towing_time,
        num_vessels=towing_vessels,
        num_ahts_vessels=ahts_vessels,
        constraints={
            "windspeed": le(group.max_windspeed),
            "waveheight": le(group.max_waveheight),
        },
    )

    # At Site
    yield group.group_task(
        "Position Substructure",
        2,
        num_vessels=towing_vessels,
        num_ahts_vessels=ahts_vessels,
        constraints={"windspeed": le(15), "waveheight": le(2.5)},
    )

    yield group.group_task(
        "Ballast to Operational Draft",
        6,
        num_vessels=towing_vessels,
        num_ahts_vessels=ahts_vessels,
        constraints={"windspeed": le(15), "waveheight": le(2.5)},
    )

    yield group.group_task(
        "Connect Mooring Lines, Pre-tension and pre-stretch",
        20,
        num_vessels=towing_vessels,
        num_ahts_vessels=ahts_vessels,
        suspendable=True,
        constraints={"windspeed": le(15), "waveheight": le(2.5)},
    )

    yield group.group_task(
        "Check Mooring Lines",
        6,
        num_vessels=towing_vessels,
        num_ahts_vessels=ahts_vessels,
        suspendable=True,
        constraints={"windspeed": le(15), "waveheight": le(2.5)},
    )

    group.submit_debug_log(progress="Substructure")
    group.submit_debug_log(progress="Turbine")

    yield group.group_task(
        "Transit",
        transit_time,
        num_vessels=towing_vessels,
        num_ahts_vessels=ahts_vessels,
        suspendable=True,
        constraints={
            "windspeed": le(group.max_windspeed),
            "waveheight": le(group.max_waveheight),
        },
    )


@process
def install_moored_substructures(
    vessel,
    queue,
    distance,
    substructures,
    station_keeping_vessels,
):
    """
    ** DEPRECATED ** This method is deprecated and is now performed
    in towing_group_action() by the towing group with AHTS vessel.
    Logic that a Multi-Purpose Support Vessel uses at site to complete the
    installation of moored substructures.

    Parameters
    ----------
    vessel : Vessel
    queue :
    distance : int | float
        Distance between port and site (km).
    substructures : int
        Number of substructures to install before transiting back to port.
    station_keeping_vessels : int
        Number of vessels to use for substructure station keeping during final
        installation at site.
    """

    warn(
        "** DEPRECATED ** This method is deprecated and is now performed"
        " in towing_group_action() by the towing group with AHTS vessel.\n",
        stacklevel=1,
    )

    n = 0
    while n < substructures:
        if queue.vessel:

            start = vessel.env.now
            if n == 0:
                vessel.mobilize()
                yield vessel.transit(distance)

            yield vessel.task_wrapper(
                "Position Substructure",
                2,
                constraints={"windspeed": le(15), "waveheight": le(2.5)},
            )
            yield vessel.task_wrapper(
                "Ballast to Operational Draft",
                6,
                constraints={"windspeed": le(15), "waveheight": le(2.5)},
            )
            yield vessel.task_wrapper(
                "Connect Mooring Lines",
                22,
                suspendable=True,
                constraints={"windspeed": le(15), "waveheight": le(2.5)},
            )
            yield vessel.task_wrapper(
                "Check Mooring Lines",
                12,
                suspendable=True,
                constraints={"windspeed": le(15), "waveheight": le(2.5)},
            )

            group_time = vessel.env.now - start
            queue.vessel.submit_action_log(
                "Positioning Support",
                group_time,
                location="site",
                num_vessels=station_keeping_vessels,
            )
            yield queue.vessel.release.succeed()
            vessel.submit_debug_log(progress="Substructure")
            n += 1

        else:
            start = vessel.env.now
            yield queue.activate
            delay_time = vessel.env.now - start

            if n != 0:
                vessel.submit_action_log("Delay", delay_time, location="Site")

    yield vessel.transit(distance)
