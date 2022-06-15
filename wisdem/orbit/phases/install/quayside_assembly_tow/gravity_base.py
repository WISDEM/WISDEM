"""Installation strategies for gravity-base substructures."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import simpy
from marmot import le, process

from wisdem.orbit.core import Vessel, WetStorage
from wisdem.orbit.phases.install import InstallPhase

from .common import TowingGroup, TurbineAssemblyLine, SubstructureAssemblyLine


class GravityBasedInstallation(InstallPhase):
    """
    Installation module to model the quayside assembly, tow-out and
    installation of gravity based foundations.
    """

    phase = "Gravity Based Foundation Installation"
    capex_category = "Substructure"

    #:
    expected_config = {
        "support_vessel": "str",
        "towing_vessel": "str",
        "towing_vessel_groups": {
            "towing_vessels": "int",
            "station_keeping_vessels": "int",
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
        Creates an instance of GravityBasedInstallation.

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
        Initializes the production of substructures at port. The number of
        independent assembly lines and production time associated with a
        substructure can be configured with the following parameters:

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

        num = self.config["plant"]["num_turbines"]
        to_assemble = [1] * num

        self.sub_assembly_lines = []
        for i in range(lines):
            a = SubstructureAssemblyLine(to_assemble, time, self.wet_storage, i + 1)

            self.env.register(a)
            a.start()
            self.sub_assembly_lines.append(a)

    def initialize_turbine_assembly(self):
        """
        Initializes turbine assembly lines. The number of independent lines
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
            a = TurbineAssemblyLine(self.wet_storage, self.assembly_storage, turbine, i + 1)

            self.env.register(a)
            a.start()
            self.turbine_assembly_lines.append(a)

    def initialize_towing_groups(self, **kwargs):
        """
        Initializes towing groups to bring completed assemblies to site and
        stabilize the assembly during final installation.
        """

        self.installation_groups = []

        vessel = self.config["towing_vessel"]
        num_groups = self.config["towing_vessel_groups"].get("num_groups", 1)
        towing = self.config["towing_vessel_groups"]["towing_vessels"]
        towing_speed = self.config["substructure"].get("towing_speed", 6)

        for i in range(num_groups):
            g = TowingGroup(vessel, num=i + 1)
            self.env.register(g)
            g.initialize()
            self.installation_groups.append(g)

            transfer_gbf_substructures_from_storage(
                g,
                self.assembly_storage,
                self.distance,
                self.active_group,
                towing,
                towing_speed,
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

    def initialize_support_vessel(self, **kwargs):
        """
        Initializes Multi-Purpose Support Vessel to perform installation
        processes at site.
        """

        specs = self.config["support_vessel"]
        vessel = self.initialize_vessel("Multi-Purpose Support Vessel", specs)

        self.env.register(vessel)
        vessel.initialize(mobilize=False)
        self.support_vessel = vessel

        station_keeping_vessels = self.config["towing_vessel_groups"]["station_keeping_vessels"]

        install_gravity_base_foundations(
            self.support_vessel,
            self.active_group,
            self.distance,
            self.num_turbines,
            station_keeping_vessels,
            **kwargs,
        )

    @property
    def detailed_output(self):
        """"""

        return {
            "operational_delays": {
                **{k: self.operational_delay(str(k)) for k in self.sub_assembly_lines},
                **{k: self.operational_delay(str(k)) for k in self.turbine_assembly_lines},
                **{k: self.operational_delay(str(k)) for k in self.installation_groups},
                self.support_vessel: self.operational_delay(str(self.support_vessel)),
            }
        }

    def operational_delay(self, name):
        """"""

        actions = [a for a in self.env.actions if a["agent"] == name]
        delay = sum(a["duration"] for a in actions if "Delay" in a["action"])

        return delay


@process
def transfer_gbf_substructures_from_storage(group, feed, distance, queue, towing_vessels, towing_speed, **kwargs):
    """
    Process logic for the towing vessel group.

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
    towing_speed : int | float
        Configured towing speed (km/h)
    """

    towing_time = distance / towing_speed
    transit_time = distance / group.transit_speed

    while True:

        start = group.env.now
        assembly = yield feed.get()
        delay = group.env.now - start

        if delay > 0:
            group.submit_action_log("Delay: No Completed Assemblies Available", delay)

        yield group.group_task("Tow Substructure", towing_time, num_vessels=towing_vessels)

        # At Site
        with queue.request() as req:
            queue_start = group.env.now
            yield req

            queue_time = group.env.now - queue_start
            if queue_time > 0:
                group.submit_action_log("Queue", queue_time, location="Site")

            queue.vessel = group
            active_start = group.env.now
            queue.activate.succeed()

            # Released by WTIV when objects are depleted
            group.release = group.env.event()
            yield group.release
            active_time = group.env.now - active_start

            queue.vessel = None
            queue.activate = group.env.event()

        yield group.group_task("Transit", transit_time, num_vessels=towing_vessels)


@process
def install_gravity_base_foundations(vessel, queue, distance, substructures, station_keeping_vessels, **kwargs):
    """
    Logic that a Multi-Purpose Support Vessel uses at site to complete the
    installation of gravity based foundations.

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

    n = 0
    while n < substructures:
        if queue.vessel:

            start = vessel.env.now
            if n == 0:
                vessel.mobilize()
                yield vessel.transit(distance)

            yield vessel.task_wrapper(
                "Position Substructure",
                5,
                constraints={"windspeed": le(15), "waveheight": le(2)},
            )
            yield vessel.task_wrapper(
                "ROV Survey",
                1,
                constraints={"windspeed": le(25), "waveheight": le(3)},
            )

            # TODO: Model for ballast pump time
            yield vessel.task_wrapper(
                "Pump Ballast",
                12,
                # suspendable=True,
                constraints={"windspeed": le(15), "waveheight": le(2)},
            )

            # TODO: Model for GBF grout time
            yield vessel.task_wrapper(
                "Grout GBF",
                6,
                suspendable=True,
                constraints={"windspeed": le(15), "waveheight": le(2)},
            )

            group_time = vessel.env.now - start
            queue.vessel.submit_action_log(
                "Positioning Support",
                group_time,
                location="site",
                num_vessels=station_keeping_vessels,
            )
            yield queue.vessel.release.succeed()
            n += 1

        else:
            start = vessel.env.now
            yield queue.activate
            delay_time = vessel.env.now - start

            if n != 0:
                vessel.submit_action_log("Delay", delay_time, location="Site")

    yield vessel.transit(distance)
