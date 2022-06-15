"""`FloatingSubstationInstallation` and related processes."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import Agent, le, process
from marmot._exceptions import AgentNotRegistered

from wisdem.orbit.core import WetStorage
from wisdem.orbit.core.logic import position_onsite
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.phases.install.mooring_install.mooring import (
    install_mooring_line,
    install_mooring_anchor,
    perform_mooring_site_survey,
)


class FloatingSubstationInstallation(InstallPhase):
    """
    Offshore Substation (OSS) installation process using the quayside assembly
    and tow-out processes.
    """

    phase = "Offshore Substation Installation"
    capex_category = "Offshore Substation"

    #:
    expected_config = {
        "num_substations": "int",
        "oss_install_vessel": "str",
        "site": {"distance": "km", "depth": "m"},
        "offshore_substation_topside": {
            "unit_cost": "USD",
            "attach_time": "int | float (optional, default: 24)",
        },
        "offshore_substation_substructure": {
            "type": "Floating",
            "takt_time": "int | float (optional, default: 0)",
            "unit_cost": "USD",
            "mooring_cost": "USD",
            "towing_speed": "int | float (optional, default: 6 km/h)",
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

    def setup_simulation(self, **kwargs):
        """
        Initializes required objects for simulation.
        - Creates port

        - Creates support vessel + towing vessels
        """

        self.distance = self.config["site"]["distance"]
        self.num_substations = self.config["num_substations"]

        self.initialize_substructure_production()
        self.initialize_installation_vessel()

    @property
    def system_capex(self):
        """Returns total procurement cost of the substation substructures,
        topsides and mooring."""

        topside = self.config["offshore_substation_topside"]["unit_cost"]
        substructure = self.config["offshore_substation_substructure"]["unit_cost"]
        mooring = self.config["offshore_substation_substructure"]["mooring_cost"]

        return self.num_substations * (topside + substructure + mooring)

    def initialize_substructure_production(self):
        """
        Initializes the production of the floating substation substructures at
        quayside.
        """

        self.wet_storage = WetStorage(self.env, float("inf"))
        takt_time = self.config["offshore_substation_substructure"].get("takt_time", 0)
        attach_time = self.config["offshore_substation_topside"].get("attach_time", 24)
        to_assemble = [1] * self.num_substations

        self.assembly_line = SubstationAssemblyLine(to_assemble, takt_time, attach_time, self.wet_storage, 1)

        self.env.register(self.assembly_line)
        self.assembly_line.start()

    def initialize_installation_vessel(self):
        """Initialize the floating substation installation vessel."""

        support = self.config["oss_install_vessel"]
        vessel = self.initialize_vessel("Floating Substation Installation Vessel", support)
        self.env.register(vessel)
        vessel.initialize(mobilize=False)
        self.support_vessel = vessel

        depth = self.config["site"]["depth"]
        towing_speed = self.config["offshore_substation_substructure"].get("towing_speed", 6)

        install_floating_substations(
            self.support_vessel,
            self.wet_storage,
            self.distance,
            towing_speed,
            depth,
            self.num_substations,
        )

    @property
    def detailed_output(self):

        return {}


@process
def install_floating_substations(vessel, feed, distance, towing_speed, depth, number):
    """
    Process steps that installation vessel at site performs to install floating
    substations.

    Parameters
    ----------
    vessel : Agent
        Performing agent.
    feed : simply.Resource
        Wet storage for completed assemblies.
    distance : int | float
        Distance from port to site.
    towing_speed : int | float
        Speed at which completed assembly can be towed to site at (km/h).
    depth : int | float
        Site depth (m).
    number : int
        Number of substations to install.
    """

    travel_time = distance / towing_speed

    for _ in range(number):

        start = vessel.env.now
        yield feed.get()
        delay = vessel.env.now - start
        if delay > 0:
            vessel.submit_action_log("Delay: Waiting on Completed Assembly", delay)

        yield vessel.task(
            "Tow Substation to Site",
            travel_time,
            constraints=vessel.operational_limits,
        )
        yield position_onsite(vessel)
        yield vessel.task_wrapper(
            "Ballast to Operational Draft",
            6,
            constraints={"windspeed": le(15), "waveheight": le(2.5)},
        )

        for _ in range(3):
            yield perform_mooring_site_survey(vessel)
            yield install_mooring_anchor(vessel, depth, "Suction Pile")
            yield install_mooring_line(vessel, depth)
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

        yield vessel.transit(distance)


class SubstationAssemblyLine(Agent):
    """Substation Assembly Line Class."""

    def __init__(self, assigned, takt_time, attach_time, target, num):
        """
        Creates an instance of `SubstructureAssemblyLine`.

        Parameters
        ----------
        assigned : list
            List of assigned tasks. Can be shared with other assembly lines.
        takt_time : int | float
            Hours required to produce one substructure.
        attach_time : int | float
            Hours required to attach a topside to the substructure.
        target : simpy.Store
            Target storage.
        num : int
            Assembly line number designation.
        """

        super().__init__(f"Substation Assembly Line {num}")

        self.assigned = assigned
        self.takt_time = takt_time
        self.attach_time = attach_time
        self.target = target

    def submit_action_log(self, action, duration, **kwargs):
        """
        Submits a log representing a completed `action` performed over time
        `duration`.

        This method overwrites the default `submit_action_log` in
        `marmot.Agent`, adding operation cost to every submitted log within
        ORBIT.

        Parameters
        ----------
        action : str
            Performed action.
        duration : int | float
            Duration of action.

        Raises
        ------
        AgentNotRegistered
        """

        if self.env is None:
            raise AgentNotRegistered(self)

        else:
            payload = {
                **kwargs,
                "agent": str(self),
                "action": action,
                "duration": float(duration),
                "cost": 0,
            }

            self.env._submit_log(payload, level="ACTION")

    @process
    def assemble_substructure(self):
        """
        Simulation process for assembling a substructure.
        """

        yield self.task("Substation Substructure Assembly", self.takt_time)
        yield self.task("Attach Topside", self.attach_time)
        substation = FloatingSubstation()

        start = self.env.now
        yield self.target.put(substation)
        delay = self.env.now - start

        if delay > 0:
            self.submit_action_log("Delay: No Wet Storage Available", delay)

    @process
    def start(self):
        """
        Trigger the assembly line to run. Will attempt to pull a task from
        self.assigned and timeout for the assembly time. Shuts down after
        self.assigned is empty.
        """

        while True:
            try:
                _ = self.assigned.pop(0)
                yield self.assemble_substructure()

            except IndexError:
                break


class FloatingSubstation:
    """Floating Substructure Class."""

    def __init__(self):
        """Creates an instance of `Substructure`."""

        pass
