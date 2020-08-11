"""Common processes and cargo types for quayside assembly and tow-out
installations"""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import Agent, le, process
from marmot._exceptions import AgentNotRegistered


class Substructure:
    """Floating Substructure Class."""

    def __init__(self):
        """Creates an instance of `Substructure`."""

        pass


class SubstructureAssemblyLine(Agent):
    """Substructure Assembly Line Class."""

    def __init__(self, assigned, time, target, num):
        """
        Creates an instance of `SubstructureAssemblyLine`.

        Parameters
        ----------
        assigned : list
            List of assigned tasks. Can be shared with other assembly lines.
        time : int | float
            Hours required to produce one substructure.
        target : simpy.Store
            Target storage.
        num : int
            Assembly line number designation.
        """

        super().__init__(f"Substructure Assembly Line {num}")

        self.assigned = assigned
        self.time = time
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

        yield self.task("Substructure Assembly", self.time)
        substructure = Substructure()

        start = self.env.now
        yield self.target.put(substructure)
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


class TurbineAssemblyLine(Agent):
    """Turbine Assembly Line Class."""

    def __init__(self, feed, target, turbine, num):
        """
        Creates an instance of `TurbineAssemblyLine`.

        Parameters
        ----------
        feed : simpy.Store
            Storage for completed substructures.
        target : simpy.Store
            Target storage.
        num : int
            Assembly line number designation.
        """

        super().__init__(f"Turbine Assembly Line {num}")

        self.feed = feed
        self.target = target
        self.turbine = turbine

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
    def start(self):
        """
        Trigger the assembly line to run. Will attempt to pull a task from
        self.assigned and timeout for the assembly time. Shuts down after
        self.assigned is empty.
        """

        while True:
            start = self.env.now
            sub = yield self.feed.get()
            delay = self.env.now - start

            if delay > 0:
                self.submit_action_log(
                    "Delay: No Substructures in Wet Storage", delay
                )

            yield self.assemble_turbine()

    @process
    def assemble_turbine(self):
        """
        Turbine assembly process. Follows a similar process as the
        `TurbineInstallation` modules but has fixed lift times + fasten times
        instead of calculating the lift times dynamically.
        """

        yield self.move_substructure()
        yield self.prepare_for_assembly()

        sections = self.turbine["tower"].get("sections", 1)
        for _ in range(sections):
            yield self.lift_and_attach_tower_section()

        yield self.lift_and_attach_nacelle()

        for _ in range(3):
            yield self.lift_and_attach_blade()

        yield self.mechanical_completion()

        start = self.env.now
        yield self.target.put(1)
        delay = self.env.now - start

        if delay > 0:
            self.submit_action_log(
                "Delay: No Assembly Storage Available", delay
            )

        self.submit_debug_log(
            message="Assembly delievered to installation groups."
        )

    @process
    def move_substructure(self):
        """
        Task representing time associated with moving the completed
        substructure assembly to the turbine assembly line.

        TODO: Move to dynamic process involving tow groups.
        """

        yield self.task("Move Substructure", 8)

    @process
    def prepare_for_assembly(self):
        """
        Task representing time associated with preparing a substructure for
        turbine assembly.
        """

        yield self.task("Prepare for Turbine Assembly", 12)

    @process
    def lift_and_attach_tower_section(self):
        """
        Task representing time associated with lifting and attaching a tower
        section at quayside.
        """

        yield self.task(
            "Lift and Attach Tower Section",
            12,
            constraints={"windspeed": le(15)},
        )

    @process
    def lift_and_attach_nacelle(self):
        """
        Task representing time associated with lifting and attaching a nacelle
        at quayside.
        """

        yield self.task(
            "Lift and Attach Nacelle", 7, constraints={"windspeed": le(15)}
        )

    @process
    def lift_and_attach_blade(self):
        """
        Task representing time associated with lifting and attaching a turbine
        blade at quayside.
        """

        yield self.task(
            "Lift and Attach Blade", 3.5, constraints={"windspeed": le(12)}
        )

    @process
    def mechanical_completion(self):
        """
        Task representing time associated with performing mechanical compltion
        work at quayside.
        """

        yield self.task(
            "Mechanical Completion", 24, constraints={"windspeed": le(18)}
        )


class TowingGroup(Agent):
    """Class to represent an arbitrary group of towing vessels."""

    def __init__(self, vessel_specs, num=1):
        """
        Creates an instance of TowingGroup.

        Parameters
        ----------
        vessel_specs : dict
            Specs for the individual vessels used in the towing group.
            Currently restricted to one vessel specification per group.
        """

        super().__init__(f"Towing Group {num}")
        self._specs = vessel_specs
        self.day_rate = self._specs["vessel_specs"]["day_rate"]
        self.transit_speed = self._specs["transport_specs"]["transit_speed"]

    def initialize(self):
        """Initializes the towing group."""

        self.submit_debug_log(message="{self.name} initialized.")

    @process
    def group_task(
        self, name, duration, num_vessels, constraints={}, **kwargs
    ):
        """
        Submits a group task with any number of towing vessels.

        Parameters
        ----------
        name : str
            Name of task to complete. Used for submitting action logs.
        duration : float | int
            Duration of the task.
            Rounded up to the nearest int.
        num_vessels : int
            Number of individual towing vessels needed for the operation.
        """

        kwargs = {**kwargs, "num_vessels": num_vessels}
        yield self.task(name, duration, constraints=constraints, **kwargs)

    def operation_cost(self, hours, **kwargs):
        """
        Returns cost of an operation of duration `hours` using number of
        vessels, `num_vessels`.

        Parameters
        ----------
        hours : int | float
            Duration of operation in hours.
        vessels : int
            Default: 1
        """

        mult = kwargs.get("cost_multiplier", 1.0)
        vessels = kwargs.get("num_vessels", 1)
        return (self.day_rate / 24) * vessels * hours * mult

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
                "cost": self.operation_cost(duration, **kwargs),
            }

            self.env._submit_log(payload, level="ACTION")
