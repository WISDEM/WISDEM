"""Common processes and cargo types for Cable Installations."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import process

from wisdem.orbit.core.logic import position_onsite
from wisdem.orbit.core.defaults import process_times as pt


class SimpleCable:
    """Simple Cable Class"""

    def __init__(self, linear_density):
        """
        Creates an instance of SimpleCable.

        Parameters
        ----------
        linear_density : int | float
        """

        self.linear_density = linear_density


@process
def load_cable_on_vessel(vessel, cable, constraints={}, **kwargs):
    """
    Subprocess for loading `cable` onto the configured `vessel`.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Required to have configured `cable_storage`.
    cable : SimpleCable | Cable
        Cable type.
    constraints : dict
        Constraints to be applied to cable loading subprocess.
    """

    key = "cable_load_time"
    load_time = kwargs.get(key, pt[key])

    vessel.cable_storage.load_cable(cable)
    yield vessel.task_wrapper("Load Cable", load_time, constraints=constraints, **kwargs)


@process
def landfall_tasks(vessel, trench_length, **kwargs):
    """
    List of tasks that must be completed at landfall at the beginning of the
    export system installation process.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Required to have configured `cable_storage`.
    trench_length : int | float
        Length of trench that is dug through the beach (km).
    """

    yield tow_plow(vessel, trench_length, **kwargs)
    yield pull_winch(vessel, trench_length, **kwargs)
    yield prep_cable(vessel, **kwargs)
    yield pull_in_cable(vessel, **kwargs)
    yield terminate_cable(vessel, **kwargs)
    yield lower_cable(vessel, **kwargs)


@process
def prep_cable(vessel, **kwargs):
    """
    Task representing time required to prepare cable for pull-in.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `transit_limits`.
    """

    key = "cable_prep_time"
    prep_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper("Prepare Cable", prep_time, constraints=vessel.transit_limits, **kwargs)


@process
def lower_cable(vessel, **kwargs):
    """
    Task representing time required to lower cable to seafloor.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    """

    key = "cable_lower_time"
    lower_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Lower Cable",
        lower_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def pull_in_cable(vessel, **kwargs):
    """
    Task representing time required to pull cable into offshore substructure or
    at onshore trench.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    """

    key = "cable_pull_in_time"
    pull_in_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Pull In Cable",
        pull_in_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def terminate_cable(vessel, **kwargs):
    """
    Task representing time required to terminate and test cable connection.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    """

    key = "cable_termination_time"
    termination_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Terminate Cable",
        termination_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def lay_bury_cable(vessel, distance, **kwargs):
    """
    Task representing time required to lay and bury a cable section.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    distance : int | float
        Distance of cable section (km)
    cable_lay_bury_speed : int | float
        Maximum speed at which cable is dispensed (km/hr)
    """

    kwargs = {**kwargs, **getattr(vessel, "_transport_specs", {})}

    key = "cable_lay_bury_speed"
    lay_bury_speed = kwargs.get(key, pt[key])
    lay_bury_time = distance / lay_bury_speed

    yield vessel.task_wrapper(
        "Lay/Bury Cable",
        lay_bury_time,
        constraints=vessel.operational_limits,
        suspendable=True,
        **kwargs,
    )


@process
def lay_cable(vessel, distance, **kwargs):
    """
    Task representing time required to lay a cable section.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    distance : int | float
        Distance of cable section (km).
    cable_lay_speed : int | float
        Maximum speed at which cable is dispensed (km/hr)
    """

    kwargs = {**kwargs, **getattr(vessel, "_transport_specs", {})}

    key = "cable_lay_speed"
    lay_speed = kwargs.get(key, pt[key])
    lay_time = distance / lay_speed

    yield vessel.task_wrapper(
        "Lay Cable",
        lay_time,
        constraints=vessel.operational_limits,
        suspendable=True,
        **kwargs,
    )


@process
def bury_cable(vessel, distance, **kwargs):
    """
    Task representing time required to bury a cable section.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    distance : int | float
        Distance of cable section (km).
    cable_bury_speed : int | float
        Maximum speed at which cable is buried (km/hr).
    """

    kwargs = {**kwargs, **getattr(vessel, "_transport_specs", {})}

    key = "cable_bury_speed"
    bury_speed = kwargs.get(key, pt[key])
    bury_time = distance / bury_speed

    yield vessel.task_wrapper(
        "Bury Cable",
        bury_time,
        constraints=vessel.operational_limits,
        suspendable=True,
        **kwargs,
    )


@process
def splice_cable(vessel, **kwargs):
    """
    Task representing time required to splice a cable at sea.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    cable_splice_time : int | float
        Time required to splice two cable ends together (h).
    """

    key = "cable_splice_time"
    splice_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Splice Cable",
        splice_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def raise_cable(vessel, **kwargs):
    """
    Task representing time required to raise the unspliced cable from the
    seafloor.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    cable_raise_time : int | float
        Time required to raise the cable from the seafloor (h).
    """

    key = "cable_raise_time"
    raise_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Raise Cable",
        raise_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def splice_process(vessel, **kwargs):
    """
    A list of tasks representing the entire cable splicing process.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    """

    yield position_onsite(vessel)
    yield raise_cable(vessel, **kwargs)
    yield splice_cable(vessel, **kwargs)
    yield lower_cable(vessel, **kwargs)


@process
def tow_plow(vessel, distance, **kwargs):
    """
    Task representing time required to tow plow at landfall site.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    distance : int | float
        Distance to between cable laying vessel and onshore construction (km).
    tow_plow_speed : float
        Towing speed (km/h).
    """

    key = "tow_plow_speed"
    plow_speed = kwargs.get(key, pt[key])
    plow_time = distance / plow_speed

    yield vessel.task_wrapper("Tow Plow", plow_time, constraints=vessel.operational_limits, **kwargs)


@process
def pull_winch(vessel, distance, **kwargs):
    """
    Task representing time required to pull cable onshore through the
    previously dug trench.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operation_limits`.
    distance: int | float
        Distance winch wire must travel to reach the plow.
    pull_winch_speed : int | float
        Speed at wich the winch travels (km/h).
    """

    key = "pull_winch_speed"
    winch_speed = kwargs.get(key, pt[key])
    winch_time = distance / winch_speed

    yield vessel.task_wrapper(
        "Pull Winch",
        winch_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def dig_trench(vessel, distance, **kwargs):
    """
    Task representing time required to dig a trench prior to cable lay and burial

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `operational_limits`.
    distance : int | float
        Length of trench, equal to length of cable section (km).
    trench_dig_speed : int | float
        Speed at which trench is dug (km/hr).
    """

    kwargs = {**kwargs, **getattr(vessel, "_transport_specs", {})}

    key = "trench_dig_speed"
    trench_dig_speed = kwargs.get(key, pt[key])
    trench_dig_time = distance / trench_dig_speed

    yield vessel.task_wrapper(
        "Dig Trench",
        trench_dig_time,
        constraints=vessel.operational_limits,
        suspendable=True,
        **kwargs,
    )
