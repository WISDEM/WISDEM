"""Provides the process logic for scour protection installation."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


from wisdem.orbit.vessels import tasks
from wisdem.orbit.simulation.logic import get_list_of_items_from_port


def transport(env, vessel, distance, to_port, to_site, **kwargs):
    """
    Subprocess to travel between port and site.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Cable laying vessel.
    distance : int or float
        Distance between port and site.
    to_port : bool
        Indicator for travelling to port (True) or to site (False).
    """
    transit_time = vessel.transit_time(distance)

    task = {
        "agent": vessel.name,
        "action": "Transit",
        "duration": transit_time,
        "location": "At Sea",
        "type": "Operations",
        **vessel.transit_limits,
    }

    if to_port and not to_site:
        vessel.at_site = False
    elif to_site and not to_port:
        vessel.at_port = False

    yield env.process(env.task_handler(task))

    if to_port and not to_site:
        vessel.at_port = True
    elif to_site and not to_port:
        vessel.at_site = True


def get_scour_protection_from_port(env, vessel, port, **kwargs):
    """
    Retrieves scour protection rocks from port.
    Subprocesses:
     - Load rocks into vessel's storage (time and amount)

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    vessel : Vessel
        Scour protection installation vessel.
    port : Port
        Port object.
    """

    component_list = [("type", "Scour Protection")]
    yield env.process(
        get_list_of_items_from_port(
            env, vessel, component_list, port, **kwargs
        )
    )


def install_scour_protection_at_site(env, vessel, rock_tonnes, **kwargs):
    """
    Drops rocks off at turbine site. Process the time it takes to install the
    scouring protection as well as the weight of rocks removed from cargo.

    Parameters
    ----------
    env : `simpy.Environment`
        SimPy environment object
    vessel : `Vessel`
        A vessel object.
    rock_tonnes : float
        Mass of scouring protection to install at fixed substructure.
    """

    drop_time = tasks.drop_rocks(**kwargs)
    yield vessel.storage.get_item("Scour Protection", rock_tonnes)

    task = {
        "agent": vessel.name,
        "type": "Operations",
        "location": "Site",
        "action": "DropRocks",
        "duration": drop_time,
        **vessel.transit_limits,
    }

    yield env.process(env.task_handler(task))
