"""This module contains common simulation logic related to vessels."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.vessels import tasks
from wisdem.orbit.simulation.exceptions import ItemNotFound

from .port_logic import get_list_of_items_from_port


def release_map(key, **kwargs):
    """
    Maps item key to the appropriate release equation.

    Parameters
    ----------
    key : str
        Item 'type'.
    """

    if key == "Monopile":
        release_time = tasks.release_monopile(**kwargs)
        action = "ReleaseMonopile"

    elif key == "Transition Piece":
        release_time = tasks.release_transition_piece(**kwargs)
        action = "ReleaseTP"

    elif key == "Tower Section":
        release_time = tasks.release_tower_section(**kwargs)
        action = "ReleaseTowerSection"

    elif key == "Nacelle":
        release_time = tasks.release_nacelle(**kwargs)
        action = "ReleaseNacelle"

    elif key == "Blade":
        release_time = tasks.release_turbine_blade(**kwargs)
        action = "ReleaseBlade"

    elif key == "Topside":
        release_time = tasks.release_topside(**kwargs)
        action = "ReleaseTopside"

    else:
        raise Exception(f"Item release method not found for '{key}'.")

    return release_time, action


def get_item_from_storage(
    env, vessel, item_type, action_vessel=None, release=False, **kwargs
):
    """
    Generic item retrieval process.
    Subprocesses:
    - release 'item'

    Parameters
    ----------
    env : Environment
    item : str
        Hook to find 'item' in 'vessel.storage' with attr {'type': 'item'}.
    vessel : Vessel
        Vessel to pick monopile from.
    action_vessel : Vessel (optional)
        If defined, the logging statement uses this vessel.
    release : bool (optional)
        If True, triggers vessel.release.succeed() when vessel.storage is empty.
        Used for WTIV + Feeder strategy to signal when feeders can leave.
    """

    if action_vessel is None:
        action_vessel = vessel

    item_rule = ("type", item_type)

    try:
        item = vessel.storage.get_item(item_rule).value

    except ItemNotFound as e:
        env.logger.debug(
            "Item not found.",
            extra={"agent": vessel.name, "time": env.now, "type": "Status"},
        )

        raise e

    time, action = release_map(item_type, **kwargs)

    task = {
        "agent": action_vessel.name,
        "action": action,
        "duration": time,
        "type": "Operations",
        "location": "Site",
        **action_vessel.transit_limits,
    }

    yield env.process(env.task_handler(task))

    if release and vessel.storage.any_remaining(item_rule) is False:
        vessel.release.succeed()

    return item


def prep_for_site_operations(env, vessel, survey_required=False, **kwargs):
    """
    Performs preperation process at site.

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    depth : int | float
        Site depth (m).
    extension : int | float
        Jack-up extension length (m).

    Yields
    ------
    task_list : list
        List of tasks included in preperation process.
    """

    site_depth = kwargs.get("site_depth", None)
    extension = kwargs.get("extension", site_depth + 10)

    position_time = tasks.position_onsite(**kwargs)
    jackup_time = vessel.jacksys.jacking_time(extension, site_depth)

    _shared = {
        "agent": vessel.name,
        "location": "Site",
        "type": "Operations",
        **vessel.transit_limits,
    }

    task_list = [
        {"action": "PositionOnsite", "duration": position_time, **_shared},
        {"action": "Jackup", "duration": jackup_time, **_shared},
    ]

    if survey_required:
        survey_time = tasks.rov_survey(**kwargs)
        survey_task = {
            "action": "RovSurvey",
            "duration": survey_time,
            **_shared,
        }

        task_list.append(survey_task)

    yield env.process(env.task_handler(task_list))


def shuttle_items_to_queue(
    env, vessel, port, queue, distance, items, **kwargs
):
    """
    Shuttles a list of items from port to queue.

    Parameters
    ----------
    env : Environemt
    vessel : Vessel
    port : Port
    queue : simpy.Resource
        Queue object to shuttle to.
    distance : int | float
        Distance between port and site (km).
    items : list
        List of components stored as tuples to shuttle.
        - ('key', 'value')
    """

    transit_time = vessel.transit_time(distance)

    transit = {
        "agent": vessel.name,
        "location": "At Sea",
        "type": "Operations",
        "action": "Transit",
        "duration": transit_time,
        **vessel.transit_limits,
    }

    while True:

        if vessel.at_port:
            env.logger.debug(
                "{} is at port.".format(vessel.name),
                extra={
                    "agent": vessel.name,
                    "time": env.now,
                    "type": "Status",
                },
            )

            if not port.items:
                env.logger.debug(
                    "No items at port. Shutting down.",
                    extra={
                        "agent": vessel.name,
                        "time": env.now,
                        "type": "Status",
                    },
                )
                break

            # Get list of items
            try:
                yield env.process(
                    get_list_of_items_from_port(
                        env, vessel, items, port, **kwargs
                    )
                )

            except ItemNotFound:
                # If no items are at port and vessel.storage.items is empty,
                # the job is done
                if not vessel.storage.items:
                    env.logger.debug(
                        "Item not found. Shutting down.",
                        extra={
                            "agent": vessel.name,
                            "time": env.now,
                            "type": "Status",
                        },
                    )
                    break

            # Transit to site
            vessel.update_trip_data()
            vessel.at_port = False
            yield env.process(env.task_handler(transit))
            vessel.at_site = True

        if vessel.at_site:
            env.logger.debug(
                "{} is at site.".format(vessel.name),
                extra={
                    "agent": vessel.name,
                    "time": env.now,
                    "type": "Status",
                },
            )

            # Join queue to be active feeder at site
            with queue.request() as req:
                queue_start = env.now
                yield req

                queue_time = env.now - queue_start
                if queue_time > 0:
                    env.logger.info(
                        "",
                        extra={
                            "agent": vessel.name,
                            "time": env.now,
                            "type": "Delay",
                            "action": "Queue",
                            "duration": queue_time,
                            "location": "Site",
                        },
                    )

                queue.vessel = vessel
                active_start = env.now
                queue.activate.succeed()

                # Released by WTIV when objects are depleted
                vessel.release = env.event()
                yield vessel.release
                active_time = env.now - active_start
                env.logger.info(
                    "",
                    extra={
                        "agent": vessel.name,
                        "time": env.now,
                        "type": "Operations",
                        "action": "ActiveFeeder",
                        "duration": active_time,
                        "location": "Site",
                    },
                )

                queue.vessel = None
                queue.activate = env.event()

            # Transit back to port
            vessel.at_site = False
            yield env.process(env.task_handler(transit))
            vessel.at_port = True
