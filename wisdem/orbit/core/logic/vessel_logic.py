"""This module contains common simulation logic related to vessels."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import process

from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.core.exceptions import ItemNotFound, MissingComponent


@process
def prep_for_site_operations(vessel, survey_required=False, **kwargs):
    """
    Performs preperation process at site.

    Parameters
    ----------
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

    yield position_onsite(vessel, **kwargs)
    yield stabilize(vessel, **kwargs)

    if survey_required:
        survey_time = kwargs.get("rov_survey_time", pt["rov_survey_time"])
        yield vessel.task_wrapper(
            "RovSurvey",
            survey_time,
            constraints=vessel.transit_limits,
            **kwargs,
        )


@process
def stabilize(vessel, **kwargs):
    """
    Task representing time required to stabilize the vessel. If the vessel
    has a dynamic positioning system, this task does not take any time. If the
    vessel has a jacking system, the vessel will jackup.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `dynamic_positioning` or
        `jacksys`.
    """

    try:
        _ = vessel.dynamic_positioning
        return

    except MissingComponent:
        pass

    try:
        jacksys = vessel.jacksys
        site_depth = kwargs.get("site_depth", 40)
        extension = kwargs.get("extension", site_depth + 10)
        jackup_time = jacksys.jacking_time(extension, site_depth)
        yield vessel.task_wrapper("Jackup", jackup_time, constraints=vessel.transit_limits, **kwargs)

    except MissingComponent:
        raise MissingComponent(vessel, ["Dynamic Positioning", "Jacking System"])


@process
def jackdown_if_required(vessel, **kwargs):
    """
    Task representing time required to jackdown the vessel, if jacking system
    is configured. If not, this task does not take anytime and the vessel is
    released.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel.
    """

    try:
        jacksys = vessel.jacksys
        site_depth = kwargs.get("site_depth", 40)
        extension = kwargs.get("extension", site_depth + 10)
        jackdown_time = jacksys.jacking_time(extension, site_depth)
        yield vessel.task_wrapper(
            "Jackdown",
            jackdown_time,
            constraints=vessel.transit_limits,
            **kwargs,
        )

    except MissingComponent:
        return


@process
def position_onsite(vessel, **kwargs):
    """
    Task representing time required to position `vessel` onsite.

    Parameters
    ----------
    vessel : Vessel
        Performing vessel. Requires configured `transit_limits`.
    """

    position_time = kwargs.get("site_position_time", pt["site_position_time"])

    yield vessel.task_wrapper("Position Onsite", position_time, constraints=vessel.transit_limits)


@process
def shuttle_items_to_queue(vessel, port, queue, distance, items, **kwargs):
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

    while True:

        if vessel.at_port:
            vessel.submit_debug_log(message=f"{vessel} is at port.")

            if not port.items:
                vessel.submit_debug_log(message="No items at port. Shutting down.")
                break

            # Get list of items
            try:
                yield get_list_of_items_from_port(vessel, port, items, **kwargs)

            except ItemNotFound:
                # If no items are at port and vessel.storage.items is empty,
                # the job is done
                if not vessel.storage.items:
                    vessel.submit_debug_log(message="Items not found. Shutting down.")
                    break

            # Transit to site
            vessel.update_trip_data()
            vessel.at_port = False
            yield vessel.task_wrapper("Transit", transit_time, constraints=vessel.transit_limits)
            yield stabilize(vessel, **kwargs)
            vessel.at_site = True

        if vessel.at_site:
            vessel.submit_debug_log(message=f"{vessel} is at site.")

            # Join queue to be active feeder at site
            with queue.request() as req:
                queue_start = vessel.env.now
                yield req

                queue_time = vessel.env.now - queue_start
                if queue_time > 0:
                    vessel.submit_action_log("Queue", queue_time, location="Site")

                queue.vessel = vessel
                active_start = vessel.env.now
                queue.activate.succeed()

                # Released by WTIV when objects are depleted
                vessel.release = vessel.env.event()
                yield vessel.release
                active_time = vessel.env.now - active_start

                vessel.submit_action_log("ActiveFeeder", active_time, location="Site")

                queue.vessel = None
                queue.activate = vessel.env.event()

            # Transit back to port
            vessel.at_site = False
            yield jackdown_if_required(vessel, **kwargs)
            yield vessel.task_wrapper("Transit", transit_time, constraints=vessel.transit_limits)
            vessel.at_port = True


@process
def get_list_of_items_from_port(vessel, port, items, **kwargs):
    """
    Retrieves multiples of 'items' from port until full.

    Parameters
    ----------
    vessel : Vessel
    port : Port
        Port simulation object to retrieve items from.
    items : list
        List of tuples representing items to get from port.
        - ('key': 'value')
    """

    with port.crane.request() as req:
        # Join queue to be active vessel at port
        queue_start = vessel.env.now
        yield req
        queue_time = vessel.env.now - queue_start
        if queue_time > 0:
            vessel.submit_action_log("Queue", queue_time)

        if port.items:
            while True:
                buffer = []
                for i in items:
                    item = port.get_item(i)
                    buffer.append(item)

                # Calculate deck space and mass of one complete turbine
                total_deck_space = sum([item.deck_space for item in buffer])
                proposed_deck_space = vessel.storage.current_deck_space + total_deck_space

                total_mass = sum([item.mass for item in buffer])
                proposed_mass = vessel.storage.current_cargo_mass + total_mass

                if vessel.storage.current_cargo_mass == 0:

                    if proposed_deck_space > vessel.storage.max_deck_space:

                        msg = f"Warning: '{vessel}' Deck Space Capacity Exceeded"
                        vessel.submit_debug_log(message=msg)

                    if proposed_mass > vessel.storage.max_cargo_mass:

                        msg = f"Warning: '{vessel}' Cargo Mass Capacity Exceeded"
                        vessel.submit_debug_log(message=msg)

                elif proposed_deck_space > vessel.storage.max_deck_space:
                    vessel.submit_debug_log(message="Full")
                    for item in buffer:
                        port.put(item)
                    break

                elif proposed_mass > vessel.storage.max_cargo_mass:
                    vessel.submit_debug_log(message="Full")
                    for item in buffer:
                        port.put(item)
                    break

                for item in buffer:
                    action, time = item.fasten(**kwargs)
                    vessel.storage.put_item(item)

                    if time > 0:
                        yield vessel.task_wrapper(
                            action,
                            time,
                            constraints=vessel.transit_limits,
                            **kwargs,
                        )

        else:
            raise ItemNotFound(items)


@process
def shuttle_items_to_queue_wait(vessel, port, queue, distance, items, per_trip, assigned, **kwargs):
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
    per_trip : int
    assigned : int
    """

    transit_time = vessel.transit_time(distance)

    n = 0
    while n < assigned:

        vessel.submit_debug_log(message=f"{vessel} is at port.")

        # Get list of items
        per_trip = max([per_trip, 1])
        yield get_list_of_items_from_port_wait(vessel, port, items * per_trip, **kwargs)

        # Transit to site
        vessel.update_trip_data()
        yield vessel.task("Transit", transit_time, constraints=vessel.transit_limits)
        yield stabilize(vessel, **kwargs)

        vessel.submit_debug_log(message=f"{vessel} is at site.")

        # Join queue to be active feeder at site
        with queue.request() as req:
            queue_start = vessel.env.now
            yield req

            queue_time = vessel.env.now - queue_start
            if queue_time > 0:
                vessel.submit_action_log("Queue", queue_time, location="Site")

            queue.vessel = vessel
            active_start = vessel.env.now
            queue.activate.succeed()

            # Released by WTIV when objects are depleted
            vessel.release = vessel.env.event()
            yield vessel.release
            active_time = vessel.env.now - active_start

            vessel.submit_action_log("ActiveFeeder", active_time, location="Site")

            queue.vessel = None
            queue.activate = vessel.env.event()

            # Transit back to port
            vessel.at_site = False
            yield jackdown_if_required(vessel, **kwargs)
            yield vessel.task("Transit", transit_time, constraints=vessel.transit_limits)

        n += per_trip


@process
def get_list_of_items_from_port_wait(vessel, port, items, **kwargs):
    """
    Retrieves multiples of 'items' from port until full.

    Parameters
    ----------
    vessel : Vessel
    port : Port
    items : list
        List of tuples representing items to get from port.
        - ('key': 'value')
    port : Port
        Port object to get items from.
    """

    with port.crane.request() as req:
        # Join queue to be active vessel at port
        queue_start = vessel.env.now
        yield req
        queue_time = vessel.env.now - queue_start
        if queue_time > 0:
            vessel.submit_action_log("Queue", queue_time)

        for i in items:
            wait_start = vessel.env.now
            item = yield port.get(lambda x: x.type == i)
            wait_time = vessel.env.now - wait_start

            if wait_time > 0:
                vessel.submit_action_log(f"Wait for {item}", wait_time)

            action, time = item.fasten(**kwargs)
            vessel.storage.put_item(item)

            if time > 0:
                yield vessel.task(action, time, constraints=vessel.transit_limits, **kwargs)
