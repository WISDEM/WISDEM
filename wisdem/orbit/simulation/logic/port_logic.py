"""This module contains common simulation logic related to ports."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from types import SimpleNamespace

from wisdem.orbit.vessels import tasks
from wisdem.orbit.simulation.exceptions import FastenTimeNotFound, VesselCapacityError


def vessel_fasten_time(item, **kwargs):
    """
    Retrieves the amount of time it takes to fasten an item to a vessel.

    Parameters
    ----------
    item : dict
        Dictionary with key 'type' to indicate what is being fastened.

    Returns
    -------
    fasten_time : int or float
        The amount of time it takes to fasten an item to the vessel.

    Raises
    ------
    Exception
        [description]
    """

    if item["type"] == "Blade":
        fasten_time = tasks.fasten_turbine_blade(**kwargs)

    elif item["type"] == "Nacelle":
        fasten_time = tasks.fasten_nacelle(**kwargs)

    elif item["type"] == "Tower":
        fasten_time = tasks.fasten_tower(**kwargs)

    elif item["type"] == "Monopile":
        fasten_time = tasks.fasten_monopile(**kwargs)

    elif item["type"] == "Transition Piece":
        fasten_time = tasks.fasten_transition_piece(**kwargs)

    elif item["type"] == "Scour Protection":
        fasten_time = tasks.load_rocks(**kwargs)

    elif item["type"] == "Topside":
        fasten_time = tasks.fasten_topside(**kwargs)

    elif item["type"] == "Carousel":
        lift_time = tasks.lift_carousel(**kwargs)
        fasten_time = tasks.fasten_carousel(**kwargs)
        fasten_time += lift_time

    else:
        raise FastenTimeNotFound(item["type"])

    return fasten_time


def get_list_of_items_from_port(env, vessel, items, port, **kwargs):
    """
    Retrieves multiples of 'items' from port until full.

    Parameters
    ----------
    items : list
        List of tuples representing items to get from port.
        - ('key': 'value')
    port : Port
        Port object to get items from.
    partial_list : bool
        Controls if the vessel can take partial loads.
    """

    partial = kwargs.get("partial_list", False)
    if partial:
        raise NotImplemented("Partial list completion is not implemented.")

    else:
        with port.crane.request() as req:
            # Join queue to be active feeder at port
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
                        "location": "Port",
                    },
                )
                pass

            while True and port.items:
                buffer = []
                for item_rule in items:
                    item = port.get_item(item_rule).value
                    buffer.append(item)

                # Calculate deck space and weight of one complete turbine
                total_deck_space = sum([item["deck_space"] for item in buffer])
                proposed_deck_space = (
                    vessel.storage.current_deck_space + total_deck_space
                )

                total_weight = sum([item["weight"] for item in buffer])
                proposed_weight = (
                    vessel.storage.current_cargo_weight + total_weight
                )

                if proposed_deck_space > vessel.storage.max_deck_space:

                    env.logger.debug(
                        "{} is full".format(vessel.name),
                        extra={
                            "agent": vessel.name,
                            "time": env.now,
                            "type": "Status",
                            "location": "Port",
                        },
                    )

                    for item in buffer:
                        port.put(item)

                    if vessel.storage.current_cargo_weight > 0:
                        break

                    else:
                        raise VesselCapacityError(vessel, items)

                elif proposed_weight > vessel.storage.max_cargo_weight:

                    env.logger.debug(
                        "{} is full".format(vessel.name),
                        extra={
                            "agent": vessel.name,
                            "time": env.now,
                            "type": "Status",
                            "location": "Port",
                        },
                    )

                    for item in buffer:
                        port.put(item)

                    if vessel.storage.current_cargo_weight > 0:
                        break

                    else:
                        raise VesselCapacityError(vessel, items)

                else:
                    for item in buffer:
                        vessel.storage.put_item(item)

                        if item["type"] == "Carousel":
                            vessel.carousel = SimpleNamespace(**item)
                        env.logger.debug(
                            "",
                            extra={
                                "agent": vessel.name,
                                "time": env.now,
                                "type": "Operations",
                                "action": "ItemRetrieved",
                                "target": item["type"],
                            },
                        )

                        fasten_time = vessel_fasten_time(item, **kwargs)
                        yield env.timeout(fasten_time)
                        env.logger.info(
                            "",
                            extra={
                                "agent": vessel.name,
                                "time": env.now,
                                "type": "Operations",
                                "action": "FastenItem",
                                "location": "Port",
                                "duration": fasten_time,
                                "target": item["type"],
                            },
                        )
