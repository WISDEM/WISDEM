"""Provides processes specific to single WTIV monopile installation."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.vessels import tasks
from wisdem.orbit.simulation.logic import (
    get_item_from_storage,
    prep_for_site_operations,
    get_list_of_items_from_port,
)
from wisdem.orbit.simulation.exceptions import ItemNotFound
from wisdem.orbit.phases.install.monopile_install._common import (
    install_monopile,
    install_transition_piece,
)


def solo_install_monopiles(env, vessel, port, distance, number, **kwargs):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses during a single
    monopile installation process.

    Parameters
    ----------
    env : simulation.Environment
        SimPy environment that the simulation runs in.
    vessel : vessels.Vessel
        Vessel object that represents the WTIV.
    distance : int | float
        Distance between port and site (km).
    number : int
        Total monopiles to install.
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

    component_list = [("type", "Monopile"), ("type", "Transition Piece")]

    n = 0
    while n < number:
        if vessel.at_port:
            try:
                # Get substructure + transition piece from port
                yield env.process(
                    get_list_of_items_from_port(
                        env, vessel, component_list, port, **kwargs
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

            if vessel.storage.items:
                # Prep for monopile install
                yield env.process(
                    prep_for_site_operations(
                        env, vessel, survey_required=True, **kwargs
                    )
                )

                # Get monopile from internal storage
                monopile = yield env.process(
                    get_item_from_storage(
                        env, vessel, item_type="Monopile", **kwargs
                    )
                )

                upend_time = tasks.upend_monopile(
                    vessel, monopile["length"], **kwargs
                )

                upend = {
                    "action": "UpendMonopile",
                    "duration": upend_time,
                    "agent": vessel.name,
                    "location": "Stie",
                    "type": "Operations",
                    **vessel.operational_limits,
                }

                yield env.process(env.task_handler(upend))

                # Install monopile
                yield env.process(
                    install_monopile(env, vessel, monopile, **kwargs)
                )

                # Get transition piece from internal storage
                tp = yield env.process(
                    get_item_from_storage(
                        env, vessel, item_type="Transition Piece", **kwargs
                    )
                )

                # Install transition piece
                yield env.process(
                    install_transition_piece(env, vessel, tp, **kwargs)
                )

                n += 1

            else:
                # Transit to port
                vessel.at_site = False
                yield env.process(env.task_handler(transit))
                vessel.at_port = True

    env.logger.debug(
        "Monopile installation complete!",
        extra={
            "agent": vessel.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )
