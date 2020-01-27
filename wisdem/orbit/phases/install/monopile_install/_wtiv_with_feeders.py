"""Provides processes specific to monopile installation with feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.vessels import tasks
from wisdem.orbit.simulation.logic import (
    get_item_from_storage,
    prep_for_site_operations,
)
from wisdem.orbit.phases.install.monopile_install._common import (
    install_monopile,
    install_transition_piece,
)


def install_monopiles_from_queue(env, wtiv, queue, number, distance, **kwargs):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses to install
    monopiles and transition pieces from queue of feeder barges.

    Parameters
    ----------
    env : simulation.Environment
        SimPy environment that the simulation runs in.
    wtiv : vessels.Vessel
        Vessel object that represents the WTIV.
    queue : simpy.Resource
        Queue object to interact with active feeder barge.
    number : int
        Total monopiles to install.
    distance : int | float
        Distance from site to port (km).
    """

    transit_time = wtiv.transit_time(distance)

    transit = {
        "agent": wtiv.name,
        "location": "At Sea",
        "type": "Operations",
        "action": "Transit",
        "duration": transit_time,
        **wtiv.transit_limits,
    }

    n = 0
    while n < number:
        if wtiv.at_port:
            # Transit to site
            wtiv.at_port = False
            yield env.process(env.task_handler(transit))
            wtiv.at_site = True

        if wtiv.at_site:

            if queue.vessel:

                # Prep for monopile install
                yield env.process(
                    prep_for_site_operations(
                        env, wtiv, survey_required=True, **kwargs
                    )
                )

                # Get monopile
                monopile = yield env.process(
                    get_item_from_storage(
                        env=env,
                        vessel=queue.vessel,
                        item_type="Monopile",
                        action_vessel=wtiv,
                        release=False,
                        **kwargs,
                    )
                )

                upend_time = tasks.upend_monopile(
                    wtiv, monopile["length"], **kwargs
                )

                upend = {
                    "action": "UpendMonopile",
                    "duration": upend_time,
                    "agent": wtiv.name,
                    "location": "Site",
                    "type": "Operations",
                    **wtiv.operational_limits,
                }

                yield env.process(env.task_handler(upend))

                # Install monopile
                yield env.process(
                    install_monopile(env, wtiv, monopile, **kwargs)
                )

                # Get transition piece from active feeder
                tp = yield env.process(
                    get_item_from_storage(
                        env=env,
                        vessel=queue.vessel,
                        item_type="Transition Piece",
                        action_vessel=wtiv,
                        release=True,
                        **kwargs,
                    )
                )

                # Install transition piece
                yield env.process(
                    install_transition_piece(env, wtiv, tp, **kwargs)
                )

                n += 1

            else:
                start = env.now
                yield queue.activate
                delay_time = env.now - start
                env.logger.info(
                    "",
                    extra={
                        "agent": wtiv.name,
                        "time": env.now,
                        "type": "Delay",
                        "action": "WaitForFeeder",
                        "duration": delay_time,
                        "location": "Site",
                    },
                )

    # Transit to port
    wtiv.at_site = False
    yield env.process(env.task_handler(transit))
    wtiv.at_port = True

    env.logger.debug(
        "Monopile installation complete!",
        extra={
            "agent": wtiv.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )
