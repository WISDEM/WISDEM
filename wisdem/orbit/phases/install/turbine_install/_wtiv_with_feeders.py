"""Provides processes specific to turbine installation with feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.simulation.logic import (
    get_item_from_storage,
    prep_for_site_operations,
)
from wisdem.orbit.phases.install.turbine_install._common import (
    install_tower,
    install_nacelle,
    install_turbine_blade,
)


def install_turbine_components_from_queue(
    env, wtiv, queue, number, distance, **kwargs
):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses to install
    turbine componenets from a queue of feeder barges.

    Parameters
    ----------
    env : simulation.Environment
        SimPy environment that the simulation runs in.
    wtiv : vessels.Vessel
        Vessel object that represents the WTIV.
    queue : simpy.Resource
        Queue object to interact with active feeder barge.
    number : int
        Total turbine component sets to install.
    distance : int | float
        Distance from site to port (km).
    """

    transit_time = wtiv.transit_time(distance)
    reequip_time = wtiv.crane.reequip(**kwargs)

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

                # Prep for turbine install
                yield env.process(
                    prep_for_site_operations(
                        env, wtiv, survey_required=False, **kwargs
                    )
                )

                # Get tower
                tower = yield env.process(
                    get_item_from_storage(
                        env=env,
                        vessel=queue.vessel,
                        item_type="Tower",
                        action_vessel=wtiv,
                        release=False,
                        **kwargs,
                    )
                )

                # Install tower
                yield env.process(install_tower(env, wtiv, tower, **kwargs))

                # Get turbine nacelle
                nacelle = yield env.process(
                    get_item_from_storage(
                        env=env,
                        vessel=queue.vessel,
                        item_type="Nacelle",
                        action_vessel=wtiv,
                        release=False,
                        **kwargs,
                    )
                )

                # Install nacelle
                yield env.process(
                    install_nacelle(env, wtiv, nacelle, **kwargs)
                )

                reequip = {
                    "agent": wtiv.name,
                    "type": "Operations",
                    "location": "Site",
                    "duration": reequip_time,
                    "action": "CraneReequip",
                    **wtiv.operational_limits,
                }

                yield env.process(env.task_handler(reequip))

                # Install turbine blades
                for _ in range(3):
                    blade = yield env.process(
                        get_item_from_storage(
                            env=env,
                            vessel=queue.vessel,
                            item_type="Blade",
                            action_vessel=wtiv,
                            release=True,
                            **kwargs,
                        )
                    )

                    yield env.process(
                        install_turbine_blade(env, wtiv, blade, **kwargs)
                    )

                # Jack-down
                site_depth = kwargs.get("site_depth", None)
                extension = kwargs.get("extension", site_depth + 10)
                jackdown_time = wtiv.jacksys.jacking_time(
                    extension, site_depth
                )

                jackdown = {
                    "agent": wtiv.name,
                    "type": "Operations",
                    "location": "Site",
                    "duration": jackdown_time,
                    "action": "Jackdown",
                    **wtiv.transit_limits,
                }

                yield env.process(env.task_handler(jackdown))

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
        "Turbine installation complete!",
        extra={
            "agent": wtiv.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )
