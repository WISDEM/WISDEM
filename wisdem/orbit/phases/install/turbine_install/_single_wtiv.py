"""Provides processes specific to single WTIV turbine installation."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.simulation.logic import (
    get_item_from_storage,
    prep_for_site_operations,
    get_list_of_items_from_port,
)
from wisdem.orbit.simulation.exceptions import ItemNotFound
from wisdem.orbit.phases.install.turbine_install._common import (
    install_tower,
    install_nacelle,
    install_turbine_blade,
)


def solo_install_turbines(env, vessel, port, distance, number, **kwargs):
    """
    Logic that a Wind Turbine Installation Vessel (WTIV) uses during a single
    turbine installation process.

    Parameters
    ----------
    env : simulation.Environment
        SimPy environment that the simulation runs in.
    vessel : vessels.Vessel
        Vessel object that represents the WTIV.
    distance : int | float
        Distance between port and site (km).
    number : int
        Total turbine component sets to install.
    """

    transit_time = vessel.transit_time(distance)
    reequip_time = vessel.crane.reequip(**kwargs)

    transit = {
        "agent": vessel.name,
        "location": "At Sea",
        "type": "Operations",
        "action": "Transit",
        "duration": transit_time,
        **vessel.transit_limits,
    }

    component_list = [
        ("type", "Tower"),
        ("type", "Nacelle"),
        ("type", "Blade"),
        ("type", "Blade"),
        ("type", "Blade"),
    ]

    n = 0
    while n < number:
        if vessel.at_port:
            try:
                # Get turbine components
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
                    prep_for_site_operations(env, vessel, **kwargs)
                )

                # Get tower
                tower = yield env.process(
                    get_item_from_storage(
                        env, vessel, item_type="Tower", **kwargs
                    )
                )

                # Install tower
                yield env.process(install_tower(env, vessel, tower, **kwargs))

                # Get turbine nacelle
                nacelle = yield env.process(
                    get_item_from_storage(
                        env, vessel, item_type="Nacelle", **kwargs
                    )
                )

                reequip = {
                    "agent": vessel.name,
                    "type": "Operations",
                    "location": "Site",
                    "duration": reequip_time,
                    "action": "CraneReequip",
                    **vessel.operational_limits,
                }

                yield env.process(env.task_handler(reequip))

                # Install nacelle
                yield env.process(
                    install_nacelle(env, vessel, nacelle, **kwargs)
                )

                # Install turbine blades
                for _ in range(3):
                    blade = yield env.process(
                        get_item_from_storage(
                            env, vessel, item_type="Blade", **kwargs
                        )
                    )

                    yield env.process(
                        install_turbine_blade(env, vessel, blade, **kwargs)
                    )

                # Jack-down
                site_depth = kwargs.get("site_depth", None)
                extension = kwargs.get("extension", site_depth + 10)
                jackdown_time = vessel.jacksys.jacking_time(
                    extension, site_depth
                )

                jackdown = {
                    "agent": vessel.name,
                    "type": "Operations",
                    "location": "Site",
                    "duration": jackdown_time,
                    "action": "Jackdown",
                    **vessel.transit_limits,
                }

                yield env.process(env.task_handler(jackdown))

                n += 1

            else:
                # Transit to port
                vessel.at_site = False
                yield env.process(env.task_handler(transit))
                vessel.at_port = True

    env.logger.debug(
        "Turbine installation complete!",
        extra={
            "agent": vessel.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )
