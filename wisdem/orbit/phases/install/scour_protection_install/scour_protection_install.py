"""Provides the `ScourProtectionInstallation` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


from math import ceil

from wisdem.orbit.vessels import Vessel
from wisdem.orbit.simulation import Environment, VesselStorageContainer
from wisdem.orbit.phases.install import InstallPhase

from .process import (
    transport,
    get_scour_protection_from_port,
    install_scour_protection_at_site,
)


class ScourProtectionInstallation(InstallPhase):
    """Scour protection installation simulation using a single vessel."""

    #:
    expected_config = {
        "scour_protection_install_vessel": "dict | str",
        "site": {"distance": "int"},
        "plant": {
            "num_turbines": "int",
            "turbine_spacing": "int",
            "turbine_distance": "float (optional)",
        },
        "turbine": {"rotor_diameter": "int"},
        "port": {
            "num_cranes": "int",
            "monthly_rate": "float (optional)",
            "name": "str (optional)",
        },
        "scour_protection": {"tonnes_per_substructure": "int"},
    }

    phase = "Scour Protection Installation"

    def __init__(self, config, weather=None, **kwargs):
        """
        Creates an instance of WtivSim.

        Parameters
        ----------
        config : dict
            User-defined input dictionary for modeling.
        weather : str, default None
            File and path to file for a weather profile used to determine
            weather delays. If `None`, then there are no weather delays.
        kwargs : dict
            Optional user-defined inputs.
        """

        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self.scour_protection_tonnes_to_install = ceil(
            self.config["scour_protection"]["tonnes_per_substructure"]
        )

        self.extract_phase_kwargs(**kwargs)
        self.extract_defaults()

        self.env = Environment(weather)
        self.init_logger(**kwargs)
        self.setup_simulation(**kwargs)

    def setup_simulation(self, **kwargs):
        """
        Sets up the required simulation infrastructure:
            - creates a port
            - initializes a scour protection installation vessel
            - initializes vessel storage
        """

        self.initialize_port()
        self.initialize_scour_protection()
        self.initialize_scour_vessel()

        site_distance = self.config["site"]["distance"]
        rotor_diameter = self.config["turbine"]["rotor_diameter"]
        turbine_distance = self.config["plant"].get("turbine_distance", None)
        if turbine_distance is None:
            turbine_distance = (
                rotor_diameter
                * self.config["plant"]["turbine_spacing"]
                / 1000.0
            )

        install_specs = (
            site_distance,
            turbine_distance,
            self.num_turbines,
            self.scour_protection_tonnes_to_install,
        )

        self.env.process(
            install_scour_protection(
                env=self.env,
                vessel=self.scour_vessel,
                port=self.port,
                install_specs=install_specs,
                **kwargs,
            )
        )

    def initialize_scour_vessel(self):
        """
        Creates a scouring protection vessel.
        """

        # Get vessel specs
        try:
            scour_protection_specs = self.config[
                "scour_protection_install_vessel"
            ]
        except KeyError:
            raise Exception("`scour_protection_install_vessel` is undefined.")

        # Vessel name and costs
        name = scour_protection_specs.get(
            "name", "Scour Protection Install Vessel"
        )
        cost = scour_protection_specs["vessel_specs"].get(
            "day_rate", self.defaults["scour_day_rate"]
        )
        self.agent_costs[name] = cost

        # Vessel storage
        try:
            storage_specs = scour_protection_specs["storage_specs"]
        except KeyError:
            raise Exception(
                "Storage specifications must be set for the scour protection "
                "installation vessel."
            )

        self.scour_vessel = Vessel(name, scour_protection_specs)
        self.scour_vessel.storage = VesselStorageContainer(
            self.env, **storage_specs
        )

        # Vessel starting location
        self.scour_vessel.at_port = True
        self.scour_vessel.at_site = False

    def initialize_scour_protection(self):
        """
        Initialize the scour protection at port.
        """

        scour_protection = {
            "type": "Scour Protection",
            "weight": self.scour_protection_tonnes_to_install,
            "deck_space": 0,
        }

        self.num_turbines = self.config["plant"]["num_turbines"]
        for _ in range(self.num_turbines):
            self.port.put(scour_protection)

        self.logger.debug(
            "SCOUR PROTECTION INITIALIZED",
            extra={"time": self.env.now, "agent": "Director"},
        )

    @property
    def detailed_output(self):
        """Returns detailed outputs."""

        outputs = {
            **self.agent_efficiencies,
            **self.get_max_cargo_weight_utilzations([self.scour_vessel]),
            **self.get_max_deck_space_utilzations([self.scour_vessel]),
        }

        return outputs


def install_scour_protection(env, vessel, port, install_specs, **kwargs):
    """
    Installs the scour protection. Processes the traveling between site
    and turbines for when there are enough rocks leftover from a previous
    installation as well as the weight of rocks available.

    Parameters
    ----------
    env : simpy.Environment
        Simulation environment.
    port : simpy.FilterStore
        Port simulation object.
    port_to_site_distance : int | float
        Distance (km) between site and the port.
    turbine_to_turbine_distance : int | float
        Distance between any two turbines.
        For now this assumes it traverses an edge and not a diagonal.
    turbines_to_install : int
        Number of turbines where scouring protection must be installed.
    rock_tonnes_per_sub : int
        Number of tonnes required to be installed at each substation
    """

    (
        port_to_site_distance,
        turbine_to_turbine_distance,
        turbines_to_install,
        rock_tonnes_per_sub,
    ) = install_specs

    turbine_to_turbine = vessel.transit_time(turbine_to_turbine_distance)

    _shared = {
        "agent": vessel.name,
        "action": "Transit",
        "type": "Operations",
        **vessel.transit_limits,
    }

    between_turbines = {
        **_shared,
        "duration": turbine_to_turbine,
        "location": "Site",
    }

    while turbines_to_install > 0:
        if vessel.at_port:
            # Load rocks
            yield env.process(
                get_scour_protection_from_port(env, vessel, port, **kwargs)
            )

            vessel.update_trip_data(items=False)
            # Travel to site
            yield env.process(
                transport(
                    env, vessel, port_to_site_distance, False, True, **kwargs
                )
            )

        elif vessel.at_site:
            if vessel.storage.current_cargo_weight >= rock_tonnes_per_sub:
                # Install scour protection
                yield env.process(
                    install_scour_protection_at_site(
                        env, vessel, rock_tonnes_per_sub, **kwargs
                    )
                )
                turbines_to_install -= 1

                # Transit to another turbine
                if vessel.storage.current_cargo_weight >= rock_tonnes_per_sub:
                    yield env.process(env.task_handler(between_turbines))

                # Transit back to port
                else:
                    yield env.process(
                        transport(
                            env,
                            vessel,
                            port_to_site_distance,
                            True,
                            False,
                            **kwargs,
                        )
                    )
            else:
                yield env.process(
                    transport(
                        env,
                        vessel,
                        port_to_site_distance,
                        True,
                        False,
                        **kwargs,
                    )
                )

        else:
            raise Exception("Vessel is lost at sea.")

    env.logger.debug(
        "Scour protection installation complete!",
        extra={
            "agent": vessel.name,
            "time": env.now,
            "type": "Status",
            "action": "Complete",
        },
    )
