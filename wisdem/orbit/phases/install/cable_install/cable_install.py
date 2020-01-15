"""
Provides the `ArrayCableInstallation` and `ExportCableInstallation` classes.
"""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import itertools
from types import SimpleNamespace

from wisdem.orbit.vessels import Vessel
from wisdem.orbit.simulation import Environment, VesselStorageContainer
from wisdem.orbit.phases.install import InstallPhase
from wisdem.orbit.vessels.components import Carousel, CarouselSystem

from .process import bury_cables, lay_array_cables, lay_export_cables

STRATEGY_MAP = {
    "array": {
        "lay_bury": lay_array_cables,
        "lay": lay_array_cables,
        "bury": bury_cables,
    },
    "export": {
        "lay_bury": lay_export_cables,
        "lay": lay_export_cables,
        "bury": bury_cables,
    },
}


def extract_cable_bury_speeds(cables):
    """
    Extracts the cable lengths and bury speeds required for a separate burial
    strategy.

    Parameters
    ----------
    cables : dict
        Dictionary of cable types and their cable sections.

    Returns
    -------
    length_speed : None or list
        If custom cable bury speeds were not included in `cable_sections`
        return `None`, otherwise return a list of the tuple pairs of section
        lengths and burial speeds.
    """

    length_speed = list(
        itertools.chain.from_iterable(
            [[length, *speed]] * n
            for cable in cables.values()
            for length, *speed, n in cable["cable_sections"]
        )
    )
    mass = [0] * len(length_speed)

    if len(length_speed[0]) == 1:
        length = [el[0] for el in length_speed]
        speed = [-1] * len(length)
        carousel = Carousel("bury", sum(length), 0, length, mass, speed, 1)

        return carousel

    length = []
    speed = []
    for l, s in length_speed:
        length.append(l)
        speed.append(s)

    carousel = Carousel("bury", sum(length), 0, length, mass, speed, 1)
    return carousel


class CableInstallation(InstallPhase):
    """
    Single vessel cable installation simulation.

    Attributes
    ----------
    config : dict
        Configuration dictionary.
    defaults : dict
        Project defaults dictionary from library.
    env : Environment
        Simulation environment.
    x
    """

    def __init__(self, config, system, weather, **kwargs):
        """
        Creates an instance of SingleVesselArrayCableInstallation

        Parameters
        ----------
        config : dict
            A configuration dictionary matching the `expected_config` template.
        system : str
            Indicates the if an array or export cable is being installed. Takes one of "array" or "export".
        weather : str
            <pathname>/<filename> to weather profile used to determine weather
            delays.
        """

        self._system = system.lower()
        config = self.initialize_library(config, **kwargs)
        self.config = self.validate_config(config)
        self.strategy = config[f"{self._system}_system"]["strategy"].lower()
        if self.strategy not in ("lay", "bury", "lay_bury"):
            raise ValueError(
                f'{self._system.title()} system strategy must be one of "lay", "bury", or "lay_bury".'
            )

        self.extract_phase_kwargs(**kwargs)
        self.extract_defaults()
        self.env = Environment(weather)
        self.init_logger(**kwargs)
        self.setup_simulation(**kwargs)

    def initialize_cable_installation_vessel(self):
        """
        Creates a cable installation vessel.
        """

        # Get vessel specs
        try:
            cable_lay_specs = self.config[f"{self._system}_cable_lay_vessel"]
        except KeyError:
            raise Exception(f'"{self._system}_cable_lay_vessel" is undefined.')

        # Vessel name and costs
        name = f"{self._system.title()} Cable Installation Vessel"
        cost = cable_lay_specs["vessel_specs"].get(
            "day_rate", self.defaults[f"{self._system}_cable_install_day_rate"]
        )
        self.agent_costs[name] = cost

        # Vessel storage
        try:
            storage_specs = cable_lay_specs["storage_specs"]
        except KeyError:
            raise Exception(f"Storage specifications must be set for {name}")

        self.cable_lay_vessel = Vessel(name, cable_lay_specs)
        self.cable_lay_vessel.storage = VesselStorageContainer(
            self.env, **storage_specs
        )

        # Vessel starting location
        self.cable_lay_vessel.at_port = True
        self.cable_lay_vessel.at_site = False

    def initialize_carousels(self):
        """
        Creates the cable carousel(s) required for installation.
        """

        self.carousels = CarouselSystem(
            self.config[f"{self._system}_system"]["cables"],
            self.cable_lay_vessel.max_cargo,
        )
        self.carousels.create_carousels()

        self.num_sections = 0
        for carousel in list(self.carousels.carousels.values())[::-1]:
            self.num_sections += len(carousel.section_lengths)
            c = carousel.__dict__
            c["type"] = "Carousel"
            self.port.put(c)

    @property
    def detailed_output(self):
        """Returns detailed outputs."""

        outputs = {
            **self.agent_efficiencies,
            **self.get_max_cargo_weight_utilzations([self.cable_lay_vessel]),
        }

        return outputs


class ArrayCableInstallation(CableInstallation):

    expected_config = {
        "port": {
            "num_cranes": "int",
            "monthly_rate": "int | float (optional)",
        },
        "array_cable_lay_vessel": "str",
        "site": {"distance": "int | float"},
        "plant": {"num_turbines": "int"},
        "array_system": {
            "strategy": "str",
            "cables": {
                "name (variable)": {
                    "linear_density": "int | float",
                    "cable_sections": [("float", "float (optional)", "int")],
                }
            },
        },
    }

    def __init__(self, config, weather=None, **kwargs):
        """`
        Initializes the `ArrayCableInstallation` class.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        strategy : str, optional
            [description], by default "lay_bury"
        weather : [type], optional
            [description], by default None
        """

        super().__init__(config, "array", weather, **kwargs)

    def setup_simulation(self, **kwargs):
        """
        Sets up the required simulation infrastructures.
            - creates a port
            - initializes a cable installation vessel
            - initializes the cable carousel(s)
            - initializes vessel storage
        """

        self.initialize_port()
        self.initialize_cable_installation_vessel()
        if self.strategy == "bury":
            self.cable_lay_vessel.carousel = extract_cable_bury_speeds(
                self.config["array_system"]["cables"]
            )
            self.num_sections = len(
                self.cable_lay_vessel.carousel.section_lengths
            )
        else:
            self.initialize_carousels()

        self.env.process(
            STRATEGY_MAP["array"][self.strategy](
                env=self.env,
                vessel=self.cable_lay_vessel,
                port=self.port,
                num_cables=self.num_sections,
                distance_to_site=self.config["site"]["distance"],
                strategy=self.strategy,
                system=self._system,
                **kwargs,
            )
        )


class ExportCableInstallation(CableInstallation):

    expected_config = {
        "port": {
            "num_cranes": "int",
            "monthly_rate": "int | float (optional)",
        },
        "export_cable_lay_vessel": "str",
        "trench_dig_vessel": "str | dict",
        "site": {
            "distance": "int",
            "distance_to_landfall": "float",
            "distance_to_beach": "float",
            "distance_to_interconnection": "float",
        },
        "export_system": {
            "strategy": "str",
            "cables": {
                "name (variable)": {
                    "linear_density": "int | float",
                    "cable_sections": [("float", "float (optional)", "int")],
                }
            },
        },
    }

    def __init__(self, config, weather=None, **kwargs):

        super().__init__(config, "export", weather, **kwargs)

    def intialize_trench_dig_vessel(self):
        """
        Creates a trench digging "vessel".
        .. note:: It is important to note that this is not a vessel, but a
        multifaceted onshore operation but for the sake of naming consistency
        we will call this a vessel.
        """

        # Get the vessel specs
        try:
            trench_dig_specs = self.config["trench_dig_vessel"]
        except KeyError:
            raise Exception('"trench_dig_vessel" is undefined.')

        # Vessel name and costs
        name = "Trench Dig Vessel"
        cost = trench_dig_specs["vessel_specs"].get(
            "day_rate", self.defaults["trench_day_rate"]
        )
        self.agent_costs[name] = cost
        self.trench_dig_vessel = Vessel(name, trench_dig_specs)

    def setup_simulation(self, **kwargs):
        """
        Sets up the required simulation infrastructures.
            - creates a port
            - initializes a cable installation vessel
            - initializes the cable carousel(s)
            - initializes vessel storage
        """

        self.initialize_port()
        self.initialize_cable_installation_vessel()
        self.intialize_trench_dig_vessel()

        if self.strategy == "bury":
            self.cable_lay_vessel.carousel = extract_cable_bury_speeds(
                self.config["export_system"]["cables"]
            )
            self.num_sections = len(
                self.cable_lay_vessel.carousel.section_lengths
            )
        else:
            self.initialize_carousels()

        cable = self.config["export_system"]["cables"][
            [*self.config["export_system"]["cables"]][0]
        ]
        self.cable_length = cable["cable_sections"][0][0]

        self.distances = SimpleNamespace(
            **{
                el.replace("distance_to_", "")
                if el != "distance"
                else "site": self.config["site"].get(el, None)
                for el in (
                    "distance",
                    "distance_to_beach",
                    "distance_to_landfall",
                    "distance_to_interconnection",
                )
            }
        )

        self.env.process(
            STRATEGY_MAP["export"][self.strategy](
                env=self.env,
                vessel=self.cable_lay_vessel,
                trench_vessel=self.trench_dig_vessel,
                port=self.port,
                cable_length=self.cable_length,
                num_sections=self.num_sections,
                distances=self.distances,
                strategy=self.strategy,
                system=self._system,
                distance_to_site=self.distances.site,
                num_cables=self.num_sections,
                **kwargs,
            )
            # env=self.env,
            # vessel=self.cable_lay_vessel,
            # port=self.port,
            # num_cables=self.num_sections,
            # distance_to_site=self.config["site"]["distance"],
            # strategy=self.strategy,
            # system=self._system,
            # **kwargs,
        )
