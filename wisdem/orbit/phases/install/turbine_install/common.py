"""Common processes and cargo types for Turbine Installations."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import process

from wisdem.orbit.core import Cargo
from wisdem.orbit.core.defaults import process_times as pt


class TowerSection(Cargo):
    """Tower Section Cargo"""

    def __init__(self, length=None, mass=None, deck_space=None, **kwargs):
        """
        Creates an instance of `TowerSection`.
        """

        self.length = length
        self.mass = mass
        self.deck_space = deck_space

    @staticmethod
    def fasten(**kwargs):
        """Returns time required to fasten a tower section at port."""

        key = "tower_section_fasten_time"
        time = kwargs.get(key, pt[key])

        return "Fasten Tower Section", time

    @staticmethod
    def release(**kwargs):
        """Returns time required to release tower section from fastenings."""

        key = "tower_section_release_time"
        time = kwargs.get(key, pt[key])

        return "Release Tower Section", time


class Nacelle(Cargo):
    """Nacelle Cargo"""

    def __init__(self, mass=None, deck_space=None, **kwargs):
        """
        Creates an instance of `Nacelle`.
        """

        self.mass = mass
        self.deck_space = deck_space

    @staticmethod
    def fasten(**kwargs):
        """Returns time required to fasten a nacelle at port."""

        key = "nacelle_fasten_time"
        time = kwargs.get(key, pt[key])

        return "Fasten Nacelle", time

    @staticmethod
    def release(**kwargs):
        """Returns time required to release nacelle from fastenings."""

        key = "nacelle_release_time"
        time = kwargs.get(key, pt[key])

        return "Release Nacelle", time


class Blade(Cargo):
    """Blade Cargo"""

    def __init__(self, length=None, mass=None, deck_space=None, **kwargs):
        """
        Creates an instance of `Blade`.
        """

        self.length = length
        self.mass = mass
        self.deck_space = deck_space

    @staticmethod
    def fasten(**kwargs):
        """Returns time required to fasten a blade at port."""

        key = "blade_fasten_time"
        time = kwargs.get(key, pt[key])

        return "Fasten Blade", time

    @staticmethod
    def release(**kwargs):
        """Returns time required to release blade from fastenings."""

        key = "blade_release_time"
        time = kwargs.get(key, pt[key])

        return "Release Blade", time


@process
def lift_nacelle(vessel, **kwargs):
    """
    Calculates time required to lift nacelle to hub height.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    hub_height : int | float
        Hub height above MSL (m).

    Yields
    ------
    vessel.task representing time to "Lift Nacelle"
    """

    hub_height = kwargs.get("hub_height", None)
    crane_rate = vessel.crane.crane_rate(**kwargs)
    lift_time = hub_height / crane_rate

    yield vessel.task_wrapper(
        "Lift Nacelle",
        lift_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def attach_nacelle(vessel, **kwargs):
    """
    Returns time required to attach nacelle to tower.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    nacelle_attach_time : int | float
        Time required to attach nacelle.

    Yields
    ------
    vessel.task representing time to "Attach Nacelle"
    """

    _ = vessel.crane
    key = "nacelle_attach_time"
    attach_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Attach Nacelle",
        attach_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def lift_turbine_blade(vessel, **kwargs):
    """
    Calculates time required to lift turbine blade to hub height.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    hub_height : int | float
        Hub height above MSL (m).

    Yields
    ------
    vessel.task representing time to "Lift Blade"
    """

    hub_height = kwargs.get("hub_height", None)
    crane_rate = vessel.crane.crane_rate(**kwargs)
    lift_time = hub_height / crane_rate

    yield vessel.task_wrapper(
        "Lift Blade",
        lift_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def attach_turbine_blade(vessel, **kwargs):
    """
    Returns time required to attach turbine blade to hub.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    blade_attach_time : int | float
        Time required to attach turbine blade.

    Yields
    ------
    vessel.task representing time to "Attach Blade"
    """

    _ = vessel.crane
    key = "blade_attach_time"
    attach_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Attach Blade",
        attach_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def lift_tower_section(vessel, height, **kwargs):
    """
    Calculates time required to lift tower section at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    height : int | float
        Height above MSL (m) required for lift.

    Yields
    ------
    vessel.task representing time to "Lift Tower Section"
    """

    crane_rate = vessel.crane.crane_rate(**kwargs)
    lift_time = height / crane_rate

    yield vessel.task_wrapper(
        "Lift Tower Section",
        lift_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def attach_tower_section(vessel, **kwargs):
    """
    Returns time required to attach tower section at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    section_attach_time : int | float
        Time required to attach tower section (h).

    Yields
    ------
    vessel.task representing time to "Attach Tower Section"
    """

    _ = vessel.crane
    key = "tower_section_attach_time"
    attach_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Attach Tower Section",
        attach_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def install_tower_section(vessel, section, height, **kwargs):
    """
    Process logic for installing a tower at site.

    Subprocesses:

    - Lift tower,  ``tasks.lift_tower()``
    - Attach tower, ``tasks.attach_tower()``

    Parameters
    ----------
    vessel : Vessel
    tower : dict
    """
    yield lift_tower_section(vessel, height, **kwargs)

    yield attach_tower_section(vessel, **kwargs)


@process
def install_nacelle(vessel, nacelle, **kwargs):
    """
    Process logic for installing a nacelle on a pre-installed tower.

    Subprocesses:

    - Lift nacelle, ``tasks.lift_nacelle()``
    - Attach nacelle, ``tasks.attach_nacelle()``

    Parameters
    ----------
    vessel : Vessel
    tower : dictå
    """

    yield lift_nacelle(vessel, **kwargs)

    yield attach_nacelle(vessel, **kwargs)


@process
def install_turbine_blade(vessel, blade, **kwargs):
    """
    Process logic for installing a turbine blade on a pre-installed tower and
    nacelle assembly.

    Subprocesses:

    - Lift blade, ``tasks.lift_turbine_blade()``
    - Attach blade, ``tasks.attach_turbine_blade()``

    Parameters
    ----------
    vessel : Vessel
    tower : dict
    """

    yield lift_turbine_blade(vessel, **kwargs)

    yield attach_turbine_blade(vessel, **kwargs)
