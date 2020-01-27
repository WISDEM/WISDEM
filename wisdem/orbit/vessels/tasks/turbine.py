"""
Jake Nunemaker
National Renewable Energy Lab
07/11/2019

This module contains vessel processes related to the installation of
turbines and turbine components
"""


from ._defaults import defaults
from ._exceptions import MissingComponent


def lift_nacelle(vessel, **kwargs):
    """
    Calculates time required to lift nacelle to hub height.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    hub_height : int | float
        Hub height above MSL (m).

    Returns
    -------
    nacelle_lift_time : float
        Time required to lift nacelle to hub height (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Lift Nacelle")

    hub_height = kwargs.get("hub_height", None)
    crane_rate = crane.crane_rate(**kwargs)
    nacelle_lift_time = hub_height / crane_rate

    return nacelle_lift_time


def attach_nacelle(vessel, **kwargs):
    """
    Returns time required to attach nacelle to tower.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    nacelle_attach_time : int | float
        Time required to attach nacelle.

    Returns
    -------
    nacelle_attach_time : float
        Time required to attach nacelle (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Attach Nacelle")

    key = "nacelle_attach_time"
    nacelle_attach_time = kwargs.get(key, defaults[key])

    return nacelle_attach_time


def fasten_nacelle(**kwargs):
    """
    Returns time required to fasten a nacelle at port.

    Parameters
    ----------
    nacelle_fasten_time : int | float
        Time required to fasten a nacelle.

    Returns
    -------
    nacelle_fasten_time : float
        Time required to fasten nacelle (h).
    """

    key = "nacelle_fasten_time"
    nacelle_fasten_time = kwargs.get(key, defaults[key])

    return nacelle_fasten_time


def release_nacelle(**kwargs):
    """
    Returns time required to release nacelle from fastenings.

    Parameters
    ----------
    nacelle_release_time : int | float
        Time required to release nacelle.

    Returns
    -------
    nacelle_release_time : float
        Time required to release nacelle (h).
    """

    key = "nacelle_release_time"
    nacelle_release_time = kwargs.get(key, defaults[key])

    return nacelle_release_time


def lift_turbine_blade(vessel, **kwargs):
    """
    Calculates time required to lift turbine blade to hub height.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    hub_height : int | float
        Hub height above MSL (m).

    Returns
    -------
    blade_lift_time : float
        Time required to lift blade to hub height (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Lift Blade")

    hub_height = kwargs.get("hub_height", None)
    crane_rate = crane.crane_rate(**kwargs)
    blade_lift_time = hub_height / crane_rate

    return blade_lift_time


def attach_turbine_blade(vessel, **kwargs):
    """
    Returns time required to attach turbine blade to hub.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    blade_attach_time : int | float
        Time required to attach turbine blade.

    Returns
    -------
    blade_attach_time : float
        Time required to attach turbine blade (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Attach Blade")

    key = "blade_attach_time"
    blade_attach_time = kwargs.get(key, defaults[key])

    return blade_attach_time


def fasten_turbine_blade(**kwargs):
    """
    Returns time required to fasten a blade at port.

    Parameters
    ----------
    blade_fasten_time : int | float
        Time required to fasten a blade at port.

    Returns
    -------
    blade_fasten_time : float
        Time required to fasten blade (h).
    """

    key = "blade_fasten_time"
    blade_fasten_time = kwargs.get(key, defaults[key])

    return blade_fasten_time


def release_turbine_blade(**kwargs):
    """
    Returns time required to release turbine blade from fastening.

    Parameters
    ----------
    blade_release_time : int | float
        Time required to release turbine blade.

    Returns
    -------
    blade_release_time : float
        Time required to release turbine blade (h).
    """

    key = "blade_release_time"
    blade_release_time = kwargs.get(key, defaults[key])

    return blade_release_time


def lift_tower_section(vessel, height, **kwargs):
    """
    Calculates time required to lift tower section at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    height : int | float
        Height above MSL (m) required for lift.

    Returns
    -------
    section_lift_time : float
        Time required to lift tower section (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Lift Tower Section")

    crane_rate = crane.crane_rate(**kwargs)
    section_lift_time = height / crane_rate

    return section_lift_time


def attach_tower_section(vessel, **kwargs):
    """
    Returns time required to attach tower section at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    section_attach_time : int | float
        Time required to attach tower section (h).

    Returns
    -------
    section_attach_time : float
        Time required to attach tower section (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Attach Tower Section")

    key = "tower_section_attach_time"
    section_attach_time = kwargs.get(key, defaults[key])

    return section_attach_time


def fasten_tower_section(**kwargs):
    """
    Returns time required to fasten a tower section at port.

    Parameters
    ----------
    section_fasten_time : int | float
        Time required to fasten a tower section (h).

    Returns
    -------
    section_fasten_time : float
        Time required to fasten tower section (h).
    """

    key = "tower_section_fasten_time"
    section_fasten_time = kwargs.get(key, defaults[key])

    return section_fasten_time


def release_tower_section(**kwargs):
    """
    Returns time required to release tower section from fastenings.

    Parameters
    ----------
    tower_section_release_time : int | float
        Time required to release tower section (h).

    Returns
    -------
    section_release_time : float
        Time required to release tower section (h).
    """

    key = "tower_section_release_time"
    tower_release_time = kwargs.get(key, defaults[key])

    return tower_release_time
