"""
Jake Nunemaker
National Renewable Energy Lab
07/11/2019

This module contains vessel processes related to the installation of offshore
substations.
"""


from ._defaults import defaults
from ._exceptions import MissingComponent


def lift_topside(vessel, **kwargs):
    """
    Calculates time required to lift topside at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.

    Returns
    -------
    topside_lift_time : float
        Time required to lift topside (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Lift Topside")

    lift_height = 5  # small lift just to clear the deck
    crane_rate = crane.crane_rate(**kwargs)
    topside_lift_time = lift_height / crane_rate

    return topside_lift_time


def attach_topside(vessel, **kwargs):
    """
    Returns time required to attach topside at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    topside_attach_time : int | float
        Time required to attach topside.

    Returns
    -------
    topside_attach_time : float
        Time required to attach topside (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Attach Topside")

    key = "topside_attach_time"
    topside_attach_time = kwargs.get(key, defaults[key])

    return topside_attach_time


def fasten_topside(**kwargs):
    """
    Returns time required to fasten topside at port.

    Parameters
    ----------
    topside_fasten_time : int | float
        Time required to fasten topside.

    Returns
    -------
    topside_fasten_time : float
        Time required to fasten topside (h).
    """

    key = "topside_fasten_time"
    topside_fasten_time = kwargs.get(key, defaults[key])

    return topside_fasten_time


def release_topside(**kwargs):
    """
    Returns time required to release topside from fastening.

    Parameters
    ----------
    topside_release_time : int | float
        Time required to release topside.

    Returns
    -------
    topside_release_time : float
        Time required to release topside (h).
    """

    key = "topside_release_time"
    topside_release_time = kwargs.get(key, defaults[key])

    return topside_release_time
