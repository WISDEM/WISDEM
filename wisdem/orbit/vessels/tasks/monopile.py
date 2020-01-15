"""
Jake Nunemaker
National Renewable Energy Lab
07/11/2019

This module contains vessel processes related to the installation of
monopile substructures.
"""


from ._defaults import defaults
from ._exceptions import MissingComponent


def upend_monopile(vessel, length, **kwargs):
    """
    Calculates time required to upend monopile to vertical position.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    length : int | float
        Overall length of monopile (m).

    Returns
    -------
    mono_upend_time : float
        Time required to upened monopile (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Upend Monopile")

    crane_rate = crane.crane_rate(**kwargs)
    mono_upend_time = length / crane_rate

    return mono_upend_time


def lower_monopile(vessel, **kwargs):
    """
    Calculates time required to lower monopile to seafloor.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    site_depth : int | float
        Seafloor depth at site (m).

    Returns
    -------
    mono_lower_time : float
        Time required to lower monopile (h).
    """

    crane = getattr(vessel, "crane", None)
    jacksys = getattr(vessel, "jacksys", None)

    missing = [c for c in [crane, jacksys] if c is None]
    if missing:
        raise MissingComponent(vessel, missing, "Lower Monopile")

    depth = kwargs.get("site_depth", None)
    rate = crane.crane_rate(**kwargs)

    height = (jacksys.air_gap + jacksys.leg_pen + depth) / rate
    mono_lower_time = height / rate

    return mono_lower_time


def drive_monopile(vessel, **kwargs):
    """
    Calculates time required to drive monopile into seafloor.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    mono_embed_len : int | float
        Monopile embedment length below seafloor (m).
    mono_drive_rate : int | float
        Driving rate (m/hr).

    Returns
    -------
    drive_time : float
        Time required to drive monopile to 'drive_length' (h).
    """

    crane = getattr(vessel, "crane", None)
    if crane is None:
        raise MissingComponent(vessel, "crane", "Drive Monopile")

    mono_embed_len = kwargs.get("mono_embed_len", defaults["mono_embed_len"])
    mono_drive_rate = kwargs.get(
        "mono_drive_rate", defaults["mono_drive_rate"]
    )

    drive_time = mono_embed_len / mono_drive_rate

    return drive_time


def lower_transition_piece(vessel, **kwargs):
    """
    Calculates time required to lower a transition piece onto monopile.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.

    Returns
    -------
    tp_lower_time : float
        Time required to lower transition piece.
    """

    crane = getattr(vessel, "crane", None)
    jacksys = getattr(vessel, "jacksys", None)

    missing = [c for c in [crane, jacksys] if c is None]
    if missing:
        raise MissingComponent(vessel, missing, "Lower Transition Piece")

    rate = crane.crane_rate(**kwargs)
    tp_lower_time = jacksys.air_gap / rate

    return tp_lower_time


def bolt_transition_piece(**kwargs):
    """
    Returns time required to bolt transition piece to monopile.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    tp_bolt_time : int | float
        Time required to attach transition piece.

    Returns
    -------
    tp_bolt_time : float
        Time required to attach transition piece (h).
    """

    key = "tp_bolt_time"
    tp_bolt_time = kwargs.get(key, defaults[key])

    return tp_bolt_time


def pump_transition_piece_grout(**kwargs):
    """
    Returns time required to pump grout at the transition piece interface.

    Parameters
    ----------
    grout_pump_time : int | float
        Time required to pump grout at the interface.

    Returns
    -------
    grout_pump_time : float
        Time required to pump grout at the interface (h).
    """

    key = "grout_pump_time"
    grout_pump_time = kwargs.get(key, defaults[key])

    return grout_pump_time


def cure_transition_piece_grout(**kwargs):
    """
    Returns time required for the transition piece grout to cure.

    Parameters
    ----------
    grout_cure_time : int | float
        Time required for the grout to cure.

    Returns
    -------
    grout_cure_time : float
        Time required for the grout to cure (h).
    """

    key = "grout_cure_time"
    grout_cure_time = kwargs.get(key, defaults[key])

    return grout_cure_time


def fasten_monopile(**kwargs):
    """
    Returns time required to fasten a monopile at quayside.

    Parameters
    ----------
    mono_fasten_time : int | float
        Time required to fasten a monopile at quayside.

    Returns
    -------
    mono_fasten_time : float
        Time required to fasten monopile (h).
    """

    key = "mono_fasten_time"
    mono_fasten_time = kwargs.get(key, defaults[key])

    return mono_fasten_time


def release_monopile(**kwargs):
    """
    Returns time required to release monopile from fastening.

    Parameters
    ----------
    mono_release_time : int | float
        Time required to release monopile.

    Returns
    -------
    mono_release_time : float
        Time required to release monopile (h).
    """

    key = "mono_release_time"
    mono_release_time = kwargs.get(key, defaults[key])

    return mono_release_time


def fasten_transition_piece(**kwargs):
    """
    Returns time required to fasten a transition piece at quayside.

    Parameters
    ----------
    tp_fasten_time : int | float
        Time required to fasten a transition piece at quayside.

    Returns
    -------
    tp_fasten_time : float
        Time required to fasten transition piece (h).
    """

    key = "tp_fasten_time"
    tp_fasten_time = kwargs.get(key, defaults[key])

    return tp_fasten_time


def release_transition_piece(**kwargs):
    """
    Returns time required to release transition piece from fastening.

    Parameters
    ----------
    tp_release_time : int | float
        Time required to release transition piece.

    Returns
    -------
    tp_release_time : float
        Time required to release transition piece (h).
    """

    key = "tp_release_time"
    tp_release_time = kwargs.get(key, defaults[key])

    return tp_release_time
