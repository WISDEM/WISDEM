"""Provides the cable laying subprocess functions."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__credits__ = ["Jake Nunemaker", "Matt Shields"]
# __version__ = "0.0.1"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"
__status__ = "Development"


from ._defaults import defaults


def dig_trench(distance, **kwargs):
    """
    Calculates time required to dig the trench from the transmission
    cable traveling from the landfall site to the onshore substation.

    Parameters
    ----------
    distance : int or float
        Distance between landfall and the offshore substation.
    trench_dig_speed : int or float
        Trench digging speed, km/h.
    Returns
    -------
    trench_dig_time : int or float
        Time required to dig the onshore trench, h.
    """

    key = "trench_dig_speed"
    trench_dig_speed = kwargs.get(key, defaults[key])
    trench_dig_time = distance / trench_dig_speed
    return trench_dig_time


def lift_carousel(**kwargs):
    """
    Returns time required to lift a cable carousel at quayside.

    Parameters
    ----------
    carousel_lift_time : int | float
        Time required to lift a carousel at quayside.

    Returns
    -------
    carousel_lift_time : float
        Time required to lift carousel (h).
    """

    key = "carousel_lift_time"
    carousel_lift_time = kwargs.get(key, defaults[key])

    return carousel_lift_time


def fasten_carousel(**kwargs):
    """
    Returns time required to fasten a cable carousel at quayside.

    Parameters
    ----------
    carousel_fasten_time : int | float
        Time required to fasten a carousel at quayside.

    Returns
    -------
    carousel_fasten_time : float
        Time required to fasten carousel (h).
    """

    key = "carousel_fasten_time"
    carousel_fasten_time = kwargs.get(key, defaults[key])

    return carousel_fasten_time


def prep_cable(**kwargs):
    """
    Calculates time required to prepare cable for pull-in.

    Parameters
    ----------
    cable_prep_time : int | float
        Time required to prepare cable for pull-in.
        Default: 1h

    Returns
    -------
    cable_prep_time : float
        Time required to prepare cable for pull-in (h).
    """

    key = "cable_prep_time"
    cable_prep_time = kwargs.get(key, defaults[key])

    return cable_prep_time


def lower_cable(**kwargs):
    """
    Calculates time required to lower cable to seafloor

    Parameters
    ----------
    cable_lower_time : int | float
        Time required to lower cable to seafloor.
        Default: 1h

    Returns
    -------
    cable_lower_time : float
        Time required to lower cable to seafloor (h).
    """

    key = "cable_lower_time"
    cable_lower_time = kwargs.get(key, defaults[key])

    return cable_lower_time


def pull_in_cable(**kwargs):
    """
    Calculates time required to pull cable into offshore substructure.

    Parameters
    ----------
    cable_pull_in_time : int | float
        Time required to pull cable into offshore substructure.
        Default: 5.5h

    Returns
    -------
    cable_pull_in_time : float
        Time required to pull cable into offshore substructure (h).
    """

    key = "cable_pull_in_time"
    cable_pull_in_time = kwargs.get(key, defaults[key])

    return cable_pull_in_time


def test_cable(**kwargs):
    """
    Calculates time required to terminate and test cable connection with
    substructure.

    Parameters
    ----------
    cable_termination_time : int | float
        Time required to terminate and test cable connection with
        substructure.
        Default: 5.5h

    Returns
    -------
    cable_termination_time : float
        Time required to terminate and test cable connection with
        substructure (h).
    """

    key = "cable_termination_time"
    cable_termination_time = kwargs.get(key, defaults[key])

    return cable_termination_time


def lay_bury_cable(distance, **kwargs):
    """
    Calculates time required to lay and bury cable between substructures.

    Parameters
    ----------
    distance : int | float
        Distance between substructures (km)
    cable_lay_bury_speed : int | float
        Maximum speed at which cable is dispensed (km/hr)

    Returns
    -------
    lay_cable_time : float
        Time required to lay cable between adjacent substructures (h).
    """

    key = "cable_lay_bury_speed"
    cable_lay_bury_speed = kwargs.get(key, defaults[key])
    lay_bury_cable_time = distance / cable_lay_bury_speed

    return lay_bury_cable_time


def lay_cable(distance, **kwargs):
    """
    Calculates time required to lay cable between substructures.

    Parameters
    ----------
    distance : int | float
        Distance between substructures (km)
    cable_lay_speed : int | float
        Maximum speed at which cable is dispensed (km/hr)

    Returns
    -------
    lay_cable_time : float
        Time required to lay cable between adjacent substructures (h).
    """

    key = "cable_lay_speed"
    cable_lay_speed = kwargs.get(key, defaults[key])
    lay_cable_time = distance / cable_lay_speed

    return lay_cable_time


def bury_cable(distance, **kwargs):
    """
    Calculates time required to bury a cable between substructures.

    Parameters
    ----------
    distance : int | float
        Distance between substructures (km)
    cable_bury_speed : int | float
        Maximum speed at which cable is burried (km/hr)

    Returns
    -------
    bury_cable_time : float
        Time required to bury cable between adjacent substructures (h).
    """

    key = "cable_bury_speed"
    cable_bury_speed = kwargs.get(key, defaults[key])
    bury_cable_time = distance / cable_bury_speed

    return bury_cable_time


def splice_cable(**kwargs):
    """
    Calculates time required to lay cable between substructures.
    Assumes simultaneous lay/burial.

    Parameters
    ----------
    cable_splice_time : int | float
        Time required to splice two cable ends together (h).

    Returns
    -------
    cable_splice_time : float
        Time required to splice two cable ends together (h).
    """

    key = "cable_splice_time"
    cable_splice_time = kwargs.get(key, defaults[key])

    return cable_splice_time


def raise_cable(**kwargs):
    """
    Calculates time required to raise the unspliced cable from the seafloor.

    Parameters
    ----------
    cable_raise_time : int | float
        Time required to raise the cable from the seafloor (h).

    Returns
    -------
    cable_raise_time : float
        Time required to raise the cable from the seafloor (h).
    """

    key = "cable_raise_time"
    cable_raise_time = kwargs.get(key, defaults[key])

    return cable_raise_time


def dig_trench(distance, **kwargs):
    """
    Calculates time required to dig the trench from the transmission
    cable traveling from the landfall site to the onshore substation.

    Parameters
    ----------
    distance : int | float
        Distance between landfall and the offshore substation.
    trench_dig_speed : int | float
        Trench digging speed (km/h).

    Returns
    -------
    trench_dig_time : float
        Time required to dig the onshore trench (h).
    """

    key = "trench_dig_speed"
    trench_dig_speed = kwargs.get(key, defaults[key])
    trench_dig_time = distance / trench_dig_speed

    return trench_dig_time


def tow_plow(distance, **kwargs):
    """
    Calculates time required to tow the plow to the landfall site.

    Parameters
    ----------
    distance : int | float
        Distance to between cable laying vessel and landfall (km).
    tow_plow_speed : float
        Towing speed (km/h).

    Returns
    -------
    tow_plow_time : float
        Time required to the plow to landfall site (h).
    """

    key = "tow_plow_speed"
    tow_plow_speed = kwargs.get(key, defaults[key])
    tow_plow_time = distance / tow_plow_speed

    return tow_plow_time


def pull_winch(distance, **kwargs):
    """
    Calculates time required to pull the cable onshore and into the previously
    dug trench.

    Parameters
    ----------
    distance: int | float
        Distance winch wire must travel to reach the plow.
    pull_winch_speed : int | float
        Speed at wich the winch travels (km/h).

    Returns
    -------
    pull_winch_time : float
        Time required to pull in the winch (h).
    """

    key = "pull_winch_speed"
    pull_winch_speed = kwargs.get(key, defaults[key])
    pull_winch_time = distance / pull_winch_speed

    return pull_winch_time
