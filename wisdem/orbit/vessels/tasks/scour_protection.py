"""
Jake Nunemaker
National Renewable Energy Lab
07/11/2019

This module contains vessel processes related to the installation of scour
protection.
"""


from ._defaults import defaults


def drop_rocks(**kwargs):
    """
    Returns the time it takes to drop rocks around a turbine.

    Parameters
    ----------
    drop_rocks_time : float
        Time required to drop rocks at site (h).

    Returns
    -------
    drop_rocks_time : float
        Time required to drop rocks at site (h).
    """

    key = "drop_rocks_time"
    drop_rocks_time = kwargs.get(key, defaults[key])

    return drop_rocks_time


def load_rocks(**kwargs):
    """
    Returns the time it takes to load rocks at port.

    Parameters
    ----------
    drop_rocks_time : float
        Time required to drop rocks at site (h).

    Returns
    -------
    load_rocks_time : float
        Time required to load rocks at port (h).
    """

    key = "load_rocks_time"
    load_rocks_time = kwargs.get(key, defaults[key])

    return load_rocks_time
