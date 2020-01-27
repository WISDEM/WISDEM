"""
Jake Nunemaker
National Renewable Energy Lab
07/11/2019

This module contains miscellaneous vessel processes.
"""


from ._defaults import defaults


def position_onsite(**kwargs):
    """
    Calculates time required to position vessel onsite.

    Parameters
    ----------
    site_position_time : int | float
        Time required to position vessel onsite.

    Returns
    -------
    site_position_time : float
        Time required to position vessel onsite (h).
    """

    key = "site_position_time"
    site_position_time = kwargs.get(key, defaults[key])

    return site_position_time


def rov_survey(**kwargs):
    """
    Calculates time required to survey site with an ROV.

    Parameters
    ----------
    rov_survey_time : int | float
        Time required to survey site with an ROV.

    Returns
    -------
    rov_survey_time : float
        Time required to survey site with an ROV (h).
    """

    key = "rov_survey_time"
    rov_survey_time = kwargs.get(key, defaults[key])

    return rov_survey_time
