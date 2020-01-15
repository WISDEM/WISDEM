"""Tests for the cable laying subprocesses."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import pytest

from wisdem.orbit.vessels.tasks import (
    defaults,
    tow_plow,
    lay_cable,
    bury_cable,
    dig_trench,
    prep_cable,
    pull_winch,
    test_cable,
    lower_cable,
    raise_cable,
    splice_cable,
    lift_carousel,
    pull_in_cable,
    lay_bury_cable,
    fasten_carousel,
)


@pytest.mark.parametrize(
    "fn,key,expected",
    (
        (fn, key, defaults[key])
        for fn, key in (
            (lift_carousel, "carousel_lift_time"),
            (fasten_carousel, "carousel_fasten_time"),
            (prep_cable, "cable_prep_time"),
            (lower_cable, "cable_lower_time"),
            (pull_in_cable, "cable_pull_in_time"),
            (test_cable, "cable_termination_time"),
            (splice_cable, "cable_splice_time"),
            (raise_cable, "cable_raise_time"),
        )
    ),
)
def test_time_only_functions(fn, key, expected):
    assert fn() == expected

    for n in (0, 1, 20):
        kwargs = {key: n}
        assert fn(**kwargs) == n


@pytest.mark.parametrize(
    "fn,key,speed",
    (
        (fn, key, defaults[key])
        for fn, key in (
            (lay_cable, "cable_lay_speed"),
            (bury_cable, "cable_bury_speed"),
            (lay_bury_cable, "cable_lay_bury_speed"),
            (tow_plow, "tow_plow_speed"),
            (pull_winch, "pull_winch_speed"),
        )
    ),
)
def test_distance_functions(fn, key, speed):
    for dist, expected in ((0, 0), (1, 1 / speed), (speed, 1)):
        assert fn(dist) == expected

    for dist, n, expected in ((1, 1, 1), (1, 20, 0.05), (20, 10, 2)):
        kwargs = {key: n}
        assert fn(dist, **kwargs) == expected


def test_dig_trench_defaults():
    assert dig_trench(1) == 1 / defaults["trench_dig_speed"]
    assert dig_trench(0) == 0


def test_dig_trench_inputs():
    assert dig_trench(10, **{"trench_dig_speed": 10}) == 1
    assert dig_trench(10, **{"trench_dig_speed": 4}) == 2.5
