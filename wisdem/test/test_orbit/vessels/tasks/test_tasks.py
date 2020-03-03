"""Tests for the cable laying subprocesses."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2019, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.test.test_orbit.vessels import WTIV, FEEDER

# more complex tests
# simple tests
from wisdem.orbit.vessels.tasks import (  # cabling tasks; scour protection tasks; monopile tasks; turbine tasks; oss tasks
    MissingComponent,
    defaults,
    tow_plow,
    lay_cable,
    bury_cable,
    dig_trench,
    drop_rocks,
    load_rocks,
    prep_cable,
    pull_winch,
    rov_survey,
    test_cable,
    lower_cable,
    raise_cable,
    fasten_tower_section,
    splice_cable,
    lift_carousel,
    pull_in_cable,
    release_tower_section,
    drive_monopile,
    fasten_nacelle,
    fasten_topside,
    lay_bury_cable,
    lower_monopile,
    upend_monopile,
    fasten_carousel,
    fasten_monopile,
    position_onsite,
    release_nacelle,
    release_topside,
    release_monopile,
    fasten_turbine_blade,
    bolt_transition_piece,
    release_turbine_blade,
    lower_transition_piece,
    fasten_transition_piece,
    release_transition_piece,
    cure_transition_piece_grout,
    pump_transition_piece_grout,
)

WTIV_no_crane = deepcopy(WTIV)
delattr(WTIV_no_crane, "crane")

WTIV_no_jacksys = deepcopy(WTIV)
delattr(WTIV_no_jacksys, "jacksys")

WTIV_no_crane_jacksys = deepcopy(WTIV)
delattr(WTIV_no_crane_jacksys, "crane")
delattr(WTIV_no_crane_jacksys, "jacksys")

jacksys_required_tasks = (lower_monopile, lower_transition_piece)


@pytest.mark.parametrize(
    "fn,key,expected",
    (
        (fn, key, defaults[key])
        for fn, key in (
            (position_onsite, "site_position_time"),
            (rov_survey, "rov_survey_time"),
            (lift_carousel, "carousel_lift_time"),
            (fasten_carousel, "carousel_fasten_time"),
            (prep_cable, "cable_prep_time"),
            (lower_cable, "cable_lower_time"),
            (pull_in_cable, "cable_pull_in_time"),
            (test_cable, "cable_termination_time"),
            (splice_cable, "cable_splice_time"),
            (raise_cable, "cable_raise_time"),
            (bolt_transition_piece, "tp_bolt_time"),
            (pump_transition_piece_grout, "grout_pump_time"),
            (cure_transition_piece_grout, "grout_cure_time"),
            (fasten_monopile, "mono_fasten_time"),
            (release_monopile, "mono_release_time"),
            (fasten_transition_piece, "tp_fasten_time"),
            (release_transition_piece, "tp_release_time"),
            (fasten_nacelle, "nacelle_fasten_time"),
            (release_nacelle, "nacelle_release_time"),
            (fasten_turbine_blade, "blade_fasten_time"),
            (release_turbine_blade, "blade_release_time"),
            (fasten_tower_section, "tower_section_fasten_time"),
            (release_tower_section, "tower_section_release_time"),
            (fasten_topside, "topside_fasten_time"),
            (release_topside, "topside_release_time"),
        )
    ),
)
def test_time_only_functions(fn, key, expected):
    assert expected == pytest.approx(fn(), rel=1e-10)

    for n in (0, 1, 10.2, 100):
        kwargs = {key: n}
        assert n == pytest.approx(fn(**kwargs), rel=1e-10)


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
            (dig_trench, "trench_dig_speed"),
        )
    ),
)
def test_distance_functions(fn, key, speed):
    for dist, expected in ((0, 0), (1, 1 / speed), (speed, 1)):
        assert expected == pytest.approx(fn(dist), rel=1e-10)

    for dist, n, expected in ((1, 1, 1), (1, 20, 0.05), (20, 10, 2)):
        kwargs = {key: n}
        assert expected == pytest.approx(fn(dist, **kwargs), rel=1e-10)


@pytest.mark.parametrize(
    "fn",
    (upend_monopile, lower_monopile, drive_monopile, lower_transition_piece),
)
def test_time_with_vessel_component(fn):
    try:
        assert isinstance(fn(WTIV, 10, site_depth=10), (float, int))
    except TypeError:
        assert isinstance(fn(WTIV, site_depth=10), (float, int))

    with pytest.raises(MissingComponent):
        fn(WTIV_no_crane, site_depth=10)

    if fn in jacksys_required_tasks:
        with pytest.raises(MissingComponent):
            fn(WTIV_no_crane_jacksys, site_depth=10)


@pytest.mark.parametrize(
    "fn,non_vessel_args",
    (
        (upend_monopile, [10]),
        (lower_monopile, []),
        (drive_monopile, []),
        (lower_transition_piece, []),
    ),
)
def test_time_with_vessel_component(fn, non_vessel_args):
    assert isinstance(fn(WTIV, *non_vessel_args, site_depth=10), (float, int))

    with pytest.raises(MissingComponent):
        fn(WTIV_no_crane, *non_vessel_args, site_depth=10)

    if fn in jacksys_required_tasks:
        with pytest.raises(MissingComponent):
            fn(WTIV_no_crane_jacksys, *non_vessel_args, site_depth=10)
