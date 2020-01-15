"""Tests for monopile related vessel tasks"""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.orbit.vessels import tasks
from wisdem.test.test_orbit.vessels import WTIV, FEEDER


def test_upend_monopile():

    res = tasks.upend_monopile(WTIV, 10, site_depth=10)
    assert isinstance(res, (float, int))

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.upend_monopile(FEEDER, 10, site_depth=10)


def test_lower_monopile():

    res = tasks.lower_monopile(WTIV, site_depth=10)
    assert isinstance(res, (float, int))

    WTIV_no_crane = deepcopy(WTIV)
    delattr(WTIV_no_crane, "crane")

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.lower_monopile(WTIV_no_crane, site_depth=10)

    WTIV_no_jacksys = deepcopy(WTIV)
    delattr(WTIV_no_jacksys, "jacksys")

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.lower_monopile(WTIV_no_jacksys, site_depth=10)

    WTIV_no_crane_or_jacksys = deepcopy(WTIV)
    delattr(WTIV_no_crane_or_jacksys, "crane")
    delattr(WTIV_no_crane_or_jacksys, "jacksys")

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.lower_monopile(WTIV_no_crane_or_jacksys, site_depth=10)


def test_drive_monopile():

    res = tasks.drive_monopile(WTIV)
    assert isinstance(res, (float, int))

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.drive_monopile(FEEDER)

    res2 = tasks.drive_monopile(WTIV, mono_embed_len=100)
    assert res2 != res

    res3 = tasks.drive_monopile(WTIV, mono_drive_rate=50)
    assert res3 != res


def test_lower_transition_piece():

    res = tasks.lower_transition_piece(WTIV)
    assert isinstance(res, (float, int))

    WTIV_no_crane = deepcopy(WTIV)
    delattr(WTIV_no_crane, "crane")

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.lower_transition_piece(WTIV_no_crane)

    WTIV_no_jacksys = deepcopy(WTIV)
    delattr(WTIV_no_jacksys, "jacksys")

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.lower_transition_piece(WTIV_no_jacksys)

    WTIV_no_crane_or_jacksys = deepcopy(WTIV)
    delattr(WTIV_no_crane_or_jacksys, "crane")
    delattr(WTIV_no_crane_or_jacksys, "jacksys")

    with pytest.raises(tasks.MissingComponent):
        _ = tasks.lower_transition_piece(WTIV_no_crane_or_jacksys)


def test_bolt_transition_piece():

    res = tasks.bolt_transition_piece()
    res1 = tasks.bolt_transition_piece(tp_bolt_time=100)
    assert res != res1


def test_pump_transition_piece_grout():

    res = tasks.pump_transition_piece_grout()
    res1 = tasks.pump_transition_piece_grout(grout_pump_time=100)
    assert res != res1


def test_cure_transition_piece_grout():

    res = tasks.cure_transition_piece_grout()
    res1 = tasks.cure_transition_piece_grout(grout_cure_time=100)
    assert res != res1


def test_fasten_monopile():

    res = tasks.fasten_monopile()
    res1 = tasks.fasten_monopile(mono_fasten_time=100)
    assert res != res1


def test_release_monopile():

    res = tasks.release_monopile()
    res1 = tasks.release_monopile(mono_release_time=100)
    assert res != res1


def test_fasten_transition_piece():

    res = tasks.fasten_transition_piece()
    res1 = tasks.fasten_transition_piece(tp_fasten_time=100)
    assert res != res1


def test_release_transition_piece():

    res = tasks.release_transition_piece()
    res1 = tasks.release_transition_piece(tp_release_time=100)
    assert res != res1
