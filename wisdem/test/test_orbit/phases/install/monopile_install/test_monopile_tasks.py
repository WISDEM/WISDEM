"""
Testing framework for common monopile installation tasks.
"""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


import pytest

from wisdem.orbit.core.exceptions import MissingComponent
from wisdem.orbit.phases.install.monopile_install.common import (
    drive_monopile,
    lower_monopile,
    upend_monopile,
    bolt_transition_piece,
    lower_transition_piece,
    cure_transition_piece_grout,
    pump_transition_piece_grout,
)


@pytest.mark.parametrize(
    "task, log, args",
    [
        (upend_monopile, "Upend Monopile", [100]),
        (lower_monopile, "Lower Monopile", []),
        (drive_monopile, "Drive Monopile", []),
        (lower_transition_piece, "Lower TP", []),
        (bolt_transition_piece, "Bolt TP", []),
        (pump_transition_piece_grout, "Pump TP Grout", []),
        (cure_transition_piece_grout, "Cure TP Grout", []),
    ],
)
def test_task(env, wtiv, task, log, args):

    env.register(wtiv)
    wtiv.initialize(mobilize=False)

    task(wtiv, *args, site_depth=10)
    env.run()

    actions = [a["action"] for a in env.actions]
    assert log in actions


@pytest.mark.parametrize(
    "task, log, args",
    [
        (upend_monopile, "Upend Monopile", [100]),
        (lower_monopile, "Lower Monopile", []),
        (drive_monopile, "Drive Monopile", []),
    ],
)
def test_task_fails(env, feeder, task, log, args):

    env.register(feeder)
    feeder.initialize(mobilize=False)

    with pytest.raises(MissingComponent):
        task(feeder, *args, site_depth=10)
        env.run()

    actions = [a["action"] for a in env.actions]
    assert log not in actions
