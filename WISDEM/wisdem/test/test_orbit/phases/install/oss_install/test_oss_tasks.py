"""
Testing framework for common oss installation tasks.
"""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


import pytest

from wisdem.orbit.core.exceptions import MissingComponent
from wisdem.orbit.phases.install.oss_install.common import (
    lift_topside,
    attach_topside,
)


@pytest.mark.parametrize(
    "task, log, args",
    [
        (lift_topside, "Lift Topside", []),
        (attach_topside, "Attach Topside", []),
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
        (lift_topside, "Lift Topside", []),
        (attach_topside, "Attach Topside", []),
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
