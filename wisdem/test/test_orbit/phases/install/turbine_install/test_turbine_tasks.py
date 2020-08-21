"""
Testing framework for common turbine installation tasks.
"""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


import pytest

from wisdem.orbit.core.exceptions import MissingComponent
from wisdem.orbit.phases.install.turbine_install.common import (
    lift_nacelle,
    attach_nacelle,
    lift_tower_section,
    lift_turbine_blade,
    attach_tower_section,
    attach_turbine_blade,
)


@pytest.mark.parametrize(
    "task, log, args",
    [
        (lift_nacelle, "Lift Nacelle", []),
        (attach_nacelle, "Attach Nacelle", []),
        (lift_turbine_blade, "Lift Blade", []),
        (attach_turbine_blade, "Attach Blade", []),
        (lift_tower_section, "Lift Tower Section", [50]),
        (attach_tower_section, "Attach Tower Section", []),
    ],
)
def test_task(env, wtiv, task, log, args):

    env.register(wtiv)
    wtiv.initialize(mobilize=False)

    task(wtiv, *args, hub_height=100)
    env.run()

    actions = [a["action"] for a in env.actions]
    assert log in actions


@pytest.mark.parametrize(
    "task, log, args",
    [
        (lift_nacelle, "Lift Nacelle", []),
        (attach_nacelle, "Attach Nacelle", []),
        (lift_turbine_blade, "Lift Blade", []),
        (attach_turbine_blade, "Attach Blade", []),
        (lift_tower_section, "Lift Tower Section", [50]),
        (attach_tower_section, "Attach Tower Section", []),
    ],
)
def test_task_fails(env, feeder, task, log, args):

    env.register(feeder)
    feeder.initialize(mobilize=False)

    with pytest.raises(MissingComponent):
        task(feeder, *args, hub_height=100)
        env.run()

    actions = [a["action"] for a in env.actions]
    assert log not in actions
