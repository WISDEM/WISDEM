"""
Testing framework for common cable installation tasks.
"""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


import pytest

from wisdem.orbit.core.exceptions import MissingComponent
from wisdem.orbit.phases.install.cable_install.common import (
    tow_plow,
    lay_cable,
    bury_cable,
    prep_cable,
    pull_winch,
    lower_cable,
    raise_cable,
    splice_cable,
    pull_in_cable,
    lay_bury_cable,
    terminate_cable,
    load_cable_on_vessel,
)


def test_load_cable_on_vessel(env, cable_vessel, feeder, simple_cable):

    env.register(cable_vessel)
    cable_vessel.initialize(mobilize=False)

    env.register(feeder)
    feeder.initialize(mobilize=False)

    load_cable_on_vessel(cable_vessel, simple_cable)
    env.run()

    with pytest.raises(MissingComponent):
        load_cable_on_vessel(feeder, simple_cable)
        env.run()


@pytest.mark.parametrize(
    "task, log, args",
    [
        (prep_cable, "Prepare Cable", []),
        (lower_cable, "Lower Cable", []),
        (pull_in_cable, "Pull In Cable", []),
        (terminate_cable, "Terminate Cable", []),
        (lay_bury_cable, "Lay/Bury Cable", [10]),
        (lay_cable, "Lay Cable", [10]),
        (splice_cable, "Splice Cable", []),
        (raise_cable, "Raise Cable", []),
        (tow_plow, "Tow Plow", [100]),
        (pull_winch, "Pull Winch", [100]),
    ],
)
def test_task(env, cable_vessel, task, log, args):

    env.register(cable_vessel)
    cable_vessel.initialize(mobilize=False)

    task(cable_vessel, *args)
    env.run()

    actions = [a["action"] for a in env.actions]
    assert log in actions
