"""Tests for common infrastructure for quayside assembly tow-out simulations"""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import pandas as pd
import pytest

from wisdem.orbit.core import WetStorage
from wisdem.orbit.phases.install.quayside_assembly_tow.common import TurbineAssemblyLine, SubstructureAssemblyLine


@pytest.mark.parametrize(
    "num, assigned, expected",
    [
        (1, [], 0),
        (1, [1] * 10, 100),
        (2, [1] * 10, 50),
        (3, [1] * 10, 40),
        (5, [1] * 10, 20),
        (10, [1] * 10, 10),
    ],
)
def test_SubstructureAssemblyLine(env, num, assigned, expected):

    _assigned = len(assigned)
    storage = WetStorage(env, capacity=float("inf"))

    for a in range(num):
        assembly = SubstructureAssemblyLine(assigned, 10, storage, a + 1)
        env.register(assembly)
        assembly.start()

    env.run()

    assert len(env.actions) == _assigned
    assert env.now == expected


@pytest.mark.parametrize(
    "num, assigned",
    [
        (1, [1] * 10),
        (2, [1] * 10),
        (3, [1] * 10),
        (5, [1] * 10),
        (10, [1] * 10),
    ],
)
def test_TurbineAssemblyLine(env, num, assigned):

    _assigned = len(assigned)
    feed = WetStorage(env, capacity=float("inf"))
    target = WetStorage(env, capacity=float("inf"))

    for i in assigned:
        feed.put(0)

    for a in range(num):
        assembly = TurbineAssemblyLine(feed, target, {"tower": {"sections": 1}}, a + 1)
        env.register(assembly)
        assembly.start()

    env.run()

    df = pd.DataFrame(env.actions)
    assert len(df.loc[df["action"] == "Mechanical Completion"]) == len(assigned)


@pytest.mark.parametrize(
    "sub_lines, turb_lines",
    [
        (1, 1),
        (1, 10),
        (1, 100),
        (10, 1),
        (10, 10),
        (10, 100),
        (100, 1),
        (100, 10),
        (100, 100),
    ],
)
def test_Sub_to_Turbine_assembly_interaction(env, sub_lines, turb_lines):

    num_turbines = 50
    assigned = [1] * num_turbines

    feed = WetStorage(env, capacity=2)
    target = WetStorage(env, capacity=float("inf"))

    for a in range(sub_lines):
        assembly = SubstructureAssemblyLine(assigned, 10, feed, a + 1)
        env.register(assembly)
        assembly.start()

    for a in range(turb_lines):
        assembly = TurbineAssemblyLine(feed, target, {"tower": {"sections": 1}}, a + 1)
        env.register(assembly)
        assembly.start()

    env.run()

    df = pd.DataFrame(env.actions)
    assert len(df.loc[df["action"] == "Mechanical Completion"]) == num_turbines
