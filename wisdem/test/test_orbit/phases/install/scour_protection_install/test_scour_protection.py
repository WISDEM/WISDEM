"""
Testing framework for the `ScourProtectionInstallation` class.
"""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


from copy import deepcopy

import pandas as pd
import pytest

from wisdem.orbit import ProjectManager
from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.phases.install import ScourProtectionInstallation
from wisdem.test.test_orbit.data import test_weather

config = extract_library_specs("config", "scour_protection_install")


def test_simulation_creation():
    sim = ScourProtectionInstallation(config)

    assert sim.config == config
    assert sim.env
    assert sim.port
    assert sim.spi_vessel
    assert sim.num_turbines
    assert sim.tonnes_per_substructure


@pytest.mark.parametrize("weather", (None, test_weather), ids=["no_weather", "test_weather"])
def test_full_run_logging(weather):
    sim = ScourProtectionInstallation(config, weather=weather)
    sim.run()

    df = pd.DataFrame(sim.env.actions)
    df = df.assign(shift=(df.time - df.time.shift(1)))
    assert (df.duration - df["shift"]).fillna(0.0).abs().max() < 1e-9
    assert df[df.action == "Drop SP Material"].shape[0] == sim.num_turbines

    assert ~df["cost"].isnull().any()
    _ = sim.agent_efficiencies
    _ = sim.detailed_output


def test_kwargs():

    sim = ScourProtectionInstallation(config)
    sim.run()
    baseline = sim.total_phase_time

    keywords = ["drop_rocks_time"]

    failed = []

    for kw in keywords:

        default = pt[kw]
        kwargs = {kw: default + 2}

        new_sim = ScourProtectionInstallation(config, **kwargs)
        new_sim.run()
        new_time = new_sim.total_phase_time

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"'{failed}' not affecting results.")

    else:
        assert True


def test_kwargs_in_ProjectManager():

    base = deepcopy(config)
    base["install_phases"] = ["ScourProtectionInstallation"]

    project = ProjectManager(base)
    project.run()
    baseline = project.phase_times["ScourProtectionInstallation"]

    keywords = ["drop_rocks_time"]

    failed = []

    for kw in keywords:

        default = pt[kw]
        processes = {kw: default + 2}

        new_config = deepcopy(base)
        new_config["processes"] = processes

        new_project = ProjectManager(new_config)
        new_project.run()
        new_time = new_project.phase_times["ScourProtectionInstallation"]

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"'{failed}' not affecting results.")

    else:
        assert True
