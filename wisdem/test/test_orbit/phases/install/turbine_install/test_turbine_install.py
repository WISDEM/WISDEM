"""Tests for the `MonopileInstallation` class without feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2019, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pandas as pd
import pytest

from wisdem.orbit import ProjectManager
from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.phases.install import TurbineInstallation
from wisdem.test.test_orbit.data import test_weather

config_wtiv = extract_library_specs("config", "turbine_install_wtiv")
config_long_mobilize = extract_library_specs("config", "turbine_install_long_mobilize")
config_wtiv_feeder = extract_library_specs("config", "turbine_install_feeder")
config_wtiv_multi_feeder = deepcopy(config_wtiv_feeder)
config_wtiv_multi_feeder["num_feeders"] = 2
floating = extract_library_specs("config", "floating_turbine_install_feeder")


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder, floating),
    ids=["wtiv_only", "single_feeder", "multi_feeder", "floating"],
)
def test_simulation_setup(config):

    sim = TurbineInstallation(config)
    assert sim.config == config
    assert sim.env
    assert sim.port.crane.capacity == config["port"]["num_cranes"]
    assert sim.num_turbines == config["plant"]["num_turbines"]

    t = len([i for i in sim.port.items if i.type == "TowerSection"])
    assert t == sim.num_turbines

    n = len([i for i in sim.port.items if i.type == "Nacelle"])
    assert n == sim.num_turbines

    b = len([i for i in sim.port.items if i.type == "Blade"])
    assert b == sim.num_turbines * 3


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder, floating),
    ids=["wtiv_only", "single_feeder", "multi_feeder", "floating"],
)
def test_vessel_creation(config):

    sim = TurbineInstallation(config)
    assert sim.wtiv
    assert sim.wtiv.crane
    assert sim.wtiv.storage

    js = sim.wtiv._jacksys_specs
    dp = sim.wtiv._dp_specs

    if not any([js, dp]):
        assert False

    if config.get("feeder", None) is not None:
        assert len(sim.feeders) == config["num_feeders"]

        for feeder in sim.feeders:
            # assert feeder.jacksys
            assert feeder.storage


@pytest.mark.parametrize("config, expected", [(config_wtiv, 72), (config_long_mobilize, 14 * 24)])
def test_vessel_mobilize(config, expected):

    sim = TurbineInstallation(config)
    assert sim.wtiv

    mobilize = [a for a in sim.env.actions if a["action"] == "Mobilize"][0]
    assert mobilize["duration"] == expected


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder, floating),
    ids=["wtiv_only", "single_feeder", "multi_feeder", "floating"],
)
@pytest.mark.parametrize("weather", (None, test_weather), ids=["no_weather", "test_weather"])
def test_for_complete_logging(weather, config):

    sim = TurbineInstallation(config, weather=weather)
    sim.run()

    df = pd.DataFrame(sim.env.actions)
    df = df.assign(shift=(df["time"] - df["time"].shift(1)))

    for vessel in df["agent"].unique():
        _df = df[df["agent"] == vessel].copy()
        _df = _df.assign(shift=(_df["time"] - _df["time"].shift(1)))
        assert (_df["shift"] - _df["duration"]).abs().max() < 1e-9

    assert ~df["cost"].isnull().any()
    _ = sim.agent_efficiencies
    _ = sim.detailed_output


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder, floating),
    ids=["wtiv_only", "single_feeder", "multi_feeder", "floating"],
)
def test_for_complete_installation(config):

    sim = TurbineInstallation(config)
    sim.run()

    installed_nacelles = len([a for a in sim.env.actions if a["action"] == "Attach Nacelle"])
    assert installed_nacelles == sim.num_turbines


def test_kwargs():

    sim = TurbineInstallation(config_wtiv)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "tower_section_fasten_time",
        "tower_section_release_time",
        "tower_section_attach_time",
        "nacelle_fasten_time",
        "nacelle_release_time",
        "nacelle_attach_time",
        "blade_fasten_time",
        "blade_release_time",
        "blade_attach_time",
        "site_position_time",
        "crane_reequip_time",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]
        kwargs = {kw: default + 2}

        new_sim = TurbineInstallation(config_wtiv, **kwargs)
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

    base = deepcopy(config_wtiv)
    base["install_phases"] = ["TurbineInstallation"]

    project = ProjectManager(base)
    project.run()
    baseline = project.phase_times["TurbineInstallation"]

    keywords = [
        "tower_section_fasten_time",
        "tower_section_release_time",
        "tower_section_attach_time",
        "nacelle_fasten_time",
        "nacelle_release_time",
        "nacelle_attach_time",
        "blade_fasten_time",
        "blade_release_time",
        "blade_attach_time",
        "site_position_time",
        "crane_reequip_time",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]
        processes = {kw: default + 2}

        new_config = deepcopy(base)
        new_config["processes"] = processes

        new_project = ProjectManager(new_config)
        new_project.run()
        new_time = new_project.phase_times["TurbineInstallation"]

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"'{failed}' not affecting results.")

    else:
        assert True


def test_multiple_tower_sections():

    sim = TurbineInstallation(config_wtiv)
    sim.run()
    baseline = len([a for a in sim.env.actions if a["action"] == "Attach Tower Section"])

    two_sections = deepcopy(config_wtiv)
    two_sections["turbine"]["tower"]["sections"] = 2

    sim2 = TurbineInstallation(two_sections)
    sim2.run()
    new = len([a for a in sim2.env.actions if a["action"] == "Attach Tower Section"])

    assert new == 2 * baseline

    df = pd.DataFrame(sim.env.actions)
    for vessel in df["agent"].unique():

        vl = df[df["agent"] == vessel].copy()
        vl = vl.assign(shift=(vl["time"] - vl["time"].shift(1)))

        assert (vl["shift"] - vl["duration"]).abs().max() < 1e-9
