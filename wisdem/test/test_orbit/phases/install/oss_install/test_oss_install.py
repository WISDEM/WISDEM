"""Tests for the `OffshoreSubstationInstallation` class using feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pandas as pd
import pytest

from wisdem.orbit import ProjectManager
from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.phases.install import FloatingSubstationInstallation, OffshoreSubstationInstallation
from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.core.exceptions import MissingComponent

config_single = extract_library_specs("config", "oss_install")
config_floating = extract_library_specs("config", "floating_oss_install")
config_multi = extract_library_specs("config", "oss_install")
config_multi["num_feeders"] = 2


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
def test_simulation_setup(config):

    sim = OffshoreSubstationInstallation(config)
    assert sim.config == config
    assert sim.env
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]
    assert sim.num_substations == sim.config["num_substations"]

    assert len(sim.port.items) == 2 * sim.num_substations


def test_floating_simulation_setup():

    sim = FloatingSubstationInstallation(config_floating)
    assert sim.config == config_floating
    assert sim.env
    assert sim.num_substations == sim.config["num_substations"]


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
def test_vessel_initialization(config):

    sim = OffshoreSubstationInstallation(config)
    assert sim.oss_vessel
    assert sim.oss_vessel.crane

    js = sim.oss_vessel._jacksys_specs
    dp = sim.oss_vessel._dp_specs

    if not any([js, dp]):
        assert False

    for feeder in sim.feeders:
        assert feeder.storage


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize("weather", (None, test_weather), ids=["no_weather", "test_weather"])
def test_for_complete_logging(weather, config):

    # No weather
    sim = OffshoreSubstationInstallation(config, weather=weather)
    sim.run()

    df = pd.DataFrame(sim.env.actions)
    df = df.assign(shift=(df["time"] - df["time"].shift(1)))

    for vessel in df["agent"].unique():
        _df = df[df["agent"] == vessel].copy()
        _df = _df.assign(shift=(_df["time"] - _df["time"].shift(1)))
        assert (_df["shift"] - _df["duration"]).fillna(0.0).abs().max() < 1e-9

    assert ~df["cost"].isnull().any()
    _ = sim.agent_efficiencies
    _ = sim.detailed_output


@pytest.mark.parametrize("weather", (None, test_weather), ids=["no_weather", "test_weather"])
def test_for_complete_logging_floating(weather):

    sim = FloatingSubstationInstallation(config_floating, weather=weather)
    sim.run()

    df = pd.DataFrame(sim.env.actions)
    df = df.assign(shift=(df["time"] - df["time"].shift(1)))

    for vessel in df["agent"].unique():
        _df = df[df["agent"] == vessel].copy()
        _df = _df.assign(shift=(_df["time"] - _df["time"].shift(1)))
        assert (_df["shift"] - _df["duration"]).fillna(0.0).abs().max() < 1e-9


def test_kwargs():

    sim = OffshoreSubstationInstallation(config_single)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "mono_embed_len",
        "mono_drive_rate",
        "mono_fasten_time",
        "mono_release_time",
        "tp_bolt_time",
        "site_position_time",
        "crane_reequip_time",
        "rov_survey_time",
        "topside_fasten_time",
        "topside_release_time",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]

        if kw == "mono_drive_rate":
            _new = default - 2

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            kwargs = {kw: _new}

        else:
            kwargs = {kw: default + 2}

        new_sim = OffshoreSubstationInstallation(config_single, **kwargs)
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

    base = deepcopy(config_single)
    base["install_phases"] = ["OffshoreSubstationInstallation"]

    project = ProjectManager(base)
    project.run()
    baseline = project.phase_times["OffshoreSubstationInstallation"]

    keywords = [
        "mono_embed_len",
        "mono_drive_rate",
        "mono_fasten_time",
        "mono_release_time",
        "tp_bolt_time",
        "site_position_time",
        "crane_reequip_time",
        "rov_survey_time",
        "topside_fasten_time",
        "topside_release_time",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]

        if kw == "mono_drive_rate":
            _new = default - 2

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            processes = {kw: _new}

        else:
            processes = {kw: default + 2}

        new_config = deepcopy(base)
        new_config["processes"] = processes

        new_project = ProjectManager(new_config)
        new_project.run()
        new_time = new_project.phase_times["OffshoreSubstationInstallation"]

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"'{failed}' not affecting results.")

    else:
        assert True
