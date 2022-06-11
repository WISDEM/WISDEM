"""Tests for the `MonopileInstallation` class without feeder barges."""

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
from wisdem.orbit.phases.install import MonopileInstallation
from wisdem.test.test_orbit.data import test_weather

config_wtiv = extract_library_specs("config", "single_wtiv_mono_install")
config_wtiv_feeder = extract_library_specs("config", "multi_wtiv_mono_install")
config_wtiv_multi_feeder = deepcopy(config_wtiv_feeder)
config_wtiv_multi_feeder["num_feeders"] = 2


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_simulation_setup(config):

    sim = MonopileInstallation(config)
    assert sim.config == config
    assert sim.env
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]
    assert sim.num_monopiles == config["plant"]["num_turbines"]

    mp = len([i for i in sim.port.items if i.type == "Monopile"])
    assert mp == sim.num_monopiles

    tp = len([i for i in sim.port.items if i.type == "TransitionPiece"])
    assert tp == sim.num_monopiles


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_vessel_initialization(config):

    sim = MonopileInstallation(config)
    assert sim.wtiv
    assert sim.wtiv.jacksys
    assert sim.wtiv.crane
    assert sim.wtiv.storage

    if config.get("feeder", None) is not None:
        assert len(sim.feeders) == config["num_feeders"]

        for feeder in sim.feeders:
            assert feeder.jacksys
            assert feeder.storage


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize("weather", (None, test_weather), ids=["no_weather", "test_weather"])
def test_for_complete_logging(weather, config):

    sim = MonopileInstallation(config, weather=weather)
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


def test_kwargs():

    sim = MonopileInstallation(config_wtiv)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "mono_embed_len",
        "mono_drive_rate",
        "mono_fasten_time",
        "mono_release_time",
        "tp_fasten_time",
        "tp_release_time",
        "tp_bolt_time",
        "site_position_time",
        "crane_reequip_time",
        "rov_survey_time",
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

        new_sim = MonopileInstallation(config_wtiv, **kwargs)
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
    base["install_phases"] = ["MonopileInstallation"]
    project = ProjectManager(base)
    project.run()
    baseline = project.phase_times["MonopileInstallation"]

    keywords = [
        "mono_embed_len",
        "mono_drive_rate",
        "mono_fasten_time",
        "mono_release_time",
        "tp_fasten_time",
        "tp_release_time",
        "tp_bolt_time",
        "site_position_time",
        "crane_reequip_time",
        "rov_survey_time",
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
        new_time = new_project.phase_times["MonopileInstallation"]

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"'{failed}' not affecting results.")

    else:
        assert True


def test_grout_kwargs():

    sim = MonopileInstallation(config_wtiv)
    sim.run()

    assert "Bolt TP" in list([a["action"] for a in sim.env.actions])

    sim = MonopileInstallation(config_wtiv, tp_connection_type="grouted")
    sim.run()

    assert "Pump TP Grout" in list([a["action"] for a in sim.env.actions])
    assert "Cure TP Grout" in list([a["action"] for a in sim.env.actions])
