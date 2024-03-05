"""Tests for the `JacketInstallation` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pandas as pd
import pytest

from wisdem.orbit import ProjectManager
from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.phases.install import JacketInstallation
from wisdem.test.test_orbit.data import test_weather

config_wtiv = extract_library_specs("config", "single_wtiv_jacket_install")
config_feeder = extract_library_specs("config", "feeder_jacket_install")
config_multi_feeder = deepcopy(config_feeder)
config_multi_feeder["num_feeders"] = 2


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_feeder, config_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_simulation_setup(config):

    sim = JacketInstallation(config)
    assert sim.config == config
    assert sim.env
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]
    assert sim.num_jackets == config["plant"]["num_turbines"]

    ja = len([i for i in sim.port.items if i.type == "Jacket"])
    assert ja == sim.num_jackets

    tp = len([i for i in sim.port.items if i.type == "TransitionPiece"])
    assert tp == sim.num_jackets


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_feeder, config_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_vessel_initialization(config):

    sim = JacketInstallation(config)
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
    (config_wtiv, config_feeder, config_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize("weather", (None, test_weather), ids=["no_weather", "test_weather"])
def test_for_complete_logging(weather, config):

    sim = JacketInstallation(config, weather=weather)
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
    (config_wtiv, config_feeder, config_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_num_legs(config):

    base = JacketInstallation(config)
    base.run()

    new_config = deepcopy(config)
    new_config["jacket"]["num_legs"] = 6

    sim = JacketInstallation(new_config)
    sim.run()

    assert sim.total_phase_time > base.total_phase_time


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_feeder, config_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_foundation_type(config):

    base = JacketInstallation(config)
    base.run()

    base_actions = [a["action"] for a in base.env.actions]
    assert "Drive Pile" in base_actions

    new_config = deepcopy(config)
    new_config["jacket"]["foundation_type"] = "suction"

    sim = JacketInstallation(new_config)
    sim.run()

    actions = [a["action"] for a in sim.env.actions]
    assert "Install Suction Bucket" in actions


def test_kwargs_piles():

    sim = JacketInstallation(config_wtiv)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "jacket_fasten_time",
        "jacket_release_time",
        "jacket_lift_time",
        "jacket_lower_time",
        "jacket_grout_time",
        "jacket_pin_template_time",
        "jacket_pile_drive_time",
        "jacket_position_pile",
        "jacket_vessel_reposition",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]
        kwargs = {kw: default + 2}

        new_sim = JacketInstallation(config_wtiv, **kwargs)
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


def test_kwargs_suction():

    config_wtiv_suction = deepcopy(config_wtiv)
    config_wtiv_suction["jacket"]["foundation_type"] = "suction"

    sim = JacketInstallation(config_wtiv_suction)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "jacket_fasten_time",
        "jacket_release_time",
        "jacket_lift_time",
        "jacket_lower_time",
        "jacket_grout_time",
        "jacket_vessel_reposition",
        "jacket_suction_bucket",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]
        kwargs = {kw: default + 2}

        new_sim = JacketInstallation(config_wtiv_suction, **kwargs)
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
