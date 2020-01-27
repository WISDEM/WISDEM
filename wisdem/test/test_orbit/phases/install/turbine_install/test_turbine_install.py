"""Tests for the `MonopileInstallation` class without feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2019, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.library import initialize_library, extract_library_specs
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import TurbineInstallation

initialize_library(pytest.library)
config_wtiv = extract_library_specs("config", "turbine_install_wtiv")
config_wtiv_feeder = extract_library_specs("config", "turbine_install_feeder")
config_wtiv_multi_feeder = deepcopy(config_wtiv_feeder)
config_wtiv_multi_feeder["num_feeders"] = 2


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_creation(config):

    sim = TurbineInstallation(config, print_logs=False)
    assert sim.config == config
    assert sim.env
    assert sim.env.logger


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_port_creation(config):

    sim = TurbineInstallation(config, print_logs=False)
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_vessel_creation(config):

    sim = TurbineInstallation(config, print_logs=False)
    assert sim.wtiv
    assert sim.wtiv.jacksys
    assert sim.wtiv.crane

    if config.get("feeder", None) is not None:
        assert len(sim.feeders) == config["num_feeders"]

        for feeder in sim.feeders:
            assert feeder.jacksys


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_turbine_creation(config):

    sim = TurbineInstallation(config, print_logs=False)
    assert sim.num_turbines == config["plant"]["num_turbines"]

    t = len(
        [item for item in sim.port.items if item["type"] == "Tower Section"]
    )
    assert sim.num_turbines == t

    n = len([item for item in sim.port.items if item["type"] == "Nacelle"])
    assert sim.num_turbines == n

    b = len([item for item in sim.port.items if item["type"] == "Blade"])
    assert sim.num_turbines * 3 == b


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "log_level,expected", (("INFO", 20), ("DEBUG", 10)), ids=["info", "debug"]
)
def test_logger_creation(config, log_level, expected):

    sim = TurbineInstallation(config, log_level=log_level)
    assert sim.env.logger.level == expected


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_full_run(config, weather):

    sim = TurbineInstallation(config, weather=weather, log_level="INFO")
    sim.run()

    complete = float(sim.logs["time"].max())

    assert complete > 0


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_for_complete_logging(weather, config):

    sim = TurbineInstallation(config, log_level="INFO")
    sim.run()

    df = sim.phase_dataframe.copy()
    df = df.loc[~df["agent"].isin(["Port", "Test Port"])]
    df = df.assign(shift=(df["time"] - df["time"].shift(1)))

    for vessel in df["agent"].unique():
        _df = df[df["agent"] == vessel].copy()
        _df = _df.assign(shift=(_df["time"] - _df["time"].shift(1)))
        assert (_df["shift"] - _df["duration"]).abs().max() < 1e-9


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_for_complete_installation(config):

    sim = TurbineInstallation(config, log_level="INFO", print_logs=False)
    sim.run()

    installed_towers = len(
        sim.logs.loc[sim.logs["action"] == "AttachTowerSection"]
    )
    assert installed_towers == sim.num_turbines

    installed_nacelles = len(
        sim.logs.loc[sim.logs["action"] == "AttachNacelle"]
    )
    assert installed_nacelles == sim.num_turbines

    installed_blades = len(sim.logs.loc[sim.logs["action"] == "AttachBlade"])
    assert installed_blades == sim.num_turbines * 3


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_for_efficiencies(config):

    sim = TurbineInstallation(config)
    sim.run()

    assert 0 <= sim.detailed_output["Example WTIV_operational_efficiency"] <= 1
    if sim.feeders is None:
        assert (
            0
            <= sim.detailed_output["Example WTIV_cargo_weight_utilization"]
            <= 1
        )
        assert (
            0
            <= sim.detailed_output["Example WTIV_deck_space_utilization"]
            <= 1
        )
    else:
        for feeder in sim.feeders:
            name = feeder.name
            assert (
                0 <= sim.detailed_output[f"{name}_operational_efficiency"] <= 1
            )
            assert (
                0
                <= sim.detailed_output[f"{name}_cargo_weight_utilization"]
                <= 1
            )
            assert (
                0 <= sim.detailed_output[f"{name}_deck_space_utilization"] <= 1
            )


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_kwargs(config):

    sim = TurbineInstallation(config, log_level="INFO", print_logs=False)
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

        default = defaults[kw]
        kwargs = {kw: default + 2}

        new_sim = TurbineInstallation(
            config, log_level="INFO", print_logs=False, **kwargs
        )
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


@pytest.mark.parametrize(
    "config", (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder)
)
def test_multiple_tower_sections(config):

    sim = TurbineInstallation(config, log_level="INFO", print_logs=False)
    sim.run()
    baseline = len(sim.logs.loc[sim.logs["action"] == "AttachTowerSection"])

    two_sections = deepcopy(config)
    two_sections["turbine"]["tower"]["sections"] = 2

    sim2 = TurbineInstallation(
        two_sections, log_level="INFO", print_logs=False
    )
    sim2.run()
    new = len(sim2.logs.loc[sim2.logs["action"] == "AttachTowerSection"])

    assert new == 2 * baseline

    df = sim2.phase_dataframe.copy()
    df = df.loc[~df["agent"].isin(["Port"])]

    for vessel in df["agent"].unique():

        vl = df[df["agent"] == vessel].copy()
        vl = vl.assign(shift=(vl["time"] - vl["time"].shift(1)))

        assert (vl["shift"] - vl["duration"]).abs().max() < 1e-9
