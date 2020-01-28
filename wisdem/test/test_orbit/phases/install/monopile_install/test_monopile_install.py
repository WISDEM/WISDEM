"""Tests for the `MonopileInstallation` class without feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.library import initialize_library, extract_library_specs
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import MonopileInstallation

initialize_library(pytest.library)
config_wtiv = extract_library_specs("config", "single_wtiv_mono_install")
config_wtiv_feeder = extract_library_specs("config", "multi_wtiv_mono_install")
config_wtiv_multi_feeder = deepcopy(config_wtiv_feeder)
config_wtiv_multi_feeder["num_feeders"] = 2


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_creation(config):

    sim = MonopileInstallation(config, print_logs=False)
    assert sim.config == config
    assert sim.env
    assert sim.env.logger


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_port_creation(config):

    sim = MonopileInstallation(config, print_logs=False)
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_vessel_creation(config):

    sim = MonopileInstallation(config, print_logs=False)
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
def test_monopile_creation(config):

    sim = MonopileInstallation(config, print_logs=False)
    assert sim.num_monopiles == config["plant"]["num_turbines"]

    mp = len([item for item in sim.port.items if item["type"] == "Monopile"])
    assert sim.num_monopiles == mp

    tp = len(
        [item for item in sim.port.items if item["type"] == "Transition Piece"]
    )
    assert sim.num_monopiles == tp


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "log_level,expected", (("INFO", 20), ("DEBUG", 10)), ids=["info", "debug"]
)
def test_logger_creation(config, log_level, expected):

    sim = MonopileInstallation(config, log_level=log_level)
    assert sim.env.logger.level == expected


@pytest.mark.parametrize(
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_full_run(weather, config):

    sim = MonopileInstallation(config, weather=weather, log_level="INFO")
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

    sim = MonopileInstallation(config, weather=weather, log_level="INFO")
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
def test_for_efficiencies(config):

    sim = MonopileInstallation(config)
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

    sim = MonopileInstallation(config, log_level="INFO", print_logs=False)
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

        default = defaults[kw]

        if kw == "mono_drive_rate":
            _new = default - 2

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            kwargs = {kw: _new}

        else:
            kwargs = {kw: default + 2}

        new_sim = MonopileInstallation(
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
    "config",
    (config_wtiv, config_wtiv_feeder, config_wtiv_multi_feeder),
    ids=["wtiv_only", "single_feeder", "multi_feeder"],
)
def test_grout_kwargs(config):

    sim = MonopileInstallation(
        config,
        log_level="INFO",
        print_logs=False,
        tp_connection_type="grouted",
    )
    sim.run()
    baseline = sim.total_phase_time

    assert "PumpGrout" in list(sim.logs["action"])
    assert "CureGrout" in list(sim.logs["action"])

    keywords = ["grout_cure_time", "grout_pump_time"]

    failed = []

    for kw in keywords:

        default = defaults[kw]

        if kw == "mono_drive_rate":
            _new = default - 2

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            kwargs = {kw: _new}

        else:
            kwargs = {kw: default + 2}

        new_sim = MonopileInstallation(
            config,
            log_level="INFO",
            print_logs=False,
            tp_connection_type="grouted",
            **kwargs,
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
