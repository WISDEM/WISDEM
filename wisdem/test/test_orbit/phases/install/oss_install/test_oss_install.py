"""Tests for the `OffshoreSubstationInstallation` class using feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.library import initialize_library, extract_library_specs
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import OffshoreSubstationInstallation

initialize_library(pytest.library)
config_single = extract_library_specs("config", "oss_install")
config_multi = extract_library_specs("config", "oss_install")
config_multi["num_feeders"] = 2

WTIV_SPECS = extract_library_specs("wtiv", "test_wtiv")
FEEDER_SPECS = extract_library_specs("feeder", "test_feeder")


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
def test_creation(config):

    sim = OffshoreSubstationInstallation(config)
    assert sim.config == config
    assert sim.env
    assert sim.env.logger


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
def test_port_creation(config):

    sim = OffshoreSubstationInstallation(config)
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]


@pytest.mark.parametrize(
    "oss_vessel, feeder",
    [
        (WTIV_SPECS, FEEDER_SPECS),  # Passing in dictionaries
        (
            "test_wtiv",
            "test_feeder",
        ),  # Passing names to be pulled from library
    ],
    ids=["dictionary", "names"],
)
def test_vessel_creation(oss_vessel, feeder):

    _config = deepcopy(config_single)
    _config["oss_install_vessel"] = oss_vessel
    _config["feeder"] = feeder

    sim = OffshoreSubstationInstallation(_config)
    assert sim.oss_vessel
    assert sim.oss_vessel.jacksys
    assert sim.oss_vessel.crane

    assert len(sim.feeders) == _config["num_feeders"]

    for feeder in sim.feeders:
        assert feeder.jacksys


def test_component_creation():

    sim = OffshoreSubstationInstallation(config_single)
    assert sim.num_substations == config_single["num_substations"]

    mp = len([item for item in sim.port.items if item["type"] == "Monopile"])
    assert sim.num_substations == mp

    ts = len([item for item in sim.port.items if item["type"] == "Topside"])
    assert sim.num_substations == ts


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "log_level,expected", (("INFO", 20), ("DEBUG", 10)), ids=["info", "debug"]
)
def test_logger_creation(config, log_level, expected):

    sim = OffshoreSubstationInstallation(config, log_level=log_level)
    assert sim.env.logger.level == expected


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_full_run(config, weather):

    sim = OffshoreSubstationInstallation(
        config, weather=weather, log_level="INFO"
    )
    sim.run()

    complete = float(sim.logs["time"].max())

    assert complete > 0


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_for_complete_logging(weather, config):

    # No weather
    sim = OffshoreSubstationInstallation(
        config, weather=weather, log_level="INFO"
    )
    sim.run()
    df = sim.phase_dataframe.copy()
    df = df.loc[~df["agent"].isin(["Port", "Test Port"])]

    for vessel in df["agent"].unique():
        _df = df[df["agent"] == vessel].copy()
        _df = _df.assign(shift=(_df["time"] - _df["time"].shift(1)))
        assert (_df["shift"] - _df["duration"]).abs().max() < 1e-9


@pytest.mark.parametrize(
    "config",
    (config_single, config_multi),
    ids=["single_feeder", "multi_feeder"],
)
def test_for_efficiencies(config):

    sim = OffshoreSubstationInstallation(config)
    sim.run()

    assert (
        0
        <= sim.detailed_output["Heavy Lift Vessel_operational_efficiency"]
        <= 1
    )
    if sim.feeders is None:
        assert (
            0
            <= sim.detailed_output[
                "Heavy Lift Vessel_cargo_weight_utilization"
            ]
            <= 1
        )
        assert (
            0
            <= sim.detailed_output["Heavy Lift Vessel_deck_space_utilization"]
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


def test_kwargs():

    sim = OffshoreSubstationInstallation(
        config_single, log_level="INFO", print_logs=False
    )
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "mono_embed_len",
        "mono_drive_rate",
        "mono_fasten_time",
        "mono_release_time",
        "grout_cure_time",
        "grout_pump_time",
        "site_position_time",
        "crane_reequip_time",
        "rov_survey_time",
        "topside_fasten_time",
        "topside_release_time",
        "topside_attach_time",
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

        new_sim = OffshoreSubstationInstallation(
            config_single, log_level="INFO", print_logs=False, **kwargs
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
