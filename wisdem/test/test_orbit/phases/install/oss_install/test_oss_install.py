"""Tests for the `OffshoreSubstationInstallation` class using feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.test.test_orbit.vessels import WTIV_SPECS, FEEDER_SPECS
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import OffshoreSubstationInstallation

config = {
    "oss_install_vessel": "example_heavy_lift_vessel",
    "feeder": "example_feeder",
    "num_feeders": 1,
    "num_substations": 1,
    "port": {"monthly_rate": 100000, "num_cranes": 1},
    "site": {"distance": 40, "depth": 15},
    "offshore_substation_topside": {
        "type": "Topside",
        "deck_space": 200,
        "weight": 400,
    },
    "offshore_substation_substructure": {
        "type": "Monopile",
        "deck_space": 200,
        "weight": 400,
        "length": 50,
    },
}


def test_creation():

    sim = OffshoreSubstationInstallation(config)
    assert sim.config == config
    assert sim.env
    assert sim.env.logger


def test_port_creation():

    sim = OffshoreSubstationInstallation(config)
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]


@pytest.mark.parametrize(
    "oss_vessel, feeder",
    [
        (WTIV_SPECS, FEEDER_SPECS),  # Passing in dictionaries
        ("example_wtiv", "example_feeder")  # Passing in vessel names to be
        # pulled from vessel library
    ],
)
def test_vessel_creation(oss_vessel, feeder):

    _config = deepcopy(config)
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

    sim = OffshoreSubstationInstallation(config)
    assert sim.num_substations == config["num_substations"]

    mp = len([item for item in sim.port.items if item["type"] == "Monopile"])
    assert sim.num_substations == mp

    ts = len([item for item in sim.port.items if item["type"] == "Topside"])
    assert sim.num_substations == ts


def test_logger_creation():

    sim = OffshoreSubstationInstallation(config)
    assert sim.env.logger.level == 20

    sim = OffshoreSubstationInstallation(config, log_level="DEBUG")
    assert sim.env.logger.level == 10


def test_full_run():

    sim = OffshoreSubstationInstallation(config, log_level="INFO")
    sim.run()

    complete = float(sim.logs["time"].max())

    assert complete > 0

    sim = OffshoreSubstationInstallation(
        config, weather=test_weather, log_level="INFO"
    )
    sim.run()

    with_weather = float(sim.logs["time"].max())

    assert with_weather >= complete


def test_for_complete_logging():

    # No weather
    sim = OffshoreSubstationInstallation(config, log_level="INFO")
    sim.run()
    df = sim.phase_dataframe.copy()
    df = df.loc[~df["agent"].isin(["Port"])]

    for vessel in df["agent"].unique():

        vl = df[df["agent"] == vessel].copy()
        vl = vl.assign(shift=(vl["time"] - vl["time"].shift(1)))

        assert (vl["shift"] - vl["duration"]).abs().max() < 1e-9

    # With weather
    sim = OffshoreSubstationInstallation(
        config, weather=test_weather, log_level="INFO"
    )
    sim.run()
    df = sim.phase_dataframe.copy()
    df = df.loc[~df["agent"].isin(["Port"])]

    for vessel in df["agent"].unique():

        vl = df[df["agent"] == vessel].copy()
        vl = vl.assign(shift=(vl["time"] - vl["time"].shift(1)))

        assert (vl["shift"] - vl["duration"]).abs().max() < 1e-9

    # With weather, multiple feeders
    config_multiple_feeders = deepcopy(config)
    config_multiple_feeders["num_feeders"] = 2
    config_multiple_feeders["num_substations"] = 2

    sim = OffshoreSubstationInstallation(
        config, weather=test_weather, log_level="INFO"
    )
    sim.run()
    df = sim.phase_dataframe.copy()
    df = df.loc[~df["agent"].isin(["Port"])]

    for vessel in df["agent"].unique():

        vl = df[df["agent"] == vessel].copy()
        vl = vl.assign(shift=(vl["time"] - vl["time"].shift(1)))

        assert (vl["shift"] - vl["duration"]).abs().max() < 1e-9


def test_for_efficiencies():

    sim = OffshoreSubstationInstallation(config)
    sim.run()

    assert (
        0
        <= sim.detailed_output["Heavy Lift Vessel_operational_efficiency"]
        <= 1
    )

    assert 0 <= sim.detailed_output["Feeder 0_operational_efficiency"] <= 1
    assert 0 <= sim.detailed_output["Feeder 0_cargo_weight_utilization"] <= 1
    assert 0 <= sim.detailed_output["Feeder 0_deck_space_utilization"] <= 1


def test_kwargs():

    sim = OffshoreSubstationInstallation(
        config, log_level="INFO", print_logs=False
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
