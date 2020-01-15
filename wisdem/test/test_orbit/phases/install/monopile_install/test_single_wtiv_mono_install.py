"""Tests for the `MonopileInstallation` class without feeder barges."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.test.test_orbit.vessels import WTIV_SPECS
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import MonopileInstallation

config = {
    "wtiv": WTIV_SPECS,
    "site": {"depth": 40, "distance": 50},
    "plant": {"num_turbines": 20},
    "turbine": {"hub_height": 100},
    "port": {"num_cranes": 1, "monthly_rate": 100000},
    "monopile": {
        "type": "Monopile",
        "length": 50,
        "diameter": 10,
        "deck_space": 500,
        "weight": 350,
    },
    "transition_piece": {
        "type": "Transition Piece",
        "deck_space": 250,
        "weight": 350,
    },
}


def test_creation():

    sim = MonopileInstallation(config, print_logs=False)
    assert sim.config == config
    assert sim.env
    assert sim.env.logger


def test_port_creation():

    sim = MonopileInstallation(config, print_logs=False)
    assert sim.port
    assert sim.port.crane.capacity == config["port"]["num_cranes"]


@pytest.mark.parametrize("wtiv", [(WTIV_SPECS), ("example_wtiv")])
def test_vessel_creation(wtiv):

    _config = deepcopy(config)
    _config["wtiv"] = wtiv

    sim = MonopileInstallation(_config, print_logs=False)
    assert sim.wtiv
    assert sim.wtiv.jacksys
    assert sim.wtiv.crane


def test_monopile_creation():

    sim = MonopileInstallation(config, print_logs=False)
    assert sim.num_monopiles == config["plant"]["num_turbines"]

    mp = len([item for item in sim.port.items if item["type"] == "Monopile"])
    assert sim.num_monopiles == mp

    tp = len(
        [item for item in sim.port.items if item["type"] == "Transition Piece"]
    )
    assert sim.num_monopiles == tp


def test_logger_creation():

    sim = MonopileInstallation(config, log_level="INFO")
    assert sim.env.logger.level == 20

    sim = MonopileInstallation(config, log_level="DEBUG")
    assert sim.env.logger.level == 10


def test_full_run():

    sim = MonopileInstallation(config, log_level="INFO")
    sim.run()

    complete = float(sim.logs["time"].max())

    assert complete > 0

    sim = MonopileInstallation(config, weather=test_weather, log_level="INFO")
    sim.run()

    with_weather = float(sim.logs["time"].max())

    assert with_weather >= complete


def test_for_complete_logging():

    sim = MonopileInstallation(config, log_level="INFO")
    sim.run()

    df = sim.phase_dataframe.copy()
    df = df.loc[~df["agent"].isin(["Port"])]
    df = df.assign(shift=(df["time"] - df["time"].shift(1)))

    assert (df["duration"] - df["shift"]).max() < 1e-9


def test_for_efficiencies():

    sim = MonopileInstallation(config)
    sim.run()

    assert 0 <= sim.detailed_output["Example WTIV_operational_efficiency"] <= 1
    assert (
        0 <= sim.detailed_output["Example WTIV_cargo_weight_utilization"] <= 1
    )
    assert 0 <= sim.detailed_output["Example WTIV_deck_space_utilization"] <= 1


def test_kwargs():

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


def test_grout_kwargs():

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
