"""
Provides a testing framework for the `ArrayCableInstallation` and
`ExportCableInstallation` classes.
"""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


import itertools
from copy import deepcopy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import ArrayCableInstallation as ArrInstall
from wisdem.orbit.phases.install import ExportCableInstallation as ExpInstall

config = {
    "port": {"num_cranes": 1},
    "array_cable_lay_vessel": "example_cable_lay_vessel",
    "export_cable_lay_vessel": "example_cable_lay_vessel",
    "trench_dig_vessel": "example_trench_dig_vessel",
    "site": {
        "distance": 50,
        "depth": 20,
        "distance_to_landfall": 30,
        "distance_to_beach": 0.0,
        "distance_to_interconnection": 3,
    },
    "plant": {"layout": "grid", "turbine_spacing": 5, "num_turbines": 40},
    "turbine": {"rotor_diameter": 154, "turbine_rating": 9},
    "array_system": {
        "strategy": "lay_bury",
        "cables": {
            "XLPE_400mm_36kV": {
                "cable_sections": [(0.81, 28)],
                "linear_density": 34.56,
            },
            "XLPE_630mm_36kV": {
                "cable_sections": [
                    (3.3644031043, 2),
                    (0.81, 8),
                    (2.3293745871000002, 2),
                    (1.3647580911000001, 2),
                ],
                "linear_density": 43.29,
            },
        },
    },
    "export_system": {
        "strategy": "lay_bury",
        "cables": {
            "XLPE_300mm_36kV": {
                "cable_sections": [(33.3502, 10)],
                "linear_density": 50.0,
            }
        },
    },
}

installs = (ArrInstall, ExpInstall)
weather = (None, test_weather)
strategies = ("lay_bury", "lay", "bury")


@pytest.mark.parametrize(
    "CableInstall,weather", itertools.product(installs, weather)
)
def test_creation(CableInstall, weather):
    sim = CableInstall(config, weather=weather, print_logs=False)

    assert sim.config == config
    assert sim.env
    assert sim.env.logger


@pytest.mark.parametrize(
    "CableInstall,weather", itertools.product(installs, weather)
)
def test_vessel_creation(CableInstall, weather):
    sim = CableInstall(config, weather=weather, log_level="INFO")

    assert sim.cable_lay_vessel
    assert sim.cable_lay_vessel.storage.deck_space == 0
    assert sim.cable_lay_vessel.at_port
    assert not sim.cable_lay_vessel.at_site


@pytest.mark.parametrize(
    "CableInstall,weather", itertools.product(installs, weather)
)
def test_carousel_system_creation(CableInstall, weather):
    sim = CableInstall(config, weather=weather, log_level="INFO")

    for carousel in sim.carousels.carousels.values():
        carousel = carousel.__dict__
        carousel["type"] = "Carousel"
        assert carousel in sim.port.items


@pytest.mark.parametrize(
    "CableInstall,weather,log_level,expected",
    (
        (c, w, l, n)
        for c in (ArrInstall, ExpInstall)
        for w in (None, test_weather)
        for l, n in (("INFO", 20), ("DEBUG", 10))
    ),
)
def test_logger_creation(CableInstall, weather, log_level, expected):
    sim = CableInstall(config, weather=weather, log_level=log_level)
    assert sim.env.logger.level == expected


@pytest.mark.parametrize(
    "CableInstall,strategy,weather",
    itertools.product(installs, strategies, weather),
)
def test_full_run_completes(CableInstall, strategy, weather):
    strategy_config = deepcopy(config)
    strategy_config["array_system"]["strategy"] = strategy
    strategy_config["export_system"]["strategy"] = strategy

    sim = CableInstall(strategy_config, weather=weather, log_level="DEBUG")
    sim.run()

    assert float(sim.logs[sim.logs.action == "Complete"]["time"]) > 0


@pytest.mark.parametrize(
    "CableInstall,weather", itertools.product(installs, weather)
)
def test_full_run_is_valid(CableInstall, weather):
    sim = CableInstall(config, weather=weather, log_level="INFO")
    sim.run()
    n_complete = sim.logs[sim.logs.action == "TestCable"].shape[0] / 2
    assert n_complete == sim.num_sections


@pytest.mark.parametrize("weather", (None, test_weather))
def test_trench_install_creation(weather):
    sim = ExpInstall(config, weather=weather, print_logs=False)
    sim.run()

    assert "DigTrench" in sim.phase_dataframe.action.tolist()


@pytest.mark.parametrize(
    "CableInstall,weather", itertools.product(installs, weather)
)
def test_full_run_logging(CableInstall, weather):
    sim = CableInstall(config, weather=weather, log_level="INFO")
    sim.run()

    df = sim.phase_dataframe[
        (~sim.phase_dataframe.agent.isin(("Director", "Port")))
        & (sim.phase_dataframe.action != "Complete")
    ]
    df = df.assign(shift=(df.time - df.time.shift(1)))
    assert (df.duration - df["shift"]).max() == pytest.approx(0, abs=1e-9)


def test_for_array_install_efficiencies():

    sim = ArrInstall(config)
    sim.run()

    assert (
        0
        <= sim.detailed_output[
            "Array Cable Installation Vessel_operational_efficiency"
        ]
        <= 1
    )
    assert (
        0
        <= sim.detailed_output[
            "Array Cable Installation Vessel_cargo_weight_utilization"
        ]
        <= 1
    )


def test_for_export_install_efficiencies():

    sim = ExpInstall(config)
    sim.run()

    assert (
        0
        <= sim.detailed_output[
            "Export Cable Installation Vessel_operational_efficiency"
        ]
        <= 1
    )
    assert (
        0
        <= sim.detailed_output[
            "Export Cable Installation Vessel_cargo_weight_utilization"
        ]
        <= 1
    )


def test_kwargs_for_array_install():

    sim = ArrInstall(config, log_level="INFO", print_logs=False)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "carousel_lift_time",
        "carousel_fasten_time",
        "site_position_time",
        "cable_prep_time",
        "cable_lower_time",
        "cable_pull_in_time",
        "cable_termination_time",
        # "cable_lay_speed",
        "cable_lay_bury_speed",
        # "cable_bury_speed",
        # "cable_splice_time",
        # "tow_plow_speed",
        # "pull_winch_speed",
        # "cable_raise_time",
        # "trench_dig_speed",
    ]

    failed = []

    for kw in keywords:

        default = defaults[kw]

        if "speed" in kw:
            _new = default - 0.05

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            kwargs = {kw: _new}

        else:
            kwargs = {kw: default + 2}

        new_sim = ArrInstall(
            config, log_level="INFO", print_logs=False, **kwargs
        )
        new_sim.run()
        new_time = new_sim.total_phase_time

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"ArrInstall: '{failed}' not affecting results.")

    else:
        assert True


def test_kwargs_for_export_install():

    new_export_system = {
        "strategy": "lay_bury",
        "cables": {
            "XLPE_300mm_36kV": {
                "cable_sections": [(88.02, 10)],
                "linear_density": 50.0,
            }
        },
    }
    new_site = {
        "distance": 50,
        "depth": 20,
        "distance_to_landfall": 30,  # landfall to site, km
        "distance_to_beach": 0.5,  # vessel has to anchor 2km from landfall site
        "distance_to_interconnection": 4,  # landfall to interconnection, km
    }

    new_config = deepcopy(config)
    new_config["export_system"] = new_export_system
    new_config["site"] = new_site

    sim = ExpInstall(new_config, log_level="INFO", print_logs=False)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "carousel_lift_time",
        "carousel_fasten_time",
        "site_position_time",
        "cable_prep_time",
        "cable_lower_time",
        "cable_pull_in_time",
        "cable_termination_time",
        # "cable_lay_speed",
        "cable_lay_bury_speed",
        # "cable_bury_speed",
        "cable_splice_time",
        "tow_plow_speed",
        "pull_winch_speed",
        "cable_raise_time",
        "trench_dig_speed",
    ]

    failed = []

    for kw in keywords:

        default = defaults[kw]

        if "speed" in kw:
            _new = default - 0.05

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            kwargs = {kw: _new}

        else:
            kwargs = {kw: default + 2}

        new_sim = ExpInstall(
            new_config, log_level="INFO", print_logs=False, **kwargs
        )
        new_sim.run()
        new_time = new_sim.total_phase_time

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"ExpInstall: '{failed}' not affecting results.")

    else:
        assert True
