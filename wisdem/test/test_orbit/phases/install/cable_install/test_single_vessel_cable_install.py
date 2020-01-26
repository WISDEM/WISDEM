"""
Provides a testing framework for the `ArrayCableInstallation` and
`ExportCableInstallation` classes.
"""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "robert.hammond@nrel.gov"


from copy import deepcopy

import pytest

from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.library import initialize_library, extract_library_specs
from wisdem.orbit.vessels.tasks import defaults
from wisdem.orbit.phases.install import ArrayCableInstallation as ArrInstall
from wisdem.orbit.phases.install import ExportCableInstallation as ExpInstall

initialize_library(pytest.library)
config_array = extract_library_specs("config", "array_cable_install")
config_export = extract_library_specs("config", "export_cable_install")

installs = ((ArrInstall, config_array), (ExpInstall, config_export))
weather = (None, test_weather)
strategies = ("simultaneous", "separate", "lay", "bury")
strategies_keys = (
    ("simultaneous", "cable_lay_bury_speed"),
    ("separate", "cable_lay_speed|cable_bury_speed"),
    ("lay", "cable_lay_speed"),
    ("bury", "cable_bury_speed"),
)


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "weather", weather, ids=["no_weather", "test_weather"]
)
def test_creation(CableInstall, config, weather):
    sim = CableInstall(config, weather=weather, print_logs=False)

    assert sim.config == config
    assert sim.env
    assert sim.env.logger


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "weather", weather, ids=["no_weather", "test_weather"]
)
@pytest.mark.parametrize("strategy", strategies)
def test_vessel_creation(CableInstall, config, weather, strategy):

    print(config.keys())

    _config = deepcopy(config)
    try:
        _config["array_system"]["strategy"] = strategy
    except KeyError:
        pass

    try:
        _config["export_system"]["strategy"] = strategy
    except KeyError:
        pass

    sim = CableInstall(_config, weather=weather, log_level="INFO")

    if strategy in ("lay", "simultaneous", "separate"):
        assert sim.cable_lay_vessel
        assert sim.cable_lay_vessel.storage.deck_space == 0
        assert sim.cable_lay_vessel.at_port
        assert not sim.cable_lay_vessel.at_site

    if strategy in ("bury", "separate"):
        assert sim.cable_bury_vessel
        assert sim.cable_bury_vessel.storage.deck_space == 0
        assert sim.cable_bury_vessel.at_port
        assert not sim.cable_bury_vessel.at_site


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "weather", weather, ids=["no_weather", "test_weather"]
)
def test_carousel_system_creation(CableInstall, config, weather):
    sim = CableInstall(config, weather=weather, log_level="INFO")

    for carousel in sim.carousels.carousels.values():
        carousel = carousel.__dict__
        carousel["type"] = "Carousel"
        assert carousel in sim.port.items


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "weather", weather, ids=["no_weather", "test_weather"]
)
@pytest.mark.parametrize(
    "log_level,expected", (("INFO", 20), ("DEBUG", 10)), ids=["info", "debug"]
)
def test_logger_creation(CableInstall, config, weather, log_level, expected):
    sim = CableInstall(config, weather=weather, log_level=log_level)
    assert sim.env.logger.level == expected


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "weather", weather, ids=["no_weather", "test_weather"]
)
@pytest.mark.parametrize(
    "strategy", strategies, ids=["simultaneous", "separate", "lay", "bury"]
)
def test_full_run_completes(CableInstall, config, weather, strategy):
    strategy_config = deepcopy(config)
    if "array_system" in strategy_config:
        strategy_config["array_system"]["strategy"] = strategy
    elif "export_system" in strategy_config:
        strategy_config["export_system"]["strategy"] = strategy

    sim = CableInstall(strategy_config, weather=weather, log_level="DEBUG")
    sim.run()

    for t in sim.logs[sim.logs.action == "Complete"]["time"]:
        assert float(t) > 0


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "weather", weather, ids=["no_weather", "test_weather"]
)
def test_full_run_is_valid(CableInstall, config, weather):
    sim = CableInstall(config, weather=weather, log_level="INFO")
    sim.run()
    n_complete = sim.logs[sim.logs.action == "TestCable"].shape[0] / 2
    assert n_complete == sim.num_sections


@pytest.mark.parametrize(
    "weather", (None, test_weather), ids=["no_weather", "test_weather"]
)
def test_trench_install_creation(weather):
    sim = ExpInstall(config_export, weather=weather, print_logs=False)
    sim.run()

    assert "DigTrench" in sim.phase_dataframe.action.tolist()


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "weather", weather, ids=["no_weather", "test_weather"]
)
def test_full_run_logging(CableInstall, config, weather):
    sim = CableInstall(config, weather=weather, log_level="INFO")
    sim.run()

    df = sim.phase_dataframe[
        (~sim.phase_dataframe.agent.isin(("Director", "Port")))
        & (sim.phase_dataframe.action != "Complete")
    ]
    df = df.assign(shift=(df.time - df.time.shift(1)))
    assert (df.duration - df["shift"]).max() == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
def test_for_array_install_efficiencies(CableInstall, config):

    sim = CableInstall(config)
    sim.run()

    vessel = sim.cable_lay_vessel.name
    assert 0 <= sim.detailed_output[f"{vessel}_operational_efficiency"] <= 1
    assert 0 <= sim.detailed_output[f"{vessel}_cargo_weight_utilization"] <= 1


@pytest.mark.parametrize(
    "CableInstall,config", installs, ids=["array", "export"]
)
@pytest.mark.parametrize(
    "strategy,key",
    strategies_keys,
    ids=["simultaneous", "separate", "lay", "bury"],
)
def test_strategy_kwargs(CableInstall, config, strategy, key):
    strategy_config = deepcopy(config)
    if "array_system" in strategy_config:
        strategy_config["array_system"]["strategy"] = strategy
    elif "export_system" in strategy_config:
        strategy_config["export_system"]["strategy"] = strategy

    sim = CableInstall(strategy_config, log_level="DEBUG")
    sim.run()
    baseline = sim.total_phase_time

    for _key in key.split("|"):
        kwargs = {_key: defaults[_key] * 0.1}
        sim = CableInstall(strategy_config, log_level="DEBUG", **kwargs)
        sim.run()
        updated = sim.total_phase_time

        assert updated > baseline


def test_kwargs_for_array_install():

    sim = ArrInstall(config_array, log_level="INFO", print_logs=False)
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
    ]

    failed = []

    for kw in keywords:

        default = defaults[kw]
        kwargs = {kw: default + 2}

        new_sim = ArrInstall(
            config_array, log_level="INFO", print_logs=False, **kwargs
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
        "strategy": "simultaneous",
        "cables": {
            "XLPE_300mm_36kV": {
                "cable_sections": [(1000, 1)],
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

    new_config = deepcopy(config_export)
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
