"""
Testing framework for the `ExportCableInstallation` class.
"""

__author__ = ["Rob Hammond", "Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


from copy import deepcopy

import pandas as pd
import pytest

from wisdem.orbit import ProjectManager
from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.phases.install import ExportCableInstallation
from wisdem.test.test_orbit.data import test_weather

base_config = extract_library_specs("config", "export_cable_install")
simul_config = deepcopy(base_config)
_ = simul_config.pop("export_cable_bury_vessel")


@pytest.mark.parametrize("config", (base_config, simul_config), ids=["separate", "simultaneous"])
def test_simulation_setup(config):

    sim = ExportCableInstallation(config)
    assert sim.env
    assert sim.cable
    assert sim.cable.linear_density

    actions = [a["action"] for a in sim.env.actions]
    assert "Onshore Construction" in actions


@pytest.mark.parametrize("config", (base_config, simul_config), ids=["separate", "simultaneous"])
def test_vessel_initialization(config):

    sim = ExportCableInstallation(config)
    assert sim.install_vessel
    assert sim.install_vessel.cable_storage

    if config.get("export_cable_bury_vessel", None):
        assert sim.bury_vessel


@pytest.mark.parametrize("config", (base_config, simul_config), ids=["separate", "simultaneous"])
@pytest.mark.parametrize("weather", (None, test_weather), ids=["no_weather", "test_weather"])
def test_for_complete_logging(config, weather):

    sim = ExportCableInstallation(config, weather=weather)
    sim.run()

    df = pd.DataFrame(sim.env.actions)
    df = df.loc[df["action"] != "Mobilize"].reset_index(drop=True)
    df = df.assign(shift=(df["time"] - df["time"].shift(1)))

    for vessel in df["agent"].unique():
        _df = df[df["agent"] == vessel].copy()
        _df = _df.assign(shift=(_df["time"] - _df["time"].shift(1)))
        assert (_df["shift"] - _df["duration"]).fillna(0.0).abs().max() < 1e-9

    assert ~df["cost"].isnull().any()
    _ = sim.agent_efficiencies
    _ = sim.detailed_output


def test_simultaneous_speed_kwargs():

    sim = ExportCableInstallation(simul_config)
    sim.run()
    baseline = sim.total_phase_time

    key = "cable_lay_bury_speed"
    val = pt[key] * 0.1

    kwargs = {key: val}

    sim = ExportCableInstallation(simul_config, **kwargs)
    sim.run()

    assert sim.total_phase_time > baseline


def test_separate_speed_kwargs():

    sim = ExportCableInstallation(base_config)
    sim.run()
    df = pd.DataFrame(sim.env.actions)

    base_lay = sum(df.loc[df["action"] == "Lay Cable"]["duration"])
    base_bury = sum(df.loc[df["action"] == "Bury Cable"]["duration"])

    kwargs = {
        "cable_lay_speed": pt["cable_lay_speed"] * 0.1,
        "cable_bury_speed": pt["cable_bury_speed"] * 0.1,
    }

    new = ExportCableInstallation(base_config, **kwargs)
    new.run()
    df = pd.DataFrame(new.env.actions)

    new_lay = sum(df.loc[df["action"] == "Lay Cable"]["duration"])
    assert new_lay > base_lay

    new_bury = sum(df.loc[df["action"] == "Bury Cable"]["duration"])
    assert new_bury > base_bury


def test_kwargs_for_export_install():

    new_export_system = {"cable": {"linear_density": 50.0, "sections": [1000], "number": 1}}
    new_site = {"distance": 50, "depth": 20}

    new_config = deepcopy(base_config)
    new_config["export_system"] = new_export_system
    new_config["site"] = new_site

    sim = ExportCableInstallation(new_config)
    sim.run()
    baseline = sim.total_phase_time

    keywords = [
        "onshore_construction_time",
        "cable_load_time",
        "site_position_time",
        "cable_prep_time",
        "cable_lower_time",
        "cable_pull_in_time",
        "cable_termination_time",
        "cable_splice_time",
        "tow_plow_speed",
        "pull_winch_speed",
        "cable_raise_time",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]

        if "speed" in kw:
            _new = default - 0.05

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            kwargs = {kw: _new}

        else:
            kwargs = {kw: default + 2}

        new_sim = ExportCableInstallation(new_config, **kwargs)
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


def test_kwargs_for_export_install_in_ProjectManager():

    new_export_system = {
        "cable": {"linear_density": 50.0, "sections": [1000], "number": 1},
        "system_cost": 200e6,
    }
    new_site = {"distance": 50, "depth": 20}
    base = deepcopy(base_config)
    base["export_system"] = new_export_system
    base["site"] = new_site
    base["install_phases"] = ["ExportCableInstallation"]

    project = ProjectManager(base)
    project.run()
    baseline = project.phase_times["ExportCableInstallation"]

    keywords = [
        "onshore_construction_time",
        "cable_load_time",
        "site_position_time",
        "cable_prep_time",
        "cable_lower_time",
        "cable_pull_in_time",
        "cable_termination_time",
        "cable_splice_time",
        "tow_plow_speed",
        "pull_winch_speed",
        "cable_raise_time",
    ]

    failed = []

    for kw in keywords:

        default = pt[kw]

        if "speed" in kw:
            _new = default - 0.05

            if _new <= 0:
                raise Exception(f"'{kw}' is less than 0.")

            processes = {kw: _new}

        else:
            processes = {kw: default + 2}

        new_config = deepcopy(base)
        new_config["processes"] = processes

        new_project = ProjectManager(new_config)
        new_project.run()
        new_time = new_project.phase_times["ExportCableInstallation"]

        if new_time > baseline:
            pass

        else:
            failed.append(kw)

    if failed:
        raise Exception(f"ExpInstall: '{failed}' not affecting results.")

    else:
        assert True
