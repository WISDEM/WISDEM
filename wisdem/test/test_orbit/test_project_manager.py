__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy
from datetime import datetime

import pandas as pd
import pytest

from wisdem.orbit import ProjectManager
from wisdem.test.test_orbit.data import test_weather
from wisdem.orbit.library import initialize_library, extract_library_specs
from wisdem.orbit.manager import PhaseNotFound, WeatherProfileError
from wisdem.orbit.simulation.exceptions import MissingInputs

initialize_library(pytest.library)
config = extract_library_specs("config", "project_manager")


def test_for_required_phase_structure():
    """
    Automated integration test to verify that all classes listed in
    ProjectManager.possible_phases are structured correctly.
    """

    for p in ProjectManager._install_phases:

        assert isinstance(p.expected_config, dict)

    for p in ProjectManager._design_phases:

        assert isinstance(p.expected_config, dict)
        assert isinstance(p.output_config, dict)


def test_exceptions():

    incomplete_config = deepcopy(config)
    _ = incomplete_config["site"].pop("depth")

    with pytest.raises(MissingInputs):
        project = ProjectManager(incomplete_config)
        project.run_project()

    wrong_phases = deepcopy(config)
    wrong_phases["install_phases"].append("IncorrectPhaseName")

    with pytest.raises(PhaseNotFound):
        project = ProjectManager(wrong_phases)
        project.run_project()

    bad_dates = deepcopy(config)
    bad_dates["install_phases"] = {
        "MonopileInstallation": "03/01/2015",
        "TurbineInstallation": "05/01/2015",
    }

    with pytest.raises(WeatherProfileError):
        project = ProjectManager(bad_dates, weather=test_weather)
        project.run_project()


def test_phase_specific_definitions():
    """
    Tests that phase specific information makes it to phase_config.
    """

    project = ProjectManager(config)

    phase_config = project.create_config_for_phase("MonopileInstallation")

    assert phase_config["wtiv"]["name"] == "Phase Specific WTIV"
    assert phase_config["site"]["distance"] == 500

    phase_config = project.create_config_for_phase("TurbineInstallation")

    assert phase_config["wtiv"]["name"] == "Example WTIV"
    assert phase_config["site"]["distance"] == 50

    project.run_project()


def test_expected_config_merging():
    """
    Tests for merging of expected configs
    """

    config1 = {
        "site": {"distance": "float", "depth": "float"},
        "plant": {"num_turbines": "int"},
    }

    config2 = {
        "site": {"distance": "float", "wave_height": "float"},
        "monopile": {"diameter": "float"},
    }

    config = ProjectManager.merge_dicts(config1, config2)

    assert config == {
        "site": {
            "distance": "float",
            "depth": "float",
            "wave_height": "float",
        },
        "plant": {"num_turbines": "int"},
        "monopile": {"diameter": "float"},
    }


@pytest.mark.parametrize("weather", (None, test_weather))
def test_complete_run(weather):

    project = ProjectManager(config, weather=weather)
    project.run_project()

    assert len(project._output_dfs) == 2
    assert isinstance(project.project_dataframe, pd.DataFrame)

    phases = ["MonopileInstallation", "TurbineInstallation"]
    assert all(p in list(project.project_dataframe["phase"]) for p in phases)


@pytest.mark.parametrize(
    "m_start, t_start",
    [
        ("03/01/2010", "03/01/2010"),
        ("03/01/2010", "04/01/2010"),
        ("03/01/2010", "05/01/2010"),
        ("04/01/2010", "06/01/2010"),
    ],
)
def test_phase_start_dates(m_start, t_start):
    """
    Tests functionality related to passing start dates into 'install_phases' sub-dict.
    """
    config_with_start_dates = deepcopy(config)
    config_with_start_dates["install_phases"] = {
        "MonopileInstallation": m_start,
        "TurbineInstallation": t_start,
    }

    project = ProjectManager(config_with_start_dates)
    project.run_project()

    df = project.project_dataframe.copy()
    _fmt = "%m/%d/%Y"
    _target_diff = (
        datetime.strptime(t_start, _fmt) - datetime.strptime(m_start, _fmt)
    ).days * 24

    _m = df.loc[df["phase"] == "MonopileInstallation"].iloc[0]
    _t = df.loc[df["phase"] == "TurbineInstallation"].iloc[0]

    _diff = (_t["time"] - _t["duration"]) - (_m["time"] - _m["duration"])
    assert _diff == _target_diff


def test_phase_start_dates_with_weather():
    m_start = "03/01/2010"
    t_start = "05/01/2010"

    config_with_start_dates = deepcopy(config)
    config_with_start_dates["install_phases"] = {
        "MonopileInstallation": m_start,
        "TurbineInstallation": t_start,
    }

    project = ProjectManager(config_with_start_dates, weather=test_weather)
    project.run_project()

    df = project.project_dataframe.copy()
    _fmt = "%m/%d/%Y"
    _target_diff = (
        datetime.strptime(t_start, _fmt) - datetime.strptime(m_start, _fmt)
    ).days * 24

    _m = df.loc[df["phase"] == "MonopileInstallation"].iloc[0]
    _t = df.loc[df["phase"] == "TurbineInstallation"].iloc[0]

    _diff = (_t["time"] - _t["duration"]) - (_m["time"] - _m["duration"])
    assert _diff == _target_diff


def test_duplicate_phase_simulations():
    config_with_duplicates = deepcopy(config)
    config_with_duplicates["MonopileInstallation_1"] = {
        "plant": {"num_turbines": 5}
    }

    config_with_duplicates["MonopileInstallation_2"] = {
        "plant": {"num_turbines": 5},
        "site": {"distance": 100},
    }

    config_with_duplicates["install_phases"] = {
        "MonopileInstallation_1": "03/01/2010",
        "MonopileInstallation_2": "04/01/2010",
        "TurbineInstallation": "05/01/2010",
    }

    project = ProjectManager(config_with_duplicates)
    project.run_project()

    df = project.project_dataframe.groupby(["phase", "action"]).count()["time"]

    assert df.loc[("MonopileInstallation_1", "DriveMonopile")] == 5
    assert df.loc[("MonopileInstallation_2", "DriveMonopile")] == 5
    assert df.loc[("TurbineInstallation", "AttachTowerSection")] == 10


def test_design_phases():

    config_with_design = deepcopy(config)

    # Add MonopileDesign
    config_with_design["design_phases"] = ["MonopileDesign"]

    # Add required parameters
    config_with_design["site"]["mean_windspeed"] = 9
    config_with_design["turbine"]["rotor_diameter"] = 200
    config_with_design["turbine"]["rated_windspeed"] = 10
    config_with_design["monopile_design"] = {}

    # Remove monopile sub dictionary
    _ = config_with_design.pop("monopile")
    project = ProjectManager(config_with_design)
    project.run_project()

    assert isinstance(project.config["monopile"], dict)

    config_with_design["install_phases"] = {
        "MonopileInstallation": "03/01/2010",
        "TurbineInstallation": "05/01/2010",
    }

    project = ProjectManager(config_with_design, weather=test_weather)
    project.run_project()


def test_find_key_match():
    class SpecificTurbineInstallation:
        expected_config = {}

    TestProjectManager = deepcopy(ProjectManager)
    TestProjectManager._install_phases.append(SpecificTurbineInstallation)

    phase_dict = TestProjectManager.phase_dict()
    assert "SpecificTurbineInstallation" in phase_dict.keys()

    tests = [
        ("TurbineInstallation", "TurbineInstallation"),
        ("TurbineInstallation_Test", "TurbineInstallation"),
        ("TurbineInstallation Test", "TurbineInstallation"),
        ("TurbineInstallation Test_1", "TurbineInstallation"),
        ("SpecificTurbineInstallation", "SpecificTurbineInstallation"),
        ("SpecificTurbineInstallation_Test", "SpecificTurbineInstallation"),
        ("SpecificTurbineInstallation Test", "SpecificTurbineInstallation"),
        ("SpecificTurbineInstallation Test_1", "SpecificTurbineInstallation"),
    ]

    for test in tests:

        i, expected = test
        response = TestProjectManager.find_key_match(i)

        assert response.__name__ == expected

    fails = [
        "DifferentTurbineInstallation",
        "Other TurbineInstallation",
        "Extra Different TurbineInstallation_1",
    ]

    for f in fails:

        assert TestProjectManager.find_key_match(f) is None


def test_resolve_project_capacity():

    # Missing turbine rating
    config1 = {"plant": {"capacity": 600, "num_turbines": 40}}

    out1 = ProjectManager.resolve_project_capacity(config1)
    assert out1["plant"]["capacity"] == config1["plant"]["capacity"]
    assert out1["plant"]["num_turbines"] == config1["plant"]["num_turbines"]
    assert out1["turbine"]["turbine_rating"] == 15

    # Missing plant capacity
    config2 = {
        "plant": {"num_turbines": 40},
        "turbine": {"turbine_rating": 15},
    }

    out2 = ProjectManager.resolve_project_capacity(config2)
    assert out2["plant"]["capacity"] == 600
    assert out2["plant"]["num_turbines"] == config2["plant"]["num_turbines"]
    assert (
        out2["turbine"]["turbine_rating"]
        == config2["turbine"]["turbine_rating"]
    )

    # Missing number of turbines
    config3 = {"plant": {"capacity": 600}, "turbine": {"turbine_rating": 15}}

    out3 = ProjectManager.resolve_project_capacity(config3)
    assert out3["plant"]["capacity"] == config3["plant"]["capacity"]
    assert out3["plant"]["num_turbines"] == 40
    assert (
        out3["turbine"]["turbine_rating"]
        == config3["turbine"]["turbine_rating"]
    )

    # Test for float precision
    config4 = {
        "plant": {"capacity": 600, "num_turbines": 40},
        "turbine": {"turbine_rating": 15.0},
    }

    out4 = ProjectManager.resolve_project_capacity(config4)
    assert out4["plant"]["capacity"] == config4["plant"]["capacity"]
    assert out4["plant"]["num_turbines"] == config4["plant"]["num_turbines"]
    assert (
        out4["turbine"]["turbine_rating"]
        == config4["turbine"]["turbine_rating"]
    )

    # Non matching calculated value
    config5 = {
        "plant": {"capacity": 700, "num_turbines": 40},
        "turbine": {"turbine_rating": 15.0},
    }

    with pytest.raises(AttributeError):
        out5 = ProjectManager.resolve_project_capacity(config5)

    # Test for not enough information
    config6 = {"plant": {"capacity": 600}}

    out6 = ProjectManager.resolve_project_capacity(config6)
    assert out6["plant"]["capacity"] == config6["plant"]["capacity"]

    with pytest.raises(KeyError):
        turbine_rating = out6["turbine"]["turbine_rating"]

    with pytest.raises(KeyError):
        num_turbines = out6["plant"]["num_turbines"]
