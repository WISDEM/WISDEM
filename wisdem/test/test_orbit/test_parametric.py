__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

from copy import deepcopy

import pandas as pd
import pytest
from benedict import benedict

from wisdem.orbit import ProjectManager, ParametricManager
from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.phases.install import TurbineInstallation
from wisdem.test.test_orbit.data import test_weather

complete_project = extract_library_specs("config", "complete_project")
weather_df = pd.DataFrame(test_weather).set_index("datetime")


params = {"site.distance": [20, 40, 60]}

funcs = {"bos_capex": lambda project: project.bos_capex}


def test_for_equal_results():

    config = benedict(deepcopy(complete_project))
    config["site.distance"] = 20
    project = ProjectManager(config)
    project.run()

    parametric = ParametricManager(config, params, funcs)
    parametric.run()
    df = parametric.results.set_index("site.distance")
    assert df.loc[20]["bos_capex"] == project.bos_capex


def test_weather():

    without = ParametricManager(complete_project, params, funcs)
    without.run()

    weathered = ParametricManager(complete_project, params, funcs, weather=weather_df)
    weathered.run()

    assert all(weathered.results["bos_capex"] > without.results["bos_capex"])


def test_individual_phase():

    config = benedict(deepcopy(complete_project))
    config["site.distance"] = 20
    phase = TurbineInstallation(config)
    phase.run()

    funcs = {"time": lambda phase: phase.total_phase_time}

    parametric = ParametricManager(complete_project, params, funcs, module=TurbineInstallation)
    parametric.run()
    df = parametric.results.set_index("site.distance")
    assert df.loc[20]["time"] == phase.total_phase_time


def test_bad_result_attribute():

    funcs = {"result": lambda phase: phase.nonexistent_result}

    parametric = ParametricManager(complete_project, params, funcs, module=TurbineInstallation)
    parametric.run()
    df = parametric.results
    assert df["result"].isnull().all()


def test_bad_result_structure():

    funcs = {"result": "bos_capex"}

    parametric = ParametricManager(complete_project, params, funcs, module=TurbineInstallation)

    with pytest.raises(TypeError):
        parametric.run()


def test_product_option():

    params = {"site.distance": [20, 40, 60], "site.depth": [20, 40, 60]}

    parametric = ParametricManager(complete_project, params, funcs, module=TurbineInstallation)

    assert parametric.num_runs == 3

    product = ParametricManager(
        complete_project,
        params,
        funcs,
        module=TurbineInstallation,
        product=True,
    )

    assert product.num_runs == 9
