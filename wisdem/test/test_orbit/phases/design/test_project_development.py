"""Tests for the `ProjectDevelopment` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

import os
from copy import deepcopy

import pytest

from wisdem.orbit.library import extract_library_specs
from wisdem.orbit.phases.design import ProjectDevelopment

base = {
    "project_development": {
        "site_auction_cost": 100e6,  # USD
        "site_auction_duration": 0,  # hrs
        "site_assessment_plan_cost": 0.5e6,  # USD
        "site_assessment_plan_duration": 8760,  # hrs
        "site_assessment_cost": 50e6,  # USD
        "site_assessment_duration": 43800,  # hrs
        "construction_operations_plan_cost": 1e6,  # USD
        "construction_operations_plan_duration": 43800,  # hrs
        "boem_review_cost": 0,  # No cost to developer
        "boem_review_duration": 8760,  # hrs
        "design_install_plan_cost": 0.25e6,  # USD
        "design_install_plan_duration": 8760,  # hrs
    }
}


def test_run():

    dev = ProjectDevelopment(base)
    dev.run()


def test_defaults():

    for k, _ in base["project_development"].items():

        _config = deepcopy(base)
        _config["project_development"].pop(k)

        dev = ProjectDevelopment(_config, library_path=pytest.library)
        dev.run()

        defaults = extract_library_specs("defaults", "project")

        _split = k.split("_")
        n = "_".join(_split[:-1])
        t = _split[-1]
        val = dev._outputs[n][t]
        assert val == defaults[k]
