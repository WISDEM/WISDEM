__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


from copy import deepcopy
from itertools import product

import pytest

from wisdem.orbit.phases.design import SemiSubmersibleDesign

base = {
    "site": {"depth": 500},
    "plant": {"num_turbines": 50},
    "turbine": {"turbine_rating": 12},
    "semisubmersible_design": {},
}


@pytest.mark.parametrize("depth,turbine_rating", product(range(100, 1201, 200), range(3, 15, 1)))
def test_parameter_sweep(depth, turbine_rating):

    config = {
        "site": {"depth": depth},
        "plant": {"num_turbines": 50},
        "turbine": {"turbine_rating": turbine_rating},
        "substation_design": {},
    }

    s = SemiSubmersibleDesign(config)
    s.run()

    assert s.detailed_output["stiffened_column_mass"] > 0
    assert s.detailed_output["truss_mass"] > 0
    assert s.detailed_output["heave_plate_mass"] > 0
    assert s.detailed_output["secondary_steel_mass"] > 0


def test_design_kwargs():

    test_kwargs = {
        "stiffened_column_CR": 3000,
        "truss_CR": 6000,
        "heave_plate_CR": 6000,
        "secondary_steel_CR": 7000,
    }

    s = SemiSubmersibleDesign(base)
    s.run()
    base_cost = s.total_cost

    for k, v in test_kwargs.items():

        config = deepcopy(base)
        config["semisubmersible_design"] = {}
        config["semisubmersible_design"][k] = v

        s = SemiSubmersibleDesign(config)
        s.run()
        cost = s.total_cost

        assert cost != base_cost
