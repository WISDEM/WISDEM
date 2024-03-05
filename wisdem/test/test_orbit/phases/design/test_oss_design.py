__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


from copy import deepcopy
from itertools import product

import pytest

from wisdem.orbit.phases.design import OffshoreSubstationDesign

base = {
    "site": {"depth": 30},
    "plant": {"num_turbines": 50},
    "turbine": {"turbine_rating": 6},
    "substation_design": {},
}


@pytest.mark.parametrize(
    "depth,num_turbines,turbine_rating",
    product(range(10, 51, 10), range(3, 13, 1), range(20, 80, 10)),
)
def test_parameter_sweep(depth, num_turbines, turbine_rating):

    config = {
        "site": {"depth": depth},
        "plant": {"num_turbines": num_turbines},
        "turbine": {"turbine_rating": turbine_rating},
        "substation_design": {},
    }

    o = OffshoreSubstationDesign(config)
    o.run()

    # Check valid substructure length
    assert 10 <= o._outputs["offshore_substation_substructure"]["length"] <= 80

    # Check valid substructure mass
    assert 200 <= o._outputs["offshore_substation_substructure"]["mass"] <= 2500

    # Check valid topside mass
    assert 200 <= o._outputs["offshore_substation_topside"]["mass"] <= 5000

    # Check valid substation cost
    assert 1e6 <= o.total_cost <= 300e6


def test_oss_kwargs():

    test_kwargs = {
        "mpt_cost_rate": 13500,
        "topside_fab_cost_rate": 15500,
        "topside_design_cost": 5e6,
        "shunt_cost_rate": 40000,
        "switchgear_cost": 15e5,
        "backup_gen_cost": 2e6,
        "workspace_cost": 3e6,
        "other_ancillary_cost": 4e6,
        "topside_assembly_factor": 0.08,
        "oss_substructure_cost_rate": 7250,
        "oss_pile_cost_rate": 2500,
        "num_substations": 2,
    }

    o = OffshoreSubstationDesign(base)
    o.run()
    base_cost = o.total_cost

    for k, v in test_kwargs.items():

        config = deepcopy(base)
        config["substation_design"] = {}
        config["substation_design"][k] = v

        o = OffshoreSubstationDesign(config)
        o.run()
        cost = o.total_cost

        assert cost != base_cost
