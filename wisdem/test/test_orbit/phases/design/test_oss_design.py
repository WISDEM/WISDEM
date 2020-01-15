__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "Jake.Nunemaker@nrel.gov"


import itertools
from copy import deepcopy

from wisdem.orbit.phases.design import OffshoreSubstationDesign

base = {
    "site": {"depth": 30},
    "plant": {"num_turbines": 50},
    "turbine": {"turbine_rating": 6},
    "substation_design": {},
}


def test_parameter_sweep():

    config = deepcopy(base)

    for i in itertools.product(
        range(10, 51, 10), range(3, 13, 1), range(20, 80, 10)
    ):

        config["site"]["depth"] = i[0]
        config["plant"]["num_turbines"] = i[2]
        config["turbine"]["turbine_rating"] = i[1]

        o = OffshoreSubstationDesign(config)
        o.run()

        # Substructure length
        if (
            not 10
            <= o._outputs["offshore_substation_substructure"]["length"]
            <= 80
        ):
            print(
                f"Invalid substructure length encountered: f{o._outputs['offshore_substation_substructure']['length']}"
            )
            assert False

        # Substructure mass
        if (
            not 200
            <= o._outputs["offshore_substation_substructure"]["weight"]
            <= 2500
        ):
            print(
                f"Invalid substructure weight encountered: f{o._outputs['offshore_substation_substructure']['weight']}"
            )
            assert False

        # Topside mass
        if (
            not 200
            <= o._outputs["offshore_substation_topside"]["weight"]
            <= 5000
        ):
            print(
                f"Invalid topside weight encountered: f{o._outputs['offshore_substation_topside']['weight']}"
            )
            assert False

        # Substation cost
        if not 1e6 <= o.total_phase_cost <= 300e6:
            print(f"Invalid substation cost: f{o.total_phase_cost}")
            assert False


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
        "substation_jacket_cost_rate": 7250,
        "substation_pile_cost_rate": 2500,
        "num_substations": 2,
    }

    o = OffshoreSubstationDesign(base)
    o.run()
    base_cost = o.total_phase_cost

    for k, v in test_kwargs.items():

        config = deepcopy(base)
        config["substation_design"] = {}
        config["substation_design"][k] = v

        o = OffshoreSubstationDesign(config)
        o.run()
        cost = o.total_phase_cost

        assert cost != base_cost
