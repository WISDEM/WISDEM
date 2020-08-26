"""Tests for the Monopile Wisdem API"""

__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import openmdao.api as om

from wisdem.orbit.api.wisdem import OrbitWisdemFixed


def test_wisdem_monopile_api_default():

    prob = om.Problem()
    prob.model = OrbitWisdemFixed()
    prob.setup()

    prob.run_driver()

    prob.model.list_inputs()
    prob.model.list_outputs()
