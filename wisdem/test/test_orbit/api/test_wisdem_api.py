"""Tests for the Monopile Wisdem API"""

__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

import pytest
import openmdao.api as om
from wisdem.orbit.api.wisdem import Orbit


def test_wisdem_monopile_api_default():

    prob = om.Problem()
    prob.model = Orbit(floating=False, jacket=False)
    prob.setup()

    prob.run_model()

    prob.model.list_inputs()
    prob.model.list_outputs()


@pytest.mark.skip(reason="Waiting on ORBIT fix to scour model for this")
def test_wisdem_jacket_api_default():

    prob = om.Problem()
    prob.model = Orbit(floating=False, jacket=True)
    prob.setup()

    prob.run_model()

    prob.model.list_inputs()
    prob.model.list_outputs()


def test_wisdem_floating_api_default():

    prob = om.Problem()
    prob.model = Orbit(floating=True, jacket=False)
    prob.setup()

    prob.run_model()

    prob.model.list_inputs()
    prob.model.list_outputs()
