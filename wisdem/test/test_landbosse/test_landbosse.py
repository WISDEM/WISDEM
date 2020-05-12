import pytest

from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
import openmdao.api as om


def test_landbosse():
    prob = om.Problem()
    prob.model = LandBOSSE()
    prob.model.options['topLevelFlag'] = True
    prob.setup()
    prob.run_model()

    assert True
