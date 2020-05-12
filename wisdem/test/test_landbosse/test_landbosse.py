import pytest

from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
import openmdao.api as om


def test_landbosse():
    prob = om.Problem()
    prob.model = LandBOSSE()
    prob.model.options['topLevelFlag'] = True
    prob.setup()
    prob.run_model()

    landbosse_costs_by_module_type_operation = prob['landbosse_costs_by_module_type_operation']

    print(landbosse_costs_by_module_type_operation)

    assert True
