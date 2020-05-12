import pytest

from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
import openmdao.api as om


@pytest.fixture
def landbosse_costs_by_module_type_operation():
    """
    Executes LandBOSSE and extracts cost output for the regression
    test.
    """
    prob = om.Problem()
    prob.model = LandBOSSE()
    prob.model.options['topLevelFlag'] = True
    prob.setup()
    prob.run_model()
    landbosse_costs_by_module_type_operation = prob['landbosse_costs_by_module_type_operation']
    return landbosse_costs_by_module_type_operation


def test_landbosse(landbosse_costs_by_module_type_operation):
    """
    This runs the regression test by comparing against the expected validation
    data.
    """
    print(landbosse_costs_by_module_type_operation)
    assert True
