import pytest

from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
import openmdao.api as om


def test_landbosse():
    prob = om.Problem()
    prob.model = LandBOSSE()
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.setup()
    prob.run_driver()

    assert True
