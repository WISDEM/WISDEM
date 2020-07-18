import unittest
import os

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from wisdem.commonse.distribution import WeibullCDF, WeibullWithMeanCDF, RayleighCDF
    

np.random.seed(314)

class Test(unittest.TestCase):

    def test_distributions(self):
        nspline = 10
    
        prob = om.Problem()
    
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    
        # Add some arbitrary inputs
        ivc.add_output('x', val=np.random.rand(nspline), units='m/s')
        ivc.add_output('xbar', val=1.5, units='m/s')
        ivc.add_output('k', val=1.5)
        ivc.add_output('A', val=1.2)
    
        prob.model.add_subsystem('comp1', WeibullCDF(nspline=nspline), promotes_inputs=['*'])
        prob.model.add_subsystem('comp2', WeibullWithMeanCDF(nspline=nspline), promotes_inputs=['*'])
        prob.model.add_subsystem('comp3', RayleighCDF(nspline=nspline), promotes_inputs=['*'])
    
        prob.setup(force_alloc_complex=True)
    
        prob.run_model()
    
        check = prob.check_partials(out_stream=None, compact_print=True, method='fd')
    
        assert_check_partials(check)
    
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
