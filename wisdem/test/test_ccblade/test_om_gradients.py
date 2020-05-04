import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from wisdem.ccblade.ccblade_component import CCBladeGeometry, CCBladePower, \
    CCBladeLoads, AeroHubLoads
    

np.random.seed(314)

class Test(unittest.TestCase):

    def test_ccblade_geometry(self):
        n_input = 10
        
        prob = om.Problem()
        
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        
        # Add some arbitrary inputs
        ivc.add_output('Rtip', val=80., units='m')
        ivc.add_output('precurve_in', val=np.random.rand(n_input), units='m')
        ivc.add_output('presweep_in', val=np.random.rand(n_input), units='m')
        ivc.add_output('precone', val=2.2, units='deg')
        
        comp = CCBladeGeometry(NINPUT=n_input)
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        
        prob.setup(force_alloc_complex=True)

        prob.run_model()

        check = prob.check_partials(compact_print=True, method='fd')

        assert_check_partials(check)
        

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())