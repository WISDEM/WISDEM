import numpy as np
import numpy.testing as npt
import unittest
import wisdem.plant_financese.plant_finance as pf
from openmdao.api import Problem, Group

class TestPlantFinance(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resids = {}

        self.params['sea_depth'] = 0.0
        self.params['turbine_number'] = 50
        self.params['turbine_cost'] = 1.2e6
        self.params['bos_costs'] = 7.7e6
        self.params['avg_annual_opex'] = 7e5
        self.params['fixed_charge_rate'] = 0.12
        self.params['tax_rate'] = 0.4
        self.params['discount_rate'] = 0.07
        self.params['net_aep'] = 1.6e7
        self.params['construction_time'] = 1.0
        self.params['project_lifetime'] = 20.0
        
        self.mypfin = pf.PlantFinance()

    def testRun(self):
        self.mypfin.solve_nonlinear(self.params, self.unknowns, self.resids)

        r = 0.07
        a = (1 + 0.5*((1+r)**1.0 - 1)) * (r/(1-(1+r)**(-20.0)))
        coe = (0.12*(50*1.2e6+7.7e6) + 0.6*7e5)/1.6e7
        lcoe = (a*(50*1.2e6+7.7e6) + 7e5)/1.6e7
        self.assertEqual(self.unknowns['coe'], coe)
        self.assertEqual(self.unknowns['lcoe'], lcoe)


    def testDerivatives(self):
        prob = Problem(root=Group())
        root = prob.root
        root.add('pf', pf.PlantFinance(), promotes=['*'])
        prob.setup()
        for k in self.params.keys(): prob[k] = self.params[k]        
        prob.check_total_derivatives()
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPlantFinance))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
