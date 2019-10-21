import numpy as np
import numpy.testing as npt
import unittest
import wisdem.plant_financese.plant_finance as pf
from openmdao.api import Problem, Group

class TestPlantFinance(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.discrete_inputs = {}
        self.outputs = {}

        self.inputs['machine_rating'] = 1000.0
        self.inputs['tcc_per_kW'] = 1.2e3
        self.inputs['bos_per_kW'] = 7.7e3
        self.inputs['opex_per_kW'] = 7e2
        self.inputs['fixed_charge_rate'] = 0.12
        self.inputs['turbine_aep'] = 1.6e7
        self.inputs['park_aep'] = 1.6e7*50
        self.inputs['wake_loss_factor'] = 0.15

        self.discrete_inputs['turbine_number'] = 50
        
        self.mypfin = pf.PlantFinance()

    def testRun(self):
        # Park AEP way
        self.mypfin.compute(self.inputs, self.outputs, self.discrete_inputs, {})

        lcoe = 1e3*50*(0.12*(1.2e3+7.7e3) + 7e2) / (1.6e7*50.0)
        self.assertEqual(self.outputs['lcoe'], lcoe)

        # Wake loss way
        self.inputs['park_aep'] = 0.0
        self.mypfin.compute(self.inputs, self.outputs, self.discrete_inputs, {})

        lcoe = 1e3*50*(0.12*(1.2e3+7.7e3) + 7e2) / (1.6e7*50.0*(1-0.15))
        self.assertEqual(self.outputs['lcoe'], lcoe)


    def testDerivatives(self):
        prob = Problem()
        root = prob.model = Group()
        root.add_subsystem('pf', pf.PlantFinance(), promotes=['*'])
        prob.setup()
        for k in self.inputs.keys(): prob[k] = self.inputs[k]        
        for k in self.discrete_inputs.keys(): prob[k] = self.discrete_inputs[k]        
        #prob.check_partials()
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPlantFinance))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
