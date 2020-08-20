import unittest
import os
from wisdem.glue_code.runWISDEM import run_wisdem


test_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) ) + os.sep + 'examples' + os.sep + 'reference_turbines_lcoe' + os.sep
this_dir =  os.path.dirname( os.path.realpath(__file__) ) + os.sep + 'analysis_flag_modeling_options' + os.sep

fname_wt_input         = test_dir + 'IEA-15-240-RWT.yaml'
fname_analysis_options = test_dir + 'analysis_options.yaml'

class TestRegression(unittest.TestCase):
    
    def testRotor(self):
        fname_modeling_options = this_dir + 'modeling_options_rotor.yaml'
        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
    
        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 75900.73672359536)
    
    def testRotorServo(self):
        fname_modeling_options = this_dir + 'modeling_options_rotor_servo.yaml'
        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
    
        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 75900.73672359536)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 86.8147239905822)
    
    def testNoBOS(self):
        fname_modeling_options = this_dir + 'modeling_options_no_bos.yaml'
        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
    
        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 75900.73672359536)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 86.8147239905822)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 39.6324136052)
    
    def testNoBOSNoTower(self):
        fname_modeling_options = this_dir + 'modeling_options_no_bos_no_tower.yaml'
        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
    
        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 75900.73672359536)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 86.8147239905822)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 40.19587288277811)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
