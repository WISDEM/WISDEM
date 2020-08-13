import unittest
import os
import numpy as np
from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) ) + os.sep + 'examples' + os.sep + 'reference_turbines_lcoe' + os.sep

class TestRegression(unittest.TestCase):
    
    def testAssembly(self):
        
        ## NREL 5MW
        fname_wt_input         = test_dir + 'nrel5mw.yaml'
        fname_modeling_options = test_dir + 'modeling_options.yaml'
        fname_analysis_options = test_dir + 'analysis_options.yaml'
        fname_wt_output        = test_dir + 'nrel5mw_update_output.yaml'
        folder_output          = 'temp'

        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options, fname_wt_output, folder_output)

        #print(wt_opt['elastic.precomp.blade_mass'])

        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 16403.682326940743)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 24.48408190614509)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 58.515005536351104)

        ## IEA 15MW
        fname_wt_input         = test_dir + 'IEA-15-240-RWT.yaml'
        fname_wt_output        = test_dir + 'IEA-15-240-RWT_update_output.yaml'

        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options, fname_wt_output, folder_output)

        # print(wt_opt['elastic.precomp.blade_mass'])

        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 73310.0985877902)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 78.4607793182)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 60.34449341210108)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
