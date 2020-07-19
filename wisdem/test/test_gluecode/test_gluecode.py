import unittest
import os
import numpy as np
from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) ) + os.sep + 'examples' + os.sep + 'reference_turbines_lcoe' + os.sep

class TestRegression(unittest.TestCase):
    
    def testAssembly(self):
        
        ## NREL 5MW
        fname_wt_input         = test_dir + os.sep + 'nrel5mw.yaml'
        fname_modeling_options = test_dir + 'modeling_options.yaml'
        fname_analysis_options = test_dir + 'analysis_options.yaml'
        fname_wt_output        = test_dir + os.sep + 'nrel5mw_update_output.yaml'
        folder_output          = 'temp'

        wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options, fname_wt_output, folder_output)

        #print(wt_opt['elastic.precomp.blade_mass'])

        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 16403.682326940743)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 24.48408190614509)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 101.19655206273262)
        np.testing.assert_almost_equal(wt_opt['wt.elastic.rail.constr_LV_4axle_horiz'], np.array([3.60953685e-01, 3.51236129e-11]))

        # ## IEA 15MW
        # fname_wt_input         = test_dir + 'IEA-15-240-RWT_WISDEMieaontology4all.yaml'
        # fname_analysis_options = test_dir + 'analysis_options.yaml'
        # fname_opt_options      = test_dir + 'optimization_options.yaml'
        # fname_wt_output        = test_dir + 'IEA-15-240-RWT_WISDEMieaontology4all2.yaml'
        # folder_output          = 'temp'

        # wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_analysis_options, fname_opt_options, fname_wt_output, folder_output)

        # print(wt_opt['elastic.precomp.blade_mass'])

        # self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 16403.682326940743)
        # self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 23.84091289784652)
        # self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 104.10860204482952)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
