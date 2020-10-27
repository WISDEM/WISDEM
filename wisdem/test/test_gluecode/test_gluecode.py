import unittest
import os
from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( os.path.realpath(__file__) ) ) ) ) + os.sep + 'examples' + os.sep + 'reference_turbines_lcoe' + os.sep
fname_modeling_options = test_dir + 'modeling_options.yaml'
fname_analysis_options = test_dir + 'analysis_options.yaml'

class TestRegression(unittest.TestCase):

    def test5MW(self):

        ## NREL 5MW
        fname_wt_input         = test_dir + 'nrel5mw.yaml'

        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 16403.682326940743, 2)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 24.0796812417, 2)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 49.848388653574716, 2)

    def test15MW(self):
        ## IEA 15MW
        fname_wt_input         = test_dir + 'IEA-15-240-RWT.yaml'
        wt_opt, modeling_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        self.assertAlmostEqual(wt_opt['elastic.precomp.blade_mass'][0], 73310.0985877902, 1)
        self.assertAlmostEqual(wt_opt['sse.AEP'][0]*1.e-6, 78.0371305939, 1)
        self.assertAlmostEqual(wt_opt['financese.lcoe'][0]*1.e3, 66.14272881152604, 1)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
