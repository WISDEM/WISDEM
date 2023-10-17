"""
A test where we can modify different parts of the modeling_options to ensure
they work with a top-to-bottom WISDEM run.
"""

import os
import unittest

from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    + os.sep
    + "examples"
    + os.sep
    + "09_floating"
    + os.sep
)
fname_analysis_options = test_dir + "analysis_options.yaml"
fname_modeling_options = test_dir + "modeling_options.yaml"


class TestRegression(unittest.TestCase):
    def test15MW(self):
        ## IEA 15MW
        fname_wt_input = test_dir + "IEA-15-240-RWT_VolturnUS-S.yaml"
        wt_opt, modeling_options, opt_options = run_wisdem(
            fname_wt_input, fname_modeling_options, fname_analysis_options
        )

        self.assertAlmostEqual(wt_opt["rotorse.blade_mass"][0], 68638.59685256994, 1) # new value: improved interpolation
        self.assertAlmostEqual(wt_opt["rotorse.rp.AEP"][0] * 1.0e-6, 77.93762832082594, 1)
        self.assertAlmostEqual(wt_opt["financese.lcoe"][0] * 1.0e3, 94.1990688526057, 1)
        self.assertAlmostEqual(wt_opt["rotorse.rs.tip_pos.tip_deflection"][0], 25.5319886366, 1)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
