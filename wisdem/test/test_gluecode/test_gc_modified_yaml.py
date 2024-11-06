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
    + "02_reference_turbines"
    + os.sep
)
fname_analysis_options = test_dir + "analysis_options.yaml"
this_dir = os.path.dirname(os.path.realpath(__file__))
fname_modeling_options = this_dir + os.sep + "modified_modeling_options.yaml"


class TestRegression(unittest.TestCase):
    def test15MW(self):
        ## IEA 15MW
        fname_wt_input = test_dir + "IEA-15-240-RWT.yaml"
        wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        self.assertAlmostEqual(wt_opt["rotorse.blade_mass"][0], 68206.4068005262, 1) # new value: improved interpolation
        self.assertAlmostEqual(wt_opt["rotorse.rp.AEP"][0] * 1.0e-6, 76.75615927114029, 1)
        self.assertAlmostEqual(wt_opt["financese.lcoe"][0] * 1.0e3, 85.9014276331142, 1)
        self.assertAlmostEqual(wt_opt["rotorse.rs.tip_pos.tip_deflection"][0], 23.997879562354576, 1)
        self.assertAlmostEqual(wt_opt["towerse.z_param"][-1], 144.386, 3)


if __name__ == "__main__":
    unittest.main()
