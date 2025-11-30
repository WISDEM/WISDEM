"""
A test where we can modify different parts of the modeling_options to ensure
they work with a top-to-bottom WISDEM run.
"""

import os
import unittest

from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
    "examples",
    "09_floating"
)
fname_analysis_options = os.path.join(test_dir, "analysis_options.yaml")
fname_modeling_options = os.path.join(test_dir, "modeling_options.yaml")


class TestRegression(unittest.TestCase):
    def test15MW(self):
        ## IEA 15MW
        fname_wt_input = os.path.join(test_dir, "IEA-15-240-RWT_VolturnUS-S.yaml")
        wt_opt, _, _ = run_wisdem(
            fname_wt_input, fname_modeling_options, fname_analysis_options
        )

        self.assertAlmostEqual(wt_opt["rotorse.rp.AEP"][0] * 1.0e-6, 77.90013659314998, 1)
        self.assertAlmostEqual(wt_opt["rotorse.blade_mass"][0], 68233.0936092383, -1) # new value: improved interpolation
        self.assertAlmostEqual(wt_opt["financese.lcoe"][0] * 1.0e3, 85.58193678481872, 1)
        self.assertAlmostEqual(wt_opt["rotorse.rs.tip_pos.tip_deflection"][0], 25.981457969698813, 1)


if __name__ == "__main__":
    unittest.main()
