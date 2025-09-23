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


class TestRegression(unittest.TestCase):
    def test5MW(self):
        ## NREL 5MW
        fname_wt_input = test_dir + "nrel5mw.yaml"
        fname_modeling_options = test_dir + "modeling_options_nrel5.yaml"

        wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        self.assertAlmostEqual(wt_opt["rotorse.rp.AEP"][0] * 1.0e-6, 23.65816579217035, 2)
        self.assertAlmostEqual(wt_opt["rotorse.blade_mass"][0], 16485.00727402099, 2)  # new value: improved interpolation
        self.assertAlmostEqual(wt_opt["financese.lcoe"][0] * 1.0e3, 52.752607160772826, 1)
        self.assertAlmostEqual(wt_opt["rotorse.rs.tip_pos.tip_deflection"][0], 4.484994228050976, 1)
        self.assertAlmostEqual(wt_opt["towerse.z_param"][-1], 87.7, 2)

    def test15MW(self):
        ## IEA 15MW
        fname_wt_input = test_dir + "IEA-15-240-RWT.yaml"
        fname_modeling_options = test_dir + "modeling_options_iea15.yaml"
        wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        self.assertAlmostEqual(wt_opt["rotorse.rp.AEP"][0] * 1.0e-6, 77.8999255816596, 1)
        self.assertAlmostEqual(wt_opt["rotorse.blade_mass"][0], 68233.0936092383, -1) # new value: improved interpolation
        self.assertAlmostEqual(wt_opt["financese.lcoe"][0] * 1.0e3, 75.25720380792154, 1)
        self.assertAlmostEqual(wt_opt["rotorse.rs.tip_pos.tip_deflection"][0], 25.838654170633227, 1)
        self.assertAlmostEqual(wt_opt["towerse.z_param"][-1], 144.386, 3)

    def test3p4MW(self):
        ## IEA 3.4MW
        fname_wt_input = test_dir + "IEA-3p4-130-RWT.yaml"
        fname_modeling_options = test_dir + "modeling_options_iea3p4.yaml"
        wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        self.assertAlmostEqual(wt_opt["rotorse.rp.AEP"][0] * 1.0e-6, 13.586538779055665, 1)
        self.assertAlmostEqual(wt_opt["rotorse.blade_mass"][0], 14534.711602944584, 1)  # new value: improved interpolation
        self.assertAlmostEqual(wt_opt["financese.lcoe"][0] * 1.0e3, 38.4591820608806, 1)
        self.assertAlmostEqual(wt_opt["rotorse.rs.tip_pos.tip_deflection"][0], 8.09983034562966, 1)
        self.assertAlmostEqual(wt_opt["towerse.z_param"][-1], 108.0, 3)


if __name__ == "__main__":
    unittest.main()
