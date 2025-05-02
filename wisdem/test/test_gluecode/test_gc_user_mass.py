"""
A test where we can modify different parts of the modeling_options to ensure
they work with a top-to-bottom WISDEM run.
"""

import os
import unittest

from wisdem.glue_code.runWISDEM import run_wisdem
import numpy.testing as npt

test_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
    "examples", "11_user_custom")


class TestRegression(unittest.TestCase):
    def test15MW(self):
        ## IEA 15MW
        fname_wt_input = os.path.join(test_dir, "nrel5mw-spar_oc3_user_mass.yaml")
        fname_modeling_options = os.path.join(test_dir, "modeling_options.yaml")
        fname_analysis_options = os.path.join(test_dir, "analysis_options.yaml")
        wt_opt, modeling_options, opt_options = run_wisdem(
            fname_wt_input, fname_modeling_options, fname_analysis_options
        )

        self.assertEqual(wt_opt["drivese.hub_mass"][0], 2700.0)
        self.assertEqual(wt_opt["drivese.spinner_mass"][0], 500.0)
        self.assertEqual(wt_opt["drivese.pitch_mass"][0], 8300.0)
        self.assertEqual(wt_opt["drivese.mb1_mass"][0], 1500.0)
        self.assertEqual(wt_opt["drivese.mb2_mass"][0], 1350.0)
        self.assertEqual(wt_opt["drivese.bedplate_mass"][0], 20000.0)
        self.assertEqual(wt_opt["drivese.brake_mass"][0], 5500.0)
        self.assertEqual(wt_opt["drivese.converter_mass"][0], 4200.0)
        self.assertEqual(wt_opt["drivese.transformer_mass"][0], 11500.0)
        self.assertEqual(wt_opt["drivese.gearbox_mass"][0], 21500.0)
        self.assertEqual(wt_opt["drivese.generator_mass"][0], 19500.0)
        self.assertEqual(wt_opt["towerse.tower_mass"][0], 250000.0)
        self.assertAlmostEqual(wt_opt["floatingse.platform_hull_mass"][0], 203000.0)


    def testElastic(self):
        fname_wt_input = os.path.join(test_dir, "IEA-15-240-RWT_VolturnUS-S_user_elastic.yaml")
        fname_modeling_options = os.path.join(test_dir, "modeling_options_user_elastic.yaml")
        fname_analysis_options = os.path.join(test_dir, "analysis_options_null.yaml")
        wt_opt, modeling_options, opt_options = run_wisdem(
            fname_wt_input, fname_modeling_options, fname_analysis_options
        )

        npt.assert_almost_equal(wt_opt['drivese.hub_system_mass'][0], 69360.0)
        npt.assert_almost_equal(wt_opt['drivese.hub_system_I'], [973520., 619970., 619970., 0.0, 0.0, 0.0])
        npt.assert_almost_equal(wt_opt['drivese.generator_rotor_I'], [9912933., 10862815., 10360761.])
        npt.assert_almost_equal(wt_opt['drivese.above_yaw_mass'][0], 675175.0)
        npt.assert_almost_equal(wt_opt['drivese.yaw_mass'][0], 0)
        npt.assert_almost_equal(wt_opt['drivese.above_yaw_cm'], [-4.528, -0.14 ,  4.098])
        npt.assert_almost_equal(wt_opt['drivese.above_yaw_I_TT'], [ 9912933., 10862815., 10360761., 0.0, 0.0, 0.0])
        npt.assert_almost_equal(wt_opt['drivese.above_yaw_I'], [ 9912933., 10862815., 10360761., 0.0, 0.0, 0.0])
        npt.assert_almost_equal(wt_opt['nacelle.uptilt'][0], 0.10471976)
        npt.assert_almost_equal(wt_opt['nacelle.overhang'][0], 12.032)
        npt.assert_almost_equal(wt_opt['nacelle.distance_tt_hub'][0], 5.6141)
        
if __name__ == "__main__":
    unittest.main()
