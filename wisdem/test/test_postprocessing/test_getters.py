import os
import unittest
import numpy.testing as npt
import numpy as np

from wisdem.glue_code.runWISDEM import run_wisdem
import wisdem.postprocessing.wisdem_get as getter

test_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
    "examples",
    "02_reference_turbines")

fname_modeling_options = os.path.join(test_dir, "modeling_options.yaml")
fname_analysis_options = os.path.join(test_dir, "analysis_options.yaml")


class TestGetters(unittest.TestCase):

    def testAll(self):
        ## IEA 15MW
        fname_wt_input = os.path.join(test_dir, "IEA-15-240-RWT.yaml")
        prob, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        z_global = np.array([-75.0, -60.0, -45.0, -30.0, -29.166666666666664, -28.333333333333336, -27.5, -26.666666666666664, -25.833333333333336, -25.0, -24.999666666666663, -24.999333333333325, -24.998999999999988, -23.33266666666666, -21.666333333333327, -19.999999999999993, -19.999666666666663, -19.999333333333333, -19.999000000000002, -18.332666666666668, -16.666333333333334, -15.0, -14.99966666666667, -14.99933333333334, -14.99900000000001, -13.332666666666668, -11.666333333333341, -10.0, -9.99966666666667, -9.999333333333325, -9.998999999999995, -8.332666666666668, -6.666333333333327, -5.0, -4.99966666666667, -4.999333333333325, -4.998999999999995, -3.332666666666668, -1.666333333333327, 0.0, 0.00033333333334439885, 0.000666666666660376, 0.0010000000000047748, 1.6673333333333318, 3.333666666666673, 5.0, 5.00033333333333, 5.00066666666666, 5.001000000000005, 6.667333333333332, 8.333666666666673, 10.0, 10.00033333333333, 10.000666666666675, 10.001000000000005, 11.667333333333332, 13.333666666666673, 15.0, 19.333333333333332, 23.666666666666664, 28.0, 28.00033333333333, 28.000666666666667, 28.001, 32.334, 36.667, 41.0, 41.00033333333333, 41.00066666666666, 41.001, 45.333999999999996, 49.666999999999994, 54.0, 54.00033333333333, 54.00066666666667, 54.001, 58.333999999999996, 62.667, 67.0, 67.00033333333334, 67.00066666666666, 67.001, 71.334, 75.667, 80.0, 80.00033333333333, 80.00066666666666, 80.001, 84.334, 88.66699999999999, 93.0, 93.00033333333333, 93.00066666666666, 93.001, 97.334, 101.66699999999999, 106.0, 106.00033333333333, 106.00066666666666, 106.001, 110.334, 114.66699999999999, 119.0, 119.00033333333333, 119.00066666666667, 119.001, 123.334, 127.667, 132.0, 132.00033333333334, 132.0006666666667, 132.001, 136.12933333333336, 140.25766666666667, 144.386])
        diam_global = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 9.975333333333333, 9.950666666666667, 9.926, 9.926, 9.926, 9.926, 9.765, 9.604, 9.443, 9.443, 9.443, 9.443, 9.239666666666666, 9.036333333333333, 8.833, 8.833, 8.833, 8.833, 8.605666666666666, 8.378333333333334, 8.151, 8.151, 8.151, 8.151, 7.897333333333333, 7.643666666666666, 7.39, 7.39, 7.39, 7.39, 7.229666666666666, 7.069333333333334, 6.909, 6.909, 6.909, 6.909, 6.855333333333333, 6.801666666666667, 6.748, 6.748, 6.748, 6.748, 6.689333333333334, 6.6306666666666665, 6.572, 6.572, 6.572, 6.572, 6.548, 6.524, 6.5])
        t_global = np.array([0.055341, 0.055341, 0.055341, 0.055341, 0.055341, 0.055341, 0.055341, 0.055341, 0.055341, 0.054395, 0.054395, 0.054395, 0.053449, 0.053449, 0.053449, 0.052479, 0.052479, 0.052479, 0.051509, 0.051509, 0.051509, 0.050518, 0.050518, 0.050518, 0.049527, 0.049527, 0.049527, 0.048521999999999996, 0.048521999999999996, 0.048521999999999996, 0.047517, 0.047517, 0.047517, 0.046517, 0.046517, 0.046517, 0.045517, 0.045517, 0.045517, 0.044522000000000006, 0.044522000000000006, 0.044522000000000006, 0.043527, 0.043527, 0.043527, 0.042884500000000006, 0.042884500000000006, 0.042884500000000006, 0.042242, 0.042242, 0.042242, 0.04165, 0.04165, 0.04165, 0.041058, 0.041058, 0.041058, 0.039496, 0.039496, 0.039496, 0.037976, 0.037976, 0.037976, 0.036456, 0.036456, 0.036456, 0.0351175, 0.0351175, 0.0351175, 0.033779, 0.033779, 0.033779, 0.0329855, 0.0329855, 0.0329855, 0.032192, 0.032192, 0.032192, 0.03145, 0.03145, 0.03145, 0.030708, 0.030708, 0.030708, 0.0299045, 0.0299045, 0.0299045, 0.029101, 0.029101, 0.029101, 0.028157, 0.028157, 0.028157, 0.027213, 0.027213, 0.027213, 0.025611000000000002, 0.025611000000000002, 0.025611000000000002, 0.024009, 0.024009, 0.024009, 0.0224175, 0.0224175, 0.0224175, 0.020826, 0.020826, 0.020826, 0.022412, 0.022412, 0.022412, 0.023998, 0.023998, 0.023998])
        
        self.assertFalse(getter.is_floating(prob))
        self.assertTrue(getter.is_monopile(prob))
        npt.assert_almost_equal(getter.get_tower_diameter(prob), diam_global[::3])
        npt.assert_almost_equal(getter.get_tower_thickness(prob), t_global[::3])
        npt.assert_almost_equal(getter.get_zpts(prob), z_global[::3])
        npt.assert_almost_equal(getter.get_section_height(prob), np.diff(z_global[::3]))
        npt.assert_equal(getter.get_transition_height(prob), 15.0)
        npt.assert_equal(getter.get_tower_outfitting(prob), 1.07)
        npt.assert_equal(getter.get_tower_E(prob), 2e11)
        npt.assert_equal(getter.get_tower_G(prob), 7.93e10)
        npt.assert_almost_equal(getter.get_tower_rho(prob), 7800.0)
        npt.assert_almost_equal(getter.get_tower_mass(prob), 853463.2377388056)
        npt.assert_almost_equal(getter.get_tower_cost(prob), 2452297.697051666)
        npt.assert_almost_equal(getter.get_monopile_mass(prob), 1309947.6409178686)
        npt.assert_almost_equal(getter.get_monopile_cost(prob), 3196816.013637404)
        npt.assert_almost_equal(getter.get_structural_mass(prob), getter.get_tower_mass(prob)+getter.get_monopile_mass(prob))
        npt.assert_almost_equal(getter.get_structural_cost(prob), getter.get_tower_cost(prob)+getter.get_monopile_cost(prob))
        npt.assert_almost_equal(getter.get_tower_freqs(prob), np.array([0.17388500570847204, 0.1748064961937791, 0.7484669710917229, 0.8630606109609893, 0.9464114651995353, 1.8926736461662579]))
        npt.assert_almost_equal(getter.get_tower_cm(prob), 52.18670343496422)
        npt.assert_almost_equal(getter.get_tower_cg(prob), 52.18670343496422)

        shapeDF = getter.get_blade_shape(prob)
        self.assertEqual(len(shapeDF), 30)
        self.assertEqual(len(shapeDF.columns), 8)

        elasticDF = getter.get_blade_elasticity(prob)
        self.assertEqual(len(elasticDF), 30)
        self.assertEqual(len(elasticDF.columns), 17)

        rotorDF = getter.get_rotor_performance(prob)
        self.assertEqual(len(rotorDF), 20)
        self.assertEqual(len(rotorDF.columns), 13)

        nacelleDF = getter.get_nacelle_mass(prob)
        self.assertEqual(len(nacelleDF), 22)
        self.assertEqual(len(nacelleDF.columns), 16)

        towDF = getter.get_tower_table(prob)
        self.assertEqual(len(towDF), 76)
        self.assertEqual(len(towDF.columns), 11)


if __name__ == "__main__":
    unittest.main()
