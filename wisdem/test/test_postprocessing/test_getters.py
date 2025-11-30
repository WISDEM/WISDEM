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

fname_modeling_options = os.path.join(test_dir, "modeling_options_iea15.yaml")
fname_analysis_options = os.path.join(test_dir, "analysis_options.yaml")


class TestGetters(unittest.TestCase):

    def testAll(self):
        ## IEA 15MW
        fname_wt_input = os.path.join(test_dir, "IEA-15-240-RWT.yaml")
        prob, _, _ = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

        z_global = np.array([-75.   , -30.   , -20.   , -10.   ,   0.   ,  10.   ,  15.   ,  28.   ,  41.   ,  54.   ,  67.   ,  80.   ,  93.   , 106.   ,  119.   , 132.   , 144.386])
        diam_global = np.array([10.   , 10.   , 10.   , 10.   , 10.   , 10.   , 10.   , 10.   , 9.926,  9.443,  8.833,  8.151,  7.39 ,  6.909,  6.748,  6.572, 6.5  ])
        t_global = np.array([0.055341, 0.055341, 0.051509, 0.047517, 0.043527, 0.041058, 0.039496, 0.036456, 0.033779, 0.032192, 0.030708, 0.029101, 0.027213, 0.024009, 0.020826, 0.023998])
        
        self.assertFalse(getter.is_floating(prob))
        self.assertTrue(getter.is_monopile(prob))
        npt.assert_almost_equal(getter.get_tower_diameter(prob), diam_global)
        npt.assert_almost_equal(getter.get_tower_thickness(prob), t_global)
        npt.assert_almost_equal(getter.get_zpts(prob), z_global)
        npt.assert_almost_equal(getter.get_section_height(prob), np.diff(z_global))
        npt.assert_equal(getter.get_transition_height(prob), 15.0)
        npt.assert_equal(getter.get_tower_outfitting(prob), 1.07)
        npt.assert_equal(getter.get_tower_E(prob), 2e11)
        npt.assert_equal(getter.get_tower_G(prob), 7.93e10)
        npt.assert_almost_equal(getter.get_tower_rho(prob), 7800.0)
        npt.assert_almost_equal(getter.get_tower_mass(prob), 853460.1173568, 1)
        npt.assert_almost_equal(getter.get_tower_cost(prob), 2394873.3738557, 1)
        npt.assert_almost_equal(getter.get_monopile_mass(prob), 1319239.1, 1)
        npt.assert_almost_equal(getter.get_monopile_cost(prob), 3108099.1, 1)
        npt.assert_almost_equal(getter.get_structural_mass(prob), getter.get_tower_mass(prob)+getter.get_monopile_mass(prob))
        npt.assert_almost_equal(getter.get_structural_cost(prob), getter.get_tower_cost(prob)+getter.get_monopile_cost(prob))
        npt.assert_almost_equal(getter.get_tower_freqs(prob), np.array([0.1744686, 0.175398 , 0.7489153, 0.8640253, 0.9475382, 1.8933329]))
        npt.assert_almost_equal(getter.get_tower_cm(prob), 52.18670343496422, 2)
        npt.assert_almost_equal(getter.get_tower_cg(prob), 52.18670343496422, 2)

        shapeDF = getter.get_blade_shape(prob)
        self.assertEqual(len(shapeDF), 30)
        self.assertEqual(len(shapeDF.columns), 9)

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
        self.assertEqual(len(towDF), 32)
        self.assertEqual(len(towDF.columns), 11)


if __name__ == "__main__":
    unittest.main()
