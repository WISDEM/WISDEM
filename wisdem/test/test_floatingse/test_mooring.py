import unittest

import numpy as np
import numpy.testing as npt

import wisdem.floatingse.mooring as mm


class TestMooring(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}

        myones = np.ones(1)
        self.inputs["fairlead"] = 10.0 * myones
        self.inputs["fairlead_radius"] = 11.0 * myones
        self.inputs["anchor_radius"] = 250.0 * myones
        self.inputs["anchor_mass"] = 100.0 * myones
        self.inputs["anchor_cost"] = 10.0 * myones
        self.inputs["anchor_max_vertical_load"] = 1e10
        self.inputs["anchor_max_lateral_load"] = 1e10

        self.inputs["rho_water"] = 1e3
        self.inputs["water_depth"] = 200.0  # 100.0

        self.inputs["line_length"] = 0.6 * (self.inputs["water_depth"] + self.inputs["anchor_radius"]) * myones
        self.inputs["line_diameter"] = 0.1 * myones
        self.inputs["line_mass_density_coeff"] = 10.0 * myones
        self.inputs["line_stiffness_coeff"] = 2e6 * myones
        self.inputs["line_breaking_load_coeff"] = 30.0 * myones
        self.inputs["line_cost_rate_coeff"] = 40.0 * myones
        self.inputs["max_surge_fraction"] = 0.2
        self.inputs["survival_heel"] = np.deg2rad(10.0)
        self.inputs["operational_heel"] = np.deg2rad(5.0)

        opt = {}
        opt["n_attach"] = 3
        opt["n_anchors"] = 6
        opt["line_anchor"] = ["custom"]  # ["drag_embedment"]
        opt["line_material"] = ["custom"]  # ["chain"]

        self.mymap = mm.Mooring(options=opt, gamma=1.35)

    def testGeometry(self):
        self.mymap.compute(self.inputs, self.outputs)
        npt.assert_equal(self.outputs["constr_mooring_length"], 0.6 * 450 / (0.95 * 440))

        self.inputs["line_length"] = 150.0
        self.mymap.compute(self.inputs, self.outputs)
        npt.assert_equal(self.outputs["constr_mooring_length"], 150 / (200 - 10 - 1.35 * 11 * np.sin(np.deg2rad(10))))

    def testRunMap(self):
        self.mymap.compute(self.inputs, self.outputs)

        npt.assert_almost_equal(self.outputs["mooring_neutral_load"].sum(axis=0)[:2], 0.0, decimal=2)
        self.assertEqual(self.outputs["mooring_neutral_load"].shape[0], 3)
        self.assertGreater(self.outputs["constr_axial_load"], 1.0)
        self.assertGreater(self.outputs["max_surge_restoring_force"], 1e3)

        npt.assert_almost_equal(self.outputs["operational_heel_restoring_force"][[1, 3, 5]], 0.0, 2)
        self.assertGreater(self.outputs["operational_heel_restoring_force"][0], 0.0)
        self.assertGreater(np.abs(self.outputs["operational_heel_restoring_force"][2]), 1e3)
        self.assertGreater(np.abs(self.outputs["operational_heel_restoring_force"][4]), 1e3)

        npt.assert_almost_equal(self.outputs["survival_heel_restoring_force"][[1, 3, 5]], 0.0, 2)
        self.assertGreater(
            self.outputs["survival_heel_restoring_force"][0], self.outputs["operational_heel_restoring_force"][0]
        )
        self.assertGreater(np.abs(self.outputs["survival_heel_restoring_force"][2]), 1e3)
        self.assertGreater(
            np.abs(self.outputs["survival_heel_restoring_force"][4]),
            np.abs(self.outputs["operational_heel_restoring_force"][4]),
        )

        self.assertAlmostEqual(self.outputs["mooring_mass"], 6 * 270 * 0.1 + 6 * 100)
        self.assertAlmostEqual(self.outputs["mooring_cost"], 6 * 270 * 0.4 + 6 * 10)

    def testRunMap_MoorProps(self):

        opt = {}
        opt["n_attach"] = 3
        opt["n_anchors"] = 6
        opt["line_anchor"] = ["drag_embedment"]
        opt["line_material"] = ["chain"]

        mymap = mm.Mooring(options=opt, gamma=1.35)

        mymap.compute(self.inputs, self.outputs)
        self.assertAlmostEqual(self.outputs["mooring_mass"], 6 * 270 * 199.0 + 6 * 0)
        self.assertAlmostEqual(self.outputs["mooring_cost"], 6 * 270 * 514.415)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMooring))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
