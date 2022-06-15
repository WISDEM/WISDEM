import unittest

import numpy as np
import numpy.testing as npt

import wisdem.drivetrainse.gearbox as gb


class TestGearbox(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        # 5MW inputs
        self.discrete_inputs["gear_configuration"] = "eep"
        self.discrete_inputs["shaft_factor"] = "normal"
        self.discrete_inputs["planet_numbers"] = [3, 3, 0]
        self.inputs["gear_ratio"] = 97.0
        self.inputs["rotor_diameter"] = 126.0
        self.inputs["rated_torque"] = 3946e3
        self.inputs["machine_rating"] = 5e3
        self.inputs["gearbox_mass_user"] = 0.0
        self.inputs["gearbox_length_user"] = 0.0
        self.inputs["gearbox_radius_user"] = 0.0

        self.myobj = gb.Gearbox(direct_drive=False)

    def testDirectDrive(self):
        self.myobj = gb.Gearbox(direct_drive=True)
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["stage_ratios"], 0.0)
        self.assertEqual(self.outputs["gearbox_mass"], 0.0)
        npt.assert_equal(self.outputs["gearbox_I"], 0.0)
        self.assertEqual(self.outputs["L_gearbox"], 0.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.0)

    def testEEP(self):
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # print("eep", self.outputs["stage_ratios"], self.outputs["gearbox_mass"])
        self.assertAlmostEqual(np.prod(self.outputs["stage_ratios"]), self.inputs["gear_ratio"], 1)
        # self.assertEqual(self.outputs['gearbox_mass'], 0.0)
        npt.assert_equal(
            self.outputs["gearbox_I"][0], 0.5 * self.outputs["gearbox_mass"] * 0.25 * self.outputs["D_gearbox"] ** 2
        )
        npt.assert_almost_equal(
            self.outputs["gearbox_I"][1:],
            self.outputs["gearbox_mass"]
            * (0.75 * self.outputs["D_gearbox"] ** 2 + self.outputs["L_gearbox"] ** 2)
            / 12.0,
        )
        self.assertEqual(self.outputs["L_gearbox"], 0.012 * 126.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.75 * 0.015 * 126.0)

    def testEEP3(self):
        self.discrete_inputs["gear_configuration"] = "eep_3"
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # print("eep3", self.outputs["stage_ratios"], self.outputs["gearbox_mass"])
        self.assertAlmostEqual(np.prod(self.outputs["stage_ratios"]), self.inputs["gear_ratio"], 1)
        self.assertEqual(self.outputs["stage_ratios"][-1], 3.0)
        # self.assertEqual(self.outputs['gearbox_mass'], 0.0)
        npt.assert_equal(
            self.outputs["gearbox_I"][0], 0.5 * self.outputs["gearbox_mass"] * 0.25 * self.outputs["D_gearbox"] ** 2
        )
        npt.assert_almost_equal(
            self.outputs["gearbox_I"][1:],
            self.outputs["gearbox_mass"]
            * (0.75 * self.outputs["D_gearbox"] ** 2 + self.outputs["L_gearbox"] ** 2)
            / 12.0,
        )
        self.assertEqual(self.outputs["L_gearbox"], 0.012 * 126.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.75 * 0.015 * 126.0)

    def testEEP2(self):
        self.discrete_inputs["gear_configuration"] = "eep_2"
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # print("eep2", self.outputs["stage_ratios"], self.outputs["gearbox_mass"])
        self.assertAlmostEqual(np.prod(self.outputs["stage_ratios"]), self.inputs["gear_ratio"], 1)
        self.assertEqual(self.outputs["stage_ratios"][-1], 2.0)
        # self.assertEqual(self.outputs['gearbox_mass'], 0.0)
        npt.assert_equal(
            self.outputs["gearbox_I"][0], 0.5 * self.outputs["gearbox_mass"] * 0.25 * self.outputs["D_gearbox"] ** 2
        )
        npt.assert_almost_equal(
            self.outputs["gearbox_I"][1:],
            self.outputs["gearbox_mass"]
            * (0.75 * self.outputs["D_gearbox"] ** 2 + self.outputs["L_gearbox"] ** 2)
            / 12.0,
        )
        self.assertEqual(self.outputs["L_gearbox"], 0.012 * 126.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.75 * 0.015 * 126.0)

    def testEEP_planet4_1(self):
        self.discrete_inputs["gear_configuration"] = "eep"
        self.discrete_inputs["planet_numbers"] = [4, 3, 0]
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # print("eep_4-1", self.outputs["stage_ratios"], self.outputs["gearbox_mass"])
        self.assertAlmostEqual(np.prod(self.outputs["stage_ratios"]), self.inputs["gear_ratio"], 1)
        # self.assertEqual(self.outputs['gearbox_mass'], 0.0)
        npt.assert_equal(
            self.outputs["gearbox_I"][0], 0.5 * self.outputs["gearbox_mass"] * 0.25 * self.outputs["D_gearbox"] ** 2
        )
        npt.assert_almost_equal(
            self.outputs["gearbox_I"][1:],
            self.outputs["gearbox_mass"]
            * (0.75 * self.outputs["D_gearbox"] ** 2 + self.outputs["L_gearbox"] ** 2)
            / 12.0,
        )
        self.assertEqual(self.outputs["L_gearbox"], 0.012 * 126.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.75 * 0.015 * 126.0)

    def testEEP_planet4_2(self):
        self.discrete_inputs["gear_configuration"] = "eep"
        self.discrete_inputs["planet_numbers"] = [3, 4, 0]
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # print("eep_4-2", self.outputs["stage_ratios"], self.outputs["gearbox_mass"])
        self.assertAlmostEqual(np.prod(self.outputs["stage_ratios"]), self.inputs["gear_ratio"], 1)
        # self.assertEqual(self.outputs['gearbox_mass'], 0.0)
        npt.assert_equal(
            self.outputs["gearbox_I"][0], 0.5 * self.outputs["gearbox_mass"] * 0.25 * self.outputs["D_gearbox"] ** 2
        )
        npt.assert_almost_equal(
            self.outputs["gearbox_I"][1:],
            self.outputs["gearbox_mass"]
            * (0.75 * self.outputs["D_gearbox"] ** 2 + self.outputs["L_gearbox"] ** 2)
            / 12.0,
        )
        self.assertEqual(self.outputs["L_gearbox"], 0.012 * 126.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.75 * 0.015 * 126.0)

    def testEPP(self):
        self.discrete_inputs["gear_configuration"] = "epp"
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # print("epp", self.outputs["stage_ratios"], self.outputs["gearbox_mass"])
        self.assertAlmostEqual(np.prod(self.outputs["stage_ratios"]), self.inputs["gear_ratio"], 1)
        # self.assertEqual(self.outputs['gearbox_mass'], 0.0)
        npt.assert_equal(
            self.outputs["gearbox_I"][0], 0.5 * self.outputs["gearbox_mass"] * 0.25 * self.outputs["D_gearbox"] ** 2
        )
        npt.assert_almost_equal(
            self.outputs["gearbox_I"][1:],
            self.outputs["gearbox_mass"]
            * (0.75 * self.outputs["D_gearbox"] ** 2 + self.outputs["L_gearbox"] ** 2)
            / 12.0,
        )
        self.assertEqual(self.outputs["L_gearbox"], 0.012 * 126.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.75 * 0.015 * 126.0)

    def testLargeMachine(self):
        self.inputs["gear_ratio"] = 200.0
        self.inputs["rotor_diameter"] = 200.0
        self.inputs["rotor_torque"] = 10e3
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # print("large", self.outputs["stage_ratios"], self.outputs["gearbox_mass"])
        self.assertAlmostEqual(np.prod(self.outputs["stage_ratios"]), self.inputs["gear_ratio"], 1)
        # self.assertEqual(self.outputs['gearbox_mass'], 0.0)
        npt.assert_equal(
            self.outputs["gearbox_I"][0], 0.5 * self.outputs["gearbox_mass"] * 0.25 * self.outputs["D_gearbox"] ** 2
        )
        npt.assert_almost_equal(
            self.outputs["gearbox_I"][1:],
            self.outputs["gearbox_mass"]
            * (0.75 * self.outputs["D_gearbox"] ** 2 + self.outputs["L_gearbox"] ** 2)
            / 12.0,
        )
        self.assertEqual(self.outputs["L_gearbox"], 0.012 * 200.0)
        self.assertEqual(self.outputs["D_gearbox"], 0.75 * 0.015 * 200.0)

    def testUserOverride(self):
        self.inputs["gear_ratio"] = 200.0
        self.inputs["rotor_diameter"] = 200.0
        self.inputs["rotor_torque"] = 10e3
        self.inputs["gearbox_mass_user"] = 10.0
        self.inputs["gearbox_length_user"] = 3.0
        self.inputs["gearbox_radius_user"] = 2.0
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertEqual(self.outputs["gearbox_mass"], 10.0)
        self.assertEqual(self.outputs["L_gearbox"], 3.0)
        self.assertEqual(self.outputs["D_gearbox"], 4.0)
        npt.assert_equal(self.outputs["gearbox_I"][0], 0.5 * 10 * 0.25 * 4**2)
        npt.assert_almost_equal(self.outputs["gearbox_I"][1:], 10 * (0.75 * 4**2 + 3**2) / 12.0)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGearbox))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
