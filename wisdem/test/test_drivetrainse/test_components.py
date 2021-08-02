import unittest

import numpy as np
import numpy.testing as npt
import wisdem.drivetrainse.drive_components as dc


class TestComponents(unittest.TestCase):
    def testBearing(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.MainBearing()

        discrete_inputs["bearing_type"] = "carb"
        inputs["D_bearing"] = 2.0
        inputs["D_shaft"] = 3.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        npt.assert_equal(
            outputs["mb_I"] / outputs["mb_mass"], 0.125 * np.r_[2 * (4 * 1.5 ** 2 + 3), (4 * 1.5 ** 2 + 5) * np.ones(2)]
        )
        self.assertAlmostEqual(outputs["mb_mass"], (1 + 80 / 27) * 1561.4 * 3 ** 2.6007)
        self.assertAlmostEqual(outputs["mb_max_defl_ang"], 0.5 * np.pi / 180)

        # Other valid types
        discrete_inputs["bearing_type"] = "crb"
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        discrete_inputs["bearing_type"] = "srb"
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        discrete_inputs["bearing_type"] = "trb"
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        # Invalid type
        try:
            discrete_inputs["bearing_type"] = 1
            myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        except ValueError:
            self.assertTrue(True)
        # Unknown type
        try:
            discrete_inputs["bearing_type"] = "trb1"
            myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        except ValueError:
            self.assertTrue(True)

    def testBrake(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.Brake(direct_drive=True)

        discrete_inputs["direct_drive"] = True
        inputs["rotor_diameter"] = 200.0
        inputs["rated_torque"] = 10e6
        inputs["s_rotor"] = 3.0
        inputs["s_gearbox"] = 0.0
        inputs["brake_mass_user"] = 0.0
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["brake_mass"], 12200)
        self.assertEqual(outputs["brake_cm"], 3)
        npt.assert_equal(outputs["brake_I"], 12200 * np.r_[0.5, 0.25, 0.25])

        discrete_inputs["direct_drive"] = False
        inputs["s_gearbox"] = 5.0
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["brake_mass"], 12200)
        self.assertEqual(outputs["brake_cm"], 3)
        npt.assert_equal(outputs["brake_I"], 12200 * np.r_[0.5, 0.25, 0.25])

        inputs["brake_mass_user"] = 42.0
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["brake_mass"], 42.0)
        self.assertEqual(outputs["brake_cm"], 3)
        npt.assert_equal(outputs["brake_I"], 42.0 * np.r_[0.5, 0.25, 0.25])

    def testRPM_In(self):
        inputs = {}
        outputs = {}
        myobj = dc.RPM_Input(n_pc=20)

        inputs["minimum_rpm"] = 2.0
        inputs["rated_rpm"] = 10.0
        inputs["gear_ratio"] = 100.0
        myobj.compute(inputs, outputs)

        x = np.linspace(2, 10, 20)
        npt.assert_equal(outputs["lss_rpm"], x)
        npt.assert_equal(outputs["hss_rpm"], 100 * x)

    def testGeneratorSimple(self):
        inputs = {}
        outputs = {}
        myobj = dc.GeneratorSimple(direct_drive=True, n_pc=20)

        inputs["rotor_diameter"] = 200.0
        inputs["machine_rating"] = 10e3
        inputs["rated_torque"] = 10e6
        inputs["lss_rpm"] = x = np.linspace(0.1, 10.0, 20)
        inputs["L_generator"] = 3.6 * 1.5
        inputs["generator_mass_user"] = 0.0
        inputs["generator_radius_user"] = 0.0
        inputs["generator_efficiency_user"] = 0.0
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["R_generator"], 1.5)
        m = 37.68 * 10e3
        self.assertEqual(outputs["generator_mass"], m)
        self.assertEqual(outputs["generator_rotor_mass"], 0.5 * m)
        self.assertEqual(outputs["generator_stator_mass"], 0.5 * m)
        npt.assert_equal(
            outputs["generator_I"], m * np.r_[0.5 * 1.5 ** 2, (3 * 1.5 ** 2 + (3.6 * 1.5) ** 2) / 12 * np.ones(2)]
        )
        npt.assert_equal(outputs["generator_rotor_I"], 0.5 * outputs["generator_I"])
        npt.assert_equal(outputs["generator_stator_I"], 0.5 * outputs["generator_I"])

        eff = 1.0 - (0.01007 / x * x[-1] + 0.02 + 0.06899 * x / x[-1])
        eff = np.maximum(1e-3, eff)
        npt.assert_almost_equal(outputs["generator_efficiency"], eff)

        myobj = dc.GeneratorSimple(direct_drive=False)
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["R_generator"], 1.5)
        m = np.mean([6.4737, 10.51, 5.34]) * 10e3 ** 0.9223
        self.assertEqual(outputs["generator_mass"], m)
        self.assertEqual(outputs["generator_rotor_mass"], 0.5 * m)
        self.assertEqual(outputs["generator_stator_mass"], 0.5 * m)
        npt.assert_equal(
            outputs["generator_I"], m * np.r_[0.5 * 1.5 ** 2, (3 * 1.5 ** 2 + (3.6 * 1.5) ** 2) / 12 * np.ones(2)]
        )
        npt.assert_equal(outputs["generator_rotor_I"], 0.5 * outputs["generator_I"])
        npt.assert_equal(outputs["generator_stator_I"], 0.5 * outputs["generator_I"])

        eff = 1.0 - (0.01289 / x * x[-1] + 0.0851 + 0.0 * x / x[-1])
        eff = np.maximum(1e-3, eff)
        npt.assert_almost_equal(outputs["generator_efficiency"], eff)

        eff = np.linspace(0.5, 1.0, 20)
        inputs["generator_efficiency_user"] = np.c_[x, eff]
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["generator_efficiency"], eff)

        inputs["generator_mass_user"] = 2.0
        inputs["generator_radius_user"] = 3.0
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["R_generator"], 3.0)
        self.assertEqual(outputs["generator_mass"], 2.0)
        self.assertEqual(outputs["generator_rotor_mass"], 1.0)
        self.assertEqual(outputs["generator_stator_mass"], 1.0)
        npt.assert_equal(outputs["generator_rotor_I"], 0.5 * outputs["generator_I"])
        npt.assert_equal(outputs["generator_stator_I"], 0.5 * outputs["generator_I"])

    def testElectronics(self):
        inputs = {}
        outputs = {}
        myobj = dc.Electronics()

        inputs["rotor_diameter"] = 200.0
        inputs["machine_rating"] = 10e3
        inputs["D_top"] = 5.0
        inputs["converter_mass_user"] = 0.0
        inputs["transformer_mass_user"] = 0.0
        myobj.compute(inputs, outputs)
        s = 0.015 * 200
        m = np.mean([740.0, 817.5]) * 10 + np.mean([101.37, 503.83])
        self.assertAlmostEqual(outputs["converter_mass"], m)
        npt.assert_equal(outputs["converter_cm"], np.r_[0.0, -(2.5 + 0.5 * s), 0.5 * s])
        npt.assert_almost_equal(outputs["converter_I"], (1.0 / 6.0) * m * s ** 2)

        m = 1915 * 10 + 1910.0
        self.assertEqual(outputs["transformer_mass"], m)
        npt.assert_equal(outputs["transformer_cm"], np.r_[0.0, -(2.5 + 0.5 * s), 0.5 * s])
        npt.assert_almost_equal(outputs["transformer_I"], (1.0 / 6.0) * m * s ** 2)

        inputs["converter_mass_user"] = 42.0
        inputs["transformer_mass_user"] = 420.0
        myobj.compute(inputs, outputs)
        self.assertAlmostEqual(outputs["converter_mass"], 42.0)
        npt.assert_equal(outputs["converter_cm"], np.r_[0.0, -(2.5 + 0.5 * s), 0.5 * s])
        npt.assert_almost_equal(outputs["converter_I"], (1.0 / 6.0) * 42 * s ** 2)
        self.assertEqual(outputs["transformer_mass"], 420.0)
        npt.assert_equal(outputs["transformer_cm"], np.r_[0.0, -(2.5 + 0.5 * s), 0.5 * s])
        npt.assert_almost_equal(outputs["transformer_I"], (1.0 / 6.0) * 420 * s ** 2)

    def testYaw(self):
        inputs = {}
        outputs = {}
        myobj = dc.YawSystem()

        inputs["rotor_diameter"] = 200.0
        inputs["machine_rating"] = 10e3
        inputs["D_top"] = 5.0
        inputs["rho"] = 5e3
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["yaw_mass"], 5e3 * np.pi * 0.1 * 5 ** 2 * 0.2 + 190 * 12)
        npt.assert_equal(outputs["yaw_cm"], 0.0)
        npt.assert_equal(outputs["yaw_I"], 0.0)

    def testMiscDirect(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.MiscNacelleComponents(direct_drive=True)

        discrete_inputs["upwind"] = False
        inputs["machine_rating"] = 10e3
        inputs["L_bedplate"] = 5.0
        inputs["H_bedplate"] = 4.0
        inputs["D_top"] = 6.0
        inputs["R_generator"] = 2.0
        inputs["overhang"] = 10.0
        inputs["generator_cm"] = 6.0
        inputs["rho_fiberglass"] = 2e3
        inputs["rho_castiron"] = 3e3
        inputs["hvac_mass_coeff"] = 0.1
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)

        L = 1.1 * 5  # (10 + 6.0)
        W = 1.1 * 2 * 2
        H = 1.1 * (2 + 4)
        self.assertEqual(outputs["cover_mass"], 0.02 * 2e3 * 2 * (L * W + L * H + W * H))
        npt.assert_equal(outputs["cover_cm"], np.array([0.5 * (L - 5), 0.0, 0.5 * H]))

        self.assertEqual(outputs["hvac_mass"], 0.1 * 10e3 * 2 * np.pi * 0.75 * 2)
        self.assertEqual(outputs["hvac_cm"], 6.0)
        npt.assert_equal(outputs["hvac_I"], outputs["hvac_mass"] * 1.5 ** 2 * np.r_[1.0, 0.5, 0.5])

        t = 0.04
        self.assertEqual(outputs["platform_mass"], t * 3e3 * 12 ** 2)
        npt.assert_equal(outputs["platform_cm"], 0.0)
        npt.assert_equal(
            outputs["platform_I"],
            outputs["platform_mass"] * np.array([t ** 2 + 12 ** 2, t ** 2 + 12 ** 2, 2 * 12 ** 2]) / 12.0,
        )

        discrete_inputs["upwind"] = True
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["cover_mass"], 0.02 * 2e3 * 2 * (L * W + L * H + W * H))
        npt.assert_equal(outputs["cover_cm"], np.array([-0.5 * (L - 5), 0.0, 0.5 * H]))

        self.assertEqual(outputs["hvac_mass"], 0.1 * 10e3 * 2 * np.pi * 0.75 * 2)
        self.assertEqual(outputs["hvac_cm"], 6.0)
        npt.assert_equal(outputs["hvac_I"], outputs["hvac_mass"] * 1.5 ** 2 * np.r_[1.0, 0.5, 0.5])

        self.assertEqual(outputs["platform_mass"], t * 3e3 * 12 ** 2)
        npt.assert_equal(outputs["platform_cm"], 0.0)
        npt.assert_equal(
            outputs["platform_I"],
            outputs["platform_mass"] * np.array([t ** 2 + 12 ** 2, t ** 2 + 12 ** 2, 2 * 12 ** 2]) / 12.0,
        )

    def testMiscGeared(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.MiscNacelleComponents(direct_drive=False)

        discrete_inputs["upwind"] = False
        inputs["machine_rating"] = 10e3
        inputs["L_bedplate"] = 5.0
        inputs["H_bedplate"] = 4.0
        inputs["D_top"] = 6.0
        inputs["R_generator"] = 2.0
        inputs["overhang"] = 10.0
        inputs["generator_cm"] = 6.0
        inputs["rho_fiberglass"] = 2e3
        inputs["rho_castiron"] = 3e3
        inputs["hvac_mass_coeff"] = 0.1
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)

        L = 1.1 * (10 + 6.0)
        W = 1.1 * 2 * 2
        H = 1.1 * (2 + 4)
        self.assertEqual(outputs["cover_mass"], 0.02 * 2e3 * 2 * (L * W + L * H + W * H))
        npt.assert_equal(outputs["cover_cm"], np.array([0.5 * (L - 5), 0.0, 0.5 * H]))

        self.assertEqual(outputs["hvac_mass"], 0.1 * 10e3 * 2 * np.pi * 0.75 * 2)
        self.assertEqual(outputs["hvac_cm"], 6.0)
        npt.assert_equal(outputs["hvac_I"], outputs["hvac_mass"] * 1.5 ** 2 * np.r_[1.0, 0.5, 0.5])

        t = 0.04
        self.assertEqual(outputs["platform_mass"], t * 3e3 * L * W)
        npt.assert_equal(outputs["platform_cm"], 0.0)
        npt.assert_equal(
            outputs["platform_I"],
            outputs["platform_mass"] * np.array([t ** 2 + W ** 2, t ** 2 + L ** 2, L ** 2 + W ** 2]) / 12.0,
        )

        discrete_inputs["upwind"] = True
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["cover_mass"], 0.02 * 2e3 * 2 * (L * W + L * H + W * H))
        npt.assert_equal(outputs["cover_cm"], np.array([-0.5 * (L - 5), 0.0, 0.5 * H]))

        self.assertEqual(outputs["hvac_mass"], 0.1 * 10e3 * 2 * np.pi * 0.75 * 2)
        self.assertEqual(outputs["hvac_cm"], 6.0)
        npt.assert_equal(outputs["hvac_I"], outputs["hvac_mass"] * 1.5 ** 2 * np.r_[1.0, 0.5, 0.5])

        self.assertEqual(outputs["platform_mass"], t * 3e3 * L * W)
        npt.assert_equal(outputs["platform_cm"], 0.0)
        npt.assert_equal(
            outputs["platform_I"],
            outputs["platform_mass"] * np.array([t ** 2 + W ** 2, t ** 2 + L ** 2, L ** 2 + W ** 2]) / 12.0,
        )

    def testNacelle_noTilt(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.NacelleSystemAdder()

        discrete_inputs["upwind"] = True
        discrete_inputs["uptower"] = True
        inputs["tilt"] = 0.0
        inputs["constr_height"] = 2.0
        inputs["x_bedplate"] = -2 * np.ones(5)
        components = [
            "mb1",
            "mb2",
            "lss",
            "hss",
            "gearbox",
            "generator",
            "generator_stator",
            "generator_rotor",
            "hvac",
            "brake",
            "nose",
            "bedplate",
            "platform",
            "yaw",
            "cover",
            "transformer",
            "converter",
        ]
        cm3 = ["gearbox", "transformer", "converter", "yaw", "bedplate", "platform", "cover"]
        for k in components:
            inputs[k + "_mass"] = 1e3
            inputs[k + "_I"] = 1e3 * np.array([1, 2, 3])
            if k in cm3:
                inputs[k + "_cm"] = np.array([-5.0, 0.0, 2.0])
            else:
                inputs[k + "_cm"] = [3.0]

        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["other_mass"], 1e3 * 6)
        self.assertEqual(outputs["nacelle_mass"], 1e3 * (len(components) - 2))  # gen stator / rotor duplication
        npt.assert_equal(outputs["nacelle_cm"], np.r_[-5.0, 0.0, 2.0])
        npt.assert_equal(outputs["nacelle_I"], 1e3 * (len(components) - 2) * np.r_[1.0, 2.0, 3.0, np.zeros(3)])

        discrete_inputs["upwind"] = False
        for k in cm3:
            inputs[k + "_cm"][0] *= -1.0
        inputs["x_bedplate"] *= -1.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["other_mass"], 1e3 * 6)
        self.assertEqual(outputs["nacelle_mass"], 1e3 * (len(components) - 2))
        npt.assert_equal(outputs["nacelle_cm"], np.r_[5.0, 0.0, 2.0])
        npt.assert_equal(outputs["nacelle_I"], 1e3 * (len(components) - 2) * np.r_[1.0, 2.0, 3.0, np.zeros(3)])

        discrete_inputs["uptower"] = False
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["other_mass"], 1e3 * 6)
        self.assertEqual(outputs["nacelle_mass"], 1e3 * (len(components) - 4))
        npt.assert_equal(outputs["nacelle_cm"], np.r_[5.0, 0.0, 2.0])
        npt.assert_equal(outputs["nacelle_I"], 1e3 * (len(components) - 4) * np.r_[1.0, 2.0, 3.0, np.zeros(3)])

    def testNacelle_withTilt(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.NacelleSystemAdder()

        discrete_inputs["upwind"] = True
        discrete_inputs["uptower"] = True
        inputs["tilt"] = 5.0
        tr = 5 * np.pi / 180.0
        inputs["constr_height"] = 2.0
        inputs["x_bedplate"] = -2 * np.ones(5)
        components = [
            "mb1",
            "mb2",
            "lss",
            "hss",
            "gearbox",
            "generator",
            "generator_stator",
            "generator_rotor",
            "hvac",
            "brake",
            "nose",
            "bedplate",
            "platform",
            "yaw",
            "cover",
            "transformer",
            "converter",
        ]
        cm3 = ["gearbox", "transformer", "converter", "yaw", "bedplate", "platform", "cover"]
        for k in components:
            inputs[k + "_mass"] = 1e3
            inputs[k + "_I"] = 1e3 * np.array([1, 2, 3])
            if k in cm3:
                inputs[k + "_cm"] = np.array([-3.0 * np.cos(tr) - 2, 0.0, 2 + 3.0 * np.sin(tr)])
            else:
                inputs[k + "_cm"] = [3.0]

        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["other_mass"], 1e3 * 6)
        self.assertEqual(outputs["nacelle_mass"], 1e3 * (len(components) - 2))  # gen stator / rotor duplication
        npt.assert_almost_equal(outputs["nacelle_cm"], np.r_[-3.0 * np.cos(tr) - 2, 0.0, 2 + 3.0 * np.sin(tr)])
        # npt.assert_equal(outputs['nacelle_I'], 1e3*len(components)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])

        discrete_inputs["upwind"] = False
        for k in cm3:
            inputs[k + "_cm"][0] *= -1.0
        inputs["x_bedplate"] *= -1.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["other_mass"], 1e3 * 6)
        self.assertEqual(outputs["nacelle_mass"], 1e3 * (len(components) - 2))
        npt.assert_almost_equal(outputs["nacelle_cm"], np.r_[3.0 * np.cos(tr) + 2, 0.0, 2 + 3.0 * np.sin(tr)])
        # npt.assert_equal(outputs['nacelle_I'], 1e3*len(components)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])

        discrete_inputs["uptower"] = False
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["other_mass"], 1e3 * 6)
        self.assertEqual(outputs["nacelle_mass"], 1e3 * (len(components) - 4))
        npt.assert_almost_equal(outputs["nacelle_cm"], np.r_[3.0 * np.cos(tr) + 2, 0.0, 2 + 3.0 * np.sin(tr)])

    def testRNA(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.RNA_Adder()

        discrete_inputs["upwind"] = True
        inputs["tilt"] = 0.0
        inputs["L_drive"] = 10.0
        inputs["blades_mass"] = 100e3
        inputs["blades_I"] = 100e3 * np.arange(1, 7)
        inputs["nacelle_mass"] = 200e3
        inputs["nacelle_I_TT"] = 200e3 * np.arange(1, 7)
        inputs["nacelle_cm"] = np.array([-5.0, 0.0, 0.0])
        inputs["hub_system_mass"] = 25e3
        inputs["hub_system_I"] = 25e3 * np.arange(1, 7)
        inputs["hub_system_cm"] = 2.0
        inputs["shaft_start"] = np.zeros(3)

        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["rotor_mass"], 125e3)
        self.assertEqual(outputs["rna_mass"], 325e3)
        npt.assert_equal(outputs["rna_cm"], np.r_[(-125 * 12 - 200 * 5) / 325, 0.0, 0.0])
        I0 = 325e3 * np.arange(1, 7)
        I0[1:3] += 125e3 * 12 ** 2
        npt.assert_equal(outputs["rna_I_TT"], I0)

        discrete_inputs["upwind"] = False
        inputs["nacelle_cm"] = np.array([5.0, 0.0, 0.0])
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["rotor_mass"], 125e3)
        self.assertEqual(outputs["rna_mass"], 325e3)
        npt.assert_equal(outputs["rna_cm"], np.r_[(125 * 12 + 200 * 5) / 325, 0.0, 0.0])
        npt.assert_equal(outputs["rna_I_TT"], I0)

        inputs["tilt"] = 5.0
        tr = 5 * np.pi / 180.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["rotor_mass"], 125e3)
        self.assertEqual(outputs["rna_mass"], 325e3)
        npt.assert_almost_equal(
            outputs["rna_cm"], np.r_[(125 * 12 * np.cos(tr) + 200 * 5) / 325, 0.0, 125 * 12 * np.sin(tr) / 325]
        )

    def testDynamics(self):
        inputs = {}
        outputs = {}
        myobj = dc.DriveDynamics()

        inputs["lss_spring_constant"] = 2.0
        inputs["hss_spring_constant"] = 3.0
        inputs["gear_ratio"] = 1.0
        inputs["damping_ratio"] = 0.5
        inputs["blades_I"] = 30 * np.ones(6)
        inputs["hub_system_I"] = 20 * np.ones(6)

        myobj.compute(inputs, outputs)
        self.assertEqual(outputs["drivetrain_spring_constant"], 2.0)
        self.assertEqual(outputs["drivetrain_damping_coefficient"], 10)

        inputs["gear_ratio"] = 5.0
        myobj.compute(inputs, outputs)
        self.assertAlmostEqual(outputs["drivetrain_spring_constant"], 150.0 / (2 + 75))
        self.assertAlmostEqual(outputs["drivetrain_damping_coefficient"], np.sqrt(50 * 150.0 / (2 + 75)))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestComponents))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
