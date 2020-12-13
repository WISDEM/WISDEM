import unittest

import numpy as np
import openmdao.api as om
import wisdem.drivetrainse.hub as hub


class TestHub(unittest.TestCase):
    def testMaxTorque(self):
        torq = hub.FindMaxTorque()
        inputs = {}
        outputs = {}

        inputs["blades_I"] = np.ones(6)
        inputs["blades_I"][0] = 1e3
        inputs["rated_rpm"] = 10.0
        inputs["stop_time"] = 5.0

        torq.compute(inputs, outputs)
        self.assertAlmostEqual(outputs["max_torque"], 1e4 * np.pi / 30.0 / 5.0)

    def testPitch(self):

        pitch = hub.PitchSystem()
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        inputs["blade_mass"] = 17740.0  # kg
        discrete_inputs["n_blades"] = 3
        inputs["hub_diameter"] = 4.0
        inputs["rho"] = 7850.0
        inputs["Xy"] = 250e6

        AirDensity = 1.225  # kg/(m^3)
        Solidity = 0.0517
        RatedWindSpeed = 11.05  # m/s
        rotor_diameter = 126.0
        inputs["BRFM"] = (
            (3.06 * np.pi / 8)
            * AirDensity
            * (RatedWindSpeed ** 2)
            * (Solidity * (rotor_diameter ** 3))
            / discrete_inputs["n_blades"]
        )

        inputs["pitch_system_scaling_factor"] = 1.0

        pitch.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        m = outputs["pitch_mass"]
        self.assertAlmostEqual(outputs["pitch_I"][0], m * 4)
        self.assertAlmostEqual(outputs["pitch_I"][1], m * 2)
        self.assertAlmostEqual(outputs["pitch_I"][2], m * 2)

    def testRegression(self):
        opt = {}
        opt["hub_gamma"] = 2.0
        opt["spinner_gamma"] = 1.5
        hub_prob = om.Problem(model=hub.Hub_System(modeling_options=opt))
        hub_prob.setup()

        hub_prob["blades_I"] = 1e3 * np.ones(6)
        hub_prob["blades_I"][0] = 199200777.51 * 30.0 * 5 / 10.0 / np.pi
        hub_prob["rated_rpm"] = 10.0
        hub_prob["stop_time"] = 5.0
        hub_prob["n_blades"] = 3
        hub_prob["flange_t2shell_t"] = 4.0
        hub_prob["flange_OD2hub_D"] = 0.5
        hub_prob["flange_ID2flange_OD"] = 0.8
        hub_prob["hub_in2out_circ"] = 1.2
        hub_prob["hub_stress_concentration"] = 2.5
        hub_prob["n_front_brackets"] = 3
        hub_prob["n_rear_brackets"] = 3
        hub_prob["clearance_hub_spinner"] = 0.5
        hub_prob["spin_hole_incr"] = 1.2
        hub_prob["blade_root_diameter"] = 4.5
        hub_prob["hub_diameter"] = 6.235382907247958

        hub_prob["pitch_system.blade_mass"] = 17000.0
        hub_prob["pitch_system.BRFM"] = 1.0e6
        hub_prob["pitch_system_scaling_factor"] = 0.54
        hub_prob["pitch_system.rho"] = 7850.0
        hub_prob["pitch_system.Xy"] = 371.0e6

        hub_prob["hub_shell.rho"] = 7200.0
        # hub_prob['hub_shell.max_torque']           = 199200777.51
        hub_prob["hub_shell.Xy"] = 200.0e6
        hub_prob["hub_shell.metal_cost"] = 3.00

        hub_prob["spinner_gust_ws"] = 70
        hub_prob["spinner.composite_Xt"] = 60.0e6
        # hub_prob['spinner.composite_SF']           = 1.5
        hub_prob["spinner.composite_rho"] = 1600.0
        hub_prob["spinner.Xy"] = 225.0e6
        # hub_prob['spinner.metal_SF']               = 1.5
        hub_prob["spinner.metal_rho"] = 7850.0
        hub_prob["spinner.composite_cost"] = 7.00
        hub_prob["spinner.metal_cost"] = 3.00

        hub_prob.run_model()

        self.assertAlmostEqual(hub_prob["pitch_system.pitch_mass"][0], 6202.76603773585)
        self.assertAlmostEqual(hub_prob["hub_shell.hub_mass"][0], 79417.52737564275)
        self.assertAlmostEqual(hub_prob["hub_shell.hub_cost"][0], 238252.58212692826)
        self.assertAlmostEqual(hub_prob["hub_shell.hub_cm"][0], 2.8681137824204033)
        self.assertAlmostEqual(hub_prob["hub_shell.hub_I"][0], 514625.57739416)
        self.assertAlmostEqual(hub_prob["spinner.spinner_mass"][0], 1393.1009978176617)  # 1704.965737284796)
        self.assertAlmostEqual(hub_prob["spinner.spinner_diameter"][0], 7.235382907247958)
        self.assertAlmostEqual(hub_prob["spinner.spinner_cost"][0], 7678.042964192897)  # 9395.05078685929)
        self.assertAlmostEqual(hub_prob["spinner.spinner_cm"][0], 3.117691453623979)
        self.assertAlmostEqual(hub_prob["spinner.spinner_I"][0], 12154.984015448832)  # 14876.04367239)
        self.assertAlmostEqual(hub_prob["adder.hub_system_mass"][0], 87013.39441119626)  # 87325.2591506634)
        self.assertAlmostEqual(hub_prob["adder.hub_system_cost"][0], 245930.62509112115)  # 247647.63291378756)
        self.assertAlmostEqual(hub_prob["adder.hub_system_cm"][0], 2.872109568415645)  # 2.8729866151188928)
        self.assertAlmostEqual(hub_prob["adder.hub_system_I"][0], 587071.4472964063)  # 589792.50695335)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHub))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
