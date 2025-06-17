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
        inputs["rated_rpm"] = np.array([ 10.0 ])
        inputs["stop_time"] = np.array([ 5.0 ])

        torq.compute(inputs, outputs)
        self.assertAlmostEqual(outputs["max_torque"], 1e4 * np.pi / 30.0 / 5.0)

    def testPitch(self):
        pitch = hub.PitchSystem()
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        inputs["blade_mass"] = np.array([ 17740.0 ])  # kg
        discrete_inputs["n_blades"] = 3
        inputs["hub_diameter"] = np.array([ 4.0 ])
        inputs["rho"] = np.array([ 7850.0 ])
        inputs["Xy"] = np.array([ 250e6 ])
        inputs["pitch_system_mass_user"] = np.array([ 0.0 ])

        AirDensity = 1.225  # kg/(m^3)
        Solidity = 0.0517
        RatedWindSpeed = 11.05  # m/s
        rotor_diameter = 126.0
        inputs["BRFM"] = (
            (3.06 * np.pi / 8)
            * AirDensity
            * (RatedWindSpeed**2)
            * (Solidity * (rotor_diameter**3))
            / discrete_inputs["n_blades"]
        )

        inputs["pitch_system_scaling_factor"] = 1.0

        pitch.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        m = outputs["pitch_mass"]
        self.assertAlmostEqual(outputs["pitch_I"][0], m * 4)
        self.assertAlmostEqual(outputs["pitch_I"][1], m * 2)
        self.assertAlmostEqual(outputs["pitch_I"][2], m * 2)

        inputs["pitch_system_mass_user"] = np.array([ 100.0 ])
        pitch.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["pitch_mass"], 100.0)

    def testHubMass(self):
        myhub = hub.HubShell()
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        discrete_inputs["n_blades"] = 3
        inputs["flange_t2shell_t"] = 4.0
        inputs["flange_OD2hub_D"] = 0.5
        inputs["flange_ID2flange_OD"] = 0.8
        inputs["hub_in2out_circ"] = 1.2
        inputs["hub_stress_concentration"] = 2.5
        discrete_inputs["n_front_brackets"] = 3
        discrete_inputs["n_rear_brackets"] = 3
        inputs["clearance_hub_spinner"] = 0.5
        inputs["spin_hole_incr"] = 1.2
        inputs["blade_root_diameter"] = 4.5
        inputs["hub_diameter"] = 6.235382907247958
        inputs["rho"] = 7200.0
        inputs["Xy"] = 200.0e6
        inputs["metal_cost"] = 3.00
        inputs['max_torque'] = 199200777.51

        # Regression check
        inputs["hub_shell_mass_user"] = np.array([0.0])
        myhub.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertAlmostEqual(outputs["hub_mass"], 79417.52737564275)

        # Override check
        inputs["hub_shell_mass_user"] = np.array([100.0])
        myhub.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["hub_mass"], 100.0)

    def testSpinnerMass(self):
        spinner = hub.Spinner()
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        discrete_inputs["n_blades"] = 3
        inputs["flange_t2shell_t"] = 4.0
        inputs["flange_OD2hub_D"] = 0.5
        inputs["flange_ID2flange_OD"] = 0.8
        inputs["hub_in2out_circ"] = 1.2
        inputs["hub_stress_concentration"] = 2.5
        discrete_inputs["n_front_brackets"] = 3
        discrete_inputs["n_rear_brackets"] = 3
        inputs["clearance_hub_spinner"] = 0.5
        inputs["spin_hole_incr"] = 1.2
        inputs["blade_root_diameter"] = 4.5
        inputs["hub_diameter"] = 6.235382907247958
        inputs["rho"] = 7200.0
        inputs['max_torque'] = 199200777.51
        inputs["spinner_gust_ws"] = 70
        inputs["composite_Xt"] = 60.0e6
        # inputs['spinner.composite_SF']           = 1.5
        inputs["composite_rho"] = 1600.0
        inputs["Xy"] = 225.0e6
        # inputs['spinner.metal_SF']               = 1.5
        inputs["metal_rho"] = 7850.0
        inputs["composite_cost"] = 7.00
        inputs["metal_cost"] = 3.00

        # Regression check
        inputs["spinner_mass_user"] = np.array([0.0])
        spinner.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertAlmostEqual(outputs["spinner_mass"], 1393.1009978176617)

        # Override check
        inputs["spinner_mass_user"] = np.array([100.0])
        spinner.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs["spinner_mass"], 100.0)

        
    def testRegression(self):
        opt = {}
        opt["hub_gamma"] = 2.0
        opt["spinner_gamma"] = 1.5
        hub_prob = om.Problem(reports=False, model=hub.Hub_System(modeling_options=opt))
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


if __name__ == "__main__":
    unittest.main()
