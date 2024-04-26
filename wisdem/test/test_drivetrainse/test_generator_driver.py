import unittest

import numpy as np
import numpy.testing as npt

import wisdem.drivetrainse.generator as gen


class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.inputs["machine_rating"] = 5e6
        self.inputs["rated_torque"] = 4.143289e6

        self.inputs["rho_Fe"] = 7700.0
        self.inputs["rho_Fes"] = 7850.0
        self.inputs["rho_Copper"] = 8900.0
        self.inputs["rho_PM"] = 7450.0

        self.inputs["B_r"] = 1.2
        self.inputs["E"] = 2e11
        self.inputs["G"] = 79.3e9
        self.inputs["P_Fe0e"] = 1.0
        self.inputs["P_Fe0h"] = 4.0
        self.inputs["S_N"] = -0.002
        self.inputs["alpha_p"] = 0.5 * np.pi * 0.7
        self.inputs["b_r_tau_r"] = 0.45
        self.inputs["b_ro"] = 0.004
        self.inputs["b_s_tau_s"] = 0.45
        self.inputs["b_so"] = 0.004
        self.inputs["cofi"] = 0.85
        self.inputs["freq"] = 60
        self.inputs["h_i"] = 0.001
        self.inputs["h_sy0"] = 0.0
        self.inputs["h_w"] = 0.005
        self.inputs["k_fes"] = 0.9
        self.inputs["k_fillr"] = 0.7
        self.inputs["k_fills"] = 0.65
        self.inputs["k_s"] = 0.2
        self.discrete_inputs["m"] = 3
        self.inputs["mu_0"] = np.pi * 4e-7
        self.inputs["mu_r"] = 1.06
        self.inputs["p"] = 3.0
        self.inputs["phi"] = np.deg2rad(90)
        self.discrete_inputs["q1"] = 6
        self.discrete_inputs["q2"] = 4
        self.inputs["ratio_mw2pp"] = 0.7
        self.inputs["resist_Cu"] = 1.8e-8 * 1.4
        self.inputs["sigma"] = 40e3
        self.inputs["v"] = 0.3
        self.inputs["y_tau_p"] = 1.0
        self.inputs["y_tau_pr"] = 10.0 / 12
        for k in self.inputs:
            self.inputs[k] = np.array( [self.inputs[k]] )

    def testConstraints(self):
        pass

    def testMofI(self):
        inputs = {}
        outputs = {}
        myobj = gen.MofI()

        inputs["R_out"] = np.array([ 3.0 ])
        inputs["stator_mass"] = np.array([ 60.0 ])
        inputs["rotor_mass"] = np.array([ 40.0 ])
        inputs["generator_mass"] = np.array([ 100.0 ])
        inputs["len_s"] = np.array([ 2.0 ])
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["generator_I"], np.r_[50.0 * 9, 25.0 * 9 + 100.0 / 3.0, 25.0 * 9 + 100.0 / 3.0])
        npt.assert_almost_equal(outputs["rotor_I"], 0.4 * outputs["generator_I"])
        npt.assert_almost_equal(outputs["stator_I"], 0.6 * outputs["generator_I"])

    def testCost(self):
        inputs = {}
        outputs = {}
        myobj = gen.Cost()

        inputs["C_Cu"] = np.array([ 2.0 ])
        inputs["C_Fe"] = np.array([ 0.5 ])
        inputs["C_Fes"] = np.array([ 4.0 ])
        inputs["C_PM"] = np.array([ 3.0 ])

        inputs["Copper"] = np.array([ 10.0 ])
        inputs["Iron"] = np.array([ 0.0 ])
        inputs["mass_PM"] = np.array([ 0.0 ])
        inputs["Structural_mass"] = np.array([ 0.0 ])
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["generator_cost"], (1.26 * 2.0 * 10 + 96.2 * 0.064 * 10.0) / 0.619)
 
        inputs["Copper"] = np.array([ 0.0 ])
        inputs["Iron"] = np.array([ 10.0 ])
        inputs["mass_PM"] = np.array([ 0.0 ])
        inputs["Structural_mass"] = np.array([ 0.0 ])
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["generator_cost"], (1.21 * 0.5 * 10 + 26.9 * 0.064 * 10.0) / 0.684)

        inputs["Copper"] = np.array([ 0.0 ])
        inputs["Iron"] = np.array([ 0.0 ])
        inputs["mass_PM"] = np.array([ 10.0 ])
        inputs["Structural_mass"] = np.array([ 0.0 ])
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["generator_cost"], (1.0 * 3.0 * 10 + 79.0 * 0.064 * 10.0) / 0.619)

        inputs["Copper"] = np.array([ 0.0 ])
        inputs["Iron"] = np.array([ 0.0 ])
        inputs["mass_PM"] = np.array([ 0.0 ])
        inputs["Structural_mass"] = np.array([ 10.0 ])
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs["generator_cost"], (1.21 * 4.0 * 10 + 15.9 * 0.064 * 10.0) / 0.684)

    def testEff(self):
        inputs = {}
        outputs = {}
        myobj = gen.PowerElectronicsEff()

        inputs["machine_rating"] = np.array([ 2e6 ])
        inputs["shaft_rpm"] = np.arange(10.1)
        inputs["shaft_rpm"][0] = 0.1
        inputs["eandm_efficiency"] = np.ones(inputs["shaft_rpm"].shape)
        myobj.compute(inputs, outputs)
        npt.assert_array_less(outputs["converter_efficiency"], 1.0)
        npt.assert_array_less(0.97, outputs["converter_efficiency"][1:])
        npt.assert_array_less(outputs["transformer_efficiency"], 1.0)
        npt.assert_array_less(0.97, outputs["transformer_efficiency"][1:])
        npt.assert_almost_equal(
            outputs["generator_efficiency"], outputs["transformer_efficiency"] * outputs["converter_efficiency"]
        )

        inputs["machine_rating"] = 20e6
        myobj.compute(inputs, outputs)
        npt.assert_array_less(outputs["converter_efficiency"], 1.0)
        npt.assert_array_less(0.97, outputs["converter_efficiency"][1:])
        npt.assert_array_less(outputs["transformer_efficiency"], 1.0)
        npt.assert_array_less(0.97, outputs["transformer_efficiency"][1:])
        npt.assert_almost_equal(
            outputs["generator_efficiency"], outputs["transformer_efficiency"] * outputs["converter_efficiency"]
        )


if __name__ == "__main__":
    unittest.main()
