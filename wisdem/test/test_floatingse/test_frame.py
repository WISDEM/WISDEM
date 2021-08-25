import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.floatingse.floating_frame as frame
from wisdem.commonse import gravity as g
from wisdem.floatingse.member import NULL, MEMMAX


class TestPlatform(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.opt = {}
        self.opt["floating"] = {}
        self.opt["WISDEM"] = {}
        self.opt["WISDEM"]["FloatingSE"] = {}
        self.opt["floating"]["members"] = {}
        self.opt["floating"]["members"]["n_members"] = n_member = 6
        self.opt["WISDEM"]["FloatingSE"]["frame3dd"] = {}
        self.opt["WISDEM"]["FloatingSE"]["frame3dd"]["shear"] = True
        self.opt["WISDEM"]["FloatingSE"]["frame3dd"]["geom"] = True
        self.opt["WISDEM"]["FloatingSE"]["frame3dd"]["tol"] = 1e-8
        self.opt["WISDEM"]["FloatingSE"]["frame3dd"]["modal"] = False
        self.opt["WISDEM"]["FloatingSE"]["gamma_f"] = 1.35  # Safety factor on loads
        self.opt["WISDEM"]["FloatingSE"]["gamma_m"] = 1.3  # Safety factor on materials
        self.opt["WISDEM"]["FloatingSE"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
        self.opt["WISDEM"]["FloatingSE"]["gamma_b"] = 1.1  # Safety factor on buckling
        self.opt["WISDEM"]["FloatingSE"]["gamma_fatigue"] = 1.755  # Not used
        self.opt["mooring"] = {}
        self.opt["mooring"]["n_attach"] = 3

        self.inputs["tower_nodes"] = NULL * np.ones((MEMMAX, 3))
        self.inputs["tower_Rnode"] = NULL * np.ones(MEMMAX)
        for k in range(n_member):
            self.inputs[f"member{k}:nodes_xyz"] = NULL * np.ones((MEMMAX, 3))
            self.inputs[f"member{k}:nodes_r"] = NULL * np.ones(MEMMAX)

        for var in [
            "D",
            "t",
            "A",
            "Asx",
            "Asy",
            "Ixx",
            "Iyy",
            "Izz",
            "rho",
            "E",
            "G",
            "sigma_y",
            "Px",
            "Py",
            "Pz",
            "qdyn",
        ]:
            self.inputs["tower_elem_" + var] = NULL * np.ones(MEMMAX)
            for k in range(n_member):
                if var in ["Px", "Py", "Pz", "qdyn"]:
                    self.inputs[f"member{k}:{var}"] = NULL * np.ones(MEMMAX)
                else:
                    self.inputs[f"member{k}:section_{var}"] = NULL * np.ones(MEMMAX)

        self.inputs["member0:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [1, 0, 0]])
        self.inputs["member1:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0.5, 1, 0]])
        self.inputs["member2:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 0]])
        self.inputs["member3:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [0, 0, 1]])
        self.inputs["member4:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0, 0, 1]])
        self.inputs["member5:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 1]])
        for k in range(n_member):
            L = np.sqrt(np.sum(np.diff(self.inputs[f"member{k}:nodes_xyz"][:2, :], axis=0) ** 2))
            self.inputs[f"member{k}:nodes_r"][:2] = 0.1 * k * np.ones(2)
            self.inputs[f"member{k}:section_D"][:1] = 2.0
            self.inputs[f"member{k}:section_t"][:1] = 0.1
            self.inputs[f"member{k}:section_A"][:1] = 0.5 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_Asx"][:1] = 0.5 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_Asy"][:1] = 0.5 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_Ixx"][:1] = 2 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_Iyy"][:1] = 2 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_Izz"][:1] = 2 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_rho"][:1] = 1e3 / (0.5 * k * np.ones(1) + 1) / L
            self.inputs[f"member{k}:section_E"][:1] = 3 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_G"][:1] = 4 * k * np.ones(1) + 1
            self.inputs[f"member{k}:section_sigma_y"][:1] = 5 * k * np.ones(1) + 1
            self.inputs[f"member{k}:idx_cb"] = 0
            self.inputs[f"member{k}:buoyancy_force"] = 1e2
            self.inputs[f"member{k}:displacement"] = 1e1
            self.inputs[f"member{k}:center_of_buoyancy"] = self.inputs[f"member{k}:nodes_xyz"][:2, :].mean(axis=0)
            self.inputs[f"member{k}:center_of_mass"] = self.inputs[f"member{k}:nodes_xyz"][:2, :].mean(axis=0)
            self.inputs[f"member{k}:total_mass"] = 1e3
            self.inputs[f"member{k}:total_cost"] = 2e3
            self.inputs[f"member{k}:I_total"] = 1e2 + np.arange(6)
            self.inputs[f"member{k}:Awater"] = 5.0
            self.inputs[f"member{k}:Iwater"] = 15.0
            self.inputs[f"member{k}:added_mass"] = np.arange(6)
            self.inputs[f"member{k}:ballast_mass"] = 1e2
            self.inputs[f"member{k}:variable_ballast_capacity"] = 10 + k
            self.inputs[f"member{k}:variable_ballast_spts"] = np.linspace(0, 0.5, 10)
            self.inputs[f"member{k}:variable_ballast_Vpts"] = np.arange(10)
            self.inputs[f"member{k}:waterline_centroid"] = self.inputs[f"member{k}:nodes_xyz"][:2, :2].mean(axis=0)
            self.inputs[f"member{k}:Px"][:2] = 1.0
            self.inputs[f"member{k}:Py"][:2] = 2.0
            self.inputs[f"member{k}:Pz"][:2] = 3.0
            self.inputs[f"member{k}:qdyn"][:2] = 4.0

        myones = np.ones(2)
        self.inputs["tower_nodes"][:3, :] = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 51.0], [0.0, 0.0, 101.0]])
        # self.inputs["tower_Fnode"] = np.zeros
        self.inputs["tower_Rnode"][:3] = np.zeros(3)
        # self.inputs["tower_elem_n1"] = [0, 1]
        # self.inputs["tower_elem_n2"] = [1, 2]
        self.inputs["tower_elem_D"][:2] = 2.0 * myones
        self.inputs["tower_elem_t"][:2] = 0.1 * myones
        self.inputs["tower_elem_A"][:2] = 4.0 * myones
        self.inputs["tower_elem_Asx"][:2] = 4.1 * myones
        self.inputs["tower_elem_Asy"][:2] = 4.1 * myones
        self.inputs["tower_elem_Ixx"][:2] = 5.0 * myones
        self.inputs["tower_elem_Iyy"][:2] = 5.0 * myones
        self.inputs["tower_elem_Izz"][:2] = 10.0 * myones
        self.inputs["tower_elem_rho"][:2] = 5e3 / 4 / 100 * myones
        self.inputs["tower_elem_E"][:2] = 30.0 * myones
        self.inputs["tower_elem_G"][:2] = 40.0 * myones
        self.inputs["tower_elem_sigma_y"][:2] = 50.0 * myones
        self.inputs["tower_elem_Px"][:3] = 1.0
        self.inputs["tower_elem_Py"][:3] = 2.0
        self.inputs["tower_elem_Pz"][:3] = 3.0
        self.inputs["tower_elem_qdyn"][:3] = 4.0
        self.inputs["tower_center_of_mass"] = self.inputs["tower_nodes"][:3, :].mean(axis=0)
        self.inputs["tower_mass"] = 5e3

        self.inputs["mooring_neutral_load"] = np.zeros((3, 3))
        self.inputs["mooring_neutral_load"][:, 0] = [200, -100.0, -100]
        self.inputs["mooring_neutral_load"][:, 1] = [0.0, 50, -50]
        self.inputs["mooring_neutral_load"][:, 2] = -1e3
        self.inputs["mooring_fairlead_joints"] = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
        self.inputs["mooring_stiffness"] = 5 * np.eye(6)
        self.inputs["transition_node"] = self.inputs["tower_nodes"][0, :]
        self.inputs["tower_top_node"] = self.inputs["tower_nodes"][2, :]
        self.inputs["rna_mass"] = 1e4
        self.inputs["rna_cg"] = np.ones(3)
        self.inputs["rna_I"] = 1e4 * np.arange(6)
        self.inputs["rna_F"] = np.array([1e2, 1e1, 0.0])
        self.inputs["rna_M"] = np.array([2e1, 2e2, 0.0])
        self.inputs["transition_piece_mass"] = 1e3
        self.inputs["transition_piece_cost"] = 3e3
        self.inputs["rho_water"] = 1e3

    def testRunFrame(self):
        myobj = frame.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs:
            self.inputs[k] = self.outputs[k]
        for k in self.discrete_outputs:
            self.discrete_inputs[k] = self.discrete_outputs[k]
        myobj = frame.PlatformTowerFrame()
        myobj.compute(self.inputs, self.outputs)
        for k in self.outputs:
            self.inputs[k] = self.outputs[k]
        for k in self.discrete_outputs:
            self.discrete_inputs[k] = self.discrete_outputs[k]
        myobj = frame.FrameAnalysis(options=self.opt)
        myobj.compute(self.inputs, self.outputs)
        self.assertTrue(True)


class TestGroup(unittest.TestCase):
    def testRunAll(self):

        opt = {}
        opt["floating"] = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["FloatingSE"] = {}
        opt["floating"]["members"] = {}
        opt["floating"]["members"]["n_members"] = n_member = 6
        opt["floating"]["members"]["n_height"] = [2]
        opt["floating"]["members"]["n_bulkheads"] = [4]
        opt["floating"]["members"]["n_layers"] = [1]
        opt["floating"]["members"]["n_ballasts"] = [0]
        opt["floating"]["members"]["n_axial_joints"] = [1]
        opt["WISDEM"]["FloatingSE"]["frame3dd"] = {}
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["shear"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["geom"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["tol"] = 1e-7
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["modal"] = False  # True
        opt["WISDEM"]["FloatingSE"]["gamma_f"] = 1.35  # Safety factor on loads
        opt["WISDEM"]["FloatingSE"]["gamma_m"] = 1.3  # Safety factor on materials
        opt["WISDEM"]["FloatingSE"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
        opt["WISDEM"]["FloatingSE"]["gamma_b"] = 1.1  # Safety factor on buckling
        opt["WISDEM"]["FloatingSE"]["gamma_fatigue"] = 1.755  # Not used
        opt["mooring"] = {}
        opt["mooring"]["n_attach"] = 3
        opt["mooring"]["n_anchors"] = 3

        opt["materials"] = {}
        opt["materials"]["n_mat"] = 2

        prob = om.Problem()
        prob.model = frame.FloatingFrame(modeling_options=opt)
        prob.setup()

        # Material properties
        prob["rho_mat"] = np.array([7850.0, 5000.0])  # Steel, ballast slurry [kg/m^3]
        prob["E_mat"] = 200e9 * np.ones((2, 3))  # Young's modulus [N/m^2]
        prob["G_mat"] = 79.3e9 * np.ones((2, 3))  # Shear modulus [N/m^2]
        prob["sigma_y_mat"] = 3.45e8 * np.ones(2)  # Elastic yield stress [N/m^2]
        prob["sigma_ult_mat"] = 5e8 * np.ones((2, 3))
        prob["wohler_exp_mat"] = 4.0 * np.ones(2)
        prob["wohler_A_mat"] = 7.5e8 * np.ones(2)
        prob["unit_cost_mat"] = np.array([2.0, 1.0])
        prob["material_names"] = ["steel", "slurry"]

        # Mass and cost scaling factors
        prob["labor_cost_rate"] = 1.0  # Cost factor for labor time [$/min]
        prob["painting_cost_rate"] = 14.4  # Cost factor for column surface finishing [$/m^2]

        prob["member0:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [1, 0, 0]])
        prob["member1:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0.5, 1, 0]])
        prob["member2:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 0]])
        prob["member3:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [0, 0, 1]])
        prob["member4:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0, 0, 1]])
        prob["member5:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 1]])
        for k in range(n_member):
            L = np.sqrt(np.sum(np.diff(prob[f"member{k}:nodes_xyz"][:, 2], axis=0) ** 2))
            prob[f"member{k}:nodes_r"][:2] = 0.1 * k * np.ones(2)
            prob[f"member{k}:section_D"][:1] = 2.0
            prob[f"member{k}:section_t"][:1] = 0.1
            prob[f"member{k}:section_A"][:1] = 0.5 * k * np.ones(1) + 1
            prob[f"member{k}:section_Asx"][:1] = 0.5 * k * np.ones(1) + 1
            prob[f"member{k}:section_Asy"][:1] = 0.5 * k * np.ones(1) + 1
            prob[f"member{k}:section_Ixx"][:1] = 2 * k * np.ones(1) + 1
            prob[f"member{k}:section_Iyy"][:1] = 2 * k * np.ones(1) + 1
            prob[f"member{k}:section_Izz"][:1] = 2 * k * np.ones(1) + 1
            prob[f"member{k}:section_rho"][:1] = 1e3 / (0.5 * k * np.ones(1) + 1) / L
            prob[f"member{k}:section_E"][:1] = 3 * k * np.ones(1) + 1
            prob[f"member{k}:section_G"][:1] = 4 * k * np.ones(1) + 1
            prob[f"member{k}:section_sigma_y"][:1] = 5 * k * np.ones(1) + 1
            prob[f"member{k}:idx_cb"] = 0
            prob[f"member{k}:buoyancy_force"] = 1e2
            prob[f"member{k}:displacement"] = 1e1
            prob[f"member{k}:center_of_buoyancy"] = prob[f"member{k}:nodes_xyz"][:2, :].mean(axis=0)
            prob[f"member{k}:center_of_mass"] = prob[f"member{k}:nodes_xyz"][:2, :].mean(axis=0)
            prob[f"member{k}:total_mass"] = 1e3
            prob[f"member{k}:total_cost"] = 2e3
            prob[f"member{k}:Awater"] = 5.0
            prob[f"member{k}:Iwater"] = 15.0
            prob[f"member{k}:added_mass"] = np.arange(6)
            prob[f"member{k}:Px"][:2] = 1.0
            prob[f"member{k}:Py"][:2] = 2.0
            prob[f"member{k}:Pz"][:2] = 3.0

        # Set environment to that used in OC3 testing campaign
        prob["rho_air"] = 1.226  # Density of air [kg/m^3]
        prob["mu_air"] = 1.78e-5  # Viscosity of air [kg/m/s]
        prob["rho_water"] = 1025.0  # Density of water [kg/m^3]
        prob["mu_water"] = 1.08e-3  # Viscosity of water [kg/m/s]
        prob["water_depth"] = 320.0  # Distance to sea floor [m]
        prob["Hsig_wave"] = 0.0  # Significant wave height [m]
        prob["Tsig_wave"] = 1e3  # Wave period [s]
        prob["shearExp"] = 0.11  # Shear exponent in wind power law
        prob["cm"] = 2.0  # Added mass coefficient
        prob["Uc"] = 0.0  # Mean current speed
        prob["beta_wind"] = prob["beta_wave"] = 0.0
        prob["cd_usr"] = -1.0  # Compute drag coefficient
        prob["Uref"] = 10.0
        prob["zref"] = 100.0

        prob["mooring_neutral_load"] = np.zeros((3, 3))
        prob["mooring_neutral_load"][:, 0] = [200, -100.0, -100]
        prob["mooring_neutral_load"][:, 1] = [0.0, 50, -50]
        prob["mooring_neutral_load"][:, 2] = -1e3
        prob["mooring_fairlead_joints"] = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
        prob["transition_piece_mass"] = 1e3
        prob["transition_node"] = prob["member4:nodes_xyz"][0, :]

        prob.run_model()
        self.assertTrue(True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPlatform))
    suite.addTest(unittest.makeSuite(TestGroup))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
