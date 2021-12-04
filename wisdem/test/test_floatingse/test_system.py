import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.floatingse.floating_system as sys
from wisdem.commonse import gravity as g
from wisdem.commonse.cylinder_member import NULL, MEMMAX


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
            "J0",
            "rho",
            "E",
            "G",
            "sigma_y",
        ]:
            for k in range(n_member):
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
            self.inputs[f"member{k}:section_J0"][:1] = 2 * k * np.ones(1) + 1
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

        myones = np.ones(2)

        self.inputs["mooring_neutral_load"] = np.zeros((3, 3))
        self.inputs["mooring_neutral_load"][:, 0] = [200, -100.0, -100]
        self.inputs["mooring_neutral_load"][:, 1] = [0.0, 50, -50]
        self.inputs["mooring_neutral_load"][:, 2] = -1e3
        self.inputs["mooring_fairlead_joints"] = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
        self.inputs["mooring_stiffness"] = 5 * np.eye(6)
        self.inputs["transition_node"] = self.inputs["member5:nodes_xyz"][1, :]
        self.inputs["turbine_mass"] = 1e4
        self.inputs["turbine_cg"] = np.array([0, 0, 50])
        self.inputs["turbine_I"] = 1e6 * np.ones(6)
        self.inputs["transition_piece_mass"] = 1e3
        self.inputs["transition_piece_cost"] = 3e3
        self.inputs["rho_water"] = 1e3

    def testTetrahedron(self):
        myobj = sys.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # Check NULLs and implied number of nodes / elements
        npt.assert_equal(self.outputs["platform_nodes"][4:, :], NULL)
        npt.assert_equal(self.outputs["platform_Fnode"][4:, :], NULL)
        npt.assert_equal(self.outputs["platform_Rnode"][4:], NULL)
        npt.assert_equal(self.outputs["platform_elem_n1"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_n2"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_D"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_t"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_A"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_Asx"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_Asy"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_Ixx"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_Iyy"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_J0"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_rho"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_E"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_G"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_sigma_y"][6:], NULL)

        npt.assert_equal(
            self.outputs["platform_nodes"][:4, :],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        npt.assert_equal(
            self.outputs["platform_Fnode"][:4, :],
            1e2 * np.array([[0.0, 0.0, 2], [0.0, 0.0, 0.0], [0.0, 0.0, 2], [0.0, 0.0, 2.0]]),
        )
        npt.assert_equal(self.outputs["platform_Rnode"][:4], 0.1 * np.r_[3, 5, 5, 4])
        npt.assert_equal(self.outputs["platform_elem_n1"][:6], np.r_[0, 3, 2, 0, 3, 2])
        npt.assert_equal(self.outputs["platform_elem_n2"][:6], np.r_[3, 2, 0, 1, 1, 1])
        npt.assert_equal(self.outputs["platform_elem_D"][:6], 2.0)
        npt.assert_equal(self.outputs["platform_elem_t"][:6], 0.1)
        npt.assert_equal(self.outputs["platform_elem_A"][:6], 0.5 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_Asx"][:6], 0.5 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_Asy"][:6], 0.5 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_Ixx"][:6], 2 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_Iyy"][:6], 2 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_J0"][:6], 2 * np.arange(6) + 1)
        # npt.assert_equal(self.outputs["platform_elem_rho"][:6], 3 * np.arange(6)+1)
        npt.assert_equal(self.outputs["platform_elem_E"][:6], 3 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_G"][:6], 4 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_sigma_y"][:6], 5 * np.arange(6) + 1)
        self.assertEqual(self.outputs["platform_displacement"], 6e1)
        centroid = np.array([0.375, 0.25, 0.25])
        R = np.zeros(6)
        R[0] = np.sum((self.inputs["member0:nodes_xyz"][:2, :2].mean(axis=0) - centroid[:2]) ** 2)
        R[1] = np.sum((self.inputs["member1:nodes_xyz"][:2, :2].mean(axis=0) - centroid[:2]) ** 2)
        R[2] = np.sum((self.inputs["member2:nodes_xyz"][:2, :2].mean(axis=0) - centroid[:2]) ** 2)
        R[3] = np.sum((self.inputs["member3:nodes_xyz"][:2, :2].mean(axis=0) - centroid[:2]) ** 2)
        R[4] = np.sum((self.inputs["member4:nodes_xyz"][:2, :2].mean(axis=0) - centroid[:2]) ** 2)
        R[5] = np.sum((self.inputs["member5:nodes_xyz"][:2, :2].mean(axis=0) - centroid[:2]) ** 2)

        npt.assert_equal(self.outputs["platform_center_of_buoyancy"], centroid)
        npt.assert_equal(self.outputs["platform_centroid"], centroid)
        cg = (6e3 * centroid + 1e3 * np.array([0.0, 0.0, 1.0])) / 7e3
        npt.assert_equal(self.outputs["platform_hull_center_of_mass"], cg)
        self.assertEqual(self.outputs["platform_ballast_mass"], 6e2)
        self.assertEqual(self.outputs["platform_hull_mass"], 6e3 + 1e3 - 6e2)
        self.assertEqual(self.outputs["platform_cost"], 6 * 2e3 + 3e3)
        self.assertEqual(self.outputs["platform_Awater"], 30)
        self.assertEqual(self.outputs["platform_Iwater"], 6 * 15 + 5 * R.sum())
        npt.assert_equal(self.outputs["platform_added_mass"], 6 * np.arange(6))
        npt.assert_equal(self.outputs["platform_variable_capacity"], 10 + np.arange(6))
        npt.assert_equal(self.outputs["transition_piece_I"], 1e3 * 0.5 ** 2 * np.r_[0.5, 0.5, 1.0, np.zeros(3)])
        npt.assert_array_less(1e2, self.outputs["platform_I_hull"] - self.outputs["transition_piece_I"])

    def testPlatformTurbineSystem(self):
        myobj = sys.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs:
            self.inputs[k] = self.outputs[k]
        for k in self.discrete_outputs:
            self.discrete_inputs[k] = self.discrete_outputs[k]
        myobj = sys.PlatformTurbineSystem()
        myobj.compute(self.inputs, self.outputs)

        self.assertEqual(self.outputs["system_structural_mass"], 6e3 + 1e4 + 1e3)
        npt.assert_equal(
            self.outputs["system_structural_center_of_mass"],
            (6e3 * np.array([0.375, 0.25, 0.25]) + 1e3 * np.array([0.0, 0.0, 1.0]) + 1e4 * np.array([0.0, 0.0, 50.0]))
            / 1.7e4,
        )
        self.assertEqual(self.outputs["variable_ballast_mass"], 6e4 - self.outputs["system_structural_mass"] - 3e3 / g)
        self.assertAlmostEqual(
            self.outputs["constr_variable_margin"],
            self.outputs["variable_ballast_mass"] / 1e3 / (10 + np.arange(6)).sum(),
        )
        frac = (10 + np.arange(6)) / (10 + np.arange(6)).sum()
        V_frac = self.outputs["variable_ballast_mass"] / 1e3 * frac
        npt.assert_almost_equal(self.outputs["member_variable_volume"], V_frac)
        s_cg = np.interp(0.5 * V_frac, np.arange(10), np.linspace(0, 0.5, 10))
        s_end = np.interp(V_frac, np.arange(10), np.linspace(0, 0.5, 10))
        npt.assert_almost_equal(self.outputs["member_variable_height"], s_end)
        cg_mem = np.zeros((6, 3))
        for k in range(6):
            cg_mem[k, :] = (
                s_cg[k] * np.diff(self.inputs[f"member{k}:nodes_xyz"][:2, :], axis=0)
                + self.inputs[f"member{k}:nodes_xyz"][0, :]
            )
        cg_var = np.dot(V_frac, cg_mem) / (self.outputs["variable_ballast_mass"] / 1e3)
        self.assertEqual(self.outputs["system_mass"], 6e3 + 1e4 + 1e3 + self.outputs["variable_ballast_mass"])
        npt.assert_almost_equal(
            self.outputs["system_center_of_mass"],
            (
                (6e3 + 1e4 + 1e3) * self.outputs["system_structural_center_of_mass"]
                + self.outputs["variable_ballast_mass"] * cg_var
            )
            / self.outputs["system_mass"],
        )
        npt.assert_array_less(self.outputs["platform_I_hull"], self.outputs["platform_I_total"])
        self.assertEqual(self.outputs["platform_mass"], 6e3 + 1e3 + self.outputs["variable_ballast_mass"])


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
        opt["mooring"] = {}
        opt["mooring"]["n_attach"] = 3
        opt["mooring"]["n_anchors"] = 3

        opt["materials"] = {}
        opt["materials"]["n_mat"] = 2

        prob = om.Problem()
        prob.model = sys.FloatingSystem(modeling_options=opt)
        prob.setup()

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
            prob[f"member{k}:section_J0"][:1] = 2 * k * np.ones(1) + 1
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

        # Set environment to that used in OC3 testing campaign
        prob["rho_water"] = 1025.0  # Density of water [kg/m^3]

        prob["mooring_neutral_load"] = np.zeros((3, 3))
        prob["mooring_neutral_load"][:, 0] = [200, -100.0, -100]
        prob["mooring_neutral_load"][:, 1] = [0.0, 50, -50]
        prob["mooring_neutral_load"][:, 2] = -1e3
        prob["transition_piece_mass"] = 1e3
        prob["transition_piece_cost"] = 1e4
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
