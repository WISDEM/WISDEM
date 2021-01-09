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
        self.opt["mooring"] = {}
        self.opt["mooring"]["n_attach"] = 3

        self.inputs["tower_nodes"] = NULL * np.ones((MEMMAX, 3))
        self.inputs["tower_Rnode"] = NULL * np.ones(MEMMAX)
        for k in range(n_member):
            self.inputs["member" + str(k) + ":nodes_xyz"] = NULL * np.ones((MEMMAX, 3))
            self.inputs["member" + str(k) + ":nodes_r"] = NULL * np.ones(MEMMAX)

        for var in ["D", "t", "A", "Asx", "Asy", "Ixx", "Iyy", "Izz", "rho", "E", "G"]:
            self.inputs["tower_elem_" + var] = NULL * np.ones(MEMMAX)
            for k in range(n_member):
                self.inputs["member" + str(k) + ":section_" + var] = NULL * np.ones(MEMMAX)

        self.inputs["member0:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [1, 0, 0]])
        self.inputs["member1:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0.5, 1, 0]])
        self.inputs["member2:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 0]])
        self.inputs["member3:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [0, 0, 1]])
        self.inputs["member4:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0, 0, 1]])
        self.inputs["member5:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 1]])
        for k in range(n_member):
            L = np.sqrt(np.sum(np.diff(self.inputs["member" + str(k) + ":nodes_xyz"][:2, :], axis=0) ** 2))
            self.inputs["member" + str(k) + ":nodes_r"][:2] = 0.1 * k * np.ones(2)
            self.inputs["member" + str(k) + ":section_D"][:1] = 2.0
            self.inputs["member" + str(k) + ":section_t"][:1] = 0.1
            self.inputs["member" + str(k) + ":section_A"][:1] = 0.5 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Asx"][:1] = 0.5 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Asy"][:1] = 0.5 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Ixx"][:1] = 2 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Iyy"][:1] = 2 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Izz"][:1] = 2 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_rho"][:1] = 1e3 / (0.5 * k * np.ones(1) + 1) / L
            self.inputs["member" + str(k) + ":section_E"][:1] = 3 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_G"][:1] = 4 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":idx_cb"] = 0
            self.inputs["member" + str(k) + ":buoyancy_force"] = 1e2
            self.inputs["member" + str(k) + ":displacement"] = 1e1
            self.inputs["member" + str(k) + ":center_of_buoyancy"] = self.inputs["member" + str(k) + ":nodes_xyz"][
                :2, :
            ].mean(axis=0)
            self.inputs["member" + str(k) + ":center_of_mass"] = self.inputs["member" + str(k) + ":nodes_xyz"][
                :2, :
            ].mean(axis=0)
            self.inputs["member" + str(k) + ":total_mass"] = 1e3
            self.inputs["member" + str(k) + ":total_cost"] = 2e3
            self.inputs["member" + str(k) + ":Awater"] = 5.0
            self.inputs["member" + str(k) + ":Iwater"] = 15.0
            self.inputs["member" + str(k) + ":added_mass"] = np.arange(6)
            self.inputs["member" + str(k) + ":transition_node"] = NULL * np.ones(3)

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
        self.inputs["tower_center_of_mass"] = self.inputs["tower_nodes"][:3, :].mean(axis=0)
        self.inputs["tower_mass"] = 5e3

        self.inputs["mooring_neutral_load"] = np.zeros((3, 3))
        self.inputs["mooring_neutral_load"][:, 0] = [200, -100.0, -100]
        self.inputs["mooring_neutral_load"][:, 1] = [0.0, 50, -50]
        self.inputs["mooring_neutral_load"][:, 2] = -1e3
        self.inputs["mooring_fairlead_joints"] = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
        self.inputs["transition_node"] = self.inputs["tower_nodes"][0, :]
        self.inputs["hub_node"] = self.inputs["tower_nodes"][2, :]
        self.inputs["rna_mass"] = 1e4
        self.inputs["rna_cg"] = np.ones(3)
        self.inputs["rna_I"] = 1e4 * np.arange(6)
        self.inputs["rna_F"] = np.array([1e2, 1e1, 0.0])
        self.inputs["rna_M"] = np.array([2e1, 2e2, 0.0])
        self.inputs["transition_piece_mass"] = 1e3
        self.inputs["rho_water"] = 1e3

    def testTetrahedron(self):
        myobj = frame.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs)

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
        npt.assert_equal(self.outputs["platform_elem_Izz"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_rho"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_E"][6:], NULL)
        npt.assert_equal(self.outputs["platform_elem_G"][6:], NULL)

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
        npt.assert_equal(self.outputs["platform_elem_Izz"][:6], 2 * np.arange(6) + 1)
        # npt.assert_equal(self.outputs["platform_elem_rho"][:6], 3 * np.arange(6)+1)
        npt.assert_equal(self.outputs["platform_elem_E"][:6], 3 * np.arange(6) + 1)
        npt.assert_equal(self.outputs["platform_elem_G"][:6], 4 * np.arange(6) + 1)
        self.assertEqual(self.outputs["platform_displacement"], 6e1)
        npt.assert_equal(
            self.outputs["platform_center_of_buoyancy"], self.outputs["platform_nodes"][:4, :].mean(axis=0)
        )
        npt.assert_equal(self.outputs["platform_center_of_mass"], self.outputs["platform_nodes"][:4, :].mean(axis=0))
        self.assertEqual(self.outputs["platform_mass"], 6e3)
        self.assertEqual(self.outputs["platform_cost"], 6 * 2e3)
        self.assertEqual(self.outputs["platform_Awater"], 30)
        self.assertEqual(self.outputs["platform_Iwater"], 6 * 15)
        npt.assert_equal(self.outputs["platform_added_mass"], 6 * np.arange(6))
        # Should find a transition mode even though one wasn't set
        npt.assert_equal(self.outputs["transition_node"], [0.0, 0.0, 1.0])

        # Test with set transition node
        self.inputs["member1:transition_node"] = [0.5, 1.0, 0.0]
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs)
        npt.assert_equal(self.outputs["transition_node"], [0.5, 1.0, 0.0])

    def testPre(self):
        inputs = {}
        outputs = {}
        inputs["transition_node"] = np.array([1, 1, 2])
        inputs["hub_height"] = 100.0
        inputs["distance_tt_hub"] = 5.0
        myobj = frame.TowerPreMember()
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["hub_node"], np.array([1, 1, 95]))

    def testPlatformTower(self):
        myobj = frame.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs)
        for k in self.outputs:
            self.inputs[k] = self.outputs[k]
        for k in self.discrete_outputs:
            self.discrete_inputs[k] = self.discrete_outputs[k]
        myobj = frame.PlatformTowerFrame()
        myobj.compute(self.inputs, self.outputs)

        npt.assert_equal(
            self.outputs["system_nodes"][:6, :],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 51.0],
                    [0.0, 0.0, 101.0],
                ]
            ),
        )
        npt.assert_equal(
            self.outputs["system_Fnode"][:6, :],
            1e2
            * np.array(
                [[0.0, 0.0, 2], [0.0, 0.0, 0.0], [0.0, 0.0, 2], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        )
        npt.assert_equal(self.outputs["system_Rnode"][:6], 0.1 * np.r_[3, 5, 5, 4, 0, 0])
        npt.assert_equal(self.outputs["system_elem_n1"][:8], np.r_[0, 3, 2, 0, 3, 2, 1, 4])
        npt.assert_equal(self.outputs["system_elem_n2"][:8], np.r_[3, 2, 0, 1, 1, 1, 4, 5])
        npt.assert_equal(self.outputs["system_elem_D"][:8], 2.0)
        npt.assert_equal(self.outputs["system_elem_t"][:8], 0.1)
        npt.assert_equal(self.outputs["system_elem_A"][:8], np.r_[0.5 * np.arange(6) + 1, 4, 4])
        npt.assert_equal(self.outputs["system_elem_Asx"][:8], np.r_[0.5 * np.arange(6) + 1, 4.1, 4.1])
        npt.assert_equal(self.outputs["system_elem_Asy"][:8], np.r_[0.5 * np.arange(6) + 1, 4.1, 4.1])
        npt.assert_equal(self.outputs["system_elem_Ixx"][:8], np.r_[2 * np.arange(6) + 1, 5, 5])
        npt.assert_equal(self.outputs["system_elem_Iyy"][:8], np.r_[2 * np.arange(6) + 1, 5, 5])
        npt.assert_equal(self.outputs["system_elem_Izz"][:8], np.r_[2 * np.arange(6) + 1, 10, 10])
        # npt.assert_equal(self.outputs["system_elem_rho"][:8], np.r_[3 * np.arange(6)+1, 20, 20])
        npt.assert_equal(self.outputs["system_elem_E"][:8], np.r_[3 * np.arange(6) + 1, 30, 30])
        npt.assert_equal(self.outputs["system_elem_G"][:8], np.r_[4 * np.arange(6) + 1, 40, 40])
        self.assertEqual(self.outputs["system_mass"], 6e3 + 5e3 + 1e4 + 1e3)
        self.assertEqual(self.outputs["variable_ballast_mass"], 6e4 - self.outputs["system_mass"])
        npt.assert_equal(
            self.outputs["system_center_of_mass"],
            (
                6e3 * np.array([0.375, 0.25, 0.25])
                + 5e3 * np.array([0.0, 0.0, 51.0])
                + 1e3 * np.array([0.0, 0.0, 1.0])
                + 1e4 * np.array([1.0, 1.0, 102.0])
            )
            / 2.2e4,
        )
        npt.assert_equal(self.outputs["transition_piece_I"], 1e3 * 0.25 * np.r_[0.5, 0.5, 1.0, np.zeros(3)])

    def testRunFrame(self):
        myobj = frame.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs)
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
        opt["floating"]["tower"] = {}
        opt["floating"]["tower"]["n_height"] = [3]
        opt["floating"]["tower"]["n_bulkheads"] = [0]
        opt["floating"]["tower"]["n_layers"] = [1]
        opt["floating"]["tower"]["n_ballasts"] = [0]
        opt["floating"]["tower"]["n_axial_joints"] = [0]
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
            L = np.sqrt(np.sum(np.diff(prob["member" + str(k) + ":nodes_xyz"][:, 2], axis=0) ** 2))
            prob["member" + str(k) + ":nodes_r"][:2] = 0.1 * k * np.ones(2)
            prob["member" + str(k) + ":section_D"][:1] = 2.0
            prob["member" + str(k) + ":section_t"][:1] = 0.1
            prob["member" + str(k) + ":section_A"][:1] = 0.5 * k * np.ones(1) + 1
            prob["member" + str(k) + ":section_Asx"][:1] = 0.5 * k * np.ones(1) + 1
            prob["member" + str(k) + ":section_Asy"][:1] = 0.5 * k * np.ones(1) + 1
            prob["member" + str(k) + ":section_Ixx"][:1] = 2 * k * np.ones(1) + 1
            prob["member" + str(k) + ":section_Iyy"][:1] = 2 * k * np.ones(1) + 1
            prob["member" + str(k) + ":section_Izz"][:1] = 2 * k * np.ones(1) + 1
            prob["member" + str(k) + ":section_rho"][:1] = 1e3 / (0.5 * k * np.ones(1) + 1) / L
            prob["member" + str(k) + ":section_E"][:1] = 3 * k * np.ones(1) + 1
            prob["member" + str(k) + ":section_G"][:1] = 4 * k * np.ones(1) + 1
            prob["member" + str(k) + ":idx_cb"] = 0
            prob["member" + str(k) + ":buoyancy_force"] = 1e2
            prob["member" + str(k) + ":displacement"] = 1e1
            prob["member" + str(k) + ":center_of_buoyancy"] = prob["member" + str(k) + ":nodes_xyz"][:2, :].mean(axis=0)
            prob["member" + str(k) + ":center_of_mass"] = prob["member" + str(k) + ":nodes_xyz"][:2, :].mean(axis=0)
            prob["member" + str(k) + ":total_mass"] = 1e3
            prob["member" + str(k) + ":total_cost"] = 2e3
            prob["member" + str(k) + ":Awater"] = 5.0
            prob["member" + str(k) + ":Iwater"] = 15.0
            prob["member" + str(k) + ":added_mass"] = np.arange(6)

        # Set environment to that used in OC3 testing campaign
        # prob["rho_air"] = 1.226  # Density of air [kg/m^3]
        # prob["mu_air"] = 1.78e-5  # Viscosity of air [kg/m/s]
        prob["rho_water"] = 1025.0  # Density of water [kg/m^3]
        # prob["mu_water"] = 1.08e-3  # Viscosity of water [kg/m/s]
        # prob["water_depth"] = 320.0  # Distance to sea floor [m]
        # prob["Hsig_wave"] = 0.0  # Significant wave height [m]
        # prob["Tsig_wave"] = 1e3  # Wave period [s]
        # prob["shearExp"] = 0.11  # Shear exponent in wind power law
        # prob["cm"] = 2.0  # Added mass coefficient
        # prob["Uc"] = 0.0  # Mean current speed
        # prob["yaw"] = 0.0  # Turbine yaw angle
        # prob["beta_wind"] = prob["beta_wave"] = 0.0
        # prob["cd_usr"] = -1.0  # Compute drag coefficient
        # prob["Uref"] = 10.0
        # prob["zref"] = 100.0

        # Porperties of turbine tower
        nTower = prob.model.options["modeling_options"]["floating"]["tower"]["n_height"][0]
        prob["hub_height"] = 85.0
        prob["distance_tt_hub"] = 5.0
        prob["tower.s"] = np.linspace(0.0, 1.0, nTower)
        prob["tower.outer_diameter_in"] = np.linspace(6.5, 3.87, nTower)
        prob["tower.layer_thickness"] = np.linspace(0.027, 0.019, nTower).reshape((1, nTower))
        prob["tower.layer_materials"] = ["steel"]
        prob["tower.outfitting_factor"] = 1.07

        prob["mooring_neutral_load"] = np.zeros((3, 3))
        prob["mooring_neutral_load"][:, 0] = [200, -100.0, -100]
        prob["mooring_neutral_load"][:, 1] = [0.0, 50, -50]
        prob["mooring_neutral_load"][:, 2] = -1e3
        prob["mooring_fairlead_joints"] = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
        prob["rna_mass"] = 1e4
        prob["rna_cg"] = np.ones(3)
        prob["rna_I"] = 1e4 * np.arange(6)
        prob["rna_F"] = np.array([1e2, 1e1, 0.0])
        prob["rna_M"] = np.array([2e1, 2e2, 0.0])
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
