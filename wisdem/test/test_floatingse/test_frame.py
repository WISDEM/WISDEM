import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.floatingse.floating_frame as frame
from wisdem.commonse import gravity as g


class TestPlatform(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.opt = {}
        self.opt["floating"] = {}
        self.opt["floating"]["n_member"] = n_member = 6
        self.opt["floating"]["frame3dd"] = {}
        self.opt["floating"]["frame3dd"]["shear"] = True
        self.opt["floating"]["frame3dd"]["geom"] = True
        self.opt["floating"]["frame3dd"]["tol"] = 1e-8
        self.opt["mooring"] = {}
        self.opt["mooring"]["n_lines"] = 3

        self.inputs["member0:nodes_xyz"] = np.array([[0, 0, 0], [1, 0, 0]])
        self.inputs["member1:nodes_xyz"] = np.array([[1, 0, 0], [0.5, 1, 0]])
        self.inputs["member2:nodes_xyz"] = np.array([[0.5, 1, 0], [0, 0, 0]])
        self.inputs["member3:nodes_xyz"] = np.array([[0, 0, 0], [0, 0, 1]])
        self.inputs["member4:nodes_xyz"] = np.array([[1, 0, 0], [0, 0, 1]])
        self.inputs["member5:nodes_xyz"] = np.array([[0.5, 1, 0], [0, 0, 1]])
        for k in range(n_member):
            L = np.sqrt(np.sum(np.diff(self.inputs["member" + str(k) + ":nodes_xyz"], axis=0) ** 2))
            self.inputs["member" + str(k) + ":nodes_r"] = 0.1 * k * np.ones(2)
            self.inputs["member" + str(k) + ":section_A"] = 0.5 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Asx"] = 0.5 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Asy"] = 0.5 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Ixx"] = 2 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Iyy"] = 2 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_Izz"] = 2 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_rho"] = 1e3 / (0.5 * k * np.ones(1) + 1) / L
            self.inputs["member" + str(k) + ":section_E"] = 3 * k * np.ones(1) + 1
            self.inputs["member" + str(k) + ":section_G"] = 4 * k * np.ones(1) + 1
            self.discrete_inputs["member" + str(k) + ":idx_cb"] = 0
            self.inputs["member" + str(k) + ":buoyancy_force"] = 1e2
            self.inputs["member" + str(k) + ":displacement"] = 1e1
            self.inputs["member" + str(k) + ":center_of_buoyancy"] = self.inputs["member" + str(k) + ":nodes_xyz"].mean(
                axis=0
            )
            self.inputs["member" + str(k) + ":center_of_mass"] = self.inputs["member" + str(k) + ":nodes_xyz"].mean(
                axis=0
            )
            self.inputs["member" + str(k) + ":total_mass"] = 1e3
            self.inputs["member" + str(k) + ":total_cost"] = 2e3
            self.inputs["member" + str(k) + ":Awater"] = 5.0
            self.inputs["member" + str(k) + ":Iwater"] = 15.0
            self.inputs["member" + str(k) + ":added_mass"] = np.arange(6)

        myones = np.ones(frame.NELEM_MAX)
        self.inputs["tower_nodes"] = frame.NULL * np.ones((frame.NNODES_MAX, 3))
        self.inputs["tower_Fnode"] = frame.NULL * np.ones((frame.NNODES_MAX, 3))
        self.inputs["tower_Rnode"] = frame.NULL * np.ones(frame.NNODES_MAX)
        self.discrete_inputs["tower_elem_n1"] = np.int_(frame.NULL * myones)
        self.discrete_inputs["tower_elem_n2"] = np.int_(frame.NULL * myones)
        self.inputs["tower_elem_A"] = frame.NULL * myones
        self.inputs["tower_elem_Asx"] = frame.NULL * myones
        self.inputs["tower_elem_Asy"] = frame.NULL * myones
        self.inputs["tower_elem_Ixx"] = frame.NULL * myones
        self.inputs["tower_elem_Iyy"] = frame.NULL * myones
        self.inputs["tower_elem_Izz"] = frame.NULL * myones
        self.inputs["tower_elem_rho"] = frame.NULL * myones
        self.inputs["tower_elem_E"] = frame.NULL * myones
        self.inputs["tower_elem_G"] = frame.NULL * myones

        self.inputs["tower_nodes"][:3, :] = np.array([[0, 0, 1], [0, 0, 51], [0, 0, 101]])
        self.inputs["tower_Fnode"][:3, :] = 0.0
        self.inputs["tower_Rnode"][:3] = 0.0
        self.discrete_inputs["tower_elem_n1"][:2] = [0, 1]
        self.discrete_inputs["tower_elem_n2"][:2] = [1, 2]
        self.inputs["tower_elem_A"][:2] = 4.0
        self.inputs["tower_elem_Asx"][:2] = 4.1
        self.inputs["tower_elem_Asy"][:2] = 4.1
        self.inputs["tower_elem_Ixx"][:2] = 5.0
        self.inputs["tower_elem_Iyy"][:2] = 5.0
        self.inputs["tower_elem_Izz"][:2] = 10.0
        self.inputs["tower_elem_rho"][:2] = 5e3 / 4 / 100
        self.inputs["tower_elem_E"][:2] = 30.0
        self.inputs["tower_elem_G"][:2] = 40.0
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
        self.inputs["transition_piece_I"] = 1e3 * np.arange(6)
        self.inputs["rho_water"] = 1e3

    def testTetrahedron(self):
        myobj = frame.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        # Check NULLs and implied number of nodes / elements
        npt.assert_equal(self.outputs["platform_nodes"][4:, :], frame.NULL)
        npt.assert_equal(self.outputs["platform_Fnode"][4:, :], frame.NULL)
        npt.assert_equal(self.outputs["platform_Rnode"][4:], frame.NULL)
        npt.assert_equal(self.discrete_outputs["platform_elem_n1"][6:], frame.NULL)
        npt.assert_equal(self.discrete_outputs["platform_elem_n2"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_A"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_Asx"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_Asy"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_Ixx"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_Iyy"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_Izz"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_rho"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_E"][6:], frame.NULL)
        npt.assert_equal(self.outputs["platform_elem_G"][6:], frame.NULL)

        npt.assert_equal(
            self.outputs["platform_nodes"][:4, :],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        npt.assert_equal(
            self.outputs["platform_Fnode"][:4, :],
            1e2 * np.array([[0.0, 0.0, 2], [0.0, 0.0, 0.0], [0.0, 0.0, 2], [0.0, 0.0, 2.0]]),
        )
        npt.assert_equal(self.outputs["platform_Rnode"][:4], 0.1 * np.r_[3, 5, 5, 4])
        npt.assert_equal(self.discrete_outputs["platform_elem_n1"][:6], np.r_[0, 3, 2, 0, 3, 2])
        npt.assert_equal(self.discrete_outputs["platform_elem_n2"][:6], np.r_[3, 2, 0, 1, 1, 1])
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

    def testPre(self):
        inputs = {}
        outputs = {}
        inputs["transition_node"] = np.array([1, 1, 2])
        inputs["hub_height"] = 100.0
        myobj = frame.TowerPreMember()
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs["hub_node"], np.array([1, 1, 100]))

    def testPlatformTower(self):
        myobj = frame.PlatformFrame(options=self.opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs:
            self.inputs[k] = self.outputs[k]
        for k in self.discrete_outputs:
            self.discrete_inputs[k] = self.discrete_outputs[k]
        myobj = frame.PlatformTowerFrame()
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

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
        npt.assert_equal(self.discrete_outputs["system_elem_n1"][:8], np.r_[0, 3, 2, 0, 3, 2, 1, 4])
        npt.assert_equal(self.discrete_outputs["system_elem_n2"][:8], np.r_[3, 2, 0, 1, 1, 1, 4, 5])
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
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs:
            self.inputs[k] = self.outputs[k]
        for k in self.discrete_outputs:
            self.discrete_inputs[k] = self.discrete_outputs[k]
        myobj = frame.FrameAnalysis(options=self.opt)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertTrue(True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPlatform))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
