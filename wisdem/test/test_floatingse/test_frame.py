import unittest

import numpy as np
import numpy.testing as npt
import wisdem.floatingse.floating_frame as frame
import wisdem.floatingse.floating_system as sys
from wisdem.commonse.cylinder_member import NULL, MEMMAX


class TestFrame(unittest.TestCase):
    def testTetrahedron(self):
        inputs = {}
        outputs = {}

        opt = {}
        opt["floating"] = {}
        opt["flags"] = {}
        opt["flags"]["floating"] = True
        opt["flags"]["tower"] = False
        opt["WISDEM"] = {}
        opt["WISDEM"]["n_dlc"] = 1
        opt["WISDEM"]["FloatingSE"] = {}
        opt["floating"]["members"] = {}
        opt["floating"]["members"]["n_members"] = n_member = 6
        opt["WISDEM"]["FloatingSE"]["frame3dd"] = {}
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["shear"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["geom"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["tol"] = 1e-8
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["modal"] = False
        opt["WISDEM"]["FloatingSE"]["gamma_f"] = 1.35  # Safety factor on loads
        opt["WISDEM"]["FloatingSE"]["gamma_m"] = 1.3  # Safety factor on materials
        opt["WISDEM"]["FloatingSE"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
        opt["WISDEM"]["FloatingSE"]["gamma_b"] = 1.1  # Safety factor on buckling
        opt["WISDEM"]["FloatingSE"]["gamma_fatigue"] = 1.755  # Not used
        opt["mooring"] = {}
        opt["mooring"]["n_attach"] = 3

        for k in range(n_member):
            for var in ["Px", "Py", "Pz", "qdyn"]:
                inputs[f"member{k}:{var}"] = NULL * np.ones((MEMMAX, 1))

            inputs[f"member{k}:Px"][:2, :] = 1.0
            inputs[f"member{k}:Py"][:2, :] = 2.0
            inputs[f"member{k}:Pz"][:2, :] = 3.0
            inputs[f"member{k}:qdyn"][:2, :] = 4.0

        myobj = frame.PlatformLoads(options=opt)
        myobj.compute(inputs, outputs)

        npt.assert_equal(outputs["platform_elem_Px1"][6:, :], NULL)
        npt.assert_equal(outputs["platform_elem_Py1"][6:, :], NULL)
        npt.assert_equal(outputs["platform_elem_Pz1"][6:, :], NULL)
        npt.assert_equal(outputs["platform_elem_Px2"][6:, :], NULL)
        npt.assert_equal(outputs["platform_elem_Py2"][6:, :], NULL)
        npt.assert_equal(outputs["platform_elem_Pz2"][6:, :], NULL)
        npt.assert_equal(outputs["platform_elem_qdyn"][6:, :], NULL)

        npt.assert_equal(outputs["platform_elem_Px1"][:6, :], 1.0)
        npt.assert_equal(outputs["platform_elem_Py1"][:6, :], 2.0)
        npt.assert_equal(outputs["platform_elem_Pz1"][:6, :], 3.0)
        npt.assert_equal(outputs["platform_elem_Px2"][:6, :], 1.0)
        npt.assert_equal(outputs["platform_elem_Py2"][:6, :], 2.0)
        npt.assert_equal(outputs["platform_elem_Pz2"][:6, :], 3.0)
        npt.assert_equal(outputs["platform_elem_qdyn"][:6, :], 4.0)

    def testAnalysis(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        opt = {}
        opt["floating"] = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["n_dlc"] = 1
        opt["WISDEM"]["FloatingSE"] = {}
        opt["floating"]["members"] = {}
        opt["floating"]["members"]["n_members"] = n_member = 6
        opt["WISDEM"]["FloatingSE"]["frame3dd"] = {}
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["shear"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["geom"] = True
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["tol"] = 1e-8
        opt["WISDEM"]["FloatingSE"]["frame3dd"]["modal"] = False
        opt["WISDEM"]["FloatingSE"]["gamma_f"] = 1.35  # Safety factor on loads
        opt["WISDEM"]["FloatingSE"]["gamma_m"] = 1.3  # Safety factor on materials
        opt["WISDEM"]["FloatingSE"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
        opt["WISDEM"]["FloatingSE"]["gamma_b"] = 1.1  # Safety factor on buckling
        opt["WISDEM"]["FloatingSE"]["gamma_fatigue"] = 1.755  # Not used
        opt["mooring"] = {}
        opt["mooring"]["n_attach"] = 3

        inputs["tower_nodes"] = NULL * np.ones((MEMMAX, 3))
        inputs["tower_Rnode"] = NULL * np.ones(MEMMAX)
        for k in range(n_member):
            inputs[f"member{k}:nodes_xyz"] = NULL * np.ones((MEMMAX, 3))
            inputs[f"member{k}:nodes_r"] = NULL * np.ones(MEMMAX)

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
            "Px",
            "Py",
            "Pz",
            "qdyn",
        ]:
            inputs["tower_elem_" + var] = NULL * np.ones(MEMMAX)
            for k in range(n_member):
                if var in ["Px", "Py", "Pz", "qdyn"]:
                    inputs[f"member{k}:{var}"] = NULL * np.ones((MEMMAX, 1))
                else:
                    inputs[f"member{k}:section_{var}"] = NULL * np.ones(MEMMAX)

        inputs["member0:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [1, 0, 0]])
        inputs["member1:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0.5, 1, 0]])
        inputs["member2:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 0]])
        inputs["member3:nodes_xyz"][:2, :] = np.array([[0, 0, 0], [0, 0, 1]])
        inputs["member4:nodes_xyz"][:2, :] = np.array([[1, 0, 0], [0, 0, 1]])
        inputs["member5:nodes_xyz"][:2, :] = np.array([[0.5, 1, 0], [0, 0, 1]])
        for k in range(n_member):
            L = np.sqrt(np.sum(np.diff(inputs[f"member{k}:nodes_xyz"][:2, :], axis=0) ** 2))
            inputs[f"member{k}:nodes_r"][:2] = 0.1 * k * np.ones(2)
            inputs[f"member{k}:section_D"][:1] = 2.0
            inputs[f"member{k}:section_t"][:1] = 0.1
            inputs[f"member{k}:section_A"][:1] = 0.5 * k * np.ones(1) + 1
            inputs[f"member{k}:section_Asx"][:1] = 0.5 * k * np.ones(1) + 1
            inputs[f"member{k}:section_Asy"][:1] = 0.5 * k * np.ones(1) + 1
            inputs[f"member{k}:section_Ixx"][:1] = 2 * k * np.ones(1) + 1
            inputs[f"member{k}:section_Iyy"][:1] = 2 * k * np.ones(1) + 1
            inputs[f"member{k}:section_J0"][:1] = 2 * k * np.ones(1) + 1
            inputs[f"member{k}:section_rho"][:1] = 1e3 / (0.5 * k * np.ones(1) + 1) / L
            inputs[f"member{k}:section_E"][:1] = 3 * k * np.ones(1) + 1
            inputs[f"member{k}:section_G"][:1] = 4 * k * np.ones(1) + 1
            inputs[f"member{k}:section_sigma_y"][:1] = 5 * k * np.ones(1) + 1
            inputs[f"member{k}:idx_cb"] = 0
            inputs[f"member{k}:buoyancy_force"] = 1e2
            inputs[f"member{k}:displacement"] = 1e1
            inputs[f"member{k}:center_of_buoyancy"] = inputs[f"member{k}:nodes_xyz"][:2, :].mean(axis=0)
            inputs[f"member{k}:center_of_mass"] = inputs[f"member{k}:nodes_xyz"][:2, :].mean(axis=0)
            inputs[f"member{k}:total_mass"] = 1e3
            inputs[f"member{k}:total_cost"] = 2e3
            inputs[f"member{k}:I_total"] = 1e2 + np.arange(6)
            inputs[f"member{k}:Awater"] = 5.0
            inputs[f"member{k}:Iwater"] = 15.0
            inputs[f"member{k}:added_mass"] = np.arange(6)
            inputs[f"member{k}:ballast_mass"] = 1e2
            inputs[f"member{k}:variable_ballast_capacity"] = 10 + k
            inputs[f"member{k}:variable_ballast_spts"] = np.linspace(0, 0.5, 10)
            inputs[f"member{k}:variable_ballast_Vpts"] = np.arange(10)
            inputs[f"member{k}:waterline_centroid"] = inputs[f"member{k}:nodes_xyz"][:2, :2].mean(axis=0)
            inputs[f"member{k}:Px"][:2, :] = 1.0
            inputs[f"member{k}:Py"][:2, :] = 2.0
            inputs[f"member{k}:Pz"][:2, :] = 3.0
            inputs[f"member{k}:qdyn"][:2, :] = 4.0

        inputs["mooring_neutral_load"] = np.zeros((3, 3))
        inputs["mooring_neutral_load"][:, 0] = [200, -100.0, -100]
        inputs["mooring_neutral_load"][:, 1] = [0.0, 50, -50]
        inputs["mooring_neutral_load"][:, 2] = -1e3
        inputs["mooring_fairlead_joints"] = np.array([[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]])
        inputs["mooring_stiffness"] = 5 * np.eye(6)
        inputs["transition_node"] = inputs["member0:nodes_xyz"][1, :]
        inputs["turbine_mass"] = 1e4
        inputs["turbine_cg"] = np.array([0, 0, 50])
        inputs["turbine_I"] = 1e6 * np.ones(6)
        inputs["transition_piece_mass"] = 1e3
        inputs["transition_piece_cost"] = 3e3
        inputs["rho_water"] = 1e3

        inputs["turbine_F"] = 1e3 * np.ones((3, 1))
        inputs["turbine_M"] = 1e4 * np.ones((3, 1))

        myobj = sys.PlatformFrame(options=opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        for k in outputs:
            inputs[k] = outputs[k]
        for k in discrete_outputs:
            discrete_inputs[k] = discrete_outputs[k]
        myobj = sys.PlatformTurbineSystem()
        myobj.compute(inputs, outputs)
        for k in outputs:
            inputs[k] = outputs[k]
        for k in discrete_outputs:
            discrete_inputs[k] = discrete_outputs[k]
        myobj = frame.PlatformLoads(options=opt)
        myobj.compute(inputs, outputs)
        for k in outputs:
            inputs[k] = outputs[k]
        for k in discrete_outputs:
            discrete_inputs[k] = discrete_outputs[k]
        myobj = frame.FrameAnalysis(options=opt)
        myobj.compute(inputs, outputs)
        self.assertTrue(True)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFrame))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
