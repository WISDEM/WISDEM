import unittest

import numpy as np
import pytest
import openmdao.api as om
import numpy.testing as npt
import wisdem.floatingse.floating_frame as frame


class TestPlatform(unittest.TestCase):
    def testTetrahedron(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        opt = {}
        opt["floating"] = {}
        opt["floating"]["n_member"] = n_member = 6
        myobj = frame.PlatformFrame(options=opt)
        myobj.node_mem2glob = {}
        myobj.node_glob2mem = {}

        for k in range(n_member):
            inputs["member" + str(k) + ":nodes_r"] = k * np.ones(2)
            inputs["member" + str(k) + ":section_A"] = 0.5 * k * np.ones(1)
            inputs["member" + str(k) + ":section_Asx"] = 0.5 * k * np.ones(1)
            inputs["member" + str(k) + ":section_Asy"] = 0.5 * k * np.ones(1)
            inputs["member" + str(k) + ":section_Ixx"] = 2 * k * np.ones(1)
            inputs["member" + str(k) + ":section_Iyy"] = 2 * k * np.ones(1)
            inputs["member" + str(k) + ":section_Izz"] = 2 * k * np.ones(1)
            inputs["member" + str(k) + ":section_rho"] = 3 * k * np.ones(1)
            inputs["member" + str(k) + ":section_E"] = 4 * k * np.ones(1)
            inputs["member" + str(k) + ":section_G"] = 4 * k * np.ones(1)
        inputs["member0:nodes_xyz"] = np.array([[0, 0, 0], [1, 0, 0]])
        inputs["member1:nodes_xyz"] = np.array([[1, 0, 0], [0.5, 1, 0]])
        inputs["member2:nodes_xyz"] = np.array([[0.5, 1, 0], [0, 0, 0]])
        inputs["member3:nodes_xyz"] = np.array([[0, 0, 0], [0, 0, 1]])
        inputs["member4:nodes_xyz"] = np.array([[1, 0, 0], [0, 0, 1]])
        inputs["member5:nodes_xyz"] = np.array([[0.5, 1, 0], [0, 0, 1]])

        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)

        # Check NULLs and implied number of nodes / elements
        npt.assert_equal(outputs["platform_nodes"][4:, :], frame.NULL)
        npt.assert_equal(outputs["platform_Rnode"][4:], frame.NULL)
        npt.assert_equal(discrete_outputs["platform_elem_n1"][6:], frame.NULL)
        npt.assert_equal(discrete_outputs["platform_elem_n2"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_A"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_Asx"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_Asy"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_Ixx"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_Iyy"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_Izz"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_rho"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_E"][6:], frame.NULL)
        npt.assert_equal(outputs["platform_elem_G"][6:], frame.NULL)

        npt.assert_equal(
            outputs["platform_nodes"][:4, :],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        npt.assert_equal(outputs["platform_Rnode"][:4], np.r_[3, 5, 5, 4])
        npt.assert_equal(discrete_outputs["platform_elem_n1"][:6], np.r_[0, 3, 2, 0, 3, 2])
        npt.assert_equal(discrete_outputs["platform_elem_n2"][:6], np.r_[3, 2, 0, 1, 1, 1])
        npt.assert_equal(outputs["platform_elem_A"][:6], 0.5 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_Asx"][:6], 0.5 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_Asy"][:6], 0.5 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_Ixx"][:6], 2 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_Iyy"][:6], 2 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_Izz"][:6], 2 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_rho"][:6], 3 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_E"][:6], 4 * np.arange(6))
        npt.assert_equal(outputs["platform_elem_G"][:6], 4 * np.arange(6))


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
