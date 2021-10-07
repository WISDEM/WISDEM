import unittest

import numpy as np
import numpy.testing as npt

import wisdem.commonse.utilities as util
import wisdem.floatingse.constraints as cons
from wisdem.commonse.cylinder_member import NULL, MEMMAX


class TestConstraints(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}

        opt = {}
        opt["floating"] = {}
        opt["WISDEM"] = {}
        opt["WISDEM"]["FloatingSE"] = {}
        opt["floating"]["members"] = {}
        opt["floating"]["members"]["n_members"] = n_member = 6
        opt["floating"]["members"]["n_ballasts"] = [2] * 6
        opt["mooring"] = {}
        opt["mooring"]["n_attach"] = 3

        for k in range(n_member):
            inputs[f"member{k}:nodes_xyz"] = NULL * np.ones((MEMMAX, 3))
            inputs[f"member{k}:constr_ballast_capacity"] = np.array([0.6])
            inputs[f"member{k}:nodes_xyz"][:2, :] = np.array([[0, 0, -1], [0, 0, 1]])

        inputs["Hsig_wave"] = 10.0
        inputs["variable_ballast_mass"] = 3.0
        inputs["fairlead"] = 20.0
        inputs["fairlead_radius"] = 10.0
        inputs["operational_heel"] = np.deg2rad(5.0)
        inputs["survival_heel"] = np.deg2rad(20.0)
        inputs["platform_Iwater"] = 1e4
        inputs["platform_displacement"] = 1e5
        inputs["platform_center_of_buoyancy"] = np.ones(3)
        inputs["system_center_of_mass"] = np.array([0.0, 0.0, 5.0])
        inputs["transition_node"] = np.array([0.0, 0.0, 10.0])
        inputs["turbine_F"] = np.array([1e2, 1e1, 0.0]).reshape((-1, 1))
        inputs["turbine_M"] = np.array([2e1, 2e2, 0.0]).reshape((-1, 1))
        inputs["max_surge_restoring_force"] = 1e5
        inputs["operational_heel_restoring_force"] = 2e5 * np.ones(6)
        inputs["survival_heel_restoring_force"] = 3e5 * np.ones(6)

        myobj = cons.FloatingConstraints(modeling_options=opt)

        myobj.compute(inputs, outputs)
        _, free2 = util.rotate(0.0, 0.0, 0.0, -4.0, np.deg2rad(20))
        _, draft2 = util.rotate(0.0, 0.0, 0.0, -6.0, np.deg2rad(20))
        npt.assert_equal(outputs["constr_fixed_margin"], 0.6 * np.ones(6))
        self.assertEqual(outputs["constr_fairlead_wave"], 1.1 * 0.5)
        self.assertEqual(outputs["metacentric_height"], 0.1 - (5 - 1))
        self.assertEqual(outputs["constr_mooring_surge"], 1e5 - 1e2)
        self.assertEqual(outputs["constr_mooring_heel"], 10 * 2e5 + (5 + 20) * 4e5 + 2e5 - 1e2 * (10 - 5) - 2e2)
        npt.assert_equal(outputs["constr_freeboard_heel_margin"], -(-4.0 - free2))
        npt.assert_equal(outputs["constr_draft_heel_margin"], -(-6.0 - draft2))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestConstraints))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
