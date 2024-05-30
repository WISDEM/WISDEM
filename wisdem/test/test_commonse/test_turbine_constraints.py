import unittest

import numpy as np
import numpy.testing as npt

import wisdem.commonse.turbine_constraints as tc


class TestMass(unittest.TestCase):
    def testAll(self):
        inputs = {}
        outputs = {}
        inputs["joint2"] = np.array([0.0, 0.0, 125.0])
        inputs["rna_mass"] = np.array([ 1000.0 ])
        inputs["rna_I"] = 3e4 * np.ones(6)
        inputs["rna_cg"] = 20.0 * np.ones(3)
        inputs["tower_mass"] = np.array([ 3000.0 ])
        inputs["tower_center_of_mass"] = 0.5 * inputs["joint2"][-1]
        inputs["tower_I_base"] = 2e4 * np.ones(6)

        myobj = tc.TurbineMass()
        myobj.compute(inputs, outputs)

        self.assertEqual(outputs["turbine_mass"], 4e3)
        h = np.r_[0.0, 0.0, 125.0]
        npt.assert_equal(outputs["turbine_center_of_mass"], (1e3 * (inputs["rna_cg"] + h) + 3e3 * 0.5 * h) / 4e3)
        # npt.assert_array_less(5e4, np.abs(outputs["turbine_I_base"]))

if __name__ == "__main__":
    unittest.main()
