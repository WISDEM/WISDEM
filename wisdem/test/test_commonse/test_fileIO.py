import os
import glob
import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt

from wisdem.commonse import fileIO


def clear_files():
    flist = glob.glob("test.*")
    for f in flist:
        os.remove(f)


class MyComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("float_in", 0.0, units="N")
        self.add_input("fraction_in", 0.0)
        self.add_input("array_in", np.zeros(3), units="m")
        self.add_discrete_input("int_in", 0)
        self.add_discrete_input("string_in", "empty")
        self.add_discrete_input("list_in", ["empty"] * 3)

        self.add_output("float_out", 0.0, units="N")
        self.add_output("fraction_out", 0.0)
        self.add_output("array_out", np.zeros(3), units="m")
        self.add_discrete_output("int_out", 0)
        self.add_discrete_output("string_out", "empty")
        self.add_discrete_output("list_out", ["empty"] * 3)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        outputs["float_out"] = inputs["float_in"] + 1
        outputs["fraction_out"] = inputs["fraction_in"] + 0.1
        outputs["array_out"] = inputs["array_in"] + 1
        discrete_outputs["int_out"] = discrete_inputs["int_in"] + 1
        discrete_outputs["string_out"] = discrete_inputs["string_in"] + "_full"
        discrete_outputs["list_out"] = discrete_inputs["list_in"] + ["full"] * 3


class MyGroup(om.Group):
    def setup(self):
        self.add_subsystem("comp", MyComp(), promotes=["*"])


class TestFileIO(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem(reports=False, model=MyGroup())
        self.prob.setup()
        self.prob["float_in"] = 5.0
        self.prob.run_model()

    def tearDown(self):
        clear_files()

    def testSaveFile(self):
        clear_files()
        fileIO.save_data("test.junk", self.prob, mat_file=False, npz_file=False, xls_file=False)
        self.assertTrue(os.path.exists("test.pkl"))
        self.assertFalse(os.path.exists("test.npz"))
        self.assertFalse(os.path.exists("test.mat"))
        self.assertFalse(os.path.exists("test.xlsx"))

        clear_files()
        fileIO.save_data("test.junk", self.prob, mat_file=False)
        self.assertTrue(os.path.exists("test.pkl"))
        self.assertTrue(os.path.exists("test.npz"))
        self.assertFalse(os.path.exists("test.mat"))
        self.assertTrue(os.path.exists("test.xlsx"))

        clear_files()
        fileIO.save_data("test.junk", self.prob)
        self.assertTrue(os.path.exists("test.pkl"))
        self.assertTrue(os.path.exists("test.npz"))
        self.assertTrue(os.path.exists("test.mat"))
        self.assertTrue(os.path.exists("test.xlsx"))

    def testLoadFile(self):
        clear_files()
        fileIO.save_data("test", self.prob)
        self.prob = fileIO.load_data("test.pkl", self.prob)

        # Check pickle file
        self.assertEqual(self.prob["float_in"], 5.0)
        self.assertEqual(self.prob["float_out"], 6.0)
        self.assertEqual(self.prob["fraction_in"], 0.0)
        self.assertEqual(self.prob["fraction_out"], 0.1)
        npt.assert_equal(self.prob["array_in"], np.zeros(3))
        npt.assert_equal(self.prob["array_out"], np.ones(3))
        self.assertEqual(self.prob["int_in"], 0)
        self.assertEqual(self.prob["int_out"], 1)
        self.assertEqual(self.prob["string_in"], "empty")
        self.assertEqual(self.prob["string_out"], "empty_full")
        self.assertEqual(self.prob["list_in"], ["empty"] * 3)
        self.assertEqual(self.prob["list_out"], ["empty"] * 3 + ["full"] * 3)

        # Check numpy file
        npzdat = np.load("test.npz", allow_pickle=True)
        self.assertEqual(npzdat["float_in_N"], 5.0)
        self.assertEqual(npzdat["float_out_N"], 6.0)
        self.assertEqual(npzdat["fraction_in"], 0.0)
        self.assertEqual(npzdat["fraction_out"], 0.1)
        npt.assert_equal(npzdat["array_in_m"], np.zeros(3))
        npt.assert_equal(npzdat["array_out_m"], np.ones(3))
        self.assertEqual(npzdat["int_in"], 0)
        self.assertEqual(npzdat["int_out"], 1)
        self.assertEqual(npzdat["string_in"], "empty")
        self.assertEqual(npzdat["string_out"], "empty_full")
        npt.assert_equal(npzdat["list_in"], ["empty"] * 3)
        npt.assert_equal(npzdat["list_out"], ["empty"] * 3 + ["full"] * 3)


if __name__ == "__main__":
    unittest.main()
