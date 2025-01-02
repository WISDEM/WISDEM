import unittest

import numpy as np
import numpy.testing as npt

from wisdem.commonse.cross_sections import Tube, Rectangle

npts = 100


class TestTube(unittest.TestCase):
    def testTubeScalar(self):
        T = Tube(2 * 5.0, 1.0)

        self.assertAlmostEqual(T.Area, np.pi * 9.0)
        self.assertAlmostEqual(T.Ixx, np.pi * 369.0 / 4.0)
        self.assertAlmostEqual(T.Iyy, np.pi * 369.0 / 4.0)
        self.assertAlmostEqual(T.J0, np.pi * 369.0 / 2.0)
        self.assertAlmostEqual(T.S, np.pi * 369.0 / 4.0 / 5.0)
        self.assertAlmostEqual(T.C, np.pi * 369.0 / 2.0 / 5.0)

    def testTubeVector(self):
        n = 5
        my1 = np.ones(n)
        T = Tube(2 * 5.0 * my1, my1)

        npt.assert_almost_equal(T.Area, np.pi * 9.0)
        npt.assert_almost_equal(T.Ixx, np.pi * 369.0 / 4.0)
        npt.assert_almost_equal(T.Iyy, np.pi * 369.0 / 4.0)
        npt.assert_almost_equal(T.J0, np.pi * 369.0 / 2.0)
        npt.assert_almost_equal(T.S, np.pi * 369.0 / 4.0 / 5.0)
        npt.assert_almost_equal(T.C, np.pi * 369.0 / 2.0 / 5.0)

class TestRectangle(unittest.TestCase):
    def testRectangleScalar(self):
        R = Rectangle(2.0, 4.0, 0.5)

        self.assertAlmostEqual(R.Area, 5.0)
        self.assertAlmostEqual(R.Ixx, 8.41666667)
        self.assertAlmostEqual(R.Iyy, 2.41666667)
        self.assertAlmostEqual(R.J0, 10.83333333)
        self.assertAlmostEqual(R.BdgMxx, 4.20833333)
        self.assertAlmostEqual(R.BdgMyy, 2.41666667)

    def testRectangleError(self):
        try:
            Rectangle(1.0, 2.0, 3.0)
        except Exception as err:
            assert isinstance(err, AssertionError)
        else:
            raise AssertionError("1, 2, 3, should not define a rectangular propertly")

    def testRectangleVector(self):
        n = 5
        my1 = np.ones(n)
        R = Rectangle(2.0 * my1, 4.0 * my1, 0.5 * my1)

        npt.assert_almost_equal(R.Area, 5.0)
        npt.assert_almost_equal(R.Ixx, 8.41666667)
        npt.assert_almost_equal(R.Iyy, 2.41666667)
        npt.assert_almost_equal(R.J0, 10.83333333)
        npt.assert_almost_equal(R.BdgMxx, 4.20833333)
        npt.assert_almost_equal(R.BdgMyy, 2.41666667)




if __name__ == "__main__":
    unittest.main()
