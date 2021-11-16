import unittest

import numpy as np
import numpy.testing as npt
from wisdem.commonse.cross_sections import Tube

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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTube))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
