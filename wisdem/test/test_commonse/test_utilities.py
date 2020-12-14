import unittest

import numpy as np
import numpy.testing as npt
import wisdem.commonse.utilities as util

npts = 100
myones = np.ones((npts,))


class TestAny(unittest.TestCase):
    def testNodal2Sectional(self):
        x, dx = util.nodal2sectional(np.array([8.0, 10.0, 12.0]))
        npt.assert_equal(x, np.array([9.0, 11.0]))
        npt.assert_equal(dx, np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]))

    def testSectionalInterp(self):
        x = np.arange(0.0, 2.1, 0.5)
        y = np.array([-1.0, 1.0, -2.0, 2.0])
        xi = np.array([-0.1, 0.25, 0.9, 1.4, 1.5, 1.6, 2.1])
        yi = util.sectionalInterp(xi, x, y)

        y_expect = np.array([-1.0, -1.0, 1.0, -2.0, 2.0, 2.0, 2.0])
        npt.assert_array_equal(yi, y_expect)

    def testModalCoefficients(self):
        # Test exact 6-deg polynomial
        p = np.random.random((7,))
        x = np.linspace(0, 1)
        y = np.polynomial.polynomial.polyval(x, p)

        pp = util.get_modal_coefficients(x, y)
        npt.assert_almost_equal(p[2:] / p[2:].sum(), pp)

    def testGetXYModes(self):
        r = np.linspace(0, 1, 20)
        n = 10
        n2 = int(n / 2)
        dx = dy = dz = np.tile(np.r_[r ** 2 + 10.0], (n, 1))
        freqs = np.arange(n)
        xm = np.zeros(n)
        ym = np.zeros(n)
        zm = np.zeros(n)
        xm[0] = xm[3] = xm[6] = xm[9] = 1
        ym[1] = ym[4] = ym[7] = 1
        zm[2] = zm[5] = zm[8] = 1

        freq_x, freq_y, _, _ = util.get_xy_mode_shapes(r, freqs, dx, dy, dz, xm, ym, zm)
        npt.assert_array_equal(freq_x, np.r_[0, 3, 6, 9, np.zeros(n2 - 4)])
        npt.assert_array_equal(freq_y, np.r_[1, 4, 7, np.zeros(n2 - 3)])

    def testRotateI(self):
        I = np.arange(6) + 1
        th = np.deg2rad(45)
        Irot = util.rotateI(I, th, axis="z")
        npt.assert_almost_equal(Irot, np.array([-2.5, 5.5, 3, -0.5, -0.5 * np.sqrt(2), 5.5 * np.sqrt(2)]))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAny))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
