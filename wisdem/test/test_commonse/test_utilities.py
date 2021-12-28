import unittest

import numpy as np
import numpy.testing as npt
import wisdem.commonse.utilities as util
from scipy.optimize import curve_fit

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
        p[:2] = 0.0
        x = np.linspace(0, 1)
        y = np.polynomial.polynomial.polyval(x, p)

        pp = util.get_modal_coefficients(x, y)
        npt.assert_almost_equal(p[2:] / p[2:].sum(), pp)

        # Test more complex and ensure get the same answer as curve fit
        p = np.random.random((10,))
        y = np.polynomial.polynomial.polyval(x, p)

        pp = util.get_modal_coefficients(x, y)
        cc, _ = curve_fit(util.mode_fit, x, y)
        cc /= cc.sum()
        npt.assert_almost_equal(pp, cc, 4)

    def testModalFASTExample_norm(self):
        x = np.array(
            [
                -10.00000000000,
                -9.55757575758,
                -8.67272727273,
                -7.78787878788,
                -6.90303030303,
                -6.01818181818,
                -5.13333333333,
                -4.24848484848,
                -3.36363636364,
                -2.47878787879,
                -1.59393939394,
                -0.70909090909,
                0.17575757576,
                1.06060606061,
                1.94545454545,
                2.83030303030,
                3.71515151515,
                4.60000000000,
                5.48484848485,
                6.36969696970,
                7.25454545455,
                8.13939393939,
                9.02424242424,
                9.90909090909,
                10.79393939394,
                11.67878787879,
                12.56363636364,
                13.44848484848,
                14.33333333333,
                15.21818181818,
                16.10303030303,
                16.98787878788,
                17.87272727273,
                18.75757575758,
                19.64242424242,
                20.52727272727,
                21.41212121212,
                22.29696969697,
                23.18181818182,
                24.06666666667,
                24.95151515152,
                25.83636363636,
                26.72121212121,
                27.60606060606,
                28.49090909091,
                29.37575757576,
                30.26060606061,
                31.14545454545,
                32.03030303030,
                32.91515151515,
                33.80000000000,
                34.68484848485,
                35.56969696970,
                36.45454545455,
                37.33939393939,
                38.22424242424,
                39.10909090909,
                39.99393939394,
                40.87878787879,
                41.76363636364,
                42.64848484848,
                43.53333333333,
                44.41818181818,
                45.30303030303,
                46.18787878788,
                47.07272727273,
                47.95757575758,
                48.84242424242,
                49.72727272727,
                50.61212121212,
                51.49696969697,
                52.38181818182,
                53.26666666667,
                54.15151515152,
                55.03636363636,
                55.92121212121,
                56.80606060606,
                57.69090909091,
                58.57575757576,
                59.46060606061,
                60.34545454545,
                61.23030303030,
                62.11515151515,
                63.00000000000,
                63.88484848485,
                64.76969696970,
                65.65454545455,
                66.53939393939,
                67.42424242424,
                68.30909090909,
                69.19393939394,
                70.07878787879,
                70.96363636364,
                71.84848484848,
                72.73333333333,
                73.61818181818,
                74.50303030303,
                75.38787878788,
                76.27272727273,
                77.15757575758,
                77.60000000000,
            ]
        )
        y = np.array(
            [
                0.01413960000,
                0.01428800000,
                0.01440820000,
                0.01429620000,
                0.01395610000,
                0.01339210000,
                0.01260860000,
                0.01160990000,
                0.01040090000,
                0.00898622000,
                0.00737101000,
                0.00556041000,
                0.00355992000,
                0.00137533000,
                -0.00098739500,
                -0.00352213000,
                -0.00622255000,
                -0.00908218000,
                -0.01209430000,
                -0.01525220000,
                -0.01854860000,
                -0.02197640000,
                -0.02552810000,
                -0.02919580000,
                -0.03297170000,
                -0.03684750000,
                -0.04081510000,
                -0.04486600000,
                -0.04899160000,
                -0.05318320000,
                -0.05743170000,
                -0.06172820000,
                -0.06606330000,
                -0.07042760000,
                -0.07481160000,
                -0.07920560000,
                -0.08359980000,
                -0.08798440000,
                -0.09234950000,
                -0.09668490000,
                -0.10098000000,
                -0.10522600000,
                -0.10941200000,
                -0.11352700000,
                -0.11756100000,
                -0.12150500000,
                -0.12534700000,
                -0.12907800000,
                -0.13268800000,
                -0.13616600000,
                -0.13950200000,
                -0.14268600000,
                -0.14570900000,
                -0.14856100000,
                -0.15123100000,
                -0.15371200000,
                -0.15599400000,
                -0.15806700000,
                -0.15992400000,
                -0.16155500000,
                -0.16295200000,
                -0.16410600000,
                -0.16501200000,
                -0.16566000000,
                -0.16604500000,
                -0.16616100000,
                -0.16600000000,
                -0.16555700000,
                -0.16482600000,
                -0.16380300000,
                -0.16248300000,
                -0.16086200000,
                -0.15893700000,
                -0.15670500000,
                -0.15416300000,
                -0.15131100000,
                -0.14814600000,
                -0.14466900000,
                -0.14088000000,
                -0.13677800000,
                -0.13236600000,
                -0.12764400000,
                -0.12261700000,
                -0.11728700000,
                -0.11165900000,
                -0.10573800000,
                -0.09953000000,
                -0.09304080000,
                -0.08627770000,
                -0.07924820000,
                -0.07196070000,
                -0.06442470000,
                -0.05665060000,
                -0.04864970000,
                -0.04043370000,
                -0.03201530000,
                -0.02340790000,
                -0.01462530000,
                -0.00568242000,
                0.00340541000,
                0.00799859000,
            ]
        )
        xn = (x - x.min()) / (x.max() - x.min())
        p = np.polynomial.polynomial.polyfit(xn, y, [2, 3, 4, 5, 6])
        p[-1] = -p[:-1].sum() + 1e-10
        y2 = np.polynomial.polynomial.polyval(xn, p)
        pp = util.get_modal_coefficients(x, y2)
        npt.assert_almost_equal(pp, p[2:] / 1e-5)

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

        freq_x, freq_y, freq_z, _, _, _ = util.get_xyz_mode_shapes(r, freqs, dx, dy, dz, xm, ym, zm)
        npt.assert_array_equal(freq_x, np.r_[3, 6, 9, np.zeros(n2 - 3)])
        npt.assert_array_equal(freq_y, np.r_[1, 4, 7, np.zeros(n2 - 3)])
        npt.assert_array_equal(freq_z, np.r_[2, 5, 8, np.zeros(n2 - 3)])

    def testRotateI(self):
        I = np.arange(6) + 1
        th = np.deg2rad(45)
        Irot = util.rotateI(I, th, axis="z")
        npt.assert_almost_equal(Irot, np.array([-2.5, 5.5, 3, -0.5, -0.5 * np.sqrt(2), 5.5 * np.sqrt(2)]))

    def testRotateAlignVectors(self):
        a = np.array([np.cos(0.25 * np.pi), np.sin(0.25 * np.pi), 0])
        b = np.array([0.0, 0.0, 1.0])
        R = util.rotate_align_vectors(a, b)
        b2 = np.matmul(R, a.T).flatten()
        npt.assert_almost_equal(b, b2)

        a = b
        R = util.rotate_align_vectors(a, b)
        npt.assert_almost_equal(R, np.eye(3))
        a = -b
        R = util.rotate_align_vectors(a, b)
        npt.assert_almost_equal(R, np.eye(3))


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
