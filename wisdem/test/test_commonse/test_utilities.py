import unittest
import time

import numpy as np
import numpy.testing as npt
from scipy.optimize import curve_fit

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
        # Test exact 6-deg polynomial, randomly generated
        p0 = np.array(
            [0.75174649, 0.06100484, 0.09602379, 0.14829988, 0.2883315, 0.24546124, 0.40763875]
        )  # np.random.random((7,))
        p = p0.copy()
        p[:2] = 0.0
        x = np.linspace(0, 1)
        y = np.polynomial.polynomial.polyval(x, p)

        pp, _ = util.get_modal_coefficients(x, y)
        npt.assert_almost_equal(p[2:] / p[2:].sum(), pp)

        # Test without removing the 0th and 1st modes
        p = p0.copy()
        y = np.polynomial.polynomial.polyval(x, p)
        pp, _ = util.get_modal_coefficients(x, y, base_slope0=False)
        npt.assert_almost_equal(p[2:] / p[2:].sum(), pp, 2)

        # Test more complex and ensure get the same answer as curve fit
        p = np.random.random((10,))
        p[:2] = 0.0
        y = np.polynomial.polynomial.polyval(x, p)

        pp, _ = util.get_modal_coefficients(x, y)
        cc, _ = curve_fit(util.mode_fit, x, y)
        cc /= cc.sum()
        npt.assert_almost_equal(pp, cc, 4)

    def testModalFASTExample_norm(self):
        z = np.array(
            [
                -75.0,
                -60.0,
                -45.0,
                -30.0,
                -26.66666667,
                -23.33333333,
                -20.0,
                -16.66666667,
                -13.33333333,
                -10.0,
                -6.66666667,
                -3.33333333,
                0.0,
                3.33333333,
                6.66666667,
                10.0,
                11.66666667,
                13.33333333,
                15.0,
                19.33333333,
                23.66666667,
                28.0,
                32.33333333,
                36.66666667,
                41.0,
                45.33333333,
                49.66666667,
                54.0,
                58.33333333,
                62.66666667,
                67.0,
                71.33333333,
                75.66666667,
                80.0,
                84.33333333,
                88.66666667,
                93.0,
                97.33333333,
                101.66666667,
                106.0,
                110.33333333,
                114.66666667,
                119.0,
                123.33333333,
                127.66666667,
                132.0,
                136.12866667,
                140.25733333,
                144.386,
            ]
        )
        i_tow = 18
        xdsp = np.array(
            [
                -4.09391274e-08,
                -3.23193671e-07,
                -7.43334720e-07,
                2.58993116e-06,
                4.86973191e-06,
                7.73067054e-06,
                1.11607520e-05,
                1.51725912e-05,
                1.97694064e-05,
                2.49382566e-05,
                3.06944894e-05,
                3.70420200e-05,
                4.39667940e-05,
                5.14863372e-05,
                5.96046438e-05,
                6.83063785e-05,
                7.28777004e-05,
                7.75974518e-05,
                8.24636046e-05,
                9.58082949e-05,
                1.10130403e-04,
                1.25393334e-04,
                1.41608189e-04,
                1.58769858e-04,
                1.76845625e-04,
                1.95856400e-04,
                2.15822157e-04,
                2.36744856e-04,
                2.58658898e-04,
                2.81591474e-04,
                3.05554999e-04,
                3.30595362e-04,
                3.56748091e-04,
                3.84031794e-04,
                4.12507482e-04,
                4.42223204e-04,
                4.73205785e-04,
                5.05522591e-04,
                5.39184836e-04,
                5.74153553e-04,
                6.10472065e-04,
                6.48084752e-04,
                6.86858745e-04,
                7.26764776e-04,
                7.67686964e-04,
                8.09455618e-04,
                8.49806191e-04,
                8.90539841e-04,
                9.31511629e-04,
            ]
        )

        pp, _ = util.get_modal_coefficients(z[i_tow:], xdsp[i_tow:], base_slope0=False)
        sheet_pp = np.flipud([-1.3658, 3.1931, -2.7443, 1.3054, 0.6116])

        xx = np.linspace(0, 1)
        yy = np.polynomial.polynomial.polyval(xx, np.r_[0.0, 0.0, pp])
        yy_sheet = np.polynomial.polynomial.polyval(xx, np.r_[0.0, 0.0, sheet_pp])

        npt.assert_almost_equal(yy, yy_sheet, 2)

    def testGetXYModes(self):
        r = np.linspace(0, 1, 20)
        n = 10
        n2 = int(n / 2)
        dx = dy = dz = np.tile(np.r_[r**2 + 10.0], (n, 1))
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

    @unittest.skip("Did away with rank_and_file")
    def testGetXYModes_rank_and_file(self):
        np.random.seed(314)
        r = np.linspace(0, 1, 20)
        n = 10
        n2 = int(n / 2)
        dx = dy = dz = np.tile(np.r_[r**2 + 10.0], (n, 1))

        # Jitter the values randomly
        dx += np.random.random(dx.shape)
        dy += np.random.random(dy.shape)
        dz += np.random.random(dz.shape)
        freqs = np.arange(n)

        # Assign random mpfs
        xm = np.random.random(n)
        ym = np.random.random(n)
        zm = np.random.random(n)

        #t0 = time.time()
        freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
            r, freqs, dx, dy, dz, xm, ym, zm, 
        )
        #print(time.time() - t0)
        
        # Ensure each mode shape's coefficients sum to 1
        npt.assert_almost_equal(np.sum(mshapes_x, axis=1), np.ones(n2))
        npt.assert_almost_equal(np.sum(mshapes_y, axis=1), np.ones(n2))
        npt.assert_almost_equal(np.sum(mshapes_z, axis=1), np.ones(n2))

        # Assert the order for each mode for this seeded random setup
        npt.assert_array_equal(freq_x, np.r_[3, 7, 8, 4, 9])
        npt.assert_array_equal(freq_y, np.r_[3, 2, 8, 7, 9])
        npt.assert_array_equal(freq_z, np.r_[3, 9, 4, 8, 5])

        mshapes_x_truth = np.array(
            [
                [-22.92022398, 129.46333573, -246.4030288, 192.11436963, -51.25445258],
                [1336.9832145, -8154.95913311, 17975.66811644, -17015.65469796, 5858.96250013],
                [21.878196, -104.20313009, 208.34202546, -190.69851021, 65.68141883],
                [-71.51627174, 505.27856257, -1249.52043721, 1298.12252412, -481.36437775],
                [-54.10105559, 288.84391942, -577.69093167, 512.63427927, -168.68621143],
            ]
        )
        mshapes_y_truth = np.array(
            [
                [-22.92022398, 129.46333573, -246.4030288, 192.11436963, -51.25445258],
                [39.6102757, -243.30461819, 560.37887651, -551.01420603, 195.32967201],
                [21.878196, -104.20313009, 208.34202546, -190.69851021, 65.68141883],
                [1336.9832145, -8154.95913311, 17975.66811644, -17015.65469796, 5858.96250013],
                [-54.10105559, 288.84391942, -577.69093167, 512.63427927, -168.68621143],
            ]
        )
        mshapes_z_truth = np.array(
            [
                [-22.92022398, 129.46333573, -246.4030288, 192.11436963, -51.25445258],
                [-54.10105559, 288.84391942, -577.69093167, 512.63427927, -168.68621143],
                [-71.51627174, 505.27856257, -1249.52043721, 1298.12252412, -481.36437775],
                [21.878196, -104.20313009, 208.34202546, -190.69851021, 65.68141883],
                [11.53705709, -85.1199343, 217.8419065, -228.79184901, 85.53281973],
            ]
        )
        npt.assert_almost_equal(mshapes_x, mshapes_x_truth)
        npt.assert_almost_equal(mshapes_y, mshapes_y_truth)
        npt.assert_almost_equal(mshapes_z, mshapes_z_truth)

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


if __name__ == "__main__":
    unittest.main()
