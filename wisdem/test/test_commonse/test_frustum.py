import unittest

import numpy as np
import numpy.testing as npt

import wisdem.commonse.frustum as f
from wisdem.commonse import eps

myones = np.ones((100,))
# For frustum
rb = 4.0
rt = 2.0
t = 0.1
h = 3.0





class TestFrustum(unittest.TestCase):
    def testFrustumVol(self):
        V = np.pi / 3 * h * (rb**2 + rt**2 + rb * rt)

        # Test volume- scalar and vector inputs
        self.assertEqual(f.frustumVol(rb, rt, h, False), V)
        self.assertEqual(f.frustumVol(2 * rb, 2 * rt, h, True), V)
        npt.assert_equal(f.frustumVol(rb * myones, rt * myones, h * myones, False), V * myones)
        npt.assert_equal(f.frustumVol(2 * rb * myones, 2 * rt * myones, h * myones, True), V * myones)

    def testFrustumCG_solid(self):
        cg_solid = h / 4 * (rb**2 + 3 * rt**2 + 2 * rb * rt) / (rb**2 + rt**2 + rb * rt)

        # Test cg of solid- scalar and vector inputs
        self.assertEqual(f.frustumCG(rb, rt, h, False), cg_solid)
        self.assertEqual(f.frustumCG(2 * rb, 2 * rt, h, True), cg_solid)
        npt.assert_equal(f.frustumCG(rb * myones, rt * myones, h * myones, False), cg_solid * myones)
        npt.assert_equal(f.frustumCG(2 * rb * myones, 2 * rt * myones, h * myones, True), cg_solid * myones)

    def testFrustum_shell(self):
        # In limit of thickness approaching radius, should recover regular formulas
        self.assertEqual(f.frustumShellVol(rb, rb, rb, h, False), f.frustumVol(rb, rb, h, False))
        self.assertEqual(f.frustumShellVol(2 * rt, 2 * rt, rt, h, True), f.frustumVol(rt, rt, h, False))

        self.assertEqual(f.frustumShellCG(rb, rb, rb, h, False), f.frustumCG(rb, rb, h, False))
        self.assertEqual(f.frustumShellCG(2 * rt, 2 * rt, rt, h, True), f.frustumCG(rt, rt, h, False))

        self.assertEqual(f.frustumShellIzz(rb, rb, rb, h, False), f.frustumIzz(rb, rb, h, False))
        self.assertEqual(f.frustumShellIzz(2 * rt, 2 * rt, rt, h, True), f.frustumIzz(rt, rt, h, False))

        self.assertAlmostEqual(f.frustumShellIxx(rb, rb, rb - eps, h, False), f.frustumIxx(rb, rb, h, False))
        self.assertAlmostEqual(f.frustumShellIxx(2 * rt, 2 * rt, rt - eps, h, True), f.frustumIxx(rt, rt, h, False))

class TestRectangularFrustum(unittest.TestCase):
    def testRectangularFrustumVol(self):
        V = 14.0
        # For rectangular frustum
        ab = 4.0
        bb = 2.0
        at = 2.0
        bt = 1.0

        # Test volume- scalar and vector inputs
        self.assertEqual(f.RectangularFrustumVol(ab, bb, at, bt, h), V)
        npt.assert_equal(f.RectangularFrustumVol(ab * myones, bb * myones, at * myones, bt * myones, h * myones), V * myones)

    def testRectangualrFrustumCG_solid(self):
        CG = 1.5
        # For rectangular frustum
        ab = 4.0
        bb = 2.0
        at = 4.0
        bt = 2.0

        # Test volume- scalar and vector inputs
        self.assertEqual(f.RectangularFrustumCG(ab, bb, at, bt, h), CG)
        npt.assert_equal(f.RectangularFrustumCG(ab * myones, bb * myones, at * myones, bt * myones, h * myones), CG * myones)

    def testRectangualrFrustumMOI_solid(self):
    
        # For rectangular cuboid
        ab = 4.0
        bb = 2.0
        at = 4.0
        bt = 2.0
        Izz = 1/12 * (ab*bb*h) * (ab**2 + bb**2)
        Ixx = 1/12 * (ab*bb*h) * (bb**2 + h**2) + (ab*bb*h)*(h/2)**2
        Iyy = 1/12 * (ab*bb*h) * (ab**2 + h**2) + (ab*bb*h)*(h/2)**2

        # Test volume- scalar and vector inputs
        self.assertEqual(f.RectangularFrustumIzz(ab, bb, at, bt, h), Izz)
        npt.assert_equal(f.RectangularFrustumIzz(ab * myones, bb * myones, at * myones, bt * myones, h * myones), Izz * myones)
        self.assertEqual(f.RectangularFrustumIxx(ab, bb, at, bt, h), Ixx)
        npt.assert_equal(f.RectangularFrustumIxx(ab * myones, bb * myones, at * myones, bt * myones, h * myones), Ixx * myones)
        self.assertEqual(f.RectangularFrustumIyy(ab, bb, at, bt, h), Iyy)
        npt.assert_equal(f.RectangularFrustumIyy(ab * myones, bb * myones, at * myones, bt * myones, h * myones), Iyy * myones)

    def testRectangularFrustum_shell(self):
        ab = 4.0
        bb = 2.0
        at = 4.0
        bt = 2.0
        t = 0.0

        # Zero thickness should give zero volume
        self.assertEqual(f.RectangularFrustumShellVol(ab, bb, at, bt, t, h), 0.0)
        npt.assert_equal(f.RectangularFrustumShellVol(ab * myones, bb * myones, at * myones, bt * myones, t * myones, h * myones), 0.0 * myones)
        # Zero thickness should give zero Izz
        self.assertEqual(f.RectangularFrustumShellIzz(ab, bb, at, bt, t, h), 0.0)
        npt.assert_equal(f.RectangularFrustumShellIzz(ab * myones, bb * myones, at * myones, bt * myones, t * myones, h * myones), 0.0 * myones)
        # Same base and top should give the same result as the rectangular cuboid
        t = 0.02
        ab_i = ab - 2*t
        bb_i = bb - 2*t
        # Because it's rectangular cuboid, only use base parameters
        # at_i = at - 2*t
        # bt_i = bt - 2*t
        Izz = 1.0/12.0 * (ab*bb*h) * (ab**2 + bb**2) 
        Izz -= 1.0/12.0 * (ab_i*bb_i*h) * (ab_i**2 + bb_i**2)
        Ixx = 1/12 * (ab*bb*h) * (bb**2 + h**2) + (ab*bb*h)*(h/2)**2
        Ixx -= 1/12 * (ab_i*bb_i*h) * (bb_i**2 + h**2) + (ab_i*bb_i*h)*(h/2)**2
        Iyy = 1/12 * (ab*bb*h) * (ab**2 + h**2) + (ab*bb*h)*(h/2)**2
        Iyy -= 1/12 * (ab_i*bb_i*h) * (ab_i**2 + h**2) + (ab_i*bb_i*h)*(h/2)**2
        # Same base and top should give the same result as the rectangular cuboid. The difference is only ~1e-15, use almostEqual
        self.assertAlmostEqual(f.RectangularFrustumShellIzz(ab, bb, at, bt, t, h), Izz)
        npt.assert_almost_equal(f.RectangularFrustumShellIzz(ab * myones, bb * myones, at * myones, bt * myones, t * myones, h * myones), Izz * myones)
        self.assertAlmostEqual(f.RectangularFrustumShellIxx(ab, bb, at, bt, t, h), Ixx)
        npt.assert_almost_equal(f.RectangularFrustumShellIxx(ab * myones, bb * myones, at * myones, bt * myones, t * myones, h * myones), Ixx * myones)
        self.assertAlmostEqual(f.RectangularFrustumShellIyy(ab, bb, at, bt, t, h), Iyy)
        npt.assert_almost_equal(f.RectangularFrustumShellIyy(ab * myones, bb * myones, at * myones, bt * myones, t * myones, h * myones), Iyy * myones)
        # Same base and top should give the mid height
        self.assertAlmostEqual(f.RectangularFrustumShellCG(ab, bb, at, bt, t, h), h/2)
        npt.assert_almost_equal(f.RectangularFrustumShellCG(ab * myones, bb * myones, at * myones, bt * myones, t * myones, h * myones), h/2 * myones)





if __name__ == "__main__":
    unittest.main()
