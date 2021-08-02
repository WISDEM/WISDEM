import unittest

import numpy as np
import numpy.testing as npt
import wisdem.commonse.utilization_api as util
from wisdem.commonse import gravity as g

myones = np.ones((100,))


class TestUtilization(unittest.TestCase):
    def testTBeam(self):
        h_web = 10.0
        w_flange = 8.0
        t_web = 3.0
        t_flange = 4.0

        area, y_cg, Ixx, Iyy = util._TBeamProperties(h_web, t_web, w_flange, t_flange)
        self.assertEqual(area, 62.0)
        self.assertAlmostEqual(y_cg, 8.6129, 4)
        self.assertAlmostEqual(Iyy, 193.16666, 4)
        self.assertAlmostEqual(Ixx, 1051.37631867699, 4)

        area, y_cg, Ixx, Iyy = util._TBeamProperties(
            h_web * myones, t_web * myones, w_flange * myones, t_flange * myones
        )
        npt.assert_equal(area, 62.0 * myones)
        npt.assert_almost_equal(y_cg, 8.6129 * myones, 1e-4)
        npt.assert_almost_equal(Iyy, 193.16666 * myones, 1e-4)
        npt.assert_almost_equal(Ixx, 1051.37631867699 * myones, 1e-4)

    def testPlasticityRF(self):
        Fy = 4.0

        Fe = 1.0
        Fi = Fe
        self.assertEqual(util._plasticityRF(Fe, Fy), Fi)
        npt.assert_equal(util._plasticityRF(Fe * myones, Fy), Fi * myones)

        Fe = 3.0
        Fr = 4.0 / 3.0
        Fi = Fe * Fr * (1.0 + 3.75 * Fr ** 2) ** (-0.25)
        self.assertEqual(util._plasticityRF(Fe, Fy), Fi)
        npt.assert_equal(util._plasticityRF(Fe * myones, Fy), Fi * myones)

    def testSafetyFactor(self):
        Fy = 100.0
        k = 1.25
        self.assertEqual(util._safety_factor(25.0, Fy), k * 1.2)
        npt.assert_equal(util._safety_factor(25.0 * myones, Fy), k * 1.2 * myones)
        self.assertEqual(util._safety_factor(125.0, Fy), k * 1.0)
        npt.assert_equal(util._safety_factor(125.0 * myones, Fy), k * 1.0 * myones)
        self.assertAlmostEqual(util._safety_factor(80.0, Fy), k * 1.08)
        npt.assert_almost_equal(util._safety_factor(80.0 * myones, Fy), k * 1.08 * myones)

    def testAppliedHoop(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0

        R_od = 0.5 * 600 * in_to_si
        t_wall = 0.75 * in_to_si
        rho = 64.0 * lbperft3_to_si
        z = 60 * ft_to_si
        pressure = rho * g * z
        expect = 1e-3 * 64.0 * 60.0 / 144.0 * ksi_to_si

        self.assertAlmostEqual(pressure, expect, -4)
        expect *= R_od / t_wall
        self.assertAlmostEqual(util._compute_applied_hoop(pressure, R_od, t_wall), expect, -4)
        npt.assert_almost_equal(
            util._compute_applied_hoop(pressure * myones, R_od * myones, t_wall * myones), expect * myones, decimal=-4
        )

    """
    def testAppliedAxial(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        R_od = 0.5*600 * np.ones((4,)) * in_to_si
        t_wall = 0.75 * np.ones((4,)) * in_to_si
        t_web = 5./8. * np.ones((3,)) * in_to_si
        h_web = 14.0 * np.ones((3,)) * in_to_si
        t_flange = 1.0 * np.ones((3,)) * in_to_si
        w_flange = 10.0 * np.ones((3,)) * in_to_si
        h_section = 50.0 * np.ones((3,)) * ft_to_si
        L_stiffener = 5.0 * np.ones((3,)) * ft_to_si
        self.params['water_density'] = 64.0 * lbperft3_to_si
        E = 29e3 * ksi_to_si
        nu = 0.3
        sigma_y = 50 * ksi_to_si
        self.params['bulkhead_nodes'] = [False, False, False, False]
        self.params['wave_height'] = 0.0 # gives only static pressure
        self.params['stack_mass_in'] = 9000 * kip_to_si/g

        self.set_geometry()
        self.myspar.section_mass = np.zeros((3,))

        expect = 9000 * kip_to_si / (2*np.pi*(0.5*R_od[0]-0.5*t_wall[0])*t_wall[0])
        npt.assert_almost_equal(util._compute_applied_axial(self.params, self.myspar.section_mass), expect* np.ones((3,)), decimal=4)
    """

    def testStiffenerFactors(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        R_od = 0.5 * 600 * np.ones((3,)) * in_to_si
        t_wall = 0.75 * np.ones((3,)) * in_to_si
        t_web = 5.0 / 8.0 * np.ones((3,)) * in_to_si
        h_web = 14.0 * np.ones((3,)) * in_to_si
        t_flange = 1.0 * np.ones((3,)) * in_to_si
        w_flange = 10.0 * np.ones((3,)) * in_to_si
        L_stiffener = 5.0 * np.ones((3,)) * ft_to_si
        E = 29e3 * ksi_to_si * np.ones((3,))
        nu = 0.3 * np.ones((3,))

        pressure = 1e-3 * 64.0 * 60.0 / 144.0 * ksi_to_si
        axial = 9000 * kip_to_si / (2 * np.pi * (R_od[0] - 0.5 * t_wall[0]) * t_wall[0])
        self.assertAlmostEqual(axial, 0.5 * 9000 / 299.625 / 0.75 / np.pi * ksi_to_si, -4)
        KthL, KthG = util._compute_stiffener_factors(
            pressure, axial, R_od, t_wall, h_web, t_web, w_flange, t_flange, L_stiffener, E, nu
        )
        npt.assert_almost_equal(KthL, 1.0 * np.ones((3,)), decimal=1)
        npt.assert_almost_equal(KthG, 0.5748 * np.ones((3,)), decimal=4)  # 0.5642 if R_flange accounts for t_wall

    def testStressLimits(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        R_od = 0.5 * 600 * np.ones((3,)) * in_to_si
        t_wall = 0.75 * np.ones((3,)) * in_to_si
        t_web = 5.0 / 8.0 * np.ones((3,)) * in_to_si
        h_web = 14.0 * np.ones((3,)) * in_to_si
        t_flange = 1.0 * np.ones((3,)) * in_to_si
        w_flange = 10.0 * np.ones((3,)) * in_to_si
        L_stiffener = 5.0 * np.ones((3,)) * ft_to_si
        h_section = 50.0 * np.ones((3,)) * ft_to_si
        E = 29e3 * ksi_to_si * np.ones((3,))
        nu = 0.3 * np.ones((3,))
        sigma_y = 50 * ksi_to_si * np.ones((3,))

        KthG = 0.5748
        FxeL, FreL, FxeG, FreG = util._compute_elastic_stress_limits(
            R_od, t_wall, h_section, h_web, t_web, w_flange, t_flange, L_stiffener, E, nu, KthG, loading="radial"
        )
        npt.assert_almost_equal(FxeL, 16.074844135928885 * ksi_to_si * np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FreL, 19.80252150945599 * ksi_to_si * np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FxeG, 37.635953475479639 * ksi_to_si * np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FreG, 93.77314503852581 * ksi_to_si * np.ones((3,)), decimal=1)

        FxcL = util._plasticityRF(FxeL, sigma_y)
        FxcG = util._plasticityRF(FxeG, sigma_y)
        FrcL = util._plasticityRF(FreL, sigma_y)
        FrcG = util._plasticityRF(FreG, sigma_y)
        npt.assert_almost_equal(FxcL, 1.0 * 16.074844135928885 * ksi_to_si * np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FrcL, 1.0 * 19.80252150945599 * ksi_to_si * np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FxcG, 0.799647237534 * 37.635953475479639 * ksi_to_si * np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FrcG, 0.444735273606 * 93.77314503852581 * ksi_to_si * np.ones((3,)), decimal=1)

    def testCheckStresses(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        R_od = 0.5 * 600 * np.ones((3,)) * in_to_si
        t_wall = 0.75 * np.ones((3,)) * in_to_si
        t_web = 5.0 / 8.0 * np.ones((3,)) * in_to_si
        h_web = 14.0 * np.ones((3,)) * in_to_si
        t_flange = 1.0 * np.ones((3,)) * in_to_si
        w_flange = 10.0 * np.ones((3,)) * in_to_si
        h_section = 50.0 * np.ones((3,)) * ft_to_si
        L_stiffener = 5.0 * np.ones((3,)) * ft_to_si
        rho = 64.0 * lbperft3_to_si
        E = 29e3 * ksi_to_si * np.ones((3,))
        nu = 0.3 * np.ones((3,))
        sigma_y = 50 * ksi_to_si * np.ones((3,))

        # Find pressure to give "head" of 60ft- put mid-point of middle section at this depth
        z = 60 * ft_to_si
        P = rho * g * z
        sigma_ax = 9000 * kip_to_si / (2 * np.pi * (R_od[0] - 0.5 * t_wall[0]) * t_wall[0])

        (
            axial_local_unity,
            axial_general_unity,
            external_local_unity,
            external_general_unity,
            _,
            _,
            _,
            _,
        ) = util.shellBuckling_withStiffeners(
            P,
            sigma_ax,
            R_od,
            t_wall,
            h_section,
            h_web,
            t_web,
            w_flange,
            t_flange,
            L_stiffener,
            E,
            nu,
            sigma_y,
            loading="radial",
        )

        # npt.assert_almost_equal(web_compactness, 24.1/22.4 * np.ones((3,)), decimal=3)
        # npt.assert_almost_equal(flange_compactness, 9.03/5.0 * np.ones((3,)), decimal=3)
        npt.assert_almost_equal(axial_local_unity, 1.07, 1)
        npt.assert_almost_equal(axial_general_unity, 0.34, 1)
        npt.assert_almost_equal(external_local_unity, 1.07, 1)
        npt.assert_almost_equal(external_general_unity, 0.59, 1)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestUtilization))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
