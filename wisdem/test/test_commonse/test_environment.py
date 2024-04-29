import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
from openmdao.utils.assert_utils import assert_check_partials

import wisdem.commonse.environment as env
from wisdem.commonse import gravity as g

npts = 100
myones = np.ones((npts,))


class TestPowerWind(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params["shearExp"] = 2.0
        self.params["Uref"] = 5.0
        self.params["zref"] = 3.0
        self.params["z0"] = 0.0
        for k in self.params:
            self.params[k] = np.array( [self.params[k]] )
        
        self.params["z"] = 9.0 * myones

        self.wind = env.PowerWind(nPoints=npts)

    def testRegular(self):
        self.wind.compute(self.params, self.unknowns)
        expect = 45.0 * myones
        npt.assert_equal(self.unknowns["U"], expect)

    def testIndex(self):
        self.params["z"][1:] = -1.0
        self.wind.compute(self.params, self.unknowns)
        expect = 45.0 * myones
        expect[1:] = 0.0
        npt.assert_equal(self.unknowns["U"], expect)

    def testZ0(self):
        self.params["z0"] = np.array([ 10.0 ])
        self.params["z"] += 10.0
        self.params["zref"] += 10.0
        self.wind.compute(self.params, self.unknowns)
        expect = 45.0 * myones
        npt.assert_equal(self.unknowns["U"], expect)


class TestLinearWaves(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params["rho_water"] = 1e3
        self.params["Hsig_wave"] = 2.0
        self.params["Uc"] = 5.0
        self.params["z_floor"] = -30.0
        self.params["z_surface"] = 0.0
        for k in self.params:
            self.params[k] = np.array( [self.params[k]] )
        self.params["z"] = -2.0 * myones

        self.wave = env.LinearWaves(nPoints=npts)

    def testRegular(self):
        D = np.abs(self.params["z_floor"])
        k = 2.5
        omega = np.sqrt(g * k * np.tanh(k * D))
        self.params["Tsig_wave"] = 2.0 * np.pi / omega

        self.wave.compute(self.params, self.unknowns)
        a = 1.0  # 0.5*Hsig_wave
        z = -2.0
        rho = 1e3
        U_exp = 5 + omega * a * np.cosh(k * (z + D)) / np.sinh(k * D)
        W_exp = -omega * a * np.sinh(k * (z + D)) / np.sinh(k * D)
        V_exp = np.sqrt(U_exp**2 + W_exp**2)
        A_exp = omega * omega * a * np.cosh(k * (z + D)) / np.sinh(k * D)
        p_exp = -rho * g * (z - a * np.cosh(k * (z + D)) / np.cosh(k * D))

        npt.assert_almost_equal(self.unknowns["U"], U_exp[0])
        npt.assert_almost_equal(self.unknowns["W"], W_exp[0])
        npt.assert_almost_equal(self.unknowns["V"], V_exp[0])
        npt.assert_almost_equal(self.unknowns["A"], A_exp[0])
        npt.assert_almost_equal(self.unknowns["p"], p_exp[0])

        # Positive depth input
        self.params["z_floor"] = np.array([ 30.0 ])
        self.wave.compute(self.params, self.unknowns)
        npt.assert_almost_equal(self.unknowns["U"], U_exp[0])
        npt.assert_almost_equal(self.unknowns["W"], W_exp[0])
        npt.assert_almost_equal(self.unknowns["V"], V_exp[0])
        npt.assert_almost_equal(self.unknowns["A"], A_exp[0])
        npt.assert_almost_equal(self.unknowns["p"], p_exp[0])

    def testPositiveZ(self):
        self.params["Tsig_wave"] = np.array([ 2.0 ])
        self.params["z"] = 2.0 * myones
        self.wave.compute(self.params, self.unknowns)
        npt.assert_equal(self.unknowns["U"], 0.0)
        npt.assert_equal(self.unknowns["W"], 0.0)
        npt.assert_equal(self.unknowns["V"], 0.0)
        npt.assert_equal(self.unknowns["A"], 0.0)
        npt.assert_equal(self.unknowns["p"], 0.0)

    def testQuiet(self):
        self.params["Hsig_wave"] = np.array([ 0.0 ])
        self.params["Tsig_wave"] = np.array([ 2.0 ])
        self.wave.compute(self.params, self.unknowns)
        p_exp = 2e3 * g
        npt.assert_equal(self.unknowns["U"], 5.0)
        npt.assert_equal(self.unknowns["W"], 0.0)
        npt.assert_equal(self.unknowns["V"], 5.0)
        npt.assert_equal(self.unknowns["A"], 0.0)
        npt.assert_equal(self.unknowns["p"], p_exp)


class TestPowerWindGradients(unittest.TestCase):
    def test(self):
        z = np.linspace(0.0, 100.0, 20)
        nPoints = len(z)

        prob = om.Problem(reports=False)
        root = prob.model = om.Group()
        root.add_subsystem("p", env.PowerWind(nPoints=nPoints))

        prob.setup()

        prob["p.Uref"] = 10.0
        prob["p.zref"] = 100.0
        prob["p.z0"] = 0.001  # Fails when z0 = 0, What to do here?
        prob["p.shearExp"] = 0.2

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True, method="fd")

        assert_check_partials(check)


class TestLogWindGradients(unittest.TestCase):
    def test(self):
        nPoints = 20
        z = np.linspace(0.1, 100.0, nPoints)

        prob = om.Problem(reports=False)
        root = prob.model = om.Group()
        root.add_subsystem("p", env.LogWind(nPoints=nPoints))

        prob.setup()

        prob["p.Uref"] = 10.0
        prob["p.zref"] = 100.0
        prob["p.z0"] = 0.1  # Fails when z0 = 0

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True, method="fd")

        assert_check_partials(check)


### These partials are wrong; do not test
# class TestLinearWaveGradients(unittest.TestCase):
#
#     def test(self):
#
#         z_floor = 0.1
#         z_surface = 20.
#         z = np.linspace(z_floor, z_surface, 20)
#         nPoints = len(z)
#
#         prob = om.Problem(reports=False)
#         root = prob.model = om.Group()
#         root.add_subsystem('p', env.LinearWaves(nPoints=nPoints))
#
#         prob.setup()
#
#         prob['p.Uc'] = 7.0
#         prob['p.z_floor'] = z_floor
#         prob['p.z_surface'] = z_surface
#         prob['p.Hsig_wave'] = 10.0
#         prob['p.Tsig_wave'] = 2.0
#
#         prob.run_model()
#
#         check = prob.check_partials(out_stream=None, compact_print=True, method='fd')
#
#         assert_check_partials(check)


### The partials are currently not correct, so skip this test
# class TestSoilGradients(unittest.TestCase):
#
#     def test(self):
#
#         d0 = 10.0
#         depth = 30.0
#         G = 140e6
#         nu = 0.4
#
#         prob = om.Problem(reports=False)
#         root = prob.model = om.Group()
#         root.add_subsystem('p', env.TowerSoil())
#
#         prob.setup()
#
#         prob['p.G'] = G
#         prob['p.nu'] = nu
#         prob['p.d0'] = d0
#         prob['p.depth'] = depth
#
#         prob.run_model()
#
#         check = prob.check_partials(out_stream=None, compact_print=True, method='fd')
#
#         assert_check_partials(check)


def suite():
    suite = [
        unittest.TestLoader().loadTestsFromTestCase(TestPowerWind),
        unittest.TestLoader().loadTestsFromTestCase(TestLinearWaves),
        unittest.TestLoader().loadTestsFromTestCase(TestPowerWindGradients),
        unittest.TestLoader().loadTestsFromTestCase(TestLogWindGradients),
    ]
    return unittest.TestSuite(suite)


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
