import unittest

import numpy as np
import openmdao.api as om
import numpy.testing as npt
import wisdem.commonse.wind_wave_drag as wwd
from openmdao.utils.assert_utils import assert_check_partials

npts = 100
myones = np.ones((npts,))


class TestDrag(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        # variables
        self.params["U"] = 2.0 * myones
        self.params["A"] = 4.0 * myones
        self.params["p"] = 3.0 * myones
        self.params["cm"] = 1.0
        self.params["d"] = 10.0 * myones
        self.params["rho_water"] = 0.5
        self.params["mu_water"] = 1e-3
        self.params["z"] = -100.0 * myones
        self.params["beta_wave"] = 0.0
        self.params["cd_usr"] = -1.0

        self.wave = wwd.CylinderWaveDrag(nPoints=npts)

    def testRegular(self):
        U = 2.0
        A = 4.0
        # cm  = 1.0
        r = 5.0
        rho = 0.5
        # mu  = 1e-3

        # Re = rho*U*2*r/mu
        q = 0.5 * rho * U * U
        cd = 1.11
        area = 2 * r
        D = q * area * cd

        Fi = rho * A * np.pi * r * r
        Fp = Fi + D

        self.wave.compute(self.params, self.unknowns)

        npt.assert_equal(self.unknowns["waveLoads_Px"], Fp)
        npt.assert_equal(self.unknowns["waveLoads_Py"], 0.0)
        npt.assert_equal(self.unknowns["waveLoads_Pz"], 0.0)
        npt.assert_equal(self.unknowns["waveLoads_qdyn"], q)
        npt.assert_equal(self.unknowns["waveLoads_pt"], q + 3.0)
        npt.assert_equal(self.unknowns["waveLoads_z"], -100.0)
        npt.assert_equal(self.unknowns["waveLoads_beta"], 0.0)

    def testCDset(self):
        self.params["cd_usr"] = 2.0
        U = 2.0
        A = 4.0
        r = 5.0

        rho = 0.5
        q = 0.5 * rho * U * U
        area = 2 * r
        D = q * area * 2.0

        Fi = rho * A * np.pi * r * r
        Fp = Fi + D
        self.wave.compute(self.params, self.unknowns)
        npt.assert_equal(self.unknowns["waveLoads_Px"], Fp)

    def test_wave_derivs(self):
        nPoints = 5

        prob = om.Problem()

        comp = wwd.CylinderWaveDrag(nPoints=nPoints)
        prob.model.add_subsystem("comp", comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)

        # Add some arbitrary inputs
        prob.set_val("U", np.arange(nPoints), units="m/s")
        prob.set_val("A", np.ones(nPoints), units="m/s**2")
        prob.set_val("p", np.ones(nPoints) * 0.5, units="N/m**2")
        prob.set_val("z", np.linspace(0.0, 10.0, nPoints), units="m")
        prob.set_val("d", np.ones(nPoints), units="m")
        prob.set_val("beta_wave", 1.2, units="deg")
        prob.set_val("rho_water", 1.0, units="kg/m**3")
        prob.set_val("mu_water", 0.001, units="kg/(m*s)")
        prob.set_val("cm", 10.0)
        prob.set_val("cd_usr", 0.01)

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True, method="fd")

        assert_check_partials(check, rtol=5e-5, atol=1e-1)

    def test_wind_derivs(self):
        nPoints = 5

        prob = om.Problem()

        comp = wwd.CylinderWindDrag(nPoints=nPoints)
        prob.model.add_subsystem("comp", comp, promotes=["*"])

        prob.setup(force_alloc_complex=True)

        # Add some arbitrary inputs
        prob.set_val("U", np.arange(nPoints), units="m/s")
        prob.set_val("z", np.linspace(0.0, 10.0, nPoints), units="m")
        prob.set_val("d", np.ones(nPoints), units="m")
        prob.set_val("beta_wind", 1.2, units="deg")
        prob.set_val("rho_air", 1.0, units="kg/m**3")
        prob.set_val("mu_air", 0.001, units="kg/(m*s)")
        prob.set_val("cd_usr", 0.01)

        prob.run_model()

        check = prob.check_partials(out_stream=None, compact_print=True, method="fd")

        assert_check_partials(check, rtol=5e-5, atol=1e-1)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDrag))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
