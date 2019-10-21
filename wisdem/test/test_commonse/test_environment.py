import numpy as np
import numpy.testing as npt
import unittest
import wisdem.commonse.environment as env
from wisdem.commonse import gravity as g

npts = 100
myones = np.ones((npts,))

class TestPowerWind(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['shearExp'] = 2.0
        self.params['Uref'] = 5.0
        self.params['zref'] = 3.0
        self.params['z0'] = 0.0
        self.params['z'] = 9.0 * myones
        
        self.wind = env.PowerWind(nPoints=npts)

    def testRegular(self):
        self.wind.compute(self.params, self.unknowns)
        expect = 45.0*myones
        npt.assert_equal(self.unknowns['U'], expect)

    def testIndex(self):
        self.params['z'][1:] = -1.0
        self.wind.compute(self.params, self.unknowns)
        expect = 45.0*myones
        expect[1:] = 0.0
        npt.assert_equal(self.unknowns['U'], expect)

    def testZ0(self):
        self.params['z0'] = 10.0
        self.params['z'] += 10.0
        self.params['zref'] += 10.0
        self.wind.compute(self.params, self.unknowns)
        expect = 45.0*myones
        npt.assert_equal(self.unknowns['U'], expect)


class TestLinearWaves(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['rho'] = 1e3
        self.params['hmax'] = 2.0
        self.params['Uc'] = 5.0
        self.params['z_floor'] = -30.0
        self.params['z_surface'] = 0.0
        self.params['z'] = -2.0 * myones

        self.wave = env.LinearWaves(nPoints=npts)
        
    def testRegular(self):
        D = np.abs(self.params['z_floor'])
        k = 2.5
        omega = np.sqrt(g*k*np.tanh(k*D))
        self.params['T'] = 2.0 * np.pi / omega
        
        self.wave.compute(self.params, self.unknowns)
        a = 1.0 #0.5*hmax
        z = -2.0
        rho = 1e3
        U_exp = 5 + omega*a*np.cosh(k*(z+D))/np.sinh(k*D)
        W_exp = -omega*a*np.sinh(k*(z+D))/np.sinh(k*D)
        V_exp = np.sqrt(U_exp**2 + W_exp**2)
        A_exp = omega*omega*a*np.cosh(k*(z+D))/np.sinh(k*D)
        p_exp = -rho*g*(z - a*np.cosh(k*(z+D))/np.cosh(k*D))
        
        npt.assert_almost_equal(self.unknowns['U'], U_exp)
        npt.assert_almost_equal(self.unknowns['W'], W_exp)
        npt.assert_almost_equal(self.unknowns['V'], V_exp)
        npt.assert_almost_equal(self.unknowns['A'], A_exp)
        npt.assert_almost_equal(self.unknowns['p'], p_exp)

        # Positive depth input
        self.params['z_floor'] = 30.0
        self.wave.compute(self.params, self.unknowns)
        npt.assert_almost_equal(self.unknowns['U'], U_exp)
        npt.assert_almost_equal(self.unknowns['W'], W_exp)
        npt.assert_almost_equal(self.unknowns['V'], V_exp)
        npt.assert_almost_equal(self.unknowns['A'], A_exp)
        npt.assert_almost_equal(self.unknowns['p'], p_exp)

    def testPositiveZ(self):
        self.params['T'] = 2.0 
        self.params['z'] = 2.0 * myones
        self.wave.compute(self.params, self.unknowns)
        npt.assert_equal(self.unknowns['U'], 0.0)
        npt.assert_equal(self.unknowns['W'], 0.0)
        npt.assert_equal(self.unknowns['V'], 0.0)
        npt.assert_equal(self.unknowns['A'], 0.0)
        npt.assert_equal(self.unknowns['p'], 0.0)

        
    def testQuiet(self):
        self.params['hmax'] = 0.0 
        self.params['T'] = 2.0
        self.wave.compute(self.params, self.unknowns)
        p_exp = 2e3*g
        npt.assert_equal(self.unknowns['U'], 5.0)
        npt.assert_equal(self.unknowns['W'], 0.0)
        npt.assert_equal(self.unknowns['V'], 5.0)
        npt.assert_equal(self.unknowns['A'], 0.0)
        npt.assert_equal(self.unknowns['p'], p_exp)
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPowerWind))
    suite.addTest(unittest.makeSuite(TestLinearWaves))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
