import numpy as np
import numpy.testing as npt
import unittest
import wisdem.commonse.WindWaveDrag as wwd

npts = 100
myones = np.ones((npts,))

class TestDrag(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        # variables
        self.params['U'] = 2.0 * myones
        self.params['A'] = 4.0 * myones
        self.params['p'] = 3.0 * myones
        self.params['cm'] = 1.0
        self.params['d'] = 10.0 * myones
        self.params['rho'] = 0.5
        self.params['mu'] = 1e-3
        self.params['z'] = -100.0 * myones
        self.params['beta'] = 0.0
        self.params['cd_usr'] = -1.0
        
        self.wave = wwd.CylinderWaveDrag(nPoints=npts)


    def testRegular(self):
        U   = 2.0
        A   = 4.0
        #cm  = 1.0
        r   = 5.0
        rho = 0.5
        #mu  = 1e-3

        #Re = rho*U*2*r/mu
        q  = 0.5*rho*U*U
        cd = 1.11
        area  = 2*r
        D = q*area*cd

        Fi = rho * A * np.pi * r*r
        Fp = Fi + D

        self.wave.compute(self.params, self.unknowns)

        npt.assert_equal(self.unknowns['waveLoads_Px'], Fp)
        npt.assert_equal(self.unknowns['waveLoads_Py'], 0.0)
        npt.assert_equal(self.unknowns['waveLoads_Pz'], 0.0)
        npt.assert_equal(self.unknowns['waveLoads_qdyn'], q)
        npt.assert_equal(self.unknowns['waveLoads_pt'], q + 3.0)
        npt.assert_equal(self.unknowns['waveLoads_z'], -100.0)
        npt.assert_equal(self.unknowns['waveLoads_beta'], 0.0)
        npt.assert_equal(self.unknowns['waveLoads_d'], 10.0)

    def testCDset(self):
        self.params['cd_usr'] = 2.0
        U   = 2.0
        A   = 4.0
        r   = 5.0

        rho = 0.5
        q  = 0.5*rho*U*U
        area  = 2*r
        D = q*area*2.0

        Fi = rho * A * np.pi * r*r
        Fp = Fi + D
        self.wave.compute(self.params, self.unknowns)
        npt.assert_equal(self.unknowns['waveLoads_Px'], Fp)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDrag))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
