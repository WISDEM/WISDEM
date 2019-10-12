import numpy as np
import numpy.testing as npt
import unittest
import wisdem.commonse.vertical_cylinder as vc
from wisdem.commonse.utilities import nodal2sectional

npts = 100
myones = np.ones((npts,))

class TestDiscretization(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['section_height'] = np.arange(1,5)
        self.params['diameter'] = 5.0 * np.ones(5)
        self.params['wall_thickness'] = 0.05 * np.ones(4)
        self.params['foundation_height'] = 0.0

    def testRefine2(self):
        mydis = vc.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        npt.assert_array_equal(self.unknowns['z_param'], np.array([0.0, 1.0, 3.0, 6.0, 10.0]))
        npt.assert_array_equal(self.unknowns['z_full'], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]))
        npt.assert_array_equal(self.unknowns['d_full'], 5.0)
        npt.assert_array_equal(self.unknowns['t_full'], 0.05)

    def testFoundation(self):
        self.params['foundation_height'] = -30.0
        mydis = vc.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        npt.assert_array_equal(self.unknowns['z_param'], np.array([0.0, 1.0, 3.0, 6.0, 10.0])-30.0)
        npt.assert_array_equal(self.unknowns['z_full'], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0])-30.0)
        npt.assert_array_equal(self.unknowns['d_full'], 5.0)
        npt.assert_array_equal(self.unknowns['t_full'], 0.05)

    def testRefine3(self):
        mydis = vc.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        for k in self.unknowns['z_param']:
            self.assertIn(k, self.unknowns['z_full'])
            
    def testRefineInterp(self):
        self.params['diameter'] = np.array([5.0, 5.0, 6.0, 7.0, 7.0])
        self.params['wall_thickness'] = 1e-2 * np.array([5.0, 5.0, 6.0, 7.0])
        mydis = vc.CylinderDiscretization(nPoints=5, nRefine=2)
        mydis.compute(self.params, self.unknowns)
        npt.assert_array_equal(self.unknowns['z_param'], np.array([0.0, 1.0, 3.0, 6.0, 10.0]))
        npt.assert_array_equal(self.unknowns['z_full'], np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]))
        npt.assert_array_equal(self.unknowns['d_full'], np.array([5.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.0, 7.0]))
        npt.assert_array_equal(self.unknowns['t_full'], 1e-2*np.array([5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0]))

                               
class TestMass(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['d_full'] = 2.0*10.0*myones
        self.params['t_full'] = 0.5*np.ones((npts-1,))
        self.params['z_full'] = np.linspace(0, 50.0, npts)
        self.params['material_density'] = 5.0
        self.params['outfitting_factor'] = 1.5
        self.params['material_cost_rate'] = 1.5
        self.params['labor_cost_rate'] = 1.0
        self.params['painting_cost_rate'] = 10.0
        
        self.cm = vc.CylinderMass(nPoints=npts)

    def testRegular(self):
        # Straight column
        self.cm.compute(self.params, self.unknowns)

        expect = np.pi*(10.**2 - 9.5**2)*5.0*1.5*(50.0/(npts-1))
        m = expect*(npts-1)
        Iax = 0.5*m*(10.**2 + 9.5**2)
        Ix = (1/12.)*m*(3*(10.**2 + 9.5**2) + 50*50) + m*25*25 # parallel axis on last term
        z_avg,_ = nodal2sectional(self.params['z_full'])
        self.assertAlmostEqual(self.unknowns['mass'].sum(), m)
        npt.assert_almost_equal(self.unknowns['mass'], expect)
        npt.assert_almost_equal(self.unknowns['section_center_of_mass'], z_avg)
        self.assertAlmostEqual(self.unknowns['center_of_mass'], 25.0)
        npt.assert_almost_equal(self.unknowns['I_base'], [Ix, Ix, Iax, 0.0, 0.0, 0.0], decimal=5)

        '''
    def testFrustum(self):
        # Frustum shell
        self.params['t_full'] = np.array([0.5, 0.4, 0.3])
        self.params['d_full'] = 2*np.array([10.0, 8.0, 6.0])
        self.wave.compute(self.params, self.unknowns)

        expect = np.pi/3.0*5.0*1.5*np.array([20.0, 30.0])*np.array([9.75*1.4+7.8*1.3, 7.8*1.1+5.85*1.0])
        m = expect*(npts-1)
        self.assertAlmostEqual(self.unknowns['mass'].sum(), m)
        npt.assert_almost_equal(self.unknowns['mass'].sum(), expect)
        '''

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDiscretization))
    suite.addTest(unittest.makeSuite(TestMass))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
