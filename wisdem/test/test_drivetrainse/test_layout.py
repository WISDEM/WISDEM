import numpy as np
import numpy.testing as npt
import unittest
import wisdem.drivetrainse.layout as lay
import openmdao.api as om

npts = 20

class TestLayout(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.inputs['L_12'] = 2.0
        self.inputs['L_h1'] = 1.0
        self.inputs['L_2n'] = 1.5
        self.inputs['L_grs'] = 1.1
        self.inputs['L_gsn'] = 1.1
        self.inputs['L_bedplate'] = 5.0
        self.inputs['H_bedplate'] = 4.875
        self.inputs['tilt'] = 5.0
        self.inputs['access_diameter'] = 0.9

        myones = np.ones(5)
        self.inputs['lss_diameter'] = 2.3*myones
        self.inputs['nose_diameter'] = 1.33*myones
        self.inputs['lss_wall_thickness'] = 0.05*myones
        self.inputs['nose_wall_thickness'] = 0.04*myones
        
        self.inputs['bedplate_wall_thickness'] = 0.06*np.ones(npts)
        self.inputs['D_top'] = 6.5

        self.inputs['rho'] = 7850.
        
        self.discrete_inputs['upwind'] = True


    def testNoTiltUpwind(self):
        self.inputs['tilt'] = 0.0
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['L_drive'], 4.5)
        self.assertAlmostEqual(self.outputs['overhang'], self.inputs['L_bedplate']+self.outputs['L_drive'])
        self.assertAlmostEqual(self.outputs['drive_height'], self.inputs['H_bedplate'])
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        self.assertAlmostEqual(self.outputs['constr_L_gsn'], 1.5-1.1)
        self.assertAlmostEqual(self.outputs['constr_L_grs'], 3.0-1.1)
        
        myones = np.ones(6)
        npt.assert_equal(self.outputs['D_nose'], 1.33*myones)
        npt.assert_equal(self.outputs['t_nose'], 0.04*myones)
        npt.assert_equal(self.outputs['D_shaft'], 2.3*myones)
        npt.assert_equal(self.outputs['t_shaft'], 0.05*myones)

        npt.assert_array_less(self.outputs['s_nose'][:-1], self.outputs['s_nose'][1:])
        npt.assert_array_less(self.outputs['s_shaft'][:-1], self.outputs['s_shaft'][1:])

        self.assertAlmostEqual(self.outputs['s_rotor'], 2+1+1.5-1.1)
        self.assertAlmostEqual(self.outputs['s_stator'], 1.1)
        self.assertAlmostEqual(self.outputs['s_mb1'], 1.5+2.0)
        self.assertAlmostEqual(self.outputs['s_mb2'], 1.5)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1], -5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 1.33)


    def testTiltUpwind(self):
        self.inputs['tilt'] = 5.0
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['L_drive'], 4.5)
        self.assertAlmostEqual(self.outputs['overhang'], self.inputs['L_bedplate']+self.outputs['L_drive']*np.cos(np.deg2rad(5)))
        self.assertAlmostEqual(self.outputs['drive_height'], self.inputs['H_bedplate']+self.outputs['L_drive']*np.sin(np.deg2rad(5)))
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        self.assertAlmostEqual(self.outputs['constr_L_gsn'], 1.5-1.1)
        self.assertAlmostEqual(self.outputs['constr_L_grs'], 3.0-1.1)
        
        myones = np.ones(6)
        npt.assert_equal(self.outputs['D_nose'], 1.33*myones)
        npt.assert_equal(self.outputs['t_nose'], 0.04*myones)
        npt.assert_equal(self.outputs['D_shaft'], 2.3*myones)
        npt.assert_equal(self.outputs['t_shaft'], 0.05*myones)

        npt.assert_array_less(self.outputs['s_nose'][:-1], self.outputs['s_nose'][1:])
        npt.assert_array_less(self.outputs['s_shaft'][:-1], self.outputs['s_shaft'][1:])

        self.assertAlmostEqual(self.outputs['s_rotor'], 2+1+1.5-1.1)
        self.assertAlmostEqual(self.outputs['s_stator'], 1.1)
        self.assertAlmostEqual(self.outputs['s_mb1'], 1.5+2.0)
        self.assertAlmostEqual(self.outputs['s_mb2'], 1.5)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1], -5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 1.33)

        
    def testNoTiltDownwind(self):
        self.discrete_inputs['upwind'] = False
        self.inputs['tilt'] = 0.0
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['L_drive'], 4.5)
        self.assertAlmostEqual(self.outputs['overhang'], self.inputs['L_bedplate']+self.outputs['L_drive'])
        self.assertAlmostEqual(self.outputs['drive_height'], self.inputs['H_bedplate'])
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        self.assertAlmostEqual(self.outputs['constr_L_gsn'], 1.5-1.1)
        self.assertAlmostEqual(self.outputs['constr_L_grs'], 3.0-1.1)
        
        myones = np.ones(6)
        npt.assert_equal(self.outputs['D_nose'], 1.33*myones)
        npt.assert_equal(self.outputs['t_nose'], 0.04*myones)
        npt.assert_equal(self.outputs['D_shaft'], 2.3*myones)
        npt.assert_equal(self.outputs['t_shaft'], 0.05*myones)

        npt.assert_array_less(self.outputs['s_nose'][:-1], self.outputs['s_nose'][1:])
        npt.assert_array_less(self.outputs['s_shaft'][:-1], self.outputs['s_shaft'][1:])

        self.assertAlmostEqual(self.outputs['s_rotor'], 2+1+1.5-1.1)
        self.assertAlmostEqual(self.outputs['s_stator'], 1.1)
        self.assertAlmostEqual(self.outputs['s_mb1'], 1.5+2.0)
        self.assertAlmostEqual(self.outputs['s_mb2'], 1.5)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1], 5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 1.33)

        
    def testTiltDownwind(self):
        self.discrete_inputs['upwind'] = False
        self.inputs['tilt'] = 5.0
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['L_drive'], 4.5)
        self.assertAlmostEqual(self.outputs['overhang'], self.inputs['L_bedplate']+self.outputs['L_drive']*np.cos(np.deg2rad(5)))
        self.assertAlmostEqual(self.outputs['drive_height'], self.inputs['H_bedplate']+self.outputs['L_drive']*np.sin(np.deg2rad(5)))
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        self.assertAlmostEqual(self.outputs['constr_L_gsn'], 1.5-1.1)
        self.assertAlmostEqual(self.outputs['constr_L_grs'], 3.0-1.1)
        
        myones = np.ones(6)
        npt.assert_equal(self.outputs['D_nose'], 1.33*myones)
        npt.assert_equal(self.outputs['t_nose'], 0.04*myones)
        npt.assert_equal(self.outputs['D_shaft'], 2.3*myones)
        npt.assert_equal(self.outputs['t_shaft'], 0.05*myones)

        npt.assert_array_less(self.outputs['s_nose'][:-1], self.outputs['s_nose'][1:])
        npt.assert_array_less(self.outputs['s_shaft'][:-1], self.outputs['s_shaft'][1:])

        self.assertAlmostEqual(self.outputs['s_rotor'], 2+1+1.5-1.1)
        self.assertAlmostEqual(self.outputs['s_stator'], 1.1)
        self.assertAlmostEqual(self.outputs['s_mb1'], 1.5+2.0)
        self.assertAlmostEqual(self.outputs['s_mb2'], 1.5)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1], 5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 1.33)


    def testMassValues(self):
        self.discrete_inputs['upwind'] = True
        self.inputs['tilt'] = 0.0
        self.inputs['L_bedplate'] = 5.0
        self.inputs['H_bedplate'] = 5.0
        self.inputs['D_top'] = 3.0
        myones = np.ones(5)
        self.inputs['lss_diameter'] = 2.0*myones
        self.inputs['nose_diameter'] = 3.0*myones
        self.inputs['lss_wall_thickness'] = 0.05*myones
        self.inputs['nose_wall_thickness'] = 0.05*myones
        self.inputs['bedplate_wall_thickness'] = 0.05*np.ones(npts)
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        rho = self.inputs['rho']
        m_bedplate = 5*0.5*np.pi * np.pi*(1.5**2 - (1.5-.05)**2) * rho
        self.assertAlmostEqual(self.outputs['bedplate_mass'], m_bedplate)
        self.assertAlmostEqual(self.outputs['bedplate_cm'][0], np.mean(self.outputs['x_bedplate']), 1)
        self.assertAlmostEqual(self.outputs['bedplate_cm'][1], 0.0)
        self.assertAlmostEqual(self.outputs['bedplate_cm'][2], np.mean(self.outputs['z_bedplate']), 1)

        m_shaft = rho*np.pi*(1**2 - 0.95**2)*self.outputs['L_shaft']
        self.assertAlmostEqual(self.outputs['lss_mass'], m_shaft)
        self.assertAlmostEqual(self.outputs['lss_cm'], 0.5*(self.outputs['s_shaft'][0] + self.outputs['s_shaft'][-1]))
        self.assertAlmostEqual(self.outputs['lss_I'][0], 0.5*m_shaft*(1**2 + 0.95**2))
        self.assertAlmostEqual(self.outputs['lss_I'][1], (1/12)*m_shaft*(3*(1**2 + 0.95**2) + self.outputs['L_shaft']**2))

        m_nose = rho*np.pi*(1.5**2 - 1.45**2)*self.outputs['L_nose']
        self.assertAlmostEqual(self.outputs['nose_mass'], m_nose)
        self.assertAlmostEqual(self.outputs['nose_cm'], 0.5*(self.outputs['s_nose'][0] + self.outputs['s_nose'][-1]))
        self.assertAlmostEqual(self.outputs['nose_I'][0], 0.5*m_nose*(1.5**2 + 1.45**2))
        self.assertAlmostEqual(self.outputs['nose_I'][1], (1/12)*m_nose*(3*(1.5**2 + 1.45**2) + self.outputs['L_nose']**2))
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLayout))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
