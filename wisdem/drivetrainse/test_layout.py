import numpy as np
import numpy.testing as npt
import unittest
import wisdem.drivetrainse.layout as lay
import openmdao.api as om

npts = 12

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
        self.inputs['L_hub'] = 0.75
        self.inputs['L_bedplate'] = 5.0
        self.inputs['H_bedplate'] = 4.875
        self.inputs['tilt'] = 5.0
        self.inputs['access_diameter'] = 0.9

        myones = np.ones(npts)
        self.inputs['shaft_diameter'] = 2.3*myones
        self.inputs['nose_diameter'] = 1.33*myones
        self.inputs['shaft_wall_thickness'] = 0.05*myones
        self.inputs['nose_wall_thickness'] = 0.04*myones
        self.inputs['bedplate_wall_thickness'] = 0.06*myones
        self.inputs['D_top'] = 6.5

        self.inputs['rho'] = 7850.
        
        self.discrete_inputs['upwind'] = True


    def testNoTiltUpwind(self):
        self.inputs['tilt'] = 0.0
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['overhang'], 5.0+(2+1+1.5+0.75))
        self.assertAlmostEqual(self.outputs['drive_height'], 4.875)
        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        
        npt.assert_equal(self.outputs['D_nose'], 1.33*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_nose'], 0.04*np.ones(npts+2))
        npt.assert_equal(self.outputs['D_shaft'], 2.3*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_shaft'], 0.05*np.ones(npts+2))

        npt.assert_array_less(self.outputs['x_nose'][:-1], self.outputs['x_nose'][1:])
        npt.assert_array_less(self.outputs['x_shaft'][:-1], self.outputs['x_shaft'][1:])

        npt.assert_equal(self.outputs['z_nose'], 4.875)
        npt.assert_equal(self.outputs['z_shaft'], 4.875)

        self.assertAlmostEqual(self.outputs['x_hub'], -(5.0+2+1+1.5+0.75))
        self.assertAlmostEqual(self.outputs['x_rotor'], -(5.0+2+1+1.5-1.1))
        self.assertAlmostEqual(self.outputs['x_stator'], -(5.0+1.1))
        self.assertAlmostEqual(self.outputs['x_mb1'], -(5.0+1.5+2.0))
        self.assertAlmostEqual(self.outputs['x_mb2'], -(5.0+1.5))

        self.assertAlmostEqual(self.outputs['z_hub'], 4.875)
        self.assertAlmostEqual(self.outputs['z_rotor'], 4.875)
        self.assertAlmostEqual(self.outputs['z_stator'], 4.875)
        self.assertAlmostEqual(self.outputs['z_mb1'], 4.875)
        self.assertAlmostEqual(self.outputs['z_mb2'], 4.875)

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0], -5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 1.33)

        #self.add_output('bedplate_mass', val=0.0, units='kg', desc='Bedplate mass')
        #self.add_output('bedplate_center_of_mass', val=np.zeros(3), units='m', desc='Bedplate center of mass')
        

    def testNoTiltDownwind(self):
        self.inputs['tilt'] = 0.0
        self.discrete_inputs['upwind'] = False
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['overhang'], 5.0+(2+1+1.5+0.75))
        self.assertAlmostEqual(self.outputs['drive_height'], 4.875)
        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        
        npt.assert_equal(self.outputs['D_nose'], 1.33*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_nose'], 0.04*np.ones(npts+2))
        npt.assert_equal(self.outputs['D_shaft'], 2.3*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_shaft'], 0.05*np.ones(npts+2))

        npt.assert_array_less(self.outputs['x_nose'][1:], self.outputs['x_nose'][:-1])
        npt.assert_array_less(self.outputs['x_shaft'][1:], self.outputs['x_shaft'][:-1])

        npt.assert_equal(self.outputs['z_nose'], 4.875)
        npt.assert_equal(self.outputs['z_shaft'], 4.875)

        self.assertAlmostEqual(self.outputs['x_hub'], (5.0+2+1+1.5+0.75))
        self.assertAlmostEqual(self.outputs['x_rotor'], (5.0+2+1+1.5-1.1))
        self.assertAlmostEqual(self.outputs['x_stator'], (5.0+1.1))
        self.assertAlmostEqual(self.outputs['x_mb1'], (5.0+1.5+2.0))
        self.assertAlmostEqual(self.outputs['x_mb2'], (5.0+1.5))

        self.assertAlmostEqual(self.outputs['z_hub'], 4.875)
        self.assertAlmostEqual(self.outputs['z_rotor'], 4.875)
        self.assertAlmostEqual(self.outputs['z_stator'], 4.875)
        self.assertAlmostEqual(self.outputs['z_mb1'], 4.875)
        self.assertAlmostEqual(self.outputs['z_mb2'], 4.875)

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0], 5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 1.33)



    def testUpwind(self):
        ct = np.cos(np.deg2rad(self.inputs['tilt']))
        st = np.sin(np.deg2rad(self.inputs['tilt']))

        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['overhang'], 5.0+(2+1+1.5+0.75)*ct)
        self.assertAlmostEqual(self.outputs['drive_height'], 4.875+(2+1+1.5+0.75)*st)
        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        
        npt.assert_equal(self.outputs['D_nose'], 1.33*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_nose'], 0.04*np.ones(npts+2))
        npt.assert_equal(self.outputs['D_shaft'], 2.3*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_shaft'], 0.05*np.ones(npts+2))

        npt.assert_array_less(self.outputs['x_nose'][:-1], self.outputs['x_nose'][1:])
        npt.assert_array_less(self.outputs['x_shaft'][:-1], self.outputs['x_shaft'][1:])

        npt.assert_array_less(self.outputs['z_nose'][1:], self.outputs['z_nose'][:-1])
        npt.assert_array_less(self.outputs['z_shaft'][1:], self.outputs['z_shaft'][:-1])

        self.assertAlmostEqual(self.outputs['x_hub'], -(5.0+(2+1+1.5+0.75)*ct))
        self.assertAlmostEqual(self.outputs['x_rotor'], -(5.0+(2+1+1.5-1.1)*ct))
        self.assertAlmostEqual(self.outputs['x_stator'], -(5.0+1.1*ct))
        self.assertAlmostEqual(self.outputs['x_mb1'], -(5.0+(1.5+2.0)*ct))
        self.assertAlmostEqual(self.outputs['x_mb2'], -(5.0+1.5*ct))

        self.assertAlmostEqual(self.outputs['z_hub'], (4.875+(2+1+1.5+0.75)*st))
        self.assertAlmostEqual(self.outputs['z_rotor'], (4.875+(2+1+1.5-1.1)*st))
        self.assertAlmostEqual(self.outputs['z_stator'], (4.875+1.1*st))
        self.assertAlmostEqual(self.outputs['z_mb1'], (4.875+(1.5+2.0)*st))
        self.assertAlmostEqual(self.outputs['z_mb2'], (4.875+1.5*st))

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0], -5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0], -5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 1.33)


        
    def testDownwind(self):
        ct = np.cos(np.deg2rad(self.inputs['tilt']))
        st = np.sin(np.deg2rad(self.inputs['tilt']))
        self.discrete_inputs['upwind'] = False
        
        myobj = lay.Layout(n_points=npts)
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        self.assertAlmostEqual(self.outputs['overhang'], 5.0+(2+1+1.5+0.75)*ct)
        self.assertAlmostEqual(self.outputs['drive_height'], 4.875+(2+1+1.5+0.75)*st)
        self.assertAlmostEqual(self.outputs['L_nose'], 3.5)
        self.assertAlmostEqual(self.outputs['L_shaft'], 3.0)
        self.assertAlmostEqual(self.outputs['D_bearing1'], 2.3-0.05-1.33)
        self.assertAlmostEqual(self.outputs['D_bearing2'], 2.3-0.05-1.33)
        npt.assert_equal(self.outputs['constr_access'], 1.33-0.04-0.9)
        
        npt.assert_equal(self.outputs['D_nose'], 1.33*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_nose'], 0.04*np.ones(npts+2))
        npt.assert_equal(self.outputs['D_shaft'], 2.3*np.ones(npts+2))
        npt.assert_equal(self.outputs['t_shaft'], 0.05*np.ones(npts+2))

        npt.assert_array_less(self.outputs['x_nose'][1:], self.outputs['x_nose'][:-1])
        npt.assert_array_less(self.outputs['x_shaft'][1:], self.outputs['x_shaft'][:-1])

        npt.assert_array_less(self.outputs['z_nose'][1:], self.outputs['z_nose'][:-1])
        npt.assert_array_less(self.outputs['z_shaft'][1:], self.outputs['z_shaft'][:-1])

        self.assertAlmostEqual(self.outputs['x_hub'], (5.0+(2+1+1.5+0.75)*ct))
        self.assertAlmostEqual(self.outputs['x_rotor'], (5.0+(2+1+1.5-1.1)*ct))
        self.assertAlmostEqual(self.outputs['x_stator'], (5.0+1.1*ct))
        self.assertAlmostEqual(self.outputs['x_mb1'], (5.0+(1.5+2.0)*ct))
        self.assertAlmostEqual(self.outputs['x_mb2'], (5.0+1.5*ct))

        self.assertAlmostEqual(self.outputs['z_hub'], (4.875+(2+1+1.5+0.75)*st))
        self.assertAlmostEqual(self.outputs['z_rotor'], (4.875+(2+1+1.5-1.1)*st))
        self.assertAlmostEqual(self.outputs['z_stator'], (4.875+1.1*st))
        self.assertAlmostEqual(self.outputs['z_mb1'], (4.875+(1.5+2.0)*st))
        self.assertAlmostEqual(self.outputs['z_mb2'], (4.875+1.5*st))

        self.assertAlmostEqual(self.outputs['x_bedplate'][0], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][0], 5.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][0], 5.0)

        self.assertAlmostEqual(self.outputs['x_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['x_bedplate_inner'][-1],  0.5*6.5)
        self.assertAlmostEqual(self.outputs['x_bedplate_outer'][-1], -0.5*6.5)
        self.assertAlmostEqual(self.outputs['D_bedplate'][-1], 6.5)

        self.assertAlmostEqual(self.outputs['z_bedplate'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][-1], 0.0)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][-1], 0.0)
        
        self.assertAlmostEqual(self.outputs['z_bedplate'][0], 4.875)
        self.assertAlmostEqual(self.outputs['z_bedplate_inner'][0], 4.875-0.5*1.33)
        self.assertAlmostEqual(self.outputs['z_bedplate_outer'][0], 4.875+0.5*1.33)
        self.assertAlmostEqual(self.outputs['D_bedplate'][0], 1.33)



        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLayout))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
