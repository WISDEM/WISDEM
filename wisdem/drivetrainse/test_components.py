import numpy as np
import numpy.testing as npt
import unittest
import wisdem.drivetrainse.drive_components as dc
import openmdao.api as om

npts = 20

class TestComponents(unittest.TestCase):
    def testBearing(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.MainBearing()

        discrete_inputs['bearing_type'] = 'carb'
        inputs['D_bearing'] = 2.0
        inputs['D_shaft'] = 3.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        npt.assert_equal(outputs['mb_I']/outputs['mb_mass'], 0.125*np.r_[2*(4*1.5**2+3), (4*1.5**2+5)*np.ones(2)])
        self.assertAlmostEqual(outputs['mb_mass'], (1+80/27)*1561.4*3**2.6007)
        self.assertAlmostEqual(outputs['mb_max_defl_ang'], 0.5*np.pi/180)

        # Other valid types
        discrete_inputs['bearing_type'] = 'crb'
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        discrete_inputs['bearing_type'] = 'srb'
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        discrete_inputs['bearing_type'] = 'trb'
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        # Invalid type
        try:
            discrete_inputs['bearing_type'] = 1
            myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        except ValueError:
            self.assertTrue(True)
        # Unknown type
        try:
            discrete_inputs['bearing_type'] = 'trb1'
            myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        except ValueError:
            self.assertTrue(True)


    def testHSS(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.HighSpeedSide()

        discrete_inputs['direct_drive'] = True
        inputs['rotor_diameter']   = 200.0
        inputs['rotor_torque']     = 10e6
        inputs['gear_ratio']       = 1.0
        inputs['D_shaft_end']      = 1.0
        inputs['s_rotor']          = 3.0
        inputs['s_gearbox']        = 0.0
        inputs['hss_input_length'] = 0.0
        inputs['rho']              = 5e3
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['hss_mass'], 12200)
        self.assertEqual(outputs['hss_cm'], 3)
        npt.assert_equal(outputs['hss_I'], 12200*np.r_[0.5, 0.25, 0.25])
        self.assertEqual(outputs['hss_length'], 0)
        self.assertEqual(outputs['hss_diameter'], 0)

        discrete_inputs['direct_drive'] = False
        inputs['gear_ratio']       = 100.0
        inputs['s_gearbox']        = 5.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['hss_mass'], 1.5*2500)
        self.assertEqual(outputs['hss_cm'], 4)
        L = 2500/0.75**2/np.pi/5e3
        self.assertEqual(outputs['hss_length'], L)
        self.assertEqual(outputs['hss_diameter'], 1.5)
        npt.assert_equal(outputs['hss_I'], 1250*np.r_[0.5, 0.25, 0.25] + 2500*np.r_[0.5*0.75**2, (3*0.75**2+L**2)/12*np.ones(2)])

        inputs['hss_input_length'] = 1.5
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['hss_mass'], 1.5*2500)
        self.assertEqual(outputs['hss_cm'], 4)
        L = 1.5
        self.assertEqual(outputs['hss_length'], L)
        self.assertEqual(outputs['hss_diameter'], 1.5)
        npt.assert_equal(outputs['hss_I'], 1250*np.r_[0.5, 0.25, 0.25] + 2500*np.r_[0.5*0.75**2, (3*0.75**2+L**2)/12*np.ones(2)])


    def testGeneratorSimple(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.GeneratorSimple()

        discrete_inputs['direct_drive'] = True
        inputs['rotor_diameter']   = 200.0
        inputs['machine_rating']   = 10e3
        inputs['rotor_torque']     = 10e6
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['R_generator'], 1.5)
        m = 37.68*10e3
        self.assertEqual(outputs['generator_mass'], m)
        npt.assert_equal(outputs['generator_I'], m*np.r_[0.5*1.5**2, (3*1.5**2+(3.6*1.5)**2)/12*np.ones(2)])

        discrete_inputs['direct_drive'] = False
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['R_generator'], 1.5)
        m = np.mean([6.4737, 10.51, 5.34]) * 10e3**0.9223
        self.assertEqual(outputs['generator_mass'], m)
        npt.assert_equal(outputs['generator_I'], m*np.r_[0.5*1.5**2, (3*1.5**2+(3.6*1.5)**2)/12*np.ones(2)])


    def testElectronics(self):
        inputs = {}
        outputs = {}
        myobj = dc.Electronics()

        inputs['rotor_diameter'] = 200.0
        inputs['machine_rating'] = 10e3
        inputs['D_top']          = 5.0
        myobj.compute(inputs, outputs)
        s = 0.015*200
        m = 2.4445*10e3 + 1599
        self.assertEqual(outputs['electronics_mass'], m)
        npt.assert_equal(outputs['electronics_cm'], np.r_[0.0, 2.5+0.5*s, 0.5*s])
        npt.assert_equal(outputs['electronics_I'], (1./6.)*m*s**2)


    def testYaw(self):
        inputs = {}
        outputs = {}
        myobj = dc.YawSystem()

        inputs['rotor_diameter'] = 200.0
        inputs['machine_rating'] = 10e3
        inputs['D_top']          = 5.0
        inputs['rho']            = 5e3
        myobj.compute(inputs, outputs)
        self.assertEqual(outputs['yaw_mass'], 5e3*np.pi*0.1*5**2*0.2 + 190*12)
        npt.assert_equal(outputs['yaw_cm'], 0.0)
        npt.assert_equal(outputs['yaw_I'], 0.0)


    def testMisc(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.MiscNacelleComponents()

        discrete_inputs['upwind'] = False
        inputs['machine_rating'] = 10e3
        inputs['L_bedplate']     = 4.0
        inputs['H_bedplate']     = 4.0
        inputs['D_top']          = 5.0
        inputs['bedplate_mass']  = 5e3
        inputs['bedplate_cm']    = np.array([1.0, 2.0, 3.0])
        inputs['bedplate_I']     = 5e3*np.array([1.0, 2.0, 3.0])
        inputs['R_generator']    = 2.0
        inputs['overhang']       = 10.0
        inputs['s_rotor']        = 8.0
        inputs['s_stator']       = 4.0
        inputs['rho_fiberglass'] = 2e3
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)

        L = 1.1*(10 + 2.5)
        W = 1.1 * 2*2
        H = 1.1 * (2 + 4)
        self.assertEqual(outputs['cover_mass'], 0.04*2e3*2*(L*W+L*H+W*H))
        npt.assert_equal(outputs['cover_cm'], np.array([0.5*(L-5), 0.0, 0.5*H]))
        
        self.assertEqual(outputs['hvac_mass'], 0.08 * 10e3)
        self.assertEqual(outputs['hvac_cm'], 6.0)
        npt.assert_equal(outputs['hvac_I'], 0.08*10e3 * 1.5**2 * np.r_[1.0, 0.5, 0.5])

        self.assertEqual(outputs['mainframe_mass'], 0.125 * 5e3)
        npt.assert_equal(outputs['mainframe_cm'], 0.0)
        npt.assert_equal(outputs['mainframe_I'], 0.125*5e3*np.array([1.0, 2.0, 3.0]))
        
        discrete_inputs['upwind'] = True
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['cover_mass'], 0.04*2e3*2*(L*W+L*H+W*H))
        npt.assert_equal(outputs['cover_cm'], np.array([-0.5*(L-5), 0.0, 0.5*H]))
        
        self.assertEqual(outputs['hvac_mass'], 0.08 * 10e3)
        self.assertEqual(outputs['hvac_cm'], 6.0)
        npt.assert_equal(outputs['hvac_I'], 0.08*10e3 * 1.5**2 * np.r_[1.0, 0.5, 0.5])

        self.assertEqual(outputs['mainframe_mass'], 0.125 * 5e3)
        npt.assert_equal(outputs['mainframe_cm'], 0.0)
        npt.assert_equal(outputs['mainframe_I'], 0.125*5e3*np.array([1.0, 2.0, 3.0]))



    def testMisc(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.MiscNacelleComponents()

        discrete_inputs['upwind'] = False
        inputs['machine_rating'] = 10e3
        inputs['L_bedplate']     = 4.0
        inputs['H_bedplate']     = 4.0
        inputs['D_top']          = 5.0
        inputs['bedplate_mass']  = 5e3
        inputs['bedplate_cm']    = np.array([1.0, 2.0, 3.0])
        inputs['bedplate_I']     = 5e3*np.array([1.0, 2.0, 3.0])
        inputs['R_generator']    = 2.0
        inputs['overhang']       = 10.0
        inputs['s_rotor']        = 8.0
        inputs['s_stator']       = 4.0
        inputs['rho_fiberglass'] = 2e3
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)

        L = 1.1*(10 + 2.5)
        W = 1.1 * 2*2
        H = 1.1 * (2 + 4)
        self.assertEqual(outputs['cover_mass'], 0.04*2e3*2*(L*W+L*H+W*H))
        npt.assert_equal(outputs['cover_cm'], np.array([0.5*(L-5), 0.0, 0.5*H]))
        
        self.assertEqual(outputs['hvac_mass'], 0.08 * 10e3)
        self.assertEqual(outputs['hvac_cm'], 6.0)
        npt.assert_equal(outputs['hvac_I'], 0.08*10e3 * 1.5**2 * np.r_[1.0, 0.5, 0.5])

        self.assertEqual(outputs['mainframe_mass'], 0.125 * 5e3)
        npt.assert_equal(outputs['mainframe_cm'], 0.0)
        npt.assert_equal(outputs['mainframe_I'], 0.125*5e3*np.array([1.0, 2.0, 3.0]))
        
        discrete_inputs['upwind'] = True
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['cover_mass'], 0.04*2e3*2*(L*W+L*H+W*H))
        npt.assert_equal(outputs['cover_cm'], np.array([-0.5*(L-5), 0.0, 0.5*H]))
        
        self.assertEqual(outputs['hvac_mass'], 0.08 * 10e3)
        self.assertEqual(outputs['hvac_cm'], 6.0)
        npt.assert_equal(outputs['hvac_I'], 0.08*10e3 * 1.5**2 * np.r_[1.0, 0.5, 0.5])

        self.assertEqual(outputs['mainframe_mass'], 0.125 * 5e3)
        npt.assert_equal(outputs['mainframe_cm'], 0.0)
        npt.assert_equal(outputs['mainframe_I'], 0.125*5e3*np.array([1.0, 2.0, 3.0]))



    def testNacelle_noTilt(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.NacelleSystemAdder()

        discrete_inputs['upwind'] = True
        discrete_inputs['uptower'] = True
        inputs['tilt'] = 0.0
        components = ['mb1','mb2','lss','hss','gearbox','generator','hvac',
                      'nose','bedplate','mainframe','yaw','cover','electronics']
        cm3 = ['gearbox','electronics','yaw','bedplate','mainframe','cover']
        for k in components:
            inputs[k+'_mass'] = 1e3
            inputs[k+'_I'] = 1e3*np.array([1, 2, 3])
            if k in cm3:
                inputs[k+'_cm'] = np.array([-3.0, 0.0, 0.0])
            else:
                inputs[k+'_cm'] = [3.0]
                
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['other_mass'], 1e3*6)
        self.assertEqual(outputs['nacelle_mass'], 1e3*len(components))
        npt.assert_equal(outputs['nacelle_cm'], np.r_[-3.0, 0.0, 0.0])
        npt.assert_equal(outputs['nacelle_I'], 1e3*len(components)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])
        
        discrete_inputs['upwind'] = False
        for k in cm3:
            inputs[k+'_cm'] *= -1.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['other_mass'], 1e3*6)
        self.assertEqual(outputs['nacelle_mass'], 1e3*len(components))
        npt.assert_equal(outputs['nacelle_cm'], np.r_[3.0, 0.0, 0.0])
        npt.assert_equal(outputs['nacelle_I'], 1e3*len(components)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])

        discrete_inputs['uptower'] = False
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['other_mass'], 1e3*6)
        self.assertEqual(outputs['nacelle_mass'], 1e3*(len(components)-1))
        npt.assert_equal(outputs['nacelle_cm'], np.r_[3.0, 0.0, 0.0])
        npt.assert_equal(outputs['nacelle_I'], 1e3*(len(components)-1)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])



    def testNacelle_withTilt(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        myobj = dc.NacelleSystemAdder()

        discrete_inputs['upwind'] = True
        discrete_inputs['uptower'] = True
        inputs['tilt'] = 5.0
        tr = 5*np.pi/180.
        components = ['mb1','mb2','lss','hss','gearbox','generator','hvac',
                      'nose','bedplate','mainframe','yaw','cover','electronics']
        cm3 = ['gearbox','electronics','yaw','bedplate','mainframe','cover']
        for k in components:
            inputs[k+'_mass'] = 1e3
            inputs[k+'_I'] = 1e3*np.array([1, 2, 3])
            if k in cm3:
                inputs[k+'_cm'] = np.array([-3.0*np.cos(tr), 0.0, 3.0*np.sin(tr)])
            else:
                inputs[k+'_cm'] = [3.0]
                
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['other_mass'], 1e3*6)
        self.assertEqual(outputs['nacelle_mass'], 1e3*len(components))
        npt.assert_almost_equal(outputs['nacelle_cm'], np.r_[-3.0*np.cos(tr), 0.0, 3.0*np.sin(tr)])
        #npt.assert_equal(outputs['nacelle_I'], 1e3*len(components)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])
        
        discrete_inputs['upwind'] = False
        for k in cm3:
            inputs[k+'_cm'][0] *= -1.0
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['other_mass'], 1e3*6)
        self.assertEqual(outputs['nacelle_mass'], 1e3*len(components))
        npt.assert_almost_equal(outputs['nacelle_cm'], np.r_[3.0*np.cos(tr), 0.0, 3.0*np.sin(tr)])
        #npt.assert_equal(outputs['nacelle_I'], 1e3*len(components)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])

        discrete_inputs['uptower'] = False
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        self.assertEqual(outputs['other_mass'], 1e3*6)
        self.assertEqual(outputs['nacelle_mass'], 1e3*(len(components)-1))
        npt.assert_almost_equal(outputs['nacelle_cm'], np.r_[3.0*np.cos(tr), 0.0, 3.0*np.sin(tr)])
        #npt.assert_equal(outputs['nacelle_I'], 1e3*(len(components)-1)*np.r_[1.0, 2.0, 3.0, np.zeros(3)])
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestComponents))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
