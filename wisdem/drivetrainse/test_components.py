import numpy as np
import numpy.testing as npt
import unittest
import wisdem.drivetrainse.drive_components as dc
import openmdao.api as om

npts = 20

class TestBearing(unittest.TestCase):
    def testAll(self):
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



class TestHSS(unittest.TestCase):
    def testAll(self):
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



class TestGeneratorSimple(unittest.TestCase):
    def testAll(self):
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
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBearing))
    suite.addTest(unittest.makeSuite(TestHSS))
    suite.addTest(unittest.makeSuite(TestGeneratorSimple))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
