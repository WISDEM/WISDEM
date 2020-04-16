import numpy as np
import numpy.testing as npt
import unittest
import wisdem.rotorse.rotor_loads_defl_strains as rlds
import openmdao.api as om
import copy
import time
import os
ARCHIVE  = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'regulation.npz'
        
class TestRLDS(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}
        
    def testGust(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        myobj = rlds.GustETM()
        
        inputs['V_mean'] = 10.0
        inputs['V_hub']  = 15.0
        inputs['std']    = 2.5
        discrete_inputs['turbulence_class'] = 'A'
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        sigma = 0.32 * (0.072*8.0*3.5 + 10.0)
        expect = 15.0 + 2.5*sigma
        self.assertEqual(outputs['V_gust'], expect)

        # Test lower case
        discrete_inputs['turbulence_class'] = 'c'
        myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        sigma = 0.24 * (0.072*8.0*3.5 + 10.0)
        expect = 15.0 + 2.5*sigma
        self.assertEqual(outputs['V_gust'], expect)

        # Test bad class
        discrete_inputs['turbulence_class'] = 'd'
        try:
            myobj.compute(inputs, outputs, discrete_inputs, discrete_outputs)
        except ValueError:
            self.assertTrue(True)

            
    def testBladeCurvature(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        npts = 101
        myzero = np.zeros(npts)
        myone  = np.ones(npts)
        options = {}
        options['blade'] = {}
        options['blade']['n_span'] = npts

        myobj = rlds.BladeCurvature(analysis_options=options)

        # Straight blade: Z is 'r'
        inputs['r'] = np.linspace(0, 100, npts)
        inputs['precurve'] = myzero
        inputs['presweep'] = myzero
        inputs['precone' ] = 0.0
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs['3d_curv'], myzero)
        npt.assert_equal(outputs['x_az'], myzero)
        npt.assert_equal(outputs['y_az'], myzero)
        npt.assert_equal(outputs['z_az'], inputs['r'])

        # Some coning: Z is 'r'
        inputs['precone' ] = 3.0
        myobj.compute(inputs, outputs)
        npt.assert_equal(outputs['3d_curv'], 3*myone)
        npt.assert_equal(outputs['x_az'], myzero)
        npt.assert_equal(outputs['y_az'], myzero)
        npt.assert_equal(outputs['z_az'], inputs['r'])

        # Some curve: X is 'flap'
        inputs['precurve'] = np.linspace(0, 1, npts)
        inputs['precone' ] = 0.0
        myobj.compute(inputs, outputs)
        cone = -np.rad2deg(np.arctan(inputs['precurve']/(inputs['r']+1e-20)))
        cone[0] = cone[1]
        npt.assert_almost_equal(outputs['3d_curv'], cone)
        npt.assert_equal(outputs['x_az'], inputs['precurve'])
        npt.assert_equal(outputs['y_az'], myzero)
        npt.assert_equal(outputs['z_az'], inputs['r'])

        # Some curve: Y is 'edge'
        inputs['precurve'] = myzero
        inputs['presweep'] = np.linspace(0, 1, npts)
        myobj.compute(inputs, outputs)
        npt.assert_almost_equal(outputs['3d_curv'], myzero)
        npt.assert_equal(outputs['x_az'], myzero)
        npt.assert_equal(outputs['y_az'], inputs['presweep'])
        npt.assert_equal(outputs['z_az'], inputs['r'])

        # Some curve and sweep
        inputs['precurve'] = np.linspace(0, 2, npts)
        inputs['presweep'] = np.linspace(0, 1, npts)
        inputs['precone' ] = 0.0
        myobj.compute(inputs, outputs)
        cone = -np.rad2deg(np.arctan(inputs['precurve']/(inputs['r']+1e-20)))
        cone[0] = cone[1]
        npt.assert_almost_equal(outputs['3d_curv'], cone)
        npt.assert_equal(outputs['x_az'], inputs['precurve'])
        npt.assert_equal(outputs['y_az'], inputs['presweep'])
        npt.assert_equal(outputs['z_az'], inputs['r'])

            
    def testTotalLoads(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}
        npts = 101
        myzero = np.zeros(npts)
        myone  = np.ones(npts)
        options = {}
        options['blade'] = {}
        options['blade']['n_span'] = npts

        myobj = rlds.TotalLoads(analysis_options=options)

        inputs['r'] = np.linspace(0, 100, npts)
        inputs['aeroloads_Px'] = 10.0 * myone
        inputs['aeroloads_Py'] = 5.0 * myone
        inputs['aeroloads_Pz'] = 1.0 * myone
        
        self.add_input('aeroloads_Omega',   val=0.0,                units='rpm',    desc='rotor rotation speed')
        self.add_input('aeroloads_pitch',   val=0.0,                units='deg',    desc='pitch angle')
        self.add_input('aeroloads_azimuth', val=0.0,                units='deg',    desc='azimuthal angle')
        self.add_input('theta',             val=np.zeros(n_span),   units='deg',    desc='structural twist')
        self.add_input('tilt',              val=0.0,                units='deg',    desc='tilt angle')
        self.add_input('3d_curv',           val=np.zeros(n_span),   units='deg',    desc='total cone angle from precone and curvature')
        self.add_input('z_az',              val=np.zeros(n_span),   units='m',      desc='location of blade in azimuth z-coordinate system')
        self.add_input('rhoA',              val=np.zeros(n_span),   units='kg/m',   desc='mass per unit length')
        self.add_input('dynamicFactor',     val=1.0,                                desc='a dynamic amplification factor to adjust the static deflection calculation')

        # Outputs
        self.add_output('Px_af', val=np.zeros(n_span), desc='total distributed loads in airfoil x-direction')
        self.add_output('Py_af', val=np.zeros(n_span), desc='total distributed loads in airfoil y-direction')
        self.add_output('Pz_af', val=np.zeros(n_span), desc='total distributed loads in airfoil z-direction')
        
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRLDS))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
