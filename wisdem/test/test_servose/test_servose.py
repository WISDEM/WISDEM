import numpy as np
import numpy.testing as npt
import unittest
import wisdem.servose.servose as serv
import openmdao.api as om
import copy
import time
import os
ARCHIVE  = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'regulation.npz'

        
class TestServo(unittest.TestCase):
        
    def testGust(self):
        inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        myobj = serv.GustETM()
        
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
        
    def testRegulationTrajectory(self):
        prob = om.Problem()
        
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        
        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(ARCHIVE)
        ivc.add_output('airfoils_aoa', npzfile['aoa'], units='deg')
        ivc.add_output('airfoils_Re', npzfile['Re'])
        ivc.add_output('airfoils_cl', np.moveaxis(npzfile['cl'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('airfoils_cd', np.moveaxis(npzfile['cd'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('airfoils_cm', np.moveaxis(npzfile['cm'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('r', npzfile['r'], units='m')
        ivc.add_output('chord', npzfile['chord'], units='m')
        ivc.add_output('theta', npzfile['theta'], units='deg')

        n_span = npzfile['r'].size
        n_aoa = npzfile['aoa'].size
        n_Re = npzfile['Re'].size
        n_pc = 22
        
        # parameters
        ivc.add_output('v_min', 4., units='m/s')
        ivc.add_output('v_max', 25., units='m/s')
        ivc.add_output('rated_power', 5e6, units='W')
        ivc.add_output('omega_min', 0.0, units='rpm')
        ivc.add_output('omega_max', 100.0, units='rpm')
        ivc.add_output('control_maxTS', 90., units='m/s')
        ivc.add_output('tsr_operational', 10.)
        ivc.add_output('control_pitch', 0.0, units='deg')
        ivc.add_output('gearbox_efficiency', 0.975)
        ivc.add_output('generator_efficiency', 0.975)
        ivc.add_discrete_output('drivetrainType', 'GEARED')
        
        ivc.add_output('Rhub', 1., units='m')
        ivc.add_output('Rtip', 70., units='m')
        ivc.add_output('hub_height', 100., units='m')
        ivc.add_output('precone', 0., units='deg')
        ivc.add_output('tilt', 0., units='deg')
        ivc.add_output('yaw', 0., units='deg')
        ivc.add_output('precurve', np.zeros(n_span), units='m')
        ivc.add_output('precurveTip', 0., units='m')
        ivc.add_output('presweep', np.zeros(n_span), units='m')
        ivc.add_output('presweepTip', 0., units='m')
        
        ivc.add_output('rho', 1.225, units='kg/m**3')
        ivc.add_output('mu', 1.81206e-5, units='kg/(m*s)')
        ivc.add_output('shearExp', 0.25)
        ivc.add_discrete_output('nBlades', 3)
        ivc.add_discrete_output('nSector', 4)
        ivc.add_discrete_output('tiploss', True)
        ivc.add_discrete_output('hubloss', True)
        ivc.add_discrete_output('wakerotation', True)
        ivc.add_discrete_output('usecd', True)

        analysis_options = {}
        analysis_options['blade'] = {}
        analysis_options['blade']['n_span'] = n_span
        analysis_options['blade']['n_aoa'] = n_aoa
        analysis_options['blade']['n_Re'] = n_Re
        analysis_options['blade']['n_tab'] = 1
        analysis_options['servose'] = {}
        analysis_options['servose']['regulation_reg_III'] = True
        analysis_options['servose']['n_pc'] = n_pc
        analysis_options['servose']['n_pc_spline'] = n_pc

        n_span, n_aoa, n_Re, n_tab = np.moveaxis(npzfile['cl'][:,:,:,np.newaxis], 0, 1).shape
        analysis_options['airfoils'] = {}
        analysis_options['airfoils']['n_aoa'] = n_aoa
        analysis_options['airfoils']['n_Re'] = n_Re
        analysis_options['airfoils']['n_tab'] = n_tab
        
        prob.model.add_subsystem('powercurve', serv.RegulatedPowerCurve(analysis_options=analysis_options), promotes=['*'])
        
        prob.setup()

        # All reg 2: no maxTS, no max rpm, no power limit
        prob['omega_max'] = 1e3
        prob['control_maxTS'] = 1e5
        prob['rated_power'] = 1e16
        prob.run_model()
        
        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi/4.,np.pi/2.,n_pc + 1)))))
        grid1 = (grid0 - grid0[0])/(grid0[-1]-grid0[0])
        V_expect0  = grid1 * (prob['v_max'] - prob['v_min']) + prob['v_min']
        V_spline = np.linspace(prob['v_min'], prob['v_max'], n_pc)
        irated = 12
        
        V_expect1 = V_expect0.copy()
        #V_expect1[irated] = prob['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        npt.assert_equal(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_tsr)
        npt.assert_equal(prob['pitch'], np.zeros( V_expect0.shape ) )
        npt.assert_array_almost_equal(prob['Cp'], prob['Cp_aero']*0.975*0.975)
        npt.assert_allclose(prob['Cp'], prob['Cp'][0])
        npt.assert_allclose(prob['Cp_aero'], prob['Cp_aero'][0])
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp, myCp[0])
        self.assertGreater(myCp[0], 0.4)
        self.assertGreater(0.5, myCp[0])
        npt.assert_allclose(myCp, prob['Cp'])
        npt.assert_array_less(prob['P'][:-1], prob['P'][1:])
        npt.assert_array_less(prob['Q'][:-1], prob['Q'][1:])
        npt.assert_array_less(prob['T'][:-1], prob['T'][1:])
        self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob['rated_Omega'][0], Omega_tsr[-1])
        self.assertEqual(prob['rated_pitch'], 0.0)
        
        # Test no maxTS, max rpm, no power limit
        prob['omega_max'] = 15.0
        prob['control_maxTS'] = 1e5
        prob['rated_power'] = 1e16
        prob.run_model()
        V_expect1 = V_expect0.copy()
        #V_expect1[irated] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, 15.0)
        npt.assert_allclose(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_expect)
        npt.assert_equal(prob['pitch'][:irated], 0.0 )
        npt.assert_array_less(0.0, np.abs(prob['pitch'][(irated+1):]))
        npt.assert_array_almost_equal(prob['Cp'], prob['Cp_aero']*0.975*0.975)
        npt.assert_array_less(prob['P'][:-1], prob['P'][1:])
        npt.assert_array_less(prob['Q'][:-1], prob['Q'][1:])
        npt.assert_array_less(prob['T'][:-1], prob['T'][1:])
        self.assertAlmostEqual(prob['rated_V'], V_expect1[-1], 3)
        self.assertAlmostEqual(prob['rated_Omega'][0], 15.0)
        self.assertGreater(prob['rated_pitch'], 0.0)
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob['Cp'][:irated])

        # Test maxTS, no max rpm, no power limit
        prob['omega_max'] = 1e3
        prob['control_maxTS'] = 105.0
        prob['rated_power'] = 1e16
        prob.run_model()
        V_expect1 = V_expect0.copy()
        #V_expect1[irated] = 105./10.
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, 105./70./2/np.pi*60)
        npt.assert_allclose(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_expect)
        npt.assert_equal(prob['pitch'][:irated], 0.0 )
        npt.assert_array_less(0.0, np.abs(prob['pitch'][irated:]))
        npt.assert_array_almost_equal(prob['Cp'], prob['Cp_aero']*0.975*0.975) 
        npt.assert_array_less(prob['P'][:-1], prob['P'][1:])
        npt.assert_array_less(prob['Q'][:-1], prob['Q'][1:])
        npt.assert_array_less(prob['T'][:-1], prob['T'][1:])
        self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob['rated_Omega'][0], Omega_expect[-1])
        self.assertGreater(prob['rated_pitch'], 0.0)
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob['Cp'][:irated])

        # Test no maxTS, no max rpm, power limit
        prob['omega_max'] = 1e3
        prob['control_maxTS'] = 1e4
        prob['rated_power'] = 5e6
        prob.run_model()
        V_expect1 = V_expect0.copy()
        V_expect1[irated] = prob['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, prob['rated_Omega'])
        npt.assert_allclose(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_expect)
        npt.assert_equal(prob['pitch'][:irated], 0.0 )
        npt.assert_array_less(0.0, np.abs(prob['pitch'][(irated+1):]))
        npt.assert_array_almost_equal(prob['Cp'], prob['Cp_aero']*0.975*0.975)
        npt.assert_array_less(prob['P'][:irated], prob['P'][1:(irated+1)])
        npt.assert_allclose(prob['P'][irated:], 5e6, rtol=1e-4, atol=0)
        #npt.assert_array_less(prob['Q'], prob['Q'][1:])
        npt.assert_array_less(prob['T'], prob['T'][irated]+1e-1)
        #self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob['rated_Omega'][0], Omega_expect[-1])
        self.assertEqual(prob['rated_pitch'], 0.0)
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob['Cp'][:irated])
        
        # Test min & max rpm, no power limit
        prob['omega_min'] = 7.0
        prob['omega_max'] = 15.0
        prob['control_maxTS'] = 1e5
        prob['rated_power'] = 1e16
        prob.run_model()
        V_expect1 = V_expect0.copy()
        #V_expect1[irated] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.maximum( np.minimum(Omega_tsr, 15.0), 7.0)
        npt.assert_allclose(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_expect)
        npt.assert_array_less(0.0, np.abs(prob['pitch'][Omega_expect != Omega_tsr]) )
        npt.assert_array_almost_equal(prob['Cp'], prob['Cp_aero']*0.975*0.975) 
        npt.assert_array_less(prob['P'][:-1], prob['P'][1:])
        npt.assert_array_less(prob['Q'][:-1], prob['Q'][1:])
        npt.assert_array_less(prob['T'][:-1], prob['T'][1:])
        self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob['rated_Omega'][0], 15.0)
        self.assertGreater(prob['rated_pitch'], 0.0)
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[Omega_expect == Omega_tsr], myCp[6])
        npt.assert_allclose(myCp[Omega_expect == Omega_tsr], prob['Cp'][Omega_expect == Omega_tsr])
        
        # Test fixed pitch
        prob['omega_min'] = 0.0
        prob['omega_max'] = 15.0
        prob['control_maxTS'] = 1e5
        prob['rated_power'] = 1e16
        prob['control_pitch'] = 5.0
        prob.run_model()
        V_expect1 = V_expect0.copy()
        #V_expect1[irated] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, 15.0)
        npt.assert_allclose(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_expect)
        npt.assert_equal(prob['pitch'][:irated], 5.0 )
        npt.assert_array_less(0.0, np.abs(prob['pitch'][irated:]))
        npt.assert_array_almost_equal(prob['Cp'], prob['Cp_aero']*0.975*0.975)
        npt.assert_array_less(prob['P'][:-1], prob['P'][1:])
        npt.assert_array_less(prob['Q'][:-1], prob['Q'][1:])
        npt.assert_array_less(prob['T'][:-1], prob['T'][1:])
        self.assertAlmostEqual(prob['rated_V'], V_expect1[-1], 3)
        self.assertAlmostEqual(prob['rated_Omega'][0], 15.0)
        self.assertGreater(prob['rated_pitch'], 5.0)
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob['Cp'][:irated])
        
    def testRegulationTrajectoryNoRegion3(self):
        prob = om.Problem()
        
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        
        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(ARCHIVE)
        ivc.add_output('airfoils_aoa', npzfile['aoa'], units='deg')
        ivc.add_output('airfoils_Re', npzfile['Re'])
        ivc.add_output('airfoils_cl', np.moveaxis(npzfile['cl'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('airfoils_cd', np.moveaxis(npzfile['cd'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('airfoils_cm', np.moveaxis(npzfile['cm'][:,:,:,np.newaxis], 0, 1))
        ivc.add_output('r', npzfile['r'], units='m')
        ivc.add_output('chord', npzfile['chord'], units='m')
        ivc.add_output('theta', npzfile['theta'], units='deg')

        n_span = npzfile['r'].size
        n_aoa = npzfile['aoa'].size
        n_Re = npzfile['Re'].size
        n_pc = 22
        
        # parameters
        ivc.add_output('v_min', 4., units='m/s')
        ivc.add_output('v_max', 25., units='m/s')
        ivc.add_output('rated_power', 5e6, units='W')
        ivc.add_output('omega_min', 0.0, units='rpm')
        ivc.add_output('omega_max', 100.0, units='rpm')
        ivc.add_output('control_maxTS', 90., units='m/s')
        ivc.add_output('tsr_operational', 10.)
        ivc.add_output('control_pitch', 0.0, units='deg')
        ivc.add_output('gearbox_efficiency', 0.975)
        ivc.add_output('generator_efficiency', 0.975)
        ivc.add_discrete_output('drivetrainType', 'GEARED')
        
        ivc.add_output('Rhub', 1., units='m')
        ivc.add_output('Rtip', 70., units='m')
        ivc.add_output('hub_height', 100., units='m')
        ivc.add_output('precone', 0., units='deg')
        ivc.add_output('tilt', 0., units='deg')
        ivc.add_output('yaw', 0., units='deg')
        ivc.add_output('precurve', np.zeros(n_span), units='m')
        ivc.add_output('precurveTip', 0., units='m')
        ivc.add_output('presweep', np.zeros(n_span), units='m')
        ivc.add_output('presweepTip', 0., units='m')
        
        ivc.add_output('rho', 1.225, units='kg/m**3')
        ivc.add_output('mu', 1.81206e-5, units='kg/(m*s)')
        ivc.add_output('shearExp', 0.25)
        ivc.add_discrete_output('nBlades', 3)
        ivc.add_discrete_output('nSector', 4)
        ivc.add_discrete_output('tiploss', True)
        ivc.add_discrete_output('hubloss', True)
        ivc.add_discrete_output('wakerotation', True)
        ivc.add_discrete_output('usecd', True)

        analysis_options = {}
        analysis_options['blade'] = {}
        analysis_options['blade']['n_span'] = n_span
        analysis_options['blade']['n_aoa'] = n_aoa
        analysis_options['blade']['n_Re'] = n_Re
        analysis_options['blade']['n_tab'] = 1
        analysis_options['servose'] = {}
        analysis_options['servose']['regulation_reg_III'] = False
        analysis_options['servose']['n_pc'] = n_pc
        analysis_options['servose']['n_pc_spline'] = n_pc

        n_span, n_aoa, n_Re, n_tab = np.moveaxis(npzfile['cl'][:,:,:,np.newaxis], 0, 1).shape
        analysis_options['airfoils'] = {}
        analysis_options['airfoils']['n_aoa'] = n_aoa
        analysis_options['airfoils']['n_Re'] = n_Re
        analysis_options['airfoils']['n_tab'] = n_tab
        
        prob.model.add_subsystem('powercurve', serv.RegulatedPowerCurve(analysis_options=analysis_options), promotes=['*'])
        
        prob.setup()
        
        # All reg 2: no maxTS, no max rpm, no power limit
        prob['omega_max'] = 1e3
        prob['control_maxTS'] = 1e5
        prob['rated_power'] = 1e16
        prob.run_model()
        
        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi/4.,np.pi/2.,n_pc + 1)))))
        grid1 = (grid0 - grid0[0])/(grid0[-1]-grid0[0])
        V_expect0  = grid1 * (prob['v_max'] - prob['v_min']) + prob['v_min']
        V_spline = np.linspace(prob['v_min'], prob['v_max'], n_pc)
        irated = 12
        
        V_expect1 = V_expect0.copy()
        #V_expect1[irated] = prob['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        npt.assert_equal(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_tsr)
        npt.assert_equal(prob['pitch'], np.zeros( V_expect0.shape ) )
        npt.assert_array_almost_equal(prob['Cp'], prob['Cp_aero']*0.975*0.975)
        npt.assert_allclose(prob['Cp'], prob['Cp'][0])
        npt.assert_allclose(prob['Cp_aero'], prob['Cp_aero'][0])
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp, myCp[0])
        self.assertGreater(myCp[0], 0.4)
        self.assertGreater(0.5, myCp[0])
        npt.assert_allclose(myCp, prob['Cp'])
        npt.assert_array_less(prob['P'][:-1], prob['P'][1:])
        npt.assert_array_less(prob['Q'][:-1], prob['Q'][1:])
        npt.assert_array_less(prob['T'][:-1], prob['T'][1:])
        self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob['rated_Omega'][0], Omega_tsr[-1])
        self.assertEqual(prob['rated_pitch'], 0.0)
        
        # Test no maxTS, no max rpm, power limit
        prob['omega_max'] = 1e3
        prob['control_maxTS'] = 1e4
        prob['rated_power'] = 5e6
        prob.run_model()
        V_expect1 = V_expect0.copy()
        V_expect1[irated] = prob['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, prob['rated_Omega'])
        npt.assert_allclose(prob['V'], V_expect1)
        npt.assert_equal(prob['V_spline'], V_spline.flatten())
        npt.assert_allclose(prob['Omega'], Omega_expect)
        npt.assert_equal(prob['pitch'], 0.0 )
        #npt.assert_array_less(0.0, np.abs(prob['pitch'][(irated+1):]))
        npt.assert_allclose(prob['Cp'][:(irated+1)], prob['Cp_aero'][:(irated+1)]*0.975*0.975)
        npt.assert_array_less(prob['P'][:irated], prob['P'][1:(irated+1)])
        npt.assert_allclose(prob['P'][irated:], 5e6, rtol=1e-6, atol=0)
        #npt.assert_equal(prob['Q'][(irated+1):], prob['Q'][irated])
        npt.assert_equal(prob['T'][(irated+1):], 0.0)
        npt.assert_array_less(prob['T'], prob['T'][irated]+1e-1)
        #self.assertEqual(prob['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(prob['rated_Omega'][0], Omega_expect[-1])
        self.assertEqual(prob['rated_pitch'], 0.0)
        myCp = prob['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:irated], myCp[0])
        npt.assert_allclose(myCp[:irated], prob['Cp'][:irated])

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestServo))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
