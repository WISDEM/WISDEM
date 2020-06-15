import numpy as np
import numpy.testing as npt
import unittest
import wisdem.rotorse.rotor_aeropower as ra
import openmdao.api as om
import copy
import time
import os
ARCHIVE  = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'regulation.npz'
        

class TestRotorAero(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        
    def testRegulationTrajectory(self):
        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(ARCHIVE)
        self.inputs['airfoils_aoa'] = npzfile['aoa']
        self.inputs['airfoils_Re'] = npzfile['Re']
        self.inputs['airfoils_cl'] = npzfile['cl']
        self.inputs['airfoils_cd'] = npzfile['cd']
        self.inputs['airfoils_cm'] = npzfile['cm']
        self.inputs['r'] = npzfile['r']
        self.inputs['chord'] = npzfile['chord']
        self.inputs['theta'] = npzfile['theta']

        naero = self.inputs['r'].size
        n_aoa_grid = self.inputs['airfoils_aoa'].size
        n_Re_grid = self.inputs['airfoils_Re'].size
        n_pc = 22
        
        # parameters
        self.inputs['control_Vin'] = 4.
        self.inputs['control_Vout'] = 25.
        self.inputs['control_ratedPower'] = 5e6
        self.inputs['control_minOmega'] = 0.0
        self.inputs['control_maxOmega'] = 100.0
        self.inputs['control_maxTS'] = 90.
        self.inputs['control_tsr'] = 10.
        self.inputs['control_pitch'] = 0.0
        self.discrete_inputs['drivetrainType'] = 'GEARED'
        self.inputs['drivetrainEff'] = 0.95
        
        self.inputs['Rhub'] = 1.
        self.inputs['Rtip'] = 70.
        self.inputs['hub_height'] = 100.
        self.inputs['precone'] = 0.
        self.inputs['tilt'] = 0.
        self.inputs['yaw'] = 0.
        self.inputs['precurve'] = np.zeros(naero)
        self.inputs['precurveTip'] = 0.
        self.inputs['presweep'] = np.zeros(naero)
        self.inputs['presweepTip'] = 0.
        
        self.discrete_inputs['nBlades'] = 3
        self.inputs['rho'] = 1.225
        self.inputs['mu'] = 1.81206e-5
        self.inputs['shearExp'] = 0.25
        self.discrete_inputs['nSector'] = 4
        self.discrete_inputs['tiploss'] = True
        self.discrete_inputs['hubloss'] = True
        self.discrete_inputs['wakerotation'] = True
        self.discrete_inputs['usecd'] = True

        myobj = ra.RegulatedPowerCurve(naero=naero, n_aoa_grid=n_aoa_grid, n_Re_grid=n_Re_grid, n_pc=n_pc, n_pc_spline=n_pc,
                                          regulation_reg_II5=True, regulation_reg_III=True)
        myobj.naero = naero
        
        # All reg 2: no maxTS, no max rpm, no power limit
        self.inputs['control_maxOmega'] = 1e3
        self.inputs['control_maxTS'] = 1e5
        self.inputs['control_ratedPower'] = 1e16
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        #V_expect1[7] = self.outputs['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        npt.assert_equal(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_tsr)
        npt.assert_equal(self.outputs['pitch'], np.zeros( V_expect0.shape ) )
        npt.assert_equal(self.outputs['Cp'], self.outputs['Cp_aero']*0.95)
        npt.assert_allclose(self.outputs['Cp'], self.outputs['Cp'][0])
        npt.assert_allclose(self.outputs['Cp_aero'], self.outputs['Cp_aero'][0])
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp, myCp[0])
        self.assertGreater(myCp[0], 0.4)
        self.assertGreater(0.5, myCp[0])
        npt.assert_allclose(myCp, self.outputs['Cp'])
        npt.assert_array_less(self.outputs['P'][:-1], self.outputs['P'][1:])
        npt.assert_array_less(self.outputs['Q'][:-1], self.outputs['Q'][1:])
        npt.assert_array_less(self.outputs['T'][:-1], self.outputs['T'][1:])
        self.assertEqual(self.outputs['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(self.outputs['rated_Omega'], Omega_tsr[-1])
        self.assertEqual(self.outputs['rated_pitch'], 0.0)
        
        # Test no maxTS, max rpm, no power limit
        self.inputs['control_maxOmega'] = 15.0
        self.inputs['control_maxTS'] = 1e5
        self.inputs['control_ratedPower'] = 1e16
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        #V_expect1[7] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, 15.0)
        npt.assert_allclose(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_expect)
        npt.assert_equal(self.outputs['pitch'][:7], 0.0 )
        npt.assert_array_less(0.0, np.abs(self.outputs['pitch'][7:]))
        npt.assert_equal(self.outputs['Cp'], self.outputs['Cp_aero']*0.95)
        npt.assert_array_less(self.outputs['P'][:-1], self.outputs['P'][1:])
        npt.assert_array_less(self.outputs['Q'][:-1], self.outputs['Q'][1:])
        npt.assert_array_less(self.outputs['T'][:-1], self.outputs['T'][1:])
        self.assertAlmostEqual(self.outputs['rated_V'], V_expect1[-1], 3)
        self.assertAlmostEqual(self.outputs['rated_Omega'], 15.0)
        self.assertGreater(self.outputs['rated_pitch'], 0.0)
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:7], myCp[0])
        npt.assert_allclose(myCp[:7], self.outputs['Cp'][:7])

        # Test maxTS, no max rpm, no power limit
        self.inputs['control_maxOmega'] = 1e3
        self.inputs['control_maxTS'] = 105.0
        self.inputs['control_ratedPower'] = 1e16
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        #V_expect1[7] = 105./10.
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, 105./70./2/np.pi*60)
        npt.assert_allclose(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_expect)
        npt.assert_equal(self.outputs['pitch'][:7], 0.0 )
        npt.assert_array_less(0.0, np.abs(self.outputs['pitch'][7:]))
        npt.assert_equal(self.outputs['Cp'], self.outputs['Cp_aero']*0.95) 
        npt.assert_array_less(self.outputs['P'][:-1], self.outputs['P'][1:])
        npt.assert_array_less(self.outputs['Q'][:-1], self.outputs['Q'][1:])
        npt.assert_array_less(self.outputs['T'][:-1], self.outputs['T'][1:])
        self.assertEqual(self.outputs['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(self.outputs['rated_Omega'], Omega_expect[-1])
        self.assertGreater(self.outputs['rated_pitch'], 0.0)
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:7], myCp[0])
        npt.assert_allclose(myCp[:7], self.outputs['Cp'][:7])

        # Test no maxTS, no max rpm, power limit
        self.inputs['control_maxOmega'] = 1e3
        self.inputs['control_maxTS'] = 1e4
        self.inputs['control_ratedPower'] = 5e6
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        V_expect1[7] = self.outputs['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, self.outputs['rated_Omega'])
        npt.assert_allclose(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_expect)
        npt.assert_equal(self.outputs['pitch'][:7], 0.0 )
        npt.assert_array_less(0.0, np.abs(self.outputs['pitch'][8:]))
        npt.assert_equal(self.outputs['Cp'], self.outputs['Cp_aero']*0.95)
        npt.assert_array_less(self.outputs['P'][:7], self.outputs['P'][1:8])
        npt.assert_allclose(self.outputs['P'][7:], 5e6, rtol=1e-4, atol=0)
        #npt.assert_array_less(self.outputs['Q'], self.outputs['Q'][1:])
        npt.assert_array_less(self.outputs['T'], self.outputs['T'][7]+1e-1)
        #self.assertEqual(self.outputs['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(self.outputs['rated_Omega'], Omega_expect[-1])
        self.assertEqual(self.outputs['rated_pitch'], 0.0)
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:7], myCp[0])
        npt.assert_allclose(myCp[:7], self.outputs['Cp'][:7])
        
        # Test min & max rpm, no power limit
        self.inputs['control_minOmega'] = 7.0
        self.inputs['control_maxOmega'] = 15.0
        self.inputs['control_maxTS'] = 1e5
        self.inputs['control_ratedPower'] = 1e16
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        #V_expect1[7] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.maximum( np.minimum(Omega_tsr, 15.0), 7.0)
        npt.assert_allclose(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_expect)
        npt.assert_array_less(0.0, np.abs(self.outputs['pitch'][Omega_expect != Omega_tsr]) )
        npt.assert_equal(self.outputs['Cp'], self.outputs['Cp_aero']*0.95) 
        npt.assert_array_less(self.outputs['P'][:-1], self.outputs['P'][1:])
        npt.assert_array_less(self.outputs['Q'][:-1], self.outputs['Q'][1:])
        npt.assert_array_less(self.outputs['T'][:-1], self.outputs['T'][1:])
        self.assertEqual(self.outputs['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(self.outputs['rated_Omega'], 15.0)
        self.assertGreater(self.outputs['rated_pitch'], 0.0)
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[Omega_expect == Omega_tsr], myCp[6])
        npt.assert_allclose(myCp[Omega_expect == Omega_tsr], self.outputs['Cp'][Omega_expect == Omega_tsr])
        
        # Test fixed pitch
        self.inputs['control_minOmega'] = 0.0
        self.inputs['control_maxOmega'] = 15.0
        self.inputs['control_maxTS'] = 1e5
        self.inputs['control_ratedPower'] = 1e16
        self.inputs['control_pitch'] = 5.0
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        #V_expect1[7] = 15.*70*2*np.pi/(10.*60.)
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, 15.0)
        npt.assert_allclose(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_expect)
        npt.assert_equal(self.outputs['pitch'][:7], 5.0 )
        npt.assert_array_less(0.0, np.abs(self.outputs['pitch'][7:]))
        npt.assert_equal(self.outputs['Cp'], self.outputs['Cp_aero']*0.95)
        npt.assert_array_less(self.outputs['P'][:-1], self.outputs['P'][1:])
        npt.assert_array_less(self.outputs['Q'][:-1], self.outputs['Q'][1:])
        npt.assert_array_less(self.outputs['T'][:-1], self.outputs['T'][1:])
        self.assertAlmostEqual(self.outputs['rated_V'], V_expect1[-1], 3)
        self.assertAlmostEqual(self.outputs['rated_Omega'], 15.0)
        self.assertGreater(self.outputs['rated_pitch'], 5.0)
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:7], myCp[0])
        npt.assert_allclose(myCp[:7], self.outputs['Cp'][:7])

        
        
    def testRegulationTrajectoryNoRegion3(self):
        # Load in airfoil and blade shape inputs for NREL 5MW
        npzfile = np.load(ARCHIVE)
        self.inputs['airfoils_aoa'] = npzfile['aoa']
        self.inputs['airfoils_Re'] = npzfile['Re']
        self.inputs['airfoils_cl'] = npzfile['cl']
        self.inputs['airfoils_cd'] = npzfile['cd']
        self.inputs['airfoils_cm'] = npzfile['cm']
        self.inputs['r'] = npzfile['r']
        self.inputs['chord'] = npzfile['chord']
        self.inputs['theta'] = npzfile['theta']

        naero = self.inputs['r'].size
        n_aoa_grid = self.inputs['airfoils_aoa'].size
        n_Re_grid = self.inputs['airfoils_Re'].size
        n_pc = 22
        
        # parameters
        self.inputs['control_Vin'] = 4.
        self.inputs['control_Vout'] = 25.
        self.inputs['control_ratedPower'] = 5e6
        self.inputs['control_minOmega'] = 0.0
        self.inputs['control_maxOmega'] = 100.0
        self.inputs['control_maxTS'] = 90.
        self.inputs['control_tsr'] = 10.
        self.inputs['control_pitch'] = 0.0
        self.discrete_inputs['drivetrainType'] = 'GEARED'
        self.inputs['drivetrainEff'] = 0.95
        
        self.inputs['Rhub'] = 1.
        self.inputs['Rtip'] = 70.
        self.inputs['hub_height'] = 100.
        self.inputs['precone'] = 0.
        self.inputs['tilt'] = 0.
        self.inputs['yaw'] = 0.
        self.inputs['precurve'] = np.zeros(naero)
        self.inputs['precurveTip'] = 0.
        self.inputs['presweep'] = np.zeros(naero)
        self.inputs['presweepTip'] = 0.
        
        self.discrete_inputs['nBlades'] = 3
        self.inputs['rho'] = 1.225
        self.inputs['mu'] = 1.81206e-5
        self.inputs['shearExp'] = 0.25
        self.discrete_inputs['nSector'] = 4
        self.discrete_inputs['tiploss'] = True
        self.discrete_inputs['hubloss'] = True
        self.discrete_inputs['wakerotation'] = True
        self.discrete_inputs['usecd'] = True

        myobj = ra.RegulatedPowerCurve(naero=naero, n_aoa_grid=n_aoa_grid, n_Re_grid=n_Re_grid, n_pc=n_pc, n_pc_spline=n_pc,
                                          regulation_reg_II5=True, regulation_reg_III=False)
        myobj.naero = naero
        
        # All reg 2: no maxTS, no max rpm, no power limit
        self.inputs['control_maxOmega'] = 1e3
        self.inputs['control_maxTS'] = 1e5
        self.inputs['control_ratedPower'] = 1e16
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        #V_expect1[7] = self.outputs['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        npt.assert_equal(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_tsr)
        npt.assert_equal(self.outputs['pitch'], np.zeros( V_expect0.shape ) )
        npt.assert_equal(self.outputs['Cp'], self.outputs['Cp_aero']*0.95)
        npt.assert_allclose(self.outputs['Cp'], self.outputs['Cp'][0])
        npt.assert_allclose(self.outputs['Cp_aero'], self.outputs['Cp_aero'][0])
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp, myCp[0])
        self.assertGreater(myCp[0], 0.4)
        self.assertGreater(0.5, myCp[0])
        npt.assert_allclose(myCp, self.outputs['Cp'])
        npt.assert_array_less(self.outputs['P'][:-1], self.outputs['P'][1:])
        npt.assert_array_less(self.outputs['Q'][:-1], self.outputs['Q'][1:])
        npt.assert_array_less(self.outputs['T'][:-1], self.outputs['T'][1:])
        self.assertEqual(self.outputs['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(self.outputs['rated_Omega'], Omega_tsr[-1])
        self.assertEqual(self.outputs['rated_pitch'], 0.0)
        
        # Test no maxTS, no max rpm, power limit
        self.inputs['control_maxOmega'] = 1e3
        self.inputs['control_maxTS'] = 1e4
        self.inputs['control_ratedPower'] = 5e6
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V_expect0 = np.linspace(4, 25, n_pc)
        V_expect1 = V_expect0.copy()
        V_expect1[7] = self.outputs['rated_V']
        Omega_tsr = V_expect1*10*60/70./2./np.pi
        Omega_expect = np.minimum(Omega_tsr, self.outputs['rated_Omega'])
        npt.assert_allclose(self.outputs['V'], V_expect1)
        npt.assert_equal(self.outputs['V_spline'], V_expect0)
        npt.assert_allclose(self.outputs['Omega'], Omega_expect)
        npt.assert_equal(self.outputs['pitch'], 0.0 )
        #npt.assert_array_less(0.0, np.abs(self.outputs['pitch'][8:]))
        npt.assert_allclose(self.outputs['Cp'][:8], self.outputs['Cp_aero'][:8]*0.95)
        npt.assert_array_less(self.outputs['P'][:7], self.outputs['P'][1:8])
        npt.assert_allclose(self.outputs['P'][7:], 5e6, rtol=1e-6, atol=0)
        #npt.assert_equal(self.outputs['Q'][8:], self.outputs['Q'][7])
        npt.assert_equal(self.outputs['T'][8:], 0.0)
        npt.assert_array_less(self.outputs['T'], self.outputs['T'][7]+1e-1)
        #self.assertEqual(self.outputs['rated_V'], V_expect1[-1])
        self.assertAlmostEqual(self.outputs['rated_Omega'], Omega_expect[-1])
        self.assertEqual(self.outputs['rated_pitch'], 0.0)
        myCp = self.outputs['P']/(0.5*1.225*V_expect1**3.*np.pi*70**2)
        npt.assert_allclose(myCp[:7], myCp[0])
        npt.assert_allclose(myCp[:7], self.outputs['Cp'][:7])
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRotorAero))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
