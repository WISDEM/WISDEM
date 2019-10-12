import numpy as np
import numpy.testing as npt
import unittest
import wisdem.floatingse.substructure as subs

from wisdem.commonse import gravity as g
NSECTIONS = 5
NPTS = 100

class TestSubs(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resids = {}

        self.inputs['structural_mass'] = 1e4
        self.inputs['structure_center_of_mass'] = 40.0*np.ones(3)
        self.inputs['structural_frequencies'] = 100.0*np.ones(6)
        self.inputs['total_force'] = 26.0*np.ones(3)
        self.inputs['total_moment'] = 5e4*np.ones(3)
        self.inputs['total_displacement'] = 1e4
        
        self.inputs['mooring_mass'] = 20.0
        self.inputs['mooring_neutral_load'] = np.zeros((15,3))
        self.inputs['mooring_neutral_load'][:3,:] = 5.0*g
        self.inputs['mooring_surge_restoring_force'] = 1e2
        self.inputs['mooring_pitch_restoring_force'] = 1e5 * np.ones((10,3))
        self.inputs['mooring_pitch_restoring_force'][3:,:] = 0.0
        self.inputs['mooring_cost'] = 256.0
        self.inputs['mooring_stiffness'] = np.ones((6,6))
        self.inputs['mooring_moments_of_inertia'] = np.array([10.0, 10.0, 2.0, 0.0, 0.0, 0.0])
        self.inputs['fairlead'] = 0.5
        self.inputs['fairlead_location'] = 0.1
        self.inputs['fairlead_radius'] = 5.0
        self.inputs['max_survival_heel'] = 10.0
        self.inputs['operational_heel'] = 10.0

        self.inputs['pontoon_cost'] = 512.0
        

        self.inputs['Hs'] = 10.0
        self.inputs['wave_period'] = 50.0
        
        self.inputs['main_Iwaterplane'] = 150.0
        self.inputs['main_Awaterplane'] = 20.0
        self.inputs['main_mass'] = 2.0*np.ones(NPTS-1)
        self.inputs['main_cost'] = 32.0
        self.inputs['main_freeboard'] = 10.0
        self.inputs['main_center_of_mass'] = -10.0
        self.inputs['main_center_of_buoyancy'] = -8.0
        self.inputs['main_added_mass'] = 2*np.array([10.0, 10.0, 2.0, 30.0, 30.0, 0.0])
        self.inputs['main_moments_of_inertia'] = 1e2 * np.array([10.0, 10.0, 2.0, 0.0, 0.0, 0.0])

        self.inputs['offset_Iwaterplane'] = 50.0
        self.inputs['offset_Awaterplane'] = 9.0
        self.inputs['offset_cost'] = 64.0
        self.inputs['offset_mass'] = np.ones(NPTS-1)
        self.inputs['offset_center_of_mass'] = -5.0
        self.inputs['offset_center_of_buoyancy'] = -4.0
        self.inputs['offset_added_mass'] = np.array([10.0, 10.0, 2.0, 30.0, 30.0, 0.0])
        self.inputs['offset_moments_of_inertia'] = 1e1 * np.array([10.0, 10.0, 2.0, 0.0, 0.0, 0.0])
        self.inputs['offset_freeboard'] = 10.0
        self.inputs['offset_draft'] = 15.0

        self.inputs['tower_z_full'] = np.linspace(0, 90, 3*NSECTIONS+1)
        self.inputs['tower_mass'] = 2e2
        self.inputs['tower_shell_cost'] = 2e5
        self.inputs['tower_d_full'] = 5.0*np.ones(NPTS)
        self.inputs['tower_I_base'] = 1e5*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs['rna_mass'] = 6e1
        self.inputs['rna_cg'] = np.array([0.0, 0.0, 5.0])
        self.inputs['rna_I'] = 1e5*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs['Rhub'] = 3.0
        
        self.inputs['number_of_offset_columns'] = 3
        self.inputs['water_ballast_radius_vector'] = 40.0 * np.ones(5)
        self.inputs['water_ballast_zpts_vector'] = np.array([-10, -9, -8, -7, -6])
        self.inputs['radius_to_offset_column'] = 20.0
        self.inputs['z_center_of_buoyancy'] = -2.0

        self.inputs['water_density'] = 1e3
        self.inputs['wave_period_range_low'] = 2.0
        self.inputs['wave_period_range_high'] = 20.0

        self.mysemi = subs.Substructure(nFull=NPTS,nFullTow=NPTS)
        self.mysemiG = subs.SubstructureGeometry(nFull=2,nFullTow=2)
        
    def testSetGeometry(self):
        self.inputs['number_of_offset_columns'] = 3
        self.inputs['Rhub'] = 3.0
        self.inputs['tower_d_full'] = 2.0*5.0*np.ones(3)
        self.inputs['main_d_full'] = 2*np.array([10.0, 10.0, 10.0])
        self.inputs['offset_d_full'] = 2*np.array([10.0, 10.0, 10.0])
        self.inputs['offset_z_nodes'] = np.array([-35.0, -15.0, 15.0])
        self.inputs['main_z_nodes'] = np.array([-35.0, -15.0, 15.0])
        self.inputs['radius_to_offset_column'] = 25.0
        self.inputs['fairlead_location'] = 0.1
        self.inputs['fairlead_offset_from_shell'] = 1.0
        self.inputs['offset_freeboard'] = 10.0
        self.inputs['offset_draft'] = 15.0
        self.mysemiG.compute(self.inputs, self.outputs)

        # Semi
        self.assertEqual(self.outputs['fairlead'], 30.0)
        self.assertEqual(self.outputs['fairlead_radius'], 11.0+25.0)
        self.assertEqual(self.outputs['main_offset_spacing'], 25.0 - 10.0 - 10.0)
        self.assertEqual(self.outputs['tower_transition_buffer'], 10-5.0)
        self.assertEqual(self.outputs['nacelle_transition_buffer'], 4.0-5.0)
        self.assertEqual(self.outputs['offset_freeboard_heel_margin'], 10.0 - 25.0*np.sin(np.deg2rad(10.0)))
        self.assertEqual(self.outputs['offset_draft_heel_margin'], 15.0 - 25.0*np.sin(np.deg2rad(10.0)))

        # Spar
        self.inputs['number_of_offset_columns'] = 0
        self.mysemiG.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs['fairlead'], 30.0)
        self.assertEqual(self.outputs['fairlead_radius'], 11.0)
        self.assertEqual(self.outputs['main_offset_spacing'], 25.0 - 10.0 - 10.0)
        self.assertEqual(self.outputs['tower_transition_buffer'], 10-5.0)
        self.assertEqual(self.outputs['nacelle_transition_buffer'], 4.0-5.0)

        
    def testBalance(self):
        self.mysemi.balance(self.inputs, self.outputs)
        m_water = 1e3*1e4 - 1e4 - 15
        z_data = self.inputs['water_ballast_zpts_vector']
        h_data = z_data - z_data[0]
        h_expect = np.interp(m_water, 1e3*h_data*np.pi*self.inputs['water_ballast_radius_vector']**2, h_data)
        cg_expect_z = (1e4*40.0 + m_water*(-10 + 0.5*h_expect)) / (1e4+m_water)
        cg_expect_xy = 1e4*40.0/ (1e4+m_water)

        self.assertEqual(self.outputs['variable_ballast_mass'], m_water)
        self.assertEqual(self.outputs['variable_ballast_height_ratio'], h_expect/4.0)
        npt.assert_almost_equal(self.outputs['center_of_mass'], np.array([cg_expect_xy, cg_expect_xy, cg_expect_z]))
        
        self.inputs['number_of_offset_columns'] = 0
        self.mysemi.balance(self.inputs, self.outputs)

        self.assertEqual(self.outputs['variable_ballast_mass'], m_water)
        self.assertEqual(self.outputs['variable_ballast_height_ratio'], h_expect/4.0)
        npt.assert_almost_equal(self.outputs['center_of_mass'], np.array([cg_expect_xy, cg_expect_xy, cg_expect_z]))
        

    def testStability(self):
        self.inputs['mooring_pitch_restoring_force'] = 0.0 * np.ones((10,3))
        self.outputs['center_of_mass'] = np.array([0.0, 0.0, -1.0])
        self.mysemi.compute_stability(self.inputs, self.outputs)

        I_expect = 150.0 + (50.0 + 9.0*(20.0*np.cos(np.deg2rad(np.array([0.0, 120., 240.0]))) )**2).sum()
        static_expect = -1.0 + 2.0
        meta_expect = I_expect/1e4 - static_expect
        wind_fact = np.cos(np.deg2rad(10.0))**2.0
        self.assertEqual(self.outputs['buoyancy_to_gravity'], static_expect)
        self.assertEqual(self.outputs['metacentric_height'], meta_expect)
        self.assertEqual(self.outputs['offset_force_ratio'], 26.0/1e2)
        self.assertAlmostEqual(self.outputs['heel_moment_ratio'], (wind_fact*5e4)/(1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))

        self.inputs['number_of_offset_columns'] = 0
        self.mysemi.compute_stability(self.inputs, self.outputs)

        I_expect = 150.0
        meta_expect = I_expect/1e4 - static_expect
        self.assertEqual(self.outputs['buoyancy_to_gravity'], static_expect)
        self.assertEqual(self.outputs['metacentric_height'], meta_expect)
        self.assertEqual(self.outputs['offset_force_ratio'], 26.0/1e2)
        self.assertAlmostEqual(self.outputs['heel_moment_ratio'], (wind_fact*5e4)/(1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))

        self.inputs['fairlead'] = 1.0
        self.inputs['mooring_pitch_restoring_force'][:3,-1] = 1.0
        self.assertAlmostEqual(self.outputs['heel_moment_ratio'], (wind_fact*5e4)/(1*5 + 1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))


    def testPeriods(self):
        # Spar first
        self.inputs['structure_center_of_mass'] = np.array([0.0, 0.0, -40.0])
        self.inputs['number_of_offset_columns'] = 0
        self.mysemi.balance(self.inputs, self.outputs)
        self.inputs['main_center_of_mass'] = self.outputs['center_of_mass'][-1]
        self.inputs['main_center_of_buoyancy'] = self.outputs['center_of_mass'][-1]+2.0
        self.inputs['tower_mass'] = 0.0
        self.inputs['rna_mass'] = 0.0
        self.inputs['rna_I'] = 1e2*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs['tower_I_base'] = 1e2*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.mysemi.compute_stability(self.inputs, self.outputs)
        self.mysemi.compute_rigid_body_periods(self.inputs, self.outputs)

        m_struct = self.inputs['structural_mass']
        m_water  = self.outputs['variable_ballast_mass']
        z_cg     = self.inputs['main_center_of_mass']
        z_water  = self.outputs['variable_ballast_center_of_mass']
        I_water  = self.outputs['variable_ballast_moments_of_inertia']
        M_expect = np.zeros(6)
        M_expect[:3] = m_struct + m_water
        M_expect[3:] = self.inputs['main_moments_of_inertia'][:3]
        M_expect[3:5] += I_water[:2] + m_water*(z_water-z_cg)**2
        M_expect[-1] += I_water[2]
        M_expect[3:] += 2e2
        npt.assert_equal(self.outputs['mass_matrix'], M_expect)

        A_expect = np.zeros(6)
        A_expect[:3] = self.inputs['main_added_mass'][:3]
        A_expect[3:5] = self.inputs['main_added_mass'][3:5] + A_expect[0]*(2.0)**2
        npt.assert_equal(self.outputs['added_mass_matrix'], A_expect)

        rho_w = self.inputs['water_density']
        K_expect = np.zeros(6)
        K_expect[2] = rho_w * g * 20.0 # waterplane area for main
        K_expect[3:5] = rho_w * g * self.outputs['metacentric_height'] * 1e4 # Total displacement
        npt.assert_almost_equal(self.outputs['hydrostatic_stiffness'], K_expect)

        T_expect = 2 * np.pi * np.sqrt( (M_expect + A_expect) / (1e-6 + K_expect + np.diag(self.inputs['mooring_stiffness'])) )
        npt.assert_almost_equal(self.outputs['rigid_body_periods'], T_expect)

        
    def testMargins(self):
        
        self.mysemi.balance(self.inputs, self.outputs)
        self.mysemi.compute_stability(self.inputs, self.outputs)
        self.mysemi.compute_rigid_body_periods(self.inputs, self.outputs)
        self.mysemi.check_frequency_margins(self.inputs, self.outputs)

        myones = np.ones(6)
        T_sys    = self.outputs['rigid_body_periods']
        T_wave_low  = self.inputs['wave_period_range_low']*myones
        T_wave_high = self.inputs['wave_period_range_high']*myones
        f_struct    = self.inputs['structural_frequencies']
        T_struct    = 1.0 / f_struct

        T_wave_high[-1] = 1e-16
        T_wave_low[-1] = 1e30

        ind = T_sys>T_wave_low
        npt.assert_equal(self.outputs['period_margin_high'][ind], T_sys[ind]/T_wave_high[ind] )

        ind = T_sys<T_wave_high
        npt.assert_equal(self.outputs['period_margin_low'][ind], T_sys[ind]/T_wave_low[ind] )

        ind = T_struct>T_wave_low
        npt.assert_equal(self.outputs['modal_margin_high'][ind], T_struct[ind]/T_wave_high[ind] )

        ind = T_struct<T_wave_high
        npt.assert_equal(self.outputs['modal_margin_low'][ind], T_struct[ind]/T_wave_low[ind] )
        
    def testCost(self):
        self.mysemi.compute_costs(self.inputs, self.outputs)
        c_expect = 256.0 + 512.0 + 32.0 + 3*64.0 + 2e5
        self.assertEqual(self.outputs['total_cost'], c_expect)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSubs))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
