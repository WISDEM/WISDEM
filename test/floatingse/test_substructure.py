import numpy as np
import numpy.testing as npt
import unittest
import floatingse.substructure as subs

from commonse import gravity as g
NSECTIONS = 5
NPTS = 100

class TestSubs(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resids = {}

        self.params['structural_mass'] = 1e4
        self.params['structure_center_of_mass'] = 40.0*np.ones(3)
        self.params['structural_frequencies'] = 100.0*np.ones(6)
        self.params['total_force'] = 26.0*np.ones(3)
        self.params['total_moment'] = 5e4*np.ones(3)
        self.params['total_displacement'] = 1e4
        
        self.params['mooring_mass'] = 20.0
        self.params['mooring_neutral_load'] = np.zeros((15,3))
        self.params['mooring_neutral_load'][:3,:] = 5.0*g
        self.params['mooring_surge_restoring_force'] = 1e2
        self.params['mooring_pitch_restoring_force'] = 1e5 * np.ones((10,3))
        self.params['mooring_pitch_restoring_force'][3:,:] = 0.0
        self.params['mooring_cost'] = 256.0
        self.params['mooring_stiffness'] = np.ones((6,6))
        self.params['mooring_moments_of_inertia'] = np.array([10.0, 10.0, 2.0, 0.0, 0.0, 0.0])
        self.params['fairlead'] = 0.5
        self.params['fairlead_location'] = 0.1
        self.params['fairlead_radius'] = 5.0
        self.params['max_survival_heel'] = 10.0
        self.params['operational_heel'] = 10.0

        self.params['pontoon_cost'] = 512.0
        

        self.params['Hs'] = 10.0
        self.params['wave_period'] = 50.0
        
        self.params['main_Iwaterplane'] = 150.0
        self.params['main_Awaterplane'] = 20.0
        self.params['main_mass'] = 2.0*np.ones(NPTS-1)
        self.params['main_cost'] = 32.0
        self.params['main_freeboard'] = 10.0
        self.params['main_center_of_mass'] = -10.0
        self.params['main_center_of_buoyancy'] = -8.0
        self.params['main_added_mass'] = 2*np.array([10.0, 10.0, 2.0, 30.0, 30.0, 0.0])
        self.params['main_moments_of_inertia'] = 1e2 * np.array([10.0, 10.0, 2.0, 0.0, 0.0, 0.0])

        self.params['offset_Iwaterplane'] = 50.0
        self.params['offset_Awaterplane'] = 9.0
        self.params['offset_cost'] = 64.0
        self.params['offset_mass'] = np.ones(NPTS-1)
        self.params['offset_center_of_mass'] = -5.0
        self.params['offset_center_of_buoyancy'] = -4.0
        self.params['offset_added_mass'] = np.array([10.0, 10.0, 2.0, 30.0, 30.0, 0.0])
        self.params['offset_moments_of_inertia'] = 1e1 * np.array([10.0, 10.0, 2.0, 0.0, 0.0, 0.0])
        self.params['offset_freeboard'] = 10.0
        self.params['offset_draft'] = 15.0

        self.params['tower_z_full'] = np.linspace(0, 90, 3*NSECTIONS+1)
        self.params['tower_mass'] = 2e2
        self.params['tower_d_full'] = 5.0*np.ones(NPTS)
        self.params['tower_I_base'] = 1e5*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.params['rna_mass'] = 6e1
        self.params['rna_cg'] = np.array([0.0, 0.0, 5.0])
        self.params['rna_I'] = 1e5*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.params['Rhub'] = 3.0
        
        self.params['number_of_offset_columns'] = 3
        self.params['water_ballast_radius_vector'] = 40.0 * np.ones(5)
        self.params['water_ballast_zpts_vector'] = np.array([-10, -9, -8, -7, -6])
        self.params['radius_to_offset_column'] = 20.0
        self.params['z_center_of_buoyancy'] = -2.0

        self.params['water_density'] = 1e3
        self.params['wave_period_range_low'] = 2.0
        self.params['wave_period_range_high'] = 20.0

        self.mysemi = subs.Substructure(NPTS,NPTS)
        self.mysemiG = subs.SubstructureGeometry(2,2)
        
    def testSetGeometry(self):
        self.params['number_of_offset_columns'] = 3
        self.params['Rhub'] = 3.0
        self.params['tower_d_full'] = 2.0*5.0*np.ones(3)
        self.params['main_d_full'] = 2*np.array([10.0, 10.0, 10.0])
        self.params['offset_d_full'] = 2*np.array([10.0, 10.0, 10.0])
        self.params['offset_z_nodes'] = np.array([-35.0, -15.0, 15.0])
        self.params['main_z_nodes'] = np.array([-35.0, -15.0, 15.0])
        self.params['radius_to_offset_column'] = 25.0
        self.params['fairlead_location'] = 0.1
        self.params['fairlead_offset_from_shell'] = 1.0
        self.params['offset_freeboard'] = 10.0
        self.params['offset_draft'] = 15.0
        self.mysemiG.solve_nonlinear(self.params, self.unknowns, None)

        # Semi
        self.assertEqual(self.unknowns['fairlead'], 30.0)
        self.assertEqual(self.unknowns['fairlead_radius'], 11.0+25.0)
        self.assertEqual(self.unknowns['main_offset_spacing'], 25.0 - 10.0 - 10.0)
        self.assertEqual(self.unknowns['tower_transition_buffer'], 10-5.0)
        self.assertEqual(self.unknowns['nacelle_transition_buffer'], 4.0-5.0)
        self.assertEqual(self.unknowns['offset_freeboard_heel_margin'], 10.0 - 25.0*np.sin(np.deg2rad(10.0)))
        self.assertEqual(self.unknowns['offset_draft_heel_margin'], 15.0 - 25.0*np.sin(np.deg2rad(10.0)))

        # Spar
        self.params['number_of_offset_columns'] = 0
        self.mysemiG.solve_nonlinear(self.params, self.unknowns, None)
        self.assertEqual(self.unknowns['fairlead'], 30.0)
        self.assertEqual(self.unknowns['fairlead_radius'], 11.0)
        self.assertEqual(self.unknowns['main_offset_spacing'], 25.0 - 10.0 - 10.0)
        self.assertEqual(self.unknowns['tower_transition_buffer'], 10-5.0)
        self.assertEqual(self.unknowns['nacelle_transition_buffer'], 4.0-5.0)

        
    def testBalance(self):
        self.mysemi.balance(self.params, self.unknowns)
        m_water = 1e3*1e4 - 1e4 - 15
        z_data = self.params['water_ballast_zpts_vector']
        h_data = z_data - z_data[0]
        h_expect = np.interp(m_water, 1e3*h_data*np.pi*self.params['water_ballast_radius_vector']**2, h_data)
        cg_expect_z = (1e4*40.0 + m_water*(-10 + 0.5*h_expect)) / (1e4+m_water)
        cg_expect_xy = 1e4*40.0/ (1e4+m_water)

        self.assertEqual(self.unknowns['variable_ballast_mass'], m_water)
        self.assertEqual(self.unknowns['variable_ballast_height_ratio'], h_expect/4.0)
        npt.assert_almost_equal(self.unknowns['center_of_mass'], np.array([cg_expect_xy, cg_expect_xy, cg_expect_z]))
        
        self.params['number_of_offset_columns'] = 0
        self.mysemi.balance(self.params, self.unknowns)

        self.assertEqual(self.unknowns['variable_ballast_mass'], m_water)
        self.assertEqual(self.unknowns['variable_ballast_height_ratio'], h_expect/4.0)
        npt.assert_almost_equal(self.unknowns['center_of_mass'], np.array([cg_expect_xy, cg_expect_xy, cg_expect_z]))
        

    def testStability(self):
        self.params['mooring_pitch_restoring_force'] = 0.0 * np.ones((10,3))
        self.unknowns['center_of_mass'] = np.array([0.0, 0.0, -1.0])
        self.mysemi.compute_stability(self.params, self.unknowns)

        I_expect = 150.0 + (50.0 + 9.0*(20.0*np.cos(np.deg2rad(np.array([0.0, 120., 240.0]))) )**2).sum()
        static_expect = -1.0 + 2.0
        meta_expect = I_expect/1e4 - static_expect
        wind_fact = np.cos(np.deg2rad(10.0))**2.0
        self.assertEqual(self.unknowns['buoyancy_to_gravity'], static_expect)
        self.assertEqual(self.unknowns['metacentric_height'], meta_expect)
        self.assertEqual(self.unknowns['offset_force_ratio'], 26.0/1e2)
        self.assertAlmostEqual(self.unknowns['heel_moment_ratio'], (wind_fact*5e4)/(1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))

        self.params['number_of_offset_columns'] = 0
        self.mysemi.compute_stability(self.params, self.unknowns)

        I_expect = 150.0
        meta_expect = I_expect/1e4 - static_expect
        self.assertEqual(self.unknowns['buoyancy_to_gravity'], static_expect)
        self.assertEqual(self.unknowns['metacentric_height'], meta_expect)
        self.assertEqual(self.unknowns['offset_force_ratio'], 26.0/1e2)
        self.assertAlmostEqual(self.unknowns['heel_moment_ratio'], (wind_fact*5e4)/(1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))

        self.params['fairlead'] = 1.0
        self.params['mooring_pitch_restoring_force'][:3,-1] = 1.0
        self.assertAlmostEqual(self.unknowns['heel_moment_ratio'], (wind_fact*5e4)/(1*5 + 1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))


    def testPeriods(self):
        # Spar first
        self.params['structure_center_of_mass'] = np.array([0.0, 0.0, -40.0])
        self.params['number_of_offset_columns'] = 0
        self.mysemi.balance(self.params, self.unknowns)
        self.params['main_center_of_mass'] = self.unknowns['center_of_mass'][-1]
        self.params['main_center_of_buoyancy'] = self.unknowns['center_of_mass'][-1]+2.0
        self.params['tower_mass'] = 0.0
        self.params['rna_mass'] = 0.0
        self.params['rna_I'] = 1e2*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.params['tower_I_base'] = 1e2*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.mysemi.compute_stability(self.params, self.unknowns)
        self.mysemi.compute_rigid_body_periods(self.params, self.unknowns)

        m_struct = self.params['structural_mass']
        m_water  = self.unknowns['variable_ballast_mass']
        z_cg     = self.params['main_center_of_mass']
        z_water  = self.unknowns['variable_ballast_center_of_mass']
        I_water  = self.unknowns['variable_ballast_moments_of_inertia']
        M_expect = np.zeros(6)
        M_expect[:3] = m_struct + m_water
        M_expect[3:] = self.params['main_moments_of_inertia'][:3]
        M_expect[3:5] += I_water[:2] + m_water*(z_water-z_cg)**2
        M_expect[-1] += I_water[2]
        M_expect[3:] += 2e2
        npt.assert_equal(self.unknowns['mass_matrix'], M_expect)

        A_expect = np.zeros(6)
        A_expect[:3] = self.params['main_added_mass'][:3]
        A_expect[3:5] = self.params['main_added_mass'][3:5] + A_expect[0]*(2.0)**2
        npt.assert_equal(self.unknowns['added_mass_matrix'], A_expect)

        rho_w = self.params['water_density']
        K_expect = np.zeros(6)
        K_expect[2] = rho_w * g * 20.0 # waterplane area for main
        K_expect[3:5] = rho_w * g * self.unknowns['metacentric_height'] * 1e4 # Total displacement
        npt.assert_almost_equal(self.unknowns['hydrostatic_stiffness'], K_expect)

        T_expect = 2 * np.pi * np.sqrt( (M_expect + A_expect) / (1e-6 + K_expect + np.diag(self.params['mooring_stiffness'])) )
        npt.assert_almost_equal(self.unknowns['rigid_body_periods'], T_expect)

        
    def testMargins(self):
        
        self.mysemi.balance(self.params, self.unknowns)
        self.mysemi.compute_stability(self.params, self.unknowns)
        self.mysemi.compute_rigid_body_periods(self.params, self.unknowns)
        self.mysemi.check_frequency_margins(self.params, self.unknowns)

        myones = np.ones(6)
        T_sys    = self.unknowns['rigid_body_periods']
        T_wave_low  = self.params['wave_period_range_low']*myones
        T_wave_high = self.params['wave_period_range_high']*myones
        f_struct    = self.params['structural_frequencies']
        T_struct    = 1.0 / f_struct

        T_wave_high[-1] = 1e-16
        T_wave_low[-1] = 1e30

        ind = T_sys>T_wave_low
        npt.assert_equal(self.unknowns['period_margin_high'][ind], T_sys[ind]/T_wave_high[ind] )

        ind = T_sys<T_wave_high
        npt.assert_equal(self.unknowns['period_margin_low'][ind], T_sys[ind]/T_wave_low[ind] )

        ind = T_struct>T_wave_low
        npt.assert_equal(self.unknowns['modal_margin_high'][ind], T_struct[ind]/T_wave_high[ind] )

        ind = T_struct<T_wave_high
        npt.assert_equal(self.unknowns['modal_margin_low'][ind], T_struct[ind]/T_wave_low[ind] )
        
    def testCost(self):
        self.mysemi.compute_costs(self.params, self.unknowns)
        c_expect = 256.0 + 512.0 + 32.0 + 3*64.0
        self.assertEqual(self.unknowns['total_cost'], c_expect)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSubs))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
