import numpy as np
import numpy.testing as npt
import unittest
import wisdem.towerse.tower as tow
import openmdao.api as om
from wisdem.commonse import gravity as g
import copy

class TestTowerSE(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        
    def testMonopileFoundation(self):
        # Test Land
        self.inputs['tower_section_height'] = 2.*np.ones(2)
        self.inputs['tower_outer_diameter'] = 3.*np.ones(3)
        self.inputs['tower_wall_thickness'] = 0.1*np.ones(2)
        self.inputs['suctionpile_depth'] = 0.0
        self.inputs['foundation_height'] = 0.0
        myobj = tow.MonopileFoundation(nPoints=3, monopile=False)
        myobj.compute(self.inputs, self.outputs)
        npt.assert_equal(self.outputs['section_height_out'], self.inputs['tower_section_height'])
        npt.assert_equal(self.outputs['outer_diameter_out'], self.inputs['tower_outer_diameter'])
        npt.assert_equal(self.outputs['wall_thickness_out'], self.inputs['tower_wall_thickness'])
        npt.assert_equal(self.outputs['foundation_height_out'], self.inputs['foundation_height'])

        # Test Land with bad suctionpile input
        self.inputs['suctionpile_depth'] = 10.0
        myobj.compute(self.inputs, self.outputs)
        npt.assert_equal(self.outputs['section_height_out'], self.inputs['tower_section_height'])
        npt.assert_equal(self.outputs['outer_diameter_out'], self.inputs['tower_outer_diameter'])
        npt.assert_equal(self.outputs['wall_thickness_out'], self.inputs['tower_wall_thickness'])
        npt.assert_equal(self.outputs['foundation_height_out'], self.inputs['foundation_height'])
        
        # Test monopile with pile
        self.inputs['suctionpile_depth'] = 10.0
        self.inputs['foundation_height'] = -30.0
        myobj = tow.MonopileFoundation(nPoints=3, monopile=True)
        myobj.compute(self.inputs, self.outputs)
        npt.assert_equal(self.outputs['section_height_out'], np.array([10., 2., 2.]))
        npt.assert_equal(self.outputs['outer_diameter_out'], 3.*np.ones(4))
        npt.assert_equal(self.outputs['wall_thickness_out'], 0.1*np.ones(3))
        npt.assert_equal(self.outputs['foundation_height_out'], self.inputs['foundation_height']-self.inputs['suctionpile_depth'])
        
        # Test monopile with gravity
        self.inputs['suctionpile_depth'] = 0.0
        myobj = tow.MonopileFoundation(nPoints=3, monopile=True)
        myobj.compute(self.inputs, self.outputs)
        npt.assert_equal(self.outputs['section_height_out'], np.array([0.1, 2., 2.]))
        npt.assert_equal(self.outputs['outer_diameter_out'], 3.*np.ones(4))
        npt.assert_equal(self.outputs['wall_thickness_out'], 0.1*np.ones(3))
        npt.assert_equal(self.outputs['foundation_height_out'], self.inputs['foundation_height']-0.1)

        
    def testTowerDisc(self):
        # Test Land
        self.inputs['hub_height'] = 100.0
        self.inputs['z_param'] = np.array([0., 40., 80.])
        myobj = tow.TowerDiscretization(nPoints=3)
        myobj.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs['height_constraint'], 20.0)
        
        # Test monopile 
        self.inputs['z_param'] = np.array([-50., -30, 0.0, 40., 80.])
        myobj = tow.TowerDiscretization(nPoints=5)
        myobj.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs['height_constraint'], 20.0)

        
    def testTowerMass(self):

        self.inputs['z_full'] = np.array([-50., -30, 0.0, 40., 80.])
        self.inputs['cylinder_mass'] = 1e3*np.ones(4)
        self.inputs['cylinder_cost'] = 1e5
        self.inputs['cylinder_center_of_mass'] = 10.0
        self.inputs['cylinder_section_center_of_mass'] = self.inputs['z_full'][:-1] + 0.5*np.diff(self.inputs['z_full'])
        self.inputs['cylinder_I_base'] = 1e4*np.r_[np.ones(3), np.zeros(3)]
        self.inputs['transition_piece_height'] = 20.0
        self.inputs['transition_piece_mass'] = 1e2
        self.inputs['gravity_foundation_mass'] = 1e2
        self.inputs['foundation_height'] = -30.
        
        myobj = tow.TowerMass(nFull=5)
        myobj.compute(self.inputs, self.outputs)
        
        self.assertEqual(self.outputs['tower_raw_cost'], self.inputs['cylinder_cost'])
        npt.assert_equal(self.outputs['tower_I_base'], self.inputs['cylinder_I_base'])
        self.assertEqual(self.outputs['tower_center_of_mass'], (4*1e3*10.0 + 1e2*20.0 + 1e2*-30.0)/(4*1e3+2e2) )
        npt.assert_equal(self.outputs['tower_section_center_of_mass'], self.inputs['cylinder_section_center_of_mass'])
        self.assertEqual(self.outputs['monopile_mass'], 1e3*2.5 + 2*1e2)
        self.assertEqual(self.outputs['monopile_cost'], self.inputs['cylinder_cost']*2.5/4.0)
        self.assertEqual(self.outputs['monopile_length'], 70.0)
        self.assertEqual(self.outputs['tower_mass'], 1e3*(4-2.5))


    def testPreFrame(self):
        
        # Test Land 
        self.inputs['z'] = 10. * np.arange(0,7)
        self.inputs['d'] = 6. * np.ones(self.inputs['z'].shape)
        self.inputs['mass'] = 1e5
        self.inputs['mI']   = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        self.inputs['mrho'] = np.array([-3., 0.0, 1.0])
        self.inputs['transition_piece_mass'] = 0.0
        self.inputs['transition_piece_height'] = 0.0
        self.inputs['gravity_foundation_mass'] = 0.0
        self.inputs['foundation_height'] = 0.0
        self.inputs['rna_F'] = 1e5*np.array([2., 3., 4.,])
        self.inputs['rna_M'] = 1e6*np.array([2., 3., 4.,])
        self.inputs['k_monopile'] = np.zeros(6)

        myobj = tow.TowerPreFrame(nPoints=3, nFull=7, monopile=False)
        myobj.compute(self.inputs, self.outputs)

        npt.assert_equal(self.outputs['kidx'], np.array([0]))
        npt.assert_equal(self.outputs['kx'], np.array([1e16]))
        npt.assert_equal(self.outputs['ky'], np.array([1e16]))
        npt.assert_equal(self.outputs['kz'], np.array([1e16]))
        npt.assert_equal(self.outputs['ktx'], np.array([1e16]))
        npt.assert_equal(self.outputs['kty'], np.array([1e16]))
        npt.assert_equal(self.outputs['ktz'], np.array([1e16]))

        npt.assert_equal(self.outputs['midx'], np.array([6, 0, 0]))
        npt.assert_equal(self.outputs['m'], np.array([1e5, 0, 0]))
        npt.assert_equal(self.outputs['mrhox'], np.array([-3., 0., 0.]))
        npt.assert_equal(self.outputs['mrhoy'], np.array([0., 0., 0.]))
        npt.assert_equal(self.outputs['mrhoz'], np.array([1., 0., 0.]))
        npt.assert_equal(self.outputs['mIxx'], np.array([1e5, 0., 0.]))
        npt.assert_equal(self.outputs['mIyy'], np.array([1e5, 0., 0.]))
        npt.assert_equal(self.outputs['mIzz'], np.array([2e5, 0., 0.]))
        npt.assert_equal(self.outputs['mIxy'], np.zeros(3))
        npt.assert_equal(self.outputs['mIxz'], np.zeros(3))
        npt.assert_equal(self.outputs['mIyz'], np.zeros(3))

        npt.assert_equal(self.outputs['plidx'], np.array([6]))
        npt.assert_equal(self.outputs['Fx'], np.array([2e5]))
        npt.assert_equal(self.outputs['Fy'], np.array([3e5]))
        npt.assert_equal(self.outputs['Fz'], np.array([4e5]))
        npt.assert_equal(self.outputs['Mxx'], np.array([2e6]))
        npt.assert_equal(self.outputs['Myy'], np.array([3e6]))
        npt.assert_equal(self.outputs['Mzz'], np.array([4e6]))

        # Test Monopile 
        self.inputs['z'] = 10. * np.arange(-6,7)
        self.inputs['d'] = 6. * np.ones(self.inputs['z'].shape)
        self.inputs['transition_piece_mass'] = 1e3
        self.inputs['transition_piece_height'] = 10.0
        self.inputs['gravity_foundation_mass'] = 1e4
        self.inputs['foundation_height'] = -30.0
        self.inputs['rna_F'] = 1e5*np.array([2., 3., 4.,])
        self.inputs['rna_M'] = 1e6*np.array([2., 3., 4.,])
        self.inputs['k_monopile'] = 20. + np.arange(6)

        myobj = tow.TowerPreFrame(nPoints=5, nFull=13, monopile=True)
        myobj.compute(self.inputs, self.outputs)

        npt.assert_equal(self.outputs['kidx'], np.array([0, 1, 2, 3]))
        npt.assert_equal(self.outputs['kx'], 20.*np.ones(4))
        npt.assert_equal(self.outputs['ky'], 22.*np.ones(4))
        npt.assert_equal(self.outputs['kz'], 24.*np.ones(4))
        npt.assert_equal(self.outputs['ktx'], 21.*np.ones(4))
        npt.assert_equal(self.outputs['kty'], 23.*np.ones(4))
        npt.assert_equal(self.outputs['ktz'], 25.*np.ones(4))

        npt.assert_equal(self.outputs['midx'], np.array([12, 7, 0]))
        npt.assert_equal(self.outputs['m'], np.array([1e5, 1e3, 1e4]))
        npt.assert_equal(self.outputs['mrhox'], np.array([-3., 0., 0.]))
        npt.assert_equal(self.outputs['mrhoy'], np.array([0., 0., 0.]))
        npt.assert_equal(self.outputs['mrhoz'], np.array([1., 0., 0.]))
        npt.assert_equal(self.outputs['mIxx'], np.array([1e5, 1e3*9*0.5, 1e4*9*0.25]))
        npt.assert_equal(self.outputs['mIyy'], np.array([1e5, 1e3*9*0.5, 1e4*9*0.25]))
        npt.assert_equal(self.outputs['mIzz'], np.array([2e5, 1e3*9, 1e4*9*0.5]))
        npt.assert_equal(self.outputs['mIxy'], np.zeros(3))
        npt.assert_equal(self.outputs['mIxz'], np.zeros(3))
        npt.assert_equal(self.outputs['mIyz'], np.zeros(3))

        npt.assert_equal(self.outputs['plidx'], np.array([12]))
        npt.assert_equal(self.outputs['Fx'], np.array([2e5]))
        npt.assert_equal(self.outputs['Fy'], np.array([3e5]))
        npt.assert_equal(self.outputs['Fz'], np.array([4e5]))
        npt.assert_equal(self.outputs['Mxx'], np.array([2e6]))
        npt.assert_equal(self.outputs['Myy'], np.array([3e6]))
        npt.assert_equal(self.outputs['Mzz'], np.array([4e6]))


    def testProblemLand(self):
        prob = om.Problem()
        prob.model = tow.TowerSE(nLC=1, nPoints=3, nFull=7, wind='PowerWind', topLevelFlag=True, monopile=False)
        prob.setup()

        prob['shearExp'] = 0.2
        prob['hub_height'] = 80.0
        prob['foundation_height'] = 0.0
        prob['transition_piece_height'] = 0.0
        prob['transition_piece_mass'] = 0.0
        prob['gravity_foundation_mass'] = 0.0
        prob['tower_section_height'] = 40.0*np.ones(2)
        prob['tower_outer_diameter'] = 10.0*np.ones(3)
        prob['tower_wall_thickness'] = 0.1*np.ones(2)
        prob['tower_buckling_length'] = 20.0
        prob['tower_outfitting_factor'] = 1.0
        prob['yaw'] = 0.0
        prob['suctionpile_depth'] = 0.0
        prob['soil_G'] = 1e7
        prob['soil_nu'] = 0.5
        prob['E'] = 1e9
        prob['G'] = 1e8
        prob['material_density'] = 1e4
        prob['sigma_y'] = 1e8
        prob['rna_mass'] = 2e5
        prob['rna_I'] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob['rna_cg'] = np.array([-3., 0.0, 1.0])
        prob['wind_reference_height'] = 80.0
        prob['wind_z0'] = 0.0
        prob['cd_usr'] = -1.
        prob['air_density'] = 1.225
        prob['air_viscosity'] = 1.7934e-5
        prob['water_density'] = 1025.0
        prob['water_viscosity'] = 1.3351e-3
        prob['wind_beta'] = prob['wave_beta'] = 0.0
        prob['significant_wave_height'] = 0.0
        prob['significant_wave_period'] = 1e3
        prob['gamma_f'] = 1.0
        prob['gamma_m'] = 1.0
        prob['gamma_n'] = 1.0
        prob['gamma_b'] = 1.0
        prob['gamma_fatigue'] = 1.0
        prob['DC'] = 80.0
        prob['shear'] = True
        prob['geom'] = True
        prob['tower_force_discretization'] = 5.0
        prob['nM'] = 2
        prob['Mmethod'] = 1
        prob['lump'] = 0
        prob['tol'] = 1e-9
        prob['shift'] = 0.0
        prob['min_d_to_t'] = 120.0
        prob['max_taper'] = 0.2
        prob['wind.Uref'] = 15.0
        prob['pre.rna_F'] = 1e3*np.array([2., 3., 4.,])
        prob['pre.rna_M'] = 1e4*np.array([2., 3., 4.,])
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4*(5.**2-4.9**2)*np.pi
        npt.assert_equal(prob['section_height_out'], prob['tower_section_height'])
        npt.assert_equal(prob['outer_diameter_out'], prob['tower_outer_diameter'])
        npt.assert_equal(prob['wall_thickness_out'], prob['tower_wall_thickness'])
        npt.assert_equal(prob['z_param'], np.array([0., 40., 80.]))
        
        self.assertEqual(prob['height_constraint'], 0.0)
        self.assertEqual(prob['tower_raw_cost'], prob['cm.cost'])
        npt.assert_equal(prob['tower_I_base'], prob['cm.I_base'])
        npt.assert_almost_equal(prob['tower_center_of_mass'], 40.0)
        npt.assert_equal(prob['tower_section_center_of_mass'], prob['cm.section_center_of_mass'])
        self.assertEqual(prob['monopile_mass'], 0.0)
        self.assertEqual(prob['monopile_cost'], 0.0)
        self.assertEqual(prob['monopile_length'], 0.0)
        npt.assert_almost_equal(prob['tower_mass'], mass_dens*80.0)

        npt.assert_equal(prob['pre.kidx'], np.array([0], dtype=np.int_))
        npt.assert_equal(prob['pre.kx'], np.array([1e16]))
        npt.assert_equal(prob['pre.ky'], np.array([1e16]))
        npt.assert_equal(prob['pre.kz'], np.array([1e16]))
        npt.assert_equal(prob['pre.ktx'], np.array([1e16]))
        npt.assert_equal(prob['pre.kty'], np.array([1e16]))
        npt.assert_equal(prob['pre.ktz'], np.array([1e16]))

        npt.assert_equal(prob['pre.midx'], np.array([6, 0, 0]))
        npt.assert_equal(prob['pre.m'], np.array([2e5, 0, 0]))
        npt.assert_equal(prob['pre.mrhox'], np.array([-3., 0., 0.]))
        npt.assert_equal(prob['pre.mrhoy'], np.array([0., 0., 0.]))
        npt.assert_equal(prob['pre.mrhoz'], np.array([1., 0., 0.]))
        npt.assert_equal(prob['pre.mIxx'], np.array([1e5, 0., 0.]))
        npt.assert_equal(prob['pre.mIyy'], np.array([1e5, 0., 0.]))
        npt.assert_equal(prob['pre.mIzz'], np.array([2e5, 0., 0.]))
        npt.assert_equal(prob['pre.mIxy'], np.zeros(3))
        npt.assert_equal(prob['pre.mIxz'], np.zeros(3))
        npt.assert_equal(prob['pre.mIyz'], np.zeros(3))

        npt.assert_equal(prob['pre.plidx'], np.array([6]))
        npt.assert_equal(prob['pre.Fx'], np.array([2e3]))
        npt.assert_equal(prob['pre.Fy'], np.array([3e3]))
        npt.assert_equal(prob['pre.Fz'], np.array([4e3]))
        npt.assert_equal(prob['pre.Mxx'], np.array([2e4]))
        npt.assert_equal(prob['pre.Myy'], np.array([3e4]))
        npt.assert_equal(prob['pre.Mzz'], np.array([4e4]))



    def testProblemFixedPile(self):
        prob = om.Problem()
        prob.model = tow.TowerSE(nLC=1, nPoints=4, nFull=10, wind='PowerWind', topLevelFlag=True, monopile=True)
        prob.setup()

        prob['shearExp'] = 0.2
        prob['hub_height'] = 80.0
        prob['foundation_height'] = -30.0
        prob['transition_piece_height'] = 15.0
        prob['transition_piece_mass'] = 1e2
        prob['gravity_foundation_mass'] = 1e4
        prob['tower_section_height'] = 30.0*np.ones(3)
        prob['tower_outer_diameter'] = 10.0*np.ones(4)
        prob['tower_wall_thickness'] = 0.1*np.ones(3)
        prob['tower_buckling_length'] = 20.0
        prob['tower_outfitting_factor'] = 1.0
        prob['yaw'] = 0.0
        prob['suctionpile_depth'] = 15.0
        prob['soil_G'] = 1e7
        prob['soil_nu'] = 0.5
        prob['E'] = 1e9
        prob['G'] = 1e8
        prob['material_density'] = 1e4
        prob['sigma_y'] = 1e8
        prob['rna_mass'] = 2e5
        prob['rna_I'] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob['rna_cg'] = np.array([-3., 0.0, 1.0])
        prob['wind_reference_height'] = 80.0
        prob['wind_z0'] = 0.0
        prob['cd_usr'] = -1.
        prob['air_density'] = 1.225
        prob['air_viscosity'] = 1.7934e-5
        prob['water_density'] = 1025.0
        prob['water_viscosity'] = 1.3351e-3
        prob['wind_beta'] = prob['wave_beta'] = 0.0
        prob['significant_wave_height'] = 0.0
        prob['significant_wave_period'] = 1e3
        prob['gamma_f'] = 1.0
        prob['gamma_m'] = 1.0
        prob['gamma_n'] = 1.0
        prob['gamma_b'] = 1.0
        prob['gamma_fatigue'] = 1.0
        prob['DC'] = 80.0
        prob['shear'] = True
        prob['geom'] = True
        prob['tower_force_discretization'] = 5.0
        prob['nM'] = 2
        prob['Mmethod'] = 1
        prob['lump'] = 0
        prob['tol'] = 1e-9
        prob['shift'] = 0.0
        prob['min_d_to_t'] = 120.0
        prob['max_taper'] = 0.2
        prob['wind.Uref'] = 15.0
        prob['pre.rna_F'] = 1e3*np.array([2., 3., 4.,])
        prob['pre.rna_M'] = 1e4*np.array([2., 3., 4.,])
        prob.run_model()

        # All other tests from above
        mass_dens = 1e4*(5.**2-4.9**2)*np.pi
        npt.assert_equal(prob['section_height_out'], np.r_[15., 30.*np.ones(3)])
        npt.assert_equal(prob['outer_diameter_out'], 10.*np.ones(5))
        npt.assert_equal(prob['wall_thickness_out'], 0.1*np.ones(4))
        npt.assert_equal(prob['z_param'], np.array([-45., -30., 0., 30., 60.]))
        
        self.assertEqual(prob['height_constraint'], 20.0)
        self.assertEqual(prob['tower_raw_cost'], (40./105.)*prob['cm.cost'])
        npt.assert_equal(prob['tower_I_base'], prob['cm.I_base'])
        npt.assert_almost_equal(prob['tower_center_of_mass'], (7.5*mass_dens*105.+15.*1e2+1e4*-30.)/(mass_dens*105+1e2+1e4))
        npt.assert_equal(prob['tower_section_center_of_mass'], prob['cm.section_center_of_mass'])
        self.assertEqual(prob['monopile_cost'], (60./105.)*prob['cm.cost'])
        self.assertEqual(prob['monopile_length'], 60.0)
        npt.assert_almost_equal(prob['monopile_mass'], mass_dens*60.0 + 1e2+1e4)
        npt.assert_almost_equal(prob['tower_mass'], mass_dens*45.0)

        npt.assert_equal(prob['pre.kidx'], np.array([0, 1, 2, 3], dtype=np.int_))
        npt.assert_array_less(prob['pre.kx'], 1e16)
        npt.assert_array_less(prob['pre.ky'], 1e16)
        npt.assert_array_less(prob['pre.kz'], 1e16)
        npt.assert_array_less(prob['pre.ktx'], 1e16)
        npt.assert_array_less(prob['pre.kty'], 1e16)
        npt.assert_array_less(prob['pre.ktz'], 1e16)
        npt.assert_array_less(0.0, prob['pre.kx'])
        npt.assert_array_less(0.0, prob['pre.ky'])
        npt.assert_array_less(0.0, prob['pre.kz'])
        npt.assert_array_less(0.0, prob['pre.ktx'])
        npt.assert_array_less(0.0, prob['pre.kty'])
        npt.assert_array_less(0.0, prob['pre.ktz'])

        npt.assert_equal(prob['pre.midx'], np.array([12, 7, 0]))
        npt.assert_equal(prob['pre.m'], np.array([2e5, 1e2, 1e4]))
        npt.assert_equal(prob['pre.mrhox'], np.array([-3., 0., 0.]))
        npt.assert_equal(prob['pre.mrhoy'], np.array([0., 0., 0.]))
        npt.assert_equal(prob['pre.mrhoz'], np.array([1., 0., 0.]))
        npt.assert_equal(prob['pre.mIxx'], np.array([1e5, 1e2*25*0.5, 1e4*25*0.25]))
        npt.assert_equal(prob['pre.mIyy'], np.array([1e5, 1e2*25*0.5, 1e4*25*0.25]))
        npt.assert_equal(prob['pre.mIzz'], np.array([2e5, 1e2*25, 1e4*25*0.5]))
        npt.assert_equal(prob['pre.mIxy'], np.zeros(3))
        npt.assert_equal(prob['pre.mIxz'], np.zeros(3))
        npt.assert_equal(prob['pre.mIyz'], np.zeros(3))

        npt.assert_equal(prob['pre.plidx'], np.array([12]))
        npt.assert_equal(prob['pre.Fx'], np.array([2e3]))
        npt.assert_equal(prob['pre.Fy'], np.array([3e3]))
        npt.assert_equal(prob['pre.Fz'], np.array([4e3]))
        npt.assert_equal(prob['pre.Mxx'], np.array([2e4]))
        npt.assert_equal(prob['pre.Myy'], np.array([3e4]))
        npt.assert_equal(prob['pre.Mzz'], np.array([4e4]))
        
        

    def testAddedMassForces(self):
        prob = om.Problem()
        prob.model = tow.TowerSE(nLC=1, nPoints=4, nFull=10, wind='PowerWind', topLevelFlag=True, monopile=True)
        prob.setup()

        prob['shearExp'] = 0.2
        prob['hub_height'] = 80.0
        prob['foundation_height'] = -30.0
        prob['transition_piece_height'] = 15.0
        prob['transition_piece_mass'] = 0.0
        prob['gravity_foundation_mass'] = 0.0
        prob['tower_section_height'] = 30.0*np.ones(3)
        prob['tower_outer_diameter'] = 10.0*np.ones(4)
        prob['tower_wall_thickness'] = 0.1*np.ones(3)
        prob['tower_buckling_length'] = 20.0
        prob['tower_outfitting_factor'] = 1.0
        prob['yaw'] = 0.0
        prob['suctionpile_depth'] = 15.0
        prob['soil_G'] = 1e7
        prob['soil_nu'] = 0.5
        prob['E'] = 1e9
        prob['G'] = 1e8
        prob['material_density'] = 1e4
        prob['sigma_y'] = 1e8
        prob['rna_mass'] = 0.0
        prob['rna_I'] = np.r_[1e5, 1e5, 2e5, np.zeros(3)]
        prob['rna_cg'] = np.array([-3., 0.0, 1.0])
        prob['wind_reference_height'] = 80.0
        prob['wind_z0'] = 0.0
        prob['cd_usr'] = -1.
        prob['air_density'] = 1.225
        prob['air_viscosity'] = 1.7934e-5
        prob['water_density'] = 1025.0
        prob['water_viscosity'] = 1.3351e-3
        prob['wind_beta'] = prob['wave_beta'] = 0.0
        prob['significant_wave_height'] = 0.0
        prob['significant_wave_period'] = 1e3
        prob['gamma_f'] = 1.0
        prob['gamma_m'] = 1.0
        prob['gamma_n'] = 1.0
        prob['gamma_b'] = 1.0
        prob['gamma_fatigue'] = 1.0
        prob['DC'] = 80.0
        prob['shear'] = True
        prob['geom'] = True
        prob['tower_force_discretization'] = 5.0
        prob['nM'] = 2
        prob['Mmethod'] = 1
        prob['lump'] = 0
        prob['tol'] = 1e-9
        prob['shift'] = 0.0
        prob['min_d_to_t'] = 120.0
        prob['max_taper'] = 0.2
        prob['wind.Uref'] = 15.0
        prob['pre.rna_F'] = 1e3*np.array([2., 3., 4.,])
        prob['pre.rna_M'] = 1e4*np.array([2., 3., 4.,])
        prob.run_model()

        myFz = copy.copy(prob['post.Fz'])

        prob['rna_mass'] = 1e4
        prob.run_model()
        myFz[3:] -= 1e4*g
        npt.assert_almost_equal(prob['post.Fz'], myFz)

        prob['transition_piece_mass'] = 1e2
        prob.run_model()
        myFz[3:7] -= 1e2*g
        npt.assert_almost_equal(prob['post.Fz'], myFz)

        prob['gravity_foundation_mass'] = 1e3
        prob.run_model()
        #myFz[3] -= 1e3*g
        npt.assert_almost_equal(prob['post.Fz'], myFz)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTowerSE))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
