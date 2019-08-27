from __future__ import print_function

import numpy as np
import numpy.testing as npt
import unittest
import time
import floatingse.loading as sP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from commonse import gravity as g


NSECTIONS = 5
NPTS = NSECTIONS+1

def DrawTruss(mytruss):
    mynodes = {}
    for k in range(len(mytruss.myframe.nx)):
        mynodes[mytruss.myframe.nnode[k]] = np.r_[mytruss.myframe.nx[k], mytruss.myframe.ny[k], mytruss.myframe.nz[k]]
    myelem = []
    for k in range(len(mytruss.myframe.eN1)):
        myelem.append( (mytruss.myframe.eN1[k], mytruss.myframe.eN2[k]) )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for e in myelem:
        xs = np.array( [ mynodes[e[0]][0], mynodes[e[1]][0] ] )
        ys = np.array( [ mynodes[e[0]][1], mynodes[e[1]][1] ] )
        zs = np.array( [ mynodes[e[0]][2], mynodes[e[1]][2] ] )
        ax.plot(xs, ys, zs)
    #ax.auto_scale_xyz([-10, 10], [-10, 10], [-30, 50])
    plt.show()


def getParams():
    params = {}

    params['E'] = 200e9
    params['G'] = 79.3e9
    params['material_density'] = 7850.0
    params['yield_stress'] = 345e6
    params['Hs'] = 10.0

    params['water_density'] = 1025.0

    params['cross_attachment_pontoons'] = True
    params['lower_attachment_pontoons'] = True
    params['upper_attachment_pontoons'] = True
    params['lower_ring_pontoons'] = True
    params['upper_ring_pontoons'] = True
    params['outer_cross_pontoons'] = True

    params['number_of_offset_columns'] = 3
    params['connection_ratio_max'] = 0.25

    params['number_of_mooring_connections'] = 3
    params['mooring_lines_per_connection'] = 1
    params['mooring_neutral_load'] = 1e2*np.ones((15,3))
    params['mooring_stiffness'] = 1e10 * np.eye(6)
    params['mooring_moments_of_inertia'] = np.ones(6)
    
    params['fairlead'] = 7.5
    params['fairlead_radius'] = 1.0
    params['fairlead_support_outer_diameter'] = 2.0
    params['fairlead_support_wall_thickness'] = 1.0
    
    params['main_pontoon_attach_upper'] = 1.0
    params['main_pontoon_attach_lower'] = 0.0

    params['radius_to_offset_column'] = 15.0
    params['pontoon_outer_diameter'] = 2.0
    params['pontoon_wall_thickness'] = 1.0

    params['main_z_full'] = np.array([-15.0, -12.5, -10.0, 0.0, 5.0, 10.0])
    params['main_d_full'] = 2*10.0 * np.ones(NPTS)
    params['main_t_full'] = 0.1 * np.ones(NPTS-1)
    params['main_mass'] = 1e2 * np.ones(NSECTIONS)
    params['main_buckling_length'] = 2.0 * np.ones(NSECTIONS)
    params['main_displaced_volume'] = 1e2 * np.ones(NSECTIONS)
    params['main_hydrostatic_force'] = np.zeros(NSECTIONS)
    params['main_hydrostatic_force'][0] = params['main_displaced_volume'].sum()*g*params['water_density']
    params['main_center_of_buoyancy'] = -10.0
    params['main_center_of_mass'] = -6.0
    params['main_Px'] = 50.0 * np.ones(NPTS)
    params['main_Py'] = np.zeros(NPTS)
    params['main_Pz'] = np.zeros(NPTS)
    params['main_qdyn'] = 70.0 * np.ones(NPTS)

    params['offset_z_full'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 10.0])
    params['offset_d_full'] = 2*2.0 * np.ones(NPTS)
    params['offset_t_full'] = 0.05 * np.ones(NPTS-1)
    params['offset_mass'] = 1e1 * np.ones(NSECTIONS)
    params['offset_buckling_length'] = 2.0 * np.ones(NSECTIONS)
    params['offset_displaced_volume'] = 1e1 * np.ones(NSECTIONS)
    params['offset_hydrostatic_force'] = np.zeros(NSECTIONS)
    params['offset_hydrostatic_force'][0] = params['main_displaced_volume'].sum()*g*params['water_density']
    params['offset_center_of_buoyancy'] = -5.0
    params['offset_center_of_mass'] = -3.0
    params['offset_Px'] = 50.0 * np.ones(NPTS)
    params['offset_Py'] = np.zeros(NPTS)
    params['offset_Pz'] = np.zeros(NPTS)
    params['offset_qdyn'] = 70.0 * np.ones(NPTS)

    params['tower_z_full'] = np.linspace(0, 90, NPTS)
    params['tower_d_full'] = 2*7.0 * np.ones(NPTS)
    params['tower_t_full'] = 0.5 * np.ones(NPTS-1)
    params['tower_mass_section'] = 2e2 * np.ones(NSECTIONS)
    params['tower_buckling_length'] = 25.0
    params['tower_center_of_mass'] = 50.0
    params['tower_Px'] = 50.0 * np.ones(NPTS)
    params['tower_Py'] = np.zeros(NPTS)
    params['tower_Pz'] = np.zeros(NPTS)
    params['tower_qdyn'] = 70.0 * np.ones(NPTS)

    params['rna_force'] = 6e1*np.ones(3)
    params['rna_moment'] = 7e2*np.ones(3)
    params['rna_mass'] = 6e1
    params['rna_cg'] = np.array([3.05, 2.96, 2.13])
    params['rna_I'] = np.array([3.05284574e9, 2.96031642e9, 2.13639924e7, 0.0, 2.89884849e7, 0.0])

    params['gamma_f'] = 1.35
    params['gamma_m'] = 1.1
    params['gamma_n'] = 1.0
    params['gamma_b'] = 1.1
    params['gamma_fatigue'] = 1.755

    params['material_cost_rate'] = 1.0
    params['painting_cost_rate'] = 10.0
    params['labor_cost_rate'] = 2.0
    return params

    
class TestFrame(unittest.TestCase):
    def setUp(self):
        self.params = getParams()
        self.unknowns = {}
        self.resid = None

        self.unknowns['pontoon_stress'] = np.zeros(70)
        
        self.mytruss = sP.FloatingFrame(NPTS,NPTS)

    def tearDown(self):
        self.mytruss = None

    def testStandard(self):
        self.params['radius_to_offset_column'] = 20.0
        self.params['fairlead_radius'] = 30.0
        self.params['offset_z_full'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 3.0])
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        npt.assert_equal(self.unknowns['main_connection_ratio'], 0.25-0.1)
        npt.assert_equal(self.unknowns['offset_connection_ratio'], 0.25-0.5)
        #DrawTruss(self.mytruss)
        

    def testSpar(self):
        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = False
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
        self.params['number_of_offset_columns'] = 0
        
        #self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        #DrawTruss(self.mytruss)
        
        
    def testOutputsIncremental(self):
        ncyl   = self.params['number_of_offset_columns']
        R_semi = self.params['radius_to_offset_column']
        Rmain  = 0.5*self.params['main_d_full'][0]
        Roff   = 0.5*self.params['offset_d_full'][0]
        Ro     = 0.5*self.params['pontoon_outer_diameter']
        Ri     = Ro - self.params['pontoon_wall_thickness']
        rho    = self.params['material_density']
        rhoW   = self.params['water_density']

        self.params['fairlead_radius'] = 0.0
        self.params['fairlead_outer_diameter'] = 0.0
        self.params['fairlead_wall_thickness'] = 0.0
        self.params['mooring_lines_per_column'] = 0.0
        self.params['number_of_mooring_connections'] = 0.0
        self.params['fairlead_radius'] = 0.0
        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = True
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
        self.params['pontoon_main_attach_upper'] = 1.0
        self.params['pontoon_main_attach_lower'] = 0.0
        self.params['material_cost_rate'] = 6.25
        self.params['painting_cost_rate'] = 0.0
        self.params['labor_cost_rate'] = 0.0
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * (R_semi-Rmain-Roff) * ncyl
        m = np.pi * (Ro*Ro-Ri*Ri) * (R_semi-Rmain-Roff) * ncyl * rho
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)


        self.params['lower_ring_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)


        self.params['upper_attachment_pontoons'] = True # above waterline
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (R_semi * (2 + np.sqrt(3)) - 2*Rmain - 4*Roff)
        #cg = ((-15)*(1 + np.sqrt(3)) + 10) / (2+np.sqrt(3))
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        #self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], cg)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        self.params['upper_ring_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * 2 * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], -2.5)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        #self.params['cross_attachment_pontoons'] = True
        #self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        #L = np.sqrt(R_semi*R_semi + 25*25)
        #k = 15. / 25.
        #V = np.pi * Ro*Ro * ncyl * (k*L + R_semi * (1 + np.sqrt(3)))
        #m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (L + R_semi * 2 * (1 + np.sqrt(3)))
        #self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        #self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        #self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        #self.params['outer_cross_pontoons'] = True
        #self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)


    def testForces(self):
        self.params['offset_z_full'] = np.linspace(-1e-4, 0.0, NPTS)
        self.params['main_z_full'] = np.linspace(-1e-4, 0.0, NPTS)
        self.params['tower_z_full'] = np.linspace(0, 1e-4, NPTS)
        self.params['fairlead'] = 0.0
        self.params['number_of_offset_columns'] = 0
        self.params['main_mass'] = 1.0 * np.ones(NSECTIONS)
        self.params['main_center_of_mass'] = 0.0
        self.params['main_hydrostatic_force'] = 1e-12 * np.ones(NSECTIONS)
        self.params['offset_mass'] = 0.0 * np.ones(NSECTIONS)
        self.params['offset_center_of_mass'] = 0.0
        self.params['offset_hydrostatic_force'] = 1e-12 * np.ones(NSECTIONS)
        self.params['tower_mass_section'] = 1.0 * np.ones(NSECTIONS)
        self.params['tower_center_of_mass'] = 0.0
        self.params['rna_mass'] = 1.0
        self.params['rna_force'] = 10.0*np.ones(3)
        self.params['rna_moment'] = 20.0*np.ones(3)
        self.params['rna_cg'] = np.array([0.0, 0.0, 0.0])
        self.params['rna_I'] = np.zeros(6)
        self.params['main_Px'] = 0.0 * np.ones(NSECTIONS)
        self.params['offset_Px'] = 0.0 * np.ones(NSECTIONS)
        self.params['tower_Px'] = 0.0 * np.ones(NSECTIONS)
        self.params['main_qdyn'] = 0.0 * np.ones(NPTS)
        self.params['offset_qdyn'] = 0.0 * np.ones(NPTS)
        self.params['tower_qdyn'] = 0.0 * np.ones(NPTS)
        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = False
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
        self.params['main_pontoon_attach_upper'] = 1.0
        self.params['main_pontoon_attach_lower'] = 0.0
        self.params['water_density'] = 1e-12
        self.params['number_of_mooring_connections'] = 3
        self.params['mooring_lines_per_connection'] = 1
        self.params['mooring_neutral_load'] = 10.0*np.ones((15,3))
        self.params['mooring_moments_of_inertia'] = np.ones(6)
        self.params['fairlead_radius'] = 0.1
        self.params['fairlead_support_outer_diameter'] = 2*np.sqrt(2.0/np.pi)
        self.params['fairlead_support_wall_thickness'] = np.sqrt(2.0/np.pi) - np.sqrt(1.0/np.pi)
        self.params['material_density'] = 20.0
        self.params['radius_to_offset_column'] = 1.0

        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        msub = NSECTIONS + 3*20.0*1.0*0.1
        mtowrna = NSECTIONS + 1
        mtot = msub + mtowrna
        self.assertAlmostEqual(self.unknowns['structural_mass'], mtot, 4)
        self.assertAlmostEqual(self.unknowns['substructure_mass'], msub, 5)
        npt.assert_almost_equal(self.unknowns['total_force'], 10*3 + np.array([10.0, 10.0, 10-mtot*g]), decimal=1)
        npt.assert_almost_equal(self.unknowns['total_moment'], np.array([20.0, 20.0, 20.0]), decimal=2)

        self.params['rna_cg'] = np.array([5.0, 5.0, 5.0])
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        npt.assert_almost_equal(self.unknowns['total_force'], 3*10 + np.array([10.0, 10.0, 10-mtot*g]), decimal=1)
        #self.assertAlmostEqual(self.unknowns['total_moment'][-1], 20.0)
        
class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.params = getParams()
        self.unknowns = {}
        self.resid = None
        self.unknowns['pontoon_stress'] = np.zeros(70)
        self.mytruss = sP.FloatingFrame(NPTS,NPTS)

    def tearDown(self):
        self.mytruss = None

        
    def testBadInput(self):
        self.params['number_of_offset_columns'] = 1
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)

        self.params['number_of_offset_columns'] = 2
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)

        self.params['number_of_offset_columns'] = 8
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)

        self.params['number_of_offset_columns'] = 3
        self.params['main_z_full'][-2] = self.params['main_z_full'][-3] + 1e-12
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)

    def testCombinations(self):
        self.params['radius_to_offset_column'] = 30.0
        
        for nc in [0, 3, 4, 5, 6, 7]:
            
            for cap in [True, False]:
                
                for lap in [True, False]:
                    
                    for uap in [True, False]:
                        
                        for lrp in [True, False]:
                            
                            for urp in [True, False]:
                                
                                for ocp in [True, False]:
                                    self.params['number_of_offset_columns'] = nc
                                    self.params['number_of_mooring_connections'] = np.maximum(nc, 3)
                                    self.params['mooring_lines_per_connection'] = 1
                                    self.params['cross_attachment_pontoons'] = cap
                                    self.params['lower_attachment_pontoons'] = lap
                                    self.params['upper_attachment_pontoons'] = uap
                                    self.params['lower_ring_pontoons'] = lrp
                                    self.params['upper_ring_pontoons'] = urp
                                    self.params['outer_cross_pontoons'] = ocp
                                    if (nc > 0) and (not cap) and (not lap) and (not uap): continue
                                    if (nc > 0) and (ocp) and (not lrp): continue
                                    
                                    self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
                                    if self.unknowns['substructure_mass'] == 1e30:
                                        print(nc, cap, lap, uap, lrp, urp, ocp)
                                    self.assertNotEqual(self.unknowns['substructure_mass'], 1e30)
                                    time.sleep(1e-3)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFrame))
    suite.addTest(unittest.makeSuite(TestSandbox))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

