from __future__ import print_function

import numpy as np
import numpy.testing as npt
import unittest
import time
import wisdem.floatingse.loading as sP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from wisdem.commonse import gravity as g


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


def getInputs():
    inputs = {}
    discrete_inputs = {}

    inputs['E'] = 200e9
    inputs['G'] = 79.3e9
    inputs['material_density'] = 7850.0
    inputs['yield_stress'] = 345e6
    inputs['Hs'] = 10.0

    inputs['water_density'] = 1025.0

    discrete_inputs['cross_attachment_pontoons'] = True
    discrete_inputs['lower_attachment_pontoons'] = True
    discrete_inputs['upper_attachment_pontoons'] = True
    discrete_inputs['lower_ring_pontoons'] = True
    discrete_inputs['upper_ring_pontoons'] = True
    discrete_inputs['outer_cross_pontoons'] = True

    inputs['number_of_offset_columns'] = 3
    inputs['connection_ratio_max'] = 0.25

    inputs['number_of_mooring_connections'] = 3
    inputs['mooring_lines_per_connection'] = 1
    inputs['mooring_neutral_load'] = 1e2*np.ones((15,3))
    inputs['mooring_stiffness'] = 1e10 * np.eye(6)
    inputs['mooring_moments_of_inertia'] = np.ones(6)
    
    inputs['fairlead'] = 7.5
    inputs['fairlead_radius'] = 1.0
    inputs['fairlead_support_outer_diameter'] = 2.0
    inputs['fairlead_support_wall_thickness'] = 1.0
    
    inputs['main_pontoon_attach_upper'] = 1.0
    inputs['main_pontoon_attach_lower'] = 0.0

    inputs['radius_to_offset_column'] = 15.0
    inputs['pontoon_outer_diameter'] = 2.0
    inputs['pontoon_wall_thickness'] = 1.0

    inputs['main_z_full'] = np.array([-15.0, -12.5, -10.0, 0.0, 5.0, 10.0])
    inputs['main_d_full'] = 2*10.0 * np.ones(NPTS)
    inputs['main_t_full'] = 0.1 * np.ones(NPTS-1)
    inputs['main_mass'] = 1e2 * np.ones(NSECTIONS)
    inputs['main_buckling_length'] = 2.0 * np.ones(NSECTIONS)
    inputs['main_displaced_volume'] = 1e2 * np.ones(NSECTIONS)
    inputs['main_hydrostatic_force'] = np.zeros(NSECTIONS)
    inputs['main_hydrostatic_force'][0] = inputs['main_displaced_volume'].sum()*g*inputs['water_density']
    inputs['main_center_of_buoyancy'] = -10.0
    inputs['main_center_of_mass'] = -6.0
    inputs['main_Px'] = 50.0 * np.ones(NPTS)
    inputs['main_Py'] = np.zeros(NPTS)
    inputs['main_Pz'] = np.zeros(NPTS)
    inputs['main_qdyn'] = 70.0 * np.ones(NPTS)

    inputs['offset_z_full'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 10.0])
    inputs['offset_d_full'] = 2*2.0 * np.ones(NPTS)
    inputs['offset_t_full'] = 0.05 * np.ones(NPTS-1)
    inputs['offset_mass'] = 1e1 * np.ones(NSECTIONS)
    inputs['offset_buckling_length'] = 2.0 * np.ones(NSECTIONS)
    inputs['offset_displaced_volume'] = 1e1 * np.ones(NSECTIONS)
    inputs['offset_hydrostatic_force'] = np.zeros(NSECTIONS)
    inputs['offset_hydrostatic_force'][0] = inputs['main_displaced_volume'].sum()*g*inputs['water_density']
    inputs['offset_center_of_buoyancy'] = -5.0
    inputs['offset_center_of_mass'] = -3.0
    inputs['offset_Px'] = 50.0 * np.ones(NPTS)
    inputs['offset_Py'] = np.zeros(NPTS)
    inputs['offset_Pz'] = np.zeros(NPTS)
    inputs['offset_qdyn'] = 70.0 * np.ones(NPTS)

    inputs['tower_z_full'] = np.linspace(0, 90, NPTS)
    inputs['tower_d_full'] = 2*7.0 * np.ones(NPTS)
    inputs['tower_t_full'] = 0.5 * np.ones(NPTS-1)
    inputs['tower_mass_section'] = 2e2 * np.ones(NSECTIONS)
    inputs['tower_buckling_length'] = 25.0
    inputs['tower_center_of_mass'] = 50.0
    inputs['tower_Px'] = 50.0 * np.ones(NPTS)
    inputs['tower_Py'] = np.zeros(NPTS)
    inputs['tower_Pz'] = np.zeros(NPTS)
    inputs['tower_qdyn'] = 70.0 * np.ones(NPTS)

    inputs['rna_force'] = 6e1*np.ones(3)
    inputs['rna_moment'] = 7e2*np.ones(3)
    inputs['rna_mass'] = 6e1
    inputs['rna_cg'] = np.array([3.05, 2.96, 2.13])
    inputs['rna_I'] = np.array([3.05284574e9, 2.96031642e9, 2.13639924e7, 0.0, 2.89884849e7, 0.0])

    inputs['gamma_f'] = 1.35
    inputs['gamma_m'] = 1.1
    inputs['gamma_n'] = 1.0
    inputs['gamma_b'] = 1.1
    inputs['gamma_fatigue'] = 1.755

    inputs['material_cost_rate'] = 1.0
    inputs['painting_cost_rate'] = 10.0
    inputs['labor_cost_rate'] = 2.0
    return inputs, discrete_inputs

    
class TestFrame(unittest.TestCase):
    def setUp(self):
        self.inputs, self.discrete_inputs = getInputs()
        self.outputs = {}
        self.discrete_outputs = {}

        self.outputs['pontoon_stress'] = np.zeros(70)
        
        self.mytruss = sP.FloatingFrame(nFull=NPTS,nFullTow=NPTS)

    def tearDown(self):
        self.mytruss = None

    def testStandard(self):
        self.inputs['radius_to_offset_column'] = 20.0
        self.inputs['fairlead_radius'] = 30.0
        self.inputs['offset_z_full'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 3.0])
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs['main_connection_ratio'], 0.25-0.1)
        npt.assert_equal(self.outputs['offset_connection_ratio'], 0.25-0.5)
        #DrawTruss(self.mytruss)
        

    def testSpar(self):
        self.inputs['cross_attachment_pontoons'] = False
        self.inputs['lower_attachment_pontoons'] = False
        self.inputs['upper_attachment_pontoons'] = False
        self.inputs['lower_ring_pontoons'] = False
        self.inputs['upper_ring_pontoons'] = False
        self.inputs['outer_cross_pontoons'] = False
        self.inputs['number_of_offset_columns'] = 0
        
        #self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        #DrawTruss(self.mytruss)
        
        
    def testOutputsIncremental(self):
        ncyl   = self.inputs['number_of_offset_columns']
        R_semi = self.inputs['radius_to_offset_column']
        Rmain  = 0.5*self.inputs['main_d_full'][0]
        Roff   = 0.5*self.inputs['offset_d_full'][0]
        Ro     = 0.5*self.inputs['pontoon_outer_diameter']
        Ri     = Ro - self.inputs['pontoon_wall_thickness']
        rho    = self.inputs['material_density']
        rhoW   = self.inputs['water_density']

        self.inputs['fairlead_radius'] = 0.0
        self.inputs['fairlead_outer_diameter'] = 0.0
        self.inputs['fairlead_wall_thickness'] = 0.0
        self.inputs['mooring_lines_per_column'] = 0.0
        self.inputs['number_of_mooring_connections'] = 0.0
        self.discrete_inputs['cross_attachment_pontoons'] = False
        self.discrete_inputs['lower_attachment_pontoons'] = True
        self.discrete_inputs['upper_attachment_pontoons'] = False
        self.discrete_inputs['lower_ring_pontoons'] = False
        self.discrete_inputs['upper_ring_pontoons'] = False
        self.discrete_inputs['outer_cross_pontoons'] = False
        self.inputs['pontoon_main_attach_upper'] = 1.0
        self.inputs['pontoon_main_attach_lower'] = 0.0
        self.inputs['material_cost_rate'] = 6.25
        self.inputs['painting_cost_rate'] = 0.0
        self.inputs['labor_cost_rate'] = 0.0

        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        V = np.pi * Ro*Ro * (R_semi-Rmain-Roff) * ncyl
        m = np.pi * (Ro*Ro-Ri*Ri) * (R_semi-Rmain-Roff) * ncyl * rho
        self.assertAlmostEqual(self.outputs['pontoon_displacement'], V)
        self.assertAlmostEqual(self.outputs['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.outputs['pontoon_center_of_mass'], -15.0)
        self.assertAlmostEqual(self.outputs['pontoon_mass'], m)
        self.assertAlmostEqual(self.outputs['pontoon_cost'], m*6.25, 2)


        self.discrete_inputs['lower_ring_pontoons'] = True
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        V = np.pi * Ro*Ro * ncyl * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        self.assertAlmostEqual(self.outputs['pontoon_displacement'], V)
        self.assertAlmostEqual(self.outputs['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.outputs['pontoon_center_of_mass'], -15.0)
        self.assertAlmostEqual(self.outputs['pontoon_mass'], m)
        self.assertAlmostEqual(self.outputs['pontoon_cost'], m*6.25, 2)


        self.discrete_inputs['upper_attachment_pontoons'] = True # above waterline
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        V = np.pi * Ro*Ro * ncyl * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (R_semi * (2 + np.sqrt(3)) - 2*Rmain - 4*Roff)
        #cg = ((-15)*(1 + np.sqrt(3)) + 10) / (2+np.sqrt(3))
        self.assertAlmostEqual(self.outputs['pontoon_displacement'], V)
        self.assertAlmostEqual(self.outputs['pontoon_center_of_buoyancy'], -15.0)
        #self.assertAlmostEqual(self.outputs['pontoon_center_of_mass'], cg)
        self.assertAlmostEqual(self.outputs['pontoon_mass'], m)
        self.assertAlmostEqual(self.outputs['pontoon_cost'], m*6.25, 2)

        
        self.discrete_inputs['upper_ring_pontoons'] = True
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        V = np.pi * Ro*Ro * ncyl * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * 2 * (R_semi * (1 + np.sqrt(3)) - Rmain - 3*Roff)
        self.assertAlmostEqual(self.outputs['pontoon_displacement'], V)
        self.assertAlmostEqual(self.outputs['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.outputs['pontoon_center_of_mass'], -2.5)
        self.assertAlmostEqual(self.outputs['pontoon_mass'], m)
        self.assertAlmostEqual(self.outputs['pontoon_cost'], m*6.25, 2)

        
        #self.inputs['cross_attachment_pontoons'] = True
        #self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        #L = np.sqrt(R_semi*R_semi + 25*25)
        #k = 15. / 25.
        #V = np.pi * Ro*Ro * ncyl * (k*L + R_semi * (1 + np.sqrt(3)))
        #m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (L + R_semi * 2 * (1 + np.sqrt(3)))
        #self.assertAlmostEqual(self.outputs['pontoon_displacement'], V)
        #self.assertAlmostEqual(self.outputs['pontoon_mass'], m)
        #self.assertAlmostEqual(self.outputs['pontoon_cost'], m*6.25, 2)

        
        #self.inputs['outer_cross_pontoons'] = True
        #self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)


    def testForces(self):
        self.inputs['offset_z_full'] = np.linspace(-1e-4, 0.0, NPTS)
        self.inputs['main_z_full'] = np.linspace(-1e-4, 0.0, NPTS)
        self.inputs['tower_z_full'] = np.linspace(0, 1e-4, NPTS)
        self.inputs['fairlead'] = 0.0
        self.inputs['number_of_offset_columns'] = 0
        self.inputs['main_mass'] = 1.0 * np.ones(NSECTIONS)
        self.inputs['main_center_of_mass'] = 0.0
        self.inputs['main_hydrostatic_force'] = 1e-12 * np.ones(NSECTIONS)
        self.inputs['offset_mass'] = 0.0 * np.ones(NSECTIONS)
        self.inputs['offset_center_of_mass'] = 0.0
        self.inputs['offset_hydrostatic_force'] = 1e-12 * np.ones(NSECTIONS)
        self.inputs['tower_mass_section'] = 1.0 * np.ones(NSECTIONS)
        self.inputs['tower_center_of_mass'] = 0.0
        self.inputs['rna_mass'] = 1.0
        self.inputs['rna_force'] = 10.0*np.ones(3)
        self.inputs['rna_moment'] = 20.0*np.ones(3)
        self.inputs['rna_cg'] = np.array([0.0, 0.0, 0.0])
        self.inputs['rna_I'] = np.zeros(6)
        self.inputs['main_Px'] = 0.0 * np.ones(NSECTIONS)
        self.inputs['offset_Px'] = 0.0 * np.ones(NSECTIONS)
        self.inputs['tower_Px'] = 0.0 * np.ones(NSECTIONS)
        self.inputs['main_qdyn'] = 0.0 * np.ones(NPTS)
        self.inputs['offset_qdyn'] = 0.0 * np.ones(NPTS)
        self.inputs['tower_qdyn'] = 0.0 * np.ones(NPTS)
        self.discrete_inputs['cross_attachment_pontoons'] = False
        self.discrete_inputs['lower_attachment_pontoons'] = False
        self.discrete_inputs['upper_attachment_pontoons'] = False
        self.discrete_inputs['lower_ring_pontoons'] = False
        self.discrete_inputs['upper_ring_pontoons'] = False
        self.discrete_inputs['outer_cross_pontoons'] = False
        self.inputs['main_pontoon_attach_upper'] = 1.0
        self.inputs['main_pontoon_attach_lower'] = 0.0
        self.inputs['water_density'] = 1e-12
        self.inputs['number_of_mooring_connections'] = 3
        self.inputs['mooring_lines_per_connection'] = 1
        self.inputs['mooring_neutral_load'] = 10.0*np.ones((15,3))
        self.inputs['mooring_moments_of_inertia'] = np.ones(6)
        self.inputs['fairlead_radius'] = 10.1
        self.inputs['fairlead_support_outer_diameter'] = 2*np.sqrt(2.0/np.pi)
        self.inputs['fairlead_support_wall_thickness'] = np.sqrt(2.0/np.pi) - np.sqrt(1.0/np.pi)
        self.inputs['material_density'] = 20.0
        self.inputs['radius_to_offset_column'] = 1.0

        goodRun = False
        kiter = 0
        while goodRun == False:
            kiter += 1
            self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
            if self.outputs['substructure_mass'] < 1e3:
                goodRun = True
            if kiter > 10:
                self.assertTrue(goodRun)
            
        msub = NSECTIONS + 3*20.0*1.0*0.1
        mtowrna = NSECTIONS + 1
        mtot = msub + mtowrna

        self.assertAlmostEqual(self.outputs['substructure_mass'], msub, 5)
        self.assertAlmostEqual(self.outputs['structural_mass'], mtot, 4)
        npt.assert_almost_equal(self.outputs['total_force'], 10*3 + np.array([10.0, 10.0, 10-mtot*g]), decimal=1)
        npt.assert_almost_equal(self.outputs['total_moment'], np.array([20.0, 20.0, 20.0]), decimal=2)

        self.inputs['rna_cg'] = np.array([5.0, 5.0, 5.0])
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        npt.assert_almost_equal(self.outputs['total_force'], 3*10 + np.array([10.0, 10.0, 10-mtot*g]), decimal=1)
        #self.assertAlmostEqual(self.outputs['total_moment'][-1], 20.0)
        
class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.inputs, self.discrete_inputs = getInputs()
        self.outputs = {}
        self.discrete_outputs = {}
        self.outputs['pontoon_stress'] = np.zeros(70)
        self.mytruss = sP.FloatingFrame(nFull=NPTS,nFullTow=NPTS)

    def tearDown(self):
        self.mytruss = None

        
    def testBadInput(self):
        self.inputs['number_of_offset_columns'] = 1
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertEqual(self.outputs['substructure_mass'], 1e30)

        self.inputs['number_of_offset_columns'] = 2
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertEqual(self.outputs['substructure_mass'], 1e30)

        self.inputs['number_of_offset_columns'] = 8
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertEqual(self.outputs['substructure_mass'], 1e30)

        self.inputs['number_of_offset_columns'] = 3
        self.inputs['main_z_full'][-2] = self.inputs['main_z_full'][-3] + 1e-12
        self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertEqual(self.outputs['substructure_mass'], 1e30)

    def testCombinations(self):
        self.inputs['radius_to_offset_column'] = 30.0
        
        for nc in [0, 3, 4, 5, 6, 7]:
            
            for cap in [True, False]:
                
                for lap in [True, False]:
                    
                    for uap in [True, False]:
                        
                        for lrp in [True, False]:
                            
                            for urp in [True, False]:
                                
                                for ocp in [True, False]:
                                    self.inputs['number_of_offset_columns'] = nc
                                    self.inputs['number_of_mooring_connections'] = np.maximum(nc, 3)
                                    self.inputs['mooring_lines_per_connection'] = 1
                                    self.discrete_inputs['cross_attachment_pontoons'] = cap
                                    self.discrete_inputs['lower_attachment_pontoons'] = lap
                                    self.discrete_inputs['upper_attachment_pontoons'] = uap
                                    self.discrete_inputs['lower_ring_pontoons'] = lrp
                                    self.discrete_inputs['upper_ring_pontoons'] = urp
                                    self.discrete_inputs['outer_cross_pontoons'] = ocp
                                    if (nc > 0) and (not cap) and (not lap) and (not uap): continue
                                    if (nc > 0) and (ocp) and (not lrp): continue
                                    
                                    self.mytruss.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
                                    if self.outputs['substructure_mass'] == 1e30:
                                        print(nc, cap, lap, uap, lrp, urp, ocp)
                                    self.assertNotEqual(self.outputs['substructure_mass'], 1e30)
                                    time.sleep(1e-3)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFrame))
    suite.addTest(unittest.makeSuite(TestSandbox))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

