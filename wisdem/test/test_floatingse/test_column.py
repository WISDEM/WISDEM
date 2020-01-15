import numpy as np
import numpy.testing as npt
import unittest
import wisdem.floatingse.column as column
from wisdem.commonse.utilities import nodal2sectional
from wisdem.commonse import gravity as g
NPTS = 11
NSEC = 2
myones = np.ones((NPTS,))
secones = np.ones((NPTS-1,))

class TestBulk(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        self.inputs['z_full'] = np.linspace(0, 1, NPTS)
        self.inputs['z_param'] = np.linspace(0, 1, 6)
        self.inputs['d_full'] = 10.0 * myones
        self.inputs['t_full'] = 0.05 * secones
        self.inputs['rho'] = 1e3
        self.inputs['bulkhead_mass_factor'] = 1.1
        self.inputs['bulkhead_thickness'] = 0.05 * np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        self.inputs['material_cost_rate'] = 1.0
        self.inputs['painting_cost_rate'] = 10.0
        self.inputs['labor_cost_rate'] = 2.0
        self.inputs['shell_mass'] = 500.0*np.ones(NPTS-1)

        self.bulk = column.BulkheadProperties(nSection=5, nFull=NPTS)

    def testAll(self):
        self.bulk.compute(self.inputs, self.outputs)

        R_i = 0.5 * 10 - 0.05
        m_bulk = np.pi * 1e3 * 1.1 * R_i**2 * 0.05 
        expect = np.zeros( self.inputs['z_full'].shape )
        expect[[0,2,6,NPTS-1]] = m_bulk
        ind = (expect > 0.0)
        npt.assert_almost_equal(self.outputs['bulkhead_mass'], expect)

        J0 = 0.50 * m_bulk * R_i**2
        I0 = 0.25 * m_bulk * R_i**2

        I = np.zeros(6)
        I[2] = 4.0 * J0
        I[0] = I0 + m_bulk*self.inputs['z_param'][0]**2
        I[0] += I0 + m_bulk*self.inputs['z_param'][1]**2
        I[0] += I0 + m_bulk*self.inputs['z_param'][3]**2
        I[0] += I0 + m_bulk*self.inputs['z_param'][-1]**2
        I[1] = I[0]
        npt.assert_almost_equal(self.outputs['bulkhead_I_keel'], I)

        A = np.pi*R_i**2
        Kp_exp = 10.0*2*A*ind.sum()
        self.inputs['painting_cost_rate'] = 10.0
        self.inputs['material_cost_rate'] = 1.0
        self.inputs['labor_cost_rate'] = 0.0
        self.bulk.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs['bulkhead_cost'], Kp_exp + m_bulk*ind.sum())
        
        self.inputs['painting_cost_rate'] = 0.0
        self.inputs['material_cost_rate'] = 0.0
        self.inputs['labor_cost_rate'] = 1.0
        self.bulk.compute(self.inputs, self.outputs)
        self.assertGreater(self.outputs['bulkhead_cost'], 2e3)


class TestBuoyancyTank(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        self.inputs['d_full'] = 10.0*myones
        self.inputs['z_full'] = np.linspace(0, 1, NPTS) - 0.5
        self.inputs['rho'] = 1e3
        
        self.inputs['buoyancy_tank_diameter'] = 12.0
        self.inputs['buoyancy_tank_height'] = 0.25
        self.inputs['buoyancy_tank_location'] = 0.0
        self.inputs['buoyancy_tank_mass_factor'] = 1.1
        self.inputs['material_cost_rate'] = 1.0
        self.inputs['labor_cost_rate'] = 2.0
        self.inputs['painting_cost_rate'] = 10.0
        self.inputs['shell_mass'] = 500.0*np.ones(NPTS-1)

        self.box = column.BuoyancyTankProperties(nFull=NPTS)

    def testNormal(self):
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6*6 - 5*5)
        V_box = A_box * 0.25
        A_box = 2*A_box  + 0.25*2*np.pi*6
        m_expect = A_box * (6.0/50.0) * 1e3 * 1.1
        self.assertEqual(self.outputs['buoyancy_tank_mass'], m_expect)
        self.assertEqual(self.outputs['buoyancy_tank_cg'], -0.5 + 0.5*0.25)
        self.assertAlmostEqual(self.outputs['buoyancy_tank_displacement'], V_box)
        #self.assertEqual(self.outputs['buoyancy_tank_I_keel'], 0.0)
        
        self.inputs['material_cost_rate'] = 1.0
        self.inputs['labor_cost_rate'] = 0.0
        self.inputs['painting_cost_rate'] = 10.0
        self.box.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs['buoyancy_tank_cost'], m_expect + 10*2*1.5*A_box)
        
        self.inputs['material_cost_rate'] = 0.0
        self.inputs['labor_cost_rate'] = 1.0
        self.inputs['painting_cost_rate'] = 0.0
        self.box.compute(self.inputs, self.outputs)
        self.assertGreater(self.outputs['buoyancy_tank_cost'], 1e3)
        

    def testTopAbove(self):
        self.inputs['buoyancy_tank_height'] = 0.75
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6*6 - 5*5)
        V_box = np.pi * (6*6 - 5*5) * 0.5
        m_expect = (2*A_box + 0.75*2*np.pi*6) * (6.0/50.0) * 1e3 * 1.1
        self.assertAlmostEqual(self.outputs['buoyancy_tank_mass'], m_expect)
        self.assertAlmostEqual(self.outputs['buoyancy_tank_cg'], -0.5 + 0.5*0.75)
        self.assertAlmostEqual(self.outputs['buoyancy_tank_displacement'], V_box)

    def testBottomAbove(self):
        self.inputs['buoyancy_tank_location'] = 0.6
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6*6 - 5*5)
        V_box = np.pi * (6*6 - 5*5) * 0.0
        m_expect = (2*A_box + 0.25*2*np.pi*6) * (6.0/50.0) * 1e3 * 1.1
        self.assertAlmostEqual(self.outputs['buoyancy_tank_mass'], m_expect)
        self.assertAlmostEqual(self.outputs['buoyancy_tank_cg'], 0.1 + 0.5*0.25)
        self.assertAlmostEqual(self.outputs['buoyancy_tank_displacement'], V_box)

    def testTooNarrow(self):
        self.inputs['buoyancy_tank_diameter'] = 8.0
        self.box.compute(self.inputs, self.outputs)

        A_box = np.pi * (6*6 - 5*5)
        V_box = np.pi * (6*6 - 5*5) * 0.0
        m_expect = (2*A_box + 0.25*2*np.pi*6) * (6.0/50.0) * 1e3 * 1.1
        self.assertAlmostEqual(self.outputs['buoyancy_tank_mass'], 0.0)
        self.assertEqual(self.outputs['buoyancy_tank_cg'], -0.5 + 0.5*0.25)
        self.assertAlmostEqual(self.outputs['buoyancy_tank_displacement'], 0.0)


        
    
class TestStiff(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        self.inputs['t_web'] = 0.5*secones
        self.inputs['t_flange'] = 0.3*secones
        self.inputs['h_web']  = 1.0*secones
        self.inputs['w_flange'] = 2.0*secones
        self.inputs['L_stiffener'] = np.r_[0.1*np.ones(5), 0.05*np.ones(5)]
        self.inputs['rho'] = 1e3
        self.inputs['ring_mass_factor'] = 1.1
        self.inputs['material_cost_rate'] = 1.0
        self.inputs['labor_cost_rate'] = 2.0
        self.inputs['painting_cost_rate'] = 10.0
        self.inputs['shell_mass'] = 500.0*np.ones(NPTS-1)

        self.inputs['t_full'] = 0.5*secones
        self.inputs['d_full'] = 2*10.0*myones
        self.inputs['d_full'][1::2] = 2*8.0
        self.inputs['z_full'] = np.linspace(0, 1, NPTS) - 0.5
        self.inputs['z_param'] = np.linspace(0, 1, NSEC+1) - 0.5

        self.stiff = column.StiffenerProperties(nSection=NSEC, nFull=NPTS)

    def testAll(self):
        self.stiff.compute(self.inputs, self.outputs)

        Rwo = 9-0.5
        Rwi = Rwo - 1.
        Rfi = Rwi - 0.3
        V1 = np.pi*(Rwo**2 - Rwi**2)*0.5
        V2 = np.pi*(Rwi**2 - Rfi**2)*2.0 
        V = V1+V2
        expect = 1.1*V*1e3
        actual = self.outputs['stiffener_mass']

        # Test Mass
        self.assertAlmostEqual(actual.sum(), expect*(0.5/0.1 + 0.5/0.05))

        # Test cost
        A = 2*(np.pi*(Rwo**2-Rwi**2) + 2*np.pi*0.5*(Rfi+Rwi)*(0.3+2)) - 2*np.pi*Rwi*0.5
        self.inputs['material_cost_rate'] = 1.0
        self.inputs['labor_cost_rate'] = 0.0
        self.inputs['painting_cost_rate'] = 10.0
        self.stiff.compute(self.inputs, self.outputs)
        self.assertAlmostEqual(self.outputs['stiffener_cost'], (expect + 10*2*A)*(0.5/0.1 + 0.5/0.05) )
        
        self.inputs['material_cost_rate'] = 0.0
        self.inputs['labor_cost_rate'] = 1.0
        self.inputs['painting_cost_rate'] = 0.0
        self.stiff.compute(self.inputs, self.outputs)
        self.assertGreater(self.outputs['stiffener_cost'], 1e3)


        # Test moment
        self.inputs['L_stiffener'] = 1.2*secones
        self.stiff.compute(self.inputs, self.outputs)
        I_web = column.I_tube(Rwi, Rwo, 0.5, V1*1e3*1.1)
        I_fl  = column.I_tube(Rfi, Rwi, 2.0, V2*1e3*1.1)
        I_sec = I_web + I_fl
        z_sec = 0.6 + 1e-6

        I = np.zeros(6)
        I[2] = I_sec[0,2]
        I[0] += I_sec[0,0] + expect*z_sec**2.0
        I[1] = I[0]
        
        npt.assert_almost_equal(self.outputs['stiffener_I_keel'], I)
        npt.assert_equal(self.outputs['flange_spacing_ratio'], 2*2.0/1.2)
        npt.assert_equal(self.outputs['stiffener_radius_ratio'], 1.8/9.0)
        

class TestBallast(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None
        self.inputs['t_full'] = 0.5*secones
        self.inputs['d_full'] = 2*10.0*myones
        self.inputs['z_full'] = np.linspace(0, 1, NPTS) - 0.5
        self.inputs['permanent_ballast_height'] = 1.0
        self.inputs['permanent_ballast_density'] = 2e3
        self.inputs['ballast_cost_rate'] = 10.0
        self.inputs['water_density'] = 1e3

        self.ball = column.BallastProperties(nFull=NPTS)
        
    def testAll(self):
        self.ball.compute(self.inputs, self.outputs)

        area = np.pi * 9.5**2
        m_perm = area * 1.0 * 2e3
        cg_perm = self.inputs['z_full'][0] + 0.5

        I_perm = np.zeros(6)
        I_perm[2] = 0.5 * m_perm * 9.5**2
        I_perm[0] = m_perm * (3*9.5**2 + 1.0**2) / 12.0 + m_perm*0.5**2
        I_perm[1] = I_perm[0]
        
        # Unused!
        h_expect = 1e6 / area / 1000.0
        m_expect = m_perm + 1e6
        cg_water = self.inputs['z_full'][0] + 1.0 + 0.5*h_expect
        cg_expect = (m_perm*cg_perm + 1e6*cg_water) / m_expect
        
        self.assertAlmostEqual(self.outputs['ballast_mass'].sum(), m_perm)
        self.assertAlmostEqual(self.outputs['ballast_z_cg'], cg_perm)
        npt.assert_almost_equal(self.outputs['ballast_I_keel'], I_perm)


    
class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        self.inputs['z_full_in'] = np.linspace(0, 50.0, NPTS)
        self.inputs['z_param_in'] = np.array([0.0, 20.0, 50.0])
        self.inputs['section_height'] = np.array([20.0, 30.0])
        self.inputs['section_center_of_mass'],_ = nodal2sectional( self.inputs['z_full_in'] )
        self.inputs['freeboard'] = 15.0
        self.inputs['water_depth'] = 100.0
        self.inputs['stiffener_web_thickness'] = np.array([0.5, 0.5])
        self.inputs['stiffener_flange_thickness'] = np.array([0.3, 0.3])
        self.inputs['stiffener_web_height']  = np.array([1.0, 1.0])
        self.inputs['stiffener_flange_width'] = np.array([2.0, 2.0])
        self.inputs['stiffener_spacing'] = np.array([0.1, 0.1])
        self.inputs['Hs'] = 5.0
        self.inputs['max_draft'] = 70.0

        self.geom = column.ColumnGeometry(nSection=NSEC, nFull=NPTS)

    def testAll(self):
        self.geom.compute(self.inputs, self.outputs)
        self.assertEqual(self.outputs['draft'], 35.0)
        self.assertEqual(self.outputs['draft'], np.sum(self.inputs['section_height'])-self.inputs['freeboard'])
        self.assertEqual(self.outputs['draft'], -1*self.outputs['z_full'][0])
        self.assertEqual(self.outputs['draft'], -1*self.outputs['z_param'][0])
        self.assertEqual(self.outputs['draft_margin'], 0.5)
        npt.assert_equal(self.outputs['z_param'], np.array([-35.0, -15.0, 15.0]) )
        npt.assert_equal(self.outputs['z_full'], self.inputs['z_full_in']-35)
        npt.assert_equal(self.outputs['z_section'], self.inputs['section_center_of_mass']-35)
        npt.assert_equal(self.outputs['t_web'], 0.5*secones)
        npt.assert_equal(self.outputs['t_flange'], 0.3*secones)
        npt.assert_equal(self.outputs['h_web'], 1.0*secones)
        npt.assert_equal(self.outputs['w_flange'], 2.0*secones)
        npt.assert_equal(self.outputs['L_stiffener'], 0.1*secones)
        
        
        
class TestProperties(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.resid = None

        # For Geometry call
        self.inputs['z_full_in'] = np.linspace(0, 50.0, NPTS)
        self.inputs['z_param_in'] = np.array([0.0, 20.0, 50.0])
        self.inputs['section_height'] = np.array([20.0, 30.0])
        self.inputs['section_center_of_mass'],_ = nodal2sectional( self.inputs['z_full_in'] )
        self.inputs['freeboard'] = 15.0
        self.inputs['fairlead'] = 10.0
        self.inputs['water_depth'] = 100.0
        self.inputs['Hs'] = 5.0
        self.inputs['max_draft'] = 70.0
        
        self.inputs['t_full'] = 0.5*secones
        self.inputs['d_full'] = 2*10.0*myones

        self.inputs['stack_mass_in'] = 0.0

        self.inputs['shell_I_keel'] = 1e5 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs['stiffener_I_keel'] = 2e5 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs['bulkhead_I_keel'] = 3e5 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs['buoyancy_tank_I_keel'] = 5e6 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.inputs['ballast_I_keel'] = 2e3 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        self.inputs['buoyancy_tank_diameter'] = 15.0
        
        self.inputs['water_density'] = 1e3
        self.inputs['bulkhead_mass'] = 10.0*myones
        self.inputs['shell_mass'] = 500.0*np.ones(NPTS-1)
        self.inputs['stiffener_mass'] = 100.0*np.ones(NPTS-1)
        self.inputs['ballast_mass'] = 20.0*np.ones(NPTS-1)
        self.inputs['ballast_z_cg'] = -35.0
        
        self.inputs['buoyancy_tank_mass'] = 20.0
        self.inputs['buoyancy_tank_cg'] = -15.0
        self.inputs['buoyancy_tank_location'] = 0.3
        self.inputs['buoyancy_tank_displacement'] = 300.0
        self.inputs['column_mass_factor'] = 1.1
        self.inputs['outfitting_mass_fraction'] = 0.05

        self.inputs['shell_cost'] = 1.0
        self.inputs['stiffener_cost'] = 2.0
        self.inputs['bulkhead_cost'] = 3.0
        self.inputs['buoyancy_tank_cost'] = 4.0
        self.inputs['ballast_cost'] = 5.0
        
        self.inputs['mooring_mass'] = 50.0
        self.inputs['mooring_vertical_load'] = 25.0
        self.inputs['mooring_restoring_force'] = 1e5
        self.inputs['mooring_cost'] = 1e4

        self.inputs['outfitting_cost_rate'] = 1.0

        self.inputs['stiffener_web_thickness'] = np.array([0.5, 0.5])
        self.inputs['stiffener_flange_thickness'] = np.array([0.3, 0.3])
        self.inputs['stiffener_web_height']  = np.array([1.0, 1.0])
        self.inputs['stiffener_flange_width'] = np.array([2.0, 2.0])
        self.inputs['stiffener_spacing'] = np.array([0.1, 0.1])
        
        self.geom = column.ColumnGeometry(nSection=NSEC, nFull=NPTS)
        self.set_geometry()

        self.mycolumn = column.ColumnProperties(nFull=NPTS)
        
    def set_geometry(self):
        tempUnknowns = {}
        self.geom.compute(self.inputs, tempUnknowns)
        for pairs in tempUnknowns.items():
            self.inputs[pairs[0]] = pairs[1]

    def testColumnMassCG(self):
        self.mycolumn.compute_column_mass_cg(self.inputs, self.outputs)
        ibox = self.mycolumn.ibox
        
        bulk  = self.inputs['bulkhead_mass']
        stiff = self.inputs['stiffener_mass']
        shell = self.inputs['shell_mass']
        box   = self.inputs['buoyancy_tank_mass']
        boxcg = self.inputs['buoyancy_tank_cg']
        m_ballast = self.inputs['ballast_mass']
        cg_ballast = self.inputs['ballast_z_cg']

        m_column = 1.1*(bulk.sum() + stiff.sum() + shell.sum() + box)
        m_out    = 0.05 * m_column
        m_expect = m_column + m_ballast.sum() + m_out

        mysec = stiff+shell+bulk[:-1]
        mysec[-1] += bulk[-1]
        mysec[ibox] += box
        mysec *= 1.1
        mysec += m_ballast
        mysec += (m_out/len(mysec))

        mycg  = 1.1*(np.dot(bulk, self.inputs['z_full']) + box*boxcg + np.dot(stiff+shell, self.inputs['z_section']))/m_column
        cg_system = ((m_column+m_out)*mycg + m_ballast.sum()*cg_ballast) / m_expect

        Iones = np.r_[np.ones(3), np.zeros(3)]
        I_expect = 1.05 * 1.1 * 5.6e6*Iones + 2e3*Iones
        I_expect[0] = I_expect[1] = I_expect[0]-m_expect*(cg_system-self.inputs['z_full'][0])**2

        self.assertAlmostEqual(self.outputs['column_total_mass'].sum(), m_expect)
        self.assertAlmostEqual(self.outputs['z_center_of_mass'], cg_system)
        
        self.assertAlmostEqual(self.outputs['column_structural_mass'], m_column+m_out )
        self.assertEqual(self.outputs['column_outfitting_mass'], m_out )
        npt.assert_equal(self.outputs['column_total_mass'], mysec)
        npt.assert_equal(self.outputs['I_column'], I_expect)


    def testBalance(self):
        rho_w = self.inputs['water_density']

        self.mycolumn.compute_column_mass_cg(self.inputs, self.outputs)
        self.mycolumn.balance_column(self.inputs, self.outputs)

        V_column = np.pi * 100.0 * 35.0
        V_box    = self.inputs['buoyancy_tank_displacement']
        box_cg   = self.inputs['buoyancy_tank_cg']
        V_expect = V_column + V_box
        cb_expect = (-17.5*V_column + V_box*box_cg) / V_expect
        Ixx = 0.25 * np.pi * 1e4
        Axx = np.pi * 1e2
        self.assertAlmostEqual(self.outputs['displaced_volume'].sum(), V_expect)
        self.assertAlmostEqual(self.outputs['hydrostatic_force'].sum(), V_expect*rho_w*g)
        self.assertAlmostEqual(self.outputs['z_center_of_buoyancy'], cb_expect)
        self.assertAlmostEqual(self.outputs['Iwater'], Ixx)
        self.assertAlmostEqual(self.outputs['Awater'], Axx)

        m_a = np.zeros(6)
        m_a[:2] = V_expect * rho_w
        m_a[2]  = 0.5 * (8.0/3.0) * rho_w * 10.0**3
        m_a[3:5] = np.pi * rho_w * 100.0 * ((0-cb_expect)**3.0 - (-35-cb_expect)**3.0) / 3.0
        npt.assert_almost_equal(self.outputs['column_added_mass'], m_a, decimal=-4)
        
        # Test if everything under water
        dz = -1.5*self.inputs['z_full'][-1]
        self.inputs['z_section'] += dz 
        self.inputs['z_full'] += dz 
        self.mycolumn.balance_column(self.inputs, self.outputs)
        V_column = np.pi * 100.0 * 50.0
        V_expect = V_column + V_box
        cb_expect = (V_column*(-25.0 + self.inputs['z_full'][-1])  + V_box*box_cg) / V_expect
        self.assertAlmostEqual(self.outputs['displaced_volume'].sum(), V_expect)
        self.assertAlmostEqual(self.outputs['hydrostatic_force'].sum(), V_expect*rho_w*g)
        self.assertAlmostEqual(self.outputs['z_center_of_buoyancy'], cb_expect)

        # Test taper- check hydrostatic via Archimedes within 1%
        self.inputs['d_full'][5] -= 8.0
        self.mycolumn.balance_column(self.inputs, self.outputs)
        self.assertAlmostEqual(self.outputs['hydrostatic_force'].sum() / (self.outputs['displaced_volume'].sum()*rho_w*g), 1.0, delta=1e-2)

        
    def testCheckCost(self):
        self.outputs['column_outfitting_mass'] = 25.0
        self.outputs['column_total_mass'] = 25*np.ones(10)
        self.mycolumn.compute_cost(self.inputs, self.outputs)

        self.assertEqual(self.outputs['column_structural_cost'], 1.1*(1+2+3+4))
        self.assertEqual(self.outputs['column_outfitting_cost'], 1.0 * 25.0)
        self.assertEqual(self.outputs['column_total_cost'], 1.1*(1+2+3+4) + 1.0*(25.0) + 5)

        
class TestBuckle(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        onepts  = np.ones((NPTS,))
        onesec  = np.ones((NPTS-1,))
        #onesec0 = np.ones((NSEC,))
        self.inputs['d_full'] = 600 * onepts * in_to_si
        self.inputs['t_full'] = 0.75 * onesec * in_to_si
        self.inputs['t_web'] = 5./8. * onesec * in_to_si
        self.inputs['h_web'] = 14.0 * onesec * in_to_si
        self.inputs['t_flange'] = 1.0 * onesec * in_to_si
        self.inputs['w_flange'] = 10.0 * onesec * in_to_si
        self.inputs['L_stiffener'] = 5.0 * onesec * ft_to_si
        #self.inputs['section_height'] = 50.0 * onesec0 * ft_to_si
        self.inputs['pressure'] = (64.0*lbperft3_to_si) * g * (60*ft_to_si) * onepts
        self.inputs['E'] = 29e3 * ksi_to_si
        self.inputs['nu'] = 0.3
        self.inputs['yield_stress'] = 50 * ksi_to_si
        self.inputs['wave_height'] = 0.0 # gives only static pressure
        self.inputs['stack_mass_in'] = 9000 * kip_to_si/g
        self.inputs['section_mass'] = 0.0 * np.ones((NPTS-1,))
        self.discrete_inputs['loading'] = 'radial'
        self.inputs['z_full'] = np.linspace(0, 1, NPTS)
        self.inputs['z_section'],_ = nodal2sectional( self.inputs['z_full'] )
        self.inputs['z_param'] = np.linspace(0, 1, NSEC+1)
        self.inputs['gamma_f'] = 1.0
        self.inputs['gamma_b'] = 1.0

        self.buckle = column.ColumnBuckling(nSection=NSEC, nFull=NPTS)


    def testAppliedAxial(self):
        t = self.inputs['t_full'][0]
        d = self.inputs['d_full'][0]
        kip_to_si = 4.4482216 * 1e3
        expect = 9000 * kip_to_si / (2*np.pi*t*(0.5*d-0.5*t))
        npt.assert_almost_equal(self.buckle.compute_applied_axial(self.inputs), expect, decimal=4)
        
    def testCheckStresses(self):
        self.buckle.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        
        npt.assert_almost_equal(self.outputs['web_compactness'], 24.1/22.4, decimal=3)
        npt.assert_almost_equal(self.outputs['flange_compactness'], 9.03/5.0, decimal=3)
        self.assertAlmostEqual(self.outputs['axial_local_api'][1], 1.07, 1)
        self.assertAlmostEqual(self.outputs['axial_general_api'][1], 0.34, 1)
        self.assertAlmostEqual(self.outputs['external_local_api'][1], 1.07, 1)
        self.assertAlmostEqual(self.outputs['external_general_api'][1], 0.59, 1)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBulk))
    suite.addTest(unittest.makeSuite(TestBuoyancyTank))
    suite.addTest(unittest.makeSuite(TestStiff))
    suite.addTest(unittest.makeSuite(TestBallast))
    suite.addTest(unittest.makeSuite(TestGeometry))
    suite.addTest(unittest.makeSuite(TestProperties))
    suite.addTest(unittest.makeSuite(TestBuckle))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
