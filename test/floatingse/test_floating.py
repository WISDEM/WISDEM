import numpy as np
import numpy.testing as npt
import unittest
from openmdao.api import Problem
from floatingse.floating import FloatingSE

nSec = 4
nSecTow = 3
NPTS = 100

class TestOC3Mass(unittest.TestCase):
    def setUp(self):
        self.myfloat = Problem(root=FloatingSE())
        self.myfloat.setup()

        # Remove all offset columns
        self.myfloat['number_of_offset_columns'] = 0
        self.myfloat['cross_attachment_pontoons_int']   = 0
        self.myfloat['lower_attachment_pontoons_int']   = 0
        self.myfloat['upper_attachment_pontoons_int']   = 0
        self.myfloat['lower_ring_pontoons_int']         = 0
        self.myfloat['upper_ring_pontoons_int']         = 0
        self.myfloat['outer_cross_pontoons_int']        = 0

        # Wind and water properties
        self.myfloat['main.windLoads.rho'] = 1.226   # Density of air [kg/m^3]
        self.myfloat['main.windLoads.mu']  = 1.78e-5 # Viscosity of air [kg/m/s]
        self.myfloat['water_density']      = 1025.0  # Density of water [kg/m^3]
        self.myfloat['main.waveLoads.mu']  = 1.08e-3 # Viscosity of water [kg/m/s]

        # Material properties
        self.myfloat['material_density'] = 7850.0          # Steel [kg/m^3]
        self.myfloat['E']                = 200e9           # Young's modulus [N/m^2]
        self.myfloat['G']                = 79.3e9          # Shear modulus [N/m^2]
        self.myfloat['yield_stress']     = 3.45e8          # Elastic yield stress [N/m^2]
        self.myfloat['nu']               = 0.26            # Poisson's ratio
        self.myfloat['permanent_ballast_density'] = 5000.0 # [kg/m^3]

        # Mass and cost scaling factors
        self.myfloat['bulkhead_mass_factor']     = 1.0     # Scaling for unaccounted bulkhead mass
        self.myfloat['ring_mass_factor']         = 1.0     # Scaling for unaccounted stiffener mass
        self.myfloat['shell_mass_factor']        = 1.0     # Scaling for unaccounted shell mass
        self.myfloat['column_mass_factor']       = 1.0    # Scaling for unaccounted column mass
        self.myfloat['outfitting_mass_fraction'] = 0.0    # Fraction of additional outfitting mass for each column
        self.myfloat['ballast_cost_rate']        = 0.1   # Cost factor for ballast mass [$/kg]
        self.myfloat['material_cost_rate']       = 1.1  # Cost factor for column mass [$/kg]
        self.myfloat['labor_cost_rate']          = 1.0  # Cost factor for labor time [$/min]
        self.myfloat['painting_cost_rate']       = 14.4  # Cost factor for column surface finishing [$/m^2]
        self.myfloat['outfitting_cost_rate']     = 1.5*1.1  # Cost factor for outfitting mass [$/kg]
        self.myfloat['mooring_cost_rate']        = 1.1     # Cost factor for mooring mass [$/kg]

        # Safety factors
        self.myfloat['gamma_f'] = 1.0 # Safety factor on loads
        self.myfloat['gamma_b'] = 1.0  # Safety factor on buckling
        self.myfloat['gamma_m'] = 1.0  # Safety factor on materials
        self.myfloat['gamma_n'] = 1.0  # Safety factor on consequence of failure
        self.myfloat['gamma_fatigue'] = 1.0 # Not used

        # Column geometry
        self.myfloat['main_permanent_ballast_height'] = 0.0 # Height above keel for permanent ballast [m]
        self.myfloat['main_freeboard']                = 10.0 # Height extension above waterline [m]
        self.myfloat['main_section_height'] = np.array([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
        self.myfloat['main_outer_diameter'] = np.array([9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
        self.myfloat['main_wall_thickness'] = 0.05 * np.ones(nSec)               # Shell thickness at each section node (linear lofting between) [m]
        self.myfloat['main_bulkhead_thickness'] = 0.05*np.array([1, 1, 0, 1, 0]) # Locations/thickness of internal bulkheads at section interfaces [m]
        self.myfloat['main_buoyancy_tank_diameter'] = 0.0
        self.myfloat['main_buoyancy_tank_height'] = 0.0

        # Column ring stiffener parameters
        self.myfloat['main_stiffener_web_height']       = 0.10 * np.ones(nSec) # (by section) [m]
        self.myfloat['main_stiffener_web_thickness']    = 0.04 * np.ones(nSec) # (by section) [m]
        self.myfloat['main_stiffener_flange_width']     = 0.10 * np.ones(nSec) # (by section) [m]
        self.myfloat['main_stiffener_flange_thickness'] = 0.02 * np.ones(nSec) # (by section) [m]
        self.myfloat['main_stiffener_spacing']          = np.array([1.5, 2.8, 3.0, 5.0]) # (by section) [m]

        # Mooring parameters
        self.myfloat['number_of_mooring_connections']    = 3             # Evenly spaced around structure
        self.myfloat['mooring_lines_per_connection']    = 1             # Evenly spaced around structure
        self.myfloat['mooring_type']               = 'chain'       # Options are chain, nylon, polyester, fiber, or iwrc
        self.myfloat['anchor_type']                = 'suctionpile' # Options are SUCTIONPILE or DRAGEMBEDMENT
        self.myfloat['mooring_diameter']           = 0.09          # Diameter of mooring line/chain [m]
        self.myfloat['fairlead_location']          = 0.384615 # Want 70.0          # Distance below waterline for attachment [m]
        self.myfloat['fairlead_offset_from_shell'] = 0.5           # Offset from shell surface for mooring attachment [m]
        self.myfloat['mooring_line_length']        = 300+902.2         # Unstretched mooring line length
        self.myfloat['anchor_radius']              = 853.87        # Distance from centerline to sea floor landing [m]
        self.myfloat['fairlead_support_outer_diameter'] = 3.2    # Diameter of all fairlead support elements [m]
        self.myfloat['fairlead_support_wall_thickness'] = 0.0175 # Thickness of all fairlead support elements [m]

        # Mooring constraints
        self.myfloat['max_offset'] = 0.1*self.myfloat['water_depth'] # Max surge/sway offset [m]      
        self.myfloat['max_survival_heel']   = 10.0 # Max heel (pitching) angle [deg]
        self.myfloat['operational_heel']   = 5.0 # Max heel (pitching) angle [deg]

        # Design constraints
        self.myfloat['max_draft'] = 200.0                # For manufacturability of rolling steel
        self.myfloat['max_taper_ratio'] = 0.4                # For manufacturability of rolling steel
        self.myfloat['min_diameter_thickness_ratio'] = 120.0 # For weld-ability

        # API 2U flag
        self.myfloat['loading'] = 'axial' #'hydrostatic'

        # Other variables to avoid divide by zeros, even though it won't matter
        self.myfloat['radius_to_offset_column'] = 15.0
        self.myfloat['offset_section_height'] = 1.0 * np.ones(nSec)
        self.myfloat['offset_outer_diameter'] = 5.0 * np.ones(nSec+1)
        self.myfloat['offset_wall_thickness'] = 0.1 * np.ones(nSec)
        self.myfloat['offset_permanent_ballast_height'] = 0.1
        self.myfloat['offset_stiffener_web_height'] = 0.1 * np.ones(nSec)
        self.myfloat['offset_stiffener_web_thickness'] =  0.1 * np.ones(nSec)
        self.myfloat['offset_stiffener_flange_width'] =  0.1 * np.ones(nSec)
        self.myfloat['offset_stiffener_flange_thickness'] =  0.1 * np.ones(nSec)
        self.myfloat['offset_stiffener_spacing'] =  0.1 * np.ones(nSec)
        self.myfloat['offset_freeboard'] =  0.1
        self.myfloat['pontoon_outer_diameter'] = 1.0
        self.myfloat['pontoon_wall_thickness'] = 0.1

        # Set environment to that used in OC3 testing campaign
        self.myfloat['water_depth'] = 320.0  # Distance to sea floor [m]
        self.myfloat['Hs']        = 0.0    # Significant wave height [m]
        self.myfloat['T']           = 1e3    # Wave period [s]
        self.myfloat['Uref']        = 0.0    # Wind reference speed [m/s]
        self.myfloat['zref']        = 119.0  # Wind reference height [m]
        self.myfloat['shearExp']    = 0.11   # Shear exponent in wind power law
        self.myfloat['cm']          = 2.0    # Added mass coefficient
        self.myfloat['Uc']          = 0.0    # Mean current speed
        self.myfloat['z0']          = 0.0    # Water line
        self.myfloat['yaw']         = 0.0    # Turbine yaw angle
        self.myfloat['beta']        = 0.0    # Wind beta angle
        self.myfloat['cd_usr']      = np.inf # Compute drag coefficient

        # Porperties of turbine tower
        self.myfloat['hub_height']              = 77.6                              # Length from tower main to top (not including freeboard) [m]
        self.myfloat['tower_section_height']    = 77.6/nSecTow * np.ones(nSecTow) # Length of each tower section [m]
        self.myfloat['tower_outer_diameter']    = np.linspace(6.5, 3.87, nSecTow+1) # Diameter at each tower section node (linear lofting between) [m]
        self.myfloat['tower_wall_thickness']    = np.linspace(0.027, 0.019, nSecTow) # Diameter at each tower section node (linear lofting between) [m]
        self.myfloat['tower_buckling_length']   = 30.0                              # Tower buckling reinforcement spacing [m]
        self.myfloat['tower_outfitting_factor'] = 1.07                              # Scaling for unaccounted tower mass in outfitting

        # Properties of rotor-nacelle-assembly (RNA)
        self.myfloat['rna_mass']   = 350e3 # Mass [kg]
        self.myfloat['rna_I']      = 1e5*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.myfloat['rna_cg']     = np.zeros(3)
        self.myfloat['rna_force']  = np.zeros(3)
        self.myfloat['rna_moment'] = np.zeros(3)
        
        
    def testMassProperties(self):
        self.myfloat.run()

        m_top = np.pi*3.2**2.0*0.05*7850.0
        ansys_m_bulk  = 13204.0 + 2.0*27239.0 + m_top
        ansys_m_shell = 80150.0 + 32060.0 + 79701.0 + 1251800.0
        ansys_m_stiff = 1390.9*52 + 1282.2 + 1121.2 + 951.44*3
        ansys_m_spar = ansys_m_bulk + ansys_m_shell + ansys_m_stiff
        ansys_cg     = -58.926
        ansys_Ixx    = 2178400000.0 + m_top*(0.25*3.2**2.0 + (10-ansys_cg)**2)
        ansys_Iyy    = 2178400000.0 + m_top*(0.25*3.2**2.0 + (10-ansys_cg)**2)
        ansys_Izz    = 32297000.0 + 0.5*m_top*3.2**2.0
        ansys_I      = np.array([ansys_Ixx, ansys_Iyy, ansys_Izz, 0.0, 0.0, 0.0])

        npt.assert_allclose(ansys_m_bulk, self.myfloat['main.bulkhead_mass'].sum(), rtol=0.03) # ANSYS uses R_od, we use R_id, top cover seems unaccounted for
        npt.assert_allclose(ansys_m_shell, self.myfloat['main.cyl_mass.mass'].sum(), rtol=0.01)
        npt.assert_allclose(ansys_m_stiff, self.myfloat['main.stiffener_mass'].sum(), rtol=0.01)
        npt.assert_allclose(ansys_m_spar, self.myfloat['main.column_total_mass'].sum(), rtol=0.01)
        npt.assert_allclose(ansys_cg, self.myfloat['main.z_center_of_mass'], rtol=0.01)
        npt.assert_allclose(ansys_I, self.myfloat['main.I_column'], rtol=0.02)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestOC3Mass))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
