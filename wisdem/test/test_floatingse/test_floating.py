import numpy as np
import numpy.testing as npt
import unittest
from openmdao.api import Problem
from wisdem.floatingse.floating import FloatingSE
import pytest

npts = 5
nsection = npts - 1

class TestOC3Mass(unittest.TestCase):
    def testMassPropertiesSpar(self):
    
        opt = {}
        opt['platform'] = {}
        opt['platform']['columns'] = {}
        opt['platform']['columns']['main'] = {}
        opt['platform']['columns']['offset'] = {}
        opt['platform']['columns']['main']['n_height'] = npts
        opt['platform']['columns']['main']['n_layers'] = 1
        opt['platform']['columns']['main']['n_bulkhead'] = 4
        opt['platform']['columns']['main']['buckling_length'] = 30.0
        opt['platform']['columns']['offset']['n_height'] = npts
        opt['platform']['columns']['offset']['n_layers'] = 1
        opt['platform']['columns']['offset']['n_bulkhead'] = 4
        opt['platform']['columns']['offset']['buckling_length'] = 30.0
        opt['platform']['tower'] = {}
        opt['platform']['tower']['buckling_length'] = 30.0
        opt['platform']['frame3dd']            = {}
        opt['platform']['frame3dd']['shear']   = True
        opt['platform']['frame3dd']['geom']    = False
        opt['platform']['frame3dd']['dx']      = -1
        #opt['platform']['frame3dd']['nM']      = 2
        opt['platform']['frame3dd']['Mmethod'] = 1
        opt['platform']['frame3dd']['lump']    = 0
        opt['platform']['frame3dd']['tol']     = 1e-6
        #opt['platform']['frame3dd']['shift']   = 0.0
        opt['platform']['gamma_f'] = 1.35  # Safety factor on loads
        opt['platform']['gamma_m'] = 1.3   # Safety factor on materials
        opt['platform']['gamma_n'] = 1.0   # Safety factor on consequence of failure
        opt['platform']['gamma_b'] = 1.1   # Safety factor on buckling
        opt['platform']['gamma_fatigue'] = 1.755 # Not used
        opt['platform']['run_modal'] = True # Not used

        opt['tower'] = {}
        opt['tower']['n_height'] = npts
        opt['tower']['n_layers'] = 1
        opt['materials'] = {}
        opt['materials']['n_mat'] = 1

        opt['flags'] = {}
        opt['flags']['monopile'] = False
        
        prob = Problem()
        prob.model=FloatingSE(modeling_options=opt)
        prob.setup()

        # Remove all offset columns
        prob['number_of_offset_columns'] = 0
        prob['cross_attachment_pontoons_int']   = 0
        prob['lower_attachment_pontoons_int']   = 0
        prob['upper_attachment_pontoons_int']   = 0
        prob['lower_ring_pontoons_int']         = 0
        prob['upper_ring_pontoons_int']         = 0
        prob['outer_cross_pontoons_int']        = 0

        # Wind and water properties
        prob['rho_air'] = 1.226   # Density of air [kg/m^3]
        prob['mu_air']  = 1.78e-5 # Viscosity of air [kg/m/s]
        prob['rho_water']      = 1025.0  # Density of water [kg/m^3]
        prob['mu_water']  = 1.08e-3 # Viscosity of water [kg/m/s]

        # Material properties
        prob['rho_mat']     = np.array([7850.0])          # Steel [kg/m^3]
        prob['E_mat']       = 200e9*np.ones((1,3))           # Young's modulus [N/m^2]
        prob['G_mat']       = 79.3e9*np.ones((1,3))          # Shear modulus [N/m^2]
        prob['sigma_y_mat'] = np.array([3.45e8])          # Elastic yield stress [N/m^2]
        prob['permanent_ballast_density'] = 5000.0 # [kg/m^3]

        # Mass and cost scaling factors
        prob['outfitting_factor'] = 0.0    # Fraction of additional outfitting mass for each column
        prob['ballast_cost_rate']        = 0.1   # Cost factor for ballast mass [$/kg]
        prob['material_cost_rate']       = 1.1  # Cost factor for column mass [$/kg]
        prob['labor_cost_rate']          = 1.0  # Cost factor for labor time [$/min]
        prob['painting_cost_rate']       = 14.4  # Cost factor for column surface finishing [$/m^2]
        prob['outfitting_cost_rate']     = 1.5*1.1  # Cost factor for outfitting mass [$/kg]
        prob['mooring_cost_factor']        = 1.1     # Cost factor for mooring mass [$/kg]

        # Column geometry
        prob['main.permanent_ballast_height'] = 0.0 # Height above keel for permanent ballast [m]
        prob['main_freeboard']                = 10.0 # Height extension above waterline [m]

        prob['main.height'] = np.sum([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
        prob['main.s'] = np.cumsum([0.0, 49.0, 59.0, 8.0, 14.0]) / prob['main.height']
        prob['main.outer_diameter_in'] = np.array([9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
        prob['main.layer_thickness'] = 0.05 * np.ones((1,nsection))               # Shell thickness at each section node (linear lofting between) [m]
        
        prob['main.bulkhead_thickness'] = 0.05*np.ones(4) # Locations/thickness of internal bulkheads at section interfaces [m]
        prob['main.bulkhead_locations'] = np.array([0.0, 0.37692308, 0.89230769, 1.0]) # Locations/thickness of internal bulkheads at section interfaces [m]

        prob['main.buoyancy_tank_diameter'] = 0.0
        prob['main.buoyancy_tank_height'] = 0.0

        # Column ring stiffener parameters
        prob['main.stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
        prob['main.stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
        prob['main.stiffener_flange_width']     = 0.10 * np.ones(nsection) # (by section) [m]
        prob['main.stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
        prob['main.stiffener_spacing']          = np.array([1.5, 2.8, 3.0, 5.0]) # (by section) [m]

        # Mooring parameters
        prob['number_of_mooring_connections']    = 3             # Evenly spaced around structure
        prob['mooring_lines_per_connection']    = 1             # Evenly spaced around structure
        prob['mooring_type']               = 'chain'       # Options are chain, nylon, polyester, fiber, or iwrc
        prob['anchor_type']                = 'suctionpile' # Options are SUCTIONPILE or DRAGEMBEDMENT
        prob['mooring_diameter']           = 0.09          # Diameter of mooring line/chain [m]
        prob['fairlead_location']          = 0.384615 # Want 70.0          # Distance below waterline for attachment [m]
        prob['fairlead_offset_from_shell'] = 0.5           # Offset from shell surface for mooring attachment [m]
        prob['mooring_line_length']        = 300+902.2         # Unstretched mooring line length
        prob['anchor_radius']              = 853.87        # Distance from centerline to sea floor landing [m]
        prob['fairlead_support_outer_diameter'] = 3.2    # Diameter of all fairlead support elements [m]
        prob['fairlead_support_wall_thickness'] = 0.0175 # Thickness of all fairlead support elements [m]

        # Mooring constraints
        prob['max_offset'] = 0.1*prob['water_depth'] # Max surge/sway offset [m]      
        prob['max_survival_heel']   = 10.0 # Max heel (pitching) angle [deg]
        prob['operational_heel']   = 5.0 # Max heel (pitching) angle [deg]

        # Design constraints
        prob['max_draft'] = 200.0                # For manufacturability of rolling steel

        # API 2U flag
        prob['loading'] = 'axial' #'hydrostatic'

        # Other variables to avoid divide by zeros, even though it won't matter
        prob['radius_to_offset_column'] = 15.0
        prob['off.height'] = 1.0
        prob['off.s'] = np.linspace(0,1,nsection+1)
        prob['off.outer_diameter_in'] = 5.0 * np.ones(nsection+1)
        prob['off.layer_thickness'] = 0.1 * np.ones((1,nsection))
        prob['off.permanent_ballast_height'] = 0.1
        prob['off.stiffener_web_height'] = 0.1 * np.ones(nsection)
        prob['off.stiffener_web_thickness'] =  0.1 * np.ones(nsection)
        prob['off.stiffener_flange_width'] =  0.1 * np.ones(nsection)
        prob['off.stiffener_flange_thickness'] =  0.1 * np.ones(nsection)
        prob['off.stiffener_spacing'] =  0.1 * np.ones(nsection)
        prob['offset_freeboard'] =  0.1
        prob['pontoon_outer_diameter'] = 1.0
        prob['pontoon_wall_thickness'] = 0.1

        # Set environment to that used in OC3 testing campaign
        prob['water_depth']           = 320.0  # Distance to sea floor [m]
        prob['hsig_wave']             = 0.0    # Significant wave height [m]
        prob['Tsig_wave']             = 1e3    # Wave period [s]
        prob['shearExp']              = 0.11   # Shear exponent in wind power law
        prob['cm']                    = 2.0    # Added mass coefficient
        prob['Uc']                    = 0.0    # Mean current speed
        prob['yaw']                   = 0.0    # Turbine yaw angle
        prob['beta_wind']             = prob['beta_wave'] = 0.0
        prob['cd_usr']                = -1.0 # Compute drag coefficient
        prob['Uref'] = 10.0
        prob['zref'] = 100.0

        # Porperties of turbine tower
        nTower = prob.model.options['modeling_options']['tower']['n_height']-1
        prob['tower_height']            = prob['hub_height'] = 77.6
        prob['tower_s']                 = np.linspace(0.0, 1.0, nTower+1)
        prob['tower_outer_diameter_in'] = np.linspace(6.5, 3.87, nTower+1)
        prob['tower_layer_thickness']   = np.linspace(0.027, 0.019, nTower).reshape((1,nTower))
        prob['tower_outfitting_factor'] = 1.07

        # Materials
        prob['material_names'] = ['steel']
        prob['main.layer_materials'] = prob['off.layer_materials'] = prob['tow.tower_layer_materials'] = ['steel']

        # Properties of rotor-nacelle-assembly (RNA)
        prob['rna_mass']   = 350e3 # Mass [kg]
        prob['rna_I']      = 1e5*np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        prob['rna_cg']     = np.zeros(3)
        prob['rna_force']  = np.zeros(3)
        prob['rna_moment'] = np.zeros(3)
        
        
        prob.run_model()

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

        npt.assert_allclose(ansys_m_bulk, prob['main.bulkhead_mass'].sum(), rtol=0.03) # ANSYS uses R_od, we use R_id, top cover seems unaccounted for
        npt.assert_allclose(ansys_m_shell, prob['main.cyl_mass.mass'].sum(), rtol=0.01)
        npt.assert_allclose(ansys_m_stiff, prob['main.stiffener_mass'].sum(), rtol=0.01)
        npt.assert_allclose(ansys_m_spar, prob['main.column_total_mass'].sum(), rtol=0.01)
        npt.assert_allclose(ansys_cg, prob['main.z_center_of_mass'], rtol=0.01)
        npt.assert_allclose(ansys_I, prob['main.I_column'], rtol=0.02)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestOC3Mass))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
