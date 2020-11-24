# TODO: Code commenting and RST parallel

import numpy as np
import openmdao.api as om
from wisdem.floatingse import FloatingSE
from wisdem.commonse import fileIO

plot_flag = False #True
opt_flag  = False

npts = 5
nsection = npts - 1

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

opt['flags'] = {}
opt['flags']['monopile'] = False

opt['tower'] = {}
opt['tower']['n_height'] = npts
opt['tower']['n_layers'] = 1
opt['materials'] = {}
opt['materials']['n_mat'] = 1

# Initialize OpenMDAO problem and FloatingSE Group
prob = om.Problem()
prob.model = FloatingSE(modeling_options=opt)
prob.setup()

# Remove all offset columns
prob['number_of_offset_columns'] = 0
prob['cross_attachment_pontoons_int']   = 0
prob['lower_attachment_pontoons_int']   = 0
prob['upper_attachment_pontoons_int']   = 0
prob['lower_ring_pontoons_int']         = 0
prob['upper_ring_pontoons_int']         = 0
prob['outer_cross_pontoons_int']        = 0

# Set environment to that used in OC3 testing campaign
prob['water_depth']           = 320.0  # Distance to sea floor [m]
prob['hsig_wave']             = 10.8   # Significant wave height [m]
prob['Tsig_wave']             = 9.8    # Wave period [s]
prob['Uref']  = 11.0   # Wind reference speed [m/s]
prob['zref'] = 119.0  # Wind reference height [m]

# Column geometry
prob['main.permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
prob['main_freeboard']                = 10.0 # Height extension above waterline [m]
prob['main.height']                   = np.sum([49.0, 59.0, 8.0, 14.0])
prob['main.s']                        = np.cumsum([0.0, 49.0, 59.0, 8.0, 14.0]) / prob['main.height']
prob['main.outer_diameter_in']        = np.array([9.4, 9.4, 9.4, 6.5, 6.5])
prob['main.layer_thickness']          = 0.05 * np.ones((1,nsection))
prob['main.bulkhead_thickness']       = 0.05*np.ones(4)
prob['main.bulkhead_locations']       = np.array([0.0, 0.25, 0.9, 1.0])

# Column ring stiffener parameters
prob['main.stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
prob['main.stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
prob['main.stiffener_flange_width']     = 0.10 * np.ones(nsection) # (by section) [m]
prob['main.stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
prob['main.stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]

# Mooring parameters
prob['mooring_diameter']           = 0.09          # Diameter of mooring line/chain [m]
prob['fairlead']                   = 70.0          # Distance below waterline for attachment [m]
prob['fairlead_offset_from_shell'] = 0.5           # Offset from shell surface for mooring attachment [m]
prob['mooring_line_length']        = 902.2         # Unstretched mooring line length
prob['anchor_radius']              = 853.87        # Distance from centerline to sea floor landing [m]
prob['fairlead_support_outer_diameter'] = 3.2    # Diameter of all fairlead support elements [m]
prob['fairlead_support_wall_thickness'] = 0.0175 # Thickness of all fairlead support elements [m]

# Other variables to avoid divide by zeros, even though it won't matter
prob['radius_to_offset_column']        = 15.0
prob['offset_freeboard']               = 0.1
prob['off.height']                     = 1.0
prob['off.s']                          = np.linspace(0,1,nsection+1)
prob['off.outer_diameter_in']          = 5.0 * np.ones(nsection+1)
prob['off.layer_thickness']            = 0.1 * np.ones((1,nsection))
prob['off.permanent_ballast_height']   = 0.1
prob['off.stiffener_web_height']       = 0.1 * np.ones(nsection)
prob['off.stiffener_web_thickness']    = 0.1 * np.ones(nsection)
prob['off.stiffener_flange_width']     = 0.1 * np.ones(nsection)
prob['off.stiffener_flange_thickness'] = 0.1 * np.ones(nsection)
prob['off.stiffener_spacing']          = 0.1 * np.ones(nsection)
prob['pontoon_outer_diameter']         = 1.0
prob['pontoon_wall_thickness']         = 0.1


### Variables common to these spar, semi, TLP examples ###
# Set environment to that used in OC4 testing campaign
prob['shearExp']  = 0.11                    # Shear exponent in wind power law
prob['cm']        = 2.0                     # Added mass coefficient
prob['Uc']        = 0.0                     # Mean current speed
prob['main.wind_z0']   = 0.0                     # Water line
prob['yaw']       = 0.0                     # Turbine yaw angle
prob['beta_wind'] = prob['beta_wave'] = 0.0 # Wind/water beta angle
prob['cd_usr']    = -1.0                    # Compute drag coefficient

# Wind and water properties
prob['rho_air']   = 1.226   # Density of air [kg/m^3]
prob['mu_air']    = 1.78e-5 # Viscosity of air [kg/m/s]
prob['rho_water'] = 1025.0  # Density of water [kg/m^3]
prob['mu_water']  = 1.08e-3 # Viscosity of water [kg/m/s]

# Material properties
prob['rho_mat']                   = np.array([7850.0])    # Steel [kg/m^3]
prob['E_mat']                     = 200e9*np.ones((1,3))  # Young's modulus [N/m^2]
prob['G_mat']                     = 79.3e9*np.ones((1,3)) # Shear modulus [N/m^2]
prob['sigma_y_mat']               = np.array([3.45e8])    # Elastic yield stress [N/m^2]
prob['permanent_ballast_density'] = 4492.0                # [kg/m^3]

# Mass and cost scaling factors
prob['outfitting_factor']    = 0.06            # Fraction of additional outfitting mass for each column
prob['ballast_cost_rate']    = 0.1             # Cost factor for ballast mass [$/kg]
prob['unit_cost_mat']        = np.array([1.1]) # Cost factor for column mass [$/kg]
prob['labor_cost_rate']      = 1.0             # Cost factor for labor time [$/min]
prob['painting_cost_rate']   = 14.4            # Cost factor for column surface finishing [$/m^2]
prob['outfitting_cost_rate'] = 1.5*1.1         # Cost factor for outfitting mass [$/kg]
prob['mooring_cost_factor']  = 1.1             # Cost factor for mooring mass [$/kg]

# Mooring parameters
prob['number_of_mooring_connections'] = 3             # Evenly spaced around structure
prob['mooring_lines_per_connection']  = 1             # Evenly spaced around structure

# Porperties of turbine tower
nTower                          = prob.model.options['modeling_options']['tower']['n_height']-1
prob['tower_height']            = prob['hub_height'] = 77.6
prob['tower_s']                 = np.linspace(0.0, 1.0, nTower+1)
prob['tower_outer_diameter_in'] = np.linspace(8.0, 3.87, nTower+1)
prob['tower_layer_thickness']   = np.linspace(0.04, 0.02, nTower).reshape((1,nTower))
prob['tower_outfitting_factor'] = 1.07

# Materials
prob['material_names'] = ['steel']
prob['main.layer_materials'] = prob['off.layer_materials'] = prob['tow.tower_layer_materials'] = ['steel']

# Properties of rotor-nacelle-assembly (RNA)
prob['rna_mass']   = 350e3
prob['rna_I']      = 1e5*np.array([1149.307, 220.354, 187.597, 0, 5.037, 0])
prob['rna_cg']     = np.array([-1.132, 0, 0.509])
prob['rna_force']  = np.array([1284744.196, 0, -112400.5527])
prob['rna_moment'] = np.array([3963732.762, 896380.8464, -346781.682])

# Mooring constraints
prob['max_draft']        = 150.0 # Max surge/sway offset [m]      
prob['max_offset']       = 100.0 # Max surge/sway offset [m]      
prob['operational_heel'] = 10.0 # Max heel (pitching) angle [deg]

# Design constraints
prob['connection_ratio_max']         = 0.25 # For welding pontoons to columns

# API 2U flag
prob['loading'] = 'hydrostatic'


# Use FD and run optimization
prob.run_model()
prob.model.list_outputs(values=True, units=True)

# Visualize with mayavi, which can be difficult to install
if plot_flag:
    import wisdem.floatingse.visualize as viz
    vizobj = viz.Visualize(prob)
    vizobj.draw_spar()
