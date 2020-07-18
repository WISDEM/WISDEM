# TODO: Code commenting and RST parallel

import numpy as np
import openmdao.api as om
from wisdem.floatingse import FloatingSE
from wisdem.floatingse.floating import commonVars
from wisdem.commonse import fileIO


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

opt['tower'] = {}
opt['tower']['monopile'] = False
opt['tower']['n_height'] = npts
opt['tower']['n_layers'] = 1
opt['materials'] = {}
opt['materials']['n_mat'] = 1

# Initialize OpenMDAO problem and FloatingSE Group
prob = om.Problem()
prob.model = FloatingSE(analysis_options=opt)
#prob.model.nonlinear_solver = om.NonlinearBlockGS()
#prob.model.linear_solver = om.LinearBlockGS()
prob.setup()

# Variables common to both examples
prob = commonVars(prob, nsection)

# Remove all offset columns
prob['number_of_offset_columns'] = 0
prob['cross_attachment_pontoons_int']   = 0
prob['lower_attachment_pontoons_int']   = 0
prob['upper_attachment_pontoons_int']   = 0
prob['lower_ring_pontoons_int']         = 0
prob['upper_ring_pontoons_int']         = 0
prob['outer_cross_pontoons_int']        = 0

# Set environment to that used in OC3 testing campaign
prob['water_depth'] = 320.0  # Distance to sea floor [m]
prob['hsig_wave']        = 10.8   # Significant wave height [m]
prob['Tsig_wave']           = 9.8    # Wave period [s]
prob['wind_reference_speed']        = 11.0   # Wind reference speed [m/s]
prob['wind_reference_height']        = 119.0  # Wind reference height [m]

# Column geometry
prob['main.permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
prob['main_freeboard']                = 10.0 # Height extension above waterline [m]

prob['main.height'] = np.sum([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
prob['main.s'] = np.cumsum([0.0, 49.0, 59.0, 8.0, 14.0]) / prob['main.height']
prob['main.outer_diameter_in'] = np.array([9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
prob['main.layer_thickness'] = 0.05 * np.ones((1,nsection))               # Shell thickness at each section node (linear lofting between) [m]

prob['main.bulkhead_thickness'] = 0.05*np.ones(4) # Locations/thickness of internal bulkheads at section interfaces [m]
prob['main.bulkhead_locations'] = np.array([0.0, 0.25, 0.9, 1.0]) # Locations/thickness of internal bulkheads at section interfaces [m]

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
prob['radius_to_offset_column'] = 15.0
prob['offset_freeboard'] = 0.1
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
prob['pontoon_outer_diameter'] = 1.0
prob['pontoon_wall_thickness'] = 0.1

prob.run_model()

prob.model.list_outputs(values=True, units=True)

