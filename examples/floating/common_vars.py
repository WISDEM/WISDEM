import numpy as np

def set_common(prob):
    
    # Set environment to that used in OC4 testing campaign
    prob['shearExp']  = 0.11                    # Shear exponent in wind power law
    prob['cm']        = 2.0                     # Added mass coefficient
    prob['Uc']        = 0.0                     # Mean current speed
    prob['wind_z0']   = 0.0                     # Water line
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
    nTower                          = prob.model.options['analysis_options']['tower']['n_height']-1
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
    prob['max_taper_ratio']              = 0.2  # For manufacturability of rolling steel
    prob['min_diameter_thickness_ratio'] = 80.0 # For weld-ability
    prob['connection_ratio_max']         = 0.25 # For welding pontoons to columns

    # API 2U flag
    prob['loading'] = 'hydrostatic'

    return prob
