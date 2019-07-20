from openmdao.api import Group, IndepVarComp, Problem, Component
from floatingse.column import Column, ColumnGeometry
from floatingse.substructure import Substructure, SubstructureGeometry
from floatingse.loading import Loading
from floatingse.map_mooring import MapMooring
from towerse.tower import TowerLeanSE
import numpy as np

    
class FloatingSE(Group):

    def __init__(self):
        super(FloatingSE, self).__init__()

        nSection = 4
        nTower   = 3
        self.nRefine  = 3
        self.nFullSec = self.nRefine*nSection+1
        self.nFullTow = self.nRefine*nTower  +1

        # Need to enter points as nPts (nsection+1), self.nFull
        self.add('tow', TowerLeanSE(nTower+1, self.nFullTow), promotes=['material_density','tower_section_height',
                                                                        'tower_outer_diameter','tower_wall_thickness','tower_outfitting_factor',
                                                                        'tower_buckling_length','max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                                                        'tower_mass','tower_I_base','hub_height','tip_position','hub_cm','downwind',
                                                                        'material_cost_rate','labor_cost_rate','painting_cost_rate'])
        
        # Next do main and ballast columns
        # Ballast columns are replicated from same design in the components
        self.add('main', Column(nSection, self.nFullSec), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                                 'Uref','zref','shearExp','beta','yaw','Uc','Hs','T','cd_usr','cm','loading',
                                                                 'max_draft','max_taper','min_d_to_t','gamma_f','gamma_b','foundation_height',
                                                                 'permanent_ballast_density','bulkhead_mass_factor','buoyancy_tank_mass_factor',
                                                                 'ring_mass_factor','column_mass_factor','outfitting_mass_fraction','ballast_cost_rate',
                                                                 'material_cost_rate','labor_cost_rate','painting_cost_rate','outfitting_cost_rate'])
        self.add('off', Column(nSection, self.nFullSec), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                                'Uref','zref','shearExp','beta','yaw','Uc','Hs','T','cd_usr','cm','loading',
                                                                'max_draft','max_taper','min_d_to_t','gamma_f','gamma_b','foundation_height',
                                                                'permanent_ballast_density','bulkhead_mass_factor','buoyancy_tank_mass_factor',
                                                                'ring_mass_factor','column_mass_factor','outfitting_mass_fraction','ballast_cost_rate',
                                                                'material_cost_rate','labor_cost_rate','painting_cost_rate','outfitting_cost_rate'])

        # Run Semi Geometry for interfaces
        self.add('sg', SubstructureGeometry(self.nFullSec,self.nFullTow), promotes=['*'])

        # Next run MapMooring
        self.add('mm', MapMooring(), promotes=['*'])
        
        # Add in the connecting truss
        self.add('load', Loading(self.nFullSec, self.nFullTow), promotes=['*'])

        # Run main Semi analysis
        self.add('subs', Substructure(self.nFullSec,self.nFullTow), promotes=['*'])

        # Define all input variables from all models
        
        # SemiGeometry
        self.add('radius_to_offset_column', IndepVarComp('radius_to_offset_column', 0.0), promotes=['*'])
        self.add('number_of_offset_columns',  IndepVarComp('number_of_offset_columns', 0), promotes=['*'])
        
        self.add('fairlead_location',          IndepVarComp('fairlead_location', 0.0), promotes=['*'])
        self.add('fairlead_offset_from_shell', IndepVarComp('fairlead_offset_from_shell', 0.0), promotes=['*'])
        self.add('z_offset',                   IndepVarComp('z_offset', 0.0), promotes=['*'])
        self.add('max_draft',                   IndepVarComp('max_draft', 0.0), promotes=['*'])


        # Mooring
        self.add('mooring_line_length',        IndepVarComp('mooring_line_length', 0.0), promotes=['*'])
        self.add('anchor_radius',              IndepVarComp('anchor_radius', 0.0), promotes=['*'])
        self.add('mooring_diameter',           IndepVarComp('mooring_diameter', 0.0), promotes=['*'])
        self.add('number_of_mooring_connections', IndepVarComp('number_of_mooring_connections', 0), promotes=['*'])
        self.add('mooring_lines_per_connection', IndepVarComp('mooring_lines_per_connection', 0), promotes=['*'])
        self.add('mooring_type',               IndepVarComp('mooring_type', 'chain', pass_by_obj=True), promotes=['*'])
        self.add('anchor_type',                IndepVarComp('anchor_type', 'SUCTIONPILE', pass_by_obj=True), promotes=['*'])
        self.add('max_offset',         IndepVarComp('max_offset', 0.0), promotes=['*'])
        self.add('operational_heel',   IndepVarComp('operational_heel', 0.0), promotes=['*'])
        self.add('mooring_cost_rate',          IndepVarComp('mooring_cost_rate', 0.0), promotes=['*'])
        self.add('max_survival_heel',          IndepVarComp('max_survival_heel', 0.0), promotes=['*'])

        # Column
        self.add('permanent_ballast_density',  IndepVarComp('permanent_ballast_density', 0.0), promotes=['*'])
        
        self.add('main_freeboard',             IndepVarComp('main_freeboard', 0.0), promotes=['*'])
        self.add('main_section_height',        IndepVarComp('main_section_height', np.zeros((nSection,))), promotes=['*'])
        self.add('main_outer_diameter',        IndepVarComp('main_outer_diameter', np.zeros((nSection+1,))), promotes=['*'])
        self.add('main_wall_thickness',        IndepVarComp('main_wall_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('main_stiffener_web_height',       IndepVarComp('main_stiffener_web_height', np.zeros((nSection,))), promotes=['*'])
        self.add('main_stiffener_web_thickness',    IndepVarComp('main_stiffener_web_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('main_stiffener_flange_width',     IndepVarComp('main_stiffener_flange_width', np.zeros((nSection,))), promotes=['*'])
        self.add('main_stiffener_flange_thickness', IndepVarComp('main_stiffener_flange_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('main_stiffener_spacing',          IndepVarComp('main_stiffener_spacing', np.zeros((nSection,))), promotes=['*'])
        self.add('main_bulkhead_thickness',             IndepVarComp('main_bulkhead_thickness', np.zeros((nSection+1,))), promotes=['*'])
        self.add('main_permanent_ballast_height',   IndepVarComp('main_permanent_ballast_height', 0.0), promotes=['*'])
        self.add('main_buoyancy_tank_diameter',   IndepVarComp('main_buoyancy_tank_diameter', 0.0), promotes=['*'])
        self.add('main_buoyancy_tank_height',   IndepVarComp('main_buoyancy_tank_height', 0.0), promotes=['*'])
        self.add('main_buoyancy_tank_location',   IndepVarComp('main_buoyancy_tank_location', 0.0), promotes=['*'])

        self.add('offset_freeboard',          IndepVarComp('offset_freeboard', 0.0), promotes=['*'])
        self.add('offset_section_height',     IndepVarComp('offset_section_height', np.zeros((nSection,))), promotes=['*'])
        self.add('offset_outer_diameter',     IndepVarComp('offset_outer_diameter', np.zeros((nSection+1,))), promotes=['*'])
        self.add('offset_wall_thickness',     IndepVarComp('offset_wall_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('offset_stiffener_web_height',       IndepVarComp('offset_stiffener_web_height', np.zeros((nSection,))), promotes=['*'])
        self.add('offset_stiffener_web_thickness',    IndepVarComp('offset_stiffener_web_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('offset_stiffener_flange_width',     IndepVarComp('offset_stiffener_flange_width', np.zeros((nSection,))), promotes=['*'])
        self.add('offset_stiffener_flange_thickness', IndepVarComp('offset_stiffener_flange_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('offset_stiffener_spacing',          IndepVarComp('offset_stiffener_spacing', np.zeros((nSection,))), promotes=['*'])
        self.add('offset_bulkhead_thickness',             IndepVarComp('offset_bulkhead_thickness', np.zeros((nSection+1,))), promotes=['*'])
        self.add('offset_permanent_ballast_height',   IndepVarComp('offset_permanent_ballast_height', 0.0), promotes=['*'])
        self.add('offset_buoyancy_tank_diameter',   IndepVarComp('offset_buoyancy_tank_diameter', 0.0), promotes=['*'])
        self.add('offset_buoyancy_tank_height',   IndepVarComp('offset_buoyancy_tank_height', 0.0), promotes=['*'])
        self.add('offset_buoyancy_tank_location',   IndepVarComp('offset_buoyancy_tank_location', 0.0), promotes=['*'])

        self.add('bulkhead_mass_factor',       IndepVarComp('bulkhead_mass_factor', 0.0), promotes=['*'])
        self.add('ring_mass_factor',           IndepVarComp('ring_mass_factor', 0.0), promotes=['*'])
        self.add('shell_mass_factor',          IndepVarComp('shell_mass_factor', 0.0), promotes=['*'])
        self.add('column_mass_factor',         IndepVarComp('column_mass_factor', 0.0), promotes=['*'])
        self.add('outfitting_mass_fraction',   IndepVarComp('outfitting_mass_fraction', 0.0), promotes=['*'])
        self.add('ballast_cost_rate',          IndepVarComp('ballast_cost_rate', 0.0), promotes=['*'])
        self.add('material_cost_rate',         IndepVarComp('material_cost_rate', 0.0), promotes=['*'])
        self.add('labor_cost_rate',            IndepVarComp('labor_cost_rate', 0.0), promotes=['*'])
        self.add('painting_cost_rate',         IndepVarComp('painting_cost_rate', 0.0), promotes=['*'])
        self.add('outfitting_cost_rate',       IndepVarComp('outfitting_cost_rate', 0.0), promotes=['*'])
        self.add('loading',                    IndepVarComp('loading', val='hydrostatic', pass_by_obj=True), promotes=['*'])
        
        self.add('max_taper_ratio',            IndepVarComp('max_taper_ratio', 0.0), promotes=['*'])
        self.add('min_diameter_thickness_ratio', IndepVarComp('min_diameter_thickness_ratio', 0.0), promotes=['*'])

        # Pontoons
        #self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])

        # Other Constraints
        self.add('wave_period_range_low',   IndepVarComp('wave_period_range_low', 2.0), promotes=['*'])
        self.add('wave_period_range_high',  IndepVarComp('wave_period_range_high', 20.0), promotes=['*'])

        # Connect all input variables from all models
        self.connect('radius_to_offset_column', ['radius_to_offset_column', 'radius_to_offset_column'])

        self.connect('main_freeboard', ['tow.foundation_height', 'main.freeboard', 'main_freeboard'])
        self.connect('main_section_height', 'main.section_height')
        self.connect('main_outer_diameter', 'main.diameter')
        self.connect('main_wall_thickness', 'main.wall_thickness')
        self.connect('z_offset', 'foundation_height')

        self.connect('tow.d_full', ['windLoads.d','tower_d_full']) # includes tower_d_full
        self.connect('tow.t_full', 'tower_t_full')
        self.connect('tow.z_full', ['loadingWind.z','tower_z_full']) # includes tower_z_full
        self.connect('tow.cm.mass','tower_mass_section')
        self.connect('tower_buckling_length','tower_buckling_length')
        self.connect('tow.turbine_mass','main.stack_mass_in')
        self.connect('tow.tower_center_of_mass','tower_center_of_mass')
        self.connect('tow.tower_cost','tower_shell_cost')
        
        self.connect('offset_freeboard', ['off.freeboard','offset_freeboard'])
        self.connect('offset_section_height', 'off.section_height')
        self.connect('offset_outer_diameter', 'off.diameter')
        self.connect('offset_wall_thickness', 'off.wall_thickness')

        self.connect('max_taper_ratio', 'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
        
        # To do: connect these to independent variables
        self.connect('main.windLoads.rho',['off.windLoads.rho','windLoads.rho'])
        self.connect('main.windLoads.mu',['off.windLoads.mu','windLoads.mu'])
        self.connect('main.waveLoads.mu','off.waveLoads.mu')

        
        self.connect('main_stiffener_web_height', 'main.stiffener_web_height')
        self.connect('main_stiffener_web_thickness', 'main.stiffener_web_thickness')
        self.connect('main_stiffener_flange_width', 'main.stiffener_flange_width')
        self.connect('main_stiffener_flange_thickness', 'main.stiffener_flange_thickness')
        self.connect('main_stiffener_spacing', 'main.stiffener_spacing')
        self.connect('main_bulkhead_thickness', 'main.bulkhead_thickness')
        self.connect('main_permanent_ballast_height', 'main.permanent_ballast_height')
        self.connect('main.L_stiffener','main_buckling_length')
        self.connect('main_buoyancy_tank_diameter', 'main.buoyancy_tank_diameter')
        self.connect('main_buoyancy_tank_height', 'main.buoyancy_tank_height')
        self.connect('main_buoyancy_tank_location', 'main.buoyancy_tank_location')

        self.connect('offset_stiffener_web_height', 'off.stiffener_web_height')
        self.connect('offset_stiffener_web_thickness', 'off.stiffener_web_thickness')
        self.connect('offset_stiffener_flange_width', 'off.stiffener_flange_width')
        self.connect('offset_stiffener_flange_thickness', 'off.stiffener_flange_thickness')
        self.connect('offset_stiffener_spacing', 'off.stiffener_spacing')
        self.connect('offset_bulkhead_thickness', 'off.bulkhead_thickness')
        self.connect('offset_permanent_ballast_height', 'off.permanent_ballast_height')
        self.connect('off.L_stiffener','offset_buckling_length')
        self.connect('offset_buoyancy_tank_diameter', 'off.buoyancy_tank_diameter')
        self.connect('offset_buoyancy_tank_height', 'off.buoyancy_tank_height')
        self.connect('offset_buoyancy_tank_location', 'off.buoyancy_tank_location')
        
        self.connect('bulkhead_mass_factor', 'buoyancy_tank_mass_factor')
        self.connect('shell_mass_factor', ['main.cyl_mass.outfitting_factor', 'off.cyl_mass.outfitting_factor'])

        self.connect('main.z_full', ['main_z_nodes', 'main_z_full'])
        self.connect('main.d_full', 'main_d_full')
        self.connect('main.t_full', 'main_t_full')

        self.connect('off.z_full', ['offset_z_nodes', 'offset_z_full'])
        self.connect('off.d_full', 'offset_d_full')
        self.connect('off.t_full', 'offset_t_full')

        self.connect('max_offset_restoring_force', 'mooring_surge_restoring_force')
        self.connect('operational_heel_restoring_force', 'mooring_pitch_restoring_force')
        
        self.connect('main.z_center_of_mass', 'main_center_of_mass')
        self.connect('main.z_center_of_buoyancy', 'main_center_of_buoyancy')
        self.connect('main.I_column', 'main_moments_of_inertia')
        self.connect('main.Iwater', 'main_Iwaterplane')
        self.connect('main.Awater', 'main_Awaterplane')
        self.connect('main.displaced_volume', 'main_displaced_volume')
        self.connect('main.hydrostatic_force', 'main_hydrostatic_force')
        self.connect('main.column_added_mass', 'main_added_mass')
        self.connect('main.column_total_mass', 'main_mass')
        self.connect('main.column_total_cost', 'main_cost')
        self.connect('main.variable_ballast_interp_zpts', 'water_ballast_zpts_vector')
        self.connect('main.variable_ballast_interp_radius', 'water_ballast_radius_vector')
        self.connect('main.Px', 'main_Px')
        self.connect('main.Py', 'main_Py')
        self.connect('main.Pz', 'main_Pz')
        self.connect('main.qdyn', 'main_qdyn')

        self.connect('off.z_center_of_mass', 'offset_center_of_mass')
        self.connect('off.z_center_of_buoyancy', 'offset_center_of_buoyancy')
        self.connect('off.I_column', 'offset_moments_of_inertia')
        self.connect('off.Iwater', 'offset_Iwaterplane')
        self.connect('off.Awater', 'offset_Awaterplane')
        self.connect('off.displaced_volume', 'offset_displaced_volume')
        self.connect('off.hydrostatic_force', 'offset_hydrostatic_force')
        self.connect('off.column_added_mass', 'offset_added_mass')
        self.connect('off.column_total_mass', 'offset_mass')
        self.connect('off.column_total_cost', 'offset_cost')
        self.connect('off.Px', 'offset_Px')
        self.connect('off.Py', 'offset_Py')
        self.connect('off.Pz', 'offset_Pz')
        self.connect('off.qdyn', 'offset_qdyn')
        self.connect('off.draft', 'offset_draft')

         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['check_form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr




def commonVars(prob, nsection):
    # Variables common to both examples

    # Set environment to that used in OC4 testing campaign
    prob['shearExp']    = 0.11   # Shear exponent in wind power law
    prob['cm']          = 2.0    # Added mass coefficient
    prob['Uc']          = 0.0    # Mean current speed
    prob['z0']          = 0.0    # Water line
    prob['yaw']         = 0.0    # Turbine yaw angle
    prob['beta']        = 0.0    # Wind beta angle
    prob['cd_usr']      = np.inf # Compute drag coefficient

    # Wind and water properties
    prob['main.windLoads.rho'] = 1.226   # Density of air [kg/m^3]
    prob['main.windLoads.mu']  = 1.78e-5 # Viscosity of air [kg/m/s]
    prob['water_density']      = 1025.0  # Density of water [kg/m^3]
    prob['main.waveLoads.mu']  = 1.08e-3 # Viscosity of water [kg/m/s]
    
    # Material properties
    prob['material_density'] = 7850.0          # Steel [kg/m^3]
    prob['E']                = 200e9           # Young's modulus [N/m^2]
    prob['G']                = 79.3e9          # Shear modulus [N/m^2]
    prob['yield_stress']     = 3.45e8          # Elastic yield stress [N/m^2]
    prob['nu']               = 0.26            # Poisson's ratio
    prob['permanent_ballast_density'] = 4492.0 # [kg/m^3]

    # Mass and cost scaling factors
    prob['bulkhead_mass_factor']     = 1.0     # Scaling for unaccounted bulkhead mass
    prob['ring_mass_factor']         = 1.0     # Scaling for unaccounted stiffener mass
    prob['shell_mass_factor']        = 1.0     # Scaling for unaccounted shell mass
    prob['column_mass_factor']       = 1.05    # Scaling for unaccounted column mass
    prob['outfitting_mass_fraction'] = 0.06    # Fraction of additional outfitting mass for each column
    prob['ballast_cost_rate']        = 0.1   # Cost factor for ballast mass [$/kg]
    prob['material_cost_rate']       = 1.1  # Cost factor for column mass [$/kg]
    prob['labor_cost_rate']          = 1.0  # Cost factor for labor time [$/min]
    prob['painting_cost_rate']       = 14.4  # Cost factor for column surface finishing [$/m^2]
    prob['outfitting_cost_rate']     = 1.5*1.1  # Cost factor for outfitting mass [$/kg]
    prob['mooring_cost_rate']        = 1.1     # Cost factor for mooring mass [$/kg]
    
    # Safety factors
    prob['gamma_f'] = 1.35 # Safety factor on loads
    prob['gamma_b'] = 1.1  # Safety factor on buckling
    prob['gamma_m'] = 1.1  # Safety factor on materials
    prob['gamma_n'] = 1.0  # Safety factor on consequence of failure
    prob['gamma_fatigue'] = 1.755 # Not used

    # Mooring parameters
    prob['number_of_mooring_connections'] = 3             # Evenly spaced around structure
    prob['mooring_lines_per_connection'] = 1             # Evenly spaced around structure
    prob['mooring_type']               = 'chain'       # Options are chain, nylon, polyester, fiber, or iwrc
    prob['anchor_type']                = 'DRAGEMBEDMENT' # Options are SUCTIONPILE or DRAGEMBEDMENT
    
    # Porperties of turbine tower
    prob['hub_height']              = 77.6                              # Length from tower main to top (not including freeboard) [m]
    prob['tower_section_height']    = 77.6/nsection * np.ones(nsection) # Length of each tower section [m]
    prob['tower_outer_diameter']    = np.linspace(6.5, 3.87, nsection+1) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_wall_thickness']    = np.linspace(0.027, 0.019, nsection) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_buckling_length']   = 30.0                              # Tower buckling reinforcement spacing [m]
    prob['tower_outfitting_factor'] = 1.07                              # Scaling for unaccounted tower mass in outfitting

    # Properties of rotor-nacelle-assembly (RNA)
    prob['rna_mass']   = 350e3 # Mass [kg]
    prob['rna_I']      = 1e5*np.array([1149.307, 220.354, 187.597, 0, 5.037, 0]) # Moment of intertia (xx,yy,zz,xy,xz,yz) [kg/m^2]
    prob['rna_cg']     = np.array([-1.132, 0, 0.509])                       # Offset of RNA center of mass from tower top (x,y,z) [m]
    # Max thrust
    prob['rna_force']  = np.array([1284744.196, 0, -112400.5527])           # Net force acting on RNA (x,y,z) [N]
    prob['rna_moment'] = np.array([3963732.762, 896380.8464, -346781.682]) # Net moment acting on RNA (x,y,z) [N*m]
    # Max wind speed
    #prob['rna_force']  = np.array([188038.8045, 0,  -16451.2637]) # Net force acting on RNA (x,y,z) [N]
    #prob['rna_moment'] = np.array([0.0, 131196.8431,  0.0]) # Net moment acting on RNA (x,y,z) [N*m]
    
    # Mooring constraints
    prob['max_draft'] = 150.0 # Max surge/sway offset [m]      
    prob['max_offset'] = 100.0 # Max surge/sway offset [m]      
    prob['operational_heel']   = 10.0 # Max heel (pitching) angle [deg]

    # Design constraints
    prob['max_taper_ratio'] = 0.2                # For manufacturability of rolling steel
    prob['min_diameter_thickness_ratio'] = 120.0 # For weld-ability
    prob['connection_ratio_max']      = 0.25 # For welding pontoons to columns

    # API 2U flag
    prob['loading'] = 'hydrostatic'
    
    return prob


def sparExample():
    # Number of sections to be used in the design
    nsection = 5

    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem(root=FloatingSE(nsection))
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
    prob['Hs']        = 10.8   # Significant wave height [m]
    prob['T']           = 9.8    # Wave period [s]
    prob['Uref']        = 11.0   # Wind reference speed [m/s]
    prob['zref']        = 119.0  # Wind reference height [m]

    # Column geometry
    prob['main_permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
    prob['main_freeboard']                = 10.0 # Height extension above waterline [m]
    prob['main_section_height'] = np.array([49.0, 29.0, 30.0, 8.0, 14.0])  # Length of each section [m]
    prob['main_outer_diameter'] = np.array([9.4, 9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['main_wall_thickness'] = 0.05 * np.ones(nsection)               # Shell thickness at each section node (linear lofting between) [m]
    prob['main_bulkhead_thickness'] = 0.05*np.array([1,1,0,0,0,0]) # Locations/thickness of internal bulkheads at section interfaces [m]
    
    # Column ring stiffener parameters
    prob['main_stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_flange_width']     = 0.10 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]
    
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
    prob['offset_section_height'] = 1.0 * np.ones(nsection)
    prob['offset_outer_diameter'] = 5.0 * np.ones(nsection+1)
    prob['offset_wall_thickness'] = 0.1 * np.ones(nsection)
    prob['offset_permanent_ballast_height'] = 0.1
    prob['offset_stiffener_web_height'] = 0.1 * np.ones(nsection)
    prob['offset_stiffener_web_thickness'] =  0.1 * np.ones(nsection)
    prob['offset_stiffener_flange_width'] =  0.1 * np.ones(nsection)
    prob['offset_stiffener_flange_thickness'] =  0.1 * np.ones(nsection)
    prob['offset_stiffener_spacing'] =  0.1 * np.ones(nsection)
    prob['pontoon_outer_diameter'] = 1.0
    prob['pontoon_wall_thickness'] = 0.1
    
    prob.run()
    


def semiExample():
    # Number of sections to be used in the design
    nsection = 5

    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem(root=FloatingSE(nsection))
    prob.setup()

    # Variables common to both examples
    prob = commonVars(prob, nsection)
    
    # Add in offset columns and truss elements
    prob['number_of_offset_columns'] = 3
    prob['cross_attachment_pontoons_int']   = 1 # Lower-Upper main-to-offset connecting cross braces
    prob['lower_attachment_pontoons_int']   = 1 # Lower main-to-offset connecting pontoons
    prob['upper_attachment_pontoons_int']   = 1 # Upper main-to-offset connecting pontoons
    prob['lower_ring_pontoons_int']         = 1 # Lower ring of pontoons connecting offset columns
    prob['upper_ring_pontoons_int']         = 1 # Upper ring of pontoons connecting offset columns
    prob['outer_cross_pontoons_int']        = 1 # Auxiliary ring connecting V-cross braces

    # Set environment to that used in OC4 testing campaign
    prob['water_depth'] = 200.0  # Distance to sea floor [m]
    prob['Hs']        = 10.8   # Significant wave height [m]
    prob['T']           = 9.8    # Wave period [s]
    prob['Uref']        = 11.0   # Wind reference speed [m/s]
    prob['zref']        = 119.0  # Wind reference height [m]

    # Column geometry
    prob['main_permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
    prob['main_freeboard']                = 10.0 # Height extension above waterline [m]
    prob['main_section_height'] = np.array([36.0, 36.0, 36.0, 8.0, 14.0])  # Length of each section [m]
    prob['main_outer_diameter'] = np.array([9.4, 9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['main_wall_thickness'] = 0.05 * np.ones(nsection)               # Shell thickness at each section node (linear lofting between) [m]
    prob['main_bulkhead_thickness'] = 0.05*np.array([1,1,0,0,0,0]) # Locations/thickness of internal bulkheads at section interfaces [m]

    # Auxiliary column geometry
    prob['radius_to_offset_column']         = 33.333 * np.cos(np.pi/6) # Centerline of main column to centerline of offset column [m]
    prob['offset_permanent_ballast_height'] = 0.1                      # Height above keel for permanent ballast [m]
    prob['offset_freeboard']                = 12.0                     # Height extension above waterline [m]
    prob['offset_section_height']           = np.array([6.0, 0.1, 7.9, 8.0, 10]) # Length of each section [m]
    prob['offset_outer_diameter']           = np.array([24, 24, 12, 12, 12, 12]) # Diameter at each section node (linear lofting between) [m]
    prob['offset_wall_thickness']           = 0.06 * np.ones(nsection)         # Shell thickness at each section node (linear lofting between) [m]

    # Column ring stiffener parameters
    prob['main_stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_flange_width']     = 0.10 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['main_stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]

    # Auxiliary column ring stiffener parameters
    prob['offset_stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['offset_stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['offset_stiffener_flange_width']     = 0.01 * np.ones(nsection) # (by section) [m]
    prob['offset_stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['offset_stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]

    # Pontoon parameters
    prob['pontoon_outer_diameter']    = 3.2    # Diameter of all pontoon/truss elements [m]
    prob['pontoon_wall_thickness']    = 0.0175 # Thickness of all pontoon/truss elements [m]
    prob['main_pontoon_attach_lower'] = -20.0  # Lower z-coordinate on main where truss attaches [m]
    prob['main_pontoon_attach_upper'] = 10.0   # Upper z-coordinate on main where truss attaches [m]
    
    # Mooring parameters
    prob['mooring_diameter']           = 0.0766        # Diameter of mooring line/chain [m]
    prob['fairlead']                   = 14.0          # Distance below waterline for attachment [m]
    prob['fairlead_offset_from_shell'] = 0.5           # Offset from shell surface for mooring attachment [m]
    prob['mooring_line_length']        = 835.5+300         # Unstretched mooring line length
    prob['anchor_radius']              = 837.6+300.0         # Distance from centerline to sea floor landing [m]
    prob['fairlead_support_outer_diameter'] = 3.2    # Diameter of all fairlead support elements [m]
    prob['fairlead_support_wall_thickness'] = 0.0175 # Thickness of all fairlead support elements [m]
    
    prob.run()
    
    '''
    f = open('deriv_semi.dat','w')
    out = prob.check_total_derivatives(f)
    #out = prob.check_partial_derivatives(f, compact_print=True)
    f.close()
    tol = 1e-4
    for comp in out.keys():
        for k in out[comp].keys():
            if ( (out[comp][k]['rel error'][0] > tol) and (out[comp][k]['abs error'][0] > tol) ):
                print k
    '''

if __name__ == "__main__":
    from openmdao.api import Problem
    import sys

    if len(sys.argv) > 1 and sys.argv[1].lower() in ['spar','column','col','oc3']:
        sparExample()
    else:
        semiExample()
        
