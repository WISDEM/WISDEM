from openmdao.api import Group, IndepVarComp, Problem, ExplicitComponent
from wisdem.floatingse.column import Column, ColumnGeometry
from wisdem.floatingse.substructure import Substructure, SubstructureGeometry
from wisdem.floatingse.loading import Loading
from wisdem.floatingse.map_mooring import MapMooring
from wisdem.towerse.tower import TowerLeanSE
import numpy as np

    
class FloatingSE(Group):

    def initialize(self):
        self.options.declare('nSection', default=4)
        self.options.declare('nTower', default=3)
        self.options.declare('nRefine', default=3)
        self.options.declare('topLevelFlag', default=True)

    def setup(self):

        nSection = self.options['nSection']
        nTower   = self.options['nTower']
        nRefine  = self.options['nRefine']
        topLevelFlag = self.options['topLevelFlag']
        nFullSec = nRefine*nSection+1
        nFullTow = nRefine*nTower  +1

        self.add_subsystem('tow', TowerLeanSE(nPoints=nTower+1, nFull=nFullTow, topLevelFlag=False), promotes=['material_density','tower_section_height',
                                                                        'tower_outer_diameter','tower_wall_thickness','tower_outfitting_factor',
                                                                        'tower_buckling_length','max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                                                        'tower_mass','tower_I_base','hub_height',#'tip_position','hub_cm','downwind',
                                                                        'material_cost_rate','labor_cost_rate','painting_cost_rate'])
        
        # Next do main and ballast columns
        # Ballast columns are replicated from same design in the components
        self.add_subsystem('main', Column(nSection=nSection, nFull=nFullSec, topLevelFlag=False), promotes=['water_density','E','nu','yield_stress','z0',
                                                                                                            'Uref','zref','shearExp','yaw','Uc','Hs','hmax','T','cd_usr','cm','loading',
                                                                 'max_draft','max_taper','min_d_to_t','gamma_f','gamma_b','foundation_height',
                                                                 'permanent_ballast_density','bulkhead_mass_factor','buoyancy_tank_mass_factor',
                                                                 'ring_mass_factor','column_mass_factor','outfitting_mass_fraction','ballast_cost_rate',
                                                                 'material_cost_rate','labor_cost_rate','painting_cost_rate','outfitting_cost_rate'])
        self.add_subsystem('off', Column(nSection=nSection, nFull=nFullSec, topLevelFlag=False), promotes=['water_density','E','nu','yield_stress','z0',
                                                                                                           'Uref','zref','shearExp','yaw','Uc','Hs','hmax','T','cd_usr','cm','loading',
                                                                'max_draft','max_taper','min_d_to_t','gamma_f','gamma_b','foundation_height',
                                                                'permanent_ballast_density','bulkhead_mass_factor','buoyancy_tank_mass_factor',
                                                                'ring_mass_factor','column_mass_factor','outfitting_mass_fraction','ballast_cost_rate',
                                                                'material_cost_rate','labor_cost_rate','painting_cost_rate','outfitting_cost_rate'])

        # Run Semi Geometry for interfaces
        self.add_subsystem('sg', SubstructureGeometry(nFull=nFullSec, nFullTow=nFullTow), promotes=['*'])

        # Next run MapMooring
        self.add_subsystem('mm', MapMooring(), promotes=['*'])
        
        # Add in the connecting truss
        self.add_subsystem('load', Loading(nFull=nFullSec, nFullTow=nFullTow), promotes=['*'])

        # Run main Semi analysis
        self.add_subsystem('subs', Substructure(nFull=nFullSec, nFullTow=nFullTow), promotes=['*'])

        # Independent variables that may be duplicated at higher levels of aggregation
        if topLevelFlag:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('hub_height', 0.0, units='m')
            sharedIndeps.add_output('material_density', 0.0, units='kg/m**3')
            sharedIndeps.add_output('air_density', 0.0, units='kg/m**3')
            sharedIndeps.add_output('air_viscosity', 0.0, units='kg/m/s')
            sharedIndeps.add_output('shearExp', 0.0)
            sharedIndeps.add_output('wind_reference_height', 0.0, units='m')
            sharedIndeps.add_output('wind_reference_speed', 0.0, units='m/s')
            sharedIndeps.add_output('wind_z0', 0.0, units='m')
            sharedIndeps.add_output('wind_beta', 0.0, units='deg')
            sharedIndeps.add_output('cd_usr', -1.0)
            sharedIndeps.add_output('cm', 0.0)
            sharedIndeps.add_output('Uc', 0.0, units='m/s')
            sharedIndeps.add_output('water_depth', 0.0, units='m')
            sharedIndeps.add_output('water_density', 0.0, units='kg/m**3')
            sharedIndeps.add_output('water_viscosity', 0.0, units='kg/m/s')
            sharedIndeps.add_output('significant_wave_height', 0.0, units='m')
            sharedIndeps.add_output('significant_wave_period', 0.0, units='s')
            sharedIndeps.add_output('wave_beta', 0.0, units='deg')
            sharedIndeps.add_output('wave_z0', 0.0, units='m')
            sharedIndeps.add_output('yaw', 0.0, units='deg')
            sharedIndeps.add_output('E', 0.0, units='N/m**2')
            sharedIndeps.add_output('G', 0.0, units='N/m**2')
            sharedIndeps.add_output('nu', 0.0)
            sharedIndeps.add_output('yield_stress', 0.0, units='N/m**2')
            sharedIndeps.add_output('gamma_f', 0.0)
            sharedIndeps.add_output('gamma_m', 0.0)
            sharedIndeps.add_output('gamma_n', 0.0)
            sharedIndeps.add_output('gamma_b', 0.0)
            sharedIndeps.add_output('gamma_fatigue', 0.0)
            sharedIndeps.add_output('DC', 0.0)
            sharedIndeps.add_discrete_output('shear', True)
            sharedIndeps.add_discrete_output('geom', False)
            sharedIndeps.add_discrete_output('nM', 2)
            sharedIndeps.add_discrete_output('Mmethod', 1)
            sharedIndeps.add_discrete_output('lump', 0)
            sharedIndeps.add_output('tol', 0.0)
            sharedIndeps.add_output('shift', 0.0)
            sharedIndeps.add_output('life', 0.0)
            sharedIndeps.add_output('m_SN', 0.0)
            sharedIndeps.add_output('rna_mass', 0.0, units='kg')
            sharedIndeps.add_output('rna_I', np.zeros(6), units='kg*m**2')
            sharedIndeps.add_output('rna_cg', np.zeros(3), units='m')
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])

        # Define all input variables from all models
        floatingIndeps = IndepVarComp()
        
        # SemiGeometry
        floatingIndeps.add_output('radius_to_offset_column', 0.0, units='m')
        floatingIndeps.add_output('number_of_offset_columns', 0)
        
        floatingIndeps.add_output('fairlead_location', 0.0)
        floatingIndeps.add_output('fairlead_offset_from_shell', 0.0, units='m')
        floatingIndeps.add_output('z_offset', 0.0, units='m')
        floatingIndeps.add_output('max_draft', 0.0, units='m')

        # Mooring
        floatingIndeps.add_output('mooring_line_length', 0.0, units='m')
        floatingIndeps.add_output('anchor_radius', 0.0, units='m')
        floatingIndeps.add_output('mooring_diameter', 0.0, units='m')
        floatingIndeps.add_output('number_of_mooring_connections', 0)
        floatingIndeps.add_output('mooring_lines_per_connection', 0)
        floatingIndeps.add_discrete_output('mooring_type', 'chain')
        floatingIndeps.add_discrete_output('anchor_type', 'SUCTIONPILE')
        floatingIndeps.add_output('max_offset', 0.0, units='m')
        floatingIndeps.add_output('operational_heel', 0.0, units='deg')
        floatingIndeps.add_output('mooring_cost_factor', 0.0)
        floatingIndeps.add_output('max_survival_heel', 0.0, units='deg')

        # Column
        floatingIndeps.add_output('permanent_ballast_density', 0.0, units='kg/m**3')

        floatingIndeps.add_output('bulkhead_mass_factor', 0.0)
        floatingIndeps.add_output('ring_mass_factor', 0.0)
        floatingIndeps.add_output('shell_mass_factor', 0.0)
        floatingIndeps.add_output('column_mass_factor', 0.0)
        floatingIndeps.add_output('outfitting_mass_fraction', 0.0)
        floatingIndeps.add_output('ballast_cost_rate', 0.0, units='USD/kg')
        floatingIndeps.add_output('material_cost_rate', 0.0, units='USD/kg')
        floatingIndeps.add_output('labor_cost_rate', 0.0, units='USD/min')
        floatingIndeps.add_output('painting_cost_rate', 0.0, units='USD/m**2')
        floatingIndeps.add_output('outfitting_cost_rate', 0.0, units='USD/kg')
        floatingIndeps.add_discrete_output('loading', 'hydrostatic')
        
        floatingIndeps.add_output('max_taper_ratio', 0.0)
        floatingIndeps.add_output('min_diameter_thickness_ratio', 0.0)

        # Other Constraints
        floatingIndeps.add_output('wave_period_range_low', 2.0, units='s')
        floatingIndeps.add_output('wave_period_range_high', 20.0, units='s')


        self.add_subsystem('floatingIndeps', floatingIndeps, promotes=['*'])
        
        # Connect all input variables from all models
        self.connect('main.freeboard', 'tow.foundation_height')
        self.connect('z_offset', 'foundation_height')

        self.connect('tow.d_full', ['windLoads.d','tower_d_full']) # includes tower_d_full
        self.connect('tow.t_full', 'tower_t_full')
        self.connect('tow.z_full', ['loadingWind.z','windLoads.z','tower_z_full']) # includes tower_z_full
        self.connect('tow.cm.mass','tower_mass_section')
        self.connect('tow.turbine_mass','main.stack_mass_in')
        self.connect('tow.tower_center_of_mass','tower_center_of_mass')
        self.connect('tow.tower_raw_cost','tower_shell_cost')
        
        self.connect('max_taper_ratio', 'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
        
        # To do: connect these to independent variables
        if topLevelFlag:
            self.connect('air_density',['main.windLoads.rho','off.windLoads.rho','windLoads.rho'])
            self.connect('air_viscosity',['main.windLoads.mu','off.windLoads.mu','windLoads.mu'])
            self.connect('water_density',['main.wave.rho','main.waveLoads.rho','off.wave.rho','off.waveLoads.rho'])
            self.connect('water_viscosity',['main.waveLoads.mu','off.waveLoads.mu'])
            self.connect('water_depth',['main.wave.z_floor','off.wave.z_floor'])
            self.connect('wave_beta',['main.waveLoads.beta','off.waveLoads.beta'])
            self.connect('wave_z0',['main.wave.z_surface','off.wave.z_surface'])
            self.connect('wind_z0','z0')
            self.connect('significant_wave_height',['Hs', 'hmax'])
            self.connect('significant_wave_period','T')
            self.connect('wind_reference_height','zref')
            self.connect('wind_reference_speed','Uref')
            
        
        self.connect('main.L_stiffener','main_buckling_length')
        self.connect('off.L_stiffener','offset_buckling_length')
        
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




def commonVars(prob, nsection):
    # Variables common to both examples

    # Set environment to that used in OC4 testing campaign
    prob['shearExp']    = 0.11   # Shear exponent in wind power law
    prob['cm']          = 2.0    # Added mass coefficient
    prob['Uc']          = 0.0    # Mean current speed
    prob['wind_z0']     = 0.0    # Water line
    prob['yaw']         = 0.0    # Turbine yaw angle
    prob['beta']        = 0.0    # Wind beta angle
    prob['cd_usr']      = -1.0 # Compute drag coefficient

    # Wind and water properties
    prob['air_density'] = 1.226   # Density of air [kg/m^3]
    prob['air_viscosity']  = 1.78e-5 # Viscosity of air [kg/m/s]
    prob['water_density']      = 1025.0  # Density of water [kg/m^3]
    prob['water_viscosity']  = 1.08e-3 # Viscosity of water [kg/m/s]
    
    # Material properties
    prob['material_density'] = prob['main.material_density'] = prob['off.material_density'] = 7850.0          # Steel [kg/m^3]
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
    prob['mooring_cost_factor']      = 1.1     # Cost factor for mooring mass [$/kg]
    
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
    nTower = prob.model.options['nTower']
    prob['hub_height']              = 77.6                              # Length from tower main to top (not including freeboard) [m]
    prob['tower_section_height']    = 77.6/nTower * np.ones(nTower) # Length of each tower section [m]
    prob['tower_outer_diameter']    = np.linspace(6.5, 3.87, nTower+1) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_wall_thickness']    = np.linspace(0.027, 0.019, nTower) # Diameter at each tower section node (linear lofting between) [m]
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
    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem()
    prob.model = FloatingSE()
    prob.setup()

    # Variables common to both examples
    nsection = prob.model.options['nSection']
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
    prob['significant_wave_height']        = 10.8   # Significant wave height [m]
    prob['significant_wave_period']           = 9.8    # Wave period [s]
    prob['wind_reference_speed']        = 11.0   # Wind reference speed [m/s]
    prob['wind_reference_height']        = 119.0  # Wind reference height [m]

    # Column geometry
    prob['main.permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
    prob['main.freeboard']                = 10.0 # Height extension above waterline [m]
    prob['main.section_height'] = np.array([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
    prob['main.outer_diameter'] = np.array([9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['main.wall_thickness'] = 0.05 * np.ones(nsection)               # Shell thickness at each section node (linear lofting between) [m]
    prob['main.bulkhead_thickness'] = 0.05*np.array([1, 1, 0, 1, 0]) # Locations/thickness of internal bulkheads at section interfaces [m]
    
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
    prob['off.freeboard'] = 0.1
    prob['off.section_height'] = 1.0 * np.ones(nsection)
    prob['off.outer_diameter'] = 5.0 * np.ones(nsection+1)
    prob['off.wall_thickness'] = 0.1 * np.ones(nsection)
    prob['off.permanent_ballast_height'] = 0.1
    prob['off.stiffener_web_height'] = 0.1 * np.ones(nsection)
    prob['off.stiffener_web_thickness'] =  0.1 * np.ones(nsection)
    prob['off.stiffener_flange_width'] =  0.1 * np.ones(nsection)
    prob['off.stiffener_flange_thickness'] =  0.1 * np.ones(nsection)
    prob['off.stiffener_spacing'] =  0.1 * np.ones(nsection)
    prob['pontoon_outer_diameter'] = 1.0
    prob['pontoon_wall_thickness'] = 0.1
    
    prob.run_model()
    


def semiExample():
    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem()
    prob.model = FloatingSE()
    prob.setup()

    # Variables common to both examples
    nsection = prob.model.options['nSection']
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
    prob['significant_wave_height']        = 10.8   # Significant wave height [m]
    prob['significant_wave_period']           = 9.8    # Wave period [s]
    prob['wind_reference_speed']        = 11.0   # Wind reference speed [m/s]
    prob['wind_reference_height']        = 119.0  # Wind reference height [m]

    # Column geometry
    prob['main.permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
    prob['main.freeboard']                = 10.0 # Height extension above waterline [m]
    prob['main.section_height'] = np.array([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
    prob['main.outer_diameter'] = np.array([9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['main.wall_thickness'] = 0.05 * np.ones(nsection)               # Shell thickness at each section node (linear lofting between) [m]
    prob['main.bulkhead_thickness'] = 0.05*np.array([1, 1, 0, 1, 0]) # Locations/thickness of internal bulkheads at section interfaces [m]

    # Auxiliary column geometry
    prob['radius_to_offset_column']         = 33.333 * np.cos(np.pi/6) # Centerline of main column to centerline of offset column [m]
    prob['off.permanent_ballast_height'] = 0.1                      # Height above keel for permanent ballast [m]
    prob['off.freeboard']                = 12.0                     # Height extension above waterline [m]
    prob['off.section_height']           = np.array([6.0, 0.1, 15.9, 10]) # Length of each section [m]
    prob['off.outer_diameter']           = np.array([24, 24, 12, 12, 12]) # Diameter at each section node (linear lofting between) [m]
    prob['off.wall_thickness']           = 0.06 * np.ones(nsection)         # Shell thickness at each section node (linear lofting between) [m]

    # Column ring stiffener parameters
    prob['main.stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['main.stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['main.stiffener_flange_width']     = 0.10 * np.ones(nsection) # (by section) [m]
    prob['main.stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['main.stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]

    # Auxiliary column ring stiffener parameters
    prob['off.stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['off.stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['off.stiffener_flange_width']     = 0.01 * np.ones(nsection) # (by section) [m]
    prob['off.stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['off.stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]

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
    
    prob.run_model()
    
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
        
