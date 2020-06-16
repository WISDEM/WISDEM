import openmdao.api as om
from wisdem.floatingse.column import Column, ColumnGeometry
from wisdem.floatingse.substructure import Substructure, SubstructureGeometry
from wisdem.floatingse.loading import Loading
from wisdem.floatingse.map_mooring import MapMooring
from wisdem.towerse.tower import TowerLeanSE
import numpy as np

from wisdem.commonse.vertical_cylinder import get_nfull

class TempVec(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('n_height')
        
    def setup(self):
        n_height  = self.options['n_height']
        n_full    = get_nfull(n_height)
        self.add_input('rho', 0.0, units='kg/m**3')
        self.add_input('unit_cost', 0.0, units='USD/kg')
        self.add_output('rho_vec', np.zeros(n_full-1), units='kg/m**3')
        self.add_output('unit_cost_vec', np.zeros(n_full-1), units='USD/kg')
        
    def compute(self, inputs, outputs):
        npts    = get_nfull(self.options['n_height']) - 1
        outputs['rho_vec'] = inputs['rho']*np.ones(npts)
        outputs['unit_cost_vec'] = inputs['unit_cost']*np.ones(npts)
    
class FloatingSE(om.Group):

    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('topLevelFlag', default=True)

    def setup(self):
        opt = self.options['analysis_options']['platform']
        n_height_main = opt['columns']['main']['n_height']
        n_height_off  = opt['columns']['offset']['n_height']
        n_height_tow  = self.options['analysis_options']['tower']['n_height']
        n_bulk_main   = opt['columns']['main']['n_bulkhead']
        n_bulk_off    = opt['columns']['offset']['n_bulkhead']
        topLevelFlag  = self.options['topLevelFlag']

        # Define all input variables from all models
        ivc = om.IndepVarComp()
        
        # SemiGeometry
        ivc.add_output('radius_to_offset_column', 0.0, units='m')
        ivc.add_output('number_of_offset_columns', 0)
        
        ivc.add_output('fairlead_location', 0.0)
        ivc.add_output('fairlead_offset_from_shell', 0.0, units='m')
        ivc.add_output('max_draft', 0.0, units='m')
        ivc.add_output('main_freeboard', 0.0, units='m') # Have to add here because cannot promote ivc from Column before needed by tower.  Grr
        ivc.add_output('offset_freeboard', 0.0, units='m')

        # Mooring
        ivc.add_output('mooring_line_length', 0.0, units='m')
        ivc.add_output('anchor_radius', 0.0, units='m')
        ivc.add_output('mooring_diameter', 0.0, units='m')
        ivc.add_output('number_of_mooring_connections', 0)
        ivc.add_output('mooring_lines_per_connection', 0)
        ivc.add_discrete_output('mooring_type', 'chain')
        ivc.add_discrete_output('anchor_type', 'SUCTIONPILE')
        ivc.add_output('max_offset', 0.0, units='m')
        ivc.add_output('operational_heel', 0.0, units='deg')
        ivc.add_output('mooring_cost_factor', 0.0)
        ivc.add_output('max_survival_heel', 0.0, units='deg')

        # Column
        ivc.add_output('permanent_ballast_density', 0.0, units='kg/m**3')
        ivc.add_output('outfitting_factor', 0.0)
        ivc.add_output('ballast_cost_rate', 0.0, units='USD/kg')
        ivc.add_output('unit_cost', 0.0, units='USD/kg')
        ivc.add_output('labor_cost_rate', 0.0, units='USD/min')
        ivc.add_output('painting_cost_rate', 0.0, units='USD/m**2')
        ivc.add_output('outfitting_cost_rate', 0.0, units='USD/kg')
        ivc.add_discrete_output('loading', 'hydrostatic')
        
        ivc.add_output('max_taper_ratio', 0.0)
        ivc.add_output('min_diameter_thickness_ratio', 0.0)

        # Other Constraints
        ivc.add_output('wave_period_range_low', 2.0, units='s')
        ivc.add_output('wave_period_range_high', 20.0, units='s')
        self.add_subsystem('ivc', ivc, promotes=['*'])

        # Independent variables that may be duplicated at higher levels of aggregation
        if topLevelFlag:
            sharedIndeps = om.IndepVarComp()
            sharedIndeps.add_output('hub_height', 0.0, units='m')
            sharedIndeps.add_output('rho', 0.0, units='kg/m**3')
            sharedIndeps.add_output('rho_air', 0.0, units='kg/m**3')
            sharedIndeps.add_output('mu_air', 0.0, units='kg/m/s')
            sharedIndeps.add_output('shearExp', 0.0)
            sharedIndeps.add_output('wind_reference_height', 0.0, units='m')
            sharedIndeps.add_output('wind_reference_speed', 0.0, units='m/s')
            sharedIndeps.add_output('wind_z0', 0.0, units='m')
            sharedIndeps.add_output('beta_wind', 0.0, units='deg')
            sharedIndeps.add_output('cd_usr', -1.0)
            sharedIndeps.add_output('cm', 0.0)
            sharedIndeps.add_output('Uc', 0.0, units='m/s')
            sharedIndeps.add_output('water_depth', 0.0, units='m')
            sharedIndeps.add_output('rho_water', 0.0, units='kg/m**3')
            sharedIndeps.add_output('mu_water', 0.0, units='kg/m/s')
            sharedIndeps.add_output('hsig_wave', 0.0, units='m')
            sharedIndeps.add_output('Tsig_wave', 0.0, units='s')
            sharedIndeps.add_output('beta_wave', 0.0, units='deg')
            sharedIndeps.add_output('wave_z0', 0.0, units='m')
            sharedIndeps.add_output('yaw', 0.0, units='deg')
            sharedIndeps.add_output('E', 0.0, units='N/m**2')
            sharedIndeps.add_output('G', 0.0, units='N/m**2')
            sharedIndeps.add_output('nu', 0.0)
            sharedIndeps.add_output('yield_stress', 0.0, units='N/m**2')
            sharedIndeps.add_output('DC', 0.0)
            sharedIndeps.add_output('life', 0.0)
            sharedIndeps.add_output('m_SN', 0.0)
            sharedIndeps.add_output('rna_mass', 0.0, units='kg')
            sharedIndeps.add_output('rna_I', np.zeros(6), units='kg*m**2')
            sharedIndeps.add_output('rna_cg', np.zeros(3), units='m')
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
        else:
            # If using YAML for input, unpack to native variables
            n_height_tow = self.options['analysis_options']['tower']['n_height']
            n_height_mon = 0 if not monopile else self.options['analysis_options']['monopile']['n_height']
            n_height     = n_height_tow if n_height_mon==0 else n_height_tow + n_height_mon - 1 # Should have one overlapping point
            n_layers_mon = 0 if not monopile else self.options['analysis_options']['monopile']['n_layers']
            self.add_subsystem('yaml', DiscretizationYAML(n_height_tower=n_height_tow, n_height_monopile=n_height_mon,
                                                          n_layers_tower=toweropt['n_layers'], n_layers_monopile=n_layers_mon,
                                                          n_mat=self.options['analysis_options']['materials']['n_mat']),
                               promotes=['*'])

        self.add_subsystem('vec', TempVec(n_height=self.options['analysis_options']['tower']['n_height']), promotes=['*'])
        
        self.add_subsystem('tow', TowerLeanSE(analysis_options=self.options['analysis_options'], topLevelFlag=False),
                           promotes=['tower_s','tower_height','tower_outer_diameter_in','tower_layer_thickness','tower_outfitting_factor',
                                     'max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                     'tower_mass','tower_I_base','hub_height',
                                     'labor_cost_rate','painting_cost_rate'])
        self.connect('unit_cost','tow.unit_cost_mat')
        self.connect('rho','tow.rho_mat')
        
        # Next do main and ballast columns
        # Ballast columns are replicated from same design in the components
        self.add_subsystem('main', Column(n_height=n_height_main, n_bulkhead=n_bulk_main, analysis_options=opt, topLevelFlag=False),
                           promotes=['E','nu','yield_stress','z0','rho_air','mu_air','rho_water','mu_water','rho',
                                     'Uref','zref','shearExp','yaw','Uc','hsig_wave','Tsig_wave','cd_usr','cm','loading',
                                     'max_draft','max_taper','min_d_to_t',
                                     'permanent_ballast_density','outfitting_factor','ballast_cost_rate',
                                     'unit_cost','labor_cost_rate','painting_cost_rate','outfitting_cost_rate'])

        self.add_subsystem('off', Column(n_height=n_height_off, n_bulkhead=n_bulk_off, analysis_options=opt, topLevelFlag=False),
                           promotes=['E','nu','yield_stress','z0','rho_air','mu_air','rho_water','mu_water','rho',
                                     'Uref','zref','shearExp','yaw','Uc','hsig_wave','Tsig_wave','cd_usr','cm','loading',
                                     'max_draft','max_taper','min_d_to_t',
                                     'permanent_ballast_density','outfitting_factor','ballast_cost_rate',
                                     'unit_cost','labor_cost_rate','painting_cost_rate','outfitting_cost_rate'])

        # Run Semi Geometry for interfaces
        self.add_subsystem('sg', SubstructureGeometry(n_height_main=n_height_main,
                                                      n_height_off=n_height_off), promotes=['*'])

        # Next run MapMooring
        self.add_subsystem('mm', MapMooring(analysis_options=opt), promotes=['*'])
        
        # Add in the connecting truss
        self.add_subsystem('load', Loading(n_height_main=n_height_main,
                                           n_height_off=n_height_off,
                                           n_height_tow=n_height_tow,
                                           analysis_options=opt), promotes=['*'])

        # Run main Semi analysis
        self.add_subsystem('subs', Substructure(n_height_main=n_height_main,
                                                n_height_off=n_height_off,
                                                n_height_tow=n_height_tow), promotes=['*'])
        
        # Connect all input variables from all models
        self.connect('main_freeboard', ['tow.foundation_height','main.freeboard'])
        self.connect('offset_freeboard', 'off.freeboard')

        self.connect('tow.d_full', ['windLoads.d','tower_d_full'])
        self.connect('tow.d_full', 'tower_d_base', src_indices=[0])
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
            self.connect('water_depth',['main.wave.z_floor','off.wave.z_floor'])
            self.connect('wave_z0',['main.wave.z_surface','off.wave.z_surface'])
            self.connect('wind_z0','z0')
            self.connect('wind_reference_height','zref')
            self.connect('wind_reference_speed','Uref')
            
        
        self.connect('main.L_stiffener','main_buckling_length')
        self.connect('off.L_stiffener','offset_buckling_length')
        
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
    prob['beta_wind'] = prob['beta_wave'] = 0.0    # Wind/water beta angle
    prob['cd_usr']      = -1.0 # Compute drag coefficient

    # Wind and water properties
    prob['rho_air'] = 1.226   # Density of air [kg/m^3]
    prob['mu_air']  = 1.78e-5 # Viscosity of air [kg/m/s]
    prob['rho_water']      = 1025.0  # Density of water [kg/m^3]
    prob['mu_water']  = 1.08e-3 # Viscosity of water [kg/m/s]
    
    # Material properties
    prob['rho'] = 7850.0          # Steel [kg/m^3]
    prob['E']                = 200e9           # Young's modulus [N/m^2]
    prob['G']                = 79.3e9          # Shear modulus [N/m^2]
    prob['yield_stress']     = 3.45e8          # Elastic yield stress [N/m^2]
    prob['nu']               = 0.26            # Poisson's ratio
    prob['permanent_ballast_density'] = 4492.0 # [kg/m^3]

    # Mass and cost scaling factors
    prob['outfitting_factor'] = 0.06    # Fraction of additional outfitting mass for each column
    prob['ballast_cost_rate']        = 0.1   # Cost factor for ballast mass [$/kg]
    prob['unit_cost']       = 1.1  # Cost factor for column mass [$/kg]
    prob['labor_cost_rate']          = 1.0  # Cost factor for labor time [$/min]
    prob['painting_cost_rate']       = 14.4  # Cost factor for column surface finishing [$/m^2]
    prob['outfitting_cost_rate']     = 1.5*1.1  # Cost factor for outfitting mass [$/kg]
    prob['mooring_cost_factor']      = 1.1     # Cost factor for mooring mass [$/kg]
    
    # Mooring parameters
    prob['number_of_mooring_connections'] = 3             # Evenly spaced around structure
    prob['mooring_lines_per_connection'] = 1             # Evenly spaced around structure
    prob['mooring_type']               = 'chain'       # Options are chain, nylon, polyester, fiber, or iwrc
    prob['anchor_type']                = 'DRAGEMBEDMENT' # Options are SUCTIONPILE or DRAGEMBEDMENT
    
    # Porperties of turbine tower
    nTower = prob.model.options['analysis_options']['tower']['n_height']-1
    prob['tower_height'] = prob['hub_height']              = 77.6                              # Length from tower main to top (not including freeboard) [m]
    prob['tower_s'] = np.linspace(0.0, 1.0, nTower+1)
    prob['tower_outer_diameter_in']    = np.linspace(6.5, 3.87, nTower+1) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_layer_thickness']    = np.linspace(0.027, 0.019, nTower).reshape((1,nTower)) # Diameter at each tower section node (linear lofting between) [m]
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
    npts = 5
    nsection = npts - 1
    
    opt = {}
    opt['platform'] = {}
    opt['platform']['columns'] = {}
    opt['platform']['columns']['main'] = {}
    opt['platform']['columns']['offset'] = {}
    opt['platform']['columns']['main']['n_height'] = npts
    opt['platform']['columns']['offset']['n_height'] = npts
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
    prob['main.section_height'] = np.array([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
    prob['main.outer_diameter'] = np.array([9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['main.wall_thickness'] = 0.05 * np.ones(nsection)               # Shell thickness at each section node (linear lofting between) [m]
    prob['main.bulkhead_thickness'] = 0.05*np.ones(4) # Locations/thickness of internal bulkheads at section interfaces [m]
    prob['main.bulkhead_locations'] = np.array([0.0, 0.25, 0.75, 1.0]) # Locations/thickness of internal bulkheads at section interfaces [m]
    
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
    npts = 5
    nsection = npts - 1
    
    opt = {}
    opt['platform'] = {}
    opt['platform']['columns'] = {}
    opt['platform']['columns']['main'] = {}
    opt['platform']['columns']['offset'] = {}
    opt['platform']['columns']['main']['n_height'] = npts
    opt['platform']['columns']['offset']['n_height'] = npts
    opt['platform']['tower'] = {}
    opt['platform']['tower']['buckling_length'] = 30.0
    opt['platform']['frame3dd']            = {}
    opt['platform']['frame3dd']['shear']   = True
    opt['platform']['frame3dd']['geom']    = False
    opt['platform']['frame3dd']['dx']      = -1
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
    prob['hsig_wave']        = 10.8   # Significant wave height [m]
    prob['Tsig_wave']           = 9.8    # Wave period [s]
    prob['wind_reference_speed']        = 11.0   # Wind reference speed [m/s]
    prob['wind_reference_height']        = 119.0  # Wind reference height [m]

    # Column geometry
    prob['main.permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
    prob['main_freeboard']                = 10.0 # Height extension above waterline [m]
    prob['main.section_height'] = np.array([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
    prob['main.outer_diameter'] = np.array([9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['main.wall_thickness'] = 0.05 * np.ones(nsection)               # Shell thickness at each section node (linear lofting between) [m]
    prob['main.bulkhead_thickness'] = 0.05*np.ones(4) # Locations/thickness of internal bulkheads at section interfaces [m]
    prob['main.bulkhead_locations'] = np.array([0.0, 0.25, 0.75, 1.0]) # Locations/thickness of internal bulkheads at section interfaces [m]

    # Auxiliary column geometry
    prob['radius_to_offset_column']         = 33.333 * np.cos(np.pi/6) # Centerline of main column to centerline of offset column [m]
    prob['off.permanent_ballast_height'] = 0.1                      # Height above keel for permanent ballast [m]
    prob['offset_freeboard']                = 12.0                     # Height extension above waterline [m]
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
        
