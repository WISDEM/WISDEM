
from __future__ import print_function
import numpy as np
from pprint import pprint
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem, ScipyOptimizeDriver, SqliteRecorder, NonlinearRunOnce, DirectSolver
try:
    from openmdao.api import pyOptSparseDriver
except:
    pass
from wisdem.rotorse.rotor import RotorSE, Init_RotorSE_wRefBlade
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.towerse.tower import TowerSE
from wisdem.commonse import NFREQ
from wisdem.commonse.environment import PowerWind, LogWind
from wisdem.commonse.turbine_constraints import TurbineConstraints
from wisdem.turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.drivetrainse.rna  import RNA
from wisdem.orbit.api.wisdem.fixed import Orbit
from wisdem.commonse.mpi_tools import MPI

# np.seterr(all ='raise')
        
# Group to link the openmdao components
class MonopileTurbine(Group):

    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('FASTpref', default={})
        self.options.declare('Nsection_Tow', default = 6)
        self.options.declare('VerbosityCosts', default = True)
        self.options.declare('user_update_routine',     default=None)
        
        
    def setup(self):
        
        RefBlade            = self.options['RefBlade']
        Nsection_Tow        = self.options['Nsection_Tow']        
        user_update_routine = self.options['user_update_routine']
        if 'Analysis_Level' in self.options['FASTpref']:
            Analysis_Level  = self.options['FASTpref']['Analysis_Level']
        else:
            Analysis_Level  = 0
        
        # Define all input variables from all models
        myIndeps = IndepVarComp()
        myIndeps.add_discrete_output('crane',    False)

        # Turbine Costs
        myIndeps.add_discrete_output('bearing_number', 0)
        
        # Tower and Frame3DD options
        myIndeps.add_output('project_lifetime',             0.0, units='yr')
        myIndeps.add_output('max_taper_ratio',              0.0)
        myIndeps.add_output('min_diameter_thickness_ratio', 0.0)

        # Environment
        myIndeps.add_output('wind_bottom_height',   0.0, units='m')
        myIndeps.add_output('wind_beta',            0.0, units='deg')
        myIndeps.add_output('cd_usr', -1.)

        # Environment Offshore
        myIndeps.add_output('offshore',           True)
        myIndeps.add_output('water_depth',        0.0, units='m')
        myIndeps.add_output('wave_height',        0.0, units='m')
        myIndeps.add_output('wave_period',        0.0, units='s')
        myIndeps.add_output('mean_current_speed', 0.0, units='m/s')

        # Design standards
        myIndeps.add_output('gamma_b', 0.0)
        myIndeps.add_output('gamma_n', 0.0)

        # Column
        myIndeps.add_output('morison_mass_coefficient', 2.0)
        myIndeps.add_output('material_density',         0.0, units='kg/m**3')
        myIndeps.add_output('E',                        0.0, units='N/m**2')
        myIndeps.add_output('yield_stress',             0.0, units='N/m**2')

        # Pontoons
        myIndeps.add_output('G', 0.0, units='N/m**2')
        
        # LCOE
        myIndeps.add_output('labor_cost_rate',      0.0, units='USD/min')
        myIndeps.add_output('material_cost_rate',   0.0, units='USD/kg')
        myIndeps.add_output('painting_cost_rate',   0.0, units='USD/m**2')
        myIndeps.add_discrete_output('number_of_turbines', 0)
        myIndeps.add_output('annual_opex',          0.0, units='USD/kW/yr') # TODO: Replace with output connection
        myIndeps.add_output('fixed_charge_rate',    0.0)
        myIndeps.add_output('wake_loss_factor',     0.0)
        
        myIndeps.add_output('overhang', 0.0, units='m')
        myIndeps.add_output('hub_cm', np.zeros(3), units='m')
        myIndeps.add_output('nac_cm', np.zeros(3), units='m')
        myIndeps.add_output('hub_I', np.zeros(6), units='kg*m**2')
        myIndeps.add_output('nac_I', np.zeros(6), units='kg*m**2')
        myIndeps.add_output('hub_mass', 0.0, units='kg')
        myIndeps.add_output('nac_mass', 0.0, units='kg')
        myIndeps.add_output('hss_mass', 0.0, units='kg')
        myIndeps.add_output('lss_mass', 0.0, units='kg')
        myIndeps.add_output('cover_mass', 0.0, units='kg')
        myIndeps.add_output('pitch_system_mass', 0.0, units='kg')
        myIndeps.add_output('platforms_mass', 0.0, units='kg')
        myIndeps.add_output('spinner_mass', 0.0, units='kg')
        myIndeps.add_output('transformer_mass', 0.0, units='kg')
        myIndeps.add_output('vs_electronics_mass', 0.0, units='kg')
        myIndeps.add_output('yaw_mass', 0.0, units='kg')
        myIndeps.add_output('gearbox_mass', 0.0, units='kg')
        myIndeps.add_output('generator_mass', 0.0, units='kg')
        myIndeps.add_output('bedplate_mass', 0.0, units='kg')
        myIndeps.add_output('main_bearing_mass', 0.0, units='kg')

        self.add_subsystem('myIndeps', myIndeps, promotes=['*'])

        
        # Add components
        self.add_subsystem('rotorse', RotorSE(RefBlade=RefBlade,
                                              npts_coarse_power_curve=20,
                                              npts_spline_power_curve=200,
                                              regulation_reg_II5=True,
                                              regulation_reg_III=True,
                                              Analysis_Level=Analysis_Level,
                                              FASTpref=self.options['FASTpref'],
                                              topLevelFlag=True,
                                              user_update_routine=user_update_routine), promotes=['*'])
        
        self.add_subsystem('rna', RNA(nLC=1), promotes=['hub_mass','nac_mass','nac_cm','hub_cm','tilt'])
        
        # Tower and substructure
        self.add_subsystem('tow',TowerSE(nLC=1,
                                         nPoints=Nsection_Tow+1,
                                         nFull=5*Nsection_Tow+1,
                                         wind='PowerWind',
                                         topLevelFlag=False,
                                         monopile=True),
                           promotes=['water_density','water_viscosity','wave_beta',
                                     'significant_wave_height','significant_wave_period',
                                     'material_density','E','G','tower_section_height',
                                     'tower_wall_thickness', 'tower_outer_diameter',
                                     'tower_outfitting_factor','tower_buckling_length',
                                     'transition_piece_mass','transition_piece_height',
                                     'max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                     'tower_add_gravity','tower_mass','tower_I_base','hub_height',
                                     'foundation_height','soil_G','soil_nu',
                                     'monopile_mass','monopile_cost','monopile_length',
                                     'suctionpile_depth','gamma_f','gamma_m','gamma_b','gamma_n','gamma_fatigue',
                                     'labor_cost_rate','material_cost_rate','painting_cost_rate','z_full','d_full','t_full',
                                     'DC','shear','geom','tower_force_discretization','nM','Mmethod','lump','tol','shift'])

        # Turbine constraints
        self.add_subsystem('tcons', TurbineConstraints(nFull=5*(Nsection_Tow+1)+1), promotes=['*'])
        
        # Turbine costs
        self.add_subsystem('tcost', Turbine_CostsSE_2015(verbosity=self.options['VerbosityCosts'], topLevelFlag=False), promotes=['*'])

        # Use ORBIT for BOS costs
        self.add_subsystem('orbit', Orbit(), promotes=['wtiv', 'feeder', 'num_feeders', 'oss_install_vessel',
                                                       'hub_height',
                                                       'number_of_turbines',
                                                       'tower_mass',
                                                       'monopile_length',
                                                       'monopile_mass',
                                                       'transition_piece_mass',
                                                       'site_distance',
                                                       'site_distance_to_landfall',
                                                       'site_distance_to_interconnection',
                                                       'plant_turbine_spacing',
                                                       'plant_row_spacing',
                                                       'plant_substation_distance',
                                                       'tower_deck_space',
                                                       'nacelle_deck_space',
                                                       'blade_deck_space',
                                                       'port_cost_per_month',
                                                       'monopile_deck_space',
                                                       'transition_piece_deck_space',
                                                       'commissioning_pct',
                                                       'decommissioning_pct'])
        
        # LCOE Calculation
        self.add_subsystem('plantfinancese', PlantFinance(verbosity=self.options['VerbosityCosts']), promotes=['machine_rating','lcoe'])
        
    
        # Set up connections        

        # Connections to DriveSE
        self.connect('Fxyz_total',      'rna.loads.F')
        self.connect('Mxyz_total',      'rna.loads.M')
        self.connect('mass_all_blades',  'rna.blades_mass')
        self.connect('I_all_blades',        'rna.blades_I')

        self.connect('material_density', 'tow.tower.rho')

        # Connections to TowerSE
        self.connect('rna.loads.top_F',         'tow.pre.rna_F')
        self.connect('rna.loads.top_M',         'tow.pre.rna_M')
        self.connect('rna.rna_I_TT',            ['rna_I','tow.pre.mI'])
        self.connect('rna.rna_cm',              ['rna_cg','tow.pre.mrho'])
        self.connect('rna.rna_mass',            ['rna_mass','tow.pre.mass'])
        self.connect('rs.gust.V_gust',          'tow.wind.Uref')
        self.connect('wind_reference_height',   ['tow.wind.zref','wind.zref'])
        # self.connect('wind_bottom_height',      ['tow.wind.z0','tow.wave.z_surface', 'wind.z0'])  # offshore
        self.connect('wind_bottom_height',      ['tow.wind.z0', 'wind.z0'])
        self.connect('shearExp',                ['tow.wind.shearExp'])
        # self.connect('morison_mass_coefficient','tow.cm')                                         # offshore
        self.connect('yield_stress',            'tow.sigma_y')
        self.connect('max_taper_ratio',         'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')

        self.connect('rho',         'tow.windLoads.rho')
        self.connect('mu',          'tow.windLoads.mu')
        self.connect('wind_beta',   'tow.windLoads.beta')
        
        # Connections to TurbineConstraints
        self.connect('nBlades',                 'blade_number')
        self.connect('control_maxOmega',        'rotor_omega')
        self.connect('tow.post.structural_frequencies', 'tower_freq')        
                
        # Connections to TurbineCostSE
        self.connect('mass_one_blade',              ['blade_mass','orbit.blade_mass'])
        self.connect('total_blade_cost',            'blade_cost_external')

        # Connections to ORBIT
        self.connect('water_depth', 'orbit.site_depth')
        self.connect('rs.gust.V_gust', 'orbit.site_mean_windspeed')
        self.connect('machine_rating', 'orbit.turbine_rating')
        self.connect('rated_V', 'orbit.turbine_rated_windspeed')
        self.connect('turbine_cost_kW', 'orbit.turbine_capex')
        self.connect('diameter', 'orbit.turbine_rotor_diameter')
        self.connect('nac_mass', 'orbit.nacelle_mass')
        self.connect('tower_outer_diameter', 'orbit.monopile_diameter', src_indices=[0])
        
        # Connections to PlantFinanceSE
        self.connect('AEP',                 'plantfinancese.turbine_aep')
        self.connect('turbine_cost_kW',     'plantfinancese.tcc_per_kW')
        self.connect('number_of_turbines',  'plantfinancese.turbine_number')
        self.connect('orbit.total_capex_kW','plantfinancese.bos_per_kW')
        self.connect('annual_opex',         'plantfinancese.opex_per_kW')
    

    





    
