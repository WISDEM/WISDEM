from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_connect, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_VarComp
from openmdao.api import Group, Component, IndepVarComp, Problem, ScipyOptimizer, DumpRecorder
from offshorebos.wind_obos_component import WindOBOS
from rotorse.rotor import RotorSE
from towerse.tower import TowerSE
from commonse import NFREQ
from commonse.rna import RNA, RNAMass
from commonse.environment import PowerWind, LogWind
from commonse.turbine_constraints import TurbineConstraints
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015
from plant_financese.plant_finance import PlantFinance
from drivese.drivese_omdao import Drive3pt, Drive4pt

import numpy as np

wind = 'PowerWind'
nLC = 1
nDEL = 0
NSECTION = 6
    
class MonopileTower(Group):

    def __init__(self):
        super(MonopileTower, self).__init__()
        nFull   = 5*(NSECTION) + 1


        # Tower and substructure
        self.add('tow',TowerSE(nLC, NSECTION+1, nFull, nDEL, wind='PowerWind'), promotes=['material_density','E','G','tower_section_height',
                                                                                          'tower_outer_diameter','tower_wall_thickness',
                                                                                          'tower_outfitting_factor','tower_buckling_length',
                                                                                          'max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                                                                          'tower_mass','tower_I_base','hub_height',
                                                                                          'foundation_height','monopile','soil_G','soil_nu',
                                                                                          'suctionpile_depth','gamma_f','gamma_m','gamma_b','gamma_n','gamma_fatigue'
        ])
                 
        
        # Tower and Frame3DD options
        self.add('hub_height', IndepVarComp('hub_height', 0.0), promotes=['*'])
        self.add('foundation_height', IndepVarComp('foundation_height', 0.0), promotes=['*'])
        #self.add('yaw', IndepVarComp('yaw', 0.0), promotes=['*'])
        self.add('stress_standard_value',          IndepVarComp('stress_standard_value', 0.0), promotes=['*'])
        self.add('frame3dd_matrix_method',         IndepVarComp('frame3dd_matrix_method', 0, pass_by_obj=True), promotes=['*'])
        self.add('compute_stiffness',              IndepVarComp('compute_stiffness', False, pass_by_obj=True), promotes=['*'])
        self.add('project_lifetime',               IndepVarComp('project_lifetime', 0.0), promotes=['*'])
        self.add('lumped_mass_matrix',             IndepVarComp('lumped_mass_matrix', 0, pass_by_obj=True), promotes=['*'])
        self.add('slope_SN',                       IndepVarComp('slope_SN', 0, pass_by_obj=True), promotes=['*'])
        self.add('number_of_modes',                IndepVarComp('number_of_modes', NFREQ, pass_by_obj=True), promotes=['*'])
        self.add('compute_shear',                  IndepVarComp('compute_shear', True, pass_by_obj=True), promotes=['*'])
        self.add('shift_value',                    IndepVarComp('shift_value', 0.0), promotes=['*'])
        self.add('frame3dd_convergence_tolerance', IndepVarComp('frame3dd_convergence_tolerance', 1e-7), promotes=['*'])
        self.add('max_taper_ratio',                IndepVarComp('max_taper_ratio', 0.0), promotes=['*'])
        self.add('min_diameter_thickness_ratio',   IndepVarComp('min_diameter_thickness_ratio', 0.0), promotes=['*'])
        
        # Environment
        self.add('air_density',                IndepVarComp('air_density', 0.0), promotes=['*'])
        self.add('air_viscosity',              IndepVarComp('air_viscosity', 0.0), promotes=['*'])
        self.add('wind_reference_speed',       IndepVarComp('wind_reference_speed', 0.0), promotes=['*'])
        self.add('wind_reference_height',      IndepVarComp('wind_reference_height', 0.0), promotes=['*'])
        self.add('shearExp',                   IndepVarComp('shearExp', 0.0), promotes=['*'])
        self.add('wind_bottom_height',         IndepVarComp('wind_bottom_height', 0.0), promotes=['*'])
        self.add('wind_beta',                  IndepVarComp('wind_beta', 0.0), promotes=['*'])
        self.add('cd_usr',                     IndepVarComp('cd_usr', np.inf), promotes=['*'])

        # Environment
        self.add('water_depth',                IndepVarComp('water_depth', 0.0), promotes=['*'])
        self.add('water_density',              IndepVarComp('water_density', 0.0), promotes=['*'])
        self.add('water_viscosity',            IndepVarComp('water_viscosity', 0.0), promotes=['*'])
        self.add('wave_height',                IndepVarComp('wave_height', 0.0), promotes=['*'])
        self.add('wave_period',                IndepVarComp('wave_period', 0.0), promotes=['*'])
        self.add('mean_current_speed',         IndepVarComp('mean_current_speed', 0.0), promotes=['*'])
        self.add('wave_beta',                  IndepVarComp('wave_beta', 0.0), promotes=['*'])

        # Column
        self.add('morison_mass_coefficient',   IndepVarComp('morison_mass_coefficient', 0.0), promotes=['*'])
        self.add('material_density',           IndepVarComp('material_density', 0.0), promotes=['*'])
        self.add('E',                          IndepVarComp('E', 0.0), promotes=['*'])
        self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])
        self.add('yield_stress',               IndepVarComp('yield_stress', 0.0), promotes=['*'])
        
        # Design standards
        self.add('gamma_freq',      IndepVarComp('gamma_freq', 0.0), promotes=['*'])
        self.add('gamma_f',         IndepVarComp('gamma_f', 0.0), promotes=['*'])
        self.add('gamma_m',         IndepVarComp('gamma_m', 0.0), promotes=['*'])
        self.add('gamma_b',         IndepVarComp('gamma_b', 0.0), promotes=['*'])
        self.add('gamma_fatigue',   IndepVarComp('gamma_fatigue', 0.0), promotes=['*'])
        self.add('gamma_n',         IndepVarComp('gamma_n', 0.0), promotes=['*'])

        # Connect all input variables from all models
        self.connect('water_depth', 'tow.z_floor')
        
        self.connect('wind_beta', 'tow.windLoads.beta')
        self.connect('wave_beta', 'tow.waveLoads.beta')
        self.connect('cd_usr', 'tow.cd_usr')
        #self.connect('yaw', 'tow.distLoads.yaw')
        self.connect('mean_current_speed', 'tow.wave.Uc')
        
        self.connect('project_lifetime', 'tow.life')
        self.connect('number_of_modes', 'tow.nM')
        self.connect('frame3dd_convergence_tolerance', 'tow.tol')
        self.connect('lumped_mass_matrix', 'tow.lump')
        self.connect('stress_standard_value', 'tow.DC')
        self.connect('shift_value', 'tow.shift')
        self.connect('compute_shear', 'tow.shear')
        self.connect('slope_SN', 'tow.m_SN')
        self.connect('compute_stiffness', 'tow.geom')
        self.connect('frame3dd_matrix_method', 'tow.Mmethod')
        
        self.connect('air_density', 'tow.windLoads.rho')
        self.connect('air_viscosity', 'tow.windLoads.mu')
        self.connect('water_density',['tow.wave.rho','tow.waveLoads.rho'])
        self.connect('water_viscosity', 'tow.waveLoads.mu')
        self.connect('wave_height', 'tow.wave.hmax')
        self.connect('wave_period', 'tow.wave.T')
        self.connect('wind_reference_speed', 'tow.wind.Uref')
        self.connect('wind_reference_height', 'tow.wind.zref')
        self.connect('wind_bottom_height', 'tow.z0')
        self.connect('shearExp', 'tow.wind.shearExp')
        self.connect('morison_mass_coefficient', 'tow.cm')
        self.connect('yield_stress', 'tow.sigma_y')
        self.connect('max_taper_ratio', 'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
        
         # Use complex number finite differences
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1e-5
        self.deriv_options['step_calc'] = 'relative'



