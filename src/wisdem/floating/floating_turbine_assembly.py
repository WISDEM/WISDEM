from openmdao.api import Group, Component, IndepVarComp, DirectSolver, ScipyGMRES, Newton, NLGaussSeidel, Brent, RunOnce
from floatingse.floating import FloatingSE
from offshorebos.wind_obos_component import WindOBOS
from plant_financese.plant_finance import PlantFinance
from rotorse.rotor import RotorSE
from commonse.rna import RNA
from commonse.turbine_constraints import TurbineConstraints
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015

import numpy as np

        
class FloatingTurbine(Group):

    def __init__(self, RefBlade, nSection):
        super(FloatingTurbine, self).__init__()

        self.add('hub_height', IndepVarComp('hub_height', 0.0), promotes=['*'])

        # TODO: 
        #Weibull/Rayleigh CDF
        
        # Rotor
        self.add('rotor', RotorSE(RefBlade), promotes=['hubFraction','nBlades','turbine_class','sparT_in','teT_in','bladeLength','precone','tilt','yaw',
                                                       'r_max_chord','chord_in','theta_in','precurve_in','presweep_in','precurve_tip','presweep_tip',
                                                       'turbulence_class','gust_stddev','VfactorPC','shape_parameter','Rtip','precurveTip','presweepTip',
                                                       'control_Vin','control_Vout','machine_rating','control_minOmega','control_maxOmega',
                                                       'control_tsr','control_pitch','pitch_extreme','azimuth_extreme','drivetrainType',
                                                       'rstar_damage','Mxb_damage','Myb_damage','strain_ult_spar','strain_ult_te','m_damage',
                                                       'nSector','tiploss','hubloss','wakerotation','usecd','AEP_loss_factor','tip_deflection',
                                                       'dynamic_amplication_tip_deflection'])

        # RNA
        self.add('rna', RNA(1), promotes=['downwind'])
        
        # Tower and substructure
        myfloat = FloatingSE(nSection)
        self.add('sm', myfloat, promotes=['radius_to_auxiliary_column','fairlead','fairlead_offset_from_shell',
                                          'fairlead_support_wall_thickness','fairlead_support_outer_diameter',
                                          'base_freeboard','base_section_height','base_outer_diameter','base_wall_thickness',
                                          'auxiliary_freeboard','auxiliary_section_height','auxiliary_outer_diameter','auxiliary_wall_thickness',
                                          'mooring_line_length','anchor_radius','mooring_diameter','number_of_mooring_connections','mooring_lines_per_connection','mooring_type',
                                          'anchor_type','mooring_max_offset','mooring_operational_heel','max_survival_heel','mooring_cost_rate',
                                          'permanent_ballast_density','base_stiffener_web_height','base_stiffener_web_thickness',
                                          'base_stiffener_flange_width','base_stiffener_flange_thickness','base_stiffener_spacing',
                                          'base_bulkhead_thickness','base_permanent_ballast_height','base_heave_plate_diameter',
                                          'auxiliary_stiffener_web_height','auxiliary_stiffener_web_thickness','auxiliary_stiffener_flange_width',
                                          'auxiliary_stiffener_flange_thickness','auxiliary_stiffener_spacing','auxiliary_bulkhead_thickness',
                                          'auxiliary_permanent_ballast_height','auxiliary_heave_plate_diameter',
                                          'bulkhead_mass_factor','ring_mass_factor','shell_mass_factor',
                                          'column_mass_factor','outfitting_mass_fraction','ballast_cost_rate','tapered_col_cost_rate',
                                          'outfitting_cost_rate','number_of_auxiliary_columns','pontoon_outer_diameter',
                                          'pontoon_wall_thickness','cross_attachment_pontoons_int','lower_attachment_pontoons_int',
                                          'upper_attachment_pontoons_int','lower_ring_pontoons_int','upper_ring_pontoons_int','outer_cross_pontoons_int',
                                          'pontoon_cost_rate','connection_ratio_max','base_pontoon_attach_upper','base_pontoon_attach_lower',
                                          'tower_section_height','tower_outer_diameter','tower_wall_thickness','tower_outfitting_factor',
                                          'tower_buckling_length','tower_mass','loading','min_diameter_thickness_ratio','max_taper_ratio',
                                          'wave_period_range_low','wave_period_range_high'])

        # Turbine constraints
        self.add('tcons', TurbineConstraints(myfloat.nFull), promotes=['blade_number','Rtip','precurveTip','presweepTip','precone','tilt',
                                                                       'tip_deflection','downwind'])
        
        # Turbine costs
        self.add('tcost', Turbine_CostsSE_2015(), promotes=['*'])
        
        # Balance of station
        self.add('wobos', WindOBOS(), promotes=['tax_rate'])

        # LCOE Calculation
        self.add('lcoe', PlantFinance(), promotes=['*'])

        # Define all input variables from all models
        self.add('offshore',             IndepVarComp('offshore', True, pass_by_obj=True), promotes=['*'])
        self.add('crane',                IndepVarComp('crane', False, pass_by_obj=True), promotes=['*'])

        # Turbine Costs
        # REMOVE ONCE DRIVESE AND GENERATORSE ARE CONNECTED
        self.add('bearing_number',           IndepVarComp('bearing_number', 0, pass_by_obj=True), promotes=['*'])
        self.add('bedplate_mass',            IndepVarComp('bedplate_mass', 0.0), promotes=['*'])
        self.add('crane_cost',               IndepVarComp('crane_cost', 0.0), promotes=['*'])
        self.add('gearbox_mass',             IndepVarComp('gearbox_mass', 0.0), promotes=['*'])
        self.add('generator_mass',           IndepVarComp('generator_mass', 0.0), promotes=['*'])
        self.add('high_speed_side_mass',     IndepVarComp('high_speed_side_mass', 0.0), promotes=['*'])
        self.add('hydraulic_cooling_mass',   IndepVarComp('hydraulic_cooling_mass', 0.0), promotes=['*'])
        self.add('low_speed_shaft_mass',     IndepVarComp('low_speed_shaft_mass', 0.0), promotes=['*'])
        self.add('main_bearing_mass',        IndepVarComp('main_bearing_mass', 0.0), promotes=['*'])
        self.add('nacelle_cover_mass',       IndepVarComp('nacelle_cover_mass', 0.0), promotes=['*'])
        self.add('nacelle_platforms_mass',   IndepVarComp('nacelle_platforms_mass', 0.0), promotes=['*'])
        self.add('pitch_system_mass',        IndepVarComp('pitch_system_mass', 0.0), promotes=['*'])
        self.add('spinner_mass',             IndepVarComp('spinner_mass', 0.0), promotes=['*'])
        self.add('transformer_mass',         IndepVarComp('transformer_mass', 0.0), promotes=['*'])
        self.add('variable_speed_elec_mass', IndepVarComp('variable_speed_elec_mass', 0.0), promotes=['*'])
        self.add('yaw_system_mass',          IndepVarComp('yaw_system_mass', 0.0), promotes=['*'])
        
        # Tower
        #self.add('stress_standard_value',          IndepVarComp('stress_standard_value', 0.0), promotes=['*'])
        #self.add('fatigue_parameters',             IndepVarComp('fatigue_parameters', np.zeros(nDEL)), promotes=['*'])
        #self.add('fatigue_z',                      IndepVarComp('fatigue_z', np.zeros(nDEL)), promotes=['*'])
        #self.add('frame3dd_matrix_method',         IndepVarComp('frame3dd_matrix_method', 0, pass_by_obj=True), promotes=['*'])
        #self.add('compute_stiffnes',               IndepVarComp('compute_stiffnes', False, pass_by_obj=True), promotes=['*'])
        self.add('project_lifetime',               IndepVarComp('project_lifetime', 0.0), promotes=['*'])
        #self.add('lumped_mass_matrix',             IndepVarComp('lumped_mass_matrix', 0, pass_by_obj=True), promotes=['*'])
        #self.add('slope_SN',                       IndepVarComp('slope_SN', 0, pass_by_obj=True), promotes=['*'])
        #self.add('number_of_modes',                IndepVarComp('number_of_modes', 0, pass_by_obj=True), promotes=['*'])
        #self.add('compute_shear',                  IndepVarComp('compute_shear', True, pass_by_obj=True), promotes=['*'])
        #self.add('shift_value',                    IndepVarComp('shift_value', 0.0), promotes=['*'])
        #self.add('frame3dd_convergence_tolerance', IndepVarComp('frame3dd_convergence_tolerance', 1e-9), promotes=['*'])

        # TODO: Multiple load cases
        # Environment
        self.add('air_density',                IndepVarComp('air_density', 0.0), promotes=['*'])
        self.add('air_viscosity',              IndepVarComp('air_viscosity', 0.0), promotes=['*'])
        self.add('wind_reference_speed',       IndepVarComp('wind_reference_speed', 0.0), promotes=['*'])
        self.add('wind_reference_height',      IndepVarComp('wind_reference_height', 0.0), promotes=['*'])
        self.add('shearExp',                      IndepVarComp('shearExp', 0.0), promotes=['*'])
        self.add('wind_bottom_height',         IndepVarComp('wind_bottom_height', 0.0), promotes=['*'])
        self.add('wind_beta',                  IndepVarComp('wind_beta', 0.0), promotes=['*'])
        self.add('cd_usr',                     IndepVarComp('cd_usr', np.inf), promotes=['*'])

        # Environment
        self.add('water_depth',                IndepVarComp('water_depth', 0.0), promotes=['*'])
        self.add('water_density',              IndepVarComp('water_density', 0.0), promotes=['*'])
        self.add('water_viscosity',            IndepVarComp('water_viscosity', 0.0), promotes=['*'])
        self.add('wave_height',                IndepVarComp('wave_height', 0.0), promotes=['*'])
        self.add('wave_period',                IndepVarComp('wave_period', 0.0), promotes=['*'])
        self.add('mean_current_speed',          IndepVarComp('mean_current_speed', 0.0), promotes=['*'])
        #self.add('wave_beta',                  IndepVarComp('wave_beta', 0.0), promotes=['*'])
        #self.add('wave_velocity_z0',           IndepVarComp('wave_velocity_z0', 0.0), promotes=['*'])
        #self.add('wave_acceleration_z0',       IndepVarComp('wave_acceleration_z0', 0.0), promotes=['*'])
        
        # Design standards
        self.add('safety_factor_frequency',    IndepVarComp('safety_factor_frequency', 0.0), promotes=['*'])
        self.add('safety_factor_stress',       IndepVarComp('safety_factor_stress', 0.0), promotes=['*'])
        self.add('safety_factor_materials',    IndepVarComp('safety_factor_materials', 0.0), promotes=['*'])
        self.add('safety_factor_buckling',     IndepVarComp('safety_factor_buckling', 0.0), promotes=['*'])
        self.add('safety_factor_fatigue',      IndepVarComp('safety_factor_fatigue', 0.0), promotes=['*'])
        self.add('safety_factor_consequence',  IndepVarComp('safety_factor_consequence', 0.0), promotes=['*'])

        # RNA
        self.add('dummy_mass',                 IndepVarComp('dummy_mass', 0.0), promotes=['*'])
        self.add('hub_mass',                   IndepVarComp('hub_mass', 0.0), promotes=['*'])
        self.add('nac_mass',                   IndepVarComp('nac_mass', 0.0), promotes=['*'])
        self.add('hub_cm',                     IndepVarComp('hub_cm', np.zeros((3,))), promotes=['*'])
        self.add('nac_cm',                     IndepVarComp('nac_cm', np.zeros((3,))), promotes=['*'])
        self.add('hub_I',                      IndepVarComp('hub_I', np.zeros(6)), promotes=['*'])
        self.add('nac_I',                      IndepVarComp('nac_I', np.zeros(6)), promotes=['*'])
        self.add('downwind',                   IndepVarComp('downwind', False, pass_by_obj=True), promotes=['*'])
        self.add('rna_weightM',                IndepVarComp('rna_weightM', True, pass_by_obj=True), promotes=['*'])
        
        # Column
        self.add('morison_mass_coefficient',   IndepVarComp('morison_mass_coefficient', 0.0), promotes=['*'])
        self.add('material_density',           IndepVarComp('material_density', 0.0), promotes=['*'])
        self.add('E',                          IndepVarComp('E', 0.0), promotes=['*'])
        self.add('nu',                         IndepVarComp('nu', 0.0), promotes=['*'])
        self.add('yield_stress',               IndepVarComp('yield_stress', 0.0), promotes=['*'])

        # Pontoons
        self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])

        # Offshore BOS
        # Turbine / Plant parameters, ,
        self.add('nacelleL',                     IndepVarComp('nacelleL', 0.0), promotes=['*'])
        self.add('nacelleW',                     IndepVarComp('nacelleW', 0.0), promotes=['*'])
        self.add('distShore',                    IndepVarComp('distShore', 0.0), promotes=['*']) #90.0
        self.add('distPort',                     IndepVarComp('distPort', 0.0), promotes=['*']) #90.0
        self.add('distPtoA',                     IndepVarComp('distPtoA', 0.0), promotes=['*']) #90.0
        self.add('distAtoS',                     IndepVarComp('distAtoS', 0.0), promotes=['*']) #90.0
        self.add('substructure',                 IndepVarComp('substructure', 'SEMISUBMERSIBLE', pass_by_obj=True), promotes=['*'])
        self.add('anchor',                       IndepVarComp('anchor', 'DRAGEMBEDMENT', pass_by_obj=True), promotes=['*'])
        self.add('turbInstallMethod',            IndepVarComp('turbInstallMethod', 'INDIVIDUAL', pass_by_obj=True), promotes=['*'])
        self.add('towerInstallMethod',           IndepVarComp('towerInstallMethod', 'ONEPIECE', pass_by_obj=True), promotes=['*'])
        self.add('installStrategy',              IndepVarComp('installStrategy', 'PRIMARYVESSEL', pass_by_obj=True), promotes=['*'])
        self.add('cableOptimizer',               IndepVarComp('cableOptimizer', False, pass_by_obj=True), promotes=['*'])
        self.add('buryDepth',                    IndepVarComp('buryDepth', 0.0), promotes=['*']) #2.0
        self.add('arrayY',                       IndepVarComp('arrayY', 0.0), promotes=['*']) #9.0
        self.add('arrayX',                       IndepVarComp('arrayX', 0.0), promotes=['*']) #9.0
        self.add('substructCont',                IndepVarComp('substructCont', 0.0), promotes=['*']) #0.30
        self.add('turbCont',                     IndepVarComp('turbCont', 0.0), promotes=['*']) #0.30
        self.add('elecCont',                     IndepVarComp('elecCont', 0.0), promotes=['*']) #0.30
        self.add('interConVolt',                 IndepVarComp('interConVolt', 0.0), promotes=['*']) #345.0
        self.add('distInterCon',                 IndepVarComp('distInterCon', 0.0), promotes=['*']) #3.0
        self.add('scrapVal',                     IndepVarComp('scrapVal', 0.0), promotes=['*']) #0.0
        #General', , 
        self.add('inspectClear',                 IndepVarComp('inspectClear', 0.0), promotes=['*']) #2.0
        self.add('plantComm',                    IndepVarComp('plantComm', 0.0), promotes=['*']) #0.01
        self.add('procurement_contingency',      IndepVarComp('procurement_contingency', 0.0), promotes=['*']) #0.05
        self.add('install_contingency',          IndepVarComp('install_contingency', 0.0), promotes=['*']) #0.30
        self.add('construction_insurance',       IndepVarComp('construction_insurance', 0.0), promotes=['*']) #0.01
        self.add('capital_cost_year_0',          IndepVarComp('capital_cost_year_0', 0.0), promotes=['*']) #0.20
        self.add('capital_cost_year_1',          IndepVarComp('capital_cost_year_1', 0.0), promotes=['*']) #0.60
        self.add('capital_cost_year_2',          IndepVarComp('capital_cost_year_2', 0.0), promotes=['*']) #0.10
        self.add('capital_cost_year_3',          IndepVarComp('capital_cost_year_3', 0.0), promotes=['*']) #0.10
        self.add('capital_cost_year_4',          IndepVarComp('capital_cost_year_4', 0.0), promotes=['*']) #0.0
        self.add('capital_cost_year_5',          IndepVarComp('capital_cost_year_5', 0.0), promotes=['*']) #0.0
        self.add('tax_rate',                     IndepVarComp('tax_rate', 0.0), promotes=['*']) #0.40
        self.add('interest_during_construction', IndepVarComp('interest_during_construction', 0.0), promotes=['*']) #0.08
        #Substructure & Foundation', , 
        self.add('mpileCR',                      IndepVarComp('mpileCR', 0.0), promotes=['*']) #2250.0
        self.add('mtransCR',                     IndepVarComp('mtransCR', 0.0), promotes=['*']) #3230.0
        self.add('mpileD',                       IndepVarComp('mpileD', 0.0), promotes=['*']) #0.0
        self.add('mpileL',                       IndepVarComp('mpileL', 0.0), promotes=['*']) #0.0
        self.add('mpEmbedL',                     IndepVarComp('mpEmbedL', 0.0), promotes=['*']) #30.0
        self.add('jlatticeCR',                   IndepVarComp('jlatticeCR', 0.0), promotes=['*']) #4680.0
        self.add('jtransCR',                     IndepVarComp('jtransCR', 0.0), promotes=['*']) #4500.0
        self.add('jpileCR',                      IndepVarComp('jpileCR', 0.0), promotes=['*']) #2250.0
        self.add('jlatticeA',                    IndepVarComp('jlatticeA', 0.0), promotes=['*']) #26.0
        self.add('jpileL',                       IndepVarComp('jpileL', 0.0), promotes=['*']) #47.50
        self.add('jpileD',                       IndepVarComp('jpileD', 0.0), promotes=['*']) #1.60
        self.add('ssHeaveCR',                    IndepVarComp('ssHeaveCR', 0.0), promotes=['*']) #6250.0
        self.add('scourMat',                     IndepVarComp('scourMat', 0.0), promotes=['*']) #250000.0
        self.add('number_install_seasons',       IndepVarComp('number_install_seasons', 0, pass_by_obj=True), promotes=['*']) #1
        #Electrical Infrastructure', , 
        self.add('pwrFac',                       IndepVarComp('pwrFac', 0.0), promotes=['*']) #0.95
        self.add('buryFac',                      IndepVarComp('buryFac', 0.0), promotes=['*']) #0.10
        self.add('catLengFac',                   IndepVarComp('catLengFac', 0.0), promotes=['*']) #0.04
        self.add('exCabFac',                     IndepVarComp('exCabFac', 0.0), promotes=['*']) #0.10
        self.add('subsTopFab',                   IndepVarComp('subsTopFab', 0.0), promotes=['*']) #14500.0
        self.add('subsTopDes',                   IndepVarComp('subsTopDes', 0.0), promotes=['*']) #4500000.0
        self.add('topAssemblyFac',               IndepVarComp('topAssemblyFac', 0.0), promotes=['*']) #0.075
        self.add('subsJackCR',                   IndepVarComp('subsJackCR', 0.0), promotes=['*']) #6250.0
        self.add('subsPileCR',                   IndepVarComp('subsPileCR', 0.0), promotes=['*']) #2250.0
        self.add('dynCabFac',                    IndepVarComp('dynCabFac', 0.0), promotes=['*']) #2.0
        self.add('shuntCR',                      IndepVarComp('shuntCR', 0.0), promotes=['*']) #35000.0
        self.add('highVoltSG',                   IndepVarComp('highVoltSG', 0.0), promotes=['*']) #950000.0
        self.add('medVoltSG',                    IndepVarComp('medVoltSG', 0.0), promotes=['*']) #500000.0
        self.add('backUpGen',                    IndepVarComp('backUpGen', 0.0), promotes=['*']) #1000000.0
        self.add('workSpace',                    IndepVarComp('workSpace', 0.0), promotes=['*']) #2000000.0
        self.add('otherAncillary',               IndepVarComp('otherAncillary', 0.0), promotes=['*']) #3000000.0
        self.add('mptCR',                        IndepVarComp('mptCR', 0.0), promotes=['*']) #12500.0
        self.add('arrVoltage',                   IndepVarComp('arrVoltage', 0.0), promotes=['*']) #33.0
        self.add('cab1CR',                       IndepVarComp('cab1CR', 0.0), promotes=['*']) #185.889
        self.add('cab2CR',                       IndepVarComp('cab2CR', 0.0), promotes=['*']) #202.788
        self.add('cab1CurrRating',               IndepVarComp('cab1CurrRating', 0.0), promotes=['*']) #300.0
        self.add('cab2CurrRating',               IndepVarComp('cab2CurrRating', 0.0), promotes=['*']) #340.0
        self.add('arrCab1Mass',                  IndepVarComp('arrCab1Mass', 0.0), promotes=['*']) #20.384
        self.add('arrCab2Mass',                  IndepVarComp('arrCab2Mass', 0.0), promotes=['*']) #21.854
        self.add('cab1TurbInterCR',              IndepVarComp('cab1TurbInterCR', 0.0), promotes=['*']) #8410.0
        self.add('cab2TurbInterCR',              IndepVarComp('cab2TurbInterCR', 0.0), promotes=['*']) #8615.0
        self.add('cab2SubsInterCR',              IndepVarComp('cab2SubsInterCR', 0.0), promotes=['*']) #19815.0
        self.add('expVoltage',                   IndepVarComp('expVoltage', 0.0), promotes=['*']) #220.0
        self.add('expCurrRating',                IndepVarComp('expCurrRating', 0.0), promotes=['*']) #530.0
        self.add('expCabMass',                   IndepVarComp('expCabMass', 0.0), promotes=['*']) #71.90
        self.add('expCabCR',                     IndepVarComp('expCabCR', 0.0), promotes=['*']) #495.411
        self.add('expSubsInterCR',               IndepVarComp('expSubsInterCR', 0.0), promotes=['*']) #57500.0
        # Vector inputs
        #self.add('arrayCables',                  IndepVarComp('arrayCables', [33, 66], pass_by_obj=True), promotes=['*'])
        #self.add('exportCables',                 IndepVarComp('exportCables', [132, 220], pass_by_obj=True), promotes=['*'])
        #Assembly & Installation',
        self.add('moorTimeFac',                  IndepVarComp('moorTimeFac', 0.0), promotes=['*']) #0.005
        self.add('moorLoadout',                  IndepVarComp('moorLoadout', 0.0), promotes=['*']) #5.0
        self.add('moorSurvey',                   IndepVarComp('moorSurvey', 0.0), promotes=['*']) #4.0
        self.add('prepAA',                       IndepVarComp('prepAA', 0.0), promotes=['*']) #168.0
        self.add('prepSpar',                     IndepVarComp('prepSpar', 0.0), promotes=['*']) #18.0
        self.add('upendSpar',                    IndepVarComp('upendSpar', 0.0), promotes=['*']) #36.0
        self.add('prepSemi',                     IndepVarComp('prepSemi', 0.0), promotes=['*']) #12.0
        self.add('turbFasten',                   IndepVarComp('turbFasten', 0.0), promotes=['*']) #8.0
        self.add('boltTower',                    IndepVarComp('boltTower', 0.0), promotes=['*']) #7.0
        self.add('boltNacelle1',                 IndepVarComp('boltNacelle1', 0.0), promotes=['*']) #7.0
        self.add('boltNacelle2',                 IndepVarComp('boltNacelle2', 0.0), promotes=['*']) #7.0
        self.add('boltNacelle3',                 IndepVarComp('boltNacelle3', 0.0), promotes=['*']) #7.0
        self.add('boltBlade1',                   IndepVarComp('boltBlade1', 0.0), promotes=['*']) #3.50
        self.add('boltBlade2',                   IndepVarComp('boltBlade2', 0.0), promotes=['*']) #3.50
        self.add('boltRotor',                    IndepVarComp('boltRotor', 0.0), promotes=['*']) #7.0
        self.add('vesselPosTurb',                IndepVarComp('vesselPosTurb', 0.0), promotes=['*']) #2.0
        self.add('vesselPosJack',                IndepVarComp('vesselPosJack', 0.0), promotes=['*']) #8.0
        self.add('vesselPosMono',                IndepVarComp('vesselPosMono', 0.0), promotes=['*']) #3.0
        self.add('subsVessPos',                  IndepVarComp('subsVessPos', 0.0), promotes=['*']) #6.0
        self.add('monoFasten',                   IndepVarComp('monoFasten', 0.0), promotes=['*']) #12.0
        self.add('jackFasten',                   IndepVarComp('jackFasten', 0.0), promotes=['*']) #20.0
        self.add('prepGripperMono',              IndepVarComp('prepGripperMono', 0.0), promotes=['*']) #1.50
        self.add('prepGripperJack',              IndepVarComp('prepGripperJack', 0.0), promotes=['*']) #8.0
        self.add('placePiles',                   IndepVarComp('placePiles', 0.0), promotes=['*']) #12.0
        self.add('prepHamMono',                  IndepVarComp('prepHamMono', 0.0), promotes=['*']) #2.0
        self.add('prepHamJack',                  IndepVarComp('prepHamJack', 0.0), promotes=['*']) #2.0
        self.add('removeHamMono',                IndepVarComp('removeHamMono', 0.0), promotes=['*']) #2.0
        self.add('removeHamJack',                IndepVarComp('removeHamJack', 0.0), promotes=['*']) #4.0
        self.add('placeTemplate',                IndepVarComp('placeTemplate', 0.0), promotes=['*']) #4.0
        self.add('placeJack',                    IndepVarComp('placeJack', 0.0), promotes=['*']) #12.0
        self.add('levJack',                      IndepVarComp('levJack', 0.0), promotes=['*']) #24.0
        self.add('hamRate',                      IndepVarComp('hamRate', 0.0), promotes=['*']) #20.0
        self.add('placeMP',                      IndepVarComp('placeMP', 0.0), promotes=['*']) #3.0
        self.add('instScour',                    IndepVarComp('instScour', 0.0), promotes=['*']) #6.0
        self.add('placeTP',                      IndepVarComp('placeTP', 0.0), promotes=['*']) #3.0
        self.add('groutTP',                      IndepVarComp('groutTP', 0.0), promotes=['*']) #8.0
        self.add('tpCover',                      IndepVarComp('tpCover', 0.0), promotes=['*']) #1.50
        self.add('prepTow',                      IndepVarComp('prepTow', 0.0), promotes=['*']) #12.0
        self.add('spMoorCon',                    IndepVarComp('spMoorCon', 0.0), promotes=['*']) #20.0
        self.add('ssMoorCon',                    IndepVarComp('ssMoorCon', 0.0), promotes=['*']) #22.0
        self.add('spMoorCheck',                  IndepVarComp('spMoorCheck', 0.0), promotes=['*']) #16.0
        self.add('ssMoorCheck',                  IndepVarComp('ssMoorCheck', 0.0), promotes=['*']) #12.0
        self.add('ssBall',                       IndepVarComp('ssBall', 0.0), promotes=['*']) #6.0
        self.add('surfLayRate',                  IndepVarComp('surfLayRate', 0.0), promotes=['*']) #375.0
        self.add('cabPullIn',                    IndepVarComp('cabPullIn', 0.0), promotes=['*']) #5.50
        self.add('cabTerm',                      IndepVarComp('cabTerm', 0.0), promotes=['*']) #5.50
        self.add('cabLoadout',                   IndepVarComp('cabLoadout', 0.0), promotes=['*']) #14.0
        self.add('buryRate',                     IndepVarComp('buryRate', 0.0), promotes=['*']) #125.0
        self.add('subsPullIn',                   IndepVarComp('subsPullIn', 0.0), promotes=['*']) #48.0
        self.add('shorePullIn',                  IndepVarComp('shorePullIn', 0.0), promotes=['*']) #96.0
        self.add('landConstruct',                IndepVarComp('landConstruct', 0.0), promotes=['*']) #7.0
        self.add('expCabLoad',                   IndepVarComp('expCabLoad', 0.0), promotes=['*']) #24.0
        self.add('subsLoad',                     IndepVarComp('subsLoad', 0.0), promotes=['*']) #60.0
        self.add('placeTop',                     IndepVarComp('placeTop', 0.0), promotes=['*']) #24.0
        self.add('pileSpreadDR',                 IndepVarComp('pileSpreadDR', 0.0), promotes=['*']) #2500.0
        self.add('pileSpreadMob',                IndepVarComp('pileSpreadMob', 0.0), promotes=['*']) #750000.0
        self.add('groutSpreadDR',                IndepVarComp('groutSpreadDR', 0.0), promotes=['*']) #3000.0
        self.add('groutSpreadMob',               IndepVarComp('groutSpreadMob', 0.0), promotes=['*']) #1000000.0
        self.add('seaSpreadDR',                  IndepVarComp('seaSpreadDR', 0.0), promotes=['*']) #165000.0
        self.add('seaSpreadMob',                 IndepVarComp('seaSpreadMob', 0.0), promotes=['*']) #4500000.0
        self.add('compRacks',                    IndepVarComp('compRacks', 0.0), promotes=['*']) #1000000.0
        self.add('cabSurveyCR',                  IndepVarComp('cabSurveyCR', 0.0), promotes=['*']) #240.0
        self.add('cabDrillDist',                 IndepVarComp('cabDrillDist', 0.0), promotes=['*']) #500.0
        self.add('cabDrillCR',                   IndepVarComp('cabDrillCR', 0.0), promotes=['*']) #3200.0
        self.add('mpvRentalDR',                  IndepVarComp('mpvRentalDR', 0.0), promotes=['*']) #72000.0
        self.add('diveTeamDR',                   IndepVarComp('diveTeamDR', 0.0), promotes=['*']) #3200.0
        self.add('winchDR',                      IndepVarComp('winchDR', 0.0), promotes=['*']) #1000.0
        self.add('civilWork',                    IndepVarComp('civilWork', 0.0), promotes=['*']) #40000.0
        self.add('elecWork',                     IndepVarComp('elecWork', 0.0), promotes=['*']) #25000.0
        #Port & Staging', , 
        self.add('nCrane600',                    IndepVarComp('nCrane600', 0, pass_by_obj=True), promotes=['*']) #0
        self.add('nCrane1000',                   IndepVarComp('nCrane1000', 0, pass_by_obj=True), promotes=['*']) #0
        self.add('crane600DR',                   IndepVarComp('crane600DR', 0.0), promotes=['*']) #5000.0
        self.add('crane1000DR',                  IndepVarComp('crane1000DR', 0.0), promotes=['*']) #8000.0
        self.add('craneMobDemob',                IndepVarComp('craneMobDemob', 0.0), promotes=['*']) #150000.0
        self.add('entranceExitRate',             IndepVarComp('entranceExitRate', 0.0), promotes=['*']) #0.525
        self.add('dockRate',                     IndepVarComp('dockRate', 0.0), promotes=['*']) #3000.0
        self.add('wharfRate',                    IndepVarComp('wharfRate', 0.0), promotes=['*']) #2.75
        self.add('laydownCR',                    IndepVarComp('laydownCR', 0.0), promotes=['*']) #0.25
        #Engineering & Management', , 
        self.add('estEnMFac',                    IndepVarComp('estEnMFac', 0.0), promotes=['*']) #0.04
        #Development', , 
        self.add('preFEEDStudy',                 IndepVarComp('preFEEDStudy', 0.0), promotes=['*']) #5000000.0
        self.add('feedStudy',                    IndepVarComp('feedStudy', 0.0), promotes=['*']) #10000000.0
        self.add('stateLease',                   IndepVarComp('stateLease', 0.0), promotes=['*']) #250000.0
        self.add('outConShelfLease',             IndepVarComp('outConShelfLease', 0.0), promotes=['*']) #1000000.0
        self.add('saPlan',                       IndepVarComp('saPlan', 0.0), promotes=['*']) #500000.0
        self.add('conOpPlan',                    IndepVarComp('conOpPlan', 0.0), promotes=['*']) #1000000.0
        self.add('nepaEisMet',                   IndepVarComp('nepaEisMet', 0.0), promotes=['*']) #2000000.0
        self.add('physResStudyMet',              IndepVarComp('physResStudyMet', 0.0), promotes=['*']) #1500000.0
        self.add('bioResStudyMet',               IndepVarComp('bioResStudyMet', 0.0), promotes=['*']) #1500000.0
        self.add('socEconStudyMet',              IndepVarComp('socEconStudyMet', 0.0), promotes=['*']) #500000.0
        self.add('navStudyMet',                  IndepVarComp('navStudyMet', 0.0), promotes=['*']) #500000.0
        self.add('nepaEisProj',                  IndepVarComp('nepaEisProj', 0.0), promotes=['*']) #5000000.0
        self.add('physResStudyProj',             IndepVarComp('physResStudyProj', 0.0), promotes=['*']) #500000.0
        self.add('bioResStudyProj',              IndepVarComp('bioResStudyProj', 0.0), promotes=['*']) #500000.0
        self.add('socEconStudyProj',             IndepVarComp('socEconStudyProj', 0.0), promotes=['*']) #200000.0
        self.add('navStudyProj',                 IndepVarComp('navStudyProj', 0.0), promotes=['*']) #250000.0
        self.add('coastZoneManAct',              IndepVarComp('coastZoneManAct', 0.0), promotes=['*']) #100000.0
        self.add('rivsnHarbsAct',                IndepVarComp('rivsnHarbsAct', 0.0), promotes=['*']) #100000.0
        self.add('cleanWatAct402',               IndepVarComp('cleanWatAct402', 0.0), promotes=['*']) #100000.0
        self.add('cleanWatAct404',               IndepVarComp('cleanWatAct404', 0.0), promotes=['*']) #100000.0
        self.add('faaPlan',                      IndepVarComp('faaPlan', 0.0), promotes=['*']) #10000.0
        self.add('endSpecAct',                   IndepVarComp('endSpecAct', 0.0), promotes=['*']) #500000.0
        self.add('marMamProtAct',                IndepVarComp('marMamProtAct', 0.0), promotes=['*']) #500000.0
        self.add('migBirdAct',                   IndepVarComp('migBirdAct', 0.0), promotes=['*']) #500000.0
        self.add('natHisPresAct',                IndepVarComp('natHisPresAct', 0.0), promotes=['*']) #250000.0
        self.add('addLocPerm',                   IndepVarComp('addLocPerm', 0.0), promotes=['*']) #200000.0
        self.add('metTowCR',                     IndepVarComp('metTowCR', 0.0), promotes=['*']) #11518.0
        self.add('decomDiscRate',                IndepVarComp('decomDiscRate', 0.0), promotes=['*']) #0.03
        
        # LCOE
        self.add('number_of_turbines', IndepVarComp('number_of_turbines', 0, pass_by_obj=True), promotes=['*'])
        self.add('annual_opex',        IndepVarComp('annual_opex', 0.0), promotes=['*']) # TODO: Replace with output connection
        self.add('fixed_charge_rate',  IndepVarComp('fixed_charge_rate', 0.0), promotes=['*'])
        self.add('discount_rate',      IndepVarComp('discount_rate', 0.0), promotes=['*'])

        # Connect all input variables from all models
        self.connect('water_depth', ['sm.water_depth', 'wobos.waterD', 'sea_depth'])
        self.connect('hub_height', ['rotor.hub_height', 'sm.hub_height', 'wobos.hubH'])
        self.connect('tower_outer_diameter', 'wobos.towerD', src_indices=[0])
        
        self.connect('wind_beta', 'sm.beta')
        self.connect('cd_usr', 'sm.cd_usr')
        #self.connect('wave_beta', 'sm.waveLoads.beta')
        self.connect('yaw', 'sm.yaw')
        self.connect('mean_current_speed', 'sm.Uc')
        
        self.connect('project_lifetime', ['rotor.struc.lifetime','wobos.projLife'])
        #self.connect('slope_SN', 'sm.m_SN')
        #self.connect('compute_shear', 'sm.shear')
        #self.connect('compute_stiffnes', 'sm.geom')
        #self.connect('frame3dd_matrix_method', 'sm.Mmethod')
        #self.connect('shift_value', 'sm.shift')
        # TODO:
        #self.connect('number_of_modes', ['sm.nM', 'rotor.nF'])
        #self.connect('frame3dd_convergence_tolerance', 'sm.tol')
        #self.connect('lumped_mass_matrix', 'sm.lump')
        #self.connect('stress_standard_value', 'sm.DC')
        
        self.connect('safety_factor_frequency', ['rotor.gamma_freq', 'tcons.gamma_freq'])
        self.connect('safety_factor_stress', ['sm.gamma_f', 'rotor.gamma_f'])
        self.connect('safety_factor_materials', ['sm.gamma_m', 'rotor.gamma_m','tcons.gamma_m'])
        self.connect('safety_factor_buckling', 'sm.gamma_b')
        self.connect('safety_factor_fatigue', ['rotor.gamma_fatigue','sm.gamma_fatigue'])
        self.connect('safety_factor_consequence', 'sm.gamma_n')
        self.connect('rna.loads.top_F', 'sm.rna_force')
        self.connect('rna.loads.top_M', 'sm.rna_moment')
        self.connect('rna.rna_I_TT', 'sm.rna_I')
        self.connect('rna.rna_mass', ['wobos.rnaM', 'sm.rna_mass'])
        self.connect('rna.rna_cm', 'sm.rna_cg')
        self.connect('rotor.mass_all_blades', 'rna.blades_mass')
        self.connect('rotor.I_all_blades', 'rna.blades_I')
        self.connect('hub_mass', 'rna.hub_mass')
        self.connect('nac_mass', 'rna.nac_mass')
        self.connect('hub_cm', ['rna.hub_cm','tcons.hub_tt'])
        self.connect('nac_cm', 'rna.nac_cm')
        self.connect('hub_I', 'rna.hub_I')
        self.connect('nac_I', 'rna.nac_I')
        self.connect('tilt', 'rna.tilt')
        self.connect('rotor.Fxyz_total','rna.loads.F')
        self.connect('rotor.Mxyz_total','rna.loads.M')
        self.connect('rna_weightM','rna.rna_weightM')
        self.connect('rotor.Rhub', 'sm.sg.Rhub')
        
        self.connect('air_density', ['sm.base.windLoads.rho','rotor.analysis.rho'])
        self.connect('air_viscosity', ['sm.base.windLoads.mu','rotor.analysis.mu'])
        self.connect('water_density', 'sm.water_density')
        self.connect('water_viscosity', 'sm.base.waveLoads.mu')
        self.connect('wave_height', 'sm.Hs')
        self.connect('wave_period', 'sm.T')
        self.connect('wind_reference_speed', 'sm.Uref')
        self.connect('wind_reference_height', ['sm.zref','rotor.wind.zref'])
        self.connect('wind_bottom_height', ['sm.z0','rotor.wind.z0'])
        self.connect('shearExp', ['sm.shearExp', 'rotor.wind.shearExp'])
        self.connect('morison_mass_coefficient', 'sm.cm')
        self.connect('material_density', 'sm.material_density')
        self.connect('E', 'sm.E')
        self.connect('G', 'sm.G')
        self.connect('nu', 'sm.nu')
        self.connect('yield_stress', 'sm.yield_stress')
        
        self.connect('anchor', 'wobos.anchor')
        self.connect('ballast_cost_rate', 'wobos.ballCR')
        #self.connect('drag_embedment_extra_length', 'wobos.deaFixLeng')
        self.connect('mooring_cost_rate', 'wobos.moorCR')
        self.connect('sm.mm.mooring_cost', 'wobos.moorCost')
        self.connect('mooring_diameter', 'wobos.moorDia')
        self.connect('sm.mm.number_of_mooring_lines', 'wobos.moorLines')
        self.connect('outfitting_cost_rate', 'wobos.sSteelCR')
        self.connect('tapered_col_cost_rate', ['wobos.spStifColCR', 'wobos.spTapColCR', 'wobos.ssStifColCR'])
        self.connect('pontoon_cost_rate', 'wobos.ssTrussCR')

        self.connect('nBlades','blade_number')
        self.connect('rotor.mass_one_blade', 'blade_mass')
        self.connect('control_maxOmega', 'tcons.rotor_omega')
        self.connect('sm.structural_frequencies', 'tcons.tower_freq')
        self.connect('sm.tow.z_full', 'tcons.tower_z')
        self.connect('sm.tow.d_full', 'tcons.tower_d')
        
        self.connect('turbine_cost_kW', 'wobos.turbCapEx')
        self.connect('machine_rating', 'wobos.turbR')
        self.connect('rotor.diameter', 'wobos.rotorD')
        self.connect('bladeLength', 'wobos.bladeL')
        self.connect('rotor.max_chord', 'wobos.chord')
        self.connect('rotor.hub_diameter', 'wobos.hubD')
        self.connect('nacelleL', 'wobos.nacelleL') # TODO: RotorSE, remove variable
        self.connect('nacelleW', 'wobos.nacelleW') # TODO: RotorSE, remove variable
        self.connect('distShore', 'wobos.distShore')
        self.connect('distPort', 'wobos.distPort')
        self.connect('distPtoA', 'wobos.distPtoA')
        self.connect('distAtoS', 'wobos.distAtoS')
        self.connect('substructure', 'wobos.substructure')
        self.connect('turbInstallMethod', 'wobos.turbInstallMethod')
        self.connect('towerInstallMethod', 'wobos.towerInstallMethod')
        self.connect('installStrategy', 'wobos.installStrategy')
        self.connect('cableOptimizer', 'wobos.cableOptimizer')
        self.connect('buryDepth', 'wobos.buryDepth')
        self.connect('arrayY', 'wobos.arrayY') # TODO: Plant_EnergySE, remove variable
        self.connect('arrayX', 'wobos.arrayX') # TODO: Plant_EnergySE, remove variable
        self.connect('substructCont', 'wobos.substructCont')
        self.connect('turbCont', 'wobos.turbCont')
        self.connect('elecCont', 'wobos.elecCont')
        self.connect('interConVolt', 'wobos.interConVolt')
        self.connect('distInterCon', 'wobos.distInterCon')
        self.connect('scrapVal', 'wobos.scrapVal')
        self.connect('inspectClear', 'wobos.inspectClear')
        self.connect('plantComm', 'wobos.plantComm')
        self.connect('procurement_contingency', 'wobos.procurement_contingency')
        self.connect('install_contingency', 'wobos.install_contingency')
        self.connect('construction_insurance', 'wobos.construction_insurance')
        self.connect('capital_cost_year_0', 'wobos.capital_cost_year_0')
        self.connect('capital_cost_year_1', 'wobos.capital_cost_year_1')
        self.connect('capital_cost_year_2', 'wobos.capital_cost_year_2')
        self.connect('capital_cost_year_3', 'wobos.capital_cost_year_3')
        self.connect('capital_cost_year_4', 'wobos.capital_cost_year_4')
        self.connect('capital_cost_year_5', 'wobos.capital_cost_year_5')
        self.connect('interest_during_construction', 'wobos.interest_during_construction')
        self.connect('mpileCR', 'wobos.mpileCR') # TODO: JacketSE, remove variable
        self.connect('mtransCR', 'wobos.mtransCR') # TODO: JacketSE, remove variable
        self.connect('mpileD', 'wobos.mpileD') # TODO: JacketSE, remove variable
        self.connect('mpileL', 'wobos.mpileL') # TODO: JacketSE, remove variable
        self.connect('mpEmbedL', 'wobos.mpEmbedL') # TODO: JacketSE, remove variable
        self.connect('jlatticeCR', 'wobos.jlatticeCR') # TODO: JacketSE, remove variable
        self.connect('jtransCR', 'wobos.jtransCR') # TODO: JacketSE, remove variable
        self.connect('jpileCR', 'wobos.jpileCR') # TODO: JacketSE, remove variable
        self.connect('jlatticeA', 'wobos.jlatticeA') # TODO: JacketSE, remove variable
        self.connect('jpileL', 'wobos.jpileL') # TODO: JacketSE, remove variable
        self.connect('jpileD', 'wobos.jpileD') # TODO: JacketSE, remove variable
        self.connect('ssHeaveCR', 'wobos.ssHeaveCR')
        self.connect('scourMat', 'wobos.scourMat')
        self.connect('number_install_seasons', 'wobos.number_install_seasons')
        self.connect('pwrFac', 'wobos.pwrFac')
        self.connect('buryFac', 'wobos.buryFac')
        self.connect('catLengFac', 'wobos.catLengFac')
        self.connect('exCabFac', 'wobos.exCabFac')
        self.connect('subsTopFab', 'wobos.subsTopFab')
        self.connect('subsTopDes', 'wobos.subsTopDes')
        self.connect('topAssemblyFac', 'wobos.topAssemblyFac')
        self.connect('subsJackCR', 'wobos.subsJackCR')
        self.connect('subsPileCR', 'wobos.subsPileCR')
        self.connect('dynCabFac', 'wobos.dynCabFac')
        self.connect('shuntCR', 'wobos.shuntCR')
        self.connect('highVoltSG', 'wobos.highVoltSG')
        self.connect('medVoltSG', 'wobos.medVoltSG')
        self.connect('backUpGen', 'wobos.backUpGen')
        self.connect('workSpace', 'wobos.workSpace')
        self.connect('otherAncillary', 'wobos.otherAncillary')
        self.connect('mptCR', 'wobos.mptCR')
        self.connect('arrVoltage', 'wobos.arrVoltage')
        self.connect('cab1CR', 'wobos.cab1CR')
        self.connect('cab2CR', 'wobos.cab2CR')
        self.connect('cab1CurrRating', 'wobos.cab1CurrRating')
        self.connect('cab2CurrRating', 'wobos.cab2CurrRating')
        self.connect('arrCab1Mass', 'wobos.arrCab1Mass')
        self.connect('arrCab2Mass', 'wobos.arrCab2Mass')
        self.connect('cab1TurbInterCR', 'wobos.cab1TurbInterCR')
        self.connect('cab2TurbInterCR', 'wobos.cab2TurbInterCR')
        self.connect('cab2SubsInterCR', 'wobos.cab2SubsInterCR')
        self.connect('expVoltage', 'wobos.expVoltage')
        self.connect('expCurrRating', 'wobos.expCurrRating')
        self.connect('expCabMass', 'wobos.expCabMass')
        self.connect('expCabCR', 'wobos.expCabCR')
        self.connect('expSubsInterCR', 'wobos.expSubsInterCR')
        #self.connect('arrayCables', 'wobos.arrayCables')
        #self.connect('exportCables', 'wobos.exportCables')
        self.connect('moorTimeFac', 'wobos.moorTimeFac')
        self.connect('moorLoadout', 'wobos.moorLoadout')
        self.connect('moorSurvey', 'wobos.moorSurvey')
        self.connect('prepAA', 'wobos.prepAA')
        self.connect('prepSpar', 'wobos.prepSpar')
        self.connect('upendSpar', 'wobos.upendSpar')
        self.connect('prepSemi', 'wobos.prepSemi')
        self.connect('turbFasten', 'wobos.turbFasten')
        self.connect('boltTower', 'wobos.boltTower')
        self.connect('boltNacelle1', 'wobos.boltNacelle1')
        self.connect('boltNacelle2', 'wobos.boltNacelle2')
        self.connect('boltNacelle3', 'wobos.boltNacelle3')
        self.connect('boltBlade1', 'wobos.boltBlade1')
        self.connect('boltBlade2', 'wobos.boltBlade2')
        self.connect('boltRotor', 'wobos.boltRotor')
        self.connect('vesselPosTurb', 'wobos.vesselPosTurb')
        self.connect('vesselPosJack', 'wobos.vesselPosJack')
        self.connect('vesselPosMono', 'wobos.vesselPosMono')
        self.connect('subsVessPos', 'wobos.subsVessPos')
        self.connect('monoFasten', 'wobos.monoFasten')
        self.connect('jackFasten', 'wobos.jackFasten')
        self.connect('prepGripperMono', 'wobos.prepGripperMono')
        self.connect('prepGripperJack', 'wobos.prepGripperJack')
        self.connect('placePiles', 'wobos.placePiles')
        self.connect('prepHamMono', 'wobos.prepHamMono')
        self.connect('prepHamJack', 'wobos.prepHamJack')
        self.connect('removeHamMono', 'wobos.removeHamMono')
        self.connect('removeHamJack', 'wobos.removeHamJack')
        self.connect('placeTemplate', 'wobos.placeTemplate')
        self.connect('placeJack', 'wobos.placeJack')
        self.connect('levJack', 'wobos.levJack')
        self.connect('hamRate', 'wobos.hamRate')
        self.connect('placeMP', 'wobos.placeMP')
        self.connect('instScour', 'wobos.instScour')
        self.connect('placeTP', 'wobos.placeTP')
        self.connect('groutTP', 'wobos.groutTP')
        self.connect('tpCover', 'wobos.tpCover')
        self.connect('prepTow', 'wobos.prepTow')
        self.connect('spMoorCon', 'wobos.spMoorCon')
        self.connect('ssMoorCon', 'wobos.ssMoorCon')
        self.connect('spMoorCheck', 'wobos.spMoorCheck')
        self.connect('ssMoorCheck', 'wobos.ssMoorCheck')
        self.connect('ssBall', 'wobos.ssBall')
        self.connect('surfLayRate', 'wobos.surfLayRate')
        self.connect('cabPullIn', 'wobos.cabPullIn')
        self.connect('cabTerm', 'wobos.cabTerm')
        self.connect('cabLoadout', 'wobos.cabLoadout')
        self.connect('buryRate', 'wobos.buryRate')
        self.connect('subsPullIn', 'wobos.subsPullIn')
        self.connect('shorePullIn', 'wobos.shorePullIn')
        self.connect('landConstruct', 'wobos.landConstruct')
        self.connect('expCabLoad', 'wobos.expCabLoad')
        self.connect('subsLoad', 'wobos.subsLoad')
        self.connect('placeTop', 'wobos.placeTop')
        self.connect('pileSpreadDR', 'wobos.pileSpreadDR')
        self.connect('pileSpreadMob', 'wobos.pileSpreadMob')
        self.connect('groutSpreadDR', 'wobos.groutSpreadDR')
        self.connect('groutSpreadMob', 'wobos.groutSpreadMob')
        self.connect('seaSpreadDR', 'wobos.seaSpreadDR')
        self.connect('seaSpreadMob', 'wobos.seaSpreadMob')
        self.connect('compRacks', 'wobos.compRacks')
        self.connect('cabSurveyCR', 'wobos.cabSurveyCR')
        self.connect('cabDrillDist', 'wobos.cabDrillDist')
        self.connect('cabDrillCR', 'wobos.cabDrillCR')
        self.connect('mpvRentalDR', 'wobos.mpvRentalDR')
        self.connect('diveTeamDR', 'wobos.diveTeamDR')
        self.connect('winchDR', 'wobos.winchDR')
        self.connect('civilWork', 'wobos.civilWork')
        self.connect('elecWork', 'wobos.elecWork')
        self.connect('nCrane600', 'wobos.nCrane600')
        self.connect('nCrane1000', 'wobos.nCrane1000')
        self.connect('crane600DR', 'wobos.crane600DR')
        self.connect('crane1000DR', 'wobos.crane1000DR')
        self.connect('craneMobDemob', 'wobos.craneMobDemob')
        self.connect('entranceExitRate', 'wobos.entranceExitRate')
        self.connect('dockRate', 'wobos.dockRate')
        self.connect('wharfRate', 'wobos.wharfRate')
        self.connect('laydownCR', 'wobos.laydownCR')
        self.connect('estEnMFac', 'wobos.estEnMFac')
        self.connect('preFEEDStudy', 'wobos.preFEEDStudy')
        self.connect('feedStudy', 'wobos.feedStudy')
        self.connect('stateLease', 'wobos.stateLease')
        self.connect('outConShelfLease', 'wobos.outConShelfLease')
        self.connect('saPlan', 'wobos.saPlan')
        self.connect('conOpPlan', 'wobos.conOpPlan')
        self.connect('nepaEisMet', 'wobos.nepaEisMet')
        self.connect('physResStudyMet', 'wobos.physResStudyMet')
        self.connect('bioResStudyMet', 'wobos.bioResStudyMet')
        self.connect('socEconStudyMet', 'wobos.socEconStudyMet')
        self.connect('navStudyMet', 'wobos.navStudyMet')
        self.connect('nepaEisProj', 'wobos.nepaEisProj')
        self.connect('physResStudyProj', 'wobos.physResStudyProj')
        self.connect('bioResStudyProj', 'wobos.bioResStudyProj')
        self.connect('socEconStudyProj', 'wobos.socEconStudyProj')
        self.connect('navStudyProj', 'wobos.navStudyProj')
        self.connect('coastZoneManAct', 'wobos.coastZoneManAct')
        self.connect('rivsnHarbsAct', 'wobos.rivsnHarbsAct')
        self.connect('cleanWatAct402', 'wobos.cleanWatAct402')
        self.connect('cleanWatAct404', 'wobos.cleanWatAct404')
        self.connect('faaPlan', 'wobos.faaPlan')
        self.connect('endSpecAct', 'wobos.endSpecAct')
        self.connect('marMamProtAct', 'wobos.marMamProtAct')
        self.connect('migBirdAct', 'wobos.migBirdAct')
        self.connect('natHisPresAct', 'wobos.natHisPresAct')
        self.connect('addLocPerm', 'wobos.addLocPerm')
        self.connect('metTowCR', 'wobos.metTowCR')
        self.connect('decomDiscRate', 'wobos.decomDiscRate')
        
        # Link outputs from one model to inputs to another
        self.connect('tower_mass', 'wobos.towerM')
        self.connect('dummy_mass', 'sm.aux.stack_mass_in')

        self.connect('sm.total_cost', 'wobos.subTotCost')
        self.connect('sm.total_mass', 'wobos.subTotM')
        self.connect('wobos.total_bos_cost', 'bos_costs')

        self.connect('number_of_turbines', ['wobos.nTurb', 'turbine_number'])
        self.connect('annual_opex', 'avg_annual_opex')
        self.connect('rotor.AEP', 'net_aep')
        self.connect('wobos.totInstTime', 'construction_time')
        
         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr

