from floatingse.instance.floating_instance import FloatingInstance, NSECTIONS, NPTS, vecOption, Five_strings, Ten_strings
from wisdem.floating.floating_turbine_assembly import FloatingTurbine
from commonse import eps
from commonse.csystem import rotMat_x, rotMat_y, rotMat_z
from rotorse import TURBULENCE_CLASS, TURBINE_CLASS, DRIVETRAIN_TYPE, NREL5MW, DTU10MW
import numpy as np
import offshorebos.wind_obos as wind_obos
import time
from StringIO import StringIO

from mayavi import mlab

NDEL = 0


class FloatingTurbineInstance(FloatingInstance):
    def __init__(self, refStr):
        super(FloatingTurbineInstance, self).__init__()

        if type(refStr) != type(''):
            raise ValueError('Must enter reference turbine name as a string')
        if refStr in Five_strings:
            self.refBlade = NREL5MW()
        elif refStr in Ten_strings:
            self.refBlade = DTU10MW()
        else:
            raise ValueError('Unknown reference turbine name, '+refStr)
            
        # Remove what we don't need from Semi
        self.params.pop('rna_cg', None)
        self.params.pop('rna_mass', None)
        self.params.pop('rna_I', None)
        self.params.pop('rna_moment', None)
        self.params.pop('rna_force', None)
        self.params.pop('sg.Rhub', None)

        # Environmental parameters
        self.params['air_density'] = self.params['base.windLoads.rho']
        self.params.pop('base.windLoads.rho')
        
        self.params['air_viscosity'] = self.params['base.windLoads.mu']
        self.params.pop('base.windLoads.mu', None)
        
        self.params['water_viscosity'] = self.params['base.waveLoads.mu']
        self.params.pop('base.waveLoads.mu')
        
        self.params['wave_height'] = self.params['Hs']
        self.params.pop('Hs')
        
        self.params['wave_period'] = self.params['T']
        self.params.pop('T', None)
        
        self.params['mean_current_speed'] = self.params['Uc']
        self.params.pop('Uc', None)

        self.params['wind_reference_speed'] = self.params['Uref']
        self.params.pop('Uref', None)
        
        self.params['wind_reference_height'] = self.params['zref']
        self.params.pop('zref')
        
        self.params['shearExp'] = 0.11

        self.params['morison_mass_coefficient'] = self.params['cm']
        self.params.pop('cm', None)
        
        self.params['wind_bottom_height'] = self.params['z0']
        self.params.pop('z0', None)

        self.params['wind_beta'] = self.params['beta']
        self.params.pop('beta', None)
        
        #self.params['wave_beta']                            = 0.0

        self.params['hub_height']                            = 90.0
        
        self.params['safety_factor_frequency']               = 1.1
        self.params['safety_factor_stress']                  = 1.35
        self.params['safety_factor_materials']               = 1.3
        self.params['safety_factor_buckling']                = 1.1
        self.params['safety_factor_fatigue']                 = 1.35*1.3*1.0
        self.params['safety_factor_consequence']             = 1.0
        self.params.pop('gamma_f', None)
        self.params.pop('gamma_m', None)
        self.params.pop('gamma_n', None)
        self.params.pop('gamma_b', None)
        self.params.pop('gamma_fatigue', None)
      
        self.params['project_lifetime']                      = 20.0
        self.params['number_of_turbines']                    = 20
        self.params['annual_opex']                           = 7e5
        self.params['fixed_charge_rate']                     = 0.12
        self.params['discount_rate']                         = 0.07

        # For RotorSE
        self.params['hubFraction']                           = 0.025
        self.params['bladeLength']                           = 61.5
        self.params['r_max_chord']                           = 0.23577
        self.params['chord_in']                              = np.array([3.2612, 4.5709, 3.3178, 1.4621])
        self.params['theta_in']                              = np.array([13.2783, 7.46036, 2.89317, -0.0878099])
        self.params['precone']                               = 2.5
        self.params['tilt']                                  = 5.0
        self.params['control_Vin']                           = 3.0
        self.params['control_Vout']                          = 25.0
        self.params['machine_rating']                        = 5e6
        self.params['control_minOmega']                      = 0.0
        self.params['control_maxOmega']                      = 12.0
        self.params['control_tsr']                           = 7.55
        self.params['sparT_in']                              = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])
        self.params['teT_in']                                = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])

        self.params['precurve_in']                           = np.array([0.0, 0.0, 0.0])
        self.params['presweep_in']                           = np.array([0.0, 0.0, 0.0])
        self.params['precurve_tip']                          = 0.0
        self.params['presweep_tip']                          = 0.0
        self.params['yaw']                                   = 0.0
        self.params['nBlades']                               = 3
        self.params['turbine_class']                         = TURBINE_CLASS['I']
        self.params['turbulence_class']                      = TURBULENCE_CLASS['B']
        self.params['drivetrainType']                        = DRIVETRAIN_TYPE['GEARED']
        self.params['gust_stddev']                           = 3
        self.params['control_pitch']                         = 0.0
        self.params['VfactorPC']                             = 0.7
        self.params['pitch_extreme']                         = 0.0
        self.params['azimuth_extreme']                       = 0.0
        self.params['rstar_damage']                          = np.linspace(0.0, 1.0, len(self.refBlade.r)+1)
        self.params['Mxb_damage']                            = eps * np.ones(len(self.refBlade.r)+1)
        self.params['Myb_damage']                            = eps * np.ones(len(self.refBlade.r)+1)
        self.params['strain_ult_spar']                       = 1e-2
        self.params['strain_ult_te']                         = 2*2500*1e-6
        self.params['m_damage']                              = 10.0
        self.params['nSector']                               = 4
        self.params['tiploss']                               = True
        self.params['hubloss']                               = True
        self.params['wakerotation']                          = True 
        self.params['usecd']                                 = True
        self.params['AEP_loss_factor']                       = 1.0
        self.params['dynamic_amplication_tip_deflection']    = 1.35
        self.params['shape_parameter']                       = 0.0

        # For RNA
        self.params['hub_mass']                              = 0.1*285599.0
        self.params['nac_mass']                              = 0.5*285599.0
        self.params['hub_cm']                                = np.array([-1.13197635, 0.0, 0.50875268])
        self.params['nac_cm']                                = np.array([-1.13197635, 0.0, 0.50875268])
        self.params['hub_I']                                 = 0.1*np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 0.0, 5.03710467e+05])
        self.params['nac_I']                                 = 0.1*np.array([1.14930678e+08, 2.20354030e+07, 1.87597425e+07, 0.0, 0.0, 5.03710467e+05])
        self.params['downwind']                              = False
        self.params['rna_weightM']                           = True

        # For turbine costs
        self.params['blade_mass_cost_coeff']                 = 13.08
        self.params['hub_mass_cost_coeff']                   = 3.8
        self.params['pitch_system_mass_cost_coeff']          = 22.91
        self.params['spinner_mass_cost_coeff']               = 23.0
        self.params['lss_mass_cost_coeff']                   = 12.6
        self.params['bearings_mass_cost_coeff']              = 6.35
        self.params['gearbox_mass_cost_coeff']               = 17.4
        self.params['high_speed_side_mass_cost_coeff']       = 8.25
        self.params['generator_mass_cost_coeff']             = 17.43
        self.params['bedplate_mass_cost_coeff']              = 4.5
        self.params['yaw_system_mass_cost_coeff']            = 11.01
        self.params['variable_speed_elec_mass_cost_coeff']   = 26.5
        self.params['hydraulic_cooling_mass_cost_coeff']     = 163.95
        self.params['nacelle_cover_mass_cost_coeff']         = 7.61
        self.params['elec_connec_machine_rating_cost_coeff'] = 40.0
        self.params['nacelle_platforms_mass_cost_coeff']     = 8.7
        self.params['base_hardware_cost_coeff']              = 0.7
        self.params['transformer_mass_cost_coeff']           = 26.5
        self.params['tower_mass_cost_coeff']                 = 3.20
        self.params['hub_assemblyCostMultiplier']            = 0.0
        self.params['hub_overheadCostMultiplier']            = 0.0
        self.params['nacelle_assemblyCostMultiplier']        = 0.0
        self.params['nacelle_overheadCostMultiplier']        = 0.0
        self.params['tower_assemblyCostMultiplier']          = 0.0
        self.params['tower_overheadCostMultiplier']          = 0.0
        self.params['turbine_assemblyCostMultiplier']        = 0.0
        self.params['turbine_overheadCostMultiplier']        = 0.0
        self.params['hub_profitMultiplier']                  = 0.0
        self.params['nacelle_profitMultiplier']              = 0.0
        self.params['tower_profitMultiplier']                = 0.0
        self.params['turbine_profitMultiplier']              = 0.0
        self.params['hub_transportMultiplier']               = 0.0
        self.params['nacelle_transportMultiplier']           = 0.0
        self.params['tower_transportMultiplier']             = 0.0
        self.params['turbine_transportMultiplier']           = 0.0
        self.params['offshore']                              = True
        self.params['crane']                                 = False
        self.params['bearing_number']       = 2
        self.params['bedplate_mass']        = 93090.6
        self.params['controls_cost_base']   = np.array([35000.0,55900.0])
        self.params['controls_esc']         = 1.5
        self.params['crane_cost']           = 0.0
        self.params['elec_connec_cost_esc'] = 1.5
        self.params['gearbox_mass']         = 30237.60
        self.params['generator_mass']       = 16699.85
        self.params['hss_mass']             = 1492.45
        self.params['hvac_mass']            = 1e3
        self.params['lss_mass']             = 31257.3
        self.params['main_bearing_mass']    = 9731.41 / 2
        self.params['cover_mass']           = 1e3
        self.params['platforms_mass']       = 1e3
        self.params['pitch_system_mass']    = 17004.0
        self.params['spinner_mass']         = 1810.5
        self.params['transformer_mass']     = 1e3
        self.params['vs_electronics_mass']  = 1e3
        self.params['yaw_mass']             = 11878.24
        
        # Offshore BOS
        # Turbine / Plant parameters
        self.params['nacelleL']                              = -np.inf
        self.params['nacelleW']                              = -np.inf
        self.params['distShore']                             = 90.0
        self.params['distPort']                              = 90.0
        self.params['distPtoA']                              = 90.0
        self.params['distAtoS']                              = 90.0
        self.params['substructure']                          = wind_obos.Substructure.SEMISUBMERSIBLE
        self.params['anchor']                                = wind_obos.Anchor.DRAGEMBEDMENT
        self.params['turbInstallMethod']                     = wind_obos.TurbineInstall.INDIVIDUAL
        self.params['towerInstallMethod']                    = wind_obos.TowerInstall.ONEPIECE
        self.params['installStrategy']                       = wind_obos.InstallStrategy.PRIMARYVESSEL
        self.params['cableOptimizer']                        = False
        self.params['buryDepth']                             = 2.0
        self.params['arrayY']                                = 9.0
        self.params['arrayX']                                = 9.0
        self.params['substructCont']                         = 0.30
        self.params['turbCont']                              = 0.30
        self.params['elecCont']                              = 0.30
        self.params['interConVolt']                          = 345.0
        self.params['distInterCon']                          = 3.0
        self.params['scrapVal']                              = 0.0
        #General']                                           = , 
        self.params['inspectClear']                          = 2.0
        self.params['plantComm']                             = 0.01
        self.params['procurement_contingency']               = 0.05
        self.params['install_contingency']                   = 0.30
        self.params['construction_insurance']                = 0.01
        self.params['capital_cost_year_0']                   = 0.20
        self.params['capital_cost_year_1']                   = 0.60
        self.params['capital_cost_year_2']                   = 0.10
        self.params['capital_cost_year_3']                   = 0.10
        self.params['capital_cost_year_4']                   = 0.0
        self.params['capital_cost_year_5']                   = 0.0
        self.params['tax_rate']                              = 0.40
        self.params['interest_during_construction']          = 0.08
        #Substructure & Foundation']                         = , 
        self.params['mpileCR']                               = 2250.0
        self.params['mtransCR']                              = 3230.0
        self.params['mpileD']                                = 0.0
        self.params['mpileL']                                = 0.0
        self.params['mpEmbedL']                              = 30.0
        self.params['jlatticeCR']                            = 4680.0
        self.params['jtransCR']                              = 4500.0
        self.params['jpileCR']                               = 2250.0
        self.params['jlatticeA']                             = 26.0
        self.params['jpileL']                                = 47.50
        self.params['jpileD']                                = 1.60
        self.params['ssHeaveCR']                             = 6250.0
        self.params['scourMat']                              = 250000.0
        self.params['number_install_seasons']                = 1.0
        #Electrical Infrastructure']                         = , 
        self.params['pwrFac']                                = 0.95
        self.params['buryFac']                               = 0.10
        self.params['catLengFac']                            = 0.04
        self.params['exCabFac']                              = 0.10
        self.params['subsTopFab']                            = 14500.0
        self.params['subsTopDes']                            = 4500000.0
        self.params['topAssemblyFac']                        = 0.075
        self.params['subsJackCR']                            = 6250.0
        self.params['subsPileCR']                            = 2250.0
        self.params['dynCabFac']                             = 2.0
        self.params['shuntCR']                               = 35000.0
        self.params['highVoltSG']                            = 950000.0
        self.params['medVoltSG']                             = 500000.0
        self.params['backUpGen']                             = 1000000.0
        self.params['workSpace']                             = 2000000.0
        self.params['otherAncillary']                        = 3000000.0
        self.params['mptCR']                                 = 12500.0
        self.params['arrVoltage']                            = 33.0
        self.params['cab1CR']                                = 185.889
        self.params['cab2CR']                                = 202.788
        self.params['cab1CurrRating']                        = 300.0
        self.params['cab2CurrRating']                        = 340.0
        self.params['arrCab1Mass']                           = 20.384
        self.params['arrCab2Mass']                           = 21.854
        self.params['cab1TurbInterCR']                       = 8410.0
        self.params['cab2TurbInterCR']                       = 8615.0
        self.params['cab2SubsInterCR']                       = 19815.0
        self.params['expVoltage']                            = 220.0
        self.params['expCurrRating']                         = 530.0
        self.params['expCabMass']                            = 71.90
        self.params['expCabCR']                              = 495.411
        self.params['expSubsInterCR']                        = 57500.0
        # Vector inputs
        #self.params['arrayCables']                          = [33, 66]
        #self.params['exportCables']                         = [132, 220]
        #Assembly & Installation',
        self.params['moorTimeFac']                           = 0.005
        self.params['moorLoadout']                           = 5.0
        self.params['moorSurvey']                            = 4.0
        self.params['prepAA']                                = 168.0
        self.params['prepSpar']                              = 18.0
        self.params['upendSpar']                             = 36.0
        self.params['prepSemi']                              = 12.0
        self.params['turbFasten']                            = 8.0
        self.params['boltTower']                             = 7.0
        self.params['boltNacelle1']                          = 7.0
        self.params['boltNacelle2']                          = 7.0
        self.params['boltNacelle3']                          = 7.0
        self.params['boltBlade1']                            = 3.50
        self.params['boltBlade2']                            = 3.50
        self.params['boltRotor']                             = 7.0
        self.params['vesselPosTurb']                         = 2.0
        self.params['vesselPosJack']                         = 8.0
        self.params['vesselPosMono']                         = 3.0
        self.params['subsVessPos']                           = 6.0
        self.params['monoFasten']                            = 12.0
        self.params['jackFasten']                            = 20.0
        self.params['prepGripperMono']                       = 1.50
        self.params['prepGripperJack']                       = 8.0
        self.params['placePiles']                            = 12.0
        self.params['prepHamMono']                           = 2.0
        self.params['prepHamJack']                           = 2.0
        self.params['removeHamMono']                         = 2.0
        self.params['removeHamJack']                         = 4.0
        self.params['placeTemplate']                         = 4.0
        self.params['placeJack']                             = 12.0
        self.params['levJack']                               = 24.0
        self.params['hamRate']                               = 20.0
        self.params['placeMP']                               = 3.0
        self.params['instScour']                             = 6.0
        self.params['placeTP']                               = 3.0
        self.params['groutTP']                               = 8.0
        self.params['tpCover']                               = 1.50
        self.params['prepTow']                               = 12.0
        self.params['spMoorCon']                             = 20.0
        self.params['ssMoorCon']                             = 22.0
        self.params['spMoorCheck']                           = 16.0
        self.params['ssMoorCheck']                           = 12.0
        self.params['ssBall']                                = 6.0
        self.params['surfLayRate']                           = 375.0
        self.params['cabPullIn']                             = 5.50
        self.params['cabTerm']                               = 5.50
        self.params['cabLoadout']                            = 14.0
        self.params['buryRate']                              = 125.0
        self.params['subsPullIn']                            = 48.0
        self.params['shorePullIn']                           = 96.0
        self.params['landConstruct']                         = 7.0
        self.params['expCabLoad']                            = 24.0
        self.params['subsLoad']                              = 60.0
        self.params['placeTop']                              = 24.0
        self.params['pileSpreadDR']                          = 2500.0
        self.params['pileSpreadMob']                         = 750000.0
        self.params['groutSpreadDR']                         = 3000.0
        self.params['groutSpreadMob']                        = 1000000.0
        self.params['seaSpreadDR']                           = 165000.0
        self.params['seaSpreadMob']                          = 4500000.0
        self.params['compRacks']                             = 1000000.0
        self.params['cabSurveyCR']                           = 240.0
        self.params['cabDrillDist']                          = 500.0
        self.params['cabDrillCR']                            = 3200.0
        self.params['mpvRentalDR']                           = 72000.0
        self.params['diveTeamDR']                            = 3200.0
        self.params['winchDR']                               = 1000.0
        self.params['civilWork']                             = 40000.0
        self.params['elecWork']                              = 25000.0
        #Port & Staging']                                    = , 
        self.params['nCrane600']                             = 0.0
        self.params['nCrane1000']                            = 0.0
        self.params['crane600DR']                            = 5000.0
        self.params['crane1000DR']                           = 8000.0
        self.params['craneMobDemob']                         = 150000.0
        self.params['entranceExitRate']                      = 0.525
        self.params['dockRate']                              = 3000.0
        self.params['wharfRate']                             = 2.75
        self.params['laydownCR']                             = 0.25
        #Engineering & Management']                          = , 
        self.params['estEnMFac']                             = 0.04
        #Development']                                       = , 
        self.params['preFEEDStudy']                          = 5000000.0
        self.params['feedStudy']                             = 10000000.0
        self.params['stateLease']                            = 250000.0
        self.params['outConShelfLease']                      = 1000000.0
        self.params['saPlan']                                = 500000.0
        self.params['conOpPlan']                             = 1000000.0
        self.params['nepaEisMet']                            = 2000000.0
        self.params['physResStudyMet']                       = 1500000.0
        self.params['bioResStudyMet']                        = 1500000.0
        self.params['socEconStudyMet']                       = 500000.0
        self.params['navStudyMet']                           = 500000.0
        self.params['nepaEisProj']                           = 5000000.0
        self.params['physResStudyProj']                      = 500000.0
        self.params['bioResStudyProj']                       = 500000.0
        self.params['socEconStudyProj']                      = 200000.0
        self.params['navStudyProj']                          = 250000.0
        self.params['coastZoneManAct']                       = 100000.0
        self.params['rivsnHarbsAct']                         = 100000.0
        self.params['cleanWatAct402']                        = 100000.0
        self.params['cleanWatAct404']                        = 100000.0
        self.params['faaPlan']                               = 10000.0
        self.params['endSpecAct']                            = 500000.0
        self.params['marMamProtAct']                         = 500000.0
        self.params['migBirdAct']                            = 500000.0
        self.params['natHisPresAct']                         = 250000.0
        self.params['addLocPerm']                            = 200000.0
        self.params['metTowCR']                              = 11518.0
        self.params['decomDiscRate']                         = 0.03

        self.params['dummy_mass']                            = eps


    def set_reference(self, instr):
        if instr.upper() in Five_strings:
            myref = NREL5MW()

            self.params['hub_mass'] = 56.780e3
            self.params['nac_mass'] = 240e3
            self.params['hub_cm']   = np.array([-5.01910, 0.0, 1.96256])
            self.params['nac_cm']   = np.array([1.9, 0.0, 1.75])
            self.params['hub_I']    = self.params['hub_mass']*1.75**2. * np.r_[(2./3.), (5./12.), (5./12.), np.zeros(3)]
            self.params['nac_I']    = np.array([7.77616624894e7, 8.34033992e+05, 8.34033992e+05, 0.0, 2.05892434e+05, 0.0])
            self.params['rna_weightM'] = True
            
        elif instr.upper() in Ten_strings:
            myref = DTU10MW()

            self.params['hub_mass'] = 105520.0
            self.params['nac_mass'] = 446036.25
            self.params['hub_cm']   = np.array([-7.1, 0.0, 2.75])
            self.params['nac_cm']   = np.array([2.69, 0.0, 2.40])
            self.params['hub_I']    = self.params['hub_mass']*2.152**2. * np.r_[(2./3.), (5./12.), (5./12.), np.zeros(3)]
            self.params['nac_I']    = self.params['nac_mass']*(1./12.) * np.r_[(10**2+10**2), (10**2+15**2), (15**2+10**2), np.zeros(3)]
            self.params['rna_weightM'] = True

        # Set blade/rotor values from reference definition in RotorSE
        self.params['hubFraction']      = myref.hubFraction
        self.params['bladeLength']      = myref.bladeLength
        self.params['precone']          = myref.precone
        self.params['tilt']             = myref.tilt
        self.params['nBlades']          = myref.nBlades
        self.params['downwind']         = myref.downwind
        self.params['r_max_chord']      = myref.r_max_chord
        self.params['chord_in']         = myref.chord
        self.params['theta_in']         = myref.theta
        self.params['precurve_in']      = myref.precurve
        self.params['presweep_in']      = myref.presweep
        self.params['sparT_in']         = myref.spar_thickness
        self.params['teT_in']           = myref.te_thickness
        self.params['hub_height']       = myref.hub_height
        self.params['turbine_class']    = myref.turbine_class
        self.params['wind_reference_height'] = myref.hub_height
        self.params['control_Vin']      = myref.control_Vin
        self.params['control_Vout']     = myref.control_Vout
        self.params['control_minOmega'] = myref.control_minOmega
        self.params['control_maxOmega'] = myref.control_maxOmega
        self.params['control_tsr']      = myref.control_tsr
        self.params['control_pitch']    = myref.control_pitch
        self.params['machine_rating']   = myref.rating
        self.params['drivetrainType']   = myref.drivetrain

        super(FloatingTurbineInstance, self).set_reference(instr)
        
        
    def get_assembly(self): return FloatingTurbine(self.refBlade, NSECTIONS)

    def add_objective(self):
        if (len(self.prob.driver._objs) == 0):
            self.prob.driver.add_objective('lcoe')
            
        
    def get_constraints(self):
        conList = super(FloatingTurbineInstance, self).get_constraints()
        for con in conList:
            con[0] = 'sm.' + con[0]

        conList.extend( [['rotor.Pn_margin', None, 1.0, None],
                         ['rotor.P1_margin', None, 1.0, None],
                         ['rotor.Pn_margin_cfem', None, 1.0, None],
                         ['rotor.P1_margin_cfem', None, 1.0, None],
                         ['rotor.rotor_strain_sparU', -1.0, None, None],
                         ['rotor.rotor_strain_sparL', None, 1.0, None],
                         ['rotor.rotor_strain_teU', -1.0, None, None],
                         ['rotor.rotor_strain_teL', None, 1.0, None],
                         ['rotor.rotor_buckling_sparU', None, 1.0, None],
                         ['rotor.rotor_buckling_sparL', None, 1.0, None],
                         ['rotor.rotor_buckling_teU', None, 1.0, None],
                         ['rotor.rotor_buckling_teL', None, 1.0, None],
                         ['rotor.rotor_damage_sparU', None, 0.0, None],
                         ['rotor.rotor_damage_sparL', None, 0.0, None],
                         ['rotor.rotor_damage_teU', None, 0.0, None],
                         ['rotor.rotor_damage_teL', None, 0.0, None],
                         #['tcons.frequency1P_margin_low', None, 1.0, None],
                         #['tcons.frequency1P_margin_high', 1.0, None, None],
                         #['tcons.frequency3P_margin_low', None, 1.0, None],
                         #['tcons.frequency3P_margin_high', 1.0, None, None],
                         ['tcons.tip_deflection_ratio', None, 1.0, None],
                         ['tcons.ground_clearance', 20.0, None, None],
        ])
        return conList

    def draw_rna(self, fig):
        if fig is None: fig=self.init_figure()

        # Quantities from input and output simulatioin parameters
        r_cylinder   = self.refBlade.r_cylinder
        bladeLength  = self.params['bladeLength']
        hubD         = 2*self.prob['rotor.Rhub']
        nblade       = self.params['nBlades']
        pitch        = 0.0
        precone      = self.params['precone']
        tilt         = self.params['tilt']
        hubH         = self.params['hub_height']
        cm_hub       = self.prob['hub_cm']
        cm_hub[-1]  += hubH
        chord        = self.prob['rotor.chord']
        thick        = self.prob['rotor.chord'] / self.refBlade.chord_ref
        twist        = self.prob['rotor.theta']
        precurve     = self.prob['rotor.precurve']
        presweep     = self.prob['rotor.presweep']
        le_loc       = self.refBlade.le_location

        # Rotation matrices
        T_tilt    = rotMat_y(np.deg2rad(tilt))
        T_precone = rotMat_y(np.deg2rad(-precone))
        T_pitch   = rotMat_z(np.deg2rad(-pitch))

        # Spanwise coordinates
        r_blade = self.prob['rotor.r_pts']

        # Airfoil coordinates
        afcoord = self.refBlade.getAirfoilCoordinates()

        # Assemble airfoil coordinates along the span
        # Not flip x and y and flip sign to be consistent with global coordinate system at pitch=0:
        # +x downstream
        # +y towards TE
        # +z towards sky
        X = []
        Y = []
        for k in range(len(r_blade)):
            thisy = afcoord[k][:,0].copy()
            thisx = afcoord[k][:,1].copy()

            # Pre-twist modifications
            thisy -= le_loc[k]
            thisx *= chord[k] * thick[k]
            thisy *= chord[k]

            # Chord scaling and twist rotation
            thismat = np.asmatrix( np.c_[thisx, thisy] ).T
            T_th    = rotMat_z( np.deg2rad(-twist[k]) )[:2,:2]
            thismat = np.asarray( (T_th * thismat).T )

            if k==0:
                X = thismat[:,0]
                Y = thismat[:,1]
            else:
                X = np.c_[X, thismat[:,0] + precurve[k]]
                Y = np.c_[Y, thismat[:,1] + presweep[k]]

        # Set Z-positions
        Z = r_blade[np.newaxis,:] * np.ones(X.shape)

        # Create plot of blades
        bladeAng = np.linspace(0, 2*np.pi, nblade+1)[:nblade]
        orig = X.shape
        for a in bladeAng:
            T_tot   = T_tilt * rotMat_x(a) * T_precone * T_pitch
            thismat = np.asmatrix( np.c_[X.flatten(), Y.flatten(), Z.flatten()] ).T
            thismat = np.asarray( (T_tot * thismat).T ) + cm_hub[np.newaxis,:]
            Xplot, Yplot, Zplot = thismat[:,0], thismat[:,1], thismat[:,2]
            mlab.mesh(Xplot.reshape(orig), Yplot.reshape(orig), Zplot.reshape(orig), color=(1,1,1), figure=fig)

        # Now do hub
        npts    = 30
        rk      = 0.5*hubD*np.ones(npts)
        th      = np.linspace(0,2*np.pi,npts)
        x       = np.linspace(-0.5, 0.5, npts) * 1.5*chord[0]
        R, TH   = np.meshgrid(rk, th)
        X, _    = np.meshgrid(x, th)
        Y       = R*np.cos(TH)
        Z       = R*np.sin(TH)
        orig    = X.shape
        thismat = np.asmatrix( np.c_[X.flatten(), Y.flatten(), Z.flatten()] ).T
        thismat = np.asarray( (T_tilt * thismat).T ) + cm_hub[np.newaxis,:]
        Xplot, Yplot, Zplot = thismat[:,0], thismat[:,1], thismat[:,2]
        mlab.mesh(Xplot.reshape(orig), Yplot.reshape(orig), Zplot.reshape(orig), color=(0.9,)*3, figure=fig)

        ph      = np.linspace(0, 0.5*np.pi, npts) + np.pi
        PH,_    = np.meshgrid(ph,th)
        Y       = R*np.cos(TH)*np.sin(PH)
        Z       = R*np.sin(TH)*np.sin(PH)
        X       = R*np.cos(PH) - 0.75*chord[0]
        orig    = X.shape
        thismat = np.asmatrix( np.c_[X.flatten(), Y.flatten(), Z.flatten()] ).T
        thismat = np.asarray( (T_tilt * thismat).T ) + cm_hub[np.newaxis,:]
        Xplot, Yplot, Zplot = thismat[:,0], thismat[:,1], thismat[:,2]
        mlab.mesh(Xplot.reshape(orig), Yplot.reshape(orig), Zplot.reshape(orig), color=(0.9,)*3, figure=fig)

        # Now do nacelle
        nacW   = nacH = hubD + 2.0
        nacL   = nacW + 5.0
        cm_nac = cm_hub + 0.5*np.array([nacL, 0.0, 0.0])
        cm_nac[0] += 0.75*chord[0]
        xx     = np.array([-0.5, 0.0, 0.5])
        PX,PY  = np.meshgrid(xx, xx)
        PZ     = np.ones(PX.shape)
        # Top and bottom
        mlab.mesh(PX*nacL + cm_nac[0], PY*nacW + cm_nac[1], -PZ*0.5*nacH + cm_nac[2], color=(0.9,)*3, figure=fig)
        mlab.mesh(PX*nacL + cm_nac[0], PY*nacW + cm_nac[1],  PZ*0.5*nacH + cm_nac[2], color=(0.9,)*3, figure=fig)
        # Sides
        mlab.mesh(PX*nacL + cm_nac[0], -PZ*nacW*0.5 + cm_nac[1], PY*nacH + cm_nac[2], color=(0.9,)*3, figure=fig)
        mlab.mesh(PX*nacL + cm_nac[0],  PZ*nacW*0.5 + cm_nac[1], PY*nacH + cm_nac[2], color=(0.9,)*3, figure=fig)
        # Front and Back
        mlab.mesh(-PZ*nacL*0.5 + cm_nac[0], PX*nacW + cm_nac[1], PY*nacH + cm_nac[2], color=(0.9,)*3, figure=fig)
        mlab.mesh( PZ*nacL*0.5 + cm_nac[0], PX*nacW + cm_nac[1], PY*nacH + cm_nac[2], color=(0.9,)*3, figure=fig)



        
