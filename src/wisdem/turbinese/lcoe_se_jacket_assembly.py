"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np

from openmdao.main.api import Assembly, Component
from openmdao.main.datatypes.api import Int, Float, Enum, VarTree, Bool, Str, Array

import os
import copy

from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection
from jacketse.jacket import JcktGeoInputs,SoilGeoInputs,WaterInputs,WindInputs,RNAprops,TPlumpMass,Frame3DDaux,\
                    MatInputs,LegGeoInputs,XBrcGeoInputs,MudBrcGeoInputs,HBrcGeoInputs,TPGeoInputs,PileGeoInputs,\
                    TwrGeoInputs, LegGeoOutputs, TwrGeoOutputs

from fusedwind.plant_cost.fused_finance import configure_extended_financial_analysis, ExtendedFinancialAnalysis
from fusedwind.plant_cost.fused_opex import OPEXVarTree
from fusedwind.plant_cost.fused_bos_costs import BOSVarTree
from fusedwind.interface import implement_base

from wisdem.turbinese.turbine_jacket import configure_turbine_with_jacket
from turbine_costsse.turbine_costsse import Turbine_CostsSE
#from turbine_costsse.turbine_costsse.turbine_costsse import Turbine_CostsSE
from plant_costsse.nrel_csm_bos.nrel_csm_bos import bos_csm_assembly
from plant_costsse.nrel_csm_opex.nrel_csm_opex import opex_csm_assembly
from plant_costsse.ecn_offshore_opex.ecn_offshore_opex  import opex_ecn_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_assembly
from fusedwind.plant_flow.basic_aep import aep_assembly
#from landbos import LandBOS

# Current configuration assembly options for LCOE SE
# Turbine Costs
def configure_lcoe_with_turb_costs(assembly):
    """
    tcc_a inputs:
        advanced_blade = Bool
        offshore = Bool
        assemblyCostMultiplier = Float
        overheadCostMultiplier = Float
        profitMultiplier = Float
        transportMultiplier = Float
    """

    assembly.replace('tcc_a', Turbine_CostsSE())

    assembly.add('advanced_blade', Bool(True, iotype='in', desc='advanced (True) or traditional (False) blade design'))
    assembly.add('offshore', Bool(iotype='in', desc='flag for offshore site'))
    assembly.add('assemblyCostMultiplier',Float(0.0, iotype='in', desc='multiplier for assembly cost in manufacturing'))
    assembly.add('overheadCostMultiplier', Float(0.0, iotype='in', desc='multiplier for overhead'))
    assembly.add('profitMultiplier', Float(0.0, iotype='in', desc='multiplier for profit markup'))
    assembly.add('transportMultiplier', Float(0.0, iotype='in', desc='multiplier for transport costs'))

    # connections to turbine costs
    assembly.connect('rotor.mass_one_blade', 'tcc_a.blade_mass')
    assembly.connect('hub.hub_mass', 'tcc_a.hub_mass')
    assembly.connect('hub.pitch_system_mass', 'tcc_a.pitch_system_mass')
    assembly.connect('hub.spinner_mass', 'tcc_a.spinner_mass')
    assembly.connect('nacelle.low_speed_shaft_mass', 'tcc_a.low_speed_shaft_mass')
    assembly.connect('nacelle.main_bearing_mass', 'tcc_a.main_bearing_mass')
    assembly.connect('nacelle.second_bearing_mass', 'tcc_a.second_bearing_mass')
    assembly.connect('nacelle.gearbox_mass', 'tcc_a.gearbox_mass')
    assembly.connect('nacelle.high_speed_side_mass', 'tcc_a.high_speed_side_mass')
    assembly.connect('nacelle.generator_mass', 'tcc_a.generator_mass')
    assembly.connect('nacelle.bedplate_mass', 'tcc_a.bedplate_mass')
    assembly.connect('nacelle.yaw_system_mass', 'tcc_a.yaw_system_mass')
    assembly.connect('jacket.Twrouts.mass', 'tcc_a.tower_mass') # jacket input
    assembly.connect('rotor.control.ratedPower', 'tcc_a.machine_rating')
    assembly.connect('rotor.nBlades', 'tcc_a.blade_number')
    assembly.connect('nacelle.crane', 'tcc_a.crane')
    assembly.connect('year', 'tcc_a.year')
    assembly.connect('month', 'tcc_a.month')
    assembly.connect('nacelle.drivetrain_design', 'tcc_a.drivetrain_design')
    assembly.connect('advanced_blade','tcc_a.advanced_blade')
    assembly.connect('offshore','tcc_a.offshore')
    assembly.connect('assemblyCostMultiplier','tcc_a.assemblyCostMultiplier')
    assembly.connect('overheadCostMultiplier','tcc_a.overheadCostMultiplier')
    assembly.connect('profitMultiplier','tcc_a.profitMultiplier')
    assembly.connect('transportMultiplier','tcc_a.transportMultiplier')

# Balance of Station Costs
def configure_lcoe_with_csm_bos(assembly):
    """
    bos inputs:
    		bos_multiplier = Float
    """

    assembly.replace('bos_a', bos_csm_assembly())

    assembly.add('bos_multiplier', Float(1.0, iotype='in'))

    # connections to bos
    assembly.connect('machine_rating', 'bos_a.machine_rating')
    assembly.connect('rotor.diameter', 'bos_a.rotor_diameter')
    assembly.connect('rotor.hubHt', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('rotor.mass_all_blades + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

    assembly.connect('sea_depth', 'bos_a.sea_depth')
    assembly.connect('year', 'bos_a.year')
    assembly.connect('month', 'bos_a.month')
    assembly.connect('bos_multiplier','bos_a.multiplier')

def configure_lcoe_with_landbos(assembly):
    """
    if with_landbos additional inputs:
        voltage
        distInter
        terrain
        layout
        soil
    """

    assembly.replace('bos_a', LandBOS())

    assembly.add('voltage', Float(iotype='in', units='kV', desc='interconnect voltage'))
    assembly.add('distInter', Float(iotype='in', units='mi', desc='distance to interconnect'))
    assembly.add('terrain', Enum('FLAT_TO_ROLLING', ('FLAT_TO_ROLLING', 'RIDGE_TOP', 'MOUNTAINOUS'),
        iotype='in', desc='terrain options'))
    assembly.add('layout', Enum('SIMPLE', ('SIMPLE', 'COMPLEX'), iotype='in',
        desc='layout options'))
    assembly.add('soil', Enum('STANDARD', ('STANDARD', 'BOUYANT'), iotype='in',
        desc='soil options'))

    # connections to bos
    assembly.connect('machine_rating', 'bos_a.machine_rating')
    assembly.connect('rotor.diameter', 'bos_a.rotor_diameter')
    assembly.connect('rotor.hubHt', 'bos_a.hub_height')
    assembly.connect('turbine_number', 'bos_a.turbine_number')
    assembly.connect('rotor.mass_all_blades + hub.hub_system_mass + nacelle.nacelle_mass', 'bos_a.RNA_mass')

    assembly.connect('voltage', 'bos_a.voltage')
    assembly.connect('distInter', 'bos_a.distInter')
    assembly.connect('terrain', 'bos_a.terrain')
    assembly.connect('layout', 'bos_a.layout')
    assembly.connect('soil', 'bos_a.soil')

# Operational Expenditures
def configure_lcoe_with_csm_opex(assembly):
    """
    opex inputs:
       availability = Float()
    """

    assembly.replace('opex_a', opex_csm_assembly())

    # connections to opex
    assembly.connect('machine_rating', 'opex_a.machine_rating')
    assembly.connect('sea_depth', 'opex_a.sea_depth')
    assembly.connect('year', 'opex_a.year')
    assembly.connect('month', 'opex_a.month')
    assembly.connect('turbine_number', 'opex_a.turbine_number')
    assembly.connect('aep_a.net_aep', 'opex_a.net_aep')


def configure_lcoe_with_ecn_opex(assembly,ecn_file):

    assembly.replace('opex_a', opex_ecn_assembly(ecn_file))

    assembly.connect('machine_rating', 'opex_a.machine_rating')
    assembly.connect('turbine_number', 'opex_a.turbine_number')
    assembly.connect('tcc_a.turbine_cost','opex_a.turbine_cost')
    assembly.connect('project_lifetime','opex_a.project_lifetime')

# Energy Production
def configure_lcoe_with_basic_aep(assembly):
    """
    aep inputs:
        array_losses = Float
        other_losses = Float
        availability = Float
    """

    assembly.replace('aep_a', aep_assembly())

    assembly.add('array_losses',Float(0.059, iotype='in', desc='energy losses due to turbine interactions - across entire plant'))
    assembly.add('other_losses',Float(0.0, iotype='in', desc='energy losses due to blade soiling, electrical, etc'))

    # connections to aep
    assembly.connect('rotor.AEP', 'aep_a.AEP_one_turbine')
    assembly.connect('turbine_number', 'aep_a.turbine_number')
    assembly.connect('machine_rating','aep_a.machine_rating')
    assembly.connect('array_losses','aep_a.array_losses')
    assembly.connect('other_losses','aep_a.other_losses')


# Finance
def configure_lcoe_with_csm_fin(assembly):
    """
    fin inputs:
        fixed_charge_rate = Float
        construction_finance_rate = Float
        tax_rate = Float
        discount_rate = Float
        construction_time = Float
    """

    assembly.replace('fin_a', fin_csm_assembly())

    assembly.add('fixed_charge_rate', Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation'))
    assembly.add('construction_finance_rate', Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs'))
    assembly.add('tax_rate', Float(0.4, iotype = 'in', desc = 'tax rate applied to operations'))
    assembly.add('discount_rate', Float(0.07, iotype = 'in', desc = 'applicable project discount rate'))
    assembly.add('construction_time', Float(1.0, iotype = 'in', desc = 'number of years to complete project construction'))

    # connections to fin
    assembly.connect('sea_depth', 'fin_a.sea_depth')
    assembly.connect('project_lifetime','fin_a.project_lifetime')
    assembly.connect('fixed_charge_rate','fin_a.fixed_charge_rate')
    assembly.connect('construction_finance_rate','fin_a.construction_finance_rate')
    assembly.connect('tax_rate','fin_a.tax_rate')
    assembly.connect('discount_rate','fin_a.discount_rate')
    assembly.connect('construction_time','fin_a.construction_time')


# =============================================================================
# Overall LCOE Assembly
@implement_base(ExtendedFinancialAnalysis)
class lcoe_se_assembly(Assembly):

    # Base I/O
    # Inputs
    turbine_number = Int(iotype = 'in', desc = 'number of turbines at plant')

    #Outputs
    turbine_cost = Float(iotype='out', desc = 'A Wind Turbine Capital _cost')
    bos_costs = Float(iotype='out', desc='A Wind Plant Balance of Station _cost Model')
    avg_annual_opex = Float(iotype='out', desc='A Wind Plant Operations Expenditures Model')
    net_aep = Float(iotype='out', desc='A Wind Plant Annual Energy Production Model', units='kW*h')
    coe = Float(iotype='out', desc='Levelized cost of energy for the wind plant')
    opex_breakdown = VarTree(OPEXVarTree(),iotype='out')
    bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')

    # Configuration options
    with_new_nacelle = Bool(False, iotype='in', desc='configure with DriveWPACT if false, else configure with DriveSE')
    with_landbose = Bool(False, iotype='in', desc='configure with CSM BOS if false, else configure with new LandBOS model')
    flexible_blade = Bool(False, iotype='in', desc='configure rotor with flexible blade if True')
    with_3pt_drive = Bool(False, iotype='in', desc='only used if configuring DriveSE - selects 3 pt or 4 pt design option') # TODO: change nacelle selection to enumerated rather than nested boolean
    with_ecn_opex = Bool(False, iotype='in', desc='configure with CSM OPEX if flase, else configure with ECN OPEX model')
    ecn_file = Str(iotype='in', desc='location of ecn excel file if used')

    # Other I/O needed at lcoe system level
    sea_depth = Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project')
    year = Int(2009, iotype='in', desc='year of project start')
    month = Int(12, iotype='in', desc='month of project start')
    project_lifetime = Float(20.0, iotype='in', desc = 'project lifetime for wind plant')

    def __init__(self, with_new_nacelle=False, with_landbos=False, flexible_blade=False, with_3pt_drive=False, with_ecn_opex=False, ecn_file=None):
        
        self.with_new_nacelle = with_new_nacelle
        self.with_landbos = with_landbos
        self.flexible_blade = flexible_blade
        self.with_3pt_drive = with_3pt_drive
        self.with_ecn_opex = with_ecn_opex
        if ecn_file == None:
            self.ecn_file=''
        else:
            self.ecn_file = ecn_file
        
        super(lcoe_se_assembly,self).__init__()

    def configure(self):
		    """
		    tcc_a inputs:
		        advanced_blade = Bool
		        offshore = Bool
		        assemblyCostMultiplier = Float
		        overheadCostMultiplier = Float
		        profitMultiplier = Float
		        transportMultiplier = Float
		    aep inputs:
		        array_losses = Float
		        other_losses = Float
		    fin inputs:
		        fixed_charge_rate = Float
		        construction_finance_rate = Float
		        tax_rate = Float
		        discount_rate = Float
		        construction_time = Float
		    bos inputs:
		        bos_multiplier = Float
		    inputs:
		        sea_depth
		        year
		        month
		        project lifetime
		    if csm opex additional inputs:
		        availability = Float()
		    if openwind opex additional inputs:
		        power_curve 
		        rpm 
		        ct 
		    if with_landbos additional inputs:
		        voltage
		        distInter
		        terrain
		        layout
		        soil
		    """
		
		    # configure base asesmbly
		    configure_extended_financial_analysis(self)
		
		    # add TurbineSE assembly
		    configure_turbine_with_jacket(self, self.with_new_nacelle, self.flexible_blade, self.with_3pt_drive)
		
		    # replace TCC with turbine_costs
		    configure_lcoe_with_turb_costs(self)
		
		    # replace BOS with either CSM or landbos
		    if self.with_landbos:
		        configure_lcoe_with_landbos(self)
		    else:
		        configure_lcoe_with_csm_bos(self)
		    
		    # replace OPEX with CSM or ECN opex and add AEP
		    if self.with_ecn_opex:  
		        configure_lcoe_with_basic_aep(self)
		        configure_lcoe_with_ecn_opex(self,ecn_file)     
		        self.connect('opex_a.availability','aep_a.availability') # connecting here due to aep / opex reversal depending on model 
		    else:
		        configure_lcoe_with_basic_aep(self)
		        configure_lcoe_with_csm_opex(self)
		        self.add('availability',Float(0.94, iotype='in', desc='average annual availbility of wind turbines at plant'))
		        self.connect('availability','aep_a.availability') # connecting here due to aep / opex reversal depending on model
		
		    # replace Finance with CSM Finance
		    configure_lcoe_with_csm_fin(self)


def example(wind_class='I',sea_depth=0.0,with_new_nacelle=False,with_landbos=False,flexible_blade=False,with_3pt_drive=False, with_ecn_opex=False, ecn_file=None,with_openwind=False,ow_file=None,ow_wkbook=None):
    """
    Inputs:
        wind_class : str ('I', 'III', 'Offshore' - selected wind class for project)
        sea_depth : float (sea depth if an offshore wind plant)
    """

    # === Create LCOE SE assembly ========
    lcoe_se = lcoe_se_assembly(with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file)

    # === Set assembly variables and objects ===
    lcoe_se.sea_depth = sea_depth # 0.0 for land-based turbine
    lcoe_se.turbine_number = 100
    lcoe_se.year = 2009
    lcoe_se.month = 12

    rotor = lcoe_se.rotor
    nacelle = lcoe_se.nacelle
    jacket = lcoe_se.jacket
    tcc_a = lcoe_se.tcc_a
    # bos_a = lcoe_se.bos_a
    # opex_a = lcoe_se.opex_a
    aep_a = lcoe_se.aep_a
    fin_a = lcoe_se.fin_a

    # Turbine ===========
    from wisdem.reference_turbines.nrel5mw.nrel5mw_jacket import configure_nrel5mw_turbine_with_jacket
    configure_nrel5mw_turbine_with_jacket(lcoe_se,wind_class,lcoe_se.sea_depth)

    # TODO: these should be specified at the turbine level and connected to other system inputs
    lcoe_se.tower_dt = 3.87  # (Array, m): diameters along tower # float for jacket
    lcoe_se.generator_speed = 1173.7  # (Float, rpm)  # generator speed
    # extra variable constant for now
    #lcoe_se.nacelle.bedplate.rotor_bending_moment_y = -2.3250E+06 # shouldnt be needed anymore

    # tcc ====
    lcoe_se.advanced_blade = True
    lcoe_se.offshore = False
    lcoe_se.assemblyCostMultiplier = 0.30
    lcoe_se.profitMultiplier = 0.20
    lcoe_se.overheadCostMultiplier = 0.0
    lcoe_se.transportMultiplier = 0.0

    # for new landBOS
    ''' # === new landBOS ===
    lcoe_se.voltage = 137
    lcoe_se.distInter = 5
    lcoe_se.terrain = 'FLAT_TO_ROLLING'
    lcoe_se.layout = 'SIMPLE'
    lcoe_se.soil = 'STANDARD' '''

    # aep ====
    if not with_openwind:
        lcoe_se.array_losses = 0.059
    lcoe_se.other_losses = 0.0
    if not with_ecn_opex:
        lcoe_se.availability = 0.94

    # fin ===
    lcoe_se.fixed_charge_rate = 0.095
    lcoe_se.construction_finance_rate = 0.0
    lcoe_se.tax_rate = 0.4
    lcoe_se.discount_rate = 0.07
    lcoe_se.construction_time = 1.0
    lcoe_se.project_lifetime = 20.0

    # Set plant level inputs ===
    shearExp = 0.2 #TODO : should be an input to lcoe
    rotor.cdf_reference_height_wind_speed = 90.0
    if not with_openwind:
        lcoe_se.array_losses = 0.1
    lcoe_se.other_losses = 0.0
    if not with_ecn_opex:
        lcoe_se.availability = 0.98
    rotor.turbulence_class = 'B'
    lcoe_se.multiplier = 2.23

    if wind_class == 'Offshore':
        # rotor.cdf_reference_mean_wind_speed = 8.4 # TODO - aep from its own module
        # rotor.cdf_reference_height_wind_speed = 50.0
        # rotor.weibull_shape = 2.1
        shearExp = 0.14 # TODO : should be an input to lcoe
        lcoe_se.array_losses = 0.15
        if not with_ecn_opex:
            lcoe_se.availability = 0.96
        lcoe_se.offshore = True
        lcoe_se.multiplier = 2.33
        lcoe_se.fixed_charge_rate = 0.118

    rotor.shearExp = shearExp
    #tower.wind1.shearExp = shearExp # not needed for jacket
    #tower.wind2.shearExp = shearExp

    # ====
    #--- Set Jacket Input Parameters ---#
    Jcktins=JcktGeoInputs()
    Jcktins.nlegs =4
    Jcktins.nbays =5
    Jcktins.batter=12.
    Jcktins.dck_botz =16.
    Jcktins.dck_width=2*6.
    Jcktins.weld2D   =0.5
    Jcktins.VPFlag = True    #vertical pile T/F;  to enable piles in frame3DD set pileinputs.ndiv>0
    Jcktins.clamped= False    #whether or not the bottom of the structure is rigidly connected. Use False when equivalent spring constants are being used.
    Jcktins.AFflag = False  #whether or not to use apparent fixity piles
    Jcktins.PreBuildTPLvl = 5  #if >0, the TP is prebuilt according to rules per PreBuildTP

    #Soil inputs
    Soilinputs=SoilGeoInputs()
    Soilinputs.zbots   =-np.array([3.,5.,7.,15.,30.,50.])
    Soilinputs.gammas  =np.array([10000.,10000.,10000.,10000.,10000.,10000.])
    Soilinputs.cus     =np.array([60000.,60000.,60000.,60000.,60000.,60000.])
    Soilinputs.phis    =np.array([26.,26.,26.,26.,26.,26])#np.array([36.,33.,26.,37.,35.,37.5])#np.array([36.,33.,26.,37.,35.,37.5])
    Soilinputs.delta   =25.
    Soilinputs.sndflg   =True
    Soilinputs.PenderSwtch   =False #True
    Soilinputs.SoilSF   =1.

    Soilinputs2=copy.copy(Soilinputs) #Parked case. We assume same stiffness although this may not be the case under a different load

    #Water and wind inputs
    Waterinputs=WaterInputs()
    Waterinputs.wdepth   =30.
    Waterinputs.wlevel   =30. #Distance from bottom of structure to surface  THIS, I believe is no longer needed as piles may be negative in z, to check and remove in case
    Waterinputs.T=12.  #Wave Period
    Waterinputs.HW=10. #Wave Height
    Waterinputs.Cd=3.  #Drag Coefficient, enhanced to account for marine growth and other members not calculated
    Waterinputs.Cm=8.#2.  #ADded mass Coefficient

    Waterinputs2=copy.copy(Waterinputs)  #PARKED CONDITIONS - still max wave here
    Waterinputs.T=8.  #Wave Period
    Waterinputs.HW=4. #Wave Height

    Windinputs=WindInputs()
    Windinputs.Cdj=4.  #Drag Coefficient for jacket members, enhanced to account for TP drag not calculated otherwise
    Windinputs.Cdt=2  #Drag Coefficient for tower, enhanced to account for TP drag not calculated otherwise
    Windinputs.HH=100. #CHECK HOW THIS COMPLIES....
    Windinputs.U50HH=30. #assumed gust speed

    ## if turbine_jacket
    ##Windinputs.HH=90. #CHECK HOW THIS COMPLIES....
    ##Windinputs.U50HH=11.7373200354 # using rated loads
    ##Windinputs.rho = 1.225
    ##Windinputs.mu = 1.81206e-05

    Windinputs2=copy.copy(Windinputs)
    Windinputs2.U50HH=70. #assumed gust speed

    #Pile data
    Pilematin=MatInputs()
    Pilematin.matname=np.array(['steel'])
    Pilematin.E=np.array([ 25.e9])
    Dpile=2.5#0.75 # 2.0
    tpile=0.01
    Lp=20. #45

    Pileinputs=PileGeoInputs()
    Pileinputs.Pilematins=Pilematin
    Pileinputs.ndiv=0 #3
    Pileinputs.Dpile=Dpile
    Pileinputs.tpile=tpile
    Pileinputs.Lp=Lp #[m] Embedment length

    #Legs data
    legmatin=MatInputs()
    legmatin.matname=(['heavysteel','heavysteel','heavysteel','heavysteel'])
    #legmatin.E=np.array([2.0e11])
    Dleg=np.array([1.5,1.5,1.5,1.5,1.5,1.5])
    tleg=1.5*np.array([0.0254]).repeat(Dleg.size)
    leginputs=LegGeoInputs()
    leginputs.legZbot   = 1.0
    leginputs.ndiv=1
    leginputs.legmatins=legmatin
    leginputs.Dleg0=Dleg[0]
    leginputs.tleg0=tleg[0]

    legbot_stmphin =1.5  #Distance from bottom of leg to second joint along z; must be>0

    #Xbrc data
    Xbrcmatin=MatInputs()
    Xbrcmatin.matname=np.array(['heavysteel']).repeat(Jcktins.nbays)
    #Xbrcmatin.E=np.array([ 2.2e11, 2.0e11,2.0e11,2.0e11,2.0e11])
    Dbrc=np.array([1.,1.,1.0,1.0,1.0])
    tbrc=np.array([1.,1.,1.0,1.0,1.0])*0.0254

    Xbrcinputs=XBrcGeoInputs()
    Xbrcinputs.Dbrc0=Dbrc[0]
    Xbrcinputs.tbrc0=tbrc[0]
    Xbrcinputs.ndiv=2#2
    Xbrcinputs.Xbrcmatins=Xbrcmatin
    Xbrcinputs.precalc=False #True   #This can be set to true if we want Xbraces to be precalculated in D and t, in which case the above set Dbrc and tbrc would be overwritten

    #Mbrc data
    Mbrcmatin=MatInputs()
    Mbrcmatin.matname=np.array(['heavysteel'])
    #Mbrcmatin.E=np.array([ 2.5e11])
    Dbrc_mud=1.5

    Mbrcinputs=MudBrcGeoInputs()
    Mbrcinputs.Dbrc_mud=Dbrc_mud
    Mbrcinputs.ndiv=2
    Mbrcinputs.Mbrcmatins=Mbrcmatin
    Mbrcinputs.precalc=False #True   #This can be set to true if we want Mudbrace to be precalculated in D and t, in which case the above set Dbrc_mud and tbrc_mud would be overwritten

    #Hbrc data
    Hbrcmatin=MatInputs()
    Hbrcmatin.matname=np.array(['heavysteel'])
    Hbrcmatin.E=np.array([ 2.5e11])
    Dbrc_hbrc=1.1

    Hbrcinputs=HBrcGeoInputs()
    Hbrcinputs.Dbrch=Dbrc_hbrc
    Hbrcinputs.ndiv=0#2
    Hbrcinputs.Hbrcmatins=Hbrcmatin
    Hbrcinputs.precalc=True   #This can be set to true if we want Hbrace to be set=Xbrace top D and t, in which case the above set Dbrch and tbrch would be overwritten

    #TP data
    TPlumpinputs=TPlumpMass()
    TPlumpinputs.mass=200.e3 #[kg]

    TPstmpsmatin=MatInputs()
    TPbrcmatin=MatInputs()
    TPstemmatin=MatInputs()
    TPbrcmatin.matname=np.array(['heavysteel'])
    #TPbrcmatin.E=np.array([ 2.5e11])
    TPstemmatin.matname=np.array(['heavysteel']).repeat(2)
    #TPstemmatin.E=np.array([ 2.1e11]).repeat(2)

    TPinputs=TPGeoInputs()
    TPinputs.TPbrcmatins=TPbrcmatin
    TPinputs.TPstemmatins=TPstemmatin
    TPinputs.TPstmpmatins=TPstmpsmatin
    TPinputs.Dstrut=leginputs.Dleg[-1]
    TPinputs.tstrut=leginputs.tleg[-1]
    TPinputs.Dgir=Dbrc_hbrc
    TPinputs.tgir=0.0254
    TPinputs.Dbrc=1.1
    TPinputs.Dbrc=TPinputs.Dgir
    TPinputs.tbrc=TPinputs.tgir

    TPinputs.hstump=1.0#1.0
    TPinputs.Dstump=1.25#1.0
    TPinputs.stumpndiv=1#2
    TPinputs.brcndiv=1#2
    TPinputs.girndiv=1#2
    TPinputs.strutndiv=1#2
    TPinputs.stemndiv=1#2
    TPinputs.nstems=3
    TPinputs.Dstem=np.array([6.]).repeat(TPinputs.nstems)
    TPinputs.tstem=np.array([0.1,0.11,0.11])
    TPinputs.hstem=np.array([6./TPinputs.nstems]).repeat(TPinputs.nstems)

    #Tower data
    Twrmatin=MatInputs()
    Twrmatin.matname=np.array(['heavysteel'])
    #Twrmatin.E=np.array([ 2.77e11])
    Twrinputs=TwrGeoInputs()
    Twrinputs.Twrmatins=Twrmatin
    #Twrinputs.Htwr=70.  #Trumped by HH
    Twrinputs.Htwr2frac=0.2   #fraction of tower height with constant x-section
    Twrinputs.ndiv=np.array([6,12])  #ndiv for uniform and tapered section
    Twrinputs.DeltaZmax= 6. #[m], maximum FE element length allowed in the tower members (i.e. the uniform and the tapered members)
    Twrinputs.Db=5.6
    Twrinputs.DTRb=130.
    Twrinputs.DTRt=150.
    Twrinputs.Dt=0.55*Twrinputs.Db
    ## if turbine_jacket
    ##Twrinputs.Dt = 3.87

    TwrRigidTop=True #False       #False=Account for RNA via math rather than a physical rigidmember

    #RNA data
    RNAins=RNAprops()
    RNAins.mass=3*350.e3
    RNAins.I[0]=86.579E+6
    RNAins.I[1]=53.530E+6
    RNAins.I[2]=58.112E+6
    RNAins.CMoff[2]=2.34
    RNAins.yawangle=45.  #angle with respect to global X, CCW looking from above, wind from left
    RNAins.rna_weightM=True
    ## if turbine_jacket
    ##RNAins.mass=285598.806453
    ##RNAins.I = np.array([1.14930678e8, 2.20354030e7, 1.87597425e7, 0.0, 5.03710467e5, 0.0])
    ##RNAins.CMoff = np.array([-1.13197635, 0.0, 0.50875268])
    ##RNAins.yawangle=0.0  #angle with respect to global X, CCW looking from above, wind from left
    #RNAins.rna_weightM=True

    RNAins2=copy.copy(RNAins)  #PARKED CASE, for now assume the same

    #RNA loads              Fx-z,         Mxx-zz
    RNA_F=np.array([1000.e3,0.,0.,0.,0.,0.])    #operational
    RNA_F2=np.array([500.e3,0.,0.,0.,0.,0.])    #Parked
    ## if turbine_jacket
    ##RNA_F=np.array([1284744.19620519,0.,-2914124.84400512,3963732.76208099,-2275104.79420872,-346781.68192839])

    #Frame3DD parameters
    FrameAuxIns=Frame3DDaux()
    FrameAuxIns.sh_fg=1               #shear flag-->Timoshenko
    FrameAuxIns.deltaz=5.
    FrameAuxIns.geo_fg=0
    FrameAuxIns.nModes = 6             # number of desired dynamic modes of vibration
    FrameAuxIns.Mmethod = 1            # 1: subspace Jacobi     2: Stodola
    FrameAuxIns.lump = 0               # 0: consistent mass ... 1: lumped mass matrix
    FrameAuxIns.tol = 1e-9             # mode shape tolerance
    FrameAuxIns.shift = 0.0            # shift value ... for unrestrained structures
    FrameAuxIns.gvector=np.array([0.,0.,-9.8065])    #GRAVITY
    ## if turbine_jacket
    ##FrameAuxIns.gvector=np.array([0.,0.,-9.81])    #GRAVITY

    #Decide whether or not to consider DLC 6.1 as well
    twodlcs=False

    #-----Launch the assembly-----#

    #turbine.jacket=JacketSE(Jcktins.clamped,Jcktins.AFflag,twodlcs=twodlcs)
    #turbine.jacket=set_as_top(JacketSE(Jcktins.clamped,Jcktins.AFflag,twodlcs=twodlcs)) ##(Jcktins.PreBuildTPLvl>0),

    #Pass all inputs to assembly
    lcoe_se.jacket.JcktGeoIn=Jcktins

    lcoe_se.jacket.Soilinputs=Soilinputs
    lcoe_se.jacket.Soilinputs2=Soilinputs2   #Parked conditions

    lcoe_se.jacket.Waterinputs=Waterinputs
    lcoe_se.jacket.Windinputs=Windinputs
    lcoe_se.jacket.RNA_F=RNA_F
    lcoe_se.jacket.Waterinputs2=Waterinputs2 #Parked conditions
    lcoe_se.jacket.Windinputs2=Windinputs2   #Parked conditions
    lcoe_se.jacket.RNA_F2=RNA_F2            #Parked conditions

    lcoe_se.jacket.Pileinputs=Pileinputs
    lcoe_se.jacket.leginputs=leginputs
    #turbine.jacket.legbot_stmphin =legbot_stmphin
    lcoe_se.jacket.Xbrcinputs=Xbrcinputs
    lcoe_se.jacket.Mbrcinputs=Mbrcinputs
    lcoe_se.jacket.Hbrcinputs=Hbrcinputs
    lcoe_se.jacket.TPlumpinputs=TPlumpinputs
    lcoe_se.jacket.TPinputs=TPinputs
    lcoe_se.jacket.RNAinputs=RNAins
    lcoe_se.jacket.RNAinputs2=RNAins2
    lcoe_se.jacket.Twrinputs=Twrinputs
    lcoe_se.jacket.TwrRigidTop=TwrRigidTop
    lcoe_se.jacket.FrameAuxIns=FrameAuxIns




    # === Run default assembly and print results
    lcoe_se.run()
    # ====

    # === Print ===

    print "Key Turbine Outputs for NREL 5 MW Reference Turbine"
    print 'mass rotor blades:{0:.2f} (kg) '.format(lcoe_se.rotor.mass_all_blades)
    print 'mass hub system: {0:.2f} (kg) '.format(lcoe_se.hub.hub_system_mass)
    print 'mass nacelle: {0:.2f} (kg) '.format(lcoe_se.nacelle.nacelle_mass)
    print 'mass tower: {0:.2f} (kg) '.format(lcoe_se.jacket.Tower.Twrouts.mass)
    print 'maximum tip deflection: {0:.2f} (m) '.format(lcoe_se.maxdeflection.max_tip_deflection)
    print 'ground clearance: {0:.2f} (m) '.format(lcoe_se.maxdeflection.ground_clearance)
    print
    print "Key Plant Outputs for wind plant with NREL 5 MW Turbine"
    #print "LCOE: ${0:.4f} USD/kWh".format(lcoe_se.lcoe) # not in base output set (add to assembly output if desired)
    print "COE: ${0:.4f} USD/kWh".format(lcoe_se.coe)
    print
    print "AEP per turbine: {0:.1f} kWh/turbine".format(lcoe_se.net_aep / lcoe_se.turbine_number)
    print "Turbine Cost: ${0:.2f} USD".format(lcoe_se.turbine_cost)
    print "BOS costs per turbine: ${0:.2f} USD/turbine".format(lcoe_se.bos_costs / lcoe_se.turbine_number)
    print "OPEX per turbine: ${0:.2f} USD/turbine".format(lcoe_se.avg_annual_opex / lcoe_se.turbine_number)    

    # ====

if __name__ == '__main__':

    # NREL 5 MW in land-based wind plant with high winds (as class I)
    wind_class = 'I'
    sea_depth = 0.0
    with_new_nacelle = False # MB1 error when true
    with_landbos = False
    flexible_blade = False
    with_3pt_drive = False
    with_ecn_opex = False
    ecn_file = ''
    example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    #with_3pt_drive = True
    #example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) )

    #with_new_nacelle = False
    #example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    #with_landbos = True
    #example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    #flexible_blade = True
    #example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    # NREL 5 MW in land-based wind plant with low winds (as class III)
    #wind_class = 'III'
    #with_new_nacelle = True
    #example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 

    # NREL 5 MW in offshore plant with high winds and 20 m sea depth (as class I)
    #wind_class = 'Offshore'
    #sea_depth = 20.0
    #example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 
    
    # NREL 5 MW in offshore plant with high winds, 20 m sea depth and ECN opex model
    #wind_class = 'Offshore'
    #sea_depth = 20.0
    #with_ecn_opex = True
    #ecn_file = 'C:/Models/ECN Model/ECN O&M Model.xls'
    #example(wind_class,sea_depth,with_new_nacelle,with_landbos,flexible_blade,with_3pt_drive,with_ecn_opex,ecn_file) 
   
