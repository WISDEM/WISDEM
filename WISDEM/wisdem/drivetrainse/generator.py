"""generator.py
Created by Latha Sethuraman, Katherine Dykes. 
Copyright (c) NREL. All rights reserved.

Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

import openmdao.api as om
from om import Group, Problem, ExplicitComponent, ExecComp, IndepVarComp
#from openmdao.api import ScipyOptimizeDriver, pyOptSparseDriver
#from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

import numpy as np
#from scig import SCIG
#from dfig import DFIG
#from eesg import EESG
#from pmsg_arms import PMSG_Arms
#from pmsg_disc import PMSG_Disc

from scigOM import SCIG_OM
from dfigOM import DFIG_OM
from eesgOM import EESG_OM
from pmsg_armsOM import PMSG_Arms_OM
from pmsg_discOM import PMSG_Disc_OM


class Generator_Cost(ExplicitComponent):
    """ Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded"""
    def setup(self):
        
        # Specific cost of material by type
        self.add_input('C_Cu',  val=0.0, units='USD/kg', desc='Specific cost of copper')
        self.add_input('C_Fe',  val=0.0, units='USD/kg', desc='Specific cost of magnetic steel/iron')
        self.add_input('C_Fes', val=0.0, units='USD/kg', desc='Specific cost of structural steel')
        self.add_input('C_PM',  val=0.0, units='USD/kg', desc='Specific cost of Magnet')  
        
        # Mass of each material type
        self.add_input('Copper',          val=0.0, units='kg', desc='Copper mass')
        self.add_input('Iron',            val=0.0, units='kg', desc='Iron mass')
        self.add_input('mass_PM' ,        val=0.0, units='kg', desc='Magnet mass')
        self.add_input('Structural_mass', val=0.0, units='kg', desc='Structural mass')
        
        # Outputs
        self.add_output('Costs', val=0.0, units='USD', desc='Total cost')

        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs):
        Copper          = inputs['Copper']
        Iron            = inputs['Iron']
        mass_PM         = inputs['mass_PM']
        Structural_mass = inputs['Structural_mass']
        C_Cu            = inputs['C_Cu']
        C_Fes           = inputs['C_Fes']
        C_Fe            = inputs['C_Fe']
        C_PM            = inputs['C_PM']
                
        # Material cost as a function of material mass and specific cost of material
        K_gen            = Copper*C_Cu + Iron*C_Fe + C_PM*mass_PM #%M_pm*K_pm; # 
        Cost_str         = C_Fes*Structural_mass
        outputs['Costs'] = K_gen + Cost_str
        

        

class Generator(Group):

    def initialize(self):
        genTypes = ['scig','dfig','eesg','pmsg_arms','pmsg_disc']
        self.options.declare('topLevelFlag', default=True)
        self.options.declare('design', values=genTypes + [m.upper() for m in genTypes])
    
    def setup(self):
        topLevelFlag = self.options['topLevelFlag']
        genType      = self.options['design']
        
        generatorIndeps = IndepVarComp()
        generatorIndeps.add_output('shaft_cm', val=np.zeros(3), units='m')
        generatorIndeps.add_output('shaft_length', val=0.0, units='m')
        
        generatorIndeps.add_output('Gearbox_efficiency', val=0.0)
        generatorIndeps.add_output('r_s',   val=0.0, units='m')
        generatorIndeps.add_output('len_s', val=0.0, units='m')
        generatorIndeps.add_output('h_s',   val=0.0, units='m')
        generatorIndeps.add_output('h_r',   val=0.0, units='m')
        generatorIndeps.add_output('h_m',   val=0.0, units='m')
        generatorIndeps.add_output('S_Nmax', val=0.0, units='T')
        generatorIndeps.add_output('I_0',   val=0.0, units='A')

        if genType.lower() in ['eesg','pmsg_arms','pmsg_disc']:
            generatorIndeps.add_output('tau_p', val=0.0, units='m')
            generatorIndeps.add_output('h_ys',  val=0.0, units='m')
            generatorIndeps.add_output('h_yr',  val=0.0, units='m')
            generatorIndeps.add_output('b_r',   val=0.0, units='m')
        elif genType.lower() in ['scig','dfig']:
            generatorIndeps.add_output('B_symax', val=0.0)
            
        generatorIndeps.add_output('I_f',  val=0.0, units='A')
        generatorIndeps.add_output('N_f',  val=0.0, units='A')
        generatorIndeps.add_output('n_s',  val=0.0)
        generatorIndeps.add_output('b_st', val=0.0, units='m')
        generatorIndeps.add_output('n_r',  val=0.0)
        generatorIndeps.add_output('d_r',  val=0.0, units='m')
        generatorIndeps.add_output('d_s',  val=0.0, units='m')
        generatorIndeps.add_output('t_d',  val=0.0, units='m')
        generatorIndeps.add_output('t_wr', val=0.0, units='m')
        generatorIndeps.add_output('t_ws', val=0.0, units='m')
        generatorIndeps.add_output('R_o',  val=0.0, units='m')
        
        generatorIndeps.add_output('rho_Fes',    val=0.0, units='kg/m**3')
        generatorIndeps.add_output('rho_Fe',     val=0.0, units='kg/m**3')
        generatorIndeps.add_output('rho_Copper', val=0.0, units='kg/m**3')
        generatorIndeps.add_output('rho_PM',     val=0.0, units='kg/m**3')
        
        generatorIndeps.add_output('C_Cu',  val=0.0, units='USD/kg')
        generatorIndeps.add_output('C_Fe',  val=0.0, units='USD/kg')
        generatorIndeps.add_output('C_Fes', val=0.0, units='USD/kg')
        generatorIndeps.add_output('C_PM',  val=0.0, units='USD/kg')
        
        self.add_subsystem('generatorIndeps', generatorIndeps, promotes=['*'])

        if topLevelFlag:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('machine_rating', 0.0, units='W')
            sharedIndeps.add_output('n_nom', 0.0, units='rpm')
            sharedIndeps.add_output('Torque', 0.0, units='N*m')
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
        
        # Add generator design component and cost
        if genType.lower() == 'scig':
            mygen = SCIG_OM
            
        elif genType.lower() == 'dfig':
            mygen = DFIG_OM
            
        elif genType.lower() == 'eesg':
            mygen = EESG_OM
            
        elif genType.lower() == 'pmsg_arms':
            mygen = PMSG_Arms_OM
            
        elif genType.lower() == 'pmsg_disc':
            mygen = PMSG_Disc_OM
            
        self.add_subsystem('generator', mygen(), promotes=['*'])
        self.add_subsystem('gen_cost', Generator_Cost(), promotes=['*'])

        # Now add constraints
        if genType.lower() in ['eesg', 'pmsg_arms', 'pmsg_disc']:
            self.add_subsystem('con_uAs', ExecComp('con_uAs =u_all_s-u_As'), promotes=['*'])
            self.add_subsystem('con_zAs', ExecComp('con_zAs =z_all_s-z_A_s'), promotes=['*'])
            self.add_subsystem('con_yAs', ExecComp('con_yAs =y_all-y_As'), promotes=['*'])
            self.add_subsystem('con_bst', ExecComp('con_bst =b_all_s-b_st',
                                                   b_st={'units':'m'}), promotes=['*'])
            self.add_subsystem('con_uAr', ExecComp('con_uAr =u_all_r-u_Ar'), promotes=['*'])
            self.add_subsystem('con_yAr', ExecComp('con_yAr =y_all-y_Ar'), promotes=['*'])
            self.add_subsystem('con_TC2', ExecComp('con_TC2 =TC2-TC1'), promotes=['*'])
            self.add_subsystem('con_TC3', ExecComp('con_TC3 =TC3-TC1'), promotes=['*'])

        if genType.lower() in ['pmsg_arms', 'pmsg_disc']:
            self.add_subsystem('con_Bsmax', ExecComp('con_Bsmax =B_g-B_smax'), promotes=['*'])
            
        if genType.lower() in ['eesg', 'pmsg_arms']:
            self.add_subsystem('con_zAr', ExecComp('con_zAr =z_all_r-z_A_r'), promotes=['*'])
            self.add_subsystem('con_br', ExecComp('con_br =b_all_r-b_r',
                                                  b_r={'units':'m'}), promotes=['*'])
            
        if genType.lower() in ['scig','dfig']:
            self.add_subsystem('TC', ExecComp('TC = TC2-TC1'), promotes=['*'])
            self.add_subsystem('K_rad_L',ExecComp('K_rad_L = K_rad-K_rad_LL'), promotes=['*'])
            self.add_subsystem('K_rad_U',ExecComp('K_rad_U = K_rad-K_rad_UL'), promotes=['*'])
            self.add_subsystem('D_ratio_L',ExecComp('D_ratio_L = D_ratio-D_ratio_LL'), promotes=['*'])
            self.add_subsystem('D_ratio_U',ExecComp('D_ratio_U = D_ratio-D_ratio_UL'), promotes=['*'])
            

def optimization_example(genType, exportFlag=False):
    genType = genType.lower()
    
    #Example optimization of a generator for costs on a 5 MW reference turbine
    opt_problem=Problem()
    opt_problem.model = Generator(design=genType, topLevelFlag=True)
    
    # add optimizer and set-up problem (using user defined input on objective function)
    '''
    opt_problem.driver = om.pyOptSparseDriver() #ScipyOptimizeDriver()
    opt_problem.driver.options['optimizer'] = 'CONMIN'
    opt_problem.driver.opt_settings['IPRINT'] = 4
    opt_problem.driver.opt_settings['ITRM'] = 3
    opt_problem.driver.opt_settings['ITMAX'] = 10
    opt_problem.driver.opt_settings['DELFUN'] = 1e-3
    opt_problem.driver.opt_settings['DABFUN'] = 1e-3
    opt_problem.driver.opt_settings['IFILE'] = 'CONMIN_'+genType.upper()+'.out'
    '''
    opt_problem.driver = om.ScipyOptimizeDriver()
    opt_problem.driver.options['optimizer'] = 'SLSQP'
    
    # Specificiency target efficiency(%)
    Eta_Target = 93.0

    eps = 1e-6
    
    # Set up design variables and bounds for a SCIG designed for a 5MW turbine
    if genType in ['dfig', 'scig']:
        # Design variables
        opt_problem.model.add_design_var('r_s',     lower=0.2,  upper=1.0)
        opt_problem.model.add_design_var('len_s',   lower=0.4,  upper=2.0)
        opt_problem.model.add_design_var('h_s',     lower=0.04, upper=0.1)
        opt_problem.model.add_design_var('h_r',     lower=0.04, upper=0.1)
        opt_problem.model.add_design_var('B_symax', lower=1.0,  upper=2.0-eps)

        # Constraints
        opt_problem.model.add_constraint('Overall_eff',        lower=Eta_Target)
        opt_problem.model.add_constraint('E_p',                lower=500.0+eps, upper=5000.0-eps)
        opt_problem.model.add_constraint('TC',                 lower=0.0+eps)
        opt_problem.model.add_constraint('B_g',                lower=0.7,       upper=1.2)
        opt_problem.model.add_constraint('B_trmax',                             upper=2.0-eps)
        opt_problem.model.add_constraint('B_tsmax',                             upper=2.0-eps)
        opt_problem.model.add_constraint('A_1',                                 upper=60000.0-eps)
        opt_problem.model.add_constraint('J_s',                                 upper=6.0)
        opt_problem.model.add_constraint('J_r',                                 upper=6.0)
        opt_problem.model.add_constraint('Slot_aspect_ratio1', lower=4.0,       upper=10.0)

    if genType == 'scig':
        opt_problem.model.add_design_var('I_0',       lower=5.0, upper=200.0)
        opt_problem.model.add_constraint('B_rymax',   upper=2.0-eps)
        opt_problem.model.add_constraint('K_rad_L',   lower=0.0)
        opt_problem.model.add_constraint('K_rad_U',   upper=0.0)
        opt_problem.model.add_constraint('D_ratio_L', lower=0.0)
        opt_problem.model.add_constraint('D_ratio_U', upper=0.0)
        
    if genType == 'dfig':
        opt_problem.model.add_design_var('I_0',          lower=5.,   upper=100.0)
        opt_problem.model.add_design_var('S_Nmax',       lower=-0.3, upper=-0.1)
        opt_problem.model.add_constraint('K_rad',        lower=0.2,  upper=1.5)
        opt_problem.model.add_constraint('D_ratio',      lower=1.37, upper=1.4)
        opt_problem.model.add_constraint('Current_ratio',lower=0.1,  upper=0.3)

    if genType == 'eesg':
        # Design variables
        opt_problem.model.add_design_var('r_s',    lower=0.5,   upper=9.0)
        opt_problem.model.add_design_var('len_s',  lower=0.5,   upper=2.5)
        opt_problem.model.add_design_var('h_s',    lower=0.06,  upper=0.15)
        opt_problem.model.add_design_var('tau_p',  lower=0.04,  upper=0.2)
        opt_problem.model.add_design_var('N_f',    lower=10,    upper=300)
        opt_problem.model.add_design_var('I_f',    lower=1,     upper=500)
        opt_problem.model.add_design_var('n_r',    lower=5.0,   upper=15.0)
        opt_problem.model.add_design_var('h_yr',   lower=0.01,  upper=0.25)
        opt_problem.model.add_design_var('h_ys',   lower=0.01,  upper=0.25)
        opt_problem.model.add_design_var('b_r',    lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('d_r',    lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('t_wr',   lower=0.001, upper=0.2)
        opt_problem.model.add_design_var('n_s',    lower=5.0,   upper=15.0)
        opt_problem.model.add_design_var('b_st',   lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('d_s',    lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('t_ws',   lower=0.001, upper=0.2)

        # Constraints
        opt_problem.model.add_constraint('B_gfm',       lower=0.617031, upper=1.057768)
        opt_problem.model.add_constraint('B_pc',        upper=2.0)
        opt_problem.model.add_constraint('E_s',         lower=500.0, upper=5000.0)
        opt_problem.model.add_constraint('J_f',         upper=6.0)
        opt_problem.model.add_constraint('n_brushes',   upper=6)
        opt_problem.model.add_constraint('Power_ratio', upper=2-eps)
        
    if genType in ['pmsg_arms','pmsg_disc']:
        # Design variables
        opt_problem.model.add_design_var('r_s',   lower=0.5,   upper=9.0)
        opt_problem.model.add_design_var('len_s', lower=0.5,   upper=2.5)
        opt_problem.model.add_design_var('h_s',   lower=0.04,  upper=0.1)
        opt_problem.model.add_design_var('tau_p', lower=0.04,  upper=0.1)
        opt_problem.model.add_design_var('h_m',   lower=0.005, upper=0.1)
        opt_problem.model.add_design_var('n_r',   lower=5.0,   upper=15.0)
        opt_problem.model.add_design_var('h_yr',  lower=0.045, upper=0.25)
        opt_problem.model.add_design_var('h_ys',  lower=0.045, upper=0.25)
        opt_problem.model.add_design_var('n_s',   lower=5.0,   upper=15.0)
        opt_problem.model.add_design_var('b_st',  lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('d_s',   lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('t_ws',  lower=0.001, upper=0.2)
    
        opt_problem.model.add_constraint('con_Bsmax', lower=0.0+eps)
        opt_problem.model.add_constraint('E_p', lower=500.0, upper=5000.0)

    if genType == 'pmsg_arms':
        opt_problem.model.add_design_var('b_r',  lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('d_r',  lower=0.1,   upper=1.5)
        opt_problem.model.add_design_var('t_wr', lower=0.001, upper=0.2)
        
    if genType == 'pmsg_disc':
        opt_problem.model.add_design_var('t_d', lower=0.1, upper=0.25)
        
    if genType in ['eesg', 'pmsg_arms', 'pmsg_disc']:
        opt_problem.model.add_constraint('B_symax', upper=2.0-eps)
        opt_problem.model.add_constraint('B_rymax', upper=2.0-eps)
        opt_problem.model.add_constraint('B_tmax',  upper=2.0-eps)
        opt_problem.model.add_constraint('B_g',     lower=0.7, upper=1.2)
        opt_problem.model.add_constraint('con_uAs', lower=0.0+eps)
        opt_problem.model.add_constraint('con_zAs', lower=0.0+eps)
        opt_problem.model.add_constraint('con_yAs', lower=0.0+eps)
        opt_problem.model.add_constraint('con_uAr', lower=0.0+eps)
        opt_problem.model.add_constraint('con_yAr', lower=0.0+eps)
        opt_problem.model.add_constraint('con_TC2', lower=0.0+eps)
        opt_problem.model.add_constraint('con_TC3', lower=0.0+eps)
        opt_problem.model.add_constraint('con_bst', lower=0.0-eps)
        opt_problem.model.add_constraint('A_1', upper=60000.0-eps)
        opt_problem.model.add_constraint('J_s', upper=6.0)
        opt_problem.model.add_constraint('A_Cuscalc', lower=5.0, upper=300)
        opt_problem.model.add_constraint('K_rad', lower=0.2+eps, upper=0.27)
        opt_problem.model.add_constraint('Slot_aspect_ratio', lower=4.0, upper=10.0)
        opt_problem.model.add_constraint('gen_eff', lower=Eta_Target)

    if genType in ['eesg', 'pmsg_arms']:
        opt_problem.model.add_constraint('con_zAr', lower=0.0+eps)
        opt_problem.model.add_constraint('con_br', lower=0.0+eps)
    
    
    Objective_function = 'Costs'
    opt_problem.model.add_objective(Objective_function, scaler=1e-5)
    opt_problem.setup()
    
    # Specify Target machine parameters
    
    opt_problem['machine_rating'] = 5000000.0
    
    if genType in ['scig', 'dfig']:
        opt_problem['n_nom']              = 1200.0
        opt_problem['Gearbox_efficiency'] = 0.955
    elif genType in ['eesg', 'pmsg_arms','pmsg_disc']:
        opt_problem['Torque']             = 4.143289e6
        opt_problem['n_nom']              = 12.1

        
    if genType == 'scig':
        #opt_problem['r_s']     = 0.55 #0.484689156353 #0.55 #meter
        opt_problem['rad_ag']  = 0.55 #0.484689156353 #0.55 #meter
        opt_problem['len_s']   = 1.30 #1.27480124244 #1.3 #meter
        opt_problem['h_s']     = 0.090 #0.098331868116 # 0.090 #meter
        opt_problem['h_r']     = 0.050 #0.04 # 0.050 #meter
        opt_problem['I_0']     = 140  #139.995232826 #140  #Ampere
        opt_problem['B_symax'] = 1.4 #1.86140258387 #1.4 #Tesla
        
    elif genType == 'dfig':
        #opt_problem['r_s']     = 0.61 #0.493167295965 #0.61 #meter
        opt_problem['rad_ag']  = 0.61 #0.493167295965 #0.61 #meter
        opt_problem['len_s']   = 0.49 #1.06173588215 #0.49 #meter
        opt_problem['h_s']     = 0.08 #0.1 # 0.08 #meter
        opt_problem['h_r']     = 0.1 # 0.0998797703231 #0.1 #meter
        opt_problem['I_0']     = 40.0 # 40.0191207049 #40.0 #Ampere
        opt_problem['B_symax'] = 1.3 #1.59611292026 #1.3 #Tesla
        opt_problem['S_Nmax']  = -0.2 #-0.3 #-0.2

    elif genType == 'eesg':
        # Initial design variables 
        #opt_problem['r_s']     = 3.2
        opt_problem['rad_ag']  = 3.2
        opt_problem['len_s']   = 1.4
        opt_problem['h_s']     = 0.060
        opt_problem['tau_p']   = 0.170
        opt_problem['I_f']     = 69
        opt_problem['N_f']     = 100
        opt_problem['h_ys']    = 0.130
        opt_problem['h_yr']    = 0.120
        opt_problem['n_s']     = 5
        opt_problem['b_st']    = 0.470
        opt_problem['n_r']     = 5
        opt_problem['b_r']     = 0.480
        opt_problem['d_r']     = 0.510
        opt_problem['d_s']     = 0.400
        opt_problem['t_wr']    = 0.140
        opt_problem['t_ws']    = 0.070
        opt_problem['R_o']     = 0.43      #10MW: 0.523950817,#5MW: 0.43, #3MW:0.363882632 #1.5MW: 0.2775  0.75MW: 0.17625

    elif genType == 'pmsg_arms':
        #opt_problem['r_s']     = 3.26
        opt_problem['rad_ag']  = 3.26
        opt_problem['len_s']   = 1.60
        opt_problem['h_s']     = 0.070
        opt_problem['tau_p']   = 0.080
        opt_problem['h_m']     = 0.009
        opt_problem['h_ys']    = 0.075
        opt_problem['h_yr']    = 0.075
        opt_problem['n_s']     = 5.0
        opt_problem['b_st']    = 0.480
        opt_problem['n_r']     = 5.0
        opt_problem['b_r']     = 0.530
        opt_problem['d_r']     = 0.700
        opt_problem['d_s']     = 0.350
        opt_problem['t_wr']    = 0.06
        opt_problem['t_ws']    = 0.06
        opt_problem['R_o']     = 0.43           #0.523950817  #0.43  #0.523950817 #0.17625 #0.2775 #0.363882632 ##0.35 #0.523950817 #0.43 #523950817 #0.43 #0.523950817 #0.523950817 #0.17625 #0.2775 #0.363882632 #0.43 #0.523950817 #0.43

    elif genType == 'pmsg_disc':
        #opt_problem['r_s']     = 3.49 #3.494618182
        opt_problem['rad_ag']  = 3.49 #3.494618182
        opt_problem['len_s']   = 1.5 #1.506103927
        opt_problem['h_s']     = 0.06 #0.06034976
        opt_problem['tau_p']   = 0.07 #0.07541515 
        opt_problem['h_m']     = 0.0105 #0.0090100202 
        opt_problem['h_ys']    = 0.085 #0.084247994 #
        opt_problem['h_yr']    = 0.055 #0.0545789687
        opt_problem['n_s']     = 5.0 #5.0
        opt_problem['b_st']    = 0.460 #0.46381
        opt_problem['t_d']     = 0.105 #0.10 
        opt_problem['d_s']     = 0.350 #0.35031 #
        opt_problem['t_ws']    = 0.150 #=0.14720 #
        opt_problem['R_o']     = 0.43 #0.43

    #----------------- try 15MW PMSG_disc --------------
    #  testing 2019 11 04
    
    if genType in ['eesg', 'pmsg_arms','pmsg_disc']:
        opt_problem['machine_rating'] = 15000000.0
        opt_problem['Torque']         = 20.64e6
        opt_problem['n_nom']          = 7.54
        opt_problem['machine_rating'] = 10000000.0
        opt_problem['Torque']         = 12.64e6
        
    #---------------------------------------------------
        
    # Specific costs
    opt_problem['C_Cu']         = 4.786         # Unit cost of Copper $/kg
    opt_problem['C_Fe']         = 0.556         # Unit cost of Iron $/kg
    opt_problem['C_Fes']        = 0.50139       # specific cost of Structural_mass $/kg
    
    #Material properties
    opt_problem['rho_Fe']       = 7700.0        # Steel density Kg/m3
    opt_problem['rho_Fes']      = 7850          # structural Steel density Kg/m3
    opt_problem['rho_Copper']   = 8900.0        # copper density Kg/m3
    opt_problem['rho_PM']       = 7450.0        # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]
            
    opt_problem['shaft_cm']     = np.zeros(3)
    opt_problem['shaft_length'] = 2.0
    
    #Run optimization
    opt_problem.model.approx_totals()
    opt_problem.run_driver()
    opt_problem.model.list_inputs(units=True) #values = False, hierarchical=False)
    opt_problem.model.list_outputs(units=True) #values = False, hierarchical=False)    

    # Export results
    if exportFlag:
        import pandas as pd

        if genType == 'scig':
            raw_data = {'Parameters': ['Rating',
                                       'Objective function',
                                       'Air gap diameter',
                                       'Stator length',
                                       'Lambda ratio',
                                       'Diameter ratio',
                                       'Pole pitch(tau_p)',
                                       'Number of Stator Slots',
                                       'Stator slot height(h_s)',
                                       'Stator slot width(b_s)',
                                       'Stator Slot aspect ratio',
                                       'Stator tooth width(b_t)',
                                       'Stator yoke height(h_ys)',
                                       'Rotor slots',
                                       'Rotor slot height(h_r)',
                                       'Rotor slot width(b_r)',
                                       'Rotor tooth width(b_tr)',
                                       'Rotor yoke height(h_yr)',
                                       'Rotor Slot_aspect_ratio',
                                       'Peak air gap flux density',
                                       'Peak air gap flux density fundamental',
                                       'Peak stator yoke flux density',
                                       'Peak rotor yoke flux density',
                                       'Peak Stator tooth flux density',
                                       'Peak Rotor tooth flux density',
                                       'Pole pairs',
                                       'Generator output frequency',
                                       'Generator output phase voltage',
                                       'Generator Output phase current',
                                       'Slip',
                                       'Stator Turns',
                                       'Conductor cross-section',
                                       'Stator Current density',
                                       'Specific current loading',
                                       'Stator resistance',
                                       'Excited magnetic inductance',
                                       'Magnetization current',
                                       'Conductor cross-section',
                                       'Rotor Current density',
                                       'Rotor resitance',
                                       'Generator Efficiency',
                                       'Overall drivetrain Efficiency',
                                       'Copper mass',
                                       'Iron Mass',
                                       'Structural mass',
                                       'Total Mass',
                                       'Total Material Cost'],
                        
                        'Values': [opt_problem['machine_rating']/1e6,
                                   Objective_function,
                                   2*opt_problem['r_s'],
                                   opt_problem['len_s'],
                                   opt_problem['K_rad'],
                                   opt_problem['D_ratio'],
                                   opt_problem['tau_p']*1000,
                                   opt_problem['S'],
                                   opt_problem['h_s']*1000,
                                   opt_problem['b_s']*1000,
                                   opt_problem['Slot_aspect_ratio1'],
                                   opt_problem['b_t']*1000,
                                   opt_problem['h_ys']*1000,
                                   opt_problem['Q_r'],
                                   opt_problem['h_r']*1000,
                                   opt_problem['b_r']*1000,
                                   opt_problem['b_tr']*1000,
                                   opt_problem['h_yr']*1000,
                                   opt_problem['Slot_aspect_ratio2'],
                                   opt_problem['B_g'],
                                   opt_problem['B_g1'],
                                   opt_problem['B_symax'],
                                   opt_problem['B_rymax'],
                                   opt_problem['B_tsmax'],
                                   opt_problem['B_trmax'],
                                   opt_problem['p'],
                                   opt_problem['f'],
                                   opt_problem['E_p'],
                                   opt_problem['I_s'],
                                   opt_problem['S_N'],
                                   opt_problem['N_s'],
                                   opt_problem['A_Cuscalc'],
                                   opt_problem['J_s'],
                                   opt_problem['A_1']/1000,
                                   opt_problem['R_s'],
                                   opt_problem['L_sm'],
                                   opt_problem['I_0'],
                                   opt_problem['A_bar']*1e6,
                                   opt_problem['J_r'],
                                   opt_problem['R_R'],
                                   opt_problem['gen_eff'],
                                   opt_problem['Overall_eff'],
                                   opt_problem['Copper']/1000,
                                   opt_problem['Iron']/1000,
                                   opt_problem['Structural_mass']/1000,
                                   opt_problem['Mass']/1000,
                                   opt_problem['Costs']/1000],
                        
                        'Limit': ['',
                                  '',
                                  '',
                                  '',
                                  '('+str(opt_problem['K_rad_LL'])+'-'+str(opt_problem['K_rad_UL'])+')',
                                  '('+str(opt_problem['D_ratio_LL'])+'-'+str(opt_problem['D_ratio_UL'])+')',
                                  
                                  '',
                                  '',
                                  '',
                                  '',
                                  '(4-10)',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '(4-10)',
                                  '(0.7-1.2)',
                                  '',
                                  '2',
                                  '2',
                                  '2',
                                  '2',
                                  '',
                                  '(10-60)',
                                  '(500-5000)',
                                  '',
                                  '(-30% to -0.2%)',
                                  '',
                                  '',
                                  '(3-6)',
                                  '<60',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  Eta_Target,
                                  '',
                                  '',
                                  '',
                                  '',
                                  ''],
                        
                        'Units':['MW',
                                 '',
                                 'm',
                                 'm',
                                 '-',
                                 '-',
                                 'mm',
                                 '-',
                                 'mm',
                                 'mm',
                                 '-',
                                 'mm',
                                 'mm',
                                 '-',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 '',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 '-',
                                 'Hz',
                                 'V',
                                 'A',
                                 '%',
                                 'turns',
                                 'mm^2',
                                 'A/mm^2',
                                 'kA/m',
                                 'ohms',
                                 'p.u',
                                 'A',
                                 'mm^2',
                                 'A/mm^2',
                                 'ohms',
                                 '%',
                                 '%',
                                 'Tons',
                                 'Tons',
                                 'Tons',
                                 'Tons',
                                 '$1000']}
            
        elif genType == 'dfig':
            raw_data = {'Parameters': ['Rating',
                                       'Objective function',
                                       'Air gap diameter',
                                       'Stator length',
                                       'K_rad',
                                       'Diameter ratio',
                                       'Pole pitch(tau_p)',
                                       ' Number of Stator Slots',
                                       'Stator slot height(h_s)',
                                       'Slots/pole/phase',
                                       'Stator slot width(b_s)',
                                       ' Stator slot aspect ratio',
                                       'Stator tooth width(b_t)',
                                       'Stator yoke height(h_ys)',
                                       'Rotor slots',
                                       'Rotor yoke height(h_yr)',
                                       'Rotor slot height(h_r)',
                                       'Rotor slot width(b_r)',
                                       ' Rotor Slot aspect ratio',
                                       'Rotor tooth width(b_t)',
                                       'Peak air gap flux density',
                                       'Peak air gap flux density fundamental',
                                       'Peak stator yoke flux density',
                                       'Peak rotor yoke flux density',
                                       'Peak Stator tooth flux density',
                                       'Peak rotor tooth flux density',
                                       'Pole pairs',
                                       'Generator output frequency',
                                       'Generator output phase voltage',
                                       'Generator Output phase current',
                                       'Optimal Slip',
                                       'Stator Turns',
                                       'Conductor cross-section',
                                       'Stator Current density',
                                       'Specific current loading',
                                       'Stator resistance',
                                       'Stator leakage inductance',
                                       'Excited magnetic inductance',
                                       'Rotor winding turns',
                                       'Conductor cross-section',
                                       'Magnetization current',
                                       'I_mag/Is',
                                       'Rotor Current density',
                                       'Rotor resitance',
                                       'Rotor leakage inductance',
                                       'Generator Efficiency',
                                       'Overall drivetrain Efficiency',
                                       'Copper mass',
                                       'Iron mass',
                                       'Structural Steel mass',
                                       'Total Mass',
                                       'Total Material Cost'],
                        
                        'Values': [opt_problem['machine_rating']/1e6,
                                   Objective_function,
                                   2*opt_problem['r_s'],
                                   opt_problem['len_s'],
                                   opt_problem['K_rad'],
                                   opt_problem['D_ratio'],
                                   opt_problem['tau_p']*1000,
                                   opt_problem['S'],
                                   opt_problem['h_s']*1000,
                                   opt_problem['q1'],
                                   opt_problem['b_s']*1000,
                                   opt_problem['Slot_aspect_ratio1'],
                                   opt_problem['b_t']*1000,
                                   opt_problem['h_ys']*1000,
                                   opt_problem['Q_r'],
                                   opt_problem['h_yr']*1000,
                                   opt_problem['h_r']*1000,
                                   opt_problem['b_r']*1000,
                                   opt_problem['Slot_aspect_ratio2'],
                                   opt_problem['b_tr']*1000,
                                   opt_problem['B_g'],
                                   opt_problem['B_g1'],
                                   opt_problem['B_symax'],
                                   opt_problem['B_rymax'],
                                   opt_problem['B_tsmax'],
                                   opt_problem['B_trmax'],
                                   opt_problem['p'],
                                   opt_problem['f'],
                                   opt_problem['E_p'],
                                   opt_problem['I_s'],
                                   opt_problem['S_Nmax'],
                                   opt_problem['N_s'],
                                   opt_problem['A_Cuscalc'],
                                   opt_problem['J_s'],
                                   opt_problem['A_1']/1000,
                                   opt_problem['R_s'],
                                   opt_problem['L_s'],
                                   opt_problem['L_sm'],
                                   opt_problem['N_r'],
                                   opt_problem['A_Curcalc'],
                                   opt_problem['I_0'],
                                   opt_problem['Current_ratio'],
                                   opt_problem['J_r'],
                                   opt_problem['R_R'],
                                   opt_problem['L_r'],
                                   opt_problem['gen_eff'],
                                   opt_problem['Overall_eff'],
                                   opt_problem['Copper']/1000,
                                   opt_problem['Iron']/1000,
                                   opt_problem['Structural_mass']/1000,
                                   opt_problem['Mass']/1000,
                                   opt_problem['Costs']/1000],
                        
                        'Limit': ['',
                                  '',
                                  '',
                                  '',
                                  '(0.2-1.5)',
                                  '(1.37-1.4)',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '(4-10)',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '(0.7-1.2)',
                                  '',
                                  '2.',
                                  '2.',
                                  '2.',
                                  '2.',
                                  '',
                                  '',
                                  '(500-5000)',
                                  '',
                                  '(-0.002-0.3)',
                                  '',
                                  '',
                                  '(3-6)',
                                  '<60',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '(0.1-0.3)',
                                  '(3-6)',
                                  '',
                                  '',
                                  '',
                                  Eta_Target,
                                  '',
                                  '',
                                  '',
                                  '',
                                  ''],
                        
                        'Units':['MW',
                                 '',
                                 'm',
                                 'm',
                                 '-',
                                 '-',
                                 'mm',
                                 '-',
                                 'mm',
                                 '',
                                 'mm',
                                 '',
                                 'mm',
                                 'mm',
                                 '-',
                                 'mm',
                                 'mm',
                                 'mm',
                                 '-',
                                 'mm',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 '-',
                                 'Hz',
                                 'V',
                                 'A',
                                 '',
                                 
                                 'turns',
                                 'mm^2',
                                 'A/mm^2',
                                 'kA/m',
                                 'ohms',
                                 '',
                                 '',
                                 'turns',
                                 'mm^2',
                                 'A',
                                 '',
                                 'A/mm^2',
                                 'ohms',
                                 'p.u',
                                 '%',
                                 '%',
                                 'Tons',
                                 'Tons',
                                 'Tons',
                                 'Tons',
                                 '$1000']}

        elif genType == 'eesg':
            raw_data = {'Parameters': ['Rating',
                                       'Stator Arms',
                                       'Stator Axial arm dimension',
                                       'Stator Circumferential arm dimension',
                                       'Stator arm Thickness',
                                       'Rotor Arms',
                                       'Rotor Axial arm dimension',
                                       'Rotor Circumferential arm dimension',
                                       'Rotor Arm thickness',
                                       'Rotor Radial deflection',
                                       'Rotor Axial deflection',
                                       'Rotor circum deflection',
                                       'Stator Radial deflection',
                                       'Stator Axial deflection',
                                       'Stator Circumferential deflection',
                                       'Air gap diameter',
                                       'Stator length',
                                       'l/D ratio',
                                       'Pole pitch',
                                       'Stator slot height',
                                       'Stator slot width',
                                       'Slot aspect ratio',
                                       'Stator tooth width',
                                       'Stator yoke height',
                                       'Rotor yoke height',
                                       'Rotor pole height',
                                       'Rotor pole width',
                                       'Average no-load flux density',
                                       'Peak air gap flux density',
                                       'Peak stator yoke flux density',
                                       'Peak rotor yoke flux density',
                                       'Stator tooth flux density',
                                       'Rotor pole core flux density',
                                       'Pole pairs',
                                       'Generator output frequency',
                                       'Generator output phase voltage(rms value)',
                                       'Generator Output phase current',
                                       'Stator resistance',
                                       'Synchronous inductance',
                                       'Stator slots',
                                       'Stator turns',
                                       'Stator conductor cross-section',
                                       'Stator Current density ',
                                       'Specific current loading',
                                       'Field turns',
                                       'Conductor cross-section',
                                       'Field Current',
                                       'D.C Field resistance',
                                       'MMF ratio at rated load(Rotor/Stator)',
                                       'Excitation Power (% of Rated Power)',
                                       'Number of brushes/polarity',
                                       'Field Current density',
                                       'Generator Efficiency',
                                       'Iron mass',
                                       'Copper mass',
                                       'Mass of Arms',
                                       'Total Mass',
                                       'Total Cost'],
                        
                        'Values': [opt_problem['machine_rating']/1e6,
                                   opt_problem['n_s'],
                                   opt_problem['d_s']*1000,
                                   opt_problem['b_st']*1000,
                                   opt_problem['t_ws']*1000,
                                   opt_problem['n_r'],
                                   opt_problem['d_r']*1000,
                                   opt_problem['b_r']*1000,
                                   opt_problem['t_wr']*1000,
                                   opt_problem['u_Ar']*1000,
                                   opt_problem['y_Ar']*1000,
                                   opt_problem['z_A_r']*1000,
                                   opt_problem['u_As']*1000,
                                   opt_problem['y_As']*1000,
                                   opt_problem['z_A_s']*1000,
                                   2*opt_problem['r_s'],
                                   opt_problem['len_s'],
                                   opt_problem['K_rad'],
                                   opt_problem['tau_p']*1000,
                                   opt_problem['h_s']*1000,
                                   opt_problem['b_s']*1000,
                                   opt_problem['Slot_aspect_ratio'],
                                   opt_problem['b_t']*1000,
                                   opt_problem['h_ys']*1000,
                                   opt_problem['h_yr']*1000,
                                   opt_problem['h_p']*1000,
                                   opt_problem['b_p']*1000,
                                   opt_problem['B_gfm'],
                                   opt_problem['B_g'],
                                   opt_problem['B_symax'],
                                   opt_problem['B_rymax'],
                                   opt_problem['B_tmax'],
                                   opt_problem['B_pc'],
                                   opt_problem['p'],
                                   opt_problem['f'],
                                   opt_problem['E_s'],
                                   opt_problem['I_s'],
                                   opt_problem['R_s'],
                                   opt_problem['L_m'],
                                   opt_problem['S'],
                                   opt_problem['N_s'],
                                   opt_problem['A_Cuscalc'],
                                   opt_problem['J_s'],
                                   opt_problem['A_1']/1000,
                                   opt_problem['N_f'],
                                   opt_problem['A_Curcalc'],
                                   opt_problem['I_f'],
                                   opt_problem['R_r'],
                                   opt_problem['Load_mmf_ratio'],
                                   opt_problem['Power_ratio'],
                                   opt_problem['n_brushes'],
                                   opt_problem['J_f'],
                                   opt_problem['gen_eff'],
                                   opt_problem['Iron']/1000,
                                   opt_problem['Copper']/1000,
                                   opt_problem['Structural_mass']/1000,
                                   opt_problem['Mass']/1000,
                                   opt_problem['Costs']/1000],
                        
                        'Limit': ['',
                                  '',
                                  '',
                                  opt_problem['b_all_s']*1000,
                                  '',
                                  '',
                                  '',
                                  opt_problem['b_all_r']*1000,
                                  '',
                                  opt_problem['u_all_r']*1000,
                                  opt_problem['y_all']*1000,
                                  opt_problem['z_all_r']*1000,
                                  opt_problem['u_all_s']*1000,
                                  opt_problem['y_all']*1000,
                                  opt_problem['z_all_s']*1000,
                                  
                                  '',
                                  '',
                                  '(0.2-0.27)',
                                  '',
                                  '',
                                  '',
                                  '(4-10)',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '(0.62-1.05)',
                                  '1.2',
                                  '2',
                                  '2',
                                  '2',
                                  '2',
                                  '',
                                  '(10-60)',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '(3-6)',
                                  '<60',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '<2%',
                                  '',
                                  '(3-6)',
                                  Eta_Target,
                                  '',
                                  '',
                                  '',
                                  '',
                                  ''],
                        
                        'Units':['MW',
                                 'unit',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'unit',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'm',
                                 'm',
                                 '',
                                 '',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 '-',
                                 'Hz',
                                 'V',
                                 'A',
                                 'om/phase',
                                 
                                 'p.u',
                                 'slots',
                                 'turns',
                                 'mm^2',
                                 'A/mm^2',
                                 'kA/m',
                                 'turns',
                                 'mm^2',
                                 'A',
                                 'ohm',
                                 '%',
                                 '%',
                                 'brushes',
                                 'A/mm^2',
                                 'turns',
                                 '%',
                                 'tons',
                                 'tons',
                                 'tons',
                                 '1000$']} 

        elif genType == 'pmsg_arms':
                raw_data = {'Parameters': ['Rating',
                                           'Stator Arms',
                                           'Stator Axial arm dimension',
                                           'Stator Circumferential arm dimension',
                                           'Stator arm Thickness',
                                           'Rotor arms',
                                           'Rotor Axial arm dimension',
                                           'Rotor Circumferential arm dimension',
                                           'Rotor arm Thickness',
                                           'Stator Radial deflection',
                                           'Stator Axial deflection',
                                           'Stator circum deflection',
                                           'Rotor Radial deflection',
                                           'Rotor Axial deflection',
                                           'Rotor circum deflection',
                                           'Air gap diameter',
                                           'Overall Outer diameter',
                                           'Stator length',
                                           'l/d ratio',
                                           'Slot_aspect_ratio',
                                           'Pole pitch',
                                           'Stator slot height',
                                           'Stator slotwidth',
                                           'Stator tooth width',
                                           'Stator yoke height',
                                           'Rotor yoke height',
                                           'Magnet height',
                                           'Magnet width',
                                           'Peak air gap flux density fundamental',
                                           'Peak stator yoke flux density',
                                           'Peak rotor yoke flux density',
                                           'Flux density above magnet',
                                           'Maximum Stator flux density',
                                           'Maximum tooth flux density',
                                           'Pole pairs',
                                           'Generator output frequency',
                                           'Generator output phase voltage',
                                           'Generator Output phase current',
                                           'Stator resistance',
                                           'Synchronous inductance',
                                           'Stator slots',
                                           'Stator turns',
                                           'Conductor cross-section',
                                           'Stator Current density ',
                                           'Specific current loading',
                                           'Generator Efficiency ',
                                           'Iron mass',
                                           'Magnet mass',
                                           'Copper mass',
                                           'Mass of Arms',
                                           'Total Mass',
                                           'Total Material Cost'],
                            
                        'Values': [opt_problem['machine_rating']/1000000,
                                   opt_problem['n_s'],
                                   opt_problem['d_s']*1000,
                                   opt_problem['b_st']*1000,
                                   opt_problem['t_ws']*1000,
                                   opt_problem['n_r'],
                                   opt_problem['d_r']*1000,
                                   opt_problem['b_r']*1000,
                                   opt_problem['t_wr']*1000,
                                   opt_problem['u_As']*1000,
                                   opt_problem['y_As']*1000,
                                   opt_problem['z_A_s']*1000,
                                   opt_problem['u_Ar']*1000,
                                   opt_problem['y_Ar']*1000,
                                   opt_problem['z_A_r']*1000,
                                   2*opt_problem['r_s'],
                                   opt_problem['R_out']*2,
                                   opt_problem['len_s'],
                                   opt_problem['K_rad'],
                                   opt_problem['Slot_aspect_ratio'],
                                   opt_problem['tau_p']*1000,
                                   opt_problem['h_s']*1000,
                                   opt_problem['b_s']*1000,
                                   opt_problem['b_t']*1000,
                                   opt_problem['h_ys']*1000,
                                   opt_problem['h_yr']*1000,
                                   opt_problem['h_m']*1000,
                                   opt_problem['b_m']*1000,
                                   opt_problem['B_g'],
                                   opt_problem['B_symax'],
                                   opt_problem['B_rymax'],
                                   opt_problem['B_pm1'],
                                   opt_problem['B_smax'],
                                   opt_problem['B_tmax'],
                                   opt_problem['p'],
                                   opt_problem['f'],
                                   opt_problem['E_p'],
                                   opt_problem['I_s'],
                                   opt_problem['R_s'],
                                   opt_problem['L_s'],
                                   opt_problem['S'],
                                   opt_problem['N_s'],
                                   opt_problem['A_Cuscalc'],
                                   opt_problem['J_s'],
                                   opt_problem['A_1']/1000,
                                   opt_problem['gen_eff'],
                                   opt_problem['Iron']/1000,
                                   opt_problem['mass_PM']/1000,
                                   opt_problem['Copper']/1000,
                                   opt_problem['Structural_mass']/1000,
                                   opt_problem['Mass']/1000,
                                   opt_problem['Costs']/1000],
                            
                        'Limit': ['',
                                  '',
                                  '',
                                  opt_problem['b_all_s']*1000,
                                  '',
                                  '',
                                  '',
                                  opt_problem['b_all_r']*1000,
                                  '',
                                  opt_problem['u_all_s']*1000,
                                  opt_problem['y_all']*1000,
                                  opt_problem['z_all_s']*1000,
                                  opt_problem['u_all_r']*1000,
                                  opt_problem['y_all']*1000,
                                  opt_problem['z_all_r']*1000,
                                  '',
                                  '',
                                  '',
                                  '(0.2-0.27)',
                                  '(4-10)',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '<2',
                                  '<2',
                                  '<2',
                                  opt_problem['B_g'],
                                  '',
                                  '',
                                  '',
                                  '',
                                  '>500',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '5',
                                  '3-6',
                                  '60',
                                  '>93%',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  ''],
                            
                        'Units':['MW',
                                 'unit',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 '',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'm',
                                 'm',
                                 'm',
                                 '',
                                 '',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 '-',
                                 'Hz',
                                 'V',
                                 'A',
                                 'ohm/phase',
                                 'p.u',
                                 'slots',
                                 'turns',
                                 'mm^2',
                                 'A/mm^2',
                                 'kA/m',
                                 '%',
                                 'tons',
                                 'tons',
                                 'tons',
                                 'tons',
                                 'tons',
                                 'k$']}

        elif genType == 'pmsg_disc':
            raw_data = {'Parameters': ['Rating',
                                       'Stator Arms',
                                       'Stator Axial arm dimension',
                                       'Stator Circumferential arm dimension',
                                       'Stator arm Thickness',
                                       'Rotor disc Thickness',
                                       'Stator Radial deflection',
                                       'Stator Axial deflection',
                                       'Stator circum deflection',
                                       'Rotor Radial deflection',
                                       'Rotor Axial deflection',
                                       'Air gap diameter',
                                       'Overall Outer diameter',
                                       'Stator length',
                                       'l/d ratio',
                                       'Slot_aspect_ratio',
                                       'Pole pitch',
                                       'Stator slot height',
                                       'Stator slotwidth',
                                       'Stator tooth width',
                                       'Stator yoke height',
                                       'Rotor yoke height',
                                       'Magnet height',
                                       'Magnet width',
                                       'Peak air gap flux density fundamental',
                                       'Peak stator yoke flux density',
                                       'Peak rotor yoke flux density',
                                       'Flux density above magnet',
                                       'Maximum Stator flux density',
                                       'Maximum tooth flux density',
                                       'Pole pairs',
                                       'Generator output frequency',
                                       'Generator output phase voltage',
                                       'Generator Output phase current',
                                       'Stator resistance',
                                       'Synchronous inductance',
                                       'Stator slots',
                                       'Stator turns',
                                       'Conductor cross-section',
                                       'Stator Current density ',
                                       'Specific current loading',
                                       'Generator Efficiency ',
                                       'Iron mass',
                                       'Magnet mass',
                                       'Copper mass',
                                       'Mass of Arms and disc',
                                       'Total Mass',
                                       'Total Material Cost'],
                        
                        'Values': [opt_problem['machine_rating']/1000000,
                                   opt_problem['n_s'],
                                   opt_problem['d_s']*1000,
                                   opt_problem['b_st']*1000,
                                   opt_problem['t_ws']*1000,
                                   opt_problem['t_d']*1000,
                                   opt_problem['u_As']*1000,
                                   opt_problem['y_As']*1000,
                                   opt_problem['z_A_s']*1000,
                                   opt_problem['u_Ar']*1000,
                                   opt_problem['y_Ar']*1000,
                                   2*opt_problem['r_s'],
                                   opt_problem['R_out']*2,
                                   opt_problem['len_s'],
                                   opt_problem['K_rad'],
                                   opt_problem['Slot_aspect_ratio'],
                                   opt_problem['tau_p']*1000,
                                   opt_problem['h_s']*1000,
                                   opt_problem['b_s']*1000,
                                   opt_problem['b_t']*1000,
                                   opt_problem['h_ys']*1000,
                                   opt_problem['h_yr']*1000,
                                   opt_problem['h_m']*1000,
                                   opt_problem['b_m']*1000,
                                   opt_problem['B_g'],
                                   opt_problem['B_symax'],
                                   opt_problem['B_rymax'],
                                   opt_problem['B_pm1'],
                                   opt_problem['B_smax'],
                                   opt_problem['B_tmax'],
                                   opt_problem['p'],
                                   opt_problem['f'],
                                   opt_problem['E_p'],
                                   opt_problem['I_s'],
                                   opt_problem['R_s'],
                                   opt_problem['L_s'],
                                   opt_problem['S'],
                                   opt_problem['N_s'],
                                   opt_problem['A_Cuscalc'],
                                   opt_problem['J_s'],
                                   opt_problem['A_1']/1000,
                                   opt_problem['gen_eff'],
                                   opt_problem['Iron']/1000,
                                   opt_problem['mass_PM']/1000,
                                   opt_problem['Copper']/1000,
                                   opt_problem['Structural_mass']/1000,
                                   opt_problem['Mass']/1000,
                                   opt_problem['Costs']/1000],
                        
                        'Limit': ['',
                                  '',
                                  '',
                                  opt_problem['b_all_s']*1000,
                                  '',
                                  '',
                                  opt_problem['u_all_s']*1000,
                                  opt_problem['y_all']*1000,
                                  opt_problem['z_all_s']*1000,
                                  opt_problem['u_all_r']*1000,
                                  opt_problem['y_all']*1000,
                                  '',
                                  '',
                                  '',
                                  '(0.2-0.27)',
                                  '(4-10)',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '<2',
                                  '<2',
                                  '<2',
                                  '<2',
                                  '<2',
                                  opt_problem['B_g'],
                                  '<2',
                                  '',
                                  '>500',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '5',
                                  
                                  '3-6',
                                  '60',
                                  '>93%',
                                  '',
                                  '',
                                  '',
                                  '',
                                  '',
                                  ''],
                        
                        'Units':['MW',
                                 'unit',
                                 'mm',
                                 'mm',
                                 'mm',
                                 '',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'm',
                                 'm',
                                 'm',
                                 '',
                                 '',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'mm',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 'T',
                                 '-',
                                 'Hz',
                                 'V',
                                 'A',
                                 'ohm/phase',
                                 'p.u',
                                 'slots',
                                 'turns',
                                 'mm^2',
                                 'A/mm^2',
                                 'kA/m',
                                 '%',
                                 'tons',
                                 'tons',
                                 'tons',
                                 'tons',
                                 'tons',
                                 'k$']}
                
            
        df = pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
        print(df)
        df.to_excel(genType.upper() + '_' + str(float(opt_problem['machine_rating']/1e6)) + '_MW.xlsx')
        df.to_csv(genType.upper() + '_' + str(float(opt_problem['machine_rating']/1e6)) + '_MW.csv')
            
if __name__=='__main__':
    
    # Run example optimizations for all generator types
    #for m in ['eesg',]:
    #for m in ['scig','dfig','eesg','pmsg_arms','pmsg_disc']:
    for m in ['pmsg_disc']:
        optimization_example(m, exportFlag=True)
