"""generator.py
Created by Latha Sethuraman, Katherine Dykes. 
Copyright (c) NREL. All rights reserved.

Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

import openmdao.api as om
import numpy as np
import generator_models as gm
import wisdem.commonse.fileIO as fio

#----------------------------------------------------------------------------------------------

class Constraints(om.ExplicitComponent):
    """ Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded"""
    def setup(self):
        self.add_input('u_allow_s', val=0.0, units='m')
        self.add_input('u_as', val=0.0, units='m')
        self.add_input('z_allow_s', val=0.0, units='m')
        self.add_input('z_as', val=0.0, units='m')
        self.add_input('y_allow_s', val=0.0, units='m')
        self.add_input('y_as', val=0.0, units='m')
        self.add_input('b_allow_s', val=0.0, units='m')
        self.add_input('b_st', val=0.0, units='m')
        self.add_input('u_allow_r', val=0.0, units='m')
        self.add_input('u_ar', val=0.0, units='m')
        self.add_input('y_allow_r', val=0.0, units='m')
        self.add_input('y_ar', val=0.0, units='m')
        self.add_input('z_allow_r', val=0.0, units='m')
        self.add_input('z_ar', val=0.0, units='m')
        self.add_input('b_allow_r', val=0.0, units='m')
        self.add_input('b_arm', val=0.0, units='m')
        self.add_input('TC1', val=0.0, units='m**3')
        self.add_input('TC2r', val=0.0, units='m**3')
        self.add_input('TC2s', val=0.0, units='m**3')
        self.add_input('B_g', val=0.0, units='T')
        self.add_input('B_smax', val=0.0, units='T')
        self.add_input('K_rad', val=0.0)
        self.add_input('K_rad_LL', val=0.0)
        self.add_input('K_rad_UL', val=0.0)
        self.add_input('D_ratio', val=0.0)
        self.add_input('D_ratio_LL', val=0.0)
        self.add_input('D_ratio_UL', val=0.0)
        
        self.add_output('con_uas', val=0.0, units='m')
        self.add_output('con_zas',  val=0.0, units='m')
        self.add_output('con_yas',  val=0.0, units='m')
        self.add_output('con_bst',  val=0.0, units='m')
        self.add_output('con_uar',  val=0.0, units='m')
        self.add_output('con_yar',  val=0.0, units='m')
        self.add_output('con_zar',  val=0.0, units='m')
        self.add_output('con_br',  val=0.0, units='m')
        self.add_output('TCr', val=0.0, units='m**3')
        self.add_output('TCs', val=0.0, units='m**3')
        self.add_output('con_TC2r',  val=0.0, units='m**3')
        self.add_output('con_TC2s',  val=0.0, units='m**3')
        self.add_output('con_Bsmax',  val=0.0, units='T')    
        self.add_output('K_rad_L', val=0.0)
        self.add_output('K_rad_U', val=0.0)
        self.add_output('D_ratio_L', val=0.0)
        self.add_output('D_ratio_U', val=0.0)
        
    def compute(self, inputs, outputs):
        outputs['con_uas'] = inputs['u_allow_s'] - inputs['u_as']
        outputs['con_zas'] = inputs['z_allow_s'] - inputs['z_as']
        outputs['con_yas'] = inputs['y_allow_s'] - inputs['y_as']
        outputs['con_bst'] = inputs['b_allow_s'] - inputs['b_st']   #b_st={'units':'m'}
        outputs['con_uar'] = inputs['u_allow_r'] - inputs['u_ar']
        outputs['con_yar'] = inputs['y_allow_r'] - inputs['y_ar']
        outputs['con_TC2r'] = inputs['TC2s'] - inputs['TC1']
        outputs['con_TC2s'] = inputs['TC2s'] - inputs['TC1']
        outputs['con_Bsmax'] = inputs['B_g'] - inputs['B_smax']
        outputs['con_zar'] = inputs['z_allow_r'] - inputs['z_ar']
        outputs['con_br'] = inputs['b_allow_r'] - inputs['b_arm'] # b_r={'units':'m'}
        outputs['TCr'] = inputs['TC2r'] - inputs['TC1']
        outputs['TCs'] = inputs['TC2s'] - inputs['TC1']
        outputs['K_rad_L'] = inputs['K_rad'] - inputs['K_rad_LL']
        outputs['K_rad_U'] = inputs['K_rad'] - inputs['K_rad_UL']
        outputs['D_ratio_L'] = inputs['D_ratio'] - inputs['D_ratio_LL']
        outputs['D_ratio_U'] = inputs['D_ratio'] - inputs['D_ratio_UL']

#----------------------------------------------------------------------------------------------
        
class Cost(om.ExplicitComponent):
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
        


#----------------------------------------------------------------------------------------------


class Generator(om.Group):

    def initialize(self):
        genTypes = ['scig','dfig','eesg','pmsg_arms','pmsg_disc','pmsg_outer']
        self.options.declare('topLevelFlag', default=True)
        self.options.declare('design', values=genTypes + [m.upper() for m in genTypes])
    
    def setup(self):
        topLevelFlag = self.options['topLevelFlag']
        genType      = self.options['design']
        
        ivc = om.IndepVarComp()
        ivc.add_output('shaft_cm', val=np.zeros(3), units='m')
        ivc.add_output('shaft_length', val=0.0, units='m')
        
        ivc.add_output('Gearbox_efficiency', val=0.0)
        ivc.add_output('r_s',   val=0.0, units='m')
        ivc.add_output('len_s', val=0.0, units='m')
        ivc.add_output('h_s',   val=0.0, units='m')
        ivc.add_output('h_r',   val=0.0, units='m')
        ivc.add_output('h_m',   val=0.0, units='m')
        ivc.add_output('S_Nmax', val=0.0, units='T')
        ivc.add_output('I_0',   val=0.0, units='A')
        ivc.add_output('sigma', 0.0,units='N/m**2',desc='shear stress')
        ivc.add_output('p', 0.0, desc='Pole pairs')

        if genType.lower() in ['pmsg_outer']:
            ivc.add_output('r_g', 0.0,units='m',desc='Air gap radius')
            ivc.add_output('h_ys',  val=0.0, units='m')
            ivc.add_output('h_yr',  val=0.0, units='m')
            ivc.add_output('B_tmax',0.0, units='T', desc='Teeth flux density')
            ivc.add_output('t_r', 0.0,units='m',desc='Rotor disc thickness')
            ivc.add_output('t_s', 0.0,units='m',desc='Stator disc thickness' )
            ivc.add_output('h_ss',0.0,units='m',desc='Stator rim thickness')
            ivc.add_output('h_sr', 0.0,units='m',desc='Rotor rim thickness')    
            ivc.add_output('u_allow_pcent', 0.0,desc='% radial deflection')
            ivc.add_output('y_allow_pcent', 0.0,desc='% axial deflection')
            ivc.add_output('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
            
        if genType.lower() in ['eesg','pmsg_arms','pmsg_disc']:
            ivc.add_output('tau_p', val=0.0, units='m')
            ivc.add_output('h_ys',  val=0.0, units='m')
            ivc.add_output('h_yr',  val=0.0, units='m')
            ivc.add_output('b_arm',   val=0.0, units='m')
        elif genType.lower() in ['scig','dfig']:
            ivc.add_output('B_symax', val=0.0, units='T')
            
        ivc.add_output('I_f',  val=0.0, units='A')
        ivc.add_output('N_f',  val=0.0, units='A')
        ivc.add_output('n_s',  val=0.0)
        ivc.add_output('b_st', val=0.0, units='m')
        ivc.add_output('n_r',  val=0.0)
        ivc.add_output('d_r',  val=0.0, units='m')
        ivc.add_output('d_s',  val=0.0, units='m')
        ivc.add_output('t_d',  val=0.0, units='m')
        ivc.add_output('t_wr', val=0.0, units='m')
        ivc.add_output('t_ws', val=0.0, units='m')
        ivc.add_output('R_o',  val=0.0, units='m')
        
        ivc.add_output('rho_Fes',    val=0.0, units='kg/m**3')
        ivc.add_output('rho_Fe',     val=0.0, units='kg/m**3')
        ivc.add_output('rho_Copper', val=0.0, units='kg/m**3')
        ivc.add_output('rho_PM',     val=0.0, units='kg/m**3')
        
        ivc.add_output('C_Cu',  val=0.0, units='USD/kg')
        ivc.add_output('C_Fe',  val=0.0, units='USD/kg')
        ivc.add_output('C_Fes', val=0.0, units='USD/kg')
        ivc.add_output('C_PM',  val=0.0, units='USD/kg')
        
        self.add_subsystem('ivc', ivc, promotes=['*'])

        if topLevelFlag:
            sivc = om.IndepVarComp()
            sivc.add_output('machine_rating', 0.0, units='W')
            sivc.add_output('n_nom', 0.0, units='rpm')
            sivc.add_output('T_rated', 0.0, units='N*m')
            self.add_subsystem('sivc', sivc, promotes=['*'])
        
        # Add generator design component and cost
        if genType.lower() == 'scig':
            mygen = gm.SCIG
            
        elif genType.lower() == 'dfig':
            mygen = gm.DFIG
            
        elif genType.lower() == 'eesg':
            mygen = gm.EESG
            
        elif genType.lower() == 'pmsg_arms':
            mygen = gm.PMSG_Arms
            
        elif genType.lower() == 'pmsg_disc':
            mygen = gm.PMSG_Disc
            
        elif genType.lower() == 'pmsg_outer':
            mygen = gm.PMSG_Outer
            
        self.add_subsystem('generator', mygen(), promotes=['*'])
        self.add_subsystem('gen_cost', Cost(), promotes=['*'])
        self.add_subsystem('constr', Constraints(), promotes=['*'])
            

def optimization_example(genType, exportFlag=False):
    genType = genType.lower()
    
    #Example optimization of a generator for costs on a 5 MW reference turbine
    prob=om.Problem()
    prob.model = Generator(design=genType, topLevelFlag=True)
    
    # add optimizer and set-up problem (using user defined input on objective function)
    '''
    prob.driver = om.pyOptSparseDriver() #ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'CONMIN'
    prob.driver.opt_settings['IPRINT'] = 4
    prob.driver.opt_settings['ITRM'] = 3
    prob.driver.opt_settings['ITMAX'] = 10
    prob.driver.opt_settings['DELFUN'] = 1e-3
    prob.driver.opt_settings['DABFUN'] = 1e-3
    prob.driver.opt_settings['IFILE'] = 'CONMIN_'+genType.upper()+'.out'
    '''
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    
    # Specificiency target efficiency(%)
    Eta_Target = 93.0

    eps = 1e-6
    
    # Set up design variables and bounds for a SCIG designed for a 5MW turbine
    if genType in ['dfig', 'scig']:
        # Design variables
        prob.model.add_design_var('r_s',     lower=0.2,  upper=1.0)
        prob.model.add_design_var('len_s',   lower=0.4,  upper=2.0)
        prob.model.add_design_var('h_s',     lower=0.04, upper=0.1)
        prob.model.add_design_var('h_r',     lower=0.04, upper=0.1)
        prob.model.add_design_var('B_symax', lower=1.0,  upper=2.0-eps)

        # Constraints
        prob.model.add_constraint('Overall_eff',        lower=Eta_Target)
        prob.model.add_constraint('E_p',                lower=500.0+eps, upper=5000.0-eps)
        prob.model.add_constraint('TCr',                 lower=0.0+eps)
        prob.model.add_constraint('TCs',                 lower=0.0+eps)
        prob.model.add_constraint('B_g',                lower=0.7,       upper=1.2)
        prob.model.add_constraint('B_trmax',                             upper=2.0-eps)
        prob.model.add_constraint('B_tsmax',                             upper=2.0-eps)
        prob.model.add_constraint('A_1',                                 upper=60000.0-eps)
        prob.model.add_constraint('J_s',                                 upper=6.0)
        prob.model.add_constraint('J_r',                                 upper=6.0)
        prob.model.add_constraint('Slot_aspect_ratio1', lower=4.0,       upper=10.0)

    if genType == 'scig':
        prob.model.add_design_var('I_0',       lower=5.0, upper=200.0)
        prob.model.add_constraint('B_rymax',   upper=2.0-eps)
        prob.model.add_constraint('K_rad_L',   lower=0.0)
        prob.model.add_constraint('K_rad_U',   upper=0.0)
        prob.model.add_constraint('D_ratio_L', lower=0.0)
        prob.model.add_constraint('D_ratio_U', upper=0.0)
        
    if genType == 'dfig':
        prob.model.add_design_var('I_0',          lower=5.,   upper=100.0)
        prob.model.add_design_var('S_Nmax',       lower=-0.3, upper=-0.1)
        prob.model.add_constraint('K_rad',        lower=0.2,  upper=1.5)
        prob.model.add_constraint('D_ratio',      lower=1.37, upper=1.4)
        prob.model.add_constraint('Current_ratio',lower=0.1,  upper=0.3)

    if genType == 'eesg':
        # Design variables
        prob.model.add_design_var('r_s',    lower=0.5,   upper=9.0)
        prob.model.add_design_var('len_s',  lower=0.5,   upper=2.5)
        prob.model.add_design_var('h_s',    lower=0.06,  upper=0.15)
        prob.model.add_design_var('tau_p',  lower=0.04,  upper=0.2)
        prob.model.add_design_var('N_f',    lower=10,    upper=300)
        prob.model.add_design_var('I_f',    lower=1,     upper=500)
        prob.model.add_design_var('n_r',    lower=5.0,   upper=15.0)
        prob.model.add_design_var('h_yr',   lower=0.01,  upper=0.25)
        prob.model.add_design_var('h_ys',   lower=0.01,  upper=0.25)
        prob.model.add_design_var('b_arm',    lower=0.1,   upper=1.5)
        prob.model.add_design_var('d_r',    lower=0.1,   upper=1.5)
        prob.model.add_design_var('t_wr',   lower=0.001, upper=0.2)
        prob.model.add_design_var('n_s',    lower=5.0,   upper=15.0)
        prob.model.add_design_var('b_st',   lower=0.1,   upper=1.5)
        prob.model.add_design_var('d_s',    lower=0.1,   upper=1.5)
        prob.model.add_design_var('t_ws',   lower=0.001, upper=0.2)

        # Constraints
        prob.model.add_constraint('B_gfm',       lower=0.617031, upper=1.057768)
        prob.model.add_constraint('B_pc',        upper=2.0)
        prob.model.add_constraint('E_s',         lower=500.0, upper=5000.0)
        prob.model.add_constraint('J_f',         upper=6.0)
        prob.model.add_constraint('n_brushes',   upper=6)
        prob.model.add_constraint('Power_ratio', upper=2-eps)
        
    if genType in ['pmsg_outer']:
        # Design variables
        prob.model.add_design_var('r_g', lower=3.0, upper=5 ) 
        prob.model.add_design_var('len_s', lower=1.5, upper=3.5 )  
        prob.model.add_design_var('h_s', lower=0.1, upper=1.00 )  
        prob.model.add_design_var('p', lower=50.0, upper=100)
        prob.model.add_design_var('h_m', lower=0.005, upper=0.2 )  
        prob.model.add_design_var('h_yr', lower=0.035, upper=0.22 )
        prob.model.add_design_var('h_ys', lower=0.035, upper=0.22 )
        prob.model.add_design_var('B_tmax', lower=1, upper=2.0 ) 
        prob.model.add_design_var('t_r', lower=0.05, upper=0.3 ) 
        prob.model.add_design_var('t_s', lower=0.05, upper=0.3 )  
        prob.model.add_design_var('h_ss', lower=0.04, upper=0.2)
        prob.model.add_design_var('h_sr', lower=0.04, upper=0.2)

        # Constraints
        prob.model.add_constraint('B_symax',lower=0.0,upper=2.0)
        prob.model.add_constraint('B_rymax',lower=0.0,upper=2.0)
        prob.model.add_constraint('b_t',    lower=0.01)
        prob.model.add_constraint('B_g', lower=0.7,upper=1.3)
        #prob.model.add_constraint('E_p',    lower=500, upper=10000)
        prob.model.add_constraint('A_Cuscalc',lower=5.0,upper=500)
        prob.model.add_constraint('K_rad',    lower=0.15,upper=0.3)
        prob.model.add_constraint('Slot_aspect_ratio',lower=4.0, upper=10.0)
        prob.model.add_constraint('gen_eff',lower=93.)
        prob.model.add_constraint('A_1',upper=95000.0)
        prob.model.add_constraint('T_e', lower= 10.26812e6,upper=10.3e6)
        prob.model.add_constraint('J_actual',lower=3,upper=6)    
        prob.model.add_constraint('con_uar',lower = 1e-2)
        prob.model.add_constraint('con_yar', lower = 1e-2)
        prob.model.add_constraint('con_uas', lower = 1e-2)
        prob.model.add_constraint('con_yas', lower = 1e-2)   

    if genType in ['pmsg_arms','pmsg_disc']:
        # Design variables
        prob.model.add_design_var('r_s',   lower=0.5,   upper=9.0)
        prob.model.add_design_var('len_s', lower=0.5,   upper=2.5)
        prob.model.add_design_var('h_s',   lower=0.04,  upper=0.1)
        prob.model.add_design_var('tau_p', lower=0.04,  upper=0.1)
        prob.model.add_design_var('h_m',   lower=0.005, upper=0.1)
        prob.model.add_design_var('n_r',   lower=5.0,   upper=15.0)
        prob.model.add_design_var('h_yr',  lower=0.045, upper=0.25)
        prob.model.add_design_var('h_ys',  lower=0.045, upper=0.25)
        prob.model.add_design_var('n_s',   lower=5.0,   upper=15.0)
        prob.model.add_design_var('b_st',  lower=0.1,   upper=1.5)
        prob.model.add_design_var('d_s',   lower=0.1,   upper=1.5)
        prob.model.add_design_var('t_ws',  lower=0.001, upper=0.2)
    
        prob.model.add_constraint('con_Bsmax', lower=0.0+eps)
        prob.model.add_constraint('E_p', lower=500.0, upper=5000.0)

    if genType == 'pmsg_arms':
        prob.model.add_design_var('b_arm',  lower=0.1,   upper=1.5)
        prob.model.add_design_var('d_r',  lower=0.1,   upper=1.5)
        prob.model.add_design_var('t_wr', lower=0.001, upper=0.2)
        
    if genType == 'pmsg_disc':
        prob.model.add_design_var('t_d', lower=0.1, upper=0.25)
        
    if genType in ['eesg', 'pmsg_arms', 'pmsg_disc']:
        prob.model.add_constraint('B_symax', upper=2.0-eps)
        prob.model.add_constraint('B_rymax', upper=2.0-eps)
        prob.model.add_constraint('B_tmax',  upper=2.0-eps)
        prob.model.add_constraint('B_g',     lower=0.7, upper=1.2)
        prob.model.add_constraint('con_uas', lower=0.0+eps)
        prob.model.add_constraint('con_zas', lower=0.0+eps)
        prob.model.add_constraint('con_yas', lower=0.0+eps)
        prob.model.add_constraint('con_uar', lower=0.0+eps)
        prob.model.add_constraint('con_yar', lower=0.0+eps)
        prob.model.add_constraint('con_TC2r', lower=0.0+eps)
        prob.model.add_constraint('con_TC2s', lower=0.0+eps)
        prob.model.add_constraint('con_bst', lower=0.0-eps)
        prob.model.add_constraint('A_1', upper=60000.0-eps)
        prob.model.add_constraint('J_s', upper=6.0)
        prob.model.add_constraint('A_Cuscalc', lower=5.0, upper=300)
        prob.model.add_constraint('K_rad', lower=0.2+eps, upper=0.27)
        prob.model.add_constraint('Slot_aspect_ratio', lower=4.0, upper=10.0)
        prob.model.add_constraint('gen_eff', lower=Eta_Target)

    if genType in ['eesg', 'pmsg_arms']:
        prob.model.add_constraint('con_zar', lower=0.0+eps)
        prob.model.add_constraint('con_br', lower=0.0+eps)
    
    
    Objective_function = 'Costs'
    prob.model.add_objective(Objective_function, scaler=1e-5)
    prob.setup()
    
    # Specify Target machine parameters
    
    prob['machine_rating'] = 5000000.0
    
    if genType in ['scig', 'dfig']:
        prob['n_nom']              = 1200.0
        prob['Gearbox_efficiency'] = 0.955
        prob['cofi'] = 0.9
        prob['y_tau_p'] = 12./15.
        prob['sigma'] = 21.5e3
        
    elif genType in ['eesg', 'pmsg_arms','pmsg_disc']:
        prob['T_rated']             = 4.143289e6
        prob['n_nom']              = 12.1
        prob['sigma'] = 48.373e3

        
    if genType == 'scig':
        #prob['r_s']     = 0.55 #0.484689156353 #0.55 #meter
        prob['rad_ag']  = 0.55 #0.484689156353 #0.55 #meter
        prob['len_s']   = 1.30 #1.27480124244 #1.3 #meter
        prob['h_s']     = 0.090 #0.098331868116 # 0.090 #meter
        prob['h_r']     = 0.050 #0.04 # 0.050 #meter
        prob['I_0']     = 140  #139.995232826 #140  #Ampere
        prob['B_symax'] = 1.4 #1.86140258387 #1.4 #Tesla
        prob['q1']      = 6
        
    elif genType == 'dfig':
        #prob['r_s']     = 0.61 #0.493167295965 #0.61 #meter
        prob['rad_ag']  = 0.61 #0.493167295965 #0.61 #meter
        prob['len_s']   = 0.49 #1.06173588215 #0.49 #meter
        prob['h_s']     = 0.08 #0.1 # 0.08 #meter
        prob['h_r']     = 0.1 # 0.0998797703231 #0.1 #meter
        prob['I_0']     = 40.0 # 40.0191207049 #40.0 #Ampere
        prob['B_symax'] = 1.3 #1.59611292026 #1.3 #Tesla
        prob['S_Nmax']  = -0.2 #-0.3 #-0.2
        prob['k_fillr']  = 0.55
        prob['q1']      = 5

    elif genType == 'eesg':
        # Initial design variables 
        #prob['r_s']     = 3.2
        prob['rad_ag']  = 3.2
        prob['len_s']   = 1.4
        prob['h_s']     = 0.060
        prob['tau_p']   = 0.170
        prob['I_f']     = 69
        prob['N_f']     = 100
        prob['h_ys']    = 0.130
        prob['h_yr']    = 0.120
        prob['n_s']     = 5
        prob['b_st']    = 0.470
        prob['n_r']     = 5
        prob['b_r']     = 0.480
        prob['d_r']     = 0.510
        prob['d_s']     = 0.400
        prob['t_wr']    = 0.140
        prob['t_ws']    = 0.070
        prob['R_o']     = 0.43      #10MW: 0.523950817,#5MW: 0.43, #3MW:0.363882632 #1.5MW: 0.2775  0.75MW: 0.17625
        prob['q1']      = 2

    elif genType == 'pmsg_arms':
        #prob['r_s']     = 3.26
        prob['rad_ag']  = 3.26
        prob['len_s']   = 1.60
        prob['h_s']     = 0.070
        prob['tau_p']   = 0.080
        prob['h_m']     = 0.009
        prob['h_ys']    = 0.075
        prob['h_yr']    = 0.075
        prob['n_s']     = 5.0
        prob['b_st']    = 0.480
        prob['n_r']     = 5.0
        prob['b_r']     = 0.530
        prob['d_r']     = 0.700
        prob['d_s']     = 0.350
        prob['t_wr']    = 0.06
        prob['t_ws']    = 0.06
        prob['R_o']     = 0.43           #0.523950817  #0.43  #0.523950817 #0.17625 #0.2775 #0.363882632 ##0.35 #0.523950817 #0.43 #523950817 #0.43 #0.523950817 #0.523950817 #0.17625 #0.2775 #0.363882632 #0.43 #0.523950817 #0.43
        prob['q1']      = 1

    elif genType == 'pmsg_disc':
        #prob['r_s']     = 3.49 #3.494618182
        prob['rad_ag']  = 3.49 #3.494618182
        prob['len_s']   = 1.5 #1.506103927
        prob['h_s']     = 0.06 #0.06034976
        prob['tau_p']   = 0.07 #0.07541515 
        prob['h_m']     = 0.0105 #0.0090100202 
        prob['h_ys']    = 0.085 #0.084247994 #
        prob['h_yr']    = 0.055 #0.0545789687
        prob['n_s']     = 5.0 #5.0
        prob['b_st']    = 0.460 #0.46381
        prob['t_d']     = 0.105 #0.10 
        prob['d_s']     = 0.350 #0.35031 #
        prob['t_ws']    = 0.150 #=0.14720 #
        prob['R_o']     = 0.43 #0.43
        prob['q1']      = 1

    elif genType == 'pmsg_outer':
        prob['machine_rating'] = 10.321e6
        prob['T_rated']        = 10.25e6       #rev 1 9.94718e6
        prob['P_mech']         = 10.71947704e6 #rev 1 9.94718e6
        prob['n_nom']          = 10            #8.68                # rpm 9.6
        prob['r_g']            = 4.0           # rev 1  4.92
        prob['len_s']          = 1.7           # rev 2.3
        prob['h_s']            = 0.7            # rev 1 0.3
        prob['p']              = 70            #100.0    # rev 1 160
        prob['h_m']            = 0.005         # rev 1 0.034
        prob['h_ys']           = 0.04          # rev 1 0.045
        prob['h_yr']           = 0.06          # rev 1 0.045
        prob['b']              = 2.
        prob['c']              = 5.0
        prob['B_tmax']         = 1.9
        prob['E_p']            = 3300/np.sqrt(3)
        prob['R_no']           = 1.1             # Nose outer radius
        prob['R_sh']           = 1.34            # Shaft outer radius =(2+0.25*2+0.3*2)*0.5
        prob['t_r']            = 0.05          # Rotor disc thickness
        prob['h_sr']           = 0.04          # Rotor cylinder thickness
        prob['t_s']            = 0.053         # Stator disc thickness
        prob['h_ss']           = 0.04          # Stator cylinder thickness
        prob['y_sh']           = 0.0005*0      # Shaft deflection
        prob['theta_sh']       = 0.00026*0     # Slope at shaft end
        prob['y_bd']           = 0.0005*0      # deflection at bedplate
        prob['theta_bd']       = 0.00026*0      # Slope at bedplate end
        prob['u_allow_pcent']  = 8.5            # % radial deflection
        prob['y_allow_pcent']  = 1.0            # % axial deflection
        prob['z_allow_deg']    = 0.05           # torsional twist
        prob['sigma']          = 60.0e3         # Shear stress
        prob['B_r']            = 1.279
        prob['ratio_mw2pp']    = 0.8
        prob['h_0']            = 5e-3
        prob['h_w']            = 4e-3
        prob['k_fes']          = 0.8

    #----------------- try 15MW PMSG_disc --------------
    #  testing 2019 11 04
    
    if genType in ['eesg', 'pmsg_arms','pmsg_disc']:
        prob['machine_rating'] = 15000000.0
        prob['T_rated']         = 20.64e6
        prob['n_nom']          = 7.54
        #prob['machine_rating'] = 10000000.0
        #prob['T_rated']         = 12.64e6
        
    #---------------------------------------------------
        
    # Specific costs
    prob['C_Cu']         = 4.786         # Unit cost of Copper $/kg
    prob['C_Fe']         = 0.556         # Unit cost of Iron $/kg
    prob['C_Fes']        = 0.50139       # specific cost of Structural_mass $/kg
    prob['C_PM']         =   95.0
    
    #Material properties
    prob['rho_Fe']       = 7700.0        # Steel density Kg/m3
    prob['rho_Fes']      = 7850          # structural Steel density Kg/m3
    prob['rho_Copper']   = 8900.0        # copper density Kg/m3
    prob['rho_PM']       = 7450.0        # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]
            
    prob['shaft_cm']     = np.zeros(3)
    prob['shaft_length'] = 2.0
    
    #Run optimization
    prob.model.approx_totals()
    prob.run_model()
    prob.model.list_inputs(units=True) #values = False, hierarchical=False)
    prob.model.list_outputs(units=True) #values = False, hierarchical=False)    
    fio.save_data(genType.upper(), prob, npz_file=False, mat_file=False, xls_file=True)
            
if __name__=='__main__':
    
    # Run example optimizations for all generator types
    #for m in ['eesg',]:
    #for m in ['pmsg_disc']:
    for m in ['scig','dfig','eesg','pmsg_arms','pmsg_disc','pmsg_outer']:
        print('Running, '+m)
        optimization_example(m, exportFlag=True)
