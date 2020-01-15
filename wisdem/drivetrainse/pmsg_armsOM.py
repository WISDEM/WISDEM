"""PMSG_arms.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis 

The OpenMDAO part of a permanent-magnet synchronous generator module.
The pure-python part is in pmsg_discSE.py
"""

from openmdao.api import Group, Problem, ExplicitComponent,ExecComp,IndepVarComp,ScipyOptimizeDriver
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan
from pmsg_armsSE import PMSG_Arms

class PMSG_Arms_OM(ExplicitComponent):
    """ Estimates overall mass dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
        
        # PMSG_arms generator design inputs
        #self.add_input('r_s',           val=0.0, units='m',   desc='airgap radius r_s')
        self.add_input('rad_ag',         val=0.0, units='m',   desc='airgap radius')
        self.add_input('len_s',          val=0.0, units='m',   desc='Stator core length l_s')
        self.add_input('h_s',            val=0.0, units='m',   desc='Yoke height h_s')
        self.add_input('tau_p',          val=0.0, units='m',   desc='Pole pitch self.tau_p')
        self.add_input('machine_rating', val=0.0, units='W',   desc='Machine rating')
        self.add_input('n_nom',          val=0.0, units='rpm', desc='rated speed')
        self.add_input('Torque',         val=0.0, units='N*m', desc='Rated torque ')
        self.add_input('h_m',            val=0.0, units='m',   desc='magnet height')
        self.add_input('h_ys',           val=0.0, units='m',   desc='Yoke height')
        self.add_input('h_yr',           val=0.0, units='m',   desc='rotor yoke height')
        
        # structural design variables
        self.add_input('n_s',  val=0.0,            desc='number of stator arms n_s')
        self.add_input('b_st', val=0.0, units='m', desc='arm width b_st')
        self.add_input('d_s',  val=0.0, units='m', desc='arm depth d_s')
        self.add_input('t_ws', val=0.0, units='m', desc='arm depth thickness self.t_ws')
        self.add_input('n_r',  val=0.0,            desc='number of arms n')
        self.add_input('b_r',  val=0.0, units='m', desc='arm width b_r')
        self.add_input('d_r',  val=0.0, units='m', desc='arm depth d_r')
        self.add_input('t_wr', val=0.0, units='m', desc='arm depth thickness self.t_wr')
        self.add_input('R_o',  val=0.0, units='m', desc='Shaft radius')
        
        # PMSG_arms generator design outputs
        
        # Magnetic loading
        self.add_output('B_symax', val=0.0, desc='Peak Stator Yoke flux density B_ymax')
        self.add_output('B_tmax',  val=0.0, desc='Peak Teeth flux density')
        self.add_output('B_rymax', val=0.0, desc='Peak Rotor yoke flux density')
        self.add_output('B_smax',  val=0.0, desc='Peak Stator flux density')
        self.add_output('B_pm1',   val=0.0, desc='Fundamental component of peak air gap flux density')
        self.add_output('B_g',     val=0.0, desc='Peak air gap flux density B_g')
        
        # Stator design
        self.add_output('N_s',       val=0.0, desc='Number of turns in the stator winding')
        self.add_output('b_s',       val=0.0, desc='slot width')
        self.add_output('b_t',       val=0.0, desc='tooth width')
        self.add_output('A_Cuscalc', val=0.0, desc='Conductor cross-section mm^2')
        self.add_output('S'       ,  val=0.0, desc='Stator slots')
        
        # Rotor magnet dimension
        self.add_output('b_m', val=0.0, desc='magnet width')
        self.add_output('p',   val=0.0, desc='No of pole pairs')
        
        # Electrical performance
        self.add_output('E_p', val=0.0, desc='Stator phase voltage')
        self.add_output('f',   val=0.0, desc='Generator output frequency')
        self.add_output('I_s', val=0.0, desc='Generator output phase current')
        self.add_output('R_s', val=0.0, desc='Stator resistance')
        self.add_output('L_s', val=0.0, desc='Stator synchronising inductance')
        self.add_output('A_1', val=0.0, desc='Electrical loading')
        self.add_output('J_s', val=0.0, desc='Current density')
        
        # Objective functions
        self.add_output('Mass',    val=0.0, desc='Actual mass')
        self.add_output('K_rad',   val=0.0, desc='K_rad')
        self.add_output('Losses',  val=0.0, desc='Total loss')
        self.add_output('gen_eff', val=0.0, desc='Generator efficiency')
        
        # Structural performance
        self.add_output('u_Ar',    val=0.0, desc='Rotor radial deflection')
        self.add_output('y_Ar',    val=0.0, desc='Rotor axial deflection')
        self.add_output('z_A_r',   val=0.0, desc='Rotor circumferential deflection')      
        self.add_output('u_As',    val=0.0, desc='Stator radial deflection')
        self.add_output('y_As',    val=0.0, desc='Stator axial deflection')
        self.add_output('z_A_s',   val=0.0, desc='Stator circumferential deflection')  
        self.add_output('u_all_r', val=0.0, desc='Allowable radial rotor')
        self.add_output('u_all_s', val=0.0, desc='Allowable radial stator')
        self.add_output('y_all',   val=0.0, desc='Allowable axial')
        self.add_output('z_all_s', val=0.0, desc='Allowable circum stator')
        self.add_output('z_all_r', val=0.0, desc='Allowable circum rotor')
        self.add_output('b_all_s', val=0.0, desc='Allowable arm')
        self.add_output('b_all_r', val=0.0, desc='Allowable arm dimensions')
        self.add_output('TC1',     val=0.0, desc='Torque constraint')
        self.add_output('TC2',     val=0.0, desc='Torque constraint-rotor')
        self.add_output('TC3',     val=0.0, desc='Torque constraint-stator')
        
        # Other parameters
        self.add_output('R_out', val=0.0, desc='Outer radius')
        
        self.add_output('Slot_aspect_ratio', val=0.0, desc='Slot aspect ratio')
        
        # Mass Outputs
        self.add_output('mass_PM',         val=0.0, units='kg', desc='Magnet mass')
        self.add_output('Copper',          val=0.0, units='kg', desc='Copper Mass')
        self.add_output('Iron',            val=0.0, units='kg', desc='Electrical Steel Mass')
        self.add_output('Structural_mass', val=0.0, units='kg', desc='Structural Mass')
        
        # Material properties
        self.add_input('rho_Fes',    val=0.0, units='kg*m**-3', desc='Structural Steel density ')
        self.add_input('rho_Fe',     val=0.0, units='kg*m**-3', desc='Magnetic Steel density ')
        self.add_input('rho_Copper', val=0.0, units='kg*m**-3', desc='Copper density ')
        self.add_input('rho_PM',     val=0.0, units='kg*m**-3', desc='Magnet density ')
        
        # inputs/outputs for interface with drivese    
        self.add_input('shaft_cm',     val= np.zeros(3), units='m', desc='Main Shaft CM')
        self.add_input('shaft_length', val=0.0,          units='m', desc='main shaft length')
        self.add_output('I',           val=np.zeros(3),             desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('cm',          val=np.zeros(3),             desc='COM [x,y,z]')
        
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack inputs
        #r_s               = inputs['r_s']
        rad_ag            = inputs['rad_ag']
        len_s             = inputs['len_s']
        h_s               = inputs['h_s']
        tau_p             = inputs['tau_p']
        h_m               = inputs['h_m']
        h_ys              = inputs['h_ys']
        h_yr              = inputs['h_yr']
        machine_rating    = inputs['machine_rating']
        n_nom             = inputs['n_nom']
        Torque            = inputs['Torque']
        
        b_st              = inputs['b_st']
        d_s               = inputs['d_s']
        t_ws              = inputs['t_ws']
        n_r               = inputs['n_r']
        n_s               = inputs['n_s']
        b_r               = inputs['b_r']
        d_r               = inputs['d_r']
        t_wr              = inputs['t_wr']
    
        R_o               = inputs['R_o']
        rho_Fe            = inputs['rho_Fe']
        rho_Copper        = inputs['rho_Copper']
        rho_Fes           = inputs['rho_Fes']
        rho_PM            = inputs['rho_PM']
        shaft_cm          = inputs['shaft_cm']
        shaft_length      = inputs['shaft_length']

        pmsg_arms = PMSG_Arms()
		
        (outputs['B_symax'], outputs['B_tmax'], outputs['B_rymax'], outputs['B_smax'], outputs['B_pm1'], outputs['B_g'], outputs['N_s'], 
		outputs['b_s'], outputs['b_t'], outputs['A_Cuscalc'], outputs['b_m'], outputs['p'], outputs['E_p'], outputs['f'], outputs['I_s'], 
		outputs['R_s'], outputs['L_s'], outputs['A_1'], outputs['J_s'], outputs['Losses'], outputs['K_rad'], outputs['gen_eff'], 
		outputs['S'], outputs['Slot_aspect_ratio'], outputs['Copper'], outputs['Iron'], outputs['u_Ar'], outputs['y_Ar'], outputs['z_A_r'], 
		outputs['u_As'], outputs['y_As'], outputs['z_A_s'], outputs['u_all_r'], outputs['u_all_s'], outputs['y_all'], outputs['z_all_s'], 
		outputs['z_all_r'], outputs['b_all_s'], outputs['b_all_r'], outputs['TC1'], outputs['TC2'], outputs['TC3'], outputs['R_out'], 
		outputs['Structural_mass'], outputs['Mass'], outputs['mass_PM'], outputs['cm'], outputs['I']) \
            = pmsg_arms.compute(rad_ag, len_s, h_s, tau_p, h_m, h_ys, h_yr, machine_rating, n_nom, Torque,          
                                b_st, d_s, t_ws, n_r, n_s, b_r, d_r, t_wr,
								R_o, rho_Fe, rho_Copper, rho_Fes, rho_PM, shaft_cm, shaft_length)

