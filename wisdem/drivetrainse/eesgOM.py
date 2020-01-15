"""eesgOM.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis 

The OpenMDAO part of an electrically excited generator module.
The pure-python part is in eesgSE.py
"""

from openmdao.api import Group, Problem, ExplicitComponent,ExecComp,IndepVarComp,ScipyOptimizeDriver
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan
import sys, os
from eesgSE import EESG

class EESG_OM(ExplicitComponent):
    """ Estimates overall mass dimensions and Efficiency of Electrically Excited Synchronous generator. """
    
    def setup(self):
        # EESG generator design inputs
        #self.add_input('r_s', val=0.0, units ='m', desc='airgap radius r_s')
        self.add_input('rad_ag', val=0.0, units ='m', desc='airgap radius')
        self.add_input('len_s',  val=0.0, units ='m', desc='Stator core length l_s')
        self.add_input('h_s',    val=0.0, units ='m', desc='Yoke height h_s')
        self.add_input('tau_p',  val=0.0, units ='m', desc='Pole pitch self.tau_p')
        
        self.add_input('machine_rating', val=0.0, units ='W',   desc='Machine rating')
        self.add_input('n_nom',          val=0.0, units ='rpm', desc='rated speed')
        self.add_input('Torque',         val=0.0, units ='N*m', desc='Rated torque ')
        self.add_input('I_f',            val=0.0, units='A',    desc='Excitation current')
        self.add_input('N_f',            val=0.0, units='A',    desc='field turns')
        self.add_input('h_ys',           val=0.0, units ='m',   desc='Yoke height')
        self.add_input('h_yr',           val=0.0, units ='m',   desc='rotor yoke height')
        
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
        
        # EESG generator design outputs
        
        # Magnetic loading
        self.add_output('B_symax', val=0.0, units='T', desc='Peak Stator Yoke flux density B_ymax')
        self.add_output('B_tmax',  val=0.0, units='T', desc='Peak Teeth flux density')
        self.add_output('B_rymax', val=0.0, units='T', desc='Peak Rotor yoke flux density')
        self.add_output('B_gfm',   val=0.0, units='T', desc='Average air gap flux density B_g')
        self.add_output('B_g',     val=0.0, units='T', desc='Peak air gap flux density B_g')
        self.add_output('B_pc',    val=0.0, units='T', desc='Pole core flux density')
        
        # Stator design
        self.add_output('N_s',       val=0.0,                desc='Number of turns in the stator winding')
        self.add_output('b_s',       val=0.0, units ='m',    desc='slot width')
        self.add_output('b_t',       val=0.0, units ='m',    desc='tooth width')
        self.add_output('A_Cuscalc', val=0.0, units='mm**2', desc='Stator Conductor cross-section mm^2')
        self.add_output('S',         val=0.0,                desc='Stator slots')
        
        # # Output parameters : Rotor design
        self.add_output('h_p',       val=0.0, units ='m',    desc='Pole height')
        self.add_output('b_p',       val=0.0, units ='m',    desc='Pole width')
        self.add_output('p',         val=0.0,                desc='No of pole pairs')
        self.add_output('n_brushes', val=0.0,                desc='number of brushes')
        self.add_output('A_Curcalc', val=0.0, units='mm**2', desc='Rotor Conductor cross-section')
        
        # Output parameters : Electrical performance
        self.add_output('E_s',            val=0.0, units='V',       desc='Stator phase voltage')
        self.add_output('f',              val=0.0,                  desc='Generator output frequency')
        self.add_output('I_s',            val=0.0, units='A',       desc='Generator output phase current')
        self.add_output('R_s',            val=0.0, units='ohm',     desc='Stator resistance')
        self.add_output('R_r',            val=0.0, units='ohm',     desc='Rotor resistance')
        self.add_output('L_m',            val=0.0, units='H',       desc='Stator synchronising inductance')
        self.add_output('J_s',            val=0.0, units='A*m**-2', desc='Stator Current density')
        self.add_output('J_f',            val=0.0, units='A*m**-2', desc='rotor Current density')
        self.add_output('A_1',            val=0.0,                  desc='Specific current loading')
        self.add_output('Load_mmf_ratio', val=0.0,                  desc='mmf_ratio')
        
        # Objective functions and output
        self.add_output('Mass',    val=0.0, units='kg',  desc='Actual mass')
        self.add_output('K_rad',   val=0.0,              desc='K_rad')
        self.add_output('Losses',  val=0.0, units='W',   desc='Total loss')
        self.add_output('gen_eff', val=0.0,              desc='Generator efficiency')   # units='pct'
        
        # Structural performance
        self.add_output('u_Ar',    val=0.0, units='m',    desc='Rotor radial deflection')
        self.add_output('y_Ar',    val=0.0, units='m',    desc='Rotor axial deflection')
        self.add_output('z_A_r',   val=0.0, units='m',    desc='Rotor circumferential deflection')
        self.add_output('u_As',    val=0.0, units='m',    desc='Stator radial deflection')
        self.add_output('y_As',    val=0.0, units='m',    desc='Stator axial deflection')
        self.add_output('z_A_s',   val=0.0, units='m',    desc='Stator circumferential deflection')  
        self.add_output('u_all_r', val=0.0, units='m',    desc='Allowable radial rotor')
        self.add_output('u_all_s', val=0.0, units='m',    desc='Allowable radial stator')
        self.add_output('y_all',   val=0.0, units='m',    desc='Allowable axial')
        self.add_output('z_all_s', val=0.0, units='m',    desc='Allowable circum stator')
        self.add_output('z_all_r', val=0.0, units='m',    desc='Allowable circum rotor')
        self.add_output('b_all_s', val=0.0, units='m',    desc='Allowable arm')
        self.add_output('b_all_r', val=0.0, units='m',    desc='Allowable arm dimensions')
        self.add_output('TC1',     val=0.0, units='m**3', desc='Torque constraint')
        self.add_output('TC2',     val=0.0, units='m**3', desc='Torque constraint-rotor')
        self.add_output('TC3',     val=0.0, units='m**3', desc='Torque constraint-stator')
        
        # Material properties
        self.add_input('rho_Fes',    val=0.0, units='kg*m**-3', desc='Structural Steel density ')
        self.add_input('rho_Fe',     val=0.0, units='kg*m**-3', desc='Magnetic Steel density ')
        self.add_input('rho_Copper', val=0.0, units='kg*m**-3', desc='Copper density ')
        
        # Mass Outputs
        self.add_output('Copper',          val=0.0, units='kg', desc='Copper Mass')
        self.add_output('Iron',            val=0.0, units='kg', desc='Electrical Steel Mass')
        self.add_output('Structural_mass', val=0.0, units='kg', desc='Structural Mass')
        
        # Other parameters
        self.add_output('Power_ratio',       val=0.0,             desc='Power_ratio')
        self.add_output('Slot_aspect_ratio', val=0.0,             desc='Stator slot aspect ratio')
        self.add_output('R_out',             val=0.0, units ='m', desc='Outer radius')
        
        # inputs/outputs for interface with drivese
        self.add_input('shaft_cm',     val=np.zeros(3), units='m',       desc='Main Shaft CM')
        self.add_input('shaft_length', val=0.0,         units='m',       desc='main shaft length')
        self.add_output('I',           val=np.zeros(3), units='kg*m**2', desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('cm',          val=np.zeros(3), units='m',       desc='COM [x,y,z]')

        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack outputs
        rad_ag            = inputs['rad_ag']
        len_s             = inputs['len_s']
        h_s               = inputs['h_s']
        tau_p             = inputs['tau_p']
        N_f               = inputs['N_f']
        I_f               = inputs['I_f']
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
        shaft_cm          = inputs['shaft_cm']
        shaft_length      = inputs['shaft_length']
        
        eesg = EESG()
        
        (outputs['B_symax'], outputs['B_tmax'], outputs['B_rymax'], outputs['B_gfm'], outputs['B_g'], outputs['B_pc'], 
        outputs['N_s'], outputs['b_s'], outputs['b_t'], outputs['A_Cuscalc'], outputs['A_Curcalc'], outputs['b_p'], 
        outputs['h_p'], outputs['p'], outputs['E_s'], outputs['f'], outputs['I_s'], outputs['R_s'], outputs['L_m'], 
        outputs['A_1'], outputs['J_s'], outputs['R_r'], outputs['Losses'], outputs['Load_mmf_ratio'], outputs['Power_ratio'], 
        outputs['n_brushes'], outputs['J_f'], outputs['K_rad'], outputs['gen_eff'], outputs['S'], 
        outputs['Slot_aspect_ratio'], outputs['Copper'], outputs['Iron'], outputs['u_Ar'], outputs['y_Ar'], 
        outputs['z_A_r'], outputs['u_As'], outputs['y_As'], outputs['z_A_s'], outputs['u_all_r'], outputs['u_all_s'], 
        outputs['y_all'], outputs['z_all_s'], outputs['z_all_r'], outputs['b_all_s'], outputs['b_all_r'], outputs['TC1'], 
        outputs['TC2'], outputs['TC3'], outputs['R_out'], outputs['Structural_mass'], outputs['Mass'], outputs['cm'], outputs['I']) \
         = eesg.compute(rad_ag, len_s, h_s, tau_p, N_f, I_f, h_ys, h_yr, machine_rating, n_nom, Torque,               
                    b_st, d_s, t_ws, n_r, n_s, b_r, d_r, t_wr, R_o, rho_Fe, rho_Copper, rho_Fes, shaft_cm, shaft_length)
        
'''
        outputs['B_symax']           = B_symax
        outputs['B_tmax']            = B_tmax
        outputs['B_rymax']           = B_rymax
        outputs['B_gfm']             = B_gfm
        outputs['B_g']               = B_g
        outputs['B_pc']              = B_pc
        outputs['N_s']               = N_s
        outputs['b_s']               = b_s

        outputs['b_t']               = b_t
        outputs['A_Cuscalc']         = A_Cuscalc
        outputs['A_Curcalc']         = A_Curcalc
        outputs['b_p']               = b_p
        outputs['h_p']               = h_p
        outputs['p']                 = p
        outputs['E_s']               = E_s
        outputs['f']                 = f

        outputs['I_s']               = I_s
        outputs['R_s']               = R_s
        outputs['L_m']               = L_m
        outputs['A_1']               = A_1
        outputs['J_s']               = J_s
        outputs['R_r']               = R_r
        outputs['Losses']            = Losses

        outputs['Load_mmf_ratio']    = Load_mmf_ratio
        outputs['Power_ratio']       = Power_ratio
        outputs['n_brushes']         = n_brushes
        outputs['J_f']               = J_f
        outputs['K_rad']             = K_rad
        outputs['gen_eff']           = gen_eff
        outputs['S']                 = S

        outputs['Slot_aspect_ratio'] = Slot_aspect_ratio
        outputs['Copper']            = Copper
        outputs['Iron']              = Iron
        outputs['u_Ar']              = u_Ar
        outputs['y_Ar']              = y_Ar

        outputs['z_A_r']             = z_A_r
        outputs['u_As']              = u_As
        outputs['y_As']              = y_As
        outputs['z_A_s']             = z_A_s
        outputs['u_all_r']           = u_all_r
        outputs['u_all_s']           = u_all_s

        outputs['y_all']             = y_all
        outputs['z_all_s']           = z_all_s
        outputs['z_all_r']           = z_all_r
        outputs['b_all_s']           = b_all_s
        outputs['b_all_r']           = b_all_r
        outputs['TC1']               = TC1

        outputs['TC2']               = TC2
        outputs['TC3']               = TC3
        outputs['R_out']             = R_out
        outputs['Structural_mass']   = Structural_mass
        outputs['Mass']              = Mass
        outputs['cm']                = cm
        outputs['I']                 = I
        
B_symax, B_tmax, B_rymax, B_gfm, B_g, B_pc, N_s, b_s, b_t, A_Cuscalc, A_Curcalc, b_p, 
h_p, p, E_s, f, I_s, R_s, L_m, A_1, J_s, R_r, Losses, Load_mmf_ratio, Power_ratio, 
n_brushes, J_f, K_rad, gen_eff, S, Slot_aspect_ratio, Copper, Iron, u_Ar, y_Ar, 
z_A_r, u_As, y_As, z_A_s, u_all_r, u_all_s, y_all, z_all_s, z_all_r, b_all_s, b_all_r, TC1, 
TC2, TC3, R_out, Structural_mass, Mass, cm, I
'''
