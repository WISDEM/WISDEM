"""DFIG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.api import Group, Problem, ExplicitComponent,ExecComp,IndepVarComp,ScipyOptimizeDriver
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan
from dfigSE import DFIG

class DFIG_OM(ExplicitComponent):
    """ Estimates overall mass, dimensions and Efficiency of DFIG generator. """
    
    def setup(self):
        # DFIG design inputs
        #self.add_input('r_s', val=0.0, units='m', desc='airgap radius r_s')
        self.add_input('rad_ag', val=0.0, units='m', desc='airgap radius')
        self.add_input('len_s',  val=0.0, units='m', desc='Stator core length l_s')
        self.add_input('h_s',    val=0.0, units='m', desc='Stator slot height')
        self.add_input('h_r',    val=0.0, units='m', desc='Rotor slot height')
        
        self.add_input('machine_rating', val=0.0, units='W',   desc='Machine rating')
        self.add_input('n_nom',          val=0.0, units='rpm', desc='rated speed')
        
        self.add_input('B_symax', val=0.0,            desc='Peak Stator Yoke flux density B_symax')
        self.add_input('S_Nmax',  val=0.0, units='T', desc='Max rated Slip ')
        self.add_input('I_0',     val=0.0, units='A', desc='no-load excitation current')
        
        self.add_input('shaft_cm',           val= np.zeros(3), units='m', desc='Main Shaft CM')
        self.add_input('shaft_length',       val=0.0,          units='m', desc='main shaft length')
        self.add_input('Gearbox_efficiency', val=0.0,                     desc='Gearbox efficiency')
        
        # Material properties
        self.add_input('rho_Fe',     val=0.0, units='kg*m**-3', desc='Magnetic Steel density ')
        self.add_input('rho_Copper', val=0.0, units='kg*m**-3', desc='Copper density ')
        
        # DFIG generator design output
        # Magnetic loading
        self.add_output('B_g',     val=0.0, desc='Peak air gap flux density B_g')
        self.add_output('B_g1',    val=0.0, desc='air gap flux density fundamental ')
        self.add_output('B_rymax', val=0.0, desc='maximum flux density in rotor yoke')
        self.add_output('B_tsmax', val=0.0, desc='maximum tooth flux density in stator')
        self.add_output('B_trmax', val=0.0, desc='maximum tooth flux density in rotor')
        
        #Stator design
        self.add_output('N_s',                val=0.0, desc='Stator turns')
        self.add_output('S',                  val=0.0, desc='Stator slots')
        self.add_output('h_ys',               val=0.0, desc='Stator Yoke height')
        self.add_output('b_s',                val=0.0, desc='stator slot width')
        self.add_output('b_t',                val=0.0, desc='stator tooth width')
        self.add_output('A_Cuscalc',          val=0.0, desc='Stator Conductor cross-section mm^2')
        self.add_output('Slot_aspect_ratio1', val=0.0, desc='Stator slot aspect ratio')
        self.add_output('D_ratio',            val=0.0, desc='Stator diameter ratio')
        self.add_output('q1',                 val=0.0, desc='Slots per pole per phase')
        
        #Rotor design
        self.add_output('N_r',                val=0.0, desc='Rotor turns')
        self.add_output('Q_r',                val=0.0, desc='Rotor slots')
        self.add_output('h_yr',               val=0.0, desc='rotor yoke height')
        self.add_output('b_r',                val=0.0, desc='rotor slot width')
        self.add_output('b_tr',               val=0.0, desc='rotor tooth width')
        self.add_output('A_Curcalc',          val=0.0, desc='Rotor Conductor cross-section mm^2')
        self.add_output('Slot_aspect_ratio2', val=0.0, desc='Rotor slot aspect ratio')
        self.add_output('b_trmin',            val=0.0, desc='minimum tooth width')
        self.add_output('tau_p',              val=0.0, desc='Pole pitch')
        self.add_output('p',                  val=0.0, desc='No of pole pairs')
        
        # Electrical performance
        self.add_output('E_p',  val=0.0, desc='Stator phase voltage')
        self.add_output('f',    val=0.0, desc='Generator output frequency')
        self.add_output('I_s',  val=0.0, desc='Generator output phase current')
        self.add_output('A_1' , val=0.0, desc='Specific current loading')
        self.add_output('J_s',  val=0.0, desc='Stator winding Current density')
        self.add_output('J_r',  val=0.0, desc='Rotor winding Current density')
        self.add_output('R_s',  val=0.0, desc='Stator resistance')
        self.add_output('R_R',  val=0.0, desc='Rotor resistance')
        self.add_output('L_r',  val=0.0, desc='Rotor inductance')
        self.add_output('L_s',  val=0.0, desc='Stator synchronising inductance')
        self.add_output('L_sm', val=0.0, desc='mutual inductance')
        
        # Objective functions
        self.add_output('Mass',    val=0.0, desc='Actual mass')
        self.add_output('K_rad',   val=0.0, desc='Stack length ratio')
        self.add_output('Losses',  val=0.0, desc='Total loss')
        self.add_output('gen_eff', val=0.0, desc='Generator efficiency')
        
        # Mass Outputs
        self.add_output('Copper',          val=0.0, units='kg', desc='Copper Mass')
        self.add_output('Iron',            val=0.0, units='kg', desc='Electrical Steel Mass')
        self.add_output('Structural_mass', val=0.0, units='kg', desc='Structural Mass')
        
        # Structural performance
        self.add_output('TC1', val=0.0, desc='Torque constraint-stator')
        self.add_output('TC2', val=0.0, desc='Torque constraint-rotor')
        
        # Other parameters
        self.add_output('Current_ratio', val=0.0,         desc='Rotor current ratio')
        self.add_output('Overall_eff',   val=0.0,         desc='Overall drivetrain efficiency')
        self.add_output('I',             val=np.zeros(3), desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('cm',            val=np.zeros(3), desc='COM [x,y,z]')
        
        self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack inputs
        rad_ag               = inputs['rad_ag']
        len_s                = inputs['len_s']
        h_s                  = inputs['h_s']
        h_r                  = inputs['h_r']
        I_0                  = inputs['I_0']
        
        machine_rating       = inputs['machine_rating']
        n_nom                = inputs['n_nom']
        Gearbox_efficiency   = inputs['Gearbox_efficiency']
        
        rho_Fe               = inputs['rho_Fe']
        rho_Copper           = inputs['rho_Copper']
        
        B_symax              = inputs['B_symax']
        S_Nmax               = inputs['S_Nmax']
        
        shaft_cm             = inputs['shaft_cm']
        shaft_length         = inputs['shaft_length']
        
        dfig = DFIG()
        
        (outputs['B_g'], outputs['B_g1'], outputs['B_rymax'], outputs['B_tsmax'], outputs['B_trmax'], outputs['q1'], 
        outputs['N_s'], outputs['S'], outputs['h_ys'], outputs['b_s'], outputs['b_t'], outputs['D_ratio'], 
        outputs['A_Cuscalc'], outputs['Slot_aspect_ratio1'], outputs['h_yr'], outputs['tau_p'], outputs['p'], 
        outputs['Q_r'], outputs['N_r'], outputs['b_r'], outputs['b_trmin'], outputs['b_tr'], outputs['A_Curcalc'],
        outputs['Slot_aspect_ratio2'], outputs['E_p'], outputs['f'], outputs['I_s'], outputs['A_1'], outputs['J_s'], 
        outputs['J_r'], outputs['R_s'], outputs['R_R'], outputs['L_r'], outputs['L_s'], outputs['L_sm'], 
        outputs['Mass'], outputs['K_rad'], outputs['Losses'], outputs['gen_eff'], outputs['Copper'], 
        outputs['Iron'], outputs['Structural_mass'], outputs['TC1'], outputs['TC2'], outputs['Current_ratio'], 
        outputs['Overall_eff'], outputs['cm'], outputs['I']) \
            = dfig.compute(rad_ag, len_s, h_s, h_r, I_0, machine_rating, n_nom, Gearbox_efficiency, 
            rho_Fe, rho_Copper, B_symax, S_Nmax, shaft_cm, shaft_length)
            
