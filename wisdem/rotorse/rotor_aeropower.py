#!/usr/bin/env python
# encoding: utf-8
"""
rotor.py

Created by Andrew Ning on 2012-02-28.
Copyright (c)  NREL. All rights reserved.
"""

from __future__ import print_function
import numpy as np
import os
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import PchipInterpolator

# from wisdem.ccblade.ccblade_component import CCBladeGeometry, CCBladePower
from wisdem.ccblade import CCAirfoil, CCBlade

from wisdem.commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from wisdem.commonse.utilities import vstack, trapz_deriv, linspace_with_deriv, smooth_min, smooth_abs
from wisdem.commonse.environment import PowerWind
# from wisdem.commonse.akima import Akima
# from wisdem.rotorse import RPM2RS, RS2RPM
# from wisdem.rotorse.rotor_geometry import RotorGeometry
# from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.rotorse.rotor_fast import eval_unsteady

import time
# ---------------------
# Components
# ---------------------



class OutputsAero(ExplicitComponent):
    def initialize(self):
        self.options.declare('npts_coarse_power_curve')
    
    def setup(self):
        npts_coarse_power_curve = self.options['npts_coarse_power_curve']

        # --- outputs ---
        self.add_input('rated_Omega_in', val=0.0, units='rpm', desc='rotor rotation speed at rated')
        self.add_input('rated_pitch_in', val=0.0, units='deg', desc='pitch setting at rated')
        self.add_input('rated_T_in', val=0.0, units='N', desc='rotor aerodynamic thrust at rated')
        self.add_input('rated_Q_in', val=0.0, units='N*m', desc='rotor aerodynamic torque at rated')

        self.add_input('V_extreme50', val=0.0, units='m/s', desc='survival wind speed')
        self.add_input('T_extreme_in', val=0.0, units='N', desc='thrust at survival wind condition')
        self.add_input('Q_extreme_in', val=0.0, units='N*m', desc='thrust at survival wind condition')

        # --- outputs ---
        self.add_output('V_extreme', val=0.0, units='m/s', desc='survival wind speed')
        self.add_output('T_extreme', val=0.0, units='N', desc='thrust at survival wind condition')
        self.add_output('Q_extreme', val=0.0, units='N*m', desc='thrust at survival wind condition')

        #self.declare_partials('V_extreme', 'V_extreme50')
        #self.declare_partials('T_extreme', 'T_extreme_in')
        #self.declare_partials('Q_extreme', 'Q_extreme_in')

    def compute(self, inputs, outputs):
        outputs['V_extreme'] = inputs['V_extreme50']
        outputs['T_extreme'] = inputs['T_extreme_in']
        outputs['Q_extreme'] = inputs['Q_extreme_in']
        '''
    def compute_partials(self, inputs, J):
        J['V_extreme', 'V_extreme50'] = 1
        J['T_extreme', 'T_extreme_in'] = 1
        J['Q_extreme', 'Q_extreme_in'] = 1
        '''

if __name__ == '__main__':


    tt = time.time()

    # Turbine Ontology input
    fname_input  = "turbine_inputs/nrel5mw_mod_update.yaml"
    # fname_output = "turbine_inputs/nrel5mw_mod_out.yaml"
    fname_schema = "turbine_inputs/IEAontology_schema.yaml"
    
    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose = True
    refBlade.NINPUT  = 5
    refBlade.NPTS    = 50
    refBlade.spar_var = ['Spar_Cap_SS', 'Spar_Cap_PS']
    refBlade.te_var   = 'TE_reinforcement'
    # refBlade.le_var   = 'LE_reinforcement'
    refBlade.validate     = False
    refBlade.fname_schema = fname_schema
    
    blade = refBlade.initialize(fname_input)
    rotor = Problem()
    npts_coarse_power_curve = 20 # (Int): number of points to evaluate aero analysis at
    npts_spline_power_curve = 2000  # (Int): number of points to use in fitting spline to power curve
    regulation_reg_II5      = True # calculate Region 2.5 pitch schedule, False will not maximize power in region 2.5
    regulation_reg_III      = True # calculate Region 3 pitch schedule, False will return erroneous Thrust, Torque, and Moment for above rated
    flag_Cp_Ct_Cq_Tables    = True # Compute Cp-Ct-Cq-Beta-TSR tables
    
    rotor.model = RotorAeroPower(RefBlade=blade,
                                 npts_coarse_power_curve=npts_coarse_power_curve,
                                 npts_spline_power_curve=npts_spline_power_curve,
                                 regulation_reg_II5=regulation_reg_II5,
                                 regulation_reg_III=regulation_reg_III,
                                 topLevelFlag=True)
    
    #rotor.setup(check=False)
    rotor.setup()
    rotor = Init_RotorAeropower_wRefBlade(rotor, blade)

    # === run and outputs ===
    rotor.run_driver()
    #rotor.check_partials(compact_print=True, step=1e-6, form='central')#, includes='*ae*')

    print('Run time = ', time.time()-tt)
    print('AEP =', rotor['AEP'])
    print('diameter =', rotor['diameter'])
    print('ratedConditions.V =', rotor['rated_V'])
    print('ratedConditions.Omega =', rotor['rated_Omega'])
    print('ratedConditions.pitch =', rotor['rated_pitch'])
    print('ratedConditions.T =', rotor['rated_T'])
    print('ratedConditions.Q =', rotor['rated_Q'])
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rotor['V'], rotor['P']/1e6)
    plt.xlabel('wind speed (m/s)')
    plt.xlabel('power (W)')
    plt.show()
    
    if flag_Cp_Ct_Cq_Tables:
        n_pitch = len(rotor['pitch_vector'])
        n_tsr   = len(rotor['tsr_vector'])
        n_U     = len(rotor['U_vector'])
        for i in range(n_U):
            fig0, ax0 = plt.subplots()
            CS0 = ax0.contour(rotor['pitch_vector'], rotor['tsr_vector'], rotor['Cp_aero_table'][:, :, i], levels=[0.0, 0.3, 0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50 ])
            ax0.clabel(CS0, inline=1, fontsize=12)
            plt.title('Power Coefficient', fontsize=14, fontweight='bold')
            plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
            plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            plt.subplots_adjust(bottom = 0.15, left = 0.15)

            fig0, ax0 = plt.subplots()
            CS0 = ax0.contour(rotor['pitch_vector'], rotor['tsr_vector'], rotor['Ct_aero_table'][:, :, i])
            ax0.clabel(CS0, inline=1, fontsize=12)
            plt.title('Thrust Coefficient', fontsize=14, fontweight='bold')
            plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
            plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            plt.subplots_adjust(bottom = 0.15, left = 0.15)

            
            fig0, ax0 = plt.subplots()
            CS0 = ax0.contour(rotor['pitch_vector'], rotor['tsr_vector'], rotor['Cq_aero_table'][:, :, i])
            ax0.clabel(CS0, inline=1, fontsize=12)
            plt.title('Torque Coefficient', fontsize=14, fontweight='bold')
            plt.xlabel('Pitch Angle [deg]', fontsize=14, fontweight='bold')
            plt.ylabel('TSR [-]', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(color=[0.8,0.8,0.8], linestyle='--')
            plt.subplots_adjust(bottom = 0.15, left = 0.15)
            
            plt.show()
   
    
