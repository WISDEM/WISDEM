'''
Controller tuning script.

Nikhar J. Abbas
January 2020
'''

import numpy as np
import datetime
from scipy import interpolate, gradient

from wisdem.ccblade import CCAirfoil, CCBlade
from wisdem.rotorse import rotor_aeropower

from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import utilities as ROSCO_utilities

class TuneROSCO(ExplicitComponent):
    def initialize(self):
        self.options.declare('wt_init_options')

    def setup(self):
        wt_init_options = self.options['wt_init_options']

        # Input parameters that we need from a .yaml file (set to none for now...)
        self.controller_params{}
        # Controller Flags
        self.controller_params['LoggingLevel'] = None
        self.controller_params['F_LPFType'] = None
        self.controller_params['F_NotchType'] = None
        self.controller_params['IPC_ControlMode'] = None
        self.controller_params['VS_ControlMode'] = None
        self.controller_params['PC_ControlMode'] = None
        self.controller_params['Y_ControlMode'] = None
        self.controller_params['SS_Mode'] = None
        self.controller_params['WE_Mode'] = None
        self.controller_params['PS_Mode'] = None
        self.controller_params['SD_Mode'] = None
        self.controller_params['Fl_Mode'] = None
        # Necessary parameters
        self.controller_params['zeta_pc'] = None
        self.controller_params['omega_pc'] = None
        self.controller_params['zeta_vs'] = None
        self.controller_params['omega_vs'] = None

        # Turbine parameters
        self.add_input('rotor_inertia',     val=0.0,        units='kg*m**2',        desc='Rotor inertia')
        self.add_input('rho',               val=0.0,        units='kg/m**3',        desc='Air Density')
        self.add_input('R',                 val=0.0,        units='m',              desc='Rotor Radius')

        # Need to define a turbine class with parameters necessary the controller tuning
        turbine = load_turbine():


        def load_turbine(self):
            '''
            Some notes on this 
                - I'm not sure this should all be in a function. Just putting things here to get stuff down. Definitely won't work

            '''

            self.TurbineName = None

            self.rotor_performance_filename = perf_filename

            turbine.J
            turbine.rho
            turbine.rotor_radius
            turbine.Ng
            turbine.rated_rotor_speed
            turbine.v_rated
            turbine.v_min
            turbine.v_max
            turbine.TSR_operational
            turbine.TowerHt # only for floating
            turbine.twr_freq # only for floating
            turbine.ptfm_freq # only for floating


            # Load Cp, Ct, Cq tables
            self.pitch_initial_rad = Cp_Ct_Cq_Tables.compute.pitch_vector   # This is probably wrong, but similar
            self.TSR_initial = Cp_Ct_Cq_Tables.compute.tsr_vector           # This is probably wrong, but similar
            self.Cp_table = Cp_Ct_Cq_Tables.ccblade.Cp                      # This is probably wrong, but similar
            self.Ct_table = Cp_Ct_Cq_Tables.ccblade.Ct                      # This is probably wrong, but similar 
            self.Cq_table = Cp_Ct_Cq_Tables.ccblade.Cq                      # This is probably wrong, but similar

            # Parse rotor performance data
            RotorPerformance = ROSCO_turbine.RotorPerformance()
            self.Cp = RotorPerformance(self.Cp_table,self.pitch_initial_rad,self.TSR_initial)
            self.Ct = RotorPerformance(self.Ct_table,self.pitch_initial_rad,self.TSR_initial)
            self.Cq = RotorPerformance(self.Cq_table,self.pitch_initial_rad,self.TSR_initial)

            # Grab general turbine parameters
            #       - These are probably inputs?
            self.TipRad = None
            self.Rhub =  None
            self.hubHt = None
            self.NumBl = None
            self.TowerHt = None
            self.shearExpNone
            self.rho = None
            self.mu = None
            self.Ng = None
            self.GenEff = None
            self.DTTorSpr = None
            self.generator_inertia = None
            self.tilt = None
            self.precone = None
            self.yaw = 0.0
            self.rotor_inertia = None
            self.generator_inertia = None
            self.J = self.rotor_inertia + self.generator_inertia * self.Ng**2
            self.rated_torque = self.rated_power/(self.GenEff/100*self.rated_rotor_speed*self.Ng)
            self.max_torque = self.rated_torque * 1.1
            self.rotor_radius = self.TipRad
            # self.omega_dt = np.sqrt(self.DTTorSpr/self.J)


            # Define operational TSR
            #       - NJA - not sure exactly how to handle this. From .yaml?
            if not self.TSR_operational:
                self.TSR_operational = self.Cp.TSR_opt


class ServoSE(Group):
    def initialize(self):
        self.options.declare('wt_init_options')

    def setup(self):
        wt_init_options = self.options['wt_init_options']

        self.add_subsystem('powercurve',        RegulatedPowerCurve(wt_init_options   = wt_init_options), promotes = None)
        
    
if __name__ = '__main__':
    