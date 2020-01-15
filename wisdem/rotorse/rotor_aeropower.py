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

from wisdem.ccblade.ccblade_component import CCBladeGeometry, CCBladePower
from wisdem.ccblade import CCAirfoil, CCBlade

from wisdem.commonse.distribution import RayleighCDF, WeibullWithMeanCDF
from wisdem.commonse.utilities import vstack, trapz_deriv, linspace_with_deriv, smooth_min, smooth_abs
from wisdem.commonse.environment import PowerWind
from wisdem.commonse.akima import Akima
from wisdem.rotorse import RPM2RS, RS2RPM
from wisdem.rotorse.rotor_geometry import RotorGeometry
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.rotorse.rotor_fast import eval_unsteady

import time
# ---------------------
# Components
# ---------------------


class RegulatedPowerCurve(ExplicitComponent): # Implicit COMPONENT

    def initialize(self):
        self.options.declare('naero')
        self.options.declare('n_pc')
        self.options.declare('n_pc_spline')
        self.options.declare('regulation_reg_II5',default=True)
        self.options.declare('regulation_reg_III',default=False)
        self.options.declare('lock_pitchII',default=False)

        self.options.declare('n_aoa_grid')
        self.options.declare('n_Re_grid')

    
    def setup(self):
        naero       = self.naero = self.options['naero']
        n_pc        = self.options['n_pc']
        n_pc_spline = self.options['n_pc_spline']
        n_aoa_grid  = self.options['n_aoa_grid']
        n_Re_grid   = self.options['n_Re_grid']

        # parameters
        self.add_input('control_Vin',        val=0.0, units='m/s',  desc='cut-in wind speed')
        self.add_input('control_Vout',       val=0.0, units='m/s',  desc='cut-out wind speed')
        self.add_input('control_ratedPower', val=0.0, units='W',    desc='electrical rated power')
        self.add_input('control_minOmega',   val=0.0, units='rpm',  desc='minimum allowed rotor rotation speed')
        self.add_input('control_maxOmega',   val=0.0, units='rpm',  desc='maximum allowed rotor rotation speed')
        self.add_input('control_maxTS',      val=0.0, units='m/s',  desc='maximum allowed blade tip speed')
        self.add_input('control_tsr',        val=0.0,               desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_input('control_pitch',      val=0.0, units='deg',  desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_discrete_input('drivetrainType',     val='GEARED')
        self.add_input('drivetrainEff',     val=0.0,               desc='overwrite drivetrain model with a given efficiency, used for FAST analysis')
        
        self.add_input('r',         val=np.zeros(naero), units='m',   desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',     val=np.zeros(naero), units='m',   desc='chord length at each section')
        self.add_input('theta',     val=np.zeros(naero), units='deg', desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',      val=0.0,             units='m',   desc='hub radius')
        self.add_input('Rtip',      val=0.0,             units='m',   desc='tip radius')
        self.add_input('hub_height',     val=0.0,             units='m',   desc='hub height')
        self.add_input('precone',   val=0.0,             units='deg', desc='precone angle', )
        self.add_input('tilt',      val=0.0,             units='deg', desc='shaft tilt', )
        self.add_input('yaw',       val=0.0,             units='deg', desc='yaw error', )
        self.add_input('precurve',      val=np.zeros(naero),    units='m', desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,                units='m', desc='precurve at tip')
        self.add_input('presweep',      val=np.zeros(naero),    units='m', desc='presweep at each section')
        self.add_input('presweepTip',   val=0.0,                units='m', desc='presweep at tip')
        
        # self.add_discrete_input('airfoils',  val=[0]*naero,                      desc='CCAirfoil instances')
        self.add_input('airfoils_cl', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa', val=np.zeros((n_aoa_grid)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re', val=np.zeros((n_Re_grid)), desc='Reynolds numbers of polars')
        self.add_discrete_input('nBlades',         val=0,                              desc='number of blades')
        self.add_input('rho',       val=0.0,        units='kg/m**3',    desc='density of air')
        self.add_input('mu',        val=0.0,        units='kg/(m*s)',   desc='dynamic viscosity of air')
        self.add_input('shearExp',  val=0.0,                            desc='shear exponent')
        self.add_discrete_input('nSector',   val=4,                         desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',   val=True,                      desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',   val=True,                      desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation', val=True,                   desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',     val=True,                      desc='use drag coefficient in computing induction factors')

        # outputs
        self.add_output('V',        val=np.zeros(n_pc), units='m/s',        desc='wind vector')
        self.add_output('Omega',    val=np.zeros(n_pc), units='rpm',        desc='rotor rotational speed')
        self.add_output('pitch',    val=np.zeros(n_pc), units='deg',        desc='rotor pitch schedule')
        self.add_output('P',        val=np.zeros(n_pc), units='W',          desc='rotor electrical power')
        self.add_output('T',        val=np.zeros(n_pc), units='N',          desc='rotor aerodynamic thrust')
        self.add_output('Q',        val=np.zeros(n_pc), units='N*m',        desc='rotor aerodynamic torque')
        self.add_output('M',        val=np.zeros(n_pc), units='N*m',        desc='blade root moment')
        self.add_output('Cp',       val=np.zeros(n_pc),                     desc='rotor electrical power coefficient')
        self.add_output('Cp_aero',  val=np.zeros(n_pc),                     desc='rotor aerodynamic power coefficient')
        self.add_output('Ct_aero',  val=np.zeros(n_pc),                     desc='rotor aerodynamic thrust coefficient')
        self.add_output('Cq_aero',  val=np.zeros(n_pc),                     desc='rotor aerodynamic torque coefficient')
        self.add_output('Cm_aero',  val=np.zeros(n_pc),                     desc='rotor aerodynamic moment coefficient')
        self.add_output('V_spline', val=np.zeros(n_pc_spline), units='m/s', desc='wind vector')
        self.add_output('P_spline', val=np.zeros(n_pc_spline), units='W',   desc='rotor electrical power')
        self.add_output('V_R25',       val=0.0,                units='m/s', desc='region 2.5 transition wind speed')
        self.add_output('rated_V',     val=0.0,                units='m/s', desc='rated wind speed')
        self.add_output('rated_Omega', val=0.0,                units='rpm', desc='rotor rotation speed at rated')
        self.add_output('rated_pitch', val=0.0,                units='deg', desc='pitch setting at rated')
        self.add_output('rated_T',     val=0.0,                units='N',   desc='rotor aerodynamic thrust at rated')
        self.add_output('rated_Q',     val=0.0,                units='N*m', desc='rotor aerodynamic torque at rated')
        self.add_output('ax_induct_cutin',   val=np.zeros(naero),           desc='rotor axial induction at cut-in wind speed along blade span')
        self.add_output('tang_induct_cutin', val=np.zeros(naero),           desc='rotor tangential induction at cut-in wind speed along blade span')
        self.add_output('aoa_cutin',val=np.zeros(naero),       units='deg', desc='angle of attack distribution along blade span at cut-in wind speed')
        self.add_output('cl_cutin', val=np.zeros(naero),                    desc='lift coefficient distribution along blade span at cut-in wind speed')
        self.add_output('cd_cutin', val=np.zeros(naero),                    desc='drag coefficient distribution along blade span at cut-in wind speed')

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Create Airfoil class instances
        af = [None]*self.naero
        for i in range(self.naero):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][:,i,:], inputs['airfoils_cd'][:,i,:], inputs['airfoils_cm'][:,i,:])
        

        self.ccblade = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])
        
        Uhub     = np.linspace(inputs['control_Vin'],inputs['control_Vout'], self.options['n_pc']).flatten()
        
        P_aero   = np.zeros_like(Uhub)
        Cp_aero  = np.zeros_like(Uhub)
        Ct_aero  = np.zeros_like(Uhub)
        Cq_aero  = np.zeros_like(Uhub)
        Cm_aero  = np.zeros_like(Uhub)
        P        = np.zeros_like(Uhub)
        Cp       = np.zeros_like(Uhub)
        T        = np.zeros_like(Uhub)
        Q        = np.zeros_like(Uhub)
        M        = np.zeros_like(Uhub)
        Omega    = np.zeros_like(Uhub)
        pitch    = np.zeros_like(Uhub) + inputs['control_pitch']

        Omega_max = min([inputs['control_maxTS'] / inputs['Rtip'], inputs['control_maxOmega']*np.pi/30.])
        
        # Region II
        for i in range(len(Uhub)):
            Omega[i] = Uhub[i] * inputs['control_tsr'] / inputs['Rtip']
        
        # self.ccblade.induction = True
        P_aero, T, Q, M, Cp_aero, Ct_aero, Cq_aero, Cm_aero = self.ccblade.evaluate(Uhub, Omega * 30. / np.pi, pitch, coefficients=True)
        
        # print(Cp_aero)
        # exit()
        
        # print(self.ccblade.a)
        # import matplotlib.pyplot as plt

        # # Induction
        # ft, axt = plt.subplots(1,1,figsize=(5.3, 4))
        # axt.plot(inputs['r'], self.ccblade.a)
        # # axt.legend(fontsize=12)
        # plt.xlabel('Blade Span [m]', fontsize=14, fontweight='bold')
        # plt.ylabel('Induction [-]', fontsize=14, fontweight='bold')
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        # plt.subplots_adjust(bottom = 0.15, left = 0.15)
        # plt.show()
        
        # exit()
        
        P, eff  = CSMDrivetrain(P_aero, inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
        Cp      = Cp_aero*eff
        
        # search for Region 2.5 bounds
        for i in range(len(Uhub)):
        
            if Omega[i] > Omega_max and P[i] < inputs['control_ratedPower']:
                Omega[i]        = Omega_max
                Uhub[i]         = Omega[i] * inputs['Rtip'] / inputs['control_tsr']
                P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch[i]], coefficients=True)
                P[i], eff       = CSMDrivetrain(P_aero[i], inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                Cp[i]           = Cp_aero[i]*eff
                regionIIhalf    = True
                i_IIhalf_start  = i

                outputs['V_R25'] = Uhub[i]
                break


            if P[i] > inputs['control_ratedPower']:
                
                regionIIhalf = False
                break

        
        def maxPregionIIhalf(pitch, Uhub, Omega):
            Uhub_i  = Uhub
            Omega_i = Omega
            pitch   = pitch
                        
            P, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
            return -P
        
        # Solve for regoin 2.5 pitch
        options             = {}
        options['disp']     = False
        options['xatol']    = 1.e-2
        if regionIIhalf == True:
            for i in range(i_IIhalf_start + 1, len(Uhub)):   
                Omega[i]    = Omega_max
                pitch0      = pitch[i-1]
                
                bnds        = [pitch0 - 10., pitch0 + 10.]
                pitch_regionIIhalf = minimize_scalar(lambda x: maxPregionIIhalf(x, Uhub[i], Omega[i]), bounds=bnds, method='bounded', options=options)['x']
                pitch[i]    = pitch_regionIIhalf
                
                P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch[i]], coefficients=True)
                
                P[i], eff  = CSMDrivetrain(P_aero[i], inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                Cp[i]      = Cp_aero[i]*eff

                if P[i] > inputs['control_ratedPower']:    
                    break    
                        
        options             = {}
        options['disp']     = False
        def constantPregionIII(pitch, Uhub, Omega):
            Uhub_i  = Uhub
            Omega_i = Omega
            pitch   = pitch           
            P_aero, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
            P, eff          = CSMDrivetrain(P_aero, inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
            return abs(P - inputs['control_ratedPower'])
            

        
        if regionIIhalf == True:
            # Rated conditions
            
            def min_Uhub_rated_II12(min_inputs):
                return min_inputs[1]
                
            def get_Uhub_rated_II12(min_inputs):

                Uhub_i  = min_inputs[1]
                Omega_i = Omega_max
                pitch   = min_inputs[0]           
                P_aero_i, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch], coefficients=False)
                P_i,eff          = CSMDrivetrain(P_aero_i.flatten(), inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                return abs(P_i - inputs['control_ratedPower'])

            x0              = [pitch[i] + 2. , Uhub[i]]
            bnds            = [(pitch0, pitch0 + 10.),(Uhub[i-1],Uhub[i+1])]
            const           = {}
            const['type']   = 'eq'
            const['fun']    = get_Uhub_rated_II12
            params_rated    = minimize(min_Uhub_rated_II12, x0, method='SLSQP', bounds=bnds, constraints=const)
            U_rated         = params_rated.x[1]
            
            if not np.isnan(U_rated):
                Uhub[i]         = U_rated
                pitch[i]        = params_rated.x[0]
            else:
                print('Regulation trajectory is struggling to find a solution for rated wind speed. Check rotor_aeropower.py')
                U_rated         = Uhub[i]
            
            Omega[i]        = Omega_max
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch0], coefficients=True)
            P_i, eff        = CSMDrivetrain(P_aero[i], inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
            Cp[i]           = Cp_aero[i]*eff
            P[i]            = inputs['control_ratedPower']
            
            
        else:
            # Rated conditions
            def get_Uhub_rated_noII12(pitch, Uhub):
                Uhub_i  = Uhub
                Omega_i = min([Uhub_i * inputs['control_tsr'] / inputs['Rtip'], Omega_max])
                pitch_i = pitch           
                P_aero_i, _, _, _ = self.ccblade.evaluate([Uhub_i], [Omega_i * 30. / np.pi], [pitch_i], coefficients=False)
                P_i, eff          = CSMDrivetrain(P_aero_i, inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                return abs(P_i - inputs['control_ratedPower'])
            
            bnds     = [Uhub[i-1], Uhub[i+1]]
            U_rated  = minimize_scalar(lambda x: get_Uhub_rated_noII12(pitch[i], x), bounds=bnds, method='bounded', options=options)['x']
            
            if not np.isnan(U_rated):
                Uhub[i]         = U_rated
            else:
                print('Regulation trajectory is struggling to find a solution for rated wind speed. Check rotor_aeropower.py. For now, U rated is assumed equal to ' + str(Uhub[i]) + ' m/s')
                U_rated         = Uhub[i]
            
            
            Omega[i] = min([Uhub[i] * inputs['control_tsr'] / inputs['Rtip'], Omega_max])
            pitch0   = pitch[i]
            
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = self.ccblade.evaluate([Uhub[i]], [Omega[i] * 30. / np.pi], [pitch0], coefficients=True)
            P[i], eff    = CSMDrivetrain(P_aero[i], inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
            Cp[i]        = Cp_aero[i]*eff
        
        
        for j in range(i + 1,len(Uhub)):
            Omega[j] = Omega[i]
            if self.options['regulation_reg_III']:
                
                pitch0   = pitch[j-1]
                bnds     = [pitch0, pitch0 + 15.]
                pitch_regionIII = minimize_scalar(lambda x: constantPregionIII(x, Uhub[j], Omega[j]), bounds=bnds, method='bounded', options=options)['x']
                pitch[j]        = pitch_regionIII
                P_aero[j], T[j], Q[j], M[j], Cp_aero[j], Ct_aero[j], Cq_aero[j], Cm_aero[j] = self.ccblade.evaluate([Uhub[j]], [Omega[j] * 30. / np.pi], [pitch[j]], coefficients=True)
                P[j], eff       = CSMDrivetrain(P_aero[j], inputs['control_ratedPower'], discrete_inputs['drivetrainType'], inputs['drivetrainEff'])
                Cp[j]           = Cp_aero[j]*eff


                if abs(P[j] - inputs['control_ratedPower']) > 1e+4:
                    print('The pitch in region III is not being determined correctly at wind speed ' + str(Uhub[j]) + ' m/s')
                    P[j]        = inputs['control_ratedPower']
                    T[j]        = T[j-1]
                    Q[j]        = P[j] / Omega[j]
                    M[j]        = M[j-1]
                    pitch[j]    = pitch[j-1]
                    Cp[j]       = P[j] / (0.5 * inputs['rho'] * np.pi * inputs['Rtip']**2 * Uhub[i]**3)
                    Ct_aero[j]  = Ct_aero[j-1]
                    Cq_aero[j]  = Cq_aero[j-1]
                    Cm_aero[j]  = Cm_aero[j-1]

                P[j] = inputs['control_ratedPower']
                
            else:
                P[j]        = inputs['control_ratedPower']
                T[j]        = 0
                Q[j]        = Q[i]
                M[j]        = 0
                pitch[j]    = 0
                Cp[j]       = P[j] / (0.5 * inputs['rho'] * np.pi * inputs['Rtip']**2 * Uhub[i]**3)
                Ct_aero[j]  = 0
                Cq_aero[j]  = 0
                Cm_aero[j]  = 0

        
        outputs['T']       = T
        outputs['Q']       = Q
        outputs['Omega']   = Omega * 30. / np.pi


        outputs['P']       = P  
        outputs['Cp']      = Cp  
        outputs['Cp_aero'] = Cp_aero
        outputs['Ct_aero'] = Ct_aero
        outputs['Cq_aero'] = Cq_aero
        outputs['Cm_aero'] = Cm_aero
        outputs['V']       = Uhub
        outputs['M']       = M
        outputs['pitch']   = pitch
                
        self.ccblade.induction_inflow = True
        a_regII, ap_regII, alpha_regII, cl_regII, cd_regII = self.ccblade.distributedAeroLoads(Uhub[0], Omega[0] * 30. / np.pi, pitch[0], 0.0)
        
        # Fit spline to powercurve for higher grid density
        spline   = PchipInterpolator(Uhub, P)
        V_spline = np.linspace(inputs['control_Vin'],inputs['control_Vout'], num=self.options['n_pc_spline'])
        P_spline = spline(V_spline)
        
        # outputs
        idx_rated = list(Uhub).index(U_rated)
        outputs['rated_V']     = U_rated.flatten()
        outputs['rated_Omega'] = Omega[idx_rated] * 30. / np.pi
        outputs['rated_pitch'] = pitch[idx_rated]
        outputs['rated_T']     = T[idx_rated]
        outputs['rated_Q']     = Q[idx_rated]
        outputs['V_spline']    = V_spline.flatten()
        outputs['P_spline']    = P_spline.flatten()
        outputs['ax_induct_cutin']   = a_regII
        outputs['tang_induct_cutin'] = ap_regII
        outputs['aoa_cutin']         = alpha_regII
        outputs['cl_cutin']         = cl_regII
        outputs['cd_cutin']         = cd_regII



class Cp_Ct_Cq_Tables(ExplicitComponent):
    def initialize(self):
        self.options.declare('naero')
        self.options.declare('n_pitch', default=20)
        self.options.declare('n_tsr', default=20)
        self.options.declare('n_U', default=1)
        self.options.declare('n_aoa_grid')
        self.options.declare('n_Re_grid')

    def setup(self):
        naero       = self.naero = self.options['naero']
        n_aoa_grid  = self.options['n_aoa_grid']
        n_Re_grid   = self.options['n_Re_grid']
        n_pitch     = self.options['n_pitch']
        n_tsr       = self.options['n_tsr']
        n_U         = self.options['n_U']
        
        # parameters        
        self.add_input('control_Vin',   val=0.0,             units='m/s',       desc='cut-in wind speed')
        self.add_input('control_Vout',  val=0.0,             units='m/s',       desc='cut-out wind speed')
        self.add_input('r',             val=np.zeros(naero), units='m',         desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
        self.add_input('chord',         val=np.zeros(naero), units='m',         desc='chord length at each section')
        self.add_input('theta',         val=np.zeros(naero), units='deg',       desc='twist angle at each section (positive decreases angle of attack)')
        self.add_input('Rhub',          val=0.0,             units='m',         desc='hub radius')
        self.add_input('Rtip',          val=0.0,             units='m',         desc='tip radius')
        self.add_input('hub_height',    val=0.0,             units='m',         desc='hub height')
        self.add_input('precone',       val=0.0,             units='deg',       desc='precone angle')
        self.add_input('tilt',          val=0.0,             units='deg',       desc='shaft tilt')
        self.add_input('yaw',           val=0.0,             units='deg',       desc='yaw error')
        self.add_input('precurve',      val=np.zeros(naero), units='m',         desc='precurve at each section')
        self.add_input('precurveTip',   val=0.0,             units='m',         desc='precurve at tip')
        self.add_input('presweep',      val=np.zeros(naero), units='m',         desc='presweep at each section')
        self.add_input('presweepTip',   val=0.0,             units='m',         desc='presweep at tip')
        self.add_input('rho',           val=0.0,             units='kg/m**3',   desc='density of air')
        self.add_input('mu',            val=0.0,             units='kg/(m*s)',  desc='dynamic viscosity of air')
        self.add_input('shearExp',      val=0.0,                                desc='shear exponent')
        # self.add_discrete_input('airfoils',      val=[0]*naero,                 desc='CCAirfoil instances')
        self.add_input('airfoils_cl', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm', val=np.zeros((n_aoa_grid, naero, n_Re_grid)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa', val=np.zeros((n_aoa_grid)), units='deg', desc='angle of attack grid for polars')
        self.add_input('airfoils_Re', val=np.zeros((n_Re_grid)), desc='Reynolds numbers of polars')
        self.add_discrete_input('nBlades',       val=0,                         desc='number of blades')
        self.add_discrete_input('nSector',       val=4,                         desc='number of sectors to divide rotor face into in computing thrust and power')
        self.add_discrete_input('tiploss',       val=True,                      desc='include Prandtl tip loss model')
        self.add_discrete_input('hubloss',       val=True,                      desc='include Prandtl hub loss model')
        self.add_discrete_input('wakerotation',  val=True,                      desc='include effect of wake rotation (i.e., tangential induction factor is nonzero)')
        self.add_discrete_input('usecd',         val=True,                      desc='use drag coefficient in computing induction factors')
        self.add_input('pitch_vector_in',  val=np.zeros(n_pitch), units='deg',  desc='pitch vector specified by the user')
        self.add_input('tsr_vector_in',    val=np.zeros(n_tsr),                 desc='tsr vector specified by the user')
        self.add_input('U_vector_in',      val=np.zeros(n_U),     units='m/s',  desc='wind vector specified by the user')

        # outputs
        self.add_output('Cp_aero_table',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero power coefficient')
        self.add_output('Ct_aero_table',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero thrust coefficient')
        self.add_output('Cq_aero_table',   val=np.zeros((n_tsr, n_pitch, n_U)), desc='table of aero torque coefficient')
        self.add_output('pitch_vector',    val=np.zeros(n_pitch), units='deg',  desc='pitch vector used')
        self.add_output('tsr_vector',      val=np.zeros(n_tsr),                 desc='tsr vector used')
        self.add_output('U_vector',        val=np.zeros(n_U),     units='m/s',  desc='wind vector used')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Create Airfoil class instances
        af = [None]*self.naero
        for i in range(self.naero):
            af[i] = CCAirfoil(inputs['airfoils_aoa'], inputs['airfoils_Re'], inputs['airfoils_cl'][:,i,:], inputs['airfoils_cd'][:,i,:], inputs['airfoils_cm'][:,i,:])
       

        n_pitch  = self.options['n_pitch']
        n_tsr    = self.options['n_tsr']
        n_U      = self.options['n_U']
        U_vector = inputs['U_vector_in']
        V_in     = inputs['control_Vin']
        V_out    = inputs['control_Vout']
        
        tsr_vector = inputs['tsr_vector_in']
        pitch_vector = inputs['pitch_vector_in']
        
        self.ccblade = CCBlade(inputs['r'], inputs['chord'], inputs['theta'], af, inputs['Rhub'], inputs['Rtip'], discrete_inputs['nBlades'], inputs['rho'], inputs['mu'], inputs['precone'], inputs['tilt'], inputs['yaw'], inputs['shearExp'], inputs['hub_height'], discrete_inputs['nSector'], inputs['precurve'], inputs['precurveTip'],inputs['presweep'], inputs['presweepTip'], discrete_inputs['tiploss'], discrete_inputs['hubloss'],discrete_inputs['wakerotation'], discrete_inputs['usecd'])
        
        if max(U_vector) == 0.:
            U_vector    = np.linspace(V_in[0],V_out[0], n_U)
        if max(tsr_vector) == 0.:
            tsr_vector = np.linspace(7.,11., n_tsr)
        if max(pitch_vector) == 0.:
            pitch_vector = np.linspace(-5., 5., n_pitch)
        
        outputs['pitch_vector'] = pitch_vector
        outputs['tsr_vector']   = tsr_vector        
        outputs['U_vector']     = U_vector
                
        R = inputs['Rtip']
        
        Cp_aero_table = np.zeros((n_tsr, n_pitch, n_U))
        Ct_aero_table = np.zeros((n_tsr, n_pitch, n_U))
        Cq_aero_table = np.zeros((n_tsr, n_pitch, n_U))
        
        for i in range(n_U):
            for j in range(n_tsr):
                U     =  U_vector[i] * np.ones(n_pitch)
                Omega = tsr_vector[j] *  U_vector[i] / R * 30. / np.pi * np.ones(n_pitch)
                _, _, _, _, outputs['Cp_aero_table'][j,:,i], outputs['Ct_aero_table'][j,:,i], outputs['Cq_aero_table'][j,:,i], _ = self.ccblade.evaluate(U, Omega, pitch_vector, coefficients=True)


# Class to define a constraint so that the blade cannot operate in stall conditions
class NoStallConstraint(ExplicitComponent):
    def initialize(self):
        
        self.options.declare('RefBlade')
        self.options.declare('verbosity', default = False)
    
    def setup(self):
        RefBlade    = self.options['RefBlade']
        NPTS        = len(RefBlade['pf']['s'])
        n_aoa_grid  = len(RefBlade['airfoils_aoa'])
        n_Re_grid   = len(RefBlade['airfoils_Re'])
        
        self.add_input('stall_angle_along_span', val=np.zeros(NPTS), units = 'deg', desc = 'Stall angle along blade span')
        self.add_input('aoa_along_span',         val=np.zeros(NPTS), units = 'deg', desc = 'Angle of attack along blade span')
        self.add_input('stall_margin',           val=0.0,            units = 'deg', desc = 'Minimum margin from the stall angle')
        self.add_input('min_s',                  val=0.0,            desc = 'Minimum nondimensional coordinate along blade span where to define the constraint (blade root typically stalls)')
        self.add_input('airfoils_cl',       val=np.zeros((n_aoa_grid, NPTS, n_Re_grid)), desc='lift coefficients, spanwise')
        self.add_input('airfoils_cd',       val=np.zeros((n_aoa_grid, NPTS, n_Re_grid)), desc='drag coefficients, spanwise')
        self.add_input('airfoils_cm',       val=np.zeros((n_aoa_grid, NPTS, n_Re_grid)), desc='moment coefficients, spanwise')
        self.add_input('airfoils_aoa',      val=np.zeros((n_aoa_grid)), units='deg', desc='angle of attack grid for polars')
        
        self.add_output('no_stall_constraint',   val=np.zeros(NPTS), desc = 'Constraint, ratio between angle of attack plus a margin and stall angle')

    def compute(self, inputs, outputs):
        
        verbosity = self.options['verbosity']
        RefBlade  = self.options['RefBlade']
        
        i_min = np.argmin(abs(inputs['min_s'] - RefBlade['pf']['s']))
        
        for i in range(len(RefBlade['pf']['s'])):
            unsteady = eval_unsteady(inputs['airfoils_aoa'], inputs['airfoils_cl'][:,i,0], inputs['airfoils_cd'][:,i,0], inputs['airfoils_cm'][:,i,0])
            inputs['stall_angle_along_span'][i] = unsteady['alpha1']
            if inputs['stall_angle_along_span'][i] == 0:
                inputs['stall_angle_along_span'][i] = 1e-6 # To avoid nan
        
        for i in range(i_min, len(RefBlade['pf']['s'])):
            outputs['no_stall_constraint'][i] = (inputs['aoa_along_span'][i] + inputs['stall_margin']) / inputs['stall_angle_along_span'][i]
        
            if verbosity == True:
                if outputs['no_stall_constraint'][i] > 1:
                    print('Blade is stalling at span location %.2f %%' % (RefBlade['pf']['s'][i]*100.))


class AEP(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_pc_spline')
    
    def setup(self):
        n_pc_spline = self.options['n_pc_spline']
        """integrate to find annual energy production"""

        # inputs
        self.add_input('CDF_V', val=np.zeros(n_pc_spline), units='m/s', desc='cumulative distribution function evaluated at each wind speed')
        self.add_input('P', val=np.zeros(n_pc_spline), units='W', desc='power curve (power)')
        self.add_input('lossFactor', val=0.0, desc='multiplicative factor for availability and other losses (soiling, array, etc.)')

        # outputs
        self.add_output('AEP', val=0.0, units='kW*h', desc='annual energy production')

        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):

        lossFactor = inputs['lossFactor']
        P = inputs['P']
        CDF_V = inputs['CDF_V']
        factor = lossFactor/1e3*365.0*24.0
        outputs['AEP'] = factor*np.trapz(P, CDF_V)  # in kWh
        '''
        dAEP_dP, dAEP_dCDF = trapz_deriv(P, CDF_V)
        dAEP_dP *= factor
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([outputs['AEP']/lossFactor])
        self.J = {}
        self.J['AEP', 'CDF_V'] = np.reshape(dAEP_dCDF, (1, len(dAEP_dCDF)))
        self.J['AEP', 'P'] = np.reshape(dAEP_dP, (1, len(dAEP_dP)))
        self.J['AEP', 'lossFactor'] = dAEP_dlossFactor

    def compute_partials(self, inputs, J):
        J = self.J
        '''


def CSMDrivetrain(aeroPower, ratedPower, drivetrainType, drivetrainEff):

    if drivetrainEff == 0.0:
        drivetrainType = drivetrainType.upper()
        if drivetrainType == 'GEARED':
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif drivetrainType == 'SINGLE_STAGE':
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif drivetrainType == 'MULTI_DRIVE':
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif drivetrainType == 'PM_DIRECT_DRIVE':
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899
        elif drivetrainType == 'CONSTANT_EFF':
            constant = 0.00
            linear = 0.07
            quadratic = 0.0
        
        Pbar0 = aeroPower / ratedPower

        # handle negative power case (with absolute value)
        Pbar1, dPbar1_dPbar0 = smooth_abs(Pbar0, dx=0.01)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar, dPbar_dPbar1, _ = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant/Pbar + linear + quadratic*Pbar)
    else:
        eff = drivetrainEff
        
    return aeroPower * eff, eff


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

        

class RotorAeroPower(Group):
    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('npts_coarse_power_curve', default=20)
        self.options.declare('npts_spline_power_curve', default=200)
        self.options.declare('regulation_reg_II5',      default=True)
        self.options.declare('regulation_reg_III',      default=True)
        self.options.declare('flag_Cp_Ct_Cq_Tables',    default=True)
        self.options.declare('topLevelFlag',            default=False)
        self.options.declare('user_update_routine',     default=None)
    
    def setup(self):
        RefBlade = self.options['RefBlade']
        npts_coarse_power_curve     = self.options['npts_coarse_power_curve']
        npts_spline_power_curve     = self.options['npts_spline_power_curve']
        regulation_reg_II5          = self.options['regulation_reg_II5']
        regulation_reg_III          = self.options['regulation_reg_III']
        flag_Cp_Ct_Cq_Tables        = self.options['flag_Cp_Ct_Cq_Tables']
        topLevelFlag                = self.options['topLevelFlag']
        user_update_routine         = self.options['user_update_routine']
        NPTS                        = len(RefBlade['pf']['s'])
        NAFgrid                     = len(RefBlade['airfoils_aoa'])
        NRe                         = len(RefBlade['airfoils_Re'])

        aeroIndeps = IndepVarComp()
        aeroIndeps.add_output('wind_reference_height',  val=0.0, units='m',     desc='reference hub height for IEC wind speed (used in CDF calculation)')
        aeroIndeps.add_output('control_Vin',            val=0.0, units='m/s',   desc='cut-in wind speed')
        aeroIndeps.add_output('control_Vout',           val=0.0, units='m/s',   desc='cut-out wind speed')
        aeroIndeps.add_output('machine_rating',         val=0.0, units='W',     desc='rated power')
        aeroIndeps.add_output('control_minOmega',       val=0.0, units='rpm',   desc='minimum allowed rotor rotation speed')
        aeroIndeps.add_output('control_maxOmega',       val=0.0, units='rpm',   desc='maximum allowed rotor rotation speed')
        aeroIndeps.add_output('control_maxTS',          val=0.0, units='m/s',   desc='maximum allowed blade tip speed')
        aeroIndeps.add_output('control_tsr',            val=0.0,                desc='tip-speed ratio in Region 2 (should be optimized externally)')
        aeroIndeps.add_output('control_pitch',          val=0.0, units='deg',   desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        aeroIndeps.add_discrete_output('drivetrainType', val='GEARED')
        aeroIndeps.add_output('AEP_loss_factor',        val=1.0,                desc='availability and other losses (soiling, array, etc.)')
        aeroIndeps.add_output('shape_parameter',        val=0.0)
        aeroIndeps.add_output('drivetrainEff',          val=0.0,                desc='overwrite drivetrain model with a given efficiency, used for FAST analysis')
        self.add_subsystem('aeroIndeps', aeroIndeps, promotes=['*'])
        
        # --- Rotor Aero & Power ---
        if topLevelFlag:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('hub_height',   val=0.0, units='m')
            sharedIndeps.add_output('rho',          val=1.225, units='kg/m**3')
            sharedIndeps.add_output('mu',           val=1.81e-5, units='kg/(m*s)')
            sharedIndeps.add_output('shearExp',     val=0.2)
            sharedIndeps.add_discrete_output('tiploss', True)
            sharedIndeps.add_discrete_output('hubloss', True)
            sharedIndeps.add_discrete_output('wakerotation', True)
            sharedIndeps.add_discrete_output('usecd', True)
            sharedIndeps.add_discrete_output('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
            
            self.add_subsystem('rotorGeom', RotorGeometry(RefBlade=RefBlade, topLevelFlag=topLevelFlag, user_update_routine=user_update_routine), promotes=['*'])

        # self.add_subsystem('tipspeed', MaxTipSpeed())
        self.add_subsystem('powercurve', RegulatedPowerCurve(naero=NPTS,
                                                             n_pc=npts_coarse_power_curve,
                                                             n_pc_spline=npts_spline_power_curve,
                                                             regulation_reg_II5=regulation_reg_II5,
                                                             regulation_reg_III=regulation_reg_III,
                                                             n_aoa_grid=NAFgrid,
                                                             n_Re_grid=NRe), promotes=['*'])

        if flag_Cp_Ct_Cq_Tables:
            self.add_subsystem('cpctcq_tables',   Cp_Ct_Cq_Tables(naero=NPTS,n_aoa_grid=NAFgrid,n_Re_grid=NRe), promotes=['*'])
        
        self.add_subsystem('nostallconstraint', NoStallConstraint(RefBlade = RefBlade, verbosity = False), promotes=['airfoils_cl','airfoils_cd','airfoils_cm','airfoils_aoa','no_stall_constraint'])
        self.add_subsystem('wind', PowerWind(nPoints=1), promotes=['shearExp'])
        self.add_subsystem('cdf', WeibullWithMeanCDF(nspline=npts_spline_power_curve))
        #self.add_subsystem('cdf', RayleighCDF(nspline=npts_spline_power_curve))
        self.add_subsystem('aep', AEP(n_pc_spline=npts_spline_power_curve), promotes=['AEP'])

        self.add_subsystem('outputs_aero', OutputsAero(npts_coarse_power_curve=npts_coarse_power_curve), promotes=['*'])

        self.connect('machine_rating',  'control_ratedPower')
        
        # connections to nostallconstraint
        self.connect('aoa_cutin','nostallconstraint.aoa_along_span')
        
        # connections to wind
        if topLevelFlag:
            self.connect('V_mean', 'wind.Uref')
            self.connect('wind_zvec', 'wind.z')
        self.connect('wind_reference_height', 'wind.zref')

        # connections to cdf
        self.connect('V_spline', 'cdf.x')
        self.connect('wind.U', 'cdf.xbar', src_indices=[0])
        self.connect('shape_parameter', 'cdf.k')

        # connections to aep
        self.connect('cdf.F', 'aep.CDF_V')
        self.connect('P_spline', 'aep.P')
        self.connect('AEP_loss_factor', 'aep.lossFactor')


def Init_RotorAeropower_wRefBlade(rotor, blade):
    # === blade grid ===
    rotor['hubFraction']      = blade['config']['hubD']/2./blade['pf']['r'][-1] #0.025  # (Float): hub location as fraction of radius
    rotor['bladeLength']      = blade['ctrl_pts']['bladeLength'] #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    rotor['precone']          = blade['config']['cone_angle'] #2.5  # (Float, deg): precone angle
    rotor['tilt']             = blade['config']['tilt_angle'] #5.0  # (Float, deg): shaft tilt
    rotor['yaw']              = 0.0  # (Float, deg): yaw error
    rotor['nBlades']          = blade['config']['number_of_blades'] #3  # (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    rotor['r_max_chord']      = blade['ctrl_pts']['r_max_chord']  # 0.23577 #(Float): location of max chord on unit radius
    rotor['chord_in']         = np.array(blade['ctrl_pts']['chord_in']) # np.array([3.2612, 4.3254, 4.5709, 3.7355, 2.69923333, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in']         = np.array(blade['ctrl_pts']['theta_in']) # np.array([0.0, 13.2783, 12.30514836,  6.95106536,  2.72696309, -0.0878099]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in']      = np.array(blade['ctrl_pts']['precurve_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in']      = np.array(blade['ctrl_pts']['presweep_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['sparT_in']         = np.array(blade['ctrl_pts']['sparT_in']) # np.array([0.0, 0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in']           = np.array(blade['ctrl_pts']['teT_in']) # np.array([0.0, 0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # if 'le_var' in blade['precomp']['le_var']:
    #     rotor['leT_in']       = np.array(blade['ctrl_pts']['leT_in']) ## (Array, m): leading-edge thickness parameters
    rotor['airfoil_position'] = np.array(blade['outer_shape_bem']['airfoil_position']['grid'])
    # ------------------
    
    # === atmosphere ===
    rotor['rho']              = 1.225  # (Float, kg/m**3): density of air
    rotor['mu']               = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['hub_height']       = blade['config']['hub_height']  # (Float, m): hub height
    rotor['shearExp']         = 0.25  # (Float): shear exponent
    rotor['shape_parameter']  = 2.0
    rotor['turbine_class']    = blade['config']['turbine_class'].upper() #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['wind_reference_height'] = blade['config']['hub_height']  # (Float, m): hub height
    # ----------------------
    
    # === control ===
    rotor['control_Vin']      = blade['config']['Vin'] #3.0  # (Float, m/s): cut-in wind speed
    rotor['control_Vout']     = blade['config']['Vout'] #25.0  # (Float, m/s): cut-out wind speed
    rotor['machine_rating']   = blade['config']['rating'] #5e6  # (Float, W): rated power
    rotor['control_minOmega'] = blade['config']['minOmega'] #0.0  # (Float, rpm): minimum allowed rotor rotation speed
    rotor['control_maxOmega'] = blade['config']['maxOmega'] #12.0  # (Float, rpm): maximum allowed rotor rotation speed
    rotor['control_maxTS']    = blade['config']['maxTS']
    rotor['control_tsr']      = blade['config']['tsr'] #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control_pitch']    = blade['config']['pitch'] #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    # ----------------------
    
    # === no stall constraint ===
    rotor['nostallconstraint.min_s']        = 0.25  # The stall constraint is only computed from this value (nondimensional coordinate along blade span) to blade tip
    rotor['nostallconstraint.stall_margin'] = 3.0   # Values in deg of stall margin
    # ----------------------
    
    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['AEP_loss_factor'] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
    rotor['drivetrainType']   = blade['config']['drivetrain'].upper() #DRIVETRAIN_TYPE['GEARED']  # (Enum)
    # ----------------------
    return rotor

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
   
    
