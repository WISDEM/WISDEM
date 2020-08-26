"""
driveSE_components.py
New components for low speed shaft, main bearings, gearbox, bedplate and yaw bearings, 
as well as simple sizing functions for the components from the rest of the nacelle.

Created by Ryan King, Yi Guo and Taylor Parsons 2014.
Copyright (c) NREL. All rights reserved.

Most of these classes will have a corresponding *_OM(ExplicitComponent) class in drivese_omdao.py
This file should contain NO OpenMDAO code, but each class here will be a member of one of the *_OM classes.
"""

from __future__ import print_function

"""
NOTES:
  - Equation numbers (Eq. x.y) refer to the report 'DriveSE: An Analytic Formulation...'
    by Guo et al., 2014
 
  - cleaned and reorganized 2019 04 18 GNS
  - d_y[] computed but not used or returned in size_LSS_*() functions
  - Many variables calculated but not saved:
       y_gp, F_mb_x, F_gb_[xyz], xshaft, Index, L_cd, F_mb[12]_x, T, N_count
       others?
       Could these be useful in the future, or should we comment them out?
  - drivese_omdao.py gives:
      bending moments in Nm
      forces and thrusts in N
      lengths in m
      
  UNITS
    mass in kg
    weight in N = g * mass
    forces in N
    moments in N-m
  
  - rotor_bending_moment_x is in N-m. Multiply it by self.u_knm_inlb / 1000 to convert to in-lb
  
  - rotor_speed changed to rotor_rpm in Generator and Gearbox to be consistent with other components

"""

useComputeD = True    # use computeD() function to compute shaft diameter?
useFlangeModel = True # use new flange model to compute len, mass?
#useFlangeModel = False # use new flange model to compute len, mass?

import sys, os
import numpy as np
import scipy as scp
import scipy.optimize as opt
from math import pi, cos, sqrt, sin, exp, log10, log

from wisdem.drivetrainse.drivese_utils import get_rotor_mass, get_distance_hub2mb, get_My, get_Mz, resize_for_bearings, mainshaftFlangeCalc 
from wisdem.commonse.utilities import assembleI, unassembleI 

# Constants
        
U_KNM_INLB = 8850.745454036  # 1 kN-m = 8850.74577 lb-in
U_IN_M = 0.0254000508001  # 1 in = 0.0254 m
G_GRAV = 9.81 # m-s^-2

FLANGE_THICK_FACTOR = 4    # Ratio of flange thickness to shell thickness - MUST AGREE with value in sph_hubse_components.py

# Shaft material properties - note mix of metric and English units

E_STEEL_LSS = 210e9 # Young's modulus of shaft steel in N/m^2
DENSITY_STEEL_LSS = 7800.0 # density of steel in kg/m^3
SY_STEEL_LSS = 66000  # *self.S_ut/700e6 #66000 #psi # approx tensile strength of steel in psi (about 4.55E5 kPa or 4.55E8 N-m^-2)

E_CAST_IRON = 169e9 # Young's modulus of cast iron in N/m^2
DENSITY_CAST_IRON = 7100.0 # density of cast iron in kg/m^3

#---------------------

def bearing_defl_check(btype):
    ''' Identical code was used in compute() for both 3 and 4 pt cases
        See below for original code with logic errors "== 'TRB1' or 'TRB2'"
        Error fixed in 2 places, but now second assignment for 'RB' will never happen. What is correct?
        
        See section '2.2.3.3 Deflection Check' in DriveSE report
        
        Units appear to be radians. 
        GNS 2019 04 18
    '''
    if btype == 'TRB1' or btype == 'TRB2':
        Bearing_Limit = 3.0 / 60.0 / 180.0 * pi
    elif btype == 'CRB':
        Bearing_Limit = 4.0 / 60.0 / 180.0 * pi
    elif btype == 'SRB' or btype == 'RB':
        Bearing_Limit = 0.078
    elif btype == 'RB':
        Bearing_Limit = 0.002
    elif btype == 'CARB':
        Bearing_Limit = 0.5 / 180 * pi
    else:
        Bearing_Limit = False
    return Bearing_Limit

'''
        if self.mb1Type == 'TRB1' or 'TRB2':
            Bearing_Limit = 3.0 / 60.0 / 180.0 * pi
        elif self.mb1Type == 'CRB':
            Bearing_Limit = 4.0 / 60.0 / 180.0 * pi
        elif self.mb1Type == 'SRB' or 'RB':
            Bearing_Limit = 0.078
        elif self.mb1Type == 'RB':
            Bearing_Limit = 0.002
        elif self.mb1Type == 'CARB':
            Bearing_Limit = 0.5 / 180 * pi
        else:
            Bearing_Limit = False
        if self.mb2Type == 'TRB1' or 'TRB2':
            Bearing_Limit2 = 3.0 / 60.0 / 180.0 * pi
        elif self.mb2Type == 'CRB':
            Bearing_Limit2 = 4.0 / 60.0 / 180.0 * pi
        elif self.mb2Type == 'SRB' or 'RB':
            Bearing_Limit2 = 0.078
        elif self.mb2Type == 'RB':
            Bearing_Limit2 = 0.002
        elif self.mb2Type == 'CARB':
            Bearing_Limit2 = 0.5 / 180 * pi
        else:
            Bearing_Limit2 = False
'''

#%%------------------------------------

def computeD(MM, rbmx, Sy, n_safety, debug=False):
    '''
    Implement Eqn. 2.30 (Eq 2.26 in 2015 rpt) to compute shaft diameter
    
    MM   : moment in N-m
    rbmx : rotor_bending_moment_x in N-m
    Sy   : tensile strength of material in psi E.g., approx 66000 for steel
    n_safety : safety factor (in the range of 1 to 3 or so)
    
    NOTE: MM is in N-m. The original code in size_LSS_3pt converted MM to kN-m, but size_LSS_4pt did not
    '''
    d1 = (16.0 * n_safety / pi / Sy) # units: in^2 / lb
    d2 = U_KNM_INLB * (4.0 * (MM * 0.001)**2 + 3.0 * (rbmx * 0.001)**2)**0.5 # units: in-lb
    d3 = (d1 * d2) ** (1./3.) * U_IN_M
    
    if debug:
        sys.stderr.write('computeD\n  IN: MM {:.1f} rbmx {:.1f} Sy {:.1f} Nsafe {:.1f}\n'.format(MM, rbmx, Sy, n_safety))
        sys.stderr.write('  d1 {:.6f} d2 {:.1f} d3 {:.3f}\n'.format(d1,d2,d3))
    return d3

    #    self.D_max = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb)**2 + 3.0 * (
    #        self.rotor_bending_moment_x / 1000.0 * self.u_knm_inlb)**2)**0.5)**(1.0 / 3.0) * self.u_in_m

#%%----------------------------------------------------

#-------------------------------------------------------------------------
# Drivetrain component models
#-------------------------------------------------------------------------

# 4 pt Low Speed Shaft Sizing
class LowSpeedShaft4pt(object):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.

    GNS Notes:
      Bearing masses returned (self.mb[12]_mass) do NOT include bearing housings. These will be added by class MainBearing.
    '''

    def __init__(self, mb1Type, mb2Type, IEC_Class, debug=False):
        
        super(LowSpeedShaft4pt, self).__init__()

        # set LSS configuration parameters
        self.mb1Type = mb1Type #Enum('SRB',('CARB','TRB1','TRB2','SRB','CRB','RB'),iotype='in',desc='Main bearing type')
        self.mb2Type = mb2Type #Enum('SRB',('CARB','TRB1','TRB2','SRB','CRB','RB'),iotype='in',desc='Second bearing type')
        self.IEC_Class = IEC_Class #Enum('A',('A','B','C'),iotype='in',desc='IEC class letter: A, B, or C')
        
        self.debug = debug

    #----------------------------------------------------
    # Deflection functions
    #----------------------------------------------------
    '''
    These functions were defined inline in LowSpeedShaft3pt() and LowSpeedShaft4pt().
      Note that fx() and gx() from size_LSS_3pt() appear to be identical to deflection() and gx() from size_LSS_4pt_Loop_1()

    Despite the 'deflection*()' names, these functions do not return deflection. The units of the 'deflection*' and 'fx*'
    functions are N*kg*m. (See Eq. 2.33 (2.29 in 2015 rpt)) The returned value is E * I * v (where v is deflection). At the end of the 
    size_LSS*() functions, these values are divided by E*I to give actual deflections (d_y[], which is not used).
    
    The 'gx*()' functions return N*kg and the results are divided by E*I to give a dimensionless value. Eqn 2.33 implies
    that this is dv/dx, which is equivalent to a tangent. By the small angle approximation, this is also equal to the
    angle itself in radians, which is why we can assign it to theta_y[] and use bearing_defl_check().
    
    GNS 2019 05
    '''
    
    # from size_LSS_4pt_Loop_1()
    
    @staticmethod
    def deflection(F_z, W_r, gamma, M_y, f_mb_z, distance_hub2mb, W_ms, L_ms, z):
        return -F_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb_z * (z - distance_hub2mb)**3 / 6.0 + W_ms / (L_ms + distance_hub2mb) / 24.0 * z**4
    
    @staticmethod
    def gx(F_z, W_r, gamma, M_y, f_mb_z, distance_hub2mb, W_ms, L_ms, C1, z):
        return -F_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb_z * (z - distance_hub2mb)**2 / 2.0 + W_ms / (L_ms + distance_hub2mb) / 6.0 * z**3 + C1
    
    # from size_LSS_4pt_Loop_2()
    
    @staticmethod
    def deflection1(F_r_z, W_r, gamma, M_y, f_mb1_z, distance_hub2mb, W_ms, L_ms, L_mb, z):
        return -F_r_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb1_z * (z - distance_hub2mb)**3 / 6.0 + W_ms / (L_ms + L_mb) / 24.0 * z**4
    
    @staticmethod
    def gx1(F_r_z, W_r, gamma, M_y, f_mb1_z, distance_hub2mb, W_ms, L_ms, L_mb, C11, z):
        return -F_r_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb1_z * (z - distance_hub2mb)**2 / 2.0 + W_ms / (L_ms + L_mb) / 6.0 * z**3 + C11
    
    # Deflection between mb2 and gearbox
    @staticmethod
    def deflection2(F_z, W_r, gamma, M_y, f_mb1_z, f_mb2_z, distance_hub2mb, W_ms, L_ms, L_mb, z):
        return -F_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb1_z * (z - distance_hub2mb)**3 / 6.0 + -f_mb2_z * (z - distance_hub2mb - L_mb)**3 / 6.0 + W_ms / (L_ms + L_mb) / 24.0 * z**4
    
    @staticmethod
    def gx2(F_z, W_r, gamma, M_y, f_mb1_z, f_mb2_z, distance_hub2mb, W_ms, L_ms, L_mb, z):
        return -F_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb1_z * (z - distance_hub2mb)**2 / 2.0 - f_mb2_z * (z - distance_hub2mb - L_mb)**2 / 2.0 + W_ms / (L_ms + L_mb) / 6.0 * z**3

    #----------------------------
    
    def size_LSS_4pt_Loop_1(self):
        # Distances
        self.L_as = self.L_ms / 2.0  # distance from main bearing to shaft center
        self.L_cu = self.L_ms + 0.5
        # L_cu: distance from upwind main bearing to upwind carrier bearing 
        #   0.5 meter is an estimation 
        #   to add as an input
        self.L_cd = self.L_cu + 0.5
        # L_cd: distance from upwind main bearing to downwind carrier bearing
        #   0.5 meter is an estimation 
        #   to add as an input

        # Weight properties
        self.rotorWeight = self.rotor_mass * self.g  # rotor weight
        self.lssWeight = pi / 3.0 \
            * (self.D_max**2.0 + self.D_min**2.0 + self.D_max * self.D_min) \
            * self.L_ms * self.density * self.g / 4.0 # weight of a solid tapered cylinder
        self.lss_mass = self.lssWeight / self.g
        self.gearboxWeight = self.gearbox_mass * self.g  # gearbox weight
        #self.gearboxWeight = self.gearboxWeight  # needed in fatigue functions
        self.carrierWeight = self.carrier_mass * self.g  # carrier weight
        self.shrinkDiscWeight = self.shrink_disc_mass * self.g

        # define LSS
        x_ms = np.linspace(self.distance_hub2mb, 
                           self.distance_hub2mb + self.L_ms, 
                           self.len_pts) # len_pts evenly spaced along mainshaft between bearings
        x_rb = np.linspace(0.0, 
                           self.distance_hub2mb, 
                           self.len_pts) # len_pts evenly spaced along mainshaft from rotor to upwind bearing
        y_gp = np.linspace(0, self.L_gp, self.len_pts) # not used

        cosSA = cos(self.shaft_angle)
        sinSA = sin(self.shaft_angle)
        
        # implement Eqs. 2.22 (Eq. 2.18 in 2015 rpt)
        F_mb_x = -self.rotor_thrust - self.rotorWeight * sinSA # not used
        self.F_mb_y = self.rotor_bending_moment_z / self.L_bg \
            - self.rotor_force_y * (self.L_bg + self.distance_hub2mb) / self.L_bg
        self.F_mb_z = (-self.rotor_bending_moment_y 
                       + self.rotorWeight * (cosSA * (self.distance_hub2mb + self.L_bg) + sinSA * self.H_gb)               
                       + self.lssWeight * cosSA * (self.L_bg - self.L_as) 
                       + self.shrinkDiscWeight * cosSA * (self.L_bg - self.L_ms) 
                       - self.gearboxWeight * cosSA * self.L_gb 
                       - self.rotor_force_z * cosSA * (self.L_bg + self.distance_hub2mb)) / self.L_bg

        # F_gb_(xyz) not used
        F_gb_x = -(self.lssWeight + self.shrinkDiscWeight + self.gearboxWeight) * sinSA                 
        F_gb_y = -self.F_mb_y - self.rotor_force_y
        F_gb_z = -self.F_mb_z \
                 + (self.shrinkDiscWeight + self.rotorWeight + self.gearboxWeight + self.lssWeight) * cosSA \
                 - self.rotor_force_z

        # Bending moments along main shaft in pitching and yaw directions
        
        My_ms = np.zeros(2 * self.len_pts)
        Mz_ms = np.zeros(2 * self.len_pts)

        # Eqs. 2.23, 2.24 (2.19, 2.20 in 2015 rpt)
        for k in range(self.len_pts):
            My_ms[k] = -self.rotor_bending_moment_y \
                + self.rotorWeight * cosSA * x_rb[k] \
                + 0.5 * self.lssWeight / self.L_ms * x_rb[k]**2 \
                - self.rotor_force_z * x_rb[k]
            Mz_ms[k] = -self.rotor_bending_moment_z \
                - self.rotor_force_y * x_rb[k]

        for j in range(self.len_pts):
            My_ms[j + self.len_pts] = -self.rotor_force_z * x_ms[j] \
                - self.rotor_bending_moment_y \
                + self.rotorWeight * cosSA * x_ms[j] \
                - self.F_mb_z * (x_ms[j] - self.distance_hub2mb) \
                + 0.5 * self.lssWeight / self.L_ms * x_ms[j]**2
            Mz_ms[j + self.len_pts] = -self.rotor_bending_moment_z \
                - self.F_mb_y * (x_ms[j] - self.distance_hub2mb) \
                - self.rotor_force_y * x_ms[j]

        # Shaft diameters (section 2.2.3.2)
        
        x_shaft = np.concatenate([x_rb, x_ms]) # not used

        MM_max = np.amax((My_ms**2 + Mz_ms**2)**0.5)
        Index = np.argmax((My_ms**2 + Mz_ms**2)**0.5) # not used

        MM_min = ((My_ms[-1]**2 + Mz_ms[-1]**2)**0.5)
        # Design shaft OD
        MM = MM_max
        #self.D_max = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
        #                                                     3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m
        self.D_max = computeD(MM, self.rotor_bending_moment_x, self.Sy, self.n_safety)
        
        # OD at end
        MM = MM_min
        #self.D_min = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
        #                                                     3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m
        self.D_min = computeD(MM, self.rotor_bending_moment_x, self.Sy, self.n_safety)
        #if self.debug:
        #    sys.stderr.write('size4pt_1: MM_max {:.1f} MM_min {:.1f} D_max {:.3f} D_min {:.3f}\n'.format(MM_max, MM_min, self.D_max, self.D_min))
        
        # Estimate ID
        self.D_in = self.shaft_ratio * self.D_max
        self.D_max = (self.D_max**4 + self.D_in**4)**0.25
        self.D_min = (self.D_min**4 + self.D_in**4)**0.25

        self.lssWeight_new = ((pi / 3) * (self.D_max**2 + self.D_min**2 + self.D_max * self.D_min) * self.L_ms / 4 
                            - (pi / 4 * (self.D_in**2) * self.L_ms)) * self.g * self.density

        D1 = self.deflection(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                        self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.distance_hub2mb + self.L_ms)
        D2 = self.deflection(self.rotor_force_z, self.rotorWeight, self.shaft_angle,
                        self.rotor_bending_moment_y, self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.distance_hub2mb)
        C1 = -(D1 - D2) / self.L_ms
        C2 = D2 - (C1 * self.distance_hub2mb)

        I_2 = pi / 64.0 * (self.D_max**4 - self.D_in**4) # hollow shaft inertia (Eq. 2.46 (Eq. 9.5 in 2015 rpt))

        self.theta_y = np.zeros(self.len_pts)
        d_y = np.zeros(self.len_pts)

        for kk in range(self.len_pts):
            self.theta_y[kk] = self.gx(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                  self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, C1, x_ms[kk]) / self.E / I_2
            d_y[kk] = (self.deflection(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                  self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, x_ms[kk]) + C1 * x_ms[kk] + C2) / self.E / I_2

    #----------------------------
    
    def size_LSS_4pt_Loop_2(self):

        # Distances
        L_as = (self.L_ms_gb + self.L_mb) / 2.0
        L_cu = (self.L_ms_gb + self.L_mb) + 0.5
        L_cd = L_cu + 0.5 # not used

        # Weight
        self.lssWeight_new = (pi / 3 * (self.D_max**2 + self.D_min**2 + self.D_max * self.D_min) * (self.L_ms_gb + self.L_mb) / 4 
                           - (pi / 4 * (self.D_in**2) * (self.L_ms_gb + self.L_mb))) * self.g * self.density
                           # weight of tapered cylinder with hole of diameter D_in

        # define LSS
        x_ms = np.linspace(self.distance_hub2mb + self.L_mb, 
                           self.distance_hub2mb + self.L_mb + self.L_ms_gb, 
                           self.len_pts)
        x_mb = np.linspace(self.distance_hub2mb, 
                           self.distance_hub2mb + self.L_mb, 
                           self.len_pts)
        x_rb = np.linspace(0.0, self.distance_hub2mb, self.len_pts)
        y_gp = np.linspace(0, self.L_gp, self.len_pts) # not used

        cosSA = cos(self.shaft_angle)
        sinSA = sin(self.shaft_angle)
        
        F_mb2_x = -self.rotor_thrust - \
            self.rotorWeight * sinSA
        F_mb2_y = -self.rotor_bending_moment_z / self.L_mb \
            + self.rotor_force_y * (self.distance_hub2mb) / self.L_mb
        F_mb2_z = (self.rotor_bending_moment_y 
                   - self.rotorWeight * cosSA * self.distance_hub2mb
                   - self.lssWeight * L_as * cosSA 
                   - self.shrinkDiscWeight * (self.L_mb + self.L_ms_0) * cosSA
                   + self.gearboxWeight * cosSA * self.L_gb 
                   + self.rotor_force_z * cosSA * self.distance_hub2mb) / self.L_mb

        F_mb1_x = 0.0
        F_mb1_y = -self.rotor_force_y - F_mb2_y
        F_mb1_z = (self.rotorWeight + self.lssWeight + self.shrinkDiscWeight) * cosSA \
            - self.rotor_force_z - F_mb2_z

        # F_mb(12)_x, F_gb_(xyz) not used
        F_gb_x = -(self.lssWeight + self.shrinkDiscWeight + self.gearboxWeight) * sinSA
        F_gb_y = -self.F_mb_y - self.rotor_force_y
        F_gb_z = -self.F_mb_z \
            + (self.shrinkDiscWeight + self.rotorWeight + self.gearboxWeight + self.lssWeight) * cosSA \
            - self.rotor_force_z

        if self.debug:
            sys.stderr.write('LSS4L2: s.F_mb_y {:.1f} F_mb1_y {:.1f} F_mb2_y {:.1f}\n'.format(self.F_mb_y, F_mb1_y, F_mb2_y))

        # Bending moments along main shaft in pitching and yaw directions
        
        My_ms = np.zeros(3 * self.len_pts)
        Mz_ms = np.zeros(3 * self.len_pts)

        for k in range(self.len_pts):
            My_ms[k] = -self.rotor_force_z * x_rb[k] \
                      + self.rotorWeight * cosSA * x_rb[k] \
                      - self.rotor_bending_moment_y \
                      + 0.5 * self.lssWeight / (self.L_mb + self.L_ms_0) * x_rb[k]**2
                
            Mz_ms[k] = -self.rotor_bending_moment_z \
                      - self.rotor_force_y * x_rb[k]

        for j in range(self.len_pts):
            My_ms[j + self.len_pts] = -self.rotor_force_z * x_mb[j] \
                + self.rotorWeight * cosSA * x_mb[j] \
                - self.rotor_bending_moment_y \
                + 0.5 * self.lssWeight / (self.L_mb + self.L_ms_0) * x_mb[j]**2 \
                - F_mb1_z * (x_mb[j] - self.distance_hub2mb)
                
            Mz_ms[j + self.len_pts] = -self.rotor_bending_moment_z \
                                     - self.rotor_force_y * x_mb[j] \
                                     - F_mb1_y * (x_mb[j] - self.distance_hub2mb)

        for l in range(self.len_pts):
            My_ms[l + 2 * self.len_pts] = -self.rotor_force_z * x_ms[l] \
                + self.rotorWeight * cosSA * x_ms[l] \
                - self.rotor_bending_moment_y \
                + 0.5 * self.lssWeight / (self.L_mb + self.L_ms_0) * x_ms[l]**2 \
                - F_mb1_z * (x_ms[l] - self.distance_hub2mb) \
                - F_mb2_z * (x_ms[l] - self.distance_hub2mb - self.L_mb)
                
            Mz_ms[l + 2 * self.len_pts] = -self.rotor_bending_moment_z \
                                         - self.rotor_force_y * x_ms[l] \
                                         - self.F_mb_y * (x_ms[l] - self.distance_hub2mb)
            # need F_mb[12]_y????
            # Following statement is probably correct, but doesn't make a difference because other Mz_ms values are larger
            Mz_ms[l + 2 * self.len_pts] = -self.rotor_bending_moment_z \
                                         - self.rotor_force_y * x_ms[l] \
                                         - F_mb1_y * (x_ms[l] - self.distance_hub2mb) \
                                         - F_mb2_y * (x_ms[l] - self.distance_hub2mb - self.L_mb)

        x_shaft = np.concatenate([x_rb, x_mb, x_ms]) # not used

        MM_max = np.amax((My_ms**2 + Mz_ms**2)**0.5)
        Index = np.argmax((My_ms**2 + Mz_ms**2)**0.5) # not used

        MM_min = ((My_ms[-1]**2 + Mz_ms[-1]**2)**0.5)

        MM_med = ((My_ms[-1 - self.len_pts]**2 +
                   Mz_ms[-1 - self.len_pts]**2)**0.5)
                   
        # Design Shaft OD using static loading and distortion energy theory
        #MM = MM_max
        #self.D_max = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
        #                                                     3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m
        self.D_max = computeD(MM_max, self.rotor_bending_moment_x, self.Sy, self.n_safety)

        # OD at end
        #MM = MM_min
        #self.D_min = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
        #                                                     3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m
        self.D_min = computeD(MM_min, self.rotor_bending_moment_x, self.Sy, self.n_safety)

        #MM = MM_med
        #self.D_med = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb / 1000)**2 +
        #                                                     3.0 * (self.rotor_bending_moment_x * self.u_knm_inlb / 1000)**2)**0.5)**(1.0 / 3.0) * self.u_in_m
        self.D_med = computeD(MM_med, self.rotor_bending_moment_x, self.Sy, self.n_safety)

        #if self.debug:
        #    sys.stderr.write('size4pt_2: MM_max {:.1f} MM_min {:.1f} MM_med {:.1f} D_max {:.3f} D_min {:.3f} D_med {:.3f}\n'.format(
        #        MM_max, MM_min, MM_med, self.D_max, self.D_min, self.D_med))

        # Estimate ID
        self.D_in = self.shaft_ratio * self.D_max
        self.D_max = (self.D_max**4 + self.D_in**4)**0.25
        self.D_min = (self.D_min**4 + self.D_in**4)**0.25
        self.D_med = (self.D_med**4 + self.D_in**4)**0.25

        self.lssWeight_new = (pi / 12.0 * self.L_mb * (self.D_max**2 + self.D_med**2 + self.D_max * self.D_med) 
                            - pi /  4.0 * self.D_in**2 * self.L_mb) * self.g * self.density

        # deflection between mb1 and mb2
        D11 = self.deflection1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                          F_mb1_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, self.distance_hub2mb + self.L_mb)
        D21 = self.deflection1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                          F_mb1_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, self.distance_hub2mb)
        C11 = -(D11 - D21) / self.L_mb
        C21 = -D21 - C11 * (self.distance_hub2mb)

        I_2 = pi / 64.0 * (self.D_max**4 - self.D_in**4) # hollow shaft inertia (Eq. 2.46 (Eq. 9.5 in 2015 rpt))

        self.theta_y = np.zeros(2 * self.len_pts)
        d_y = np.zeros(2 * self.len_pts)

        for kk in range(self.len_pts):
            self.theta_y[kk] = self.gx1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                   F_mb1_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, C11, x_mb[kk]) / self.E / I_2
            d_y[kk] = (self.deflection1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                   F_mb1_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, x_mb[kk]) + C11 * x_mb[kk] + C21) / self.E / I_2

        D12 = self.deflection2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                          F_mb1_z, F_mb2_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, self.distance_hub2mb + self.L_mb)
        D22 = self.gx2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                  F_mb1_z, F_mb2_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, self.distance_hub2mb + self.L_mb)
        C12 = self.gx1(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                  F_mb1_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, C11, x_mb[-1]) - D22
        C22 = -D12 - C12 * (self.distance_hub2mb + self.L_mb)

        for kk in range(self.len_pts):
            self.theta_y[kk + self.len_pts] = (self.gx2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                                   F_mb1_z, F_mb2_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, x_ms[kk]) + C12) / self.E / I_2
            # d_y[] computed but discarded
            d_y[kk + self.len_pts] = (self.deflection2(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                                  F_mb1_z, F_mb2_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.L_mb, x_ms[kk]) + C12 * x_ms[kk] + C22) / self.E / I_2

    #----------------------------
    
    def compute(self, rotor_diameter, rotor_mass, rotor_thrust, rotor_force_y, rotor_force_z, 
                      rotor_bending_moment_x, rotor_bending_moment_y, rotor_bending_moment_z, \
                      overhang, machine_rating, drivetrain_efficiency, \
                      gearbox_mass, carrier_mass, gearbox_cm, gearbox_length, \
                      shrink_disc_mass, flange_length, distance_hub2mb, shaft_angle, shaft_ratio, \
                      hub_flange_thickness):

        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.rotor_mass = rotor_mass #Float(iotype='in', units='kg', desc='rotor mass')
        self.rotor_bending_moment_x = rotor_bending_moment_x #Float(iotype='in', units='N*m', desc='The bending moment about the x axis')
        self.rotor_bending_moment_y = rotor_bending_moment_y #Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
        self.rotor_bending_moment_z = rotor_bending_moment_z #Float(iotype='in', units='N*m', desc='The bending moment about the z axis')
        self.rotor_thrust = rotor_thrust #Float(iotype='in', units='N', desc='The force along the x axis applied at hub center')
        self.rotor_force_y = rotor_force_y #Float(iotype='in', units='N', desc='The force along the y axis applied at hub center')
        self.rotor_force_z = rotor_force_z #Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
        self.overhang = overhang #Float(iotype='in', units='m', desc='Overhang distance')
        self.machine_rating = machine_rating #Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
        self.drivetrain_efficiency = drivetrain_efficiency #Float(iotype = 'in', desc = 'overall drivettrain efficiency')
        self.gearbox_mass = gearbox_mass #Float(iotype='in', units='kg', desc='Gearbox mass')
        self.carrier_mass = carrier_mass #Float(iotype='in', units='kg', desc='Carrier mass')
        self.gearbox_cm = gearbox_cm #Array(iotype = 'in', units = 'm', desc = 'center of mass of gearbox')
        self.gearbox_length = gearbox_length #Float(iotype='in', units='m', desc='gearbox length')
        self.shrink_disc_mass = shrink_disc_mass #Float(iotype='in', units='kg', desc='Mass of the shrink disc')# shrink disk or flange addtional mass
        self.flange_length = flange_length #Float(iotype ='in', units='m', desc ='flange length')
        self.distance_hub2mb = distance_hub2mb #Float(iotype='in', units='m', desc='distance between hub center and upwind main bearing')
        self.shaft_angle = shaft_angle #Float(iotype='in', units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
        self.shaft_ratio = shaft_ratio #Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
        self.hub_flange_thickness = hub_flange_thickness 

        # outputs
        self.design_torque = 0.0 #Float(iotype='out', units='N*m', desc='lss design torque')
        self.design_bending_load = 0.0 #Float(iotype='out', units='N', desc='lss design bending load')
        self.length = 0.0 #Float(iotype='out', units='m', desc='lss length')
        self.diameter1 = 0.0 #Float(iotype='out', units='m', desc='lss outer diameter at main bearing')
        self.diameter2 = 0.0 #Float(iotype='out', units='m', desc='lss outer diameter at second bearing')
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I =  np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.mb1_facewidth = 0.0 #Float(iotype='out', units='m', desc='facewidth of upwind main bearing') 
        self.mb2_facewidth = 0.0 #Float(iotype='out', units='m', desc='facewidth of main bearing')     
        self.mb1_mass = 0.0 #Float(iotype='out', units = 'kg', desc='main bearing mass')
        self.mb2_mass = 0.0 #Float(iotype='out', units = 'kg', desc='second bearing mass')
        self.mb1_cm = np.zeros(3) #Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 1 center of mass')
        self.mb2_cm = np.zeros(3) #Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 2 center of mass')

        # input parameters

        if self.distance_hub2mb == 0:  # distance from hub center to main bearing
            #distance_hub2mb = 0.007835 * self.rotor_diameter + 0.9642
            distance_hub2mb = get_distance_hub2mb(self.rotor_diameter, False)  # [0] not needed without derivative
            self.distance_hub2mb = distance_hub2mb # see if this returns modified value to prob
        else:
            distance_hub2mb = self.distance_hub2mb

        # If user does not know important moments, a crude approximation is made
        if self.rotor_mass > 0 and self.rotor_bending_moment_y == 0:
            self.rotor_bending_moment_y = get_My(self.rotor_mass, distance_hub2mb)

        if self.rotor_mass > 0 and self.rotor_bending_moment_z == 0:
            self.rotor_bending_moment_z = get_Mz(self.rotor_mass, distance_hub2mb)

        if self.rotor_mass == 0:
            [self.rotor_mass] = get_rotor_mass(self.machine_rating, False)

        if self.flange_length == 0:
            ''' 2014 Report gives flange_length as 0.9918 * exp(0.0068*rotor_diameter) (after Eq.2.38) 
                which gives lengths roughly 3 times as large as the following code'''
            self.flange_length = 0.3 * (self.rotor_diameter / 100.0)**2.0 \
                - 0.1 * (self.rotor_diameter / 100.0) \
                + 0.4 # (following Eq. 2.32 in 2015 rpt) 
            if self.debug:
                sys.stderr.write('MSFlangeLen (approx): {:.2f} m\n'.format(self.flange_length))

        # constants
        self.g = 9.81 # m/s^2
        
        # material properties
        self.E = 2.1e11
        self.density = 7800.0 # density of steel in kg/m^3
        self.Sy = 66000  # *self.S_ut/700e6 #66000 #psi # approx tensile strength of steel in psi

        # Safety factors
        self.n_safety = 2.5  # According to AGMA, takes into account the peak load safety factor
        self.n_safety_brg = 1.0

        # unit conversion
        self.u_knm_inlb = 8850.745454036  # 1 kN-m = 8850.74577 lb-in
        self.u_in_m = 0.0254000508001  # 1 in = 0.0254 m

        # initialization for iterations
        self.L_ms_new = 0.0
        self.L_ms_0 = 0.5  # main shaft length downwind of main bearing
        self.L_ms = self.L_ms_0
        self.len_pts = 101
        self.D_max = 1
        self.D_min = 0.2
        self.D_med = 0.
        self.D_in  = 0.

        tol = 1e-4
        check_limit = 1.0
        dL = 0.05
        counter = 0
        N_count = 50 # not used
        N_count_2 = 2

        # Distances
        # distance from first main bearing to gearbox yokes
        # to add as an input
        self.L_bg = 6.11 - distance_hub2mb
        self.L_as = self.L_ms / 2.0  # distance from main bearing to shaft center
        self.L_gb = 0.0  # distance to gearbox center from trunnions in x-dir # to add as an input
        self.H_gb = 1.0  # distance to gearbox center from trunnions in z-dir # to add as an input
        self.L_gp = 0.825  # distance from gearbox coupling to gearbox trunnions - only used for y_gp, which is not used
        
        # distance from upwind main bearing to upwind carrier bearing
        #   0.5 meter is an estimation 
        #   to add as an input
        self.L_cu = self.L_ms + 0.5
        # distance from upwind main bearing to downwind carrier bearing
        #   0.5 meter is an estimation 
        #   to add as an input
        self.L_cd = self.L_cu + 0.5

        # Main bearing deflection check
        Bearing_Limit = bearing_defl_check(self.mb1Type)

        # Second bearing deflection check
        Bearing_Limit2 = bearing_defl_check(self.mb2Type)

        length_max = self.overhang - distance_hub2mb + \
            (self.gearbox_cm[0] - self.gearbox_length / 2.)  # modified length limit 7/29/14

        while abs(check_limit) > tol and self.L_ms_new < length_max:
            counter = counter + 1
            if self.L_ms_new > 0:
                self.L_ms = self.L_ms_new
            else:
                self.L_ms = self.L_ms_0

            self.size_LSS_4pt_Loop_1()

            check_limit = abs(abs(self.theta_y[-1]) - Bearing_Limit / self.n_safety_brg)

            if check_limit < 0:  # 'if' and 'else' clauses are identical
                self.L_ms_new = self.L_ms + dL
            else:
                self.L_ms_new = self.L_ms + dL

        # Initialization
        self.L_mb = self.L_ms_new
        counter_ms = 0
        check_limit_ms = 1.0
        self.L_mb_new = 0.0
        self.L_mb_0 = self.L_mb  # main shaft length
        self.L_ms = self.L_ms_new
        dL_ms = 0.05
        dL = 0.0025

        while abs(check_limit_ms) > tol and self.L_mb_new < length_max:
            counter_ms = counter_ms + 1
            if self.L_mb_new > 0:
                self.L_mb = self.L_mb_new
            else:
                self.L_mb = self.L_mb_0

            counter = 0.0
            check_limit = 1.0
            self.L_ms_gb_new = 0.0
            self.L_ms_0 = 0.5  # mainshaft length
            self.L_ms = self.L_ms_0

            # check_limit(_ms) are calculated with abs(), so never less than 0 - 'if' and 'else' results are equal anyway
            while abs(check_limit) > tol and counter < N_count_2:
                counter = counter + 1
                if self.L_ms_gb_new > 0.0:
                    self.L_ms_gb = self.L_ms_gb_new
                else:
                    self.L_ms_gb = self.L_ms_0

                self.size_LSS_4pt_Loop_2()

                check_limit = abs(abs(self.theta_y[-1]) - Bearing_Limit / self.n_safety_brg)

                if check_limit < 0:  # 'if' and 'else' clauses are identical
                    #self.L_ms__gb_new = self.L_ms_gb + dL
                    self.L_ms_gb_new = self.L_ms_gb + dL
                else:
                    #self.L_ms__gb_new = self.L_ms_gb + dL
                    self.L_ms_gb_new = self.L_ms_gb + dL

                check_limit_ms = abs(abs(self.theta_y[-1]) - Bearing_Limit2 / self.n_safety_brg)

                if check_limit_ms < 0:  # 'if' and 'else' clauses are identical
                    self.L_mb_new = self.L_mb + dL_ms
                else:
                    self.L_mb_new = self.L_mb + dL_ms

        # Resize low speed shaft for bearings
        [self.D_max_a, facewidth_max, bearing1mass] = resize_for_bearings(self.D_max,  self.mb1Type, False)    
        [self.D_med_a, facewidth_med, bearing2mass] = resize_for_bearings(self.D_med,  self.mb2Type, False)       

        lss_vol_new = (pi / 3) * (self.D_max_a**2 + self.D_med_a**2 + self.D_max_a * self.D_med_a) * (self.L_mb - (facewidth_max + facewidth_med) / 2) / 4 \
                     + (pi / 4) * (self.D_max_a**2 - self.D_in**2) * facewidth_max \
                     + (pi / 4) * (self.D_med_a**2 - self.D_in**2) * facewidth_med \
                     - (pi / 4) * (self.D_in**2) * (self.L_mb + (facewidth_max + facewidth_med) / 2)
                     # volume of tapered cylinder + two bearing sections - internal hole
        lss_mass_new = lss_vol_new * self.density
        
        # begin bearing routine with updated shaft mass

        if useFlangeModel:
            self.flange_length, mass_flange, cm_flange, cost_flange = mainshaftFlangeCalc(self.D_in, 
                                                                    self.D_max_a, 
                                                                    self.hub_flange_thickness * FLANGE_THICK_FACTOR, 
                                                                    debug=self.debug)
            self.mass = lss_mass_new + mass_flange
        else:
            self.mass = lss_mass_new * 1.33  # add flange mass

        # add facewidths and flange
        self.lss_length = self.L_mb_new \
            + (facewidth_max + facewidth_med) / 2 \
            + self.flange_length
        self.D_outer = self.D_max
        self.D_in    = self.D_in

        self.diameter1 = self.D_max_a
        self.diameter2 = self.D_med_a

        # calculate mass properties
        downwind_location = np.array([self.gearbox_cm[0] - self.gearbox_length / 2., self.gearbox_cm[1], self.gearbox_cm[2]])

        mb1_cm = np.zeros(3)  # upwind
        mb1_cm[0] = downwind_location[0] - (self.L_mb_new + facewidth_med / 2) * cos(self.shaft_angle)
        mb1_cm[1] = downwind_location[1]
        mb1_cm[2] = downwind_location[2] + (self.L_mb_new + facewidth_med / 2) * sin(self.shaft_angle)
        self.mb1_cm = mb1_cm

        mb2_cm = np.zeros(3)  # downwind
        mb2_cm[0] = downwind_location[0] - facewidth_med * .5 * cos(self.shaft_angle)
        mb2_cm[1] = downwind_location[1]
        mb2_cm[2] = downwind_location[2] + facewidth_med * .5 * sin(self.shaft_angle)
        self.mb2_cm = mb2_cm

        cm = np.zeros(3)
        # From solid models, center of mass with flange (not including shrink
        # disk) very nearly .65*total_length
        # TODO 2019 07 18 - we have cm_flange - can we use it instead of this approximation?
        cm[0] = downwind_location[0] - 0.65 * self.lss_length * cos(self.shaft_angle)
        cm[1] = downwind_location[1]
        cm[2] = downwind_location[2] + 0.65 * self.lss_length * sin(self.shaft_angle)

        # including shrink disk mass
        self.cm[0] = (cm[0] * self.mass + downwind_location[0] * self.shrink_disc_mass) \
                      / (self.mass + self.shrink_disc_mass)
        self.cm[1] = cm[1]
        self.cm[2] = (cm[2] * self.mass + downwind_location[2] * self.shrink_disc_mass) \
                      / (self.mass + self.shrink_disc_mass)
        self.mass += self.shrink_disc_mass

        I = np.zeros(3)
        I[0] = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1] = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0
                             + (4.0 / 3.0) * (self.lss_length ** 2.0)) / 16.0
        I[2] = I[1]
        self.I = I

        self.mb1_facewidth = facewidth_max
        self.mb2_facewidth = facewidth_med

        self.mb1_mass = bearing1mass
        self.mb2_mass = bearing2mass

        ''' self.length was never set - instead, the code was working on self.lss_length, but returning self.length 
            2019 06 11 GNS
        '''
        self.length = self.lss_length # quick fix

        if self.debug:
            sys.stderr.write('LSS4:: Len {:.3f} m (iter) + {:.3f} m (facewidth) + {:.3f} m (flange)\n'.format(self.L_mb_new, 
                                                0.5*(facewidth_max + facewidth_med), self.flange_length))
            lssfmt = 'LSS4:: Len {:.3f} m Dia1 {:.2f} m Dia2 {:.2f} m  ID {:.2f} m Mass {:.1f} kg MB1Mass {:.1f} kg MB2Mass {:.1f} kg  F_mb_y {:.1f} N  F_mb_z {:.1f} N\n'
            sys.stderr.write(lssfmt.format(self.length, self.diameter1, self.diameter2, self.D_in, self.mass, 
                                               self.mb1_mass, self.mb2_mass, self.F_mb_y, self.F_mb_z))
            sys.stderr.write(' hub2mb  {:6.3f}\n L_mb    {:6.3f}\n L_ms_gb {:6.3f}\n'.format(self.distance_hub2mb, 
                             self.L_mb_new, self.L_ms_gb_new))
            sys.stderr.write(' fwidth1 {:6.3f}\n fwidth2 {:6.3f}\n flange  {:6.3f}\n'.format(self.mb1_facewidth, 
                             self.mb2_facewidth, self.flange_length))

        return (self.design_torque, self.design_bending_load, self.length, self.diameter1, self.diameter2, self.mass, self.cm, self.I, \
                self.mb1_facewidth, self.mb2_facewidth, self.mb1_mass, self.mb2_mass, self.mb1_cm, self.mb2_cm)

#-------------------------------------------------------------------------

# Size 3 pt suspension low speed shaft
class LowSpeedShaft3pt(object):
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''
    '''
    2019 04 18 GNS
    Many variables changed to conform to LowSpeedShaft4pt:
      - names changed
      - local vbls are now members of self
    Bearing masses returned (self.mb[12]_mass) do NOT include bearing housings. These will be added by class MainBearing.
    '''

    def __init__(self, mb1Type, IEC_Class, debug=False):
        
        super(LowSpeedShaft3pt, self).__init__()

        # set LSS configuration parameters
        self.mb1Type = mb1Type #Enum('SRB',('CARB','TRB1','TRB2','SRB','CRB','RB'),iotype='in',desc='Main bearing type')
        self.IEC_Class = IEC_Class #Enum('A',('A','B','C'),iotype='in',desc='IEC class letter: A, B, or C')
        self.debug = debug
        
    #----------------------------------------------------
    # Deflection functions
    #----------------------------------------------------
    '''
    These functions were defined inline in LowSpeedShaft3pt() and LowSpeedShaft4pt().
      Note that fx() and gx() from size_LSS_3pt() appear to be identical to deflection() and gx() from size_LSS_4pt_Loop_1()
      
    See notes above for LowSpeedShaft4pt()
    '''
    # from size_LSS_3pt()
    
    @staticmethod
    def fx(F_r_z, W_r, gamma, M_y, f_mb_z, distance_hub2mb, W_ms, L_ms, z):
        #return -F_r_z * z**3 / 6.0 + W_r * cos(gamma) * z**3 / 6.0 - M_y * z**2 / 2.0 - f_mb_z * (z - distance_hub2mb)**3 / 6.0 + W_ms / (L_ms + distance_hub2mb) / 24.0 * z**4
        return -F_r_z * z**3 / 6.0 \
               + W_r * cos(gamma) * z**3 / 6.0 \
               - M_y * z**2 / 2.0 \
               - f_mb_z * (z - distance_hub2mb)**3 / 6.0 \
               + W_ms / (L_ms + distance_hub2mb) / 24.0 * z**4
    
    '''
      gx from size_LSS_3pt is the same as gx from size_LSS_4pt_Loop_1 (except for the name of the first argument)
    '''
    @staticmethod
    def gx(F_r_z, W_r, gamma, M_y, f_mb_z, distance_hub2mb, W_ms, L_ms, C1, z):
        #return -F_r_z * z**2 / 2.0 + W_r * cos(gamma) * z**2 / 2.0 - M_y * z - f_mb_z * (z - distance_hub2mb)**2 / 2.0 + W_ms / (L_ms + distance_hub2mb) / 6.0 * z**3 + C1
        return -F_r_z * z**2 / 2.0 \
               + W_r * cos(gamma) * z**2 / 2.0 \
               - M_y * z \
               - f_mb_z * (z - distance_hub2mb)**2 / 2.0 \
               + W_ms / (L_ms + distance_hub2mb) / 6.0 * z**3 \
               + C1

    #----------------------------
    
    def size_LSS_3pt(self):
        # Distances
        # distance from hub center to gearbox yokes
        self.L_bg = 6.11 * (self.machine_rating / 5.0e3)
        L_as = self.L_ms / 2.0  # distance from main bearing to shaft center
        H_gb = 1.0  # distance to gearbox center from trunnions in z-dir
        self.L_gp = 0.825  # distance from gearbox coupling to gearbox trunnions - only used for y_gp, which is not used
        self.L_cu = self.L_ms + 0.5
        self.L_cd = self.L_cu + 0.5
        self.L_gb = 0

        # Weight properties
        self.rotorWeight = self.rotor_mass * self.g
        self.lss_mass = pi / 3.0 * (self.D_max**2.0 + self.D_min**2.0 + self.D_max * self.D_min) \
                            * self.L_ms * self.density / 4.0
                            # volume of solid tapered cylinder
        self.lssWeight = self.lss_mass * self.g  # LSS weight
        self.shrinkDiscWeight = self.shrink_disc_mass * self.g  # shrink disc weight
        self.gearboxWeight = self.gearbox_mass * self.g  # gearbox weight
        self.carrierWeight = self.carrier_mass * self.g

        #len_pts = 101
        x_ms = np.linspace(self.distance_hub2mb, self.L_ms + self.distance_hub2mb, self.len_pts)
        x_rb = np.linspace(0.0, self.distance_hub2mb, self.len_pts)
        y_gp = np.linspace(0, self.L_gp, self.len_pts) # not used

        cosSA = cos(self.shaft_angle)
        sinSA = sin(self.shaft_angle)
        
        #len_my = np.arange(1,len(self.rotor_bending_moment_y)+1)
        F_mb_x = -self.rotor_thrust - self.rotorWeight * sinSA # not used
        self.F_mb_y = self.rotor_bending_moment_z / self.L_bg \
                    - self.rotor_force_y * (self.L_bg + self.distance_hub2mb) / self.L_bg
        self.F_mb_z = (-self.rotor_bending_moment_y \
                       + self.rotorWeight * (cosSA * (self.distance_hub2mb + self.L_bg) + (sinSA * H_gb)) \
                       + self.lssWeight * (self.L_bg - L_as) * cosSA \
                       + self.shrinkDiscWeight * cosSA * (self.L_bg - self.L_ms) \
                       - self.gearboxWeight * cosSA * self.L_gb \
                       - self.rotor_force_z * cosSA * (self.L_bg + self.distance_hub2mb)) / self.L_bg
        #if self.debug:
        #    sys.stderr.write('F_mb: X {:.1f} Y {:.1f} Z {:.1f} LSSwt {:.1f} rfZ {:.1f}\n'.format(F_mb_x, 
        #                     self.F_mb_y, self.F_mb_z, self.lssWeight, self.rotor_force_z))

        # F_gb_[xyz] not used
        F_gb_x = -(self.lssWeight + self.shrinkDiscWeight + self.gearboxWeight) * sinSA
        F_gb_y = -self.F_mb_y - self.rotor_force_y
        F_gb_z = -self.F_mb_z \
                 + (self.lssWeight + self.shrinkDiscWeight + self.gearboxWeight + self.rotorWeight) * cosSA \
                 - self.rotor_force_z

        # carrier bearing loads
        F_cu_z = (self.lssWeight * cosSA \
            + self.shrinkDiscWeight * cosSA \
            + self.gearboxWeight * cosSA) \
            - self.F_mb_z \
            - self.rotor_force_z \
            - (-self.rotor_bending_moment_y \
              - self.rotor_force_z * cosSA * self.distance_hub2mb \
              + self.lssWeight * (self.L_bg - L_as) * cosSA \
              - self.carrierWeight * cosSA * self.L_gb) \
            / (1 - self.L_cu / self.L_cd)
            #- (-self.rotor_bending_moment_y - self.rotor_force_z * cos(self.shaft_angle) * self.distance_hub2mb + self.lssWeight *
            # (self.L_bg - L_as) * cos(self.shaft_angle) - self.carrierWeight * cos(self.shaft_angle) * self.L_gb) / (1 - self.L_cu / self.L_cd)

        #F_cd_z = (self.lssWeight * cos(self.shaft_angle) + self.shrinkDiscWeight * cos(self.shaft_angle) +
        #          self.gearboxWeight * cos(self.shaft_angle)) - self.F_mb_z - self.rotor_force_z - F_cu_z
        F_cd_z = cosSA * (self.lssWeight + self.shrinkDiscWeight + self.gearboxWeight) \
                 - self.F_mb_z \
                 - self.rotor_force_z \
                 - F_cu_z  # not used

        # Bending moments along main shaft in pitching and yaw directions
        
        My_ms = np.zeros(2 * self.len_pts)
        Mz_ms = np.zeros(2 * self.len_pts)

        for k in range(self.len_pts):
            My_ms[k] = -self.rotor_bending_moment_y \
              + self.rotorWeight * cosSA * x_rb[k] \
              +  0.5 * self.lssWeight / self.L_ms * x_rb[k]**2 \
              - self.rotor_force_z * x_rb[k]
            Mz_ms[k] = -self.rotor_bending_moment_z \
                - self.rotor_force_y * x_rb[k]

        for j in range(self.len_pts):
            My_ms[j + self.len_pts] = -self.rotor_force_z * x_ms[j] \
              - self.rotor_bending_moment_y \
              + self.rotorWeight * cosSA * x_ms[j] \
              - self.F_mb_z * (x_ms[j] - self.distance_hub2mb) \
              + 0.5 * x_ms[j]**2 * self.lssWeight / self.L_ms
            Mz_ms[j + self.len_pts] = -self.rotor_bending_moment_z \
                - self.F_mb_y * (x_ms[j] - self.distance_hub2mb) \
                - self.rotor_force_y * x_ms[j]

        x_shaft = np.concatenate([x_rb, x_ms]) # not used

        if useComputeD:
            # Design shaft OD using distortion energy theory
            MM_max = np.amax((My_ms**2 + Mz_ms**2)**0.5)  # MM_max, min in N-m
            MM_min = ((My_ms[-1]**2 + Mz_ms[-1]**2)**0.5)
            self.D_max = computeD(MM_max, self.rotor_bending_moment_x, self.Sy, self.n_safety)
            self.D_min = computeD(MM_min, self.rotor_bending_moment_x, self.Sy, self.n_safety)

        else:
            MM_max = np.amax((My_ms**2 + Mz_ms**2)**0.5 / 1000.0)  # MM_max, min in kN-m
            Index = np.argmax((My_ms**2 + Mz_ms**2)**0.5 / 1000.0) # not used
            MM_min = ((My_ms[-1]**2 + Mz_ms[-1]**2)**0.5 / 1000.0)
    
            # Design shaft OD using distortion energy theory
            MM = MM_max
            self.D_max = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb)**2 + 3.0 * (
                self.rotor_bending_moment_x / 1000.0 * self.u_knm_inlb)**2)**0.5)**(1.0 / 3.0) * self.u_in_m
            sys.stderr.write('D_max\n  IN: MM {:.1f} rbmx {:.1f} Sy {:.1f} Nsafe {:.1f}\n'.format(MM, 
                             self.rotor_bending_moment_x, self.Sy, self.n_safety))
            sys.stderr.write('  d1 {:.6f} d2 {:.1f} d3 {:.3f}\n'.format(16.0 * self.n_safety / pi / self.Sy,
                             (4.0 * (MM * self.u_knm_inlb)**2 + 3.0 * (self.rotor_bending_moment_x / 1000.0 * self.u_knm_inlb)**2)**0.5,
                             self.D_max))

            # OD at end
            MM = MM_min
            self.D_min = (16.0 * self.n_safety / pi / self.Sy * (4.0 * (MM * self.u_knm_inlb)**2 + 3.0 * (
                self.rotor_bending_moment_x / 1000.0 * self.u_knm_inlb)**2)**0.5)**(1.0 / 3.0) * self.u_in_m
        
        
        #if self.debug:
        #    sys.stderr.write('size3pt  : MM_max {:.1f} MM_min {:.1f} D_max {:.3f} D_min {:.3f}\n'.format(MM_max, MM_min, self.D_max, self.D_min))

        # Estimate ID
        self.D_in = self.shaft_ratio * self.D_max
        self.D_max = (self.D_in**4 + self.D_max**4)**0.25
        self.D_min = (self.D_in**4 + self.D_min**4)**0.25

        self.lssWeight_new = (pi / 12.0 * self.L_ms * (self.D_max**2.0 + self.D_min**2.0 + self.D_max * self.D_min) 
                            - pi / 4.0  * self.L_ms * self.D_in**2.0
                            + pi / 4.0  * self.distance_hub2mb * self.D_max**2) * self.density * self.g
           # volume of tapered cylinder - internal hole + cylindrical section (hub to MB)
        massLSS_new = self.lssWeight_new / self.g # not used

        D1 = self.fx(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.distance_hub2mb + self.L_ms)
        D2 = self.fx(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, self.distance_hub2mb)
        C1 = -(D1 - D2) / self.L_ms
        C2 = -D2 - C1 * (self.distance_hub2mb)

        I_2 = pi / 64.0 * (self.D_max**4 - self.D_in**4) # hollow shaft inertia (Eq. 2.46  (Eq. 9.5 in 2015 rpt))

        self.theta_y = np.zeros(self.len_pts)
        d_y = np.zeros(self.len_pts)

        for kk in range(self.len_pts):
            self.theta_y[kk] = self.gx(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                                  self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, C1, x_ms[kk]) / self.E / I_2
            # d_y[] computed but discarded
            d_y[kk] = (self.fx(self.rotor_force_z, self.rotorWeight, self.shaft_angle, self.rotor_bending_moment_y,
                          self.F_mb_z, self.distance_hub2mb, self.lssWeight_new, self.L_ms, x_ms[kk]) + C1 * x_ms[kk] + C2) / self.E / I_2

    #----------------------------
    
    def compute(self, rotor_diameter, rotor_mass, rotor_thrust, rotor_force_y, rotor_force_z, 
                      rotor_bending_moment_x, rotor_bending_moment_y, rotor_bending_moment_z, \
                      overhang, machine_rating, drivetrain_efficiency, \
                      gearbox_mass, carrier_mass, gearbox_cm, gearbox_length, \
                      shrink_disc_mass, flange_length, distance_hub2mb, shaft_angle, shaft_ratio,
                      hub_flange_thickness):

        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.rotor_mass = rotor_mass #Float(iotype='in', units='kg', desc='rotor mass')
        self.rotor_bending_moment_x = rotor_bending_moment_x #Float(iotype='in', units='N*m', desc='The bending moment about the x axis')
        self.rotor_bending_moment_y = rotor_bending_moment_y #Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
        self.rotor_bending_moment_z = rotor_bending_moment_z #Float(iotype='in', units='N*m', desc='The bending moment about the z axis')
        self.rotor_thrust = rotor_thrust #Float(iotype='in', units='N', desc='The force along the x axis applied at hub center')
        self.rotor_force_y = rotor_force_y #Float(iotype='in', units='N', desc='The force along the y axis applied at hub center')
        self.rotor_force_z = rotor_force_z #Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
        self.overhang = overhang #Float(iotype='in', units='m', desc='Overhang distance')
        self.machine_rating = machine_rating #Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
        self.drivetrain_efficiency = drivetrain_efficiency #Float(iotype = 'in', desc = 'overall drivettrain efficiency')
        self.gearbox_mass = gearbox_mass #Float(iotype='in', units='kg', desc='Gearbox mass')
        self.carrier_mass = carrier_mass #Float(iotype='in', units='kg', desc='Carrier mass')
        self.gearbox_cm = gearbox_cm #Array(iotype = 'in', units = 'm', desc = 'center of mass of gearbox')
        self.gearbox_length = gearbox_length #Float(iotype='in', units='m', desc='gearbox length')
        self.shrink_disc_mass = shrink_disc_mass #Float(iotype='in', units='kg', desc='Mass of the shrink disc')# shrink disk or flange addtional mass
        self.flange_length = flange_length #Float(iotype ='in', units='m', desc ='flange length')
        self.distance_hub2mb = distance_hub2mb #Float(iotype='in', units='m', desc='distance between hub center and upwind main bearing')
        self.shaft_angle = shaft_angle #Float(iotype='in', units='rad', desc='Angle of the LSS inclindation with respect to the horizontal')
        self.shaft_ratio = shaft_ratio #Float(iotype='in', desc='Ratio of inner diameter to outer diameter.  Leave zero for solid LSS')
        self.hub_flange_thickness = hub_flange_thickness 
        
        # outputs
        self.design_torque = 0.0 #Float(iotype='out', units='N*m', desc='lss design torque')
        self.design_bending_load = 0.0 #Float(iotype='out', units='N', desc='lss design bending load')
        self.length = 0.0 #Float(iotype='out', units='m', desc='lss length')
        self.diameter1 = 0.0 #Float(iotype='out', units='m', desc='lss outer diameter at main bearing')
        self.diameter2 = 0.0 #Float(iotype='out', units='m', desc='lss outer diameter at second bearing')
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.mb1_facewidth = 0.0 #Float(iotype='out', units='m', desc='facewidth of upwind main bearing') 
        self.mb2_facewidth = 0.0 #Float(iotype='out', units='m', desc='facewidth of main bearing')     
        self.mb1_mass = 0.0 #Float(iotype='out', units = 'kg', desc='main bearing mass')
        self.mb2_mass = 0.0 #Float(iotype='out', units = 'kg', desc='second bearing mass')
        self.mb1_cm = np.zeros(3) #Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 1 center of mass')
        self.mb2_cm = np.zeros(3) #Array(np.array([0,0,0]),iotype='out', units = 'm', desc = 'main bearing 2 center of mass')

        # input parameters
        if self.distance_hub2mb == 0:  # distance from hub center to main bearing
            distance_hub2mb = get_distance_hub2mb(self.rotor_diameter, False) # [0] not needed without derivative
        else:
            distance_hub2mb = self.distance_hub2mb

        # If user does not know important moments, crude approx
        if self.rotor_mass > 0 and self.rotor_bending_moment_y == 0:
            self.rotor_bending_moment_y = get_My(self.rotor_mass, distance_hub2mb)

        if self.rotor_mass > 0 and self.rotor_bending_moment_z == 0:
            self.rotor_bending_moment_z = get_Mz(self.rotor_mass, distance_hub2mb)

        if self.flange_length == 0:
            ''' 2014 report gives flange_length as 0.9918 * exp(0.0068*rotor_diameter) (after Eq.2.38)  
                which gives lengths roughly 3 times as larges as the following code '''
            self.flange_length = 0.3 * (self.rotor_diameter / 100.0)**2.0 \
                - 0.1 * (self.rotor_diameter / 100.0) \
                + 0.4 # (following Eq. 2.32 in 2015 rpt) 
            if self.debug:
                sys.stderr.write('MSFlangeLen (approx): {:.2f} m\n'.format(self.flange_length))
                
        # constants
        self.g = 9.81 # m/s^2

        # material properties
        self.E = 2.1e11
        self.density = 7850.0
        self.n_safety = 2.5
        self.n_safety_brg = 1.0
        self.Sy = 66000  # *self.S_ut/700e6 #psi
        
        # unit conversion
        self.u_knm_inlb = 8850.745454036
        self.u_in_m = 0.0254000508001
        
        self.L_ms_new = 0.0
        self.L_ms_0 = 0.5  # main shaft length downwind of main bearing
        self.L_ms = self.L_ms_0
        tol = 1e-4
        check_limit = 1.0
        dL = 0.05
        self.len_pts = 101
        self.D_max = 1.0
        self.D_min = 0.2

        T = self.rotor_bending_moment_x / 1000.0 # rbmx in kN-m NOT USED

        # Main bearing deflection check
        Bearing_Limit = bearing_defl_check(self.mb1Type)
        
        N_count = 50 # not used

        counter = 0
        length_max = self.overhang - distance_hub2mb + \
            (self.gearbox_cm[0] - self.gearbox_length / 2.)  # modified length limit 7/29

        while abs(check_limit) > tol and self.L_ms_new < length_max:
            counter = counter + 1
            if self.L_ms_new > 0:
                self.L_ms = self.L_ms_new
            else:
                self.L_ms = self.L_ms_0

            #-----------------------
            self.size_LSS_3pt()
            #-----------------------

            check_limit = abs(abs(self.theta_y[-1]) - Bearing_Limit / self.n_safety_brg)
            self.L_ms_new = self.L_ms + dL

        # resize bearing (no fatigue check implemented)
        [self.D_max_a, facewidth_max, bearingmass] = resize_for_bearings(self.D_max,  self.mb1Type, False)

        # mb2 is a representation of the gearbox connection
        # TODO: revisit this formulation
        [self.D_min_a, facewidth_min, trash] = resize_for_bearings(self.D_min,  'SRB', False)

        ''' lss_volume = vol(solid taper) + vol(contained by MB1) + vol(contained by MB2) - vol(hole)  Eq. 2.37 (Eq. 2.31 in 2015 rpt) '''
        lss_volume_new = (pi / 3) * (self.D_max_a**2 + self.D_min_a**2 + self.D_max_a * self.D_min_a) * (self.L_ms - (facewidth_max + facewidth_min) / 2) / 4 \
                     + (pi / 4) * (self.D_max_a**2 - self.D_in**2) * facewidth_max \
                     + (pi / 4) * (self.D_min_a**2 - self.D_in**2) * facewidth_min \
                     - (pi / 4) * (self.D_in**2) * (self.L_ms + (facewidth_max + facewidth_min) / 2) 
        lss_mass_new = lss_volume_new * self.density

        if useFlangeModel:
            self.flange_length, mass_flange, cm_flange, cost_flange = mainshaftFlangeCalc(self.D_in, 
                                                                    self.D_max_a, 
                                                                    self.hub_flange_thickness * FLANGE_THICK_FACTOR, 
                                                                    debug=self.debug)
            lss_mass_new += mass_flange
        else:
            lss_mass_new *= 1.35  # add flange and shrink disk mass NOTE: sdm is added below - this approx probably just for flange

        self.mass = lss_mass_new
        
        self.lss_length = self.L_ms_new \
            + (facewidth_max + facewidth_min) / 2 \
            + self.flange_length # Eq. 2.38 (Eq. 2.32 in 2015 rpt)
            
        self.D_outer = self.D_max
        self.D_in = self.D_in
        self.diameter1 = self.D_max_a
        self.diameter2 = self.D_min_a
        # self.lss_length=self.L_ms
        self.D_outer = self.D_max_a
        self.diameter = self.D_max_a
          # diameter == D_outer == diameter1 == D_max_a

        # calculate mass properties
        downwind_location = np.array([self.gearbox_cm[0] - self.gearbox_length / 2., self.gearbox_cm[1], self.gearbox_cm[2]])

        mb1_cm = np.zeros(3)  # upwind
        mb1_cm[0] = downwind_location[0] - self.L_ms * cos(self.shaft_angle)
        mb1_cm[1] = downwind_location[1]
        mb1_cm[2] = downwind_location[2] + self.L_ms * sin(self.shaft_angle)
        self.mb1_cm = mb1_cm

        self.mb2_cm = np.zeros(3)  # downwind does not exist

        cm = np.zeros(3)
        # From solid models, center of mass with flange (not including shrink
        # disk) very nearly .65*total_length
        cm[0] = downwind_location[0] - 0.65 * self.lss_length * cos(self.shaft_angle)
        cm[1] = downwind_location[1]
        cm[2] = downwind_location[2] + 0.65 * self.lss_length * sin(self.shaft_angle)

        # including shrink disk mass
        self.cm[0] = (cm[0] * self.mass + downwind_location[0] * self.shrink_disc_mass) / (self.mass + self.shrink_disc_mass)
        self.cm[1] = cm[1]
        self.cm[2] = (cm[2] * self.mass + downwind_location[2] * self.shrink_disc_mass) / (self.mass + self.shrink_disc_mass)
        self.mass += self.shrink_disc_mass

        I = np.zeros(3)
        I[0] = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0) / 8.0
        I[1] = self.mass * (self.D_in ** 2.0 + self.D_outer ** 2.0 + (4.0 / 3.0) * (self.lss_length ** 2.0)) / 16.0
        I[2] = I[1]
        self.I = I

        self.facewidth_mb = facewidth_max
        self.mb1_mass = bearingmass
        self.mb2_mass = 0.

        ''' self.length was never set - instead, the code was working on self.lss_length, but returning self.length 
            2019 06 11 GNS
        '''
        self.length = self.lss_length # quick fix
        if self.debug:
            sys.stderr.write('LSS3:: Len {:.2f} m (iter) + {:.2f} m (facewidth) + {:.2f} m (flange)\n'.format(self.L_ms_new, 
                                                0.5*(facewidth_max + facewidth_min), self.flange_length))
            #sys.stderr.write('LowSpeedShaft3pt::compute(): ')
            lssfmt = 'LSS3:: Len {:.2f} m Dia1 {:.2f} m Dia2 {:.2f} m  ID {:.2f} m Mass {:.1f} kg MB1Mass {:.1f} kg  F_mb_y {:.1f} N  F_mb_z {:.1f} N\n'
            sys.stderr.write(lssfmt.format(self.length, self.diameter1, self.diameter2, self.D_in, self.mass, 
                                           self.mb1_mass, self.F_mb_y, self.F_mb_z))

        return (self.design_torque, self.design_bending_load, self.length, self.diameter1, self.diameter2, \
                self.mass, self.cm, self.I, \
                self.mb1_facewidth, self.mb2_facewidth, self.mb1_mass, self.mb2_mass, self.mb1_cm, self.mb2_cm)

#-------------------------------------------------------------------------

# Calculate the rest of the bearing attributes (position and mass moments of inertia)
class MainBearing(object):
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. 
          It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
          
          Mass returned by compute() includes bearing housing and is 3.92 times as large as bearing mass returned by 
          LowSpeedShaft[34]pt (which is one of the inputs to compute()).
    '''

    def __init__(self, bearing_position):

        super(MainBearing, self).__init__()

        self.bearing_position = bearing_position #Str(iotype='in',desc='Main bearing type: main or second')

    def compute(self, bearing_mass, lss_diameter, lss_design_torque, rotor_diameter, location):
        
        self.bearing_mass = bearing_mass #Float(iotype ='in', units = 'kg', desc = 'bearing mass from LSS model')
        self.lss_diameter = lss_diameter #Float(iotype='in', units='m', desc='lss outer diameter at main bearing')
        self.lss_design_torque = lss_design_torque #Float(iotype='in', units='N*m', desc='lss design torque')
        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.location = location #Array(np.array([0.,0.,0.]),iotype = 'in', units = 'm', desc = 'x,y,z location from shaft model')

        # returns
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I  = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        self.mass = self.bearing_mass
        self.mass += self.mass * (8000.0 / 2700.0)  # add housing weight
            # see Section 2.2.4.2 in report which gives a factor of 2.92 - this is 2.963

        # calculate mass properties
        inDiam = self.lss_diameter
        depth = (inDiam * 1.5) # not used

        try:
            self.bearing_position in ['main','second']
        except ValueError:
            print("Invalid variable assignment: bearing position must be 'main' or 'second'.")
        else:
            if self.bearing_position == 'main':
                if self.location[0] != 0.0:
                    cm = self.location
                else:
                    cmMB = np.zeros(3)
                    cmMB = ([- (0.035 * self.rotor_diameter),  0.0, 0.025 * self.rotor_diameter])
                    cm = cmMB
                
                b1I0 = (self.mass * inDiam ** 2) / 4.0
                self.cm = cm
                self.I = np.array([b1I0, b1I0 / 2.0, b1I0 / 2.0])
            else:
                if self.mass > 0 and self.location[0] != 0.0:
                    cm = self.location
                else:
                    cm = np.zeros(3)
                    self.mass = 0.
        
                b2I0 = (self.mass * inDiam ** 2) / 4.0
                self.cm = cm
                self.I = np.array([b2I0, b2I0 / 2.0, b2I0 / 2.0])

        return (self.mass, self.cm, self.I.flatten())

#-------------------------------------------------------------------------

#Size gearbox based on type
class Gearbox(object):
    ''' Gearbox class
          The Gearbox class is used to represent the gearbox component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
          
          This class implements much (all?) of the model described in "A wind turbine gearbox sizing model for minimizing
          its weight" by Y. Guo et al., which is in file SunderLandCostModelGearbox_Report2.pdf
    '''

    def __init__(self, gear_configuration, shaft_factor='normal', debug=False):

        super(Gearbox, self).__init__()

        self.gear_configuration = gear_configuration #Str(iotype='in', desc='string that represents the configuration of the gearbox (stage number and types)')
        self.shaft_factor = shaft_factor #Str(iotype='in', desc = 'normal or short shaft length')
        self.debug = debug

    def compute(self, gear_ratio, planet_numbers, rotor_rpm, rotor_diameter, rotor_torque, gearbox_input_cm):

        #variables
        self.gear_ratio = gear_ratio #Float(iotype='in', desc='overall gearbox speedup ratio')
        self.planet_numbers = planet_numbers #Array(np.array([0.0,0.0,0.0,]), iotype='in', desc='number of planets in each stage')
        self.rotor_rpm = rotor_rpm #Float(iotype='in', desc='rotor rpm at rated power')
        self.rotor_diameter = rotor_diameter #Float(iotype='in', desc='rotor diameter')
        self.rotor_torque = rotor_torque #Float(iotype='in', units='N*m', desc='rotor torque at rated power')
        self.gearbox_input_cm = gearbox_input_cm #Float(0,iotype = 'in', units='m', desc ='gearbox position along x-axis')
    
        # outputs
        self.stage_masses = np.zeros(4) #Array(np.array([0.0, 0.0, 0.0, 0.0]), iotype='out', units='kg', desc='individual gearbox stage masses')
        self.gearbox_mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.gearbox_cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.gearbox_I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    
        self.gearbox_length = 0.0 #Float(iotype='out', units='m', desc='gearbox length')
        self.gearbox_height = 0.0 #Float(iotype='out', units='m', desc='gearbox height')
        self.gearbox_diameter = 0.0 #Float(iotype='out', units='m', desc='gearbox diameter')

        # initialize stage ratios
        self.stageRatio = np.zeros([3, 1])

        # filled in when ebxWeightEst is called
        self.stageTorque = np.zeros([len(self.stageRatio), 1])
        # filled in when ebxWeightEst is called
        self.stageMass = np.zeros([len(self.stageRatio), 1])
        self.stageType = self.stageTypeCalc(self.gear_configuration)
        self.stageRatio = self.stageRatioCalc(self.gear_ratio, self.planet_numbers, self.gear_configuration)

        m = self.gearboxWeightEst(self.gear_configuration, self.gear_ratio, self.planet_numbers, self.shaft_factor, self.rotor_torque)
        self.gearbox_mass = float(m)
        self.stage_masses = self.stageMass
        # calculate mass properties

        self.gearbox_length = (0.012 * self.rotor_diameter)
        self.gearbox_height = (0.015 * self.rotor_diameter)
        self.gearbox_diameter = (0.75 * self.gearbox_height)

        cm0 = self.gearbox_input_cm
        cm1 = 0.0
        # TODO validate or adjust factor. origin is modified to be above
        # bedplate top
        cm2 = 0.4 * self.gearbox_height
        self.gearbox_cm = np.array([cm0, cm1, cm2])

        I0 = self.gearbox_mass * (self.gearbox_diameter ** 2) / 8 \
          + (self.gearbox_mass / 2) * (self.gearbox_height ** 2) / 8
        I1 = self.gearbox_mass * (   0.5 * (self.gearbox_diameter ** 2) 
                                 + (2/3) * (self.gearbox_length ** 2) 
                                 + 0.25 * (self.gearbox_height ** 2)) / 8
        I2 = I1
        self.gearbox_I = np.array([I0, I1, I2])
        
        if self.debug:
            sys.stderr.write('GBOX: Mass {:.1f} kg  Len/Ht/Diam (m) {:.2f} {:.2f} {:.2f}\n'.format(self.gearbox_mass, 
                             self.gearbox_length, self.gearbox_height, self.gearbox_diameter))

        return(self.stage_masses.flatten(), self.gearbox_mass, self.gearbox_cm, self.gearbox_I.flatten(), self.gearbox_length, self.gearbox_height, self.gearbox_diameter)

    def stageTypeCalc(self, config):
        temp = []
        for character in config:
                if character == 'e':
                    temp.append(2)
                if character == 'p':
                    temp.append(1)
        return temp

    def stageMassCalc(self, indStageRatio, indNp, indStageType):
        '''
        Computes the mass of an individual gearbox stage.

        Parameters
        ----------
        indStageRatio : str
          Speedup ratio of the individual stage in question.
        indNp : int
          Number of planets for the individual stage.
        indStageType : int
          Type of gear.  Use '1' for parallel and '2' for epicyclic.
        '''

        # Application factor to include ring/housing/carrier weight
        Kr = 0.4
        Kgamma = 1.1

        if indNp == 3:
            Kgamma = 1.1
        elif indNp == 4:
            Kgamma = 1.1
        elif indNp == 5:
            Kgamma = 1.35

        if indStageType == 1:
            indStageMass = 1.0 + indStageRatio + \
                indStageRatio**2 + (1.0 / indStageRatio)

        elif indStageType == 2:
            sunRatio = 0.5 * indStageRatio - 1.0
            indStageMass = Kgamma * ( (1 / indNp) 
                                    + (1 / (indNp * sunRatio))
                                    + sunRatio 
                                    + sunRatio**2 
                                    + Kr * ((indStageRatio - 1)**2) / indNp 
                                    + Kr * ((indStageRatio - 1)**2) / (indNp * sunRatio))

        if self.debug:
            sys.stderr.write('GBox::stageMassCalc(): ISR {:.3f} INP {} IST {}  Mass {:2f}\n'.format(indStageRatio, 
                             indNp, indStageType, indStageMass))
            
        return indStageMass

    def gearboxWeightEst(self, config, overallRatio, planet_numbers, shaft_factor, torque):
        '''
        Computes the gearbox weight based on a surface durability criteria.
        2019 06 10: now uses torque argument instead of self.rotor_torque for consistency (no difference in results)
          What are expected units of torque? Values of 200 to 700 seem VERY low.
        '''

        ## Define Application Factors ##
        # Application factor for weight estimate
        Ka = 0.6
        Kshaft = 0.0
        Kfact = 0.0

        # K factor for pitting analysis
        #if self.rotor_torque < 200.0:
        if torque < 200.0:
            Kfact = 850.0
        #elif self.rotor_torque < 700.0:
        elif torque < 700.0:
            Kfact = 950.0
        else:
            Kfact = 1100.0

        # Unit conversion from Nm to inlb and vice-versa
        Kunit = 8.029
        ''' Should be 8.85075 to convert from N-m to in-lb '''

        # Shaft length factor
        try:
            shaft_factor in ['normal','short']
        except ValueError:
            print("Invalid shaft_factor.  Must be either 'normal' or 'short'")
        else:
            if shaft_factor == 'normal':
                Kshaft = 1.0
            elif shaft_factor == 'short':
                Kshaft = 1.25

        # Individual stage torques
        #torqueTemp = self.rotor_torque
        torqueTemp = torque
        for s in range(len(self.stageRatio)):
            self.stageTorque[s] = torqueTemp / self.stageRatio[s]
            torqueTemp = self.stageTorque[s]
            self.stageMass[s] = Kunit * Ka / Kfact * self.stageTorque[s] \
                * self.stageMassCalc(self.stageRatio[s], self.planet_numbers[s], self.stageType[s])
            if self.debug:
                sys.stderr.write('GBOX::gbWE(): stage {} mass {:8.1f} kg  torque {:9.1f} N-m\n'.format(s, 
                                 self.stageMass[s][0], self.stageTorque[s][0]))

        gearboxWeight = (sum(self.stageMass)) * Kshaft

        return gearboxWeight

    def stageRatioCalc(self, overallRatio, planet_numbers, config):
        '''
        Calculates individual stage ratios using either:
            empirical relationships from the Sunderland model, or 
            a SciPy constrained optimization routine.
        '''

        K_r = 0

        x = np.zeros([3, 1])

        try:
            config in ['eep','eep_2','eep_3','epp']
        except ValueError:
            print("Invalid value for gearbox_configuration.  Must be one of: 'eep','eep_2','eep_3','epp'")
        else:
            if config == 'eep':
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r1 = 0
                K_r2 = 0  # 2nd stage structure weight coefficient
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) +
                    (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) + \
                    (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 +
                     K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
    
            elif config == 'eep_3':
                # fixes last stage ratio at 3
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r1 = 0
                K_r2 = 0.8  # 2nd stage structure weight coefficient
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 + K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                def constr3(x, overallRatio):
                    return x[2] - 3.0
    
                def constr4(x, overallRatio):
                    return 3.0 - x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2, constr3, constr4], consargs=[overallRatio], rhoend=1e-7)
    
            elif config == 'eep_2':
                # fixes final stage ratio at 2
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r1 = 0
                K_r2 = 1.6  # 2nd stage structure weight coefficient
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 + K_r1 * ((x[0] - 1.0)**2) / B_1 + K_r1 * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1])) * ((1.0 / B_2) + (1 / (B_2 * ((x[1] / 2.0) - 1.0))) + (x[1] / 2.0 - 1.0) + (x[1] / 2.0 - 1.0)**2.0 + K_r2 * ((x[1] - 1.0)**2.0) / B_2 + K_r2 * ((x[1] - 1.0)**2.0) / (B_2 * (x[1] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
            elif config == 'epp':
                # fixes last stage ratio at 3
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                B_2 = planet_numbers[1]
                K_r = 0
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1.0) + (x[0] / 2.0 - 1)**2 +
                        K_r * ((x[0] - 1.0)**2) / B_1 + K_r * ((x[0] - 1.0)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                    + (1.0 / (x[0] * x[1])) * (1.0 + (1.0 / x[1]) + x[1] + x[1]**2) \
                    + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                    return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
    
            else:  # Should not execute since try/except checks for acceptable gearbox configuration types
                x0 = [overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0), 
                      overallRatio ** (1.0 / 3.0)]
                B_1 = planet_numbers[0]
                K_r = 0.0
    
                def volume(x):
                    return (1.0 / (x[0])) * ((1.0 / B_1) + (1.0 / (B_1 * ((x[0] / 2.0) - 1.0))) + (x[0] / 2.0 - 1) + (x[0] / 2.0 - 1.0)**2 + K_r * ((x[0] - 1.0)**2) / B_1 + K_r * ((x[0] - 1)**2) / (B_1 * (x[0] / 2.0 - 1.0))) \
                         + (1.0 / (x[0] * x[1])) * (1.0 + (1.0 / x[1]) + x[1] + x[1]**2) \
                         + (1.0 / (x[0] * x[1] * x[2])) * (1.0 + (1.0 / x[2]) + x[2] + x[2]**2)
    
                def constr1(x, overallRatio):
                   return x[0] * x[1] * x[2] - overallRatio
    
                def constr2(x, overallRatio):
                    return overallRatio - x[0] * x[1] * x[2]
    
                x = opt.fmin_cobyla(volume, x0, [constr1, constr2], consargs=[overallRatio], rhoend=1e-7)
    
            return x

#-------------------------------------------------------------------------


class Bedplate(object):
    ''' Bedplate class
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and 
            dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, uptower_transformer=True, debug=False):

        super(Bedplate, self).__init__()

        self.uptower_transformer = uptower_transformer #Bool(iotype = 'in', desc = 'Boolean stating if transformer is uptower')

        self.debug = debug
        
    # functions used in bedplate sizing
    def midDeflection(self, totalLength, loadLength, load, E, I):
        ''' Eq. 2.154 - tip deflection for load applied at x (Eq. 2.66 in 2015 rpt) '''
        defl = load * loadLength**2.0 * \
            (3.0 * totalLength - loadLength) / (6.0 * E * I)
        return defl
    
    def distDeflection(self, totalLength, distWeight, E, I):
        ''' Eq. 2.155 - tip deflection for distributed load (Eq. 2.67 in 2015 rpt)'''
        defl = distWeight * totalLength**4 / (8.0 * E * I)
        return defl
        
    def characterize_Bedplate_Rear(self):
        '''
        Evaluate stresses and deflections on the rear (steel) section of bedplate
        
        Many of the 'self.*' member variables could be local:
            - all the *TipDefl vbls
            - rootStress, totalBendingMoment, bi, hi, I_b, A, w
        This function just needs to set: totalSteelMass rearTotalTipDefl rearBendingStress
          (similarly for characterize_Bedplate_Front)
        '''
        self.bi = (self.b0 - self.tw) / 2.0
        self.hi = self.h0 - 2.0 * self.tf
        self.I_b = self.b0 * self.h0**3 / 12.0 - 2 * self.bi * self.hi**3 / 12.0
        self.A = self.b0 * self.h0 - 2.0 * self.bi * self.hi
        self.w = self.A * self.steelDensity # mass per linear foot
        
        # Tip Deflection for load not at end

        self.hssTipDefl = self.midDeflection(
            self.rearTotalLength, self.hss_location,       self.hss_mass * self.g / 2,         self.steelE, self.I_b)
        self.genTipDefl = self.midDeflection(
            self.rearTotalLength, self.generator_location, self.generator_mass * self.g / 2,   self.steelE, self.I_b)
        self.convTipDefl = self.midDeflection(
            self.rearTotalLength, self.convLoc,            self.convMass * self.g / 2,         self.steelE, self.I_b)
        self.transTipDefl = self.midDeflection(
            self.rearTotalLength, self.transLoc,           self.transformer_mass * self.g / 2, self.steelE, self.I_b)
        self.gearboxTipDefl = self.midDeflection(
            self.rearTotalLength, self.gearbox_location,   self.gearbox_mass * self.g / 2,     self.steelE, self.I_b)
        self.selfTipDefl = self.distDeflection(
            self.rearTotalLength,                          self.w * self.g,                    self.steelE, self.I_b)
  
        self.totalTipDefl = self.hssTipDefl + self.genTipDefl + self.convTipDefl \
            + self.transTipDefl + self.gearboxTipDefl + self.selfTipDefl
  
        # root stress
        self.totalBendingMoment = (self.hss_location * self.hss_mass 
                                 + self.generator_location * self.generator_mass 
                                 + self.convLoc * self.convMass 
                                 + self.transLoc * self.transformer_mass 
                                 + self.w * self.rearTotalLength**2 / 2.0) * self.g
        self.modelStress = self.totalBendingMoment * self.h0 / (2. * self.I_b)
  
        # mass
        self.steelVolume = self.A * self.rearTotalLength
        self.steelMass = self.steelVolume * self.steelDensity
  
        # 2 parallel I-beams
        self.totalSteelMass = 2.0 * self.steelMass
  
        self.rearTotalTipDefl = self.totalTipDefl
        self.rearBendingStress = self.modelStress

    def characterize_Bedplate_Front(self):
        '''
        Evaluate stresses and deflections on the front (cast) section of bedplate
        '''
        self.bi = (self.b0 - self.tw) / 2.0
        self.hi = self.h0 - 2.0 * self.tf
        self.I_b = self.b0 * self.h0**3 / 12.0 - 2 * self.bi * self.hi**3 / 12.0
        self.A = self.b0 * self.h0 - 2.0 * self.bi * self.hi
        self.w = self.A * self.castDensity
  
        # Tip Deflection for load not at end
        self.gearboxTipDefl = self.midDeflection(
            self.frontTotalLength, self.gearbox_location, self.gearbox_mass * self.g / 2.0, self.castE, self.I_b)
            #self.frontTotalLength, self.gearbox_mass, self.gearbox_mass * self.g / 2.0, self.castE, self.I_b)  Mass should be loc!
        self.mb1TipDefl = self.midDeflection(
            self.frontTotalLength, self.mb1_cm[0],    self.mb1_mass * self.g / 2.0,     self.castE, self.I_b)
        self.mb2TipDefl = self.midDeflection(
            self.frontTotalLength, self.mb2_cm[0],    self.mb2_mass * self.g / 2.0,     self.castE, self.I_b)
        self.lssTipDefl = self.midDeflection(
            self.frontTotalLength, self.lss_location, self.lss_mass * self.g / 2.0,     self.castE, self.I_b)
        self.rotorTipDefl = self.midDeflection(
            self.frontTotalLength, self.rotorLoc,     self.rotor_mass * self.g / 2.0,   self.castE, self.I_b)
        self.rotorFzTipDefl = self.midDeflection(
            self.frontTotalLength, self.rotorLoc,     self.rotorFz / 2.0,               self.castE, self.I_b)
        self.selfTipDefl = self.distDeflection(
            self.frontTotalLength,                    self.w * self.g,                  self.castE, self.I_b)

        self.rotorMyTipDefl = self.rotorMy / 2.0 \
            * self.frontTotalLength**2 / (2.0 * self.castE * self.I_b)
  
        self.totalTipDefl = self.gearboxTipDefl + self.mb1TipDefl + self.mb2TipDefl + self.lssTipDefl  \
            + self.rotorTipDefl + self.rotorFzTipDefl + self.selfTipDefl + self.rotorMyTipDefl
  
        # root stress
        self.totalBendingMoment = (  self.mb1_cm[0] * self.mb1_mass / 2.0 
                                   + self.mb2_cm[0] * self.mb2_mass / 2.0 
                                   + self.lss_location * self.lss_mass / 2.0 
                                   + self.w * self.frontTotalLength**2 / 2.0 
                                   + self.rotorLoc * self.rotor_mass / 2.0) * self.g \
            + self.rotorLoc * self.rotorFz / 2.0 \
            + self.rotorMy / 2.0
        self.modelStress = self.totalBendingMoment * self.h0 / 2 / self.I_b
  
        # mass
        self.castVolume = self.A * self.frontTotalLength
        self.castMass = self.castVolume * self.castDensity
  
        # 2 parallel I-beams
        self.totalCastMass = 2.0 * self.castMass
        self.frontTotalTipDefl = self.totalTipDefl
        self.frontBendingStress = self.modelStress

    def compute(self, gearbox_length, gearbox_location, gearbox_mass, hss_location, hss_mass, generator_location, generator_mass, \
                      lss_location, lss_mass, lss_length, mb1_cm, mb1_facewidth, mb1_mass, mb2_cm, mb2_mass, \
                      transformer_mass, transformer_cm, \
                      tower_top_diameter, rotor_diameter, machine_rating, rotor_mass, rotor_bending_moment_y, rotor_force_z, \
                      flange_length, distance_hub2mb):

        '''Model bedplate as 2 parallel I-beams with a rear steel frame and a front cast frame
           Deflection constraints applied at each bedplate end
           Stress constraint checked at root of front and rear bedplate sections'''

        if self.debug:
            sys.stderr.write('GBox loc {} mass {}\n'.format(gearbox_location, gearbox_mass))
        
        #variables
        self.gearbox_length = gearbox_length #Float(iotype = 'in', units = 'm', desc = 'gearbox length')
        self.gearbox_location = gearbox_location #Float(iotype = 'in', units = 'm', desc = 'gearbox CM location')
        self.gearbox_mass = gearbox_mass #Float(iotype = 'in', units = 'kg', desc = 'gearbox mass')
        self.hss_location = hss_location #Float(iotype ='in', units = 'm', desc='HSS CM location')
        self.hss_mass = hss_mass #Float(iotype ='in', units = 'kg', desc='HSS mass')
        self.generator_location = generator_location #Float(iotype ='in', units = 'm', desc='generator CM location')
        self.generator_mass = generator_mass #Float(iotype ='in', units = 'kg', desc='generator mass')
        self.lss_location = lss_location #Float(iotype ='in', units = 'm', desc='LSS CM location')
        self.lss_mass = lss_mass #Float(iotype ='in', units = 'kg', desc='LSS mass')
        self.lss_length = lss_length #Float(iotype = 'in', units = 'm', desc = 'LSS length')
        self.mb1_facewidth = mb1_facewidth #Float(iotype = 'in', units = 'm', desc = 'Upwind main bearing facewidth')
        self.mb1_cm = mb1_cm #Float(iotype ='in', units = 'm', desc='Upwind main bearing CM location')
        self.mb1_mass = mb1_mass #Float(iotype ='in', units = 'kg', desc='Upwind main bearing mass')
        self.mb2_cm = mb2_cm #Float(iotype ='in', units = 'm', desc='Downwind main bearing CM location')
        self.mb2_mass = mb2_mass #Float(iotype ='in', units = 'kg', desc='Downwind main bearing mass')
        self.transformer_mass = transformer_mass #Float(iotype ='in', units = 'kg', desc='Transformer mass')
        self.transformer_location = transformer_cm[0] #Float(iotype = 'in', units = 'm', desc = 'transformer CM location')
        self.tower_top_diameter = tower_top_diameter #Float(iotype ='in', units = 'm', desc='diameter of the top tower section at the yaw gear')
        self.rotor_diameter = rotor_diameter #Float(iotype = 'in', units = 'm', desc='rotor diameter')
        self.machine_rating = machine_rating #Float(iotype='in', units='kW', desc='machine_rating machine rating of the turbine')
        self.rotor_mass = rotor_mass #Float(iotype='in', units='kg', desc='rotor mass')
        self.rotor_bending_moment_y = rotor_bending_moment_y #Float(iotype='in', units='N*m', desc='The bending moment about the y axis')
        self.rotor_force_z = rotor_force_z #Float(iotype='in', units='N', desc='The force along the z axis applied at hub center')
        self.flange_length = flange_length #Float(iotype='in', units='m', desc='flange length')
        self.distance_hub2mb = distance_hub2mb #Float(iotype = 'in', units = 'm', desc = 'length between rotor center and upwind main bearing')
    
        #outputs
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    
        self.length = 0.0 #Float(iotype='out', units='m', desc='length of bedplate')
        self.height = 0.0 #Float(iotype='out', units='m', desc='max height of bedplate')
        self.width = 0.0 #Float(iotype='out', units='m', desc='width of bedplate')

        #Standard constants and material properties
        self.g = 9.81
        self.E = 2.1e11
        self.density = 7800
        
        self.steelDensity = 7800 # kg/m^3
        self.castDensity  = 7100 # kg/m^3
        self.steelE = 210e9  # Young's modulus of steel     in N/m^2
        self.castE  = 169e9  # Young's modulus of cast iron in N/m^2
        self.steelStressMax = 620e6 # yield strength of alloy steel in MPa (1e6N/m^2)
        self.castStressMax  = 200e6 # yield strength of cast iron   in MPa (1e6N/m^2)

        if self.distance_hub2mb > 0:
            distance_hub2mb = self.distance_hub2mb
        else:
            distance_hub2mb = get_distance_hub2mb(self.rotor_diameter, False)

        # component weights and locations
        if self.transformer_mass > 0:  # only if uptower transformer
            self.transLoc = self.transformer_location
            self.convMass = 0.3 * self.transformer_mass
        else:
            self.transLoc = 0
            # (transformer mass * .3)
            self.convMass = (2.4445 * (self.machine_rating) + 1599.0) * 0.3

        self.convLoc = self.generator_location * 2.0

        mb1_cm = abs(self.mb1_cm[0])
        mb2_cm = abs(self.mb2_cm[0])  
        lss_location = abs(self.lss_location)

        if self.transLoc > 0:
          self.rearTotalLength = self.transLoc * 1.1
        else:
          self.rearTotalLength = self.generator_location * 4.237 / 2.886 \
               - self.tower_top_diameter / 2.0  # scaled off of GE1.5

        self.frontTotalLength = mb1_cm + self.mb1_facewidth / 2.

        # rotor weights and loads
        self.rotorLoc = mb1_cm + distance_hub2mb
        self.rotorFz = abs(self.rotor_force_z)
        self.rotorMy = abs(self.rotor_bending_moment_y)

        # If user does not know important moment, crude approx
        if self.rotor_mass > 0 and self.rotorMy == 0:
            self.rotorMy = get_My(self.rotor_mass, distance_hub2mb)

        if self.rotorFz == 0 and self.rotor_mass > 0:
            self.rotorFz = self.rotor_mass * self.g

        self.defl_denom = 1500.  # factor in deflection check
        self.stressTol = 5e5
        self.deflTol = 1e-4
        self.stress_mult = 8.  # modified to fit industry data
        
        # ----------- REAR -------------------
        
        # initial I-beam dimensions in m
        self.tf = 0.01905        # flange thickness
        self.tw = 0.0127         # web thickness
        self.h0 = 0.6096         # overall height
        self.b0 = self.h0 / 2.0  # overall width

        # Rear Steel Frame:
        if self.gearbox_location == 0:
            self.gearbox_location = 0
            self.gearbox_mass = 0
        else:
            self.gearbox_location = self.gearbox_location
            self.gearbox_mass = self.gearbox_mass
        if self.debug:
            sys.stderr.write('GBox REAR  loc {} mass {}\n'.format(self.gearbox_location, self.gearbox_mass))

        self.modelStress = 250e6  # initial value
        self.totalTipDefl = 1.0  # initial value

        self.stressMax = 620e6  # yield of alloy steel
        self.deflMax = self.rearTotalLength / self.defl_denom

        counter = 0
        while (self.modelStress * self.stress_mult - self.steelStressMax) > self.stressTol \
           or (self.totalTipDefl - self.deflMax) > self.deflTol:

            counter += 1

            self.characterize_Bedplate_Rear()

            self.tf += 0.002
            self.tw += 0.002
            self.b0 += 0.006
            self.h0 += 0.006
            rearCounter = counter

        self.rearHeight = self.h0
        
        # ----------- FRONT -------------------
        
        # Front cast section:
        ''' Negative gearbox_location means that the gearbox is located on the front (cast) section ??? '''
        if self.gearbox_location < 0:
            self.gearbox_location = abs(self.gearbox_location)
            self.gearbox_mass = self.gearbox_mass
        else: 
            self.gearbox_location = 0
            self.gearbox_mass = 0
        if self.debug:
            sys.stderr.write('GBox FRONT loc {} mass {}\n'.format(self.gearbox_location, self.gearbox_mass))

        self.E = 169e9 #EN-GJS-400-18-LT http://www.claasguss.de/html_e/pdf/THBl2_engl.pdf
        self.castDensity = 7100
        
        # initial I-beam dimensions in m
        self.tf = 0.01905        # flange thickness
        self.tw = 0.0127         # web thickness
        self.h0 = 0.6096         # overall height
        self.b0 = self.h0 / 2.0  # overall width
        
        self.modelStress = 250e6  # initial value
        self.totalTipDefl = 1.0  # initial value
        
        self.deflMax = self.frontTotalLength/self.defl_denom
        self.stressMax = 200e6
        
        counter = 0
        
        while (self.modelStress*self.stress_mult - self.castStressMax) >  self.stressTol \
           or (self.totalTipDefl - self.deflMax) >  self.deflTol:
            counter += 1
            self.characterize_Bedplate_Front()
            self.tf += 0.002 
            self.tw += 0.002
            self.b0 += 0.006
            self.h0 += 0.006
            
            frontCounter=counter
            
            '''
            if self.debug:
                scalc = self.modelStress*self.stress_mult - self.castStressMax
                dcalc = self.totalTipDefl - self.deflMax
                sflag = ' '
                dflag = ' '
                if scalc <= self.stressTol:
                    sflag = '*'
                if dcalc <= self.deflTol:
                    dflag = '*'
                sys.stderr.write('BP:front: {:3d} ST {:.1f} calc {:.1f} {} DT {:.5f} calc {:.5f} {}\n'.format(counter,
                        self.stressTol, scalc, sflag,
                        self.deflTol, dcalc, dflag))
            '''
        self.frontHeight = self.h0
  
        # ----------- ----- -------------------
        
        # frame multiplier for front support
        self.support_multiplier = 1.1+5e13*self.rotor_diameter**(-8) # based on solidworks estimates for GRC and GE bedplates. extraneous mass percentage decreases for larger machines
        self.totalCastMass *= self.support_multiplier
        self.totalSteelMass *= self.support_multiplier
        self.mass = self.totalCastMass + self.totalSteelMass
  
        self.bedplate_length = self.frontTotalLength + self.rearTotalLength
        self.width = self.b0 + self.tower_top_diameter
        self.height = np.max([self.frontHeight, self.rearHeight])
  
        # calculate mass properties
        cm = np.zeros(3)
        cm[0] = (self.totalSteelMass*self.rearTotalLength/2 - self.totalCastMass*self.frontTotalLength/2)/(self.mass) #previously 0.
        cm[1] = 0.0
        cm[2] = -self.height/2.
        self.cm = cm
  
        self.depth = (self.bedplate_length / 2.0)
  
        I = np.zeros(3)
        I[0]  = self.mass * (self.width ** 2 + self.depth ** 2) / 8
        I[1]  = self.mass * (self.depth ** 2 + self.width ** 2 + (4/3) * self.bedplate_length ** 2) / 16
        I[2]  = I[1]
        self.I = I

        if self.debug:
            sys.stderr.write('Bedplate: mass {:.1f} cast {:.1f} steel {:.1f} L {:.1f} m H {:.1f} m W {:.1f} m\n'.format(self.mass, 
                             self.totalCastMass, self.totalSteelMass, self.bedplate_length, self.height, self.width))
            sys.stderr.write('Bedplate: frontLen {:.1f} m rearLen {:.1f} m nFront {} nRear {} \n'.format(self.frontTotalLength, 
                             self.rearTotalLength, frontCounter, rearCounter))
            sys.stderr.write('  LSS         {:5.2f} m  {:8.1f} kg\n'.format(self.lss_location, self.lss_mass))
            sys.stderr.write('  HSS         {:5.2f} m  {:8.1f} kg\n'.format(self.hss_location, self.hss_mass))
            sys.stderr.write('  Gearbox     {:5.2f} m  {:8.1f} kg\n'.format(gearbox_location, gearbox_mass))
            sys.stderr.write('  Generator   {:5.2f} m  {:8.1f} kg\n'.format(self.generator_location, self.generator_mass))
            sys.stderr.write('  Transformer {:5.2f} m  {:8.1f} kg\n'.format(self.transformer_location, self.transformer_mass))
            
        return (self.mass, self.cm, self.I, self.bedplate_length, self.height, self.width)

#---------------------------------------------------------------------------------------------------------------

class YawSystem(object):
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''


    def __init__(self, yaw_motors_number=0):

        super(YawSystem, self).__init__()

        self.yaw_motors_number = yaw_motors_number #Int(0,iotype='in', desc='number of yaw motors')

    def compute(self, rotor_diameter, rotor_thrust, tower_top_diameter, above_yaw_mass, bedplate_height):

        #variables
        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.rotor_thrust = rotor_thrust #Float(iotype='in', units='N', desc='maximum rotor thrust')
        self.tower_top_diameter = tower_top_diameter #Float(iotype='in', units='m', desc='tower top diameter')
        self.above_yaw_mass = above_yaw_mass #Float(iotype='in', units='kg', desc='above yaw mass')
        self.bedplate_height = bedplate_height #Float(iotype = 'in', units = 'm', desc = 'bedplate height')
    
        #outputs
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        if self.yaw_motors_number == 0 :
          if self.rotor_diameter < 90.0 :
            self.yaw_motors_number = 4
          elif self.rotor_diameter < 120.0 :
            self.yaw_motors_number = 6
          else:
            self.yaw_motors_number = 8
  
        # Assume friction plate surface width is 1/10 the diameter
        # Assume friction plate thickness scales with rotor diameter
        frictionPlateVol=pi*self.tower_top_diameter*(self.tower_top_diameter*0.10)*(self.rotor_diameter/1000.0)
        steelDensity=8000.0
        frictionPlateMass=frictionPlateVol*steelDensity
  
        # Assume same yaw motors as Vestas V80 for now: Bonfiglioli 709T2M
        yawMotorMass=190.0
  
        totalYawMass=frictionPlateMass + (self.yaw_motors_number*yawMotorMass)
        self.mass= totalYawMass
  
        # calculate mass properties
        # yaw system assumed to be collocated to tower top center
        cm = np.zeros(3)
        cm[2] = -self.bedplate_height
        self.cm = cm
  
        # assuming 0 MOI for yaw system (ie mass is nonrotating)
        I = np.zeros(3)
        self.I = I

        return(self.mass, self.cm, self.I)

#-------------------------------------------------------------------------------

class Transformer(object):
    ''' Transformer class
            The transformer class is used to represent the transformer of a wind turbine drivetrain.
            It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
            It contains an update method to determine the mass, mass properties, and dimensions of the component if it is in fact uptower
    '''

    def __init__(self, uptower_transformer=True):

        super(Transformer, self).__init__()

        self.uptower_transformer = uptower_transformer #Bool(iotype='in', desc = 'uptower or downtower transformer')

    def compute(self, machine_rating, tower_top_diameter, rotor_mass, generator_cm, rotor_diameter, RNA_mass, RNA_cm):

        #inputs
        self.machine_rating = machine_rating #Float(iotype='in', units='kW', desc='machine rating of the turbine')
        self.tower_top_diameter = tower_top_diameter #Float(iotype = 'in', units = 'm', desc = 'tower top diameter for comparision of nacelle CM')
        self.rotor_mass = rotor_mass #Float(iotype='in', units='kg', desc='rotor mass')
        self.generator_cm = generator_cm #Array(iotype='in', desc='center of mass of the generator in [x,y,z]')
        self.rotor_diameter = rotor_diameter #Float(iotype='in',units='m', desc='rotor diameter of turbine')
        self.RNA_mass = RNA_mass #Float(iotype = 'in', units='kg', desc='mass of total RNA')
        self.RNA_cm = RNA_cm #Float(iotype='in', units='m', desc='RNA CM along x-axis')
    
        #outputs
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')    

        if self.uptower_transformer:
            # function places transformer where tower top CM is within tower bottom OD to reduce tower moments
            if self.rotor_mass:
                rotor_mass = self.rotor_mass
            else:
                [rotor_mass] = get_rotor_mass(self.machine_rating,False)

            bottom_OD = self.tower_top_diameter*1.7 #approximate average from industry data

            self.mass = 2.4445*(self.machine_rating) + 1599.0
            
            if self.RNA_cm <= -(bottom_OD)/2: #upwind of acceptable. Most likely
                transformer_x = (bottom_OD/2.*(self.RNA_mass+self.mass) - (self.RNA_mass*self.RNA_cm))/(self.mass)
                if transformer_x > self.generator_cm[0]*3:
                    transformer_x = self.generator_cm[0] + (1.6 * 0.015 * self.rotor_diameter) #assuming generator and transformer approximately same length
            else:
                transformer_x = self.generator_cm[0] + (1.8 * 0.015 * self.rotor_diameter) #assuming generator and transformer approximately same length

            cm = np.zeros(3)
            cm[0] = transformer_x
            cm[1] = self.generator_cm[1]
            cm[2] = self.generator_cm[2]/.75*.5 #same height as gearbox CM
            self.cm = cm
            
            width = self.tower_top_diameter+.5
            height = 0.016*self.rotor_diameter #similar to gearbox
            length = .012*self.rotor_diameter #similar to gearbox
            
            def get_I(d1,d2,mass):
                return mass*(d1**2 + d2**2)/12.
            
            I = np.zeros(3)
            I[0] = get_I(height, width,  self.mass)
            I[1] = get_I(length, height, self.mass)
            I[2] = get_I(length, width,  self.mass)
            self.I = I
            
        else:
            self.cm = np.zeros(3)
            self.I = self.cm.copy()
            self.mass = 0.

        return(self.mass, self.cm, self.I)
        
#-------------------------------------------------------------------

class HighSpeedSide(object):
    '''
    HighSpeedSide class
          The HighSpeedSide class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):

        super(HighSpeedSide, self).__init__()

    def compute(self, rotor_diameter, rotor_torque, gear_ratio, lss_diameter, gearbox_length, gearbox_height, gearbox_cm, length_in):

        # variables
        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.rotor_torque = rotor_torque #Float(iotype='in', units='N*m', desc='rotor torque at rated power')
        self.gear_ratio = gear_ratio #Float(iotype='in', desc='overall gearbox ratio')
        self.lss_diameter = lss_diameter #Float(iotype='in', units='m', desc='low speed shaft outer diameter')
        self.gearbox_length = gearbox_length #Float(iotype = 'in', units = 'm', desc='gearbox length')
        self.gearbox_height = gearbox_height #Float(iotype='in', units = 'm', desc = 'gearbox height')
        self.gearbox_cm = gearbox_cm #Array(iotype = 'in', units = 'm', desc = 'gearbox cm [x,y,z]')
        self.length_in = length_in #Float(iotype = 'in', units = 'm', desc = 'high speed shaft length determined by user. Default 0.5m')
    
        # returns
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.length = 0.0 #Float(iotype='out', desc='length of high speed shaft')

        # compute masses, dimensions and cost
        design_torque = self.rotor_torque / self.gear_ratio               # design torque [Nm] based on rotor torque and Gearbox ratio
        massFact = 0.025                                 # mass matching factor default value
        highSpeedShaftMass = (massFact * design_torque)
  
        mechBrakeMass = (0.5 * highSpeedShaftMass)      # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines
  
        self.mass = (mechBrakeMass + highSpeedShaftMass)
  
        diameter = (1.5 * self.lss_diameter)                     # based on WindPACT relationships for full HSS / mechanical brake assembly
        if self.length_in == 0:
            self.hss_length = 0.5+self.rotor_diameter/127.
        else:
            self.hss_length = self.length_in
        hss_length = self.hss_length
  
        matlDensity = 7850. # material density kg/m^3
  
        # calculate mass properties
        cm = np.zeros(3)
        cm[0]   = self.gearbox_cm[0]+self.gearbox_length/2+hss_length/2
        cm[1]   = self.gearbox_cm[1]
        cm[2]   = self.gearbox_cm[2]+self.gearbox_height*0.2
        self.cm = cm
  
        I = np.zeros(3)
        I[0]    = 0.25 * hss_length * 3.14159 * matlDensity * (diameter ** 2) * (self.gear_ratio**2) * (diameter ** 2) / 8.
        I[1]    = self.mass * ((3/4.) * (diameter ** 2) + (hss_length ** 2)) / 12.
        I[2]    = I[1]
        self.I = I

        return(self.mass, self.cm, self.I, self.hss_length)

#----------------------------------------------------------------------------------------------

class Generator(object):
    '''Generator class
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
          
      Mass is a function of machine_rating, or, in the direct-drive case, of torque.
    '''

    def __init__(self, drivetrain_design='geared'):

        super(Generator, self).__init__()

        self.drivetrain_design = drivetrain_design #Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
       
    def compute(self, rotor_diameter, machine_rating, gear_ratio, hss_length, hss_cm, rotor_rpm):

        # variables
        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.machine_rating = machine_rating #Float(iotype='in', units='kW', desc='machine rating of generator')
        self.gear_ratio = gear_ratio #Float(iotype='in', desc='overall gearbox ratio')
        self.hss_length = hss_length #Float( iotype = 'in', units = 'm', desc='length of high speed shaft and brake')
        self.hss_cm = hss_cm #Array(np.array([0.0,0.0,0.0]), iotype = 'in', units = 'm', desc='cm of high speed shaft and brake')
        self.rotor_rpm = rotor_rpm #Float(iotype='in', units='rpm', desc='Speed of rotor at rated power')
    
        # returns
        self.mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.I = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        # coefficients based on generator configuration
        massCoeff = [None, 6.4737, 10.51 ,  5.34  , 37.68  ]
        massExp   = [None, 0.9223, 0.9223,  0.9223, 1      ]
  
        if self.rotor_rpm !=0:
          CalcRPM = self.rotor_rpm
        else:
          CalcRPM    = 80 / (self.rotor_diameter*0.5*pi/30)  # assumes tip speed of 80 m/s
        CalcTorque = (self.machine_rating*1.1) / (CalcRPM * pi/30)
  
        if self.drivetrain_design == 'geared':
            drivetrain_design = 1
        elif self.drivetrain_design == 'single_stage':
            drivetrain_design = 2
        elif self.drivetrain_design == 'multi':
            drivetrain_design = 3
        elif self.drivetrain_design == 'pm_direct':
            drivetrain_design = 4
  
        if (drivetrain_design < 4):
            self.mass = (massCoeff[drivetrain_design] * self.machine_rating ** massExp[drivetrain_design])
        else:  # direct drive
            self.mass = (massCoeff[drivetrain_design] * CalcTorque ** massExp[drivetrain_design])
  
        # calculate mass properties
        length = 1.8 * 0.015 * self.rotor_diameter
        d_length_d_rotor_diameter = 1.8*.015 # not used
  
        depth = 0.015 * self.rotor_diameter
        d_depth_d_rotor_diameter = 0.015 # not used
  
        width = 0.5 * depth
        d_width_d_depth = 0.5 # not used
  
        cm = np.zeros(3)
        cm[0]  = self.hss_cm[0] + self.hss_length/2. + length/2.
        cm[1]  = self.hss_cm[1]
        cm[2]  = self.hss_cm[2]
        self.cm = cm
  
        I = np.zeros(3)
        I[0]   = 4.86e-5 * self.rotor_diameter**5.333 \
                + (2./3. * self.mass) * (depth**2 + width**2) / 8.
        I[1]   = I[0] / 2. / self.gear_ratio**2 \
                 + 1. / 3. * self.mass * length**2 / 12. \
                 + 2. / 3. * self.mass * (depth**2. + width**2. + 4./3. * length**2.) / 16.
        I[2]   = I[1]
        self.I = I 

        return(self.mass, self.cm, self.I)

#-------------------------------------------------------------------------------

class AboveYawMassAdder(object):

    def __init__(self, crane=True):

        super(AboveYawMassAdder, self).__init__()

        self.crane = crane #Bool(iotype='in', desc='flag for presence of crane')
        
    def compute(self, machine_rating, lss_mass, mb1_mass, mb2_mass, gearbox_mass, \
                      hss_mass, generator_mass, bedplate_mass, bedplate_length, bedplate_width, transformer_mass):

        # variables
        self.machine_rating = machine_rating #Float(iotype = 'in', units='kW', desc='machine rating')
        self.lss_mass = lss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb1_mass = mb1_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb2_mass = mb2_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.gearbox_mass = gearbox_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.hss_mass = hss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.generator_mass = generator_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.bedplate_mass = bedplate_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.bedplate_length = bedplate_length #Float(iotype = 'in', units='m', desc='component length')
        self.bedplate_width = bedplate_width #Float(iotype = 'in', units='m', desc='component width')
        self.transformer_mass = transformer_mass #Float(iotype = 'in', units='kg', desc='component mass')
    
        # returns
        self.electrical_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.converter_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.hvac_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.controls_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.platforms_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.crane_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.mainframe_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.cover_mass = 0.0 #Float(iotype = 'out', units='kg', desc='component mass')
        self.above_yaw_mass = 0.0 #Float(iotype = 'out', units='kg', desc='total mass above yaw system')
        self.length = 0.0 #Float(iotype = 'out', units='m', desc='component length')
        self.width = 0.0 #Float(iotype = 'out', units='m', desc='component width')
        self.height = 0.0 #Float(iotype = 'out', units='m', desc='component height')

        # electronic systems, hydraulics and controls
        self.electrical_mass = 0.0
        
        self.converter_mass = 0 #2.4445*self.machine_rating + 1599.0 accounted for in transformer calcs
        
        self.hvac_mass = 0.08 * self.machine_rating
        
        self.controls_mass     = 0.0
        
        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        self.platforms_mass = 0.125 * self.bedplate_mass
        
        if (self.crane):
            self.crane_mass =  3000.0
        else:
            self.crane_mass = 0.0
            
        self.mainframe_mass  = self.bedplate_mass + self.crane_mass + self.platforms_mass
        
        nacelleCovArea      = 2 * (self.bedplate_length ** 2)              # this calculation is based on Sunderland
        self.cover_mass = (84.1 * nacelleCovArea) / 2          # this calculation is based on Sunderland - divided by 2 in order to approach CSM
        
        # yaw system weight calculations based on total system mass above yaw system
        self.above_yaw_mass =  (self.lss_mass + 
                                self.mb1_mass + self.mb2_mass + 
                                self.gearbox_mass + 
                                self.hss_mass + 
                                self.generator_mass + 
                                self.mainframe_mass + 
                                self.transformer_mass +
                                self.electrical_mass + 
                                self.converter_mass + 
                                self.hvac_mass +
                                self.cover_mass)

        self.length      = self.bedplate_length                              # nacelle length [m] based on bedplate length
        self.width       = self.bedplate_width                        # nacelle width [m] based on bedplate width
        self.height      = (2.0 / 3.0) * self.length                         # nacelle height [m] calculated based on cladding area

        return(self.electrical_mass, self.converter_mass, self.hvac_mass, self.controls_mass, self.platforms_mass, self.crane_mass, \
               self.mainframe_mass, self.cover_mass, self.above_yaw_mass, self.length, self.width, self.height)

#--------------------------------------------
class RNASystemAdder(object):
    ''' RNASystem class
          This analysis is only to be used in placing the transformer of the drivetrain.
          The Rotor-Nacelle-Group class is used to represent the RNA of the turbine without the yaw system, transformer and bedplate (to resolve circular dependency issues).
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component. 
    '''

    def __init__(self):

        super(RNASystemAdder , self).__init__()
        
    def compute(self, lss_mass, mb1_mass, mb2_mass, gearbox_mass,  hss_mass, generator_mass, \
                      lss_cm, mb1_cm, mb2_cm, gearbox_cm, hss_cm, generator_cm, overhang, rotor_mass, machine_rating):

        #inputs
        self.lss_mass = lss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb1_mass = mb1_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb2_mass = mb2_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.gearbox_mass = gearbox_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.hss_mass = hss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.generator_mass = generator_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.lss_cm = lss_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.mb1_cm = mb1_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.mb2_cm = mb2_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.gearbox_cm = gearbox_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.hss_cm = hss_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.generator_cm = generator_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.overhang = overhang #Float(iotype = 'in', units='m', desc='nacelle overhang')
        self.rotor_mass = rotor_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.machine_rating = machine_rating #Float(iotype = 'in', units = 'kW', desc = 'machine rating ')
    
        #returns
        self.RNA_mass = 0.0 #Float(iotype = 'out', units='kg', desc='mass of total RNA')
        self.RNA_cm = 0.0 #Float(iotype='out', units='m', desc='RNA CM along x-axis')

        if self.rotor_mass > 0:
            rotor_mass = self.rotor_mass
        else:
            [rotor_mass] = get_rotor_mass(self.machine_rating,False)

        masses = np.array([rotor_mass, self.lss_mass, self.mb1_mass,self.mb2_mass,self.gearbox_mass,self.hss_mass,self.generator_mass])
        cms = np.array([(-self.overhang), self.lss_cm[0], self.mb1_cm[0], self.mb2_cm[0], self.gearbox_cm[0], self.hss_cm[0], self.generator_cm[0]])
        
        self.RNA_mass = np.sum(masses)
        self.RNA_cm = np.sum(masses*cms)/np.sum(masses)
        
        return(self.RNA_mass, self.RNA_cm)

#--------------------------------------------
class NacelleSystemAdder(object): #added to drive to include transformer
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self):

        super(NacelleSystemAdder , self).__init__()
       
    def compute(self, above_yaw_mass, yaw_mass, lss_mass, mb1_mass, mb2_mass, gearbox_mass, \
                      hss_mass, generator_mass, bedplate_mass, mainframe_mass, \
                      lss_cm, mb1_cm, mb2_cm, gearbox_cm, hss_cm, generator_cm, bedplate_cm, \
                      lss_I, mb1_I, mb2_I, gearbox_I, hss_I, generator_I, bedplate_I, \
                      transformer_mass, transformer_cm, transformer_I):

        # variables
        self.above_yaw_mass = above_yaw_mass #Float(iotype='in', units='kg', desc='mass above yaw system')
        self.yaw_mass = yaw_mass #Float(iotype='in', units='kg', desc='mass of yaw system')
        self.lss_mass = lss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb1_mass = mb1_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mb2_mass = mb2_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.gearbox_mass = gearbox_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.hss_mass = hss_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.generator_mass = generator_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.bedplate_mass = bedplate_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.mainframe_mass = mainframe_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.lss_cm = lss_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.mb1_cm = mb1_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.mb2_cm = mb2_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.gearbox_cm = gearbox_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.hss_cm = hss_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.generator_cm = generator_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.bedplate_cm = bedplate_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.lss_I = lss_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.mb1_I = mb1_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.mb2_I = mb2_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.gearbox_I = gearbox_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.hss_I = hss_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.generator_I = generator_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.bedplate_I = bedplate_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
        self.transformer_mass = transformer_mass #Float(iotype = 'in', units='kg', desc='component mass')
        self.transformer_cm = transformer_cm #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component CM')
        self.transformer_I = transformer_I #Array(np.array([0.0,0.0,0.0]),iotype = 'in', units='kg', desc='component I')
    
        # returns
        self.nacelle_mass = 0.0 #Float(0.0, iotype='out', units='kg', desc='overall component mass')
        self.nacelle_cm = np.zeros(3) #Array(np.array([0.0, 0.0, 0.0]), units='m', iotype='out', desc='center of mass of the component in [x,y,z] for an arbitrary coordinate system')
        self.nacelle_I = np.zeros(3) # Array(np.array([0.0, 0.0, 0.0]), units='kg*m**2', iotype='out', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')

        # aggregation of nacelle mass
        self.nacelle_mass = (self.above_yaw_mass + self.yaw_mass)
  
        # calculation of mass center and moments of inertia
        self.nacelle_cm = ( (self.lss_mass*self.lss_cm
                           + self.transformer_mass*self.transformer_cm 
                           + self.mb1_mass*self.mb1_cm 
                           + self.mb2_mass*self.mb2_cm 
                           + self.gearbox_mass*self.gearbox_cm 
                           + self.hss_mass*self.hss_cm 
                           + self.generator_mass*self.generator_cm 
                           + self.mainframe_mass*self.bedplate_cm 
                           + self.yaw_mass*np.zeros(3))
                      / (self.lss_mass + self.mb1_mass + self.mb2_mass + self.gearbox_mass +
                         self.hss_mass + self.generator_mass + self.mainframe_mass + self.yaw_mass) )
  
        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems

        def appendI(xmass, xcm, xI):
            r    = xcm - self.nacelle_cm
            Icg  = assembleI( np.r_[xI, np.zeros(3)] ) # not used
            Iadd = xmass*(np.dot(r, r)*np.eye(3) - np.outer(r, r))
            return Iadd

        I   = np.zeros((3,3))
        I += appendI(self.lss_mass, self.lss_cm, self.lss_I)
        I += appendI(self.hss_mass, self.hss_cm, self.hss_I)
        I += appendI(self.mb1_mass, self.mb1_cm, self.mb1_I)
        I += appendI(self.mb2_mass, self.mb2_cm, self.mb2_I)
        I += appendI(self.gearbox_mass, self.gearbox_cm, self.gearbox_I)
        I += appendI(self.transformer_mass, self.transformer_cm, self.transformer_I)
        I += appendI(self.generator_mass, self.generator_cm, self.generator_I)
        # Mainframe mass includes bedplate mass and other components that assume the bedplate cm
        I += appendI(self.mainframe_mass, self.bedplate_cm, (self.mainframe_mass/self.bedplate_mass)*self.bedplate_I)
        self.nacelle_I = unassembleI(I)

        return(self.nacelle_mass, self.nacelle_cm, self.nacelle_I)

#%%------------------
        
if __name__ == '__main__':

    '''TODO: add full drivetrain examples in pure python'''

    pass
