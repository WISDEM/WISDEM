"""PMSG_arms.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

from openmdao.api import Group, Problem, ExplicitComponent,ExecComp,IndepVarComp,ScipyOptimizeDriver
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan

class PMSG_Arms(ExplicitComponent):
    """ Estimates overall mass dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
        
        # PMSG_arms generator design inputs
        #self.add_input('r_s', val=0.0, units ='m', desc='airgap radius r_s')
        self.add_input('rad_ag', val=0.0, units ='m', desc='airgap radius')
        self.add_input('l_s', val=0.0, units ='m', desc='Stator core length l_s')
        self.add_input('h_s', val=0.0, units ='m', desc='Yoke height h_s')
        self.add_input('tau_p', val=0.0, units ='m', desc='Pole pitch self.tau_p')
        self.add_input('machine_rating', val=0.0, units ='W', desc='Machine rating')
        self.add_input('n_nom', val=0.0, units ='rpm', desc='rated speed')
        self.add_input('Torque', val=0.0, units ='N*m', desc='Rated torque ')
        self.add_input('h_m', val=0.0, units ='m', desc='magnet height')
        self.add_input('h_ys', val=0.0, units ='m', desc='Yoke height')
        self.add_input('h_yr', val=0.0, units ='m', desc='rotor yoke height')
        
        # structural design variables
        self.add_input('n_s', val=0.0, desc='number of stator arms n_s')
        self.add_input('b_st', val=0.0, units ='m', desc='arm width b_st')
        self.add_input('d_s', val=0.0,units ='m', desc='arm depth d_s')
        self.add_input('t_ws', val=0.0,units ='m', desc='arm depth thickness self.t_wr')
        self.add_input('n_r', val=0.0, desc='number of arms n')
        self.add_input('b_r', val=0.0,units ='m', desc='arm width b_r')
        self.add_input('d_r', val=0.0, units ='m', desc='arm depth d_r')
        self.add_input('t_wr', val=0.0, units ='m', desc='arm depth thickness self.t_wr')
        self.add_input('R_o', val=0.0, units ='m',desc='Shaft radius')
        
        # PMSG_arms generator design outputs
        
        # Magnetic loading
        self.add_output('B_symax', val=0.0, desc='Peak Stator Yoke flux density B_ymax')
        self.add_output('B_tmax', val=0.0, desc='Peak Teeth flux density')
        self.add_output('B_rymax', val=0.0, desc='Peak Rotor yoke flux density')
        self.add_output('B_smax', val=0.0, desc='Peak Stator flux density')
        self.add_output('B_pm1', val=0.0, desc='Fundamental component of peak air gap flux density')
        self.add_output('B_g', val=0.0, desc='Peak air gap flux density B_g')
        
        # Stator design
        self.add_output('N_s', val=0.0, desc='Number of turns in the stator winding')
        self.add_output('b_s', val=0.0, desc='slot width')
        self.add_output('b_t', val=0.0, desc='tooth width')
        self.add_output('A_Cuscalc', val=0.0, desc='Conductor cross-section mm^2')
        self.add_output('S'       , val=0.0, desc='Stator slots')
        
        # Rotor magnet dimension
        self.add_output('b_m', val=0.0, desc='magnet width')
        self.add_output('p', val=0.0, desc='No of pole pairs')
        
        # Electrical performance
        self.add_output('E_p', val=0.0, desc='Stator phase voltage')
        self.add_output('f', val=0.0, desc='Generator output frequency')
        self.add_output('I_s', val=0.0, desc='Generator output phase current')
        self.add_output('R_s', val=0.0, desc='Stator resistance')
        self.add_output('L_s', val=0.0, desc='Stator synchronising inductance')
        self.add_output('A_1', val=0.0, desc='Electrical loading')
        self.add_output('J_s', val=0.0, desc='Current density')
        
        # Objective functions
        self.add_output('Mass', val=0.0, desc='Actual mass')
        self.add_output('K_rad', val=0.0, desc='K_rad')
        self.add_output('Losses', val=0.0, desc='Total loss')
        self.add_output('gen_eff', val=0.0, desc='Generator efficiency')
        
        # Structural performance
        self.add_output('u_Ar', val=0.0, desc='Rotor radial deflection')
        self.add_output('y_Ar', val=0.0, desc='Rotor axial deflection')
        self.add_output('z_A_r', val=0.0, desc='Rotor circumferential deflection')      
        self.add_output('u_As', val=0.0, desc='Stator radial deflection')
        self.add_output('y_As', val=0.0, desc='Stator axial deflection')
        self.add_output('z_A_s', val=0.0, desc='Stator circumferential deflection')  
        self.add_output('u_all_r', val=0.0, desc='Allowable radial rotor')
        self.add_output('u_all_s', val=0.0, desc='Allowable radial stator')
        self.add_output('y_all', val=0.0, desc='Allowable axial')
        self.add_output('z_all_s', val=0.0, desc='Allowable circum stator')
        self.add_output('z_all_r', val=0.0, desc='Allowable circum rotor')
        self.add_output('b_all_s', val=0.0, desc='Allowable arm')
        self.add_output('b_all_r', val=0.0, desc='Allowable arm dimensions')
        self.add_output('TC1', val=0.0, desc='Torque constraint')
        self.add_output('TC2', val=0.0, desc='Torque constraint-rotor')
        self.add_output('TC3', val=0.0, desc='Torque constraint-stator')
        
        # Other parameters
        self.add_output('R_out', val=0.0, desc='Outer radius')
        
        self.add_output('Slot_aspect_ratio', val=0.0, desc='Slot aspect ratio')
        
        # Mass Outputs
        self.add_output('mass_PM', val=0.0, units='kg', desc='Magnet mass')
        self.add_output('Copper', val=0.0, units='kg', desc='Copper Mass')
        self.add_output('Iron', val=0.0, units='kg', desc='Electrical Steel Mass')
        self.add_output('Structural_mass', val=0.0, units='kg', desc='Structural Mass')
        
        # Material properties
        self.add_input('rho_Fes', val=0.0,units='kg*m**-3', desc='Structural Steel density ')
        self.add_input('rho_Fe', val=0.0,units='kg*m**-3', desc='Magnetic Steel density ')
        self.add_input('rho_Copper', val=0.0,units='kg*m**-3', desc='Copper density ')
        self.add_input('rho_PM', val=0.0,units='kg*m**-3', desc='Magnet density ')
        
        # inputs/outputs for interface with drivese    
        self.add_input('shaft_cm', val= np.zeros(3), units='m', desc='Main Shaft CM')
        self.add_input('shaft_length', val=0.0, units='m', desc='main shaft length')
        self.add_output('I', val=np.zeros(3),desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('cm', val=np.zeros(3),desc='COM [x,y,z]')
        
        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack inputs
        #r_s               = inputs['r_s']
        rad_ag            = inputs['rad_ag']
        l_s               = inputs['l_s']
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

        
        # Assign values to universal constants
        B_r    = 1.2                 # Tesla remnant flux density
        g1     = 9.81                # m / s^2 acceleration due to gravity
        E      = 2e11                # N / m^2 young's modulus
        sigma  = 40e3                # shear stress assumed
        ratio  = 0.7                 # ratio of magnet width to pole pitch(bm / tau_p)
        mu_0   = pi * 4e-7           # permeability of free space
        mu_r   = 1.06                # relative permeability 
        phi    = radians(90)         # tilt angle (rotor tilt -90 degrees during transportation)
        cofi   = 0.85                # power factor
        
        # Assign values to design constants
        h_w       = 0.005            # Slot wedge height
        h_i       = 0.001            # coil insulation thickness
        y_tau_p   = 1                # Coil span to pole pitch
        m         = 3                # no of phases
        q1        = 1                # no of slots per pole per phase
        b_s_tau_s = 0.45             # slot width to slot pitch ratio
        k_sfil    = 0.65             # Slot fill factor
        P_Fe0h    = 4                # specific hysteresis losses W / kg @ 1.5 T 
        P_Fe0e    = 1                # specific eddy losses W / kg @ 1.5 T 
        rho_Cu    = 1.8e-8 * 1.4     # Copper resisitivty
        k_fes     = 0.9              # Stator iron fill factor per Grauers
        b_so      =  0.004           # Slot opening
        alpha_p   =  pi / 2 * 0.7

        # back iron thickness for rotor and stator
        t_s = h_ys    
        t   = h_yr
        
        
        ###################################################### Electromagnetic design#############################################
        
        K_rad = l_s / (2 * rad_ag)           # Aspect ratio
        #T     = Torque                    # rated torque
        l_u   = k_fes * l_s               # useful iron stack length
        We    = tau_p
        l_b   = 2 * tau_p                 # end winding length
        l_e   = l_s + 2 * 0.001 * rad_ag     # equivalent core length
        b_m   = 0.7 * tau_p               # magnet width     
        
        
        # Calculating air gap length
        dia   =  2 * rad_ag                  # air gap diameter
        ag_len     =  0.001 * dia              # air gap length
        r_m   =  rad_ag + h_ys + h_s         # magnet radius
        r_r   =  rad_ag - ag_len                  # rotor radius
        
        p     =  np.round(pi * dia / (2 * tau_p))  # pole pairs
        f     =  n_nom * p / 60                    # outout frequency
        S     = 2 * p*q1 * m                       # Stator slots
        N_conductors = S * 2                    
        N_s = N_conductors / 2/3                   # Stator turns per phase
        tau_s = pi * dia / S                       # Stator slot pitch
        b_s    =  b_s_tau_s * tau_s                # slot width 
        b_t    =  tau_s - (b_s)                    # tooth width
        Slot_aspect_ratio = h_s / b_s
        
        # Calculating Carter factor for statorand effective air gap length
        gamma  =  4 / pi * (b_so / 2/(ag_len + h_m / mu_r) * atan(b_so / 2/(ag_len + h_m / mu_r)) - log(sqrt(1 + (b_so / 2/(ag_len + h_m / mu_r))**2)))
        k_C    =  tau_s / (tau_s - gamma * (ag_len + h_m / mu_r))   # carter coefficient
        g_eff  =  k_C * (ag_len + h_m / mu_r)
        
        # angular frequency in radians
        om_m            =  2 * pi * n_nom / 60
        om_e            =  p * om_m / 2

        
        # Calculating magnetic loading
        B_pm1   = B_r * h_m / mu_r / g_eff
        B_g     = B_r * h_m / mu_r / g_eff * (4 / pi) * sin(alpha_p)
        B_symax = B_g * b_m * l_e / (2 * h_ys * l_u)
        B_rymax = B_g * b_m * l_e / (2 * h_yr * l_s)
        B_tmax  = B_g * tau_s / b_t
        
        # Calculating winding factor
        k_wd    = sin(pi / 6) / q1 / sin(pi / 6/q1)      
        
        L_t = l_s + 2 * tau_p
        
        #l = L_t                          # length - now using L_t everywhere
        
        # Stator winding length, cross-section and resistance
        l_Cus     = 2 * N_s * (2 * tau_p + L_t)
        A_s       = b_s        * (h_s - h_w)        * q1 * p
        A_scalc   = b_s * 1000 * (h_s - h_w) * 1000 * q1 * p
        A_Cus     = A_s * k_sfil / N_s
        A_Cuscalc = A_scalc * k_sfil / N_s
        R_s       = l_Cus * rho_Cu / A_Cus
        
        # Calculating leakage inductance in  stator
        L_m        = 2 * mu_0 * N_s**2 / p * m * k_wd**2 * tau_p * L_t / pi**2 / g_eff
        L_ssigmas  = 2 * mu_0 * N_s**2 / p / q1 * l_s  * ((h_s - h_w) / (3 * b_s) + h_w / b_so)                        # slot leakage inductance
        L_ssigmaew = 2 * mu_0 * N_s**2 / p / q1 * l_s  * 0.34 * ag_len * (l_e - 0.64 * tau_p * y_tau_p) / l_s          # end winding leakage inductance
        L_ssigmag  = 2 * mu_0 * N_s**2 / p / q1 * l_s  * (5 * (ag_len * k_C / b_so) / (5 + 4 * (ag_len * k_C / b_so))) # tooth tip leakage inductance
        L_ssigma   = L_ssigmas + L_ssigmaew + L_ssigmag
        L_s  = L_m + L_ssigma
        
        # Calculating no-load voltage induced in the stator
        E_p    = 2 * N_s * L_t * rad_ag * k_wd * om_m * B_g / sqrt(2)
        
        Z = machine_rating / (m * E_p)        
        G = E_p**2 - (om_e * L_s * Z)**2

        # Calculating stator current and electrical loading
        I_s     = sqrt(Z**2 + (((E_p - G**0.5) / (om_e * L_s)**2)**2))
        J_s     = I_s / A_Cuscalc
        A_1     = 6 * N_s * I_s / (pi * dia)
        I_snom  = machine_rating / (m * E_p * cofi) # rated current                
        I_qnom  = machine_rating / (m * E_p)
        X_snom  = om_e * (L_m + L_ssigma)
                
        B_smax  = sqrt(2) * I_s * mu_0 / g_eff
        
        # Calculating Electromagnetically active mass
        
        V_Cus   = m * l_Cus * A_Cus              # copper volume
        V_Fest  = L_t * 2*p * q1 * m * b_t * h_s # volume of iron in stator tooth
        
        V_Fesy  = L_t * pi * ((rad_ag + h_s + h_ys)**2 - (rad_ag + h_s)**2) # volume of iron in stator yoke
        V_Fery  = L_t * pi * ((r_r - h_m)**2 - (r_r - h_m - h_yr)**2)
        Copper  = V_Cus * rho_Copper

        M_Fest  = V_Fest * rho_Fe    # Mass of stator tooth
        M_Fesy  = V_Fesy * rho_Fe    # Mass of stator yoke
        M_Fery  = V_Fery * rho_Fe    # Mass of rotor yoke
        Iron    = M_Fest + M_Fesy + M_Fery
        
        # Calculating Losses
        ##1. Copper Losses
        
        K_R = 1.2   # Skin effect correction co - efficient
        P_Cu        = m * I_snom**2 * R_s * K_R

        # Iron Losses ( from Hysteresis and eddy currents) 
        P_Hyys    = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))      # Hysteresis losses in stator yoke
        P_Ftys    = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2) # Eddy       losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys
        
        P_Hyd     = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))       # Hysteresis losses in stator teeth
        P_Ftd     = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)  # Eddy       losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd
        
        # additional stray losses due to leakage flux
        P_ad = 0.2 * (P_Hyys + P_Ftys + P_Hyd + P_Ftd ) 
        pFtm = 300 # specific magnet loss
        P_Ftm = pFtm * 2 * p * b_m * l_s
        
        Losses = P_Cu + P_Festnom + P_Fesynom + P_ad + P_Ftm
        gen_eff = machine_rating * 100 / (machine_rating + Losses)
        
        #################################################### Structural  Design ############################################################
        
        ## Deflection Calculations ##
        # rotor structure calculations
        
        a_r         = (b_r * d_r) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr))  # cross-sectional area of rotor arms
        A_r         = L_t * t                                              # cross-sectional area of rotor cylinder
        N_r         = np.round(n_r)                                        # rotor arms
        theta_r     = pi * 1 / N_r                                         # half angle between spokes
        I_r         = L_t * t**3 / 12                                      # second moment of area of rotor cylinder
        I_arm_axi_r = ((b_r * d_r**3) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr)**3)) / 12  # second moment of area of rotor arm
        I_arm_tor_r = ((d_r * b_r**3) - ((d_r - 2 * t_wr) * (b_r - 2 * t_wr)**3)) / 12  # second moment of area of rotot arm w.r.t torsion
        R           = rad_ag - ag_len - h_m - 0.5 * t                         # Rotor mean radius
        c           = R / 500                                              
        u_all_r     = c / 20                                               # allowable radial deflection
        R_1         = R - t * 0.5                                          # inner radius of rotor cylinder
        k_1         = sqrt(I_r / A_r)                                      # radius of gyration
        m1          = (k_1 / R)**2                                         
        l_ir        = R                                                    # length of rotor arm beam at which rotor cylinder acts
        l_iir       = R_1
        
        
        b_all_r     = 2 * pi * R_o / N_r                                   # allowable circumferential arm dimension for rotor
        q3          = B_g**2 / 2 / mu_0                                    # normal component of Maxwell stress
        mass_PM     = 2 * pi * (R + 0.5 * t) * L_t * h_m * ratio * rho_PM  # magnet mass
        
        # Calculating radial deflection of the rotor
        Numer = R**3 * ((0.25 * (sin(theta_r) - (theta_r * cos(theta_r))) / (sin(theta_r))**2) - (0.5 / sin(theta_r)) + (0.5 / theta_r))
        Pov   = ((theta_r / (sin(theta_r))**2) + 1 / tan(theta_r)) * ((0.25 * R / A_r) + (0.25 * R**3 / I_r))
        Qov   = R**3 / (2 * I_r * theta_r * (m1 + 1))
        Lov   = (R_1 - R_o) / a_r
        Denom = I_r * (Pov - Qov + Lov) # radial deflection % rotor        
        u_Ar  = (q3 * R**2 / E / t) * (1 + Numer / Denom)
        
        # Calculating axial deflection of the rotor under its own weight
        
        w_r         = rho_Fes * g1 * sin(phi) * a_r * N_r                     # uniformly distributed load of the weight of the rotor arm
        mass_st_lam = rho_Fe * 2*pi * R * L_t * h_yr                          # mass of rotor yoke steel
        W           = g1 * sin(phi) * (mass_st_lam / N_r + mass_PM / N_r)     # weight of 1/nth of rotor cylinder
        
        y_a1 = W   * l_ir**3  / 12 / E / I_arm_axi_r                          # deflection from weight component of back iron
        y_a2 = w_r * l_iir**4 / 24 / E / I_arm_axi_r                          # deflection from weight component of the arms
        y_Ar = y_a1 + y_a2 # axial deflection
        
        y_all     = 2 * L_t / 100    # allowable axial deflection
        
        # Calculating # circumferential deflection of the rotor
        
        z_all_r     = radians(0.05  * R)                                   # allowable torsional deflection
        z_A_r       = (2 * pi * (R - 0.5 * t) * L_t / N_r) * sigma \
                      * (l_ir - 0.5 * t)**3 / 3 / E / I_arm_tor_r          # circumferential deflection
        
        val_str_rotor = mass_PM + (mass_st_lam + (N_r * (R_1 - R_o) * a_r * rho_Fes))           # rotor mass
        
        # stator structure deflection calculation
        a_s         = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws)) # cross-sectional area of stator armms
        A_st        = L_t * t_s                                             # cross-sectional area of stator cylinder
        N_st        = np.round(n_s)                                         # stator arms
        theta_s     = pi * 1 / N_st                                         # half angle between spokes
        I_st        = L_t * t_s**3 / 12                                     # second moment of area of stator cylinder
        k_2         = sqrt(I_st / A_st)                                     # radius of gyration
        I_arm_axi_s = ((b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws)**3)) / 12  # second moment of area of stator arm
        I_arm_tor_s = ((d_s * b_st**3) - ((d_s - 2 * t_ws) * (b_st - 2 * t_ws)**3)) / 12  # second moment of area of rotot arm w.r.t torsion
        R_st        = rad_ag + h_s + h_ys * 0.5                                # stator cylinder mean radius
        R_1s        = R_st - t_s * 0.5                                      # inner radius of stator cylinder, m
        m2          = (k_2 / R_st)**2
        d_se        = dia + 2 * (h_ys + h_s + h_w)                          # stator outer diameter
        
        # allowable radial deflection of stator
        c1      = R_st / 500
        u_all_s = c1 / 20
        R_out   = (R / 0.995 + h_s + h_ys)
        l_is    = R_st - R_o              # distance at which the weight of the stator cylinder acts
        l_iis   = l_is                    # distance at which the weight of the stator cylinder acts
        l_iiis  = l_is                    # distance at which the weight of the stator cylinder acts
        
        mass_st_lam_s = M_Fest + pi * L_t * rho_Fe * ((R_st + 0.5 * h_ys)**2 - (R_st - 0.5 * h_ys)**2)
        W_is          = 0.5 * g1 * sin(phi) * (rho_Fes * L_t  *d_s**2)                      # length of stator arm beam at which self-weight acts
        W_iis         = g1 * sin(phi) * (mass_st_lam_s + V_Cus * rho_Copper) / 2/N_st       # weight of stator cylinder and teeth
        w_s           = rho_Fes * g1 * sin(phi) * a_s * N_st                                # uniformly distributed load of the arms
                
        mass_stru_steel  = 2 * (N_st * (R_1s - R_o) * a_s * rho_Fes)                        # Structural mass of stator arms
        
        # Calculating radial deflection of the stator
        
        Numers = R_st**3 * ((0.25 * (sin(theta_s) - (theta_s * cos(theta_s))) / (sin(theta_s))**2) - (0.5 / sin(theta_s)) + (0.5 / theta_s))
        Povs   = ((theta_s / (sin(theta_s))**2) + 1 / tan(theta_s)) * ((0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st))
        Qovs   = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs   = (R_1s - R_o) * 0.5 / a_s
        Denoms = I_st * (Povs - Qovs + Lovs)
        u_As   = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)

        # Calculating axial deflection of the stator
        X_comp1 = W_is  * l_is**3   / 12 / E / I_arm_axi_s  # deflection component due to stator arm beam at which self-weight acts
        X_comp2 = W_iis * l_iis**4  / 24 / E / I_arm_axi_s  # deflection component due to 1 / nth of stator cylinder
        X_comp3 = w_s   * l_iiis**4 / 24 / E / I_arm_axi_s  # deflection component due to weight of arms
        y_As    = X_comp1 + X_comp2 + X_comp3               # axial deflection
        
        # Calculating circumferential deflection of the stator
        z_A_s   = 2 * pi * (R_st + 0.5 * t_s) * L_t / (2 * N_st) * sigma * (l_is + 0.5 * t_s)**3 / 3/E / I_arm_tor_s 
        z_all_s = radians(0.05 * R_st)               # allowable torsional deflection
        b_all_s = 2 * pi * R_o / N_st                # allowable circumferential arm dimension
        
        val_str_stator = mass_stru_steel + mass_st_lam_s   
        val_str_mass   = val_str_rotor + val_str_stator
        
        TC1 = Torque / (2 * pi * sigma)  # Desired shear stress
        TC2 = R**2 * L_t            # Evaluating Torque constraint for rotor
        TC3 = R_st**2 * L_t         # Evaluating Torque constraint for stator
        
        Structural_mass = mass_stru_steel + (N_r * (R_1 - R_o) * a_r * rho_Fes)
        Stator = mass_st_lam_s + mass_stru_steel + Copper
        Rotor = ((2 * pi * t*L_t * R * rho_Fe) + (N_r * (R_1 - R_o) * a_r * rho_Fes)) + mass_PM
        Mass = Stator + Rotor
        
        
        # Calculating mass moments of inertia and center of mass
        
        I = np.zeros(3)
        I[0]   = 0.50 * Mass * R_out**2
        I[1]   = 0.25 * Mass * R_out**2 + Mass * l_s**2 / 12
        I[2]   = I[1]
        cm = np.zeros(3)
        cm[0]  = shaft_cm[0] + shaft_length / 2. + l_s / 2
        cm[1]  = shaft_cm[1]
        cm[2]  = shaft_cm[2]

        outputs['B_symax']           =  B_symax
        outputs['B_tmax']            =  B_tmax
        outputs['B_rymax']           =  B_rymax
        outputs['B_smax']            =  B_smax
        outputs['B_pm1']             =  B_pm1
        outputs['B_g']               =  B_g
        outputs['N_s']               =  N_s
        outputs['b_s']               =  b_s
        
        outputs['b_t']               =  b_t
        outputs['A_Cuscalc']         =  A_Cuscalc
        outputs['b_m']               =  b_m
        outputs['p']                 =  p
        outputs['E_p']               =  E_p
        outputs['f']                 =  f

        outputs['I_s']               =  I_s
        outputs['R_s']               =  R_s
        outputs['L_s']               =  L_s
        outputs['A_1']               =  A_1
        outputs['J_s']               =  J_s
        outputs['Losses']            =  Losses

        outputs['K_rad']             =  K_rad
        outputs['gen_eff']           =  gen_eff
        outputs['S']                 =  S
        outputs['Slot_aspect_ratio'] =  Slot_aspect_ratio
        outputs['Copper']            =  Copper
        outputs['Iron']              =  Iron
        outputs['u_Ar']              =  u_Ar
        outputs['y_Ar']              =  y_Ar

        outputs['z_A_r']             =  z_A_r
        outputs['u_As']              =  u_As
        outputs['y_As']              =  y_As
        outputs['z_A_s']             =  z_A_s
        outputs['u_all_r']           =  u_all_r
        outputs['u_all_s']           =  u_all_s

        outputs['y_all']             =  y_all
        outputs['z_all_s']           =  z_all_s
        outputs['z_all_r']           =  z_all_r
        outputs['b_all_s']           =  b_all_s
        outputs['b_all_r']           =  b_all_r
        outputs['TC1']               =  TC1

        outputs['TC2']               =  TC2
        outputs['TC3']               =  TC3
        outputs['R_out']             =  R_out
        outputs['Structural_mass']   =  Structural_mass
        outputs['Mass']              =  Mass
        outputs['mass_PM']           =  mass_PM
        outputs['cm']                =  cm
        outputs['I']                 =  I
        

