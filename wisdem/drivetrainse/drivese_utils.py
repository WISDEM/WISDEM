# Utility functions used by drivese components

import numpy as np
import scipy as scp
from math import pi, cos, sqrt, sin, exp, log10, log
import sys

#-------------------------------------------------------------------------
# Supporting functions

#If user does not specify key information about turbine properties,
#they are estimated from curve fits to basic turbine configuration parameters

# if user inputs forces and zero rotor mass then rotor mass estimated
def get_rotor_mass(machine_rating, deriv):  
    out = [23.566 * machine_rating]
    if deriv:
        out.extend([23.566])
    return out

# moments taken to scale approximately with force (rotor mass) and distance (distance_hub2mb)
def get_My(rotor_mass, distance_hub2mb):
    if distance_hub2mb == 0:
        # approximate rotor diameter from rotor mass
        distance_hub2mb = get_distance_hub2mb((rotor_mass + 49089) / 1170.6)
    return 59.7 * rotor_mass * distance_hub2mb

# moments taken to scale roughly with force (rotor mass) and distance (distance_hub2mb)
def get_Mz(rotor_mass, distance_hub2mb):  
    if distance_hub2mb == 0:
        # approximate rotor diameter from rotor mass
        distance_hub2mb = get_distance_hub2mb((rotor_mass - 49089) / 1170.6)
    return 53.846 * rotor_mass * distance_hub2mb

# function to estimate the location of the main bearing location from the hub center
def get_distance_hub2mb(rotor_diameter, deriv=False):
    out = 0.007835 * rotor_diameter + 0.9642
    # Ignoring deriv calculation for now
    #if deriv:
    #    out.extend([.007835])
    return out

# -------------------------------------------------
# Bearing support functions
# returns facewidth, mass for bearings without fatigue analysis
def resize_for_bearings(D_shaft, type, deriv):
    # assume low load rating for bearing
    if type == 'CARB':  # p = Fr, so X=1, Y=0
        out = [D_shaft, .2663 * D_shaft + .0435, 1561.4 * D_shaft**2.6007]
        if deriv == True:
            out.extend([1., .2663, 1561.4 * 2.6007 * D_shaft**1.6007])
    elif type == 'SRB':
        out = [D_shaft, .2762 * D_shaft, 876.7 * D_shaft**1.7195]
        if deriv == True:
            out.extend([1., .2762, 876.7 * 1.7195 * D_shaft**0.7195])
    elif type == 'TRB1':
        out = [D_shaft, .0740, 92.863 * D_shaft**.8399]
        if deriv == True:
            out.extend([1., 0., 92.863 * 0.8399 * D_shaft**(0.8399 - 1.)])
    elif type == 'CRB':
        out = [D_shaft, .1136 * D_shaft, 304.19 * D_shaft**1.8885]
        if deriv == True:
            out.extend([1., .1136, 304.19 * 1.8885 * D_shaft**0.8885])
    elif type == 'TRB2':
        out = [D_shaft, .1499 * D_shaft, 543.01 * D_shaft**1.9043]
        if deriv == True:
            out.extend([1., .1499, 543.01 * 1.9043 * D_shaft**.9043])
    elif type == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        out = [D_shaft, .0839, 229.47 * D_shaft**1.8036]
        if deriv == True:
            out.extend([1.0, 0.0, 229.47 * 1.8036 * D_shaft**0.8036])

    # shaft diameter, facewidth, mass. if deriv==True, provides derivatives.
    return out


#%%---------------------------------------------------------------
# Revised mainshaft flange calculations - added July 2019
    
def mainshaftFlangeCalc(D_in, D_face_MB, flange_thickness, debug=False):
    ''' Compute mass, center of mass, and cost of mainshaft flange
    
        Rxx identifiers in comments are the row numbers in the original Excel spreadsheet
    
        Parameters
        ----------
        D_in : float
          Mainshaft Inner Bore Diameter (from DriveSE == D_max * shaft_ratio (0.10))
        D_face_MB : float
          Mainshaft Up-Wind Bearing Face Diameter (from DriveSE == D_max ??)
        flange_thickness : float
          Mainshaft Flange Thickness (from HubSE - SphHub::main_flange_thick ??)
          
        Returns
        -------
        flange_length : float
          flange length in m
        mass_flange : float
          flange mass in kg
        cm_flange : float
          flange center of mass location in m
        cost_flange : float
          flange cost in $
    '''
	
    PCT_SHOULDER_HT_ID = 3.15     # R12 Main Bearing Shaft Shoulder Height (as a percentage of ID)  3.150  %
    RATIO_SHOULDER_W2H = 6        # R15 Main Bearing Shaft Shoulder Width to Height Ratio  6.000  
    RATIO_SHOULDER_STEPDOWN = 3   # R17 Main Bearing Shaft Shoulder Width Step down ratio  3.000  
    RATIO_R2D = 0.3               # R20 r/d ratio (Petersons SCF Guide)  0.30  
    FLANGE_BOLT_SIZE = 0.048      # R23 Flange Bolt Size (Diameter)  0.048  m
    FLANGE_BOLT_DIAM_INCR = 0.003 # R24 Flange Bolt Diameter Increase  0.003  m
    RATIO_DIST2BHD = 1.5          # R26 Distance from Bolt Circle Center to Flange Edge (as ratio of Bolt Hole Diameter)  1.500  
    DENSITY_42CrMo4 = 7800        # R36 Density of Forging (42CrMo4 Steel)  7800  kg/m3
    DENSITY_S355    = 7850        # R37 Density of Structural Steel (S355 Steel)  7850  kg/m3
    COST_FORGING = 3.5            # R61 Mainshaft Flange Forging Cost (42CrMo4)  3.5  USD/kg
    COST_LOCKPLATE = 3.0          # R62 Rotor Lock Plate Cost (S355)  3  USD/kg
    
    density_forging = DENSITY_42CrMo4
    
    # Mainshaft Flange Geometry                                                                                           
                                                                                      #      
    shoulder_diam = D_face_MB * ((100 + PCT_SHOULDER_HT_ID) / 100)                    #  R13 Main Bearing Shaft Shoulder Height Diameter  1.155  m
    shoulder_height = (shoulder_diam - D_face_MB) / 2                                 #  R14 Main Bearing Shaft Shoulder Height  0.018  m
    shoulder_width = shoulder_height * RATIO_SHOULDER_W2H                             #  R16 Main Bearing Shaft Shoulder Width  0.106  m
    shoulder_stepdown_width = shoulder_width / RATIO_SHOULDER_STEPDOWN                #  R18 Main Bearing Shaft Shoulder Width Step down Width  0.035  m
    tot_shoulder_width = shoulder_width + shoulder_stepdown_width                     #      
                                                                                      #   
    flange_trans_radius = D_face_MB * RATIO_R2D                                       #  R21 Mainshaft to Mainshaft Flange Transition Radius  0.336  m
    flange_length = shoulder_width + shoulder_stepdown_width \
                  + flange_trans_radius + flange_thickness                            #  R30 Mainshaft Flange Length (Total)  0.817  m
    
    bolt_circle_ID = D_face_MB + (2 * flange_trans_radius)                            #  R22 Flange Bolt Circle  ID  1.792  m
    bolt_hole_diam = FLANGE_BOLT_SIZE + FLANGE_BOLT_DIAM_INCR                         #  R25 Flange Bolt Hole Diameter  0.051  m
    bolt_circ_flange_len = 2 * (RATIO_DIST2BHD * bolt_hole_diam)                      #  R27 Bolt Circle Flange Length (Distance between ID and OD)  0.153  m
    bolt_circle_OD = bolt_circle_ID + (2 * bolt_circ_flange_len)                      #  R28 Bolt Circle OD (Mainshaft OD)  2.098  m
                                                                                      #
    # Mainshaft Flange Design Allowable and Material Properties                       #
    # Mainshaft Flange Mass Calculations                                              #
    # Mainshaft Flange Volume(s) - subcomponents   
                                       #
    vol_shoulder  = (pi/4) * tot_shoulder_width  * (shoulder_diam**2 - D_in**2)       #  R40 Mainshaft Volume (Main bearing Shoulder)  0.120  m3
    vol_tzone1    = (pi/4) * flange_trans_radius * (D_face_MB**2     - D_in**2)       #  R41 Mainshaft Volume (Transition zone 1-base)  0.265  m3
    vol_tzone2    = (1 - pi/4) * (pi/4) * flange_trans_radius \
                             * ((D_face_MB + flange_trans_radius)**2 - D_in**2)       #  R42 Mainshaft Volume (Transition zone 2-fillet)  0.106  m3
    vol_bcflange  = (pi/4) * flange_thickness *   (bolt_circle_OD**2 - D_in**2)       #  R43 Mainshaft Volume (Bolt Circle Flange)  1.109  m3
    vol_flange = vol_shoulder + vol_tzone1 + vol_tzone2 + vol_bcflange                #  R44 Mainshaft Volume (TOTAL)  1.600  m3
                                                                                      #   
    mass_flange = vol_flange * density_forging                                        #  R46 Mainshaft Flange Mass (TOTAL)  12,478.4  kg
                                                                                      #  
    # Mainshaft Flange Centroid Calculations 
                                             #  
    mass_shoulder = vol_shoulder * density_forging                                    #  R49 Mainshaft Mass (Main bearing Shoulder)  937.7  kg
    mass_tzone1   = vol_tzone1   * density_forging                                    #  R51 Mainshaft Mass (Transition zone 1-base)  2,067.4  kg
    mass_tzone2   = vol_tzone2   * density_forging                                    #  R53 Mainshaft Mass (Transition zone 2-fillet)  826.0  kg
    mass_bcflange = vol_bcflange * density_forging                                    #  R55 Mainshaft Mass (Bolt Circle Flange)  8,647.3  kg
                                                                    
    cm_shoulder = 0.5 * tot_shoulder_width                                            #  R50 Center of Mass (Main bearing Shoulder)    0.071  m
    cm_tzone1   = tot_shoulder_width + (flange_trans_radius/2)                        #  R52 Center of Mass (Transition zone 1-base)  0.309  m
    cm_tzone2   = tot_shoulder_width + (2 * flange_trans_radius/3)                    #  R54 Center of Mass (Transition zone 2-fillet)   0.365  m
    cm_bcflange = tot_shoulder_width + (flange_thickness/2) + flange_trans_radius     #  R56 Center of Mass (Bolt Circle Flange)  0.647  m
                                                                                      #  
    cm_flange = ((mass_shoulder * cm_shoulder) \
                  + (mass_tzone1   * cm_tzone1) \
                  + (mass_tzone2   * cm_tzone2) \
                  + (mass_bcflange * cm_bcflange)) \
             / (mass_shoulder + mass_tzone1 + mass_tzone2 + mass_bcflange)            #  R58 Mainshaft Center of Mass  0.529  m

    cost_flange = mass_flange * COST_FORGING                                          #  R64 Total Mainshaft Flange Cost  43,674.50  USD 
	                                                                                  #  R65 Total Rotor Lock Cost    USD
    # TODO: add lockplate - mass, cm, cost
    
    if debug:
        sys.stderr.write('msFlange: len {:.2f} m  mass {:.1f} kg  cm {:.2f} m  cost ${:.2f}\n'.format(flange_length, mass_flange, cm_flange, cost_flange))

    return flange_length, mass_flange, cm_flange, cost_flange

#%% Transform functions for rotor forces and moments (not currently used)
# ---------------------------------------------------------------------
class blade_moment_transform(object):
    ''' Blade_Moment_Transform class          
          The Blade_Moment_Transform class is used to transform moments from the WISDEM rotor models to driveSE.
    '''

    def __init__(self):
        super(blade_moment_transform, self).__init__()

        # variables
        # ensure angles are in radians. Azimuth is 3-element array with blade
        # azimuths; b1, b2, b3 are 3-element arrays for each blade moment (Mx, My,
        # Mz); pitch and cone are floats
        self.add_param('azimuth_angle', val=np.array([0, 2 * pi / 3, 4 * pi / 3]), units='rad', desc='azimuth angles for each blade')
        self.add_param('pitch_angle', val=0.0, units='rad', desc='pitch angle at each blade, assumed same')
        self.add_param('cone_angle', val=0.0, units='rad', desc='cone angle at each blade, assumed same')
        self.add_param('b1', val=np.array([]), units='N*m', desc='moments in x,y,z directions along local blade coordinate system')
        self.add_param('b2', val=np.array([]), units='N*m', desc='moments in x,y,z directions along local blade coordinate system')
        self.add_param('b3', val=np.array([]), units='N*m', desc='moments in x,y,z directions along local blade coordinate system')

        # returns
        self.add_output('Mx', val=0.0, units='N*m',  desc='rotor moment in x-direction')
        self.add_output('My', val=0.0, units='N*m',  desc='rotor moment in y-direction')
        self.add_output('Mz', val=0.0, units='N*m',  desc='rotor moment in z-direction')

    def solve_nonlinear(self, params, unknowns, resids):

        # nested function for transformations
        def trans(alpha, con, phi, bMx, bMy, bMz):
            Mx = bMx * cos(con) * cos(alpha) - bMy * (sin(con) * cos(alpha) * sin(phi) - sin(
                alpha) * cos(phi)) + bMz * (sin(con) * cos(alpha) * cos(phi) - sin(alpha) * sin(phi))
            My = bMx * cos(con) * sin(alpha) - bMy * (sin(con) * sin(alpha) * sin(phi) + cos(
                alpha) * cos(phi)) + bMz * (sin(con) * sin(alpha) * cos(phi) + cos(alpha) * sin(phi))
            Mz = bMx * (-sin(alpha)) - bMy * (-cos(alpha) *
                                              sin(phi)) + bMz * (cos(alpha) * cos(phi))

            return [Mx, My, Mz]

        [b1Mx, b1My, b1Mz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   0], self.b1[0], self.b1[1], self.b1[2])
        [b2Mx, b2My, b2Mz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   1], self.b2[0], self.b2[1], self.b2[2])
        [b3Mx, b3My, b3Mz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   2], self.b3[0], self.b3[1], self.b3[2])

        self.Mx = b1Mx + b2Mx + b3Mx
        self.My = b1My + b2My + b3My
        self.Mz = b1Mz + b2Mz + b3Mz


class blade_force_transform(object):
    ''' Blade_Force_Transform class          
          The Blade_Force_Transform class is used to transform forces from the WISDEM rotor models to driveSE.
    '''

    def __init__(self):
        super(blade_force_transform, self).__init__()

        # variables
        # ensure angles are in radians. Azimuth is 3-element array with blade
        # azimuths; b1, b2, b3 are 3-element arrays for each blade force (Fx, Fy,
        # Fz); pitch and cone are floats
        self.add_param('azimuth_angle', val=np.array([0, 2 * pi / 3, 4 * pi / 3]), units='rad', desc='azimuth angles for each blade')
        self.add_param('pitch_angle', val=0.0, units='rad', desc='pitch angle at each blade, assumed same')
        self.add_param('cone_angle', val=0.0, units='rad', desc='cone angle at each blade, assumed same')
        self.add_param('b1', val=np.array([]), units='N', desc='forces in x,y,z directions along local blade coordinate system')
        self.add_param('b2', val=np.array([]), units='N', desc='forces in x,y,z directions along local blade coordinate system')
        self.add_param('b3', val=np.array([]), units='N', desc='forces in x,y,z directions along local blade coordinate system')

        # returns
        self.add_output('Fx', val=0.0, units='N',  desc='rotor force in x-direction')
        self.add_output('Fy', val=0.0, units='N',  desc='rotor force in y-direction')
        self.add_output('Fz', val=0.0, units='N',  desc='rotor force in z-direction')
        
    def solve_nonlinear(self, params, unknowns, resids):

        # nested function for transformations
        def trans(alpha, con, phi, bFx, bFy, bFz):
            Fx = bFx * cos(con) * cos(alpha) - bFy * (sin(con) * cos(alpha) * sin(phi) - sin(
                alpha) * cos(phi)) + bFz * (sin(con) * cos(alpha) * cos(phi) - sin(alpha) * sin(phi))
            Fy = bFx * cos(con) * sin(alpha) - bFy * (sin(con) * sin(alpha) * sin(phi) + cos(
                alpha) * cos(phi)) + bFz * (sin(con) * sin(alpha) * cos(phi) + cos(alpha) * sin(phi))
            Fz = bFx * (-sin(alpha)) - bFy * (-cos(alpha) *
                                              sin(phi)) + bFz * (cos(alpha) * cos(phi))

            return [Fx, Fy, Fz]

        [b1Fx, b1Fy, b1Fz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   0], self.b1[0], self.b1[1], self.b1[2])
        [b2Fx, b2Fy, b2Fz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   1], self.b2[0], self.b2[1], self.b2[2])
        [b3Fx, b3Fy, b3Fz] = trans(self.pitch_angle, self.cone_angle, self.azimuth_angle[
                                   2], self.b3[0], self.b3[1], self.b3[2])

        self.Fx = b1Fx + b2Fx + b3Fx
        self.Fy = b1Fy + b2Fy + b3Fy
        self.Fz = b1Fz + b2Fz + b3Fz

#-------------------------------------------------------------------------
# Fatigue calculations supporting functions and code for low speed shaft and main bearing(s) (not currently used)
# Developed 2014 by Taylor Parsons - requires additional testing and development to complete
#-------------------------------------------------------------------------

'''
# basic supporting functions
def Ninterp(S, a, b):
    return (S / a)**(1 / b)

def Goodman(S_alt, S_mean, Sut):
    return S_alt / (1 - (S_mean / Sut))

def standardrange(N, N_f, Beta, k_b):
    F_delta = (Beta * (log10(N_f) - log10(N))) + 0.18
    if F_delta >= 2 * k_b:
        F_delta = 0.
    return F_delta

# calculate required dynamic load rating, C
def C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing):
    Fa_ref = np.max(F_a)  # used in comparisons Fa/Fr <e
    Fr_ref = np.max(F_r)

    if Fa_ref / Fr_ref <= e:
        P = F_r + Y1 * F_a
    else:
        P = X2 * F_r + Y2 * F_a

    P_eq = ((scp.integrate.simps((P**p), x=N_array, even='avg')) /
            (N_array[-1] - N_array[0]))**(1 / p)
    C_min = P_eq * (life_bearing / 1e6)**(1. / p) / 1000  # kN
    return C_min

# fatigue analysis for bearings
def fatigue_for_bearings(D_shaft, F_r, F_a, N_array, life_bearing, type, deriv):
    # deriv is boolean, defines if derivatives are returned
    if type == 'CARB':  # p = Fr, so X=1, Y=0
        if (np.max(F_a)) > 0:
            print('---------------------------------------------------------')
            print("error: axial loads too large for CARB bearing application")
            print('---------------------------------------------------------')
        else:
            e = 1
            Y1 = 0.
            X2 = 1.
            Y2 = 0.
            p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 13980 * D_shaft**1.5602:
            out = [D_shaft, 0.4299 * D_shaft +
                   0.0382, 3682.8 * D_shaft**2.7676]
            if deriv:
                out.extend([1., 0.4299, 3682.8 * 2.7676 * D_shaft**1.7676])
        else:
            out = [D_shaft, .2663 * D_shaft + .0435, 1561.4 * D_shaft**2.6007]
            if deriv:
                out.extend([1., .2663, 1561.4 * 2.6007 * D_shaft**1.6007])

    elif type == 'SRB':
        e = 0.32
        Y1 = 2.1
        X2 = 0.67
        Y2 = 3.1
        p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 13878 * D_shaft**1.0796:
            out = [D_shaft, .4801 * D_shaft, 2688.3 * D_shaft**1.8877]
            if deriv:
                out.extend([1., .4801, 2688.3 * 1.8877 * D_shaft**0.8877])
        else:
            out = [D_shaft, .2762 * D_shaft, 876.7 * D_shaft**1.7195]
            if deriv:
                out.extend([1., .2762, 876.7 * 1.7195 * D_shaft**0.7195])

    elif type == 'TRB1':
        e = .37
        Y1 = 0
        X2 = .4
        Y2 = 1.6
        p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 670 * D_shaft + 1690:
            out = [D_shaft, .1335, 269.83 * D_shaft**.441]
            if deriv:
                out.extend([1., 0., 269.83 * 0.441 * D_shaft**(0.441 - 1.)])
        else:
            out = [D_shaft, .0740, 92.863 * D_shaft**.8399]
            if deriv:
                out.extend([1., 0., 92.863 * 0.8399 * D_shaft**(0.8399 - 1.)])

    elif type == 'CRB':
        if (np.max(F_a) / np.max(F_r) >= .5) or (np.min(F_a) / (np.min(F_r)) >= .5):
            print('--------------------------------------------------------')
            print("error: axial loads too large for CRB bearing application")
            print('--------------------------------------------------------')
        else:
            e = 0.2
            Y1 = 0
            X2 = 0.92
            Y2 = 0.6
            p = 10. / 3
            C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
            if C_min > 4526.5 * D_shaft**.9556:
                out = [D_shaft, .2603 * D_shaft, 1070.8 * D_shaft**1.8278]
                if deriv:
                    out.extend([1., .2603, 1070.8 * 1.8278 * D_shaft**0.8278])
            else:
                out = [D_shaft, .1136 * D_shaft, 304.19 * D_shaft**1.8885]
                if deriv:
                    out.extend([1., .1136, 304.19 * 1.8885 * D_shaft**0.8885])

    elif type == 'TRB2':
        e = 0.4
        Y1 = 2.5
        X2 = 0.4
        Y2 = 1.75
        p = 10. / 3
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 6579.9 * D_shaft**.8592:
            out = [D_shaft, .3689 * D_shaft, 1442.6 * D_shaft**1.8932]
            if deriv:
                out.extend([1., .3689, 1442.6 * 1.8932 * D_shaft**.8932])
        else:
            out = [D_shaft, .1499 * D_shaft, 543.01 * D_shaft**1.9043]
            if deriv:
                out.extend([1., .1499, 543.01 * 1.9043 * D_shaft**.9043])

    elif type == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        e = 0.4
        Y1 = 1.6
        X2 = 0.75
        Y2 = 2.15
        p = 3.
        C_min = C_calc(F_a, F_r, N_array, p, e, Y1, Y2, X2, life_bearing)
        if C_min > 884.5 * D_shaft**.9964:
            out = [D_shaft, .1571, 646.46 * D_shaft**2.]
            if deriv:
                out.extend([1., 0., 646.46 * 2. * D_shaft])
        else:
            out = [D_shaft, .0839, 229.47 * D_shaft**1.8036]
            if deriv:
                out.extend([1.0, 0.0, 229.47 * 1.8036 * D_shaft**0.8036])

    return out

#-------------------------------------------------------------------------
def get_Damage_Brng2(self):
    I = (pi / 64.0) * (self.D_med**4 - self.D_in**4)
    J = I * 2
    Area = pi / 4. * (self.D_med**2 - self.D_in**2)
    self.LssWeight = self.density * 9.81 * \
        (((pi / 12) * (self.D_max**2 + self.D_med**2 + self.D_max *
                       self.D_med) * (self.L_mb)) - (pi / 4 * self.L_mb * self.D_in**2))

    self.Fz1stoch = (-self.My_stoch) / (self.L_mb)
    self.Fy1stoch = self.Mz_stoch / self.L_mb
    self.My2stoch = 0.  # My_stoch - abs(Fz1stoch)*self.L_mb #=0
    self.Mz2stoch = 0.  # Mz_stoch - abs(Fy1stoch)*self.L_mb #=0

    # create stochastic loads across N
    stoch_bend2 = (self.My2stoch**2 + self.Mz2stoch **
                   2)**(0.5) * self.D_med / (2. * I)
    stoch_shear2 = abs(self.Mx_stoch * self.D_med / (2. * J))
    # all normal force held by downwind bearing
    stoch_normal2 = self.Fx_stoch / Area * cos(self.shaft_angle)
    stoch_stress2 = ((stoch_bend2 + stoch_normal2) **
                     2 + 3. * stoch_shear2**2)**(0.5)

    # create mean loads
    # Fz_mean*self.distance_hub2mb*self.D_med/(2.*I) #not mean, but deterministic
    mean_bend2 = 0.
    mean_shear2 = self.Mx_mean * self.D_med / (2. * J)
    mean_normal2 = self.Fx_mean / Area * \
        cos(self.shaft_angle) + (self.rotorWeight +
                                 self.LssWeight) * sin(self.shaft_angle)
    mean_stress2 = ((mean_bend2 + mean_normal2) **
                    2 + 3. * mean_shear2**2)**(0.5)
    # apply Goodman with compressive (-) mean stress
    S_mod_stoch2 = Goodman(stoch_stress2, -mean_stress2, self.S_ut)

    # Use Palmgren-Miner linear damage rule to add damage from stochastic
    # load ranges
    DEL_y = self.Fx_stoch.copy()  # initialize
    for i in range(self.num_pts):
        DEL_y[i] = self.N[i] / \
            (Ninterp(S_mod_stoch2[i], self.SN_a, self.SN_b))

    # damage from stochastic loading
    self.Damage = scp.integrate.simps(DEL_y, x=self.N, even='avg')

    # create deterministic loads occurring N_rotor times
    self.Fz1determ = (self.gearboxWeight * self.L_gb - self.LssWeight * .5 *
                      self.L_mb - self.rotorWeight * (self.L_mb + self.distance_hub2mb)) / (self.L_mb)
    # -rotorWeight*(self.distance_hub2mb+self.L_mb) + Fz1determ*self.L_mb - self.LssWeight*.5*self.L_mb + self.gearboxWeight*self.L_gb
    self.My2determ = self.gearboxWeight * self.L_gb
    self.determ_stress2 = abs(self.My2determ * self.D_med / (2. * I))

    S_mod_determ2 = Goodman(self.determ_stress2, -mean_stress2, self.S_ut)

    if S_mod_determ2 > 0:
        self.Damage += self.N_rotor / \
            (Ninterp(S_mod_determ2, self.SN_a, self.SN_b))

def get_Damage_Brng1(self):
    self.D_in = self.shaft_ratio * self.D_max
    self.D_max = (self.D_max**4 + self.D_in**4)**0.25
    self.D_min = (self.D_min**4 + self.D_in**4)**0.25
    I = (pi / 64.0) * (self.D_max**4 - self.D_in**4)
    J = I * 2
    Area = pi / 4. * (self.D_max**2 - self.D_in**2)
    self.LssWeight = self.density * 9.81 * \
        (((pi / 12) * (self.D_max**2 + self.D_min**2 + self.D_max *
                       self.D_min) * (self.L_ms)) - (pi / 4 * self.L_ms * self.D_in**2))

    # create stochastic loads across N
    stoch_bend1 = (self.My_stoch**2 + self.Mz_stoch **
                   2)**(0.5) * self.D_max / (2. * I)
    stoch_shear1 = abs(self.Mx_stoch * self.D_max / (2. * J))
    stoch_normal1 = self.Fx_stoch / Area * cos(self.shaft_angle)
    stoch_stress1 = ((stoch_bend1 + stoch_normal1) **
                     2 + 3. * stoch_shear1**2)**(0.5)

    # create mean loads
    # Fz_mean*self.distance_hub2mb*self.D_max/(2.*I) #not mean, but deterministic
    mean_bend1 = 0
    mean_shear1 = self.Mx_mean * self.D_max / (2. * J)
    mean_normal1 = self.Fx_mean / Area * \
        cos(self.shaft_angle) + (self.rotorWeight +
                                 self.LssWeight) * sin(self.shaft_angle)
    mean_stress1 = ((mean_bend1 + mean_normal1) **
                    2 + 3. * mean_shear1**2)**(0.5)

    # apply Goodman with compressive (-) mean stress
    S_mod_stoch1 = Goodman(stoch_stress1, -mean_stress1, self.S_ut)

    # Use Palmgren-Miner linear damage rule to add damage from stochastic
    # load ranges
    DEL_y = self.Fx_stoch.copy()  # initialize
    for i in range(self.num_pts):
        DEL_y[i] = self.N[i] / \
            (Ninterp(S_mod_stoch1[i], self.SN_a, self.SN_b))

    # damage from stochastic loading
    self.Damage = scp.integrate.simps(DEL_y, x=self.N, even='avg')

    # create deterministic loads occurring N_rotor times
    # only deterministic stress at mb1 is bending due to weights
    determ_stress1 = abs(
        self.rotorWeight * cos(self.shaft_angle) * self.distance_hub2mb * self.D_max / (2. * I))

    S_mod_determ = Goodman(determ_stress1, -mean_stress1, self.S_ut)

    self.Damage += self.N_rotor / \
        (Ninterp(S_mod_determ, self.SN_a, self.SN_b))

def setup_Fatigue_Loads(self):
    R = self.rotor_diameter / 2.0
    rotor_torque = (self.machine_rating * 1000 /
                    self.drivetrain_efficiency) / (self.rotor_freq * (pi / 30))
    Tip_speed_ratio = self.rotor_freq / 30. * pi * R / self.Vrated
    rho_air = 1.225  # kg/m^3 density of air TODO add as input
    p_o = 4. / 3 * rho_air * ((4 * pi * self.rotor_freq / 60 * R / 3)**2 + self.Vrated**2) * (
        pi * R / (self.blade_number * Tip_speed_ratio * (Tip_speed_ratio**2 + 1)**(.5)))
    # characteristic frequency on rotor from turbine of given blade number
    # [Hz]
    n_c = self.blade_number * self.rotor_freq / 60
    # number of rotor rotations based off of weibull curve. .827 comes from
    # lower rpm than rated at lower wind speeds
    self.N_f = self.availability * n_c * (self.T_life * 365 * 24 * 60 * 60) * exp(-(
        self.cut_in / self.weibull_A)**self.weibull_k) - exp(-(self.cut_out / self.weibull_A)**self.weibull_k)

    k_b = 2.5  # calculating rotor pressure from all three blades. Use kb=1 for individual blades

    if self.IEC_Class == 'A':  # From IEC 61400-1 TODO consider calculating based off of 10-minute windspeed and weibull parameters, include neighboring wake effects?
        I_t = 0.18
    elif self.IEC_Class == 'B':
        I_t = 0.14
    else:
        I_t = 0.12

    Beta = 0.11 * k_b * (I_t + 0.1) * (self.weibull_A + 4.4)

    # for analysis with N on log scale, makes larger loads contain finer
    # step sizes
    self.num_pts = 100
    # with zeros: N=np.logspace(log10(1.0),log10(N_f),endpoint=True,num=self.num_pts)
    self.N = np.logspace((log10(self.N_f) - (2 * k_b - 0.18) / Beta),
                         log10(self.N_f), endpoint=True, num=self.num_pts)
    self.N_rotor = self.N_f / 3.
    F_stoch = self.N.copy()

    k_r = 0.8  # assuming natural frequency of rotor is significantly larger than rotor rotational frequency

    for i in range(self.num_pts):
        F_stoch[i] = self.standardrange(self.N[i], self.N_f, Beta, k_b)

    Fx_factor = (.3649 * log(self.rotor_diameter) - 1.074)
    Mx_factor = (.0799 * log(self.rotor_diameter) - .2577)
    My_factor = (.172 * log(self.rotor_diameter) - .5943)
    Mz_factor = (.1659 * log(self.rotor_diameter) - .5795)

    self.Fx_stoch = (F_stoch.copy() * 0.5 * p_o * (R)) * Fx_factor
    self.Mx_stoch = (F_stoch.copy() * 0.45 * p_o *
                     (R)**2) * Mx_factor  # *0.31
    self.My_stoch = (F_stoch.copy() * 0.33 * p_o *
                     k_r * (R)**2) * My_factor  # *0.25
    self.Mz_stoch = (F_stoch.copy() * 0.33 * p_o *
                     k_r * (R)**2) * Mz_factor  # *0.25

    self.Fx_mean = 0.5 * p_o * R * self.blade_number * Fx_factor
    self.Mx_mean = 0.5 * rotor_torque * Mx_factor
    self.rotorWeight = self.rotor_mass * self.g


# Code remove from LowSpeedShaft4pt component
############################################
# inputs into LSS for fatigue calculations:

check_fatigue = Enum(0,(0,1,2),iotype = 'in', desc = 'turns on and off fatigue check')

#fatigue1 variables
rotor_freq = Float(iotype = 'in', units = 'rpm', desc='rated rotor speed')
availability = Float(.95,iotype = 'in', desc = 'turbine availability')
fatigue_exponent = Float(0,iotype = 'in', desc = 'fatigue exponent of material')
S_ut = Float(700e6,iotype = 'in', units = 'Pa', desc = 'ultimate tensile strength of material')
weibull_A = Float(iotype = 'in', units = 'm/s', desc = 'weibull scale parameter "A" of 10-minute windspeed probability distribution')
weibull_k = Float(iotype = 'in', desc = 'weibull shape parameter "k" of 10-minute windspeed probability distribution')
blade_number = Float(iotype = 'in', desc = 'number of blades on rotor, 2 or 3')
cut_in = Float(iotype = 'in', units = 'm/s', desc = 'cut-in windspeed')
cut_out = Float(iotype = 'in', units = 'm/s', desc = 'cut-out windspeed')
Vrated = Float(iotype = 'in', units = 'm/s', desc = 'rated windspeed')
T_life = Float(iotype = 'in', units = 'yr', desc = 'cut-in windspeed')

# fatigue2 variables
rotor_thrust_distribution = Array(iotype='in', units ='N', desc = 'thrust distribution across turbine life')
rotor_thrust_count = Array(iotype='in', desc = 'corresponding cycle array for thrust distribution')
rotor_Fy_distribution = Array(iotype='in', units ='N', desc = 'Fy distribution across turbine life')
rotor_Fy_count = Array(iotype='in', desc = 'corresponding cycle array for Fy distribution')
rotor_Fz_distribution = Array(iotype='in', units ='N', desc = 'Fz distribution across turbine life')
rotor_Fz_count = Array(iotype='in', desc = 'corresponding cycle array for Fz distribution') 
rotor_torque_distribution = Array(iotype='in', units ='N*m', desc = 'torque distribution across turbine life')
rotor_torque_count = Array(iotype='in', desc = 'corresponding cycle array for torque distribution') 
rotor_My_distribution = Array(iotype='in', units ='N*m', desc = 'My distribution across turbine life')
rotor_My_count = Array(iotype='in', desc = 'corresponding cycle array for My distribution') 
rotor_Mz_distribution = Array(iotype='in', units ='N*m', desc = 'Mz distribution across turbine life')
rotor_Mz_count = Array(iotype='in', desc = 'corresponding cycle array for Mz distribution')

# LSS fatigue check calculations 
# fatigue check Taylor Parsons 6/14
if self.check_fatigue == 1 or self.check_fatigue == 2:

    # checks to make sure all inputs are reasonable
    if self.rotor_mass < 100:
        [self.rotor_mass] = get_rotor_mass(self.machine_rating, False)

    # material properties 34CrNiMo6 steel +QT, large diameter
    self.n_safety = 2.5
    if self.S_ut <= 0:
        self.S_ut = 700.0e6  # Pa

    # calculate material props for fatigue
    Sm = 0.9 * self.S_ut  # for bending situations, material strength at 10^3 cycles

    if self.fatigue_exponent != 0:
        if self.fatigue_exponent > 0:
            self.SN_b = - self.fatigue_exponent
        else:
            self.SN_b = self.fatigue_exponent
    else:
        C_size = 0.6  # diameter larger than 10"
        # machined surface 272*(self.S_ut/1e6)**-.995 #forged
        C_surf = 4.51 * (self.S_ut / 1e6)**-.265
        C_temp = 1  # normal operating temps
        C_reliab = 0.814  # 99% reliability
        C_envir = 1.  # enclosed environment
        Se = C_size * C_surf * C_temp * C_reliab * C_envir * .5 * \
            self.S_ut  # modified endurance limit for infinite life (should be Sf)\
        Nfinal = 5e8  # point where fatigue limit occurs under hypothetical S-N curve TODO adjust to fit actual data
        # assuming no endurance limit (high strength steel)
        z = log10(1e3) - log10(Nfinal)
        self.SN_b = 1 / z * log10(Sm / Se)
    self.SN_a = Sm / (1000.**self.SN_b)


    if self.check_fatigue == 1:

        setup_Fatigue_Loads(self)

        # upwind bearing calculations
        iterationstep = 0.001
        diameter_limit = 5.0
        while True:
            get_Damage_Brng1(self)
            if self.Damage < 1 or self.D_max >= diameter_limit:
                break
            else:
                self.D_max += iterationstep

        # downwind bearing calculations
        diameter_limit = 5.0
        iterationstep = 0.001
        while True:
            get_Damage_Brng2(self)
            if self.Damage < 1 or self.D_med >= diameter_limit:
                break
            else:
                self.D_med += iterationstep

        # begin bearing calculations
        # counts per rotation (not defined by characteristic frequency
        # 3n_rotor)
        N_bearings = self.N / self.blade_number

        # radial stochastic + deterministic mean
        Fr1_range = ((abs(self.Fz1stoch) + abs(self.Fz1determ))
                     ** 2 + self.Fy1stoch**2)**.5
        Fa1_range = np.zeros(len(self.Fy1stoch))

        #...calculate downwind forces
        lss_weight = self.density * 9.81 * \
            (((pi / 12) * (self.D_max**2 + self.D_med**2 + self.D_max *
                           self.D_med) * (self.L_mb)) - (pi / 4 * self.L_mb * self.D_in**2))
        Fy2stoch = -self.Mz_stoch / (self.L_mb)  # = -Fy1 - Fy_stoch
        Fz2stoch = -(lss_weight * 2. / 3. * self.L_mb - self.My_stoch) / (self.L_mb) + (lss_weight + self.shrinkDiscWeight + self.gearboxWeight) * \
            cos(self.shaft_angle) - \
            self.rotorWeight  # -Fz1 +Weights*cos(gamma)-Fz_stoch+Fz_mean (Fz_mean is in negative direction)
        Fr2_range = (Fy2stoch**2 + (Fz2stoch + abs(-self.rotorWeight * distance_hub2mb +
                                                   0.5 * lss_weight + self.gearboxWeight * self.L_gb / self.L_mb))**2)**0.5
        Fa2_range = self.Fx_stoch * cos(self.shaft_angle) + (
            self.rotorWeight + lss_weight) * sin(self.shaft_angle)  # axial stochastic + mean

        life_bearing = self.N_f / self.blade_number

        [self.D_max_a, facewidth_max, bearing1mass] = fatigue_for_bearings(
            self.D_max, Fr1_range, Fa1_range, N_bearings, life_bearing, self.mb1Type, False)
        [self.D_med_a, facewidth_med, bearing2mass] = fatigue_for_bearings(
            self.D_med, Fr2_range, Fa2_range, N_bearings, life_bearing, self.mb2Type, False)

    # elif self.check_fatigue == 2: # untested and not used currently
    #   Fx = self.rotor_thrust_distribution
    #   n_Fx = self.rotor_thrust_count
    #   Fy = self.rotor_Fy_distribution
    #   n_Fy = self.rotor_Fy_count
    #   Fz = self.rotor_Fz_distribution
    #   n_Fz = self.rotor_Fz_count
    #   Mx = self.rotor_torque_distribution
    #   n_Mx = self.rotor_torque_count
    #   My = self.rotor_My_distribution
    #   n_My = self.rotor_My_count
    #   Mz = self.rotor_Mz_distribution
    #   n_Mz = self.rotor_Mz_count

    #   def Ninterp(L_ult,L_range,m):
    # return (L_ult/(.5*L_range))**m #TODO double-check that the input
    # will be the load RANGE instead of load amplitudes. May also
    # include means

    #   #upwind bearing calcs
    #   diameter_limit = 5.0
    #   iterationstep=0.001
    #   #upwind bearing calcs
    #   while True:
    #       self.Damage = 0
    #       Fx_ult = self.SN_a*(pi/4.*(self.D_max**2-self.D_in**2))
    #       Fyz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*64.)/self.distance_hub2mb
    #       Mx_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(32*(3)**.5*self.D_max)
    #       Myz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*64.)
    #       if Fx_ult !=0 and np.all(n_Fx) != 0:
    #           self.Damage+=scp.integrate.simps(n_Fx/Ninterp(Fx_ult,Fx,-1/self.SN_b),x=n_Fx,even = 'avg')
    #       if Fyz_ult !=0:
    #           if np.all(n_Fy) != 0:
    #               self.Damage+=scp.integrate.simps(abs(n_Fy/Ninterp(Fyz_ult,Fy,-1/self.SN_b)),x=n_Fy,even = 'avg')
    #           if np.all(n_Fz) != 0:
    #               self.Damage+=scp.integrate.simps(abs(n_Fz/Ninterp(Fyz_ult,Fz,-1/self.SN_b)),x=n_Fz,even = 'avg')
    #       if Mx_ult !=0 and np.all(n_Mx) != 0:
    #           self.Damage+=scp.integrate.simps(abs(n_Mx/Ninterp(Mx_ult,Mx,-1/self.SN_b)),x=n_Mx,even = 'avg')
    #       if Myz_ult!=0:
    #           if np.all(n_My) != 0:
    #               self.Damage+=scp.integrate.simps(abs(n_My/Ninterp(Myz_ult,My,-1/self.SN_b)),x=n_My,even = 'avg')
    #           if np.all(n_Mz) != 0:
    #               self.Damage+=scp.integrate.simps(abs(n_Mz/Ninterp(Myz_ult,Mz,-1/self.SN_b)),x=n_Mz,even = 'avg')

    #       if self.Damage <= 1 or self.D_max >= diameter_limit:
    #           break
    #       else:
    #           self.D_max+=iterationstep
    #   #downwind bearing calcs
    #   while True:
    #       self.Damage = 0
    #       Fx_ult = self.SN_a*(pi/4.*(self.D_med**2-self.D_in**2))
    #       Mx_ult = self.SN_a*(pi*(self.D_med**4-self.D_in**4))/(32*(3)**.5*self.D_med)
    #       if Fx_ult !=0:
    #           self.Damage+=scp.integrate.simps(n_Fx/Ninterp(Fx_ult,Fx,-1/self.SN_b),x=n_Fx,even = 'avg')
    #       if Mx_ult !=0:
    #           self.Damage+=scp.integrate.simps(n_Mx/Ninterp(Mx_ult,Mx,-1/self.SN_b),x=n_Mx,even = 'avg')

    #       if self.Damage <= 1 or self.D_med>= diameter_limit:
    #           break
    #       else:
    #           self.D_med+=iterationstep

    #   #bearing calcs
    #   if self.availability != 0 and rotor_freq != 0 and self.T_life != 0 and self.cut_out != 0 and self.weibull_A != 0:
    #       N_rotations = self.availability*rotor_freq/60.*(self.T_life*365*24*60*60)*exp(-(self.cut_in/self.weibull_A)**self.weibull_k)-exp(-(self.cut_out/self.weibull_A)**self.weibull_k)
    #   elif np.max(n_Fx > 1e6):
    #       N_rotations = np.max(n_Fx)/self.blade_number
    #   elif np.max(n_My > 1e6):
    #       N_rotations = np.max(n_My)/self.blade_number
    #   Fz1_Fz = Fz*(self.L_mb+self.distance_hub2mb)/self.L_mb
    #   Fz1_My = My/self.L_mb
    #   Fy1_Fy = -Fy*(self.L_mb+self.distance_hub2mb)/self.L_mb
    #   Fy1_Mz = Mz/self.L_mb
    #   [self.D_max_a,facewidth_max,bearing1mass] = fatigue2_for_bearings(self.D_max,self.mb1Type,np.zeros(2),np.array([1,2]),Fy1_Fy,n_Fy/self.blade_number,Fz1_Fz,n_Fz/self.blade_number,Fz1_My,n_My/self.blade_number,Fy1_Mz,n_Mz/self.blade_number,N_rotations)
    #   Fz2_Fz = Fz*self.distance_hub2mb/self.L_mb
    #   Fz2_My = My/self.L_mb
    #   Fy2_Fy = Fy*self.distance_hub2mb/self.L_mb
    #   Fy2_Mz = Mz/self.L_mb
    #   [self.D_med_a,facewidth_med,bearing2mass] = fatigue2_for_bearings(self.D_med,self.mb2Type,Fx,n_Fx/self.blade_number,Fy2_Fy,n_Fy/self.blade_number,Fz2_Fz,n_Fz/self.blade_number,Fz2_My,n_My/self.blade_number,Fy2_Mz,n_Mz/self.blade_number,N_rotations)


# Code remove from LowSpeedShaft3pt component
############################################
# inputs into LSS for fatigue calculations:

check_fatigue = Enum(0,(0,1,2),iotype = 'in', desc = 'turns on and off fatigue check')

#fatigue1 variables
rotor_freq = Float(iotype = 'in', units = 'rpm', desc='rated rotor speed')
availability = Float(.95,iotype = 'in', desc = 'turbine availability')
fatigue_exponent = Float(0,iotype = 'in', desc = 'fatigue exponent of material')
S_ut = Float(700e6,iotype = 'in', units = 'Pa', desc = 'ultimate tensile strength of material')
weibull_A = Float(iotype = 'in', units = 'm/s', desc = 'weibull scale parameter "A" of 10-minute windspeed probability distribution')
weibull_k = Float(iotype = 'in', desc = 'weibull shape parameter "k" of 10-minute windspeed probability distribution')
blade_number = Float(iotype = 'in', desc = 'number of blades on rotor, 2 or 3')
cut_in = Float(iotype = 'in', units = 'm/s', desc = 'cut-in windspeed')
cut_out = Float(iotype = 'in', units = 'm/s', desc = 'cut-out windspeed')
Vrated = Float(iotype = 'in', units = 'm/s', desc = 'rated windspeed')
T_life = Float(iotype = 'in', units = 'yr', desc = 'cut-in windspeed')

# fatigue2 variables
rotor_thrust_distribution = Array(iotype='in', units ='N', desc = 'thrust distribution across turbine life')
rotor_thrust_count = Array(iotype='in', desc = 'corresponding cycle array for thrust distribution')
rotor_Fy_distribution = Array(iotype='in', units ='N', desc = 'Fy distribution across turbine life')
rotor_Fy_count = Array(iotype='in', desc = 'corresponding cycle array for Fy distribution')
rotor_Fz_distribution = Array(iotype='in', units ='N', desc = 'Fz distribution across turbine life')
rotor_Fz_count = Array(iotype='in', desc = 'corresponding cycle array for Fz distribution') 
rotor_torque_distribution = Array(iotype='in', units ='N*m', desc = 'torque distribution across turbine life')
rotor_torque_count = Array(iotype='in', desc = 'corresponding cycle array for torque distribution') 
rotor_My_distribution = Array(iotype='in', units ='N*m', desc = 'My distribution across turbine life')
rotor_My_count = Array(iotype='in', desc = 'corresponding cycle array for My distribution') 
rotor_Mz_distribution = Array(iotype='in', units ='N*m', desc = 'Mz distribution across turbine life')
rotor_Mz_count = Array(iotype='in', desc = 'corresponding cycle array for Mz distribution')

# fatigue check Taylor Parsons 6/2014
if self.check_fatigue == 1 or 2:
	  #start_time = time.time()
	  #material properties 34CrNiMo6 steel +QT, large diameter
	  self.E=2.1e11
	  self.density=7800.0
	  self.n_safety = 2.5
	  if self.S_ut <= 0:
	    self.S_ut=700.0e6 #Pa
	  Sm=0.9*self.S_ut #for bending situations, material strength at 10^3 cycles
	  C_size=0.6 #diameter larger than 10"
	  C_surf=4.51*(self.S_ut/1e6)**-.265 #machined surface 272*(self.S_ut/1e6)**-.995 #forged
	  C_temp=1 #normal operating temps
	  C_reliab=0.814 #99% reliability
	  C_envir=1. #enclosed environment
	  Se=C_size*C_surf*C_temp*C_reliab*C_envir*.5*self.S_ut #modified endurance limit for infinite life
	
	  if self.fatigue_exponent!=0:
	    if self.fatigue_exponent > 0:
	        self.SN_b = - self.fatigue_exponent
	    else:
	        self.SN_b = self.fatigue_exponent
	  else:
	    Nfinal = 5e8 #point where fatigue limit occurs under hypothetical S-N curve TODO adjust to fit actual data
	    z=log10(1e3)-log10(Nfinal)  #assuming no endurance limit (high strength steel)
	    self.SN_b=1/z*log10(Sm/Se)
	  self.SN_a=Sm/(1000.**self.SN_b)
	  # print 'm:', -1/self.SN_b
	  # print 'a:', self.SN_a
	
	  if self.check_fatigue == 1:
	      #checks to make sure all inputs are reasonable
	      if self.rotor_mass < 100:
	          [self.rotor_mass] = get_rotor_mass(self.machine_rating,False)
	
	      #Rotor Loads calculations using DS472
	      setup_Fatigue_Loads(self)
	
	      #upwind diameter calculations
	      iterationstep=0.001
	      diameter_limit = 1.5
	      while True:
	
	          get_Damage_Brng1(self)
	
	          if self.Damage < 1 or self.D_max >= diameter_limit:
	              break
	          else:
	              self.D_max+=iterationstep
	
	      #begin bearing calculations
	      N_bearings = self.N/self.blade_number #rotation number
	
	      Fz1stoch = (-self.My_stoch)/(self.L_ms)
	      Fy1stoch = self.Mz_stoch/self.L_ms
	      Fz1determ = (self.weightGbx*self.L_gb - self.LssWeight*.5*self.L_ms - self.rotorWeight*(self.L_ms+distance_hub2mb)) / (self.L_ms)
	
	      Fr_range = ((abs(Fz1stoch)+abs(Fz1determ))**2 +Fy1stoch**2)**.5 #radial stochastic + deterministic mean
	      Fa_range = self.Fx_stoch*cos(self.shaft_angle) + (self.rotorWeight+self.LssWeight)*sin(self.shaft_angle) #axial stochastic + mean
	
	      life_bearing = self.N_f/self.blade_number
	
	      [self.D_max_a,facewidth_max,bearingmass] = fatigue_for_bearings(self.D_max, Fr_range, Fa_range, N_bearings, life_bearing, self.mb1Type,False)
	
	  # elif self.check_fatigue == 2:
	  #   Fx = self.rotor_thrust_distribution
	  #   n_Fx = self.rotor_thrust_count
	  #   Fy = self.rotor_Fy_distribution
	  #   n_Fy = self.rotor_Fy_count
	  #   Fz = self.rotor_Fz_distribution
	  #   n_Fz = self.rotor_Fz_count
	  #   Mx = self.rotor_torque_distribution
	  #   n_Mx = self.rotor_torque_count
	  #   My = self.rotor_My_distribution
	  #   n_My = self.rotor_My_count
	  #   Mz = self.rotor_Mz_distribution
	  #   n_Mz = self.rotor_Mz_count
	
	  #   # print n_Fx
	  #   # print Fx*.5
	  #   # print Mx*.5
	  #   # print -1/self.SN_b
	
	  #   def Ninterp(L_ult,L_range,m):
	  #       return (L_ult/(.5*L_range))**m #TODO double-check that the input will be the load RANGE instead of load amplitudes. Also, may include means?
	
	  #   #upwind bearing calcs
	  #   diameter_limit = 5.0
	  #   iterationstep=0.001
	  #   #upwind bearing calcs
	  #   while True:
	  #       self.Damage = 0
	  #       Fx_ult = self.SN_a*(pi/4.*(self.D_max**2-self.D_in**2))
	  #       Fyz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*32*self.distance_hub2mb)
	  #       Mx_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(32*(3.**.5)*self.D_max)
	  #       Myz_ult = self.SN_a*(pi*(self.D_max**4-self.D_in**4))/(self.D_max*64.)
	  #       if Fx_ult and np.all(n_Fx):
	  #           self.Damage+=scp.integrate.simps(n_Fx/Ninterp(Fx_ult,Fx,-1/self.SN_b),x=n_Fx,even = 'avg')
	  #       if Fyz_ult:
	  #           if np.all(n_Fy):
	  #               self.Damage+=scp.integrate.simps(abs(n_Fy/Ninterp(Fyz_ult,Fy,-1/self.SN_b)),x=n_Fy,even = 'avg')
	  #           if np.all(n_Fz):
	  #               self.Damage+=scp.integrate.simps(abs(n_Fz/Ninterp(Fyz_ult,Fz,-1/self.SN_b)),x=n_Fz,even = 'avg')
	  #       if Mx_ult and np.all(n_Mx):
	  #           self.Damage+=scp.integrate.simps(abs(n_Mx/Ninterp(Mx_ult,Mx,-1/self.SN_b)),x=n_Mx,even = 'avg')
	  #       if Myz_ult:
	  #           if np.all(n_My):
	  #               self.Damage+=scp.integrate.simps(abs(n_My/Ninterp(Myz_ult,My,-1/self.SN_b)),x=n_My,even = 'avg')
	  #           if np.all(n_Mz):
	  #               self.Damage+=scp.integrate.simps(abs(n_Mz/Ninterp(Myz_ult,Mz,-1/self.SN_b)),x=n_Mz,even = 'avg')
	
	  #       print 'Upwind Bearing Diameter:', self.D_max
	  #       print 'self.Damage:', self.Damage
	
	  #       if self.Damage <= 1 or self.D_max >= diameter_limit:
	  #           # print 'Upwind Bearing Diameter:', self.D_max
	  #           # print 'self.Damage:', self.Damage
	  #           #print (time.time() - start_time), 'seconds of total simulation time'
	  #           break
	  #       else:
	  #           self.D_max+=iterationstep
	
	  #   #bearing calcs
	  #   if self.availability != 0 and rotor_freq != 0 and self.T_life != 0 and self.cut_out != 0 and self.weibull_A != 0:
	  #       N_rotations = self.availability*rotor_freq/60.*(self.T_life*365*24*60*60)*exp(-(self.cut_in/self.weibull_A)**self.weibull_k)-exp(-(self.cut_out/self.weibull_A)**self.weibull_k)
	  #   elif np.max(n_Fx > 1e6):
	  #       N_rotations = np.max(n_Fx)/self.blade_number
	  #   elif np.max(n_My > 1e6):
	  #       N_rotations = np.max(n_My)/self.blade_number
	
	  #   # Fz1 = (Fz*(self.L_ms+self.distance_hub2mb)+My)/self.L_ms
	  #   Fz1_Fz = Fz*(self.L_ms+self.distance_hub2mb)/self.L_ms #force in z direction due to Fz
	  #   Fz1_My = My/self.L_ms #force in z direction due to My
	  #   Fy1_Fy = -Fy*(self.L_ms+self.distance_hub2mb)/self.L_ms
	  #   Fy1_Mz = Mz/self.L_ms
	  #   [self.D_max_a,facewidth_max,bearingmass] = fatigue2_for_bearings(self.D_max,self.mb1Type,np.zeros(2),np.array([1,2]),Fy1_Fy,n_Fy/self.blade_number,Fz1_Fz,n_Fz/self.blade_number,Fz1_My,n_My/self.blade_number,Fy1_Mz,n_Mz/self.blade_number,N_rotations)
 
'''