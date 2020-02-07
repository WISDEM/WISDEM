"""
hubse_components.py
Copyright (c) NREL. All rights reserved.

This is a modified version of hubse_components.py that models the hub and spinner as spherical (rather than
cylindrical) shapes. It is based on Excel spreadsheets by Scott Carron.
GNS 2019 06 17
"""

import numpy as np
from math import pi, cos, sqrt, sin, exp, radians
import sys
import warnings
warnings.simplefilter("error")

from wisdem.drivetrainse.drivese_utils import get_distance_hub2mb

# -------------------------------------------------

def inertiaSphereShell(mass, diameter, thickness, debug=False):
    ''' Return moment of inertia of a spherical shell '''
    radius = 0.5 * diameter
    insideRadius = radius - thickness
    try:
        dr5 = radius ** 5 - insideRadius ** 5
        dr3 = radius ** 3 - insideRadius ** 3
        I = 0.4 * mass \
               * (radius ** 5 - insideRadius ** 5) \
               / (radius ** 3 - insideRadius ** 3)
    except RuntimeWarning:
        sys.stderr.write('\n*** inertiaSphereShell: ERROR mass {:.1f} Rad {:.4f} IRad {:.4f} Thick {:.4f}\n\n'.format(mass, 
                         radius, insideRadius, thickness))
        I = 0
        
    if debug:
        sys.stderr.write('iSphShell: mass {:.1f} kg diam {:.1f} m thick {:.2f} m\n'.format(float(mass), float(diameter), float(thickness)))
        sys.stderr.write('iSphShell: I {:.2f} kg-m2\n'.format(float(I)))
    return np.array([I, I, I])
    
# -------------------------------------------------

class Hub_System_Adder(object):
    ''' 
    Compute hub mass, cm, and I
    '''

    def __init__(self, blade_number, debug=False):

        super(Hub_System_Adder, self).__init__()
        self.mass_adder = Hub_Mass_Adder(blade_number, debug=debug)
        self.cm_adder   = Hub_CM_Adder()
        
        self.debug = debug

    def compute(self, rotor_diameter, blade_mass, distance_hub2mb, shaft_angle, MB1_location, hub_mass, hub_diameter, hub_thickness, pitch_system_mass, spinner_mass):

        (self.rotor_mass, self.hub_system_mass, self.hub_system_I, self.hub_I) = self.mass_adder.compute(blade_mass, hub_mass, hub_diameter,
                                                                                             hub_thickness, pitch_system_mass, spinner_mass)
        self.hub_system_cm = self.cm_adder.compute(rotor_diameter, distance_hub2mb, shaft_angle, MB1_location)

        return(self.rotor_mass, self.hub_system_mass, self.hub_system_cm, self.hub_system_I, self.hub_I)

# -------------------------------------------------

class Hub_Mass_Adder(object):
    ''' 
    Compute hub mass and I
    Excluding cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''

    def __init__(self, blade_number, debug=False):

        super(Hub_Mass_Adder, self).__init__()
        self.blade_number = blade_number
        
        self.debug = debug

    def compute(self, blade_mass, hub_mass, hub_diameter, hub_thickness, pitch_system_mass, spinner_mass):

        # variables
        self.blade_mass = blade_mass.flatten() #Float(iotype='in', units='kg',desc='mass of blade')
        self.hub_mass = hub_mass.flatten() #Float(iotype='in', units='kg',desc='mass of Hub')
        self.hub_diameter = hub_diameter.flatten() #Float(3.0,iotype='in', units='m', desc='hub diameter')
        self.hub_thickness = hub_thickness.flatten() #Float(iotype='in', units='m', desc='hub thickness')
        self.pitch_system_mass = pitch_system_mass.flatten() #Float(iotype='in', units='kg',desc='mass of Pitch System')
        self.spinner_mass = spinner_mass.flatten() #Float(iotype='in', units='kg',desc='mass of spinner')
        
        # outputs
        self.hub_system_I = np.zeros(3) #Array(iotype='out', desc='mass moments of Inertia of hub [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass in yaw-aligned c.s.')
        self.hub_system_mass = 0. #Float(iotype='out', units='kg',desc='mass of hub system')
        self.rotor_mass = 0.

        self.hub_system_mass = self.hub_mass + self.pitch_system_mass + self.spinner_mass
        self.rotor_mass = self.hub_system_mass + self.blade_number*self.blade_mass

        hub_rad = 0.5 * self.hub_diameter
        cav_rad = hub_rad - self.hub_thickness
        t5 = (hub_rad**5 - cav_rad**5)
        t3 = (hub_rad**3 - cav_rad**3)
        if self.debug:
            sys.stderr.write('SphHMA::compute(): Thick {:.3f} M Diam {:.2f} m H {:.3f} C {:.3f} T5 {:.3f} T3 {:.3f}\n'.format(
                float(hub_thickness), float(hub_diameter), float(hub_rad), float(cav_rad), float(t5), float(t3)))
            
        hub_I = inertiaSphereShell(self.hub_mass, self.hub_diameter, self.hub_thickness, debug=self.debug)
        '''
        hub_I[0] = 0.4 * self.hub_mass \
               * ((self.hub_diameter / 2) ** 5 - (self.hub_diameter / 2 - self.hub_thickness) ** 5) \
               / ((self.hub_diameter / 2) ** 3 - (self.hub_diameter / 2 - self.hub_thickness) ** 3)
        hub_I[1] = hub_I[0]
        hub_I[2] = hub_I[1]
        '''
        
        pitch_system_I = np.zeros(3)
        pitch_system_I[0] = self.pitch_system_mass * (self.hub_diameter ** 2) / 4
        pitch_system_I[1] = pitch_system_I[0]
        pitch_system_I[2] = pitch_system_I[1]

        if self.hub_diameter == 0:
            spinner_diameter = 3.30
        else:
            spinner_diameter = self.hub_diameter
        spinner_thickness = spinner_diameter * (0.055 / 3.30)         # 0.055 for 1.5 MW outer diameter of 3.3 - using proportional constant

        spinner_I = inertiaSphereShell(self.spinner_mass, spinner_diameter, spinner_thickness, debug=self.debug)
        '''
        spinner_I = np.zeros(3)
        spinner_I[0] = 0.4 * self.spinner_mass \
             * ((spinner_diameter / 2) ** 5 - (spinner_diameter / 2 - spinner_thickness) ** 5) \
             / ((spinner_diameter / 2) ** 3 - (spinner_diameter / 2 - spinner_thickness) ** 3)
        spinner_I[1] = spinner_I[0]
        spinner_I[2] = spinner_I[1]
        '''
        
        #add moments of inertia
        #I = np.zeros(3)
        #for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
            # calculate moments around CM
            # sum moments around each components CM
            #I[i]  =  hub_I[i] + pitch_system_I[i] + spinner_I[i]
        self.hub_system_I = np.r_[hub_I.flatten() + pitch_system_I + spinner_I.flatten(), np.zeros(3)]
        
        if self.debug:
            sys.stderr.write('SphHMA: hub_system_mass {:8.1f} kg\n'.format(float(self.hub_system_mass)))
            sys.stderr.write('               hub_mass {:8.1f} kg\n'.format(float(self.hub_mass)))
            sys.stderr.write('      pitch_system_mass {:8.1f} kg\n'.format(float(self.pitch_system_mass)))
            sys.stderr.write('           spinner_mass {:8.1f} kg\n'.format(float(self.spinner_mass)))
            sys.stderr.write('             blade_mass {:8.1f} kg = {} * {:.1f} kg\n'.format(float(self.blade_number*self.blade_mass), 
                                                                                            int(self.blade_number), float(self.blade_mass)))
            sys.stderr.write('             rotor_mass {:8.1f} kg\n'.format(float(self.rotor_mass)))
            
            #for i in range(3):
            #    sys.stderr.write('Inertia {} H {:.2f} {:.2f} S {:.2f} {:.2f}\n'.format(i, hub_I[i], hI[i], spinner_I[i], sI[i]))

        return(self.rotor_mass, self.hub_system_mass, self.hub_system_I, hub_I.flatten())

# -------------------------------------------------


class Hub_CM_Adder(object):
    ''' 
    Compute hub cm
    Separating cm here, because it has a dependency on main bearing location, which can only be calculated once the full rotor mass is set
    '''

    def __init__(self):

        super(Hub_CM_Adder, self).__init__()

    def compute(self, rotor_diameter, distance_hub2mb, shaft_angle, MB1_location):

        # variables
        self.rotor_diameter = rotor_diameter #Float(iotype='in', units='m', desc='rotor diameter')
        self.distance_hub2mb = distance_hub2mb #Float(0.0,iotype='in', units = 'm', desc = 'distance between hub center and upwind main bearing')
        self.shaft_angle = shaft_angle #Float(iotype = 'in', units = 'deg', desc = 'shaft angle')
        self.MB1_location = MB1_location #Array(iotype = 'in', units = 'm', desc = 'center of mass of main bearing in [x,y,z] for an arbitrary coordinate system')
        
        # outputs
        self.hub_system_cm = np.zeros(3) #Array(iotype='out', units='m',desc='center of mass of the hub relative to tower to in yaw-aligned c.s.')
        
        if self.distance_hub2mb > 0:
            distance_hub2mb = self.distance_hub2mb
        else:
            distance_hub2mb = get_distance_hub2mb(self.rotor_diameter)

        cm = np.zeros(3)
        cm[0]     = self.MB1_location[0] - distance_hub2mb
        cm[1]     = 0.0
        cm[2]     = self.MB1_location[2] + distance_hub2mb*sin(self.shaft_angle)
        self.hub_system_cm = (cm)

        return(self.hub_system_cm)

#%% -------------------------------------------------

class Hub(object):
    ''' Sph_Hub class    
          The Sph_Hub class is used to represent the hub component of a wind turbine. 
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component. 

        2019 04 24 - GNS
          Conversion from kW to MW actually coverted to W - proper factor of 1e-3 is now used           
    '''

    def __init__(self, blade_number, debug=False):

        super(Hub, self).__init__()
        
        self.blade_number = blade_number
        self.debug = debug
        self.main_flange_thick = None
    
    def compute(self, blade_root_diameter, rotor_rpm, blade_mass, rotor_diameter, blade_length):  
        
        if self.blade_number != 3:
            sys.stderr.write('\n***ERROR: spherical_hub only works with 3-bladed hubs\n\n')
            return None, None, None
        
        if self.debug:
            sys.stderr.write('Hub: INPUTS BRD {:.1f} m RPM {:.1f} BMass {:.1f} RDiam {:.1f} m BLen {:.1f} m\n'.format(float(blade_root_diameter), \
                                                                                                                      float(rotor_rpm), float(blade_mass),
                                                                                                                      float(rotor_diameter), float(blade_length)) )
            
        # Parameters / 'constants'
        
        HUB_CIRC_INCR_PCT = 20.     #  %    Initial Spherical Hub  Circumference Increase Factor (Percentage)  
        ROTOR_SHUTDOWN_TIME = 1.    #  sec  Rotor Shutdown Time  
        FINAL_ROTOR_RADPS = 0.      # rad/s Final Rotor Speed  
        YIELD_STRENGTH_CAST = 200.  # Mpa   Yield Strength of Hub Casting  
        RESERVE_FACTOR = 2.         #       Reserve Factor
        STRESS_CON_FACTOR = 2.5     #       Stress Concentration Factor
        HUB_DENS = 7200.            # kg/m3    Density of Hub  
        FLANGE_THICK_FACTOR = 4.    #       Ratio of flange thickness to shell thickness
        
        ''' Can we use HUB_DENS for densities of spherical cap and main flange too? '''
        
        rotor_radps = rotor_rpm * 2. * pi / 60. # rad/s   Power Production rad/s 
        ang_accel = (rotor_radps-FINAL_ROTOR_RADPS) / (ROTOR_SHUTDOWN_TIME-0)  # rad/s2 Angular Acceleration  
    
        #   Hub Design Allowable and Material Properties      
        stress_allow = YIELD_STRENGTH_CAST / (STRESS_CON_FACTOR * RESERVE_FACTOR)  # Mpa (N/mm2)   Stress Allowable   
        stress_allow_pa = stress_allow * 1000000. #  N/m2      
             
        #    Hub Geometry      
        init_hub_diam = blade_root_diameter / (sin(radians(120./2.))) # m   Initial Spherical Hub Diameter determined by a Circle enclosing an Equilateral Triangle with Length equal to Blade Diameter
        init_hub_circ = pi * init_hub_diam                     # m   Initial Spherical Hub Circumference of Cross-Section 
        dsgn_hub_circ = init_hub_circ * (1.+(HUB_CIRC_INCR_PCT/100.)) # m   Design Spherical Hub  Circumference  
        dsgn_hub_diam = dsgn_hub_circ / pi                     # m   Design Spherical Hub Diameter (OD)  
            
        #   Hub Design Load      
        blade_cm        = ((rotor_diameter/2.) - blade_length) + (blade_length/3.) #  m   Blade Center of Mass (from Hub Rotational Axis)
        blade_mmi_edge  = blade_mass * blade_cm**2. # kgm2  Mass Moment of Inertia (mr2) - Edgewise  
        blade_torque    = blade_mmi_edge * ang_accel #  Nm   Torque from Blade  
        hub_torque      = blade_torque * self.blade_number #  Nm   Torque on Hub (Total)
           
        #   Hub Mass Calculations      
        sph_hub_shell_thick = ((((dsgn_hub_diam**4.) - ((32./ pi)*(hub_torque*dsgn_hub_diam/2./stress_allow_pa)))**(1./4.)) - dsgn_hub_diam) / (-2.) #  m   Spherical Hub Shell Thickness 
        sph_hub_shell_thick_mm = sph_hub_shell_thick * 1000. #  mm          
        sph_hub_vol = (4./3.) * pi * ((dsgn_hub_diam/2.)**3. - ((dsgn_hub_diam-2.*sph_hub_shell_thick)/2.)**3.) #  m3    Spherical Hub Volume 
        sph_hub_mass = sph_hub_vol * HUB_DENS # kg    Spherical Hub Mass 
        
        sph_cap_area  = 2. * pi * (dsgn_hub_diam/2.) \
                      * ((dsgn_hub_diam/2.) - sqrt((dsgn_hub_diam/2.)**2. - (blade_root_diameter/2.)**2.)) # m2   Spherical Cap Area (1 blade root cutout) 
        sph_cap_vol   = sph_cap_area * sph_hub_shell_thick # m3   Spherical Cap Volume (1 blade root cutout)
        sph_cap_vol_tot = self.blade_number * sph_cap_vol # m3   Spherical Cap Volume (3 blade root cutouts)
        sph_cap_mass  = sph_cap_vol_tot * HUB_DENS # kg   Spherical Cap Mass (3 blade root cutouts) 
            
        #main_flange_OD = 0.6 * dsgn_hub_diam # m  CALCULATED / ASSUMPTION IN HUBSE  Main Flange OD
        #main_flange_ID = main_flange_OD - (2*(dsgn_hub_diam/10)) # m  CALCULATED / ASSUMPTION IN HUBSE  Main Flange ID  
        #main_flange_thick = 5 * sph_hub_shell_thick #  m  CALCULATED / ASSUMPTION IN HUBSE  Main Flange Thickness 
        # Rev02 changes constant terms in 3 lines above - 2019 07 08
        main_flange_OD = 0.5 * dsgn_hub_diam # m  CALCULATED / ASSUMPTION IN HUBSE  Main Flange OD
        main_flange_ID = main_flange_OD - (2.*(dsgn_hub_diam/20.)) # m  CALCULATED / ASSUMPTION IN HUBSE  Main Flange ID  
        main_flange_thick = FLANGE_THICK_FACTOR * sph_hub_shell_thick #  m  CALCULATED / ASSUMPTION IN HUBSE  Main Flange Thickness 
        main_flange_vol = pi * main_flange_thick * ((main_flange_OD/2.)**2.- (main_flange_ID/2.)**2.) # m3    Main Flange Volume 
        main_flange_mass = main_flange_vol * HUB_DENS # kg    Mass of Main Flange
            
        hub_mass = main_flange_mass + sph_hub_mass # kg   Total Hub Mass  
            
        #   Hub Centroid Calculations      
        main_flange_cm = main_flange_thick / 2. # m   Center of Mass (Main Flange) 
        mmf = main_flange_mass # kg    Mass (Main Flange) 
        sphere_cm = dsgn_hub_diam / 2. # m   Center of Mass (Sphere) 
        msph  = sph_hub_mass # kg    Mass (Sphere) 
        if (mmf + msph) < 0.01:
        	sys.stderr.write('\n*** Hub::compute() ERROR:  mmf {:.2f} msph {:.2f}\n\n'.format(mmf, msph))
        	hub_cm = 0.0
        else:
            hub_cm  = (mmf*main_flange_cm + msph*sphere_cm) / (mmf + msph) # m    Hub Center of Mass 
           
        #   Hub Mass Calculations      
        cost_cast_iron = 3. # USD/kg   Casting House Costs for Cast Iron 
        hub_cost = hub_mass * cost_cast_iron # USD   Hub Cost
        
        # Save some values
        self.main_flange_thick = main_flange_thick
        
        if self.debug:
            sys.stderr.write('Sph_Hub: mass {:.1f} kg Diam {:.1f} m CM {:.2f} m COST ${:.2f} ShellThick {:.3f} FlangeThick {:.3f}\n'.format(float(hub_mass),
                                                                                                                                            float(dsgn_hub_diam), float(hub_cm),
                                                                                                                                            float(hub_cost),
                                                                                                                                            float(sph_hub_shell_thick),
                                                                                                                                            float(main_flange_thick)) )

        return hub_mass, dsgn_hub_diam, hub_cm, hub_cost, sph_hub_shell_thick

#-------------------------------------------------------------------------

class PitchSystem(object):
    '''
     PitchSystem class
      The PitchSystem class is used to represent the pitch system of a wind turbine.
      It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
      It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, blade_number, debug=False):

        super(PitchSystem, self).__init__()
        
        self.blade_number = blade_number
        
        self.debug = debug

    def compute(self, blade_mass, rotor_bending_moment_y):

        # variables
        self.blade_mass = blade_mass #Float(iotype='in', units='kg', desc='mass of one blade')
        self.rotor_bending_moment_y = rotor_bending_moment_y #Float(iotype='in', units='N*m', desc='flapwise bending moment at blade root')
    
        # parameters
        #blade_number = Int(3, iotype='in', desc='number of turbine blades')
    
        # outputs
        self.mass = 0. #Float(0.0, iotype='out', units='kg', desc='overall component mass')

        # -------- Sunderland method for calculating pitch system masses --------
        pitchmatldensity = 7860.                             # density of pitch system material (kg / m^3) - assuming BS1503-622 (same material as LSS)
        pitchmatlstress  = 371000000.                        # allowable stress of hub material (N / m^2)

        hubpitchFact     = 1.                                # default factor is 1.0 (0.54 for modern designs)
        self.mass = hubpitchFact * (0.22 * self.blade_mass * self.blade_number \
                                    + 12.6 * np.abs(self.rotor_bending_moment_y) * (pitchmatldensity / pitchmatlstress))
                                    #+ 12.6 * self.rotor_bending_moment_y * (pitchmatldensity / pitchmatlstress))
                             # mass of pitch system based on Sunderland model
                             # 2019 04 29 - mass is probably a function of abs(rotor_moment_y) - without abs, we can get negative masses
        # -------- End Sunderland method --------
                             
        if self.debug:
            sys.stderr.write('PitchSystem IN : blade mass {:.1f} kg rbmy {:.1f} Nm\n'.format(float(blade_mass), float(self.rotor_bending_moment_y)))
            sys.stderr.write('PitchSystem OUT: mass {:.1f} kg\n'.format(float(self.mass)))
       
        return(self.mass)

#-------------------------------------------------------------------------

class Spinner(object):
    '''
       Sph_Spinner class
          The Sph_SpinnerClass is used to represent the spinner of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, blade_number, debug=False):
        self.blade_number = blade_number
        self.debug = debug

    def computeOLD(self, blade_root_diameter):

        if self.blade_number != 3:
            sys.stderr.write('\n***ERROR: spherical_spinner only works with 3-bladed hubs\n\n')
            return None, None, None
        
        # Parameters / 'constants'
        HUB_CIRC_INCR_PCT = 20.   # %      C12  Initial Spherical Hub  Circumference Increase Factor (Percentage)  
        OSHA_CLEARANCE = 0.6     # m      C15  OSHA Clearance between Spinner and Hub  
        COMP_SHELL_THICK = 0.007 # m      C17  Spinner Composite Shell Thickness      
        SPIN_HOLE_INCR_PCT = 20. # %      C19  Spinner Blade Access Hole Increase Factor  
        COMP_DENS  = 2100.        # kg/m3  C28  Density of Composite (GFRP)  
        #STEEL_DENS = 7850        # kg/m3  C29  Density of Steel (S355)      
        RATIO_COMP_STEEL = 2.   #        C47  Composite to Steel Mass Ratio  
        COST_COMPOSITE = 7.    # USD/kg C56  Composite Spinner Shell Cost  
        COST_SMALL_STEEL = 3.  # USD/kg C57  Small Steel Part Hardware Costs          
        
        #     Spinner Geometry      
        init_hub_diam = blade_root_diameter / (sin(radians(120./2.))) # m C10  CALCHUB  Initial Spherical Hub Diameter determined by a Circle enclosing an Equilateral Triangle with Length equal to Blade Diameter  
        init_hub_circ = pi * init_hub_diam                          # m C11  CALCHUB  Initial Spherical Hub Circumference of Cross-Section  
        dsgn_hub_circ = init_hub_circ * (1.+(HUB_CIRC_INCR_PCT/100.)) # m C13  CALCHUB  Design Spherical Hub  Circumference      
        dsgn_hub_diam = dsgn_hub_circ / pi                          # m C14  CALCHUB  Design Spherical Hub Diameter (OD)      
        sph_spin_diam = dsgn_hub_diam + (2. * OSHA_CLEARANCE)        # m C16  CALCSPN  Spherical Spinner Diameter (OD)          
               
        spin_acc_hole_diam = blade_root_diameter * ((100.+SPIN_HOLE_INCR_PCT)/100.) # m C20  CALCSPN  Spinner Blade Access Hole Diameter          
            
                
        #    Spinner Design Load      
        #C24  BASSSPN  ULS Load Case (Driving Load Case)  Aero/OSHA  
                  
        #    Spinner Design Allowable and Material Properties      
        #C27  BASSSPN  Stress Allowable(s)  N/A  Mpa (N/mm2)
                
        #    Spinner Mass Calculations      
        comp_shell_vol = (4./3.)*pi*((sph_spin_diam/2.)**3.-((sph_spin_diam-2.*COMP_SHELL_THICK)/2.)**3.)  # m  C32  CALCSPN  Spherical Spinner Composite Shell Volume  3
        comp_shell_mass = comp_shell_vol * COMP_DENS                                                   # kg C33  CALCSPN  Spherical Spinner Composite Shell Mass      
                 
        sph_cap_area = 2. * pi * (sph_spin_diam/2.) \
                      * ((sph_spin_diam/2.) - sqrt((sph_spin_diam/2.)**2. - (spin_acc_hole_diam/2.)**2.)) #  m2 C35  CALCSPN  Spherical Cap Area (1 blade root cutout)  
        sph_cap_vol = sph_cap_area * COMP_SHELL_THICK # m3    C36  CALCSPN  Spherical Cap Volume (1 blade root cutout)  
        sph_cap_tot_vol = self.blade_number * sph_cap_vol             # m3    C37  CALCSPN  Spherical Cap Volume (3 blade root cutouts)  
        sph_cap_mass = sph_cap_tot_vol * COMP_DENS    # kg    C38  CALCSPN  Spherical Cap Mass (3 blade root cutouts)  
                
        main_flange_od = 0.6 * dsgn_hub_diam  #  m C40  CASSHUB  Main Flange OD                      
        main_flange_cap_area = 2. * pi * (sph_spin_diam/2.) \
                               * ((sph_spin_diam/2.) - sqrt((sph_spin_diam/2)**2 - (main_flange_od/2.)**2.)) #  m2 C41  CALCSPN  Main Flange Spherical Cap Area      
        main_flange_cap_vol = main_flange_cap_area * COMP_SHELL_THICK # m3 C42  CALCSPN  Main Flange Spherical Cap Volume  
        main_flange_cap_mass = main_flange_cap_vol * COMP_DENS # kg C43  CALCSPN  Main Flange Spherical Cap Mass      
                
        tot_composite_mass = comp_shell_mass - sph_cap_mass - main_flange_cap_mass #  kg   C45  CALCSPN  Total Composite Spinner Shell Mass  
                
        tot_steel_mass = tot_composite_mass / RATIO_COMP_STEEL # kg    C48  CALCSPN  Total Steel Mass              
                                                                  
        tot_spinner_mass = tot_composite_mass + tot_steel_mass # kg    C50  CALCSPN  Total Spinner Mass  

        #    Spinner Centroid Calculations      
        spin_cm = sph_spin_diam / 2.  # m C53  CALCSPN  Spinner Center of Mass (Spherical Shell and Front and Rear Steel Hardware)  
         
        #    Spinner Inertia Calculations
        spin_I = [0.0, 0.0, 0.0]
        
        #    Spinner Cost Calculations      
        spinner_cost = (COST_COMPOSITE * tot_composite_mass) + (COST_SMALL_STEEL * tot_steel_mass) # USD  C58  CALCSPN  Total Spinner Cost              
    
        if self.debug:
            sys.stderr.write('Sph_Spinner: mass {:.1f} kg = Steel {:.1f} kg + Composite {:.1f} kg\n'.format(tot_spinner_mass,
                             tot_steel_mass, tot_composite_mass))
            sys.stderr.write('Sph_Spinner: size IHD {:.1f} m DHD {:.1f} m SAHD {:.1f} m\n'.format(init_hub_diam,
                             dsgn_hub_diam, spin_acc_hole_diam))

        return tot_spinner_mass, spin_cm, spinner_cost

    def compute(self, blade_root_diameter):
        ''' This version of compute implements the REV02 rewrite of the spinner that Scott Caron delivered on 2019 07 07 '''
        if self.blade_number != 3:
            sys.stderr.write('\n***ERROR: spherical_spinner only works with 3-bladed hubs\n\n')
            return None, None, None
        
        # Parameters / 'constants'
        OSHA_CLEARANCE = 0.5             #  m            C17  Clearance between Spinner and Hub  
        SPIN_HOLE_INCR_PCT = 20.         #  %            C22  Spinner Blade Access Hole Increase Diameter Factor  
        N_FRONT_BRACKETS = 3.            #               C24  Number of Front Spinner Brackets    
        N_REAR_BRACKETS = 3.             #               C25  Number of Rear Spinner Brackets    
        EXTR_GUST = 70.                  #  m/s          C29  Extreme Gust Velocity  
        EXTR_GUST_LOAD_FACTOR = 1.5      #               C31  Extreme Gust Load Factor  
        COMP_TENSILE_STRENGTH = 60.      #  Mpa (N/mm2)  C41  Composite Shell Tensile Strength  
        COMP_RESERVE_FACTOR = 1.5        #               C42  Composite Shell Reserve Factor    
        COMP_DENSITY = 1600.             #  kg/m3        C44  Density of Composite Shell  
        S235_YIELD_STRENGTH = 235.       #  Mpa (N/mm2)  C46  S235 Yield Strength (Base)  
        S235_YIELD_STRENGTH_THICK = 225. #  Mpa (N/mm2)  C47  S235 Yield Strength (t>16mm)  
        S235_RESERVE_FACTOR = 1.5        #               C48  S235 Reserve Factor    
        S235_DENSITY = 7850.             #  kg/m3        C50  Density of Steel (S355)  
        HUB_CIRC_INCR_PCT = 20.          #  %            C13  Initial Spherical Hub  Circumference Increase Factor (Percentage)  
        SPIN_SHELL_COMP_COST = 7.      #  USD/kg       C92  Composite Spinner Shell Cost    
        SMALL_STEEL_COST = 3.          #  USD/kg       C93  Small Steel Part Hardware Costs 
        
        # set specific material properties
        
        steel_density = S235_DENSITY
        steel_yield_strength = S235_YIELD_STRENGTH
        steel_yield_strength_thick = S235_YIELD_STRENGTH_THICK
        steel_reserve_factor = S235_RESERVE_FACTOR
        
        init_hub_diam = blade_root_diameter / (sin(radians(120./2.))) # m                                                                                                                       C11  CALCULATED IN HUBSE  Initial Spherical Hub Diameter                                     
        init_hub_circ = pi * init_hub_diam # m                                                                                                                                                C12  CALCULATED IN HUBSE  Initial Spherical Hub Circumference of Cross-Section               
        dsgn_hub_circ = init_hub_circ * (1.+(HUB_CIRC_INCR_PCT/100.)) # m                                                                                                                       C14  CALCULATED IN HUBSE  Design Spherical Hub  Circumference                                 
        dsgn_hub_diam = dsgn_hub_circ / pi # m                                                                                                                                                C15  CALCULATED IN HUBSE  Design Spherical Hub Diameter (OD)                                 
                                                                                                                                                                                                                                                                                           
        sph_spin_diam = dsgn_hub_diam + (2.*OSHA_CLEARANCE) # m                                                                                                                                C18  CALC  Spherical Spinner Diameter (OD)                                                   
        sph_spin_rad = 0.5 * sph_spin_diam                                                                                                                                                                                                                                                 
        sph_spin_circ = pi * sph_spin_diam # m                                                                                                                                                C19  CALC  Spherical Spinner Circumference                                                    
        spin_panel_width = (sph_spin_circ - dsgn_hub_circ) / 3. # m                                                                                                                            C20  CALC  Spinner Panel Width between Blade Cutouts                                         
                                                                                                                                                                                                                                                                                           
        spin_acc_hole_diam = blade_root_diameter * ((100.+SPIN_HOLE_INCR_PCT)/100.) # m                                                                                                         C23  CALC  Spinner Blade Access Hole Diameter                                                 
                                                                                                                                                                                                                                                                                           
        extr_gust_pressure = 0.5 * 1.225 * (EXTR_GUST ** 2.) # N/m2                                                                                                                            C30  CALC  Extreme Gust Pressure                                                              
        extr_gust_dsgn_pressure = extr_gust_pressure * EXTR_GUST_LOAD_FACTOR # N/m2                                                                                                           C32  CALC  Extreme Gust Design Pressure                                                       
                                                                                                                                                                                                                                                                                           
        allow_tensile_strength = COMP_TENSILE_STRENGTH / COMP_RESERVE_FACTOR # Mpa (N/mm2)                                                                                                    C43  CALC  Composite Shell Design Allowable Tensile Strength                                  
        allow_yield_strength = steel_yield_strength_thick / steel_reserve_factor # Mpa (N/mm2)                                                                                                C49  CALC  S235 Design Allowable Yield Strength                                               
                                                                                                                                                                                                                                                                                           
        flat_plate_length = sph_spin_diam # m                                                                                                                                                 C54  CALC  Flat plate length (a)                                                             
        flat_plate_width = spin_panel_width # m                                                                                                                                               C55  CALC  Flat Plate width (b)                                                              
        spin_shell_thickness = sqrt((0.75 * extr_gust_dsgn_pressure * flat_plate_width ** 2.) / ((allow_tensile_strength*1000000.)*(1.61*(flat_plate_width/flat_plate_length) ** 3. + 1.))) # m   C56  CALC  Spinner shell Thickness                                                           
        spin_shell_volume = (4./3.)  *pi * (sph_spin_rad ** 3. - ((sph_spin_diam - 2.*spin_shell_thickness)/2.) ** 3.) # m3                                                                         C57  CALC  Spherical Spinner Composite Shell Volume                                           
        spin_shell_mass = spin_shell_volume * COMP_DENSITY # kg                                                                                                                               C58  CALC  Spherical Spinner Composite Shell Mass                                            
                                                                                                                                                                                                                                                                                           
        sph_cap_area = 2.  *pi * sph_spin_rad * (sph_spin_rad - sqrt(sph_spin_rad ** 2. - (spin_acc_hole_diam/2.) ** 2.)) # m2                                                                    C60  CALC  Spherical Cap Area (1 blade root cutout)                                          
        sph_cap_volume = sph_cap_area * spin_shell_thickness # m3                                                                                                                             C61  CALC  Spherical Cap Volume (1 blade root cutout)                                        
        sph_cap_volume = 3. * sph_cap_volume # m3                                                                                                                                              C62  CALC  Spherical Cap Volume (3 blade root cutouts)                                        
        sph_cap_mass = sph_cap_volume * COMP_DENSITY # kg                                                                                                                                     C63  CALC  Spherical Cap Mass (3 blade root cutouts)                                         
                                                                                                                                                                                                                                                                                           
        main_flange_diam = 0.6 * dsgn_hub_diam # m                                                                                                                                            C65  CALCULATED / ASSUMPTION IN HUBSE  Main Flange OD                                        
        main_flange_area = 2. * pi * sph_spin_rad * (sph_spin_rad - sqrt(sph_spin_rad ** 2. - (main_flange_diam/2.) ** 2.)) # m2                                                                  C66  CALC  Main Flange Spherical Cap Area                                                    
        main_flange_volume = main_flange_area * spin_shell_thickness # m3                                                                                                                     C67  CALC  Main Flange Spherical Cap Volume                                                  
        main_flange_mass = main_flange_volume * COMP_DENSITY # kg                                                                                                                             C68  CALC  Main Flange Spherical Cap Mass                                                     
        spin_shell_mass = spin_shell_mass - sph_cap_mass - main_flange_mass # kg                                                                                                                  C70  CALC  Total Composite Spinner Shell Mass                                                 
                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                              #C73  BASE  Description in Notes                                                              
        spin_frontal_area = pi * (sph_spin_diam ** 2.)/4. # m2                                                                                                                                  C74  CALC  Spinner Frontal Area                                                              
        frontal_gust_load = spin_frontal_area * extr_gust_dsgn_pressure # N                                                                                                                   C75  CALC  Extreme Gust Load on Frontal Area                                                 
        bracket_load = frontal_gust_load / (N_FRONT_BRACKETS + N_REAR_BRACKETS) # N                                                                                                           C76  CALC  Load on Single Bracket                                                            
        bracket_bending_moment = bracket_load * OSHA_CLEARANCE # Nm                                                                                                                           C77  CALC  Bending Moment on Bracket                                                         
        bracket_width = spin_panel_width / 2. # m                                                                                                                                              C78  CALC  Steel bracket width (b)                                                           
        bracket_length = OSHA_CLEARANCE # m                                                                                                                                                   C79  CALC  Steel Bracket Length                                                              
        bracket_thickness = sqrt((6 * bracket_bending_moment) / (bracket_width * allow_yield_strength * 1000000.)) # m                                                                         C80  CALC  Steel bracket thickness                                                           
        bracket_flange_length = bracket_length * 0.25 # m                                                                                                                                     C81  CALC  Steel Bracket attachment flange Length                                            
        bracket_volume = (bracket_length + bracket_flange_length + bracket_flange_length) * bracket_width * bracket_thickness # m3                                                            C82  CALC  Steel Bracket Volume                                                              
        bracket_mass = bracket_volume * steel_density # kg                                                                                                                                    C83  CALC  Steel Bracket Mass (Individual Bracket)                                            
        bracket_mass_total = bracket_mass * (N_FRONT_BRACKETS + N_REAR_BRACKETS) # kg                                                                                                         C84  CALC  Steel Bracket Mass (Total)                                                        
                                                                                                                                                                                                                                                                                           
        spinner_mass = spin_shell_mass + bracket_mass_total # kg                                                                                                                              C86  CALC  Total Spinner Mass (Composite Shell plus Steel Hardware)

        spinner_cm = sph_spin_diam / 2. # m                                                                                                                                                    C89  CALC  Spinner Center of Mass (Sph Shell and Front/Rear Steel Hardware)                   
        spinner_cost = (spin_shell_mass * SPIN_SHELL_COMP_COST) + (bracket_mass_total * SMALL_STEEL_COST) # USD                                                                               C94  CALC  Total Spinner Cost                                                                 

        if self.debug:
            sys.stderr.write('Sph_Spinner: mass {:.1f} kg = Shell {:.1f} kg + Bracket {:.1f} kg\n'.format(float(spinner_mass),
                                                                                                          float(spin_shell_mass),
                                                                                                          float(bracket_mass_total)))
            sys.stderr.write('Sph_Spinner: size IHD {:.1f} m DHD {:.1f} m SAHD {:.1f} m\n'.format(float(init_hub_diam),
                                                                                                  float(dsgn_hub_diam),
                                                                                                  float(spin_acc_hole_diam)))
            sys.stderr.write('Sph_Spinner: cost ${:.2f}  CM {:.2f} m\n'.format(float(spinner_cost), float(spinner_cm)))

        return spinner_mass, spinner_cm, spinner_cost

#%%---------------------------------
        
if __name__ == "__main__":

    # TODO: raw python hub component examples
    
    blade_root_diameter = 4.0
    blade_mass = 17000.
    rotor_bending_moment_y = 0
    rotor_rpm = 12.1
    rotor_diameter = 126.0
    blade_length = 61.0
    
    if False: # BAR params
        blade_root_diameter = 4.5
        blade_mass = 60800
        rotor_bending_moment_y = 0.
        rotor_rpm = 7.9
        rotor_diameter = 206.0
        blade_length = 100.0
    
    spin = Spinner(blade_number=3, debug=True)
    tot_spinner_mass, spin_cm, spinner_cost = spin.compute(blade_root_diameter)
    
    pitch = PitchSystem(blade_number=3, debug=True)
    ps_mass = pitch.compute(blade_mass, rotor_bending_moment_y)
    
    hub = Hub(blade_number=3, debug=True)
    hub_mass, dsgn_hub_diam, hub_cm, hub_cost, sph_hub_shell_thick = hub.compute(blade_root_diameter, 
                                                            rotor_rpm, blade_mass, rotor_diameter, blade_length)
    
    pass
