#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import openmdao.api as om
from scipy.special import ellipeinc
import wisdem.commonse.tube as tube
from wisdem.commonse.utilities import nodal2sectional

class Geometry(om.Group):
    def initialize(self):
        self.options.declare('n_points')
    
    def setup(self):
        n_points = self.options['n_points']
        self.add_subsystem('lay',Layout(n_points=n_points), promotes=['*'])
        self.add_subsystem('mass',Mass(n_points=n_points), promotes=['*'])

                           
class Mass(om.ExplicitComponent):
    pass


class Layout(om.ExplicitComponent):
    """Calculate lengths, heights, and diameters of key drivetrain components in a direct drive system (valid for upwind or downwind)."""
    
    def initialize(self):
        self.options.declare('n_points')
    
    def setup(self):
        n_points = self.options['n_points']

        self.add_discrete_input('upwind', True, desc='Flag whether the design is upwind or downwind') 
        
        self.add_input('L_12', 0.0, units='m', desc='Length from bearing #1 to bearing #2')
        self.add_input('L_h1', 0.0, units='m', desc='Length from hub / start of shaft to bearing #1')
        self.add_input('L_2n', 0.0, units='m', desc='Length from bedplate / end of nose to bearing #2')
        self.add_input('L_grs', 0.0, units='m', desc='Length from shaft-hub flange to generator rotor attachment point on shaft')
        self.add_input('L_gsn', 0.0, units='m', desc='Length from nose-bedplate flange to generator stator attachment point on nose')
        self.add_input('L_hub', 0.0, units='m', desc='Length of hub') 
        self.add_input('L_bedplate', 0.0, units='m', desc='Length of bedplate') 
        self.add_input('H_bedplate', 0.0, units='m', desc='height of bedplate')
        self.add_input('tilt', 0.0, units='deg', desc='Shaft tilt') 
        self.add_input('access_diameter',0.0, units='m', desc='Minimum diameter required for maintenance access') 

        self.add_input('shaft_diameter', np.zeros(n_points), units='m', desc='Shaft outer diameter from hub to bearing 2')
        self.add_input('nose_diameter', np.zeros(n_points), units='m', desc='Nose outer diameter from bearing 1 to bedplate')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')
        self.add_input('shaft_wall_thickness', np.zeros(n_points), units='m', desc='Shaft wall thickness')
        self.add_input('nose_wall_thickness', np.zeros(n_points), units='m', desc='Nose wall thickness')
        self.add_input('bedplate_wall_thickness', np.zeros(n_points), units='m', desc='Shaft wall thickness')

        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        
        self.add_output('overhang',0.0, units='m', desc='Overhang of rotor from tower along x-axis in yaw-aligned c.s.') 
        self.add_output('drive_height',0.0, units='m', desc='Hub height above tower top')
        self.add_output('L_nose',0.0, units='m', desc='Length of nose') 
        self.add_output('L_shaft',0.0, units='m', desc='Length of nose') 
        self.add_output('D_bearing1',0.0, units='m', desc='Diameter of bearing #1 (closer to hub)') 
        self.add_output('D_bearing2',0.0, units='m', desc='Diameter of bearing #2 (closer to tower)') 
        self.add_output('constr_access', 0.0, units='m', desc='Margin for allowing maintenance access (should be > 0)') 

        self.add_output('x_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized x-coordinates')
        self.add_output('z_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized z-coordinates')
        self.add_output('D_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized diameter values at coordinates')
        self.add_output('t_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized thickness values at coordinates')

        self.add_output('x_shaft', val=np.zeros(n_points+2), units='m', desc='Shaft discretized x-coordinates')
        self.add_output('z_shaft', val=np.zeros(n_points+2), units='m', desc='Shaft discretized z-coordinates')
        self.add_output('D_shaft', val=np.zeros(n_points+2), units='m', desc='Shaft discretized diameter values at coordinates')
        self.add_output('t_shaft', val=np.zeros(n_points+2), units='m', desc='Shaft discretized thickness values at coordinates')

        self.add_output('x_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline x-coordinates')
        self.add_output('z_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline z-coordinates')
        self.add_output('x_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve x-coordinates')
        self.add_output('z_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve z-coordinates')
        self.add_output('x_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve x-coordinates')
        self.add_output('z_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve z-coordinates')
        self.add_output('D_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate diameters')
        self.add_output('t_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate wall thickness (mirrors input)')
        self.add_output('bedplate_mass', val=0.0, units='kg', desc='Bedplate mass')
        self.add_output('bedplate_center_of_mass', val=np.zeros(3), units='m', desc='Bedplate center of mass')

        self.add_output('x_mb1', val=0.0, units='m', desc='Bearing 1 x-coordinate')
        self.add_output('z_mb1', val=0.0, units='m', desc='Bearing 1 z-coordinate')
        self.add_output('x_mb2', val=0.0, units='m', desc='Bearing 2 x-coordinate')
        self.add_output('z_mb2', val=0.0, units='m', desc='Bearing 2 z-coordinate')
        self.add_output('x_stator', val=0.0, units='m', desc='Generator stator attachment to nose x-coordinate')
        self.add_output('z_stator', val=0.0, units='m', desc='Generator stator attachment to nose z-coordinate')
        self.add_output('x_rotor', val=0.0, units='m', desc='Generator rotor attachment to shaft x-coordinate')
        self.add_output('z_rotor', val=0.0, units='m', desc='Generator rotor attachment to shaft z-coordinate')
        self.add_output('x_hub', val=0.0, units='m', desc='Hub center (blade attachment plane) x-coordinate')
        self.add_output('z_hub', val=0.0, units='m', desc='Hub center (blade attachment plane) z-coordinate')
        
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        L_12       = float(inputs['L_12'])
        L_h1       = float(inputs['L_h1'])
        L_2n       = float(inputs['L_2n'])
        L_hub      = float(inputs['L_hub'])
        L_grs      = float(inputs['L_grs'])
        L_gsn      = float(inputs['L_gsn'])
        L_bedplate = float(inputs['L_bedplate'])
        H_bedplate = float(inputs['H_bedplate'])
        tilt       = float(np.deg2rad(inputs['tilt']))
        D_access   = float(inputs['access_diameter'])
        D_nose     = inputs['nose_diameter']
        D_shaft    = inputs['shaft_diameter']
        D_top      = float(inputs['D_top'])
        t_nose     = inputs['nose_wall_thickness']
        t_shaft    = inputs['shaft_wall_thickness']
        t_bed      = inputs['bedplate_wall_thickness']
        upwind     = discrete_inputs['upwind']
        rho        = float(inputs['rho'])
        
        # ------------ Bedplate geometry and coordinates -------------
        # Define reference/centroidal axis
        # Origin currently set like standard ellipse eqns, but will shift below to being at tower top
        # The end point of 90 deg isn't exactly right for non-zero tilt, but leaving that for later
        n_points = self.options['n_points']
        if upwind:
            rad = np.linspace(0.5*np.pi, 0, n_points)
        else:
            rad = np.linspace(0.5*np.pi, np.pi, n_points)

        # Centerline
        x_c = L_bedplate*np.cos(rad)
        z_c = H_bedplate*np.sin(rad)

        # Eccentricity of the centroidal ellipse
        ecc = np.sqrt(1 - (H_bedplate/L_bedplate)**2)
        
        # Points on the outermost ellipse
        x_outer = (L_bedplate + 0.5*D_top )*np.cos(rad)
        z_outer = (H_bedplate + 0.5*D_nose[-1])*np.sin(rad)

        # Points on the innermost ellipse
        x_inner = (L_bedplate - 0.5*D_top )*np.cos(rad)
        z_inner = (H_bedplate - 0.5*D_nose[-1])*np.sin(rad)

        # Cross-sectional properties
        D_bed   = np.sqrt( (z_outer-z_inner)**2 + (x_outer-x_inner)**2 )
        r_bed_o = 0.5*D_bed
        r_bed_i = r_bed_o - t_bed
        A_bed   = np.pi * (r_bed_o**2 - r_bed_i**2)

        # This finds the central angle (rad2) given the parametric angle (rad)
        rad2 = np.arctan( (L_bedplate + 0.5*D_top) / (H_bedplate + 0.5*D_nose[-1]) * np.tan(np.diff(rad)) )
            
        # arc length
        arc = L_bedplate*np.abs(ellipeinc(rad2,ecc))    # Arc length via incomplete elliptic integral of the second kind

        # Mass and MoI properties
        mass = nodal2sectional(A_bed)[0] * arc * rho
        outputs['bedplate_mass'] = mass.sum()
        x_c_sec = nodal2sectional( x_c )[0]
        z_c_sec = nodal2sectional( z_c )[0]
        outputs['bedplate_cm'] = np.array([np.sum(mass*x_c_sec), 0.0, np.sum(mass*z_c_sec)]) / mass.sum()

        # Now shift originn to be at tower top
        x_inner -= x_c[-1]
        x_outer -= x_c[-1]
        x_c     -= x_c[-1]
        
        # Geometry outputs
        outputs['x_bedplate'] = x_c
        outputs['z_bedplate'] = z_c
        outputs['x_bedplate_inner'] = x_inner
        outputs['z_bedplate_inner'] = z_inner
        outputs['x_bedplate_outer'] = x_outer
        outputs['z_bedplate_outer'] = z_outer
        outputs['D_bedplate'] = D_bed
        outputs['t_bedplate'] = t_bed
        # ------------------------------------

        
        '''
        s_disc = 0.5*np.ones(2)
        s_drive = np.array([0.0, L_2n*s_disc, L_12*s_disc, L_h1*s_disc, L_hub*s_disc])
        L_drive = s_drive.sum()

        s_drive = np.cumsum(s_drive)
        s_drive = np.r_[s_drive, L_drive-L_hub-L_grs, L_gsn] / L_drive
        
        x_drive = s_drive*np.cos(tilt)
        z_drive = s_drive*np.sin(tilt)
        if upwind:
            x_drive *= -1

        x_drive += x_c[0]
        z_drive += z_c[0]
        
        x_mb1 = x_drive[4]
        z_mb1 = z_drive[4]
        x_mb2 = x_drive[2]
        z_mb2 = z_drive[2]

        x_stator = x_drive[-1]
        z_stator = z_drive[-1]
        x_rotor  = x_drive[-2]
        z_rotor  = z_drive[-2]

        s_nose   = np.r_[s_drive[:5], s_drive[-1]]
        x_nose   = np.r_[x_drive[:5], x_drive[-1]]
        z_nose   = np.r_[z_drive[:5], z_drive[-1]]

        s_shaft  = np.r_[s_drive[2:-1], s_drive[-2]]
        x_shaft  = np.r_[x_drive[2:-1], x_drive[-2]]
        z_shaft  = np.r_[z_drive[2:-1], z_drive[-2]]
        '''

        
        # ------- Nose, shaft, and bearing coordinates ----------------
        # Length of shaft and nose
        L_shaft = L_12 + L_h1
        L_nose  = L_12 + L_2n
        outputs['L_shaft'] = L_shaft
        outputs['L_nose']  = L_nose

        # Set arc length and uncorrected x-z values
        npts    = len(D_nose)
        
        s_shaft = np.linspace(L_shaft, 0, npts)
        x_shaft = s_shaft*np.cos(tilt)
        z_shaft = s_shaft*np.sin(tilt)
        
        s_nose  = np.linspace(L_nose, 0, npts)
        x_nose  = s_nose*np.cos(tilt)
        z_nose  = s_nose*np.sin(tilt)
        
        if upwind:
            x_shaft *= -1.0
            x_nose  *= -1.0

        # Set starting position for nose
        x_nose  += x_c[0]
        z_nose  += z_c[0]

        # Set bearing positions
        x_mb1  = x_nose[0]
        z_mb1  = z_nose[0]
        x_mb2  = np.interp(L_nose - L_12, s_nose, x_nose, period=L_nose)
        z_mb2  = z_nose[0] if tilt == 0.0 else np.interp(L_nose - L_12, s_nose, z_nose, period=L_nose)
        outputs['x_mb1'] = x_mb1
        outputs['z_mb1'] = z_mb1
        outputs['x_mb2'] = x_mb2
        outputs['z_mb2'] = z_mb2
        
        # Set starting position for shaft
        x_shaft += x_mb2
        z_shaft += z_mb2

        # Set points for generator rotor and stator attachment
        x_rotor  = np.interp(L_shaft - L_grs, s_shaft, x_shaft, period=L_shaft)
        z_rotor  = z_shaft[0] if tilt == 0.0 else np.interp(L_shaft - L_grs, s_shaft, z_shaft, period=L_shaft)
        x_stator = np.interp(L_gsn, s_nose, x_nose, period=L_nose)
        z_stator = z_nose[0] if tilt == 0.0 else np.interp(L_gsn, s_nose, z_nose, period=L_nose)
        outputs['x_rotor']  = x_rotor
        outputs['z_rotor']  = z_rotor
        outputs['x_stator'] = x_stator
        outputs['z_stator'] = z_stator

        # Insert point for stator attachment point and bearings in nose and shaft arrays (for applied forces points)
        # For nose, mb1 point is the same as end point
        snew   = np.array( [L_gsn, L_nose-L_12] )
        D_nose = np.r_[D_nose, np.interp(snew, s_nose, D_nose, period=L_nose)]
        t_nose = np.r_[t_nose, np.interp(snew, s_nose, t_nose, period=L_nose)]
        x_nose = np.r_[x_nose, x_stator, x_mb2]
        z_nose = np.r_[z_nose, z_stator, z_mb2]
        s_nose = np.r_[s_nose, snew]

        # For shaft, mb2 point is the same as end point
        snew   = np.array( [L_shaft-L_grs, L_shaft-L_h1] )
        D_shaft = np.r_[D_shaft, np.interp(snew, s_shaft, D_shaft, period=L_shaft)]
        t_shaft = np.r_[t_shaft, np.interp(snew, s_shaft, t_shaft, period=L_shaft)]
        x_shaft = np.r_[x_shaft, x_stator, x_mb1]
        z_shaft = np.r_[z_shaft, z_stator, z_mb1]
        s_shaft = np.r_[s_shaft, snew]

        # Now is a good time to set bearing diameters
        outputs['D_bearing1'] = D_shaft[-1] - t_shaft[-1] - D_nose[0]
        outputs['D_bearing2'] = D_shaft[-1] - t_shaft[-1] - D_nose[-1]

        # For upwind, sort from lowest x-value to highest, downwind vice-versa
        isort = np.argsort(x_nose)
        if not upwind: isort = np.flipud(isort)
        x_nose = x_nose[isort]
        z_nose = z_nose[isort]
        D_nose = D_nose[isort]
        t_nose = t_nose[isort]
        s_nose = s_nose[isort]
        outputs['x_nose'] = x_nose
        outputs['z_nose'] = z_nose
        outputs['D_nose'] = D_nose
        outputs['t_nose'] = t_nose
        
        # For upwind, sort from lowest x-value to highest, downwind vice-versa
        isort = np.argsort(x_shaft)
        if not upwind: isort = np.flipud(isort)
        x_shaft = x_shaft[isort]
        z_shaft = z_shaft[isort]
        D_shaft = D_shaft[isort]
        t_shaft = t_shaft[isort]
        s_shaft = s_shaft[isort]
        outputs['x_shaft'] = x_shaft
        outputs['z_shaft'] = z_shaft
        outputs['D_shaft'] = D_shaft
        outputs['t_shaft'] = t_shaft
        # ------------------------------------
        

        # ------- Hub and summary coordinates ----------------
        x_hub = L_hub*np.cos(tilt)
        z_hub = L_hub*np.sin(tilt)
        if upwind: x_hub *= -1.0

        x_hub += x_shaft[0]
        z_hub += z_shaft[0]

        outputs['x_hub'] = x_hub
        outputs['z_hub'] = z_hub
        
        # Total length from bedplate to hub blade attachment
        L_drive = L_hub + L_h1 + L_12 + L_2n

        # Overhang from center of tower measured in yaw-aligned c.s. (parallel to ground)
        outputs['overhang'] = L_drive*np.cos(tilt) + L_bedplate

        # Total height (to ensure correct hub height can be enforced)
        outputs['drive_height'] = L_drive*np.sin(tilt) + H_bedplate

        # Ensure maintenance access space
        D_nose_i  = D_nose - t_nose
        outputs['constr_access'] = D_nose_i - D_access
        # ------------------------------------
