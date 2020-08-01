#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import openmdao.api as om
from scipy.special import ellipeinc
import wisdem.commonse.tube as tube
import wisdem.commonse.utilities as util

def rod_prop(s, D, t, rho):
    y  = 0.25 * rho * np.pi * (D**2  - (D - 2*t)**2)
    L  = s.max() - s.min()
    m  = np.trapz(y, s)
    cm = np.trapz(y*s, s) / m
    Dm = D.mean()
    tm = t.mean()
    I  = np.array([0.5     *   0.25*(Dm**2  + (Dm - 2*tm)**2),
                  (1./12.)*(3*0.25*(Dm**2  + (Dm - 2*tm)**2) + L**2),
                  (1./12.)*(3*0.25*(Dm**2  + (Dm - 2*tm)**2) + L**2) ])
    return m, cm, m*I


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
        self.add_input('L_bedplate', 0.0, units='m', desc='Length of bedplate') 
        self.add_input('H_bedplate', 0.0, units='m', desc='height of bedplate')
        self.add_input('tilt', 0.0, units='deg', desc='Shaft tilt') 
        self.add_input('access_diameter',0.0, units='m', desc='Minimum diameter required for maintenance access') 

        self.add_input('lss_diameter', np.zeros(5), units='m', desc='Shaft outer diameter from hub to bearing 2')
        self.add_input('nose_diameter', np.zeros(5), units='m', desc='Nose outer diameter from bearing 1 to bedplate')
        self.add_input('D_top', 0.0, units='m', desc='Tower top outer diameter')
        self.add_input('lss_wall_thickness', np.zeros(5), units='m', desc='Shaft wall thickness')
        self.add_input('nose_wall_thickness', np.zeros(5), units='m', desc='Nose wall thickness')
        self.add_input('bedplate_wall_thickness', np.zeros(n_points), units='m', desc='Bedplate wall thickness')

        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        
        self.add_output('overhang',0.0, units='m', desc='Overhang of rotor from tower along x-axis in yaw-aligned c.s.') 
        self.add_output('drive_height',0.0, units='m', desc='Hub height above tower top')
        self.add_output('L_nose',0.0, units='m', desc='Length of nose') 
        self.add_output('L_shaft',0.0, units='m', desc='Length of nose') 
        self.add_output('L_generator',0.0, units='m', desc='Generator stack width') 
        self.add_output('L_drive',0.0, units='m', desc='Length of drivetrain from bedplate to hub flang') 
        self.add_output('D_bearing1',0.0, units='m', desc='Diameter of bearing #1 (closer to hub)') 
        self.add_output('D_bearing2',0.0, units='m', desc='Diameter of bearing #2 (closer to tower)') 

        self.add_output('constr_access', np.zeros(5), units='m', desc='Margin for allowing maintenance access (should be > 0)') 
        self.add_output('constr_L_grs', 0.0, units='m', desc='Margin for generator rotor attachment distance (should be > 0)') 
        self.add_output('constr_L_gsn', 0.0, units='m', desc='Margin for generator stator attachment distance (should be > 0)') 

        self.add_output('s_nose', val=np.zeros(6), units='m', desc='Nose discretized hub-aligned s-coordinates')
        self.add_output('D_nose', val=np.zeros(6), units='m', desc='Nose discretized diameter values at coordinates')
        self.add_output('t_nose', val=np.zeros(6), units='m', desc='Nose discretized thickness values at coordinates')
        self.add_output('nose_mass', val=0.0, units='kg', desc='Nose mass')
        self.add_output('nose_cm', val=0.0, units='m', desc='Nose center of mass along nose axis from bedplate')
        self.add_output('nose_I', val=np.zeros(3), units='kg*m**2', desc='Nose moment of inertia around cm in axial (hub-aligned) c.s.')

        self.add_output('s_shaft', val=np.zeros(6), units='m', desc='Shaft discretized s-coordinates')
        self.add_output('D_shaft', val=np.zeros(6), units='m', desc='Shaft discretized diameter values at coordinates')
        self.add_output('t_shaft', val=np.zeros(6), units='m', desc='Shaft discretized thickness values at coordinates')
        self.add_output('lss_mass', val=0.0, units='kg', desc='LSS mass')
        self.add_output('lss_cm', val=0.0, units='m', desc='LSS center of mass along shaft axis from bedplate')
        self.add_output('lss_I', val=np.zeros(3), units='kg*m**2', desc='LSS moment of inertia around cm in axial (hub-aligned) c.s.')

        self.add_output('x_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline x-coordinates')
        self.add_output('z_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline z-coordinates')
        self.add_output('x_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve x-coordinates')
        self.add_output('z_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve z-coordinates')
        self.add_output('x_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve x-coordinates')
        self.add_output('z_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve z-coordinates')
        self.add_output('D_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate diameters')
        self.add_output('t_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate wall thickness (mirrors input)')
        self.add_output('bedplate_mass', val=0.0, units='kg', desc='Bedplate mass')
        self.add_output('bedplate_cm', val=np.zeros(3), units='m', desc='Bedplate center of mass')
        self.add_output('bedplate_I', val=np.zeros(3), units='kg*m**2', desc='Bedplate mass moment of inertia about base')

        self.add_output('s_mb1', val=0.0, units='m', desc='Bearing 1 s-coordinate along drivetrain, measured from bedplate')
        self.add_output('s_mb2', val=0.0, units='m', desc='Bearing 2 s-coordinate along drivetrain, measured from bedplate')
        self.add_output('s_stator', val=0.0, units='m', desc='Generator stator attachment to nose s-coordinate')
        self.add_output('s_rotor', val=0.0, units='m', desc='Generator rotor attachment to shaft s-coordinate')
        self.add_output('generator_cm', val=0.0, units='m', desc='Overall generator cm')
        
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        L_12       = float(inputs['L_12'])
        L_h1       = float(inputs['L_h1'])
        L_2n       = float(inputs['L_2n'])
        L_grs      = float(inputs['L_grs'])
        L_gsn      = float(inputs['L_gsn'])
        L_bedplate = float(inputs['L_bedplate'])
        H_bedplate = float(inputs['H_bedplate'])
        tilt       = float(np.deg2rad(inputs['tilt']))
        D_access   = float(inputs['access_diameter'])
        D_nose     = inputs['nose_diameter']
        D_shaft    = inputs['lss_diameter']
        D_top      = float(inputs['D_top'])
        t_nose     = inputs['nose_wall_thickness']
        t_shaft    = inputs['lss_wall_thickness']
        t_bed      = inputs['bedplate_wall_thickness']
        upwind     = discrete_inputs['upwind']
        rho        = float(inputs['rho'])
        
        # ------------ Bedplate geometry and coordinates -------------
        # Define reference/centroidal axis
        # Origin currently set like standard ellipse eqns, but will shift below to being at tower top
        # The end point of 90 deg isn't exactly right for non-zero tilt, but leaving that for later
        n_points = self.options['n_points']
        if upwind:
            rad = np.linspace(0.0, 0.5*np.pi, n_points)
        else:
            rad = np.linspace(np.pi, 0.5*np.pi, n_points)

        # Centerline
        x_c = L_bedplate*np.cos(rad)
        z_c = H_bedplate*np.sin(rad)

        # Eccentricity of the centroidal ellipse
        ecc = np.sqrt(1 - (H_bedplate/L_bedplate)**2)
        
        # Points on the outermost ellipse
        x_outer = (L_bedplate + 0.5*D_top )*np.cos(rad)
        z_outer = (H_bedplate + 0.5*D_nose[0])*np.sin(rad)

        # Points on the innermost ellipse
        x_inner = (L_bedplate - 0.5*D_top )*np.cos(rad)
        z_inner = (H_bedplate - 0.5*D_nose[0])*np.sin(rad)

        # Cross-sectional properties
        D_bed   = np.sqrt( (z_outer-z_inner)**2 + (x_outer-x_inner)**2 )
        r_bed_o = 0.5*D_bed
        r_bed_i = r_bed_o - t_bed
        A_bed   = np.pi * (r_bed_o**2 - r_bed_i**2)

        # This finds the central angle (rad2) given the parametric angle (rad)
        rad2 = np.arctan( (L_bedplate + 0.5*D_top) / (H_bedplate + 0.5*D_nose[0]) * np.tan(np.diff(rad)) )
            
        # arc length
        arc = L_bedplate*np.abs(ellipeinc(rad2,ecc))    # Arc length via incomplete elliptic integral of the second kind

        # Mass and MoI properties
        x_c_sec = util.nodal2sectional( x_c )[0]
        z_c_sec = util.nodal2sectional( z_c )[0]
        R_c_sec = np.sqrt( x_c_sec**2 + z_c_sec**2 )
        mass    = util.nodal2sectional(A_bed)[0] * arc * rho
        mass_tot = mass.sum()
        cm      = np.array([np.sum(mass*x_c_sec), 0.0, np.sum(mass*z_c_sec)]) / mass_tot
        # For I, could do integral over sectional I, rotate axes by rad2, and then parallel axis theorem
        # we simplify by assuming lumped point mass.  TODO: Find a good way to check this?  Torus shell?
        I_bed  = util.assembleI( np.zeros(6) )
        for k in range(len(mass)):
            r_bed_o_k = 0.5*(r_bed_o[k] + r_bed_o[k+1])
            r_bed_i_k = 0.5*(r_bed_i[k] + r_bed_i[k+1])
            I_sec = mass[k]*np.array([0.5     *   (r_bed_o_k**2  + r_bed_i_k**2),
                                      (1./12.)*(3*(r_bed_o_k**2  + r_bed_i_k**2) + arc[k]**2),
                                      (1./12.)*(3*(r_bed_o_k**2  + r_bed_i_k**2) + arc[k]**2) ])
            I_sec_rot = util.rotateI(I_sec, 0.5*np.pi-rad2[k], axis='y')
            R_k       = np.array([x_c_sec[k]-x_c[0], 0.0, z_c_sec[k]])
            I_bed    += util.assembleI(I_sec_rot) + mass[k]*(np.dot(R_k, R_k)*np.eye(3) - np.outer(R_k, R_k))

        # Now shift originn to be at tower top
        cm[0]   -= x_c[0]
        x_inner -= x_c[0]
        x_outer -= x_c[0]
        x_c     -= x_c[0]
        
        outputs['bedplate_mass'] = mass_tot
        outputs['bedplate_cm']   = cm
        outputs['bedplate_I']    = I_bed
        
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
        
        # ------- Discretization ----------------
        # Length of shaft and nose
        L_shaft = L_12 + L_h1
        L_nose  = L_12 + L_2n
        outputs['L_shaft'] = L_shaft
        outputs['L_nose']  = L_nose

        # Total length from bedplate to hub flange
        ds      = 0.5*np.ones(2)
        s_drive = np.r_[0.0, L_2n*ds, L_12*ds, L_h1*ds]
        L_drive = s_drive.sum()
        outputs['L_drive']  = L_drive

        # Overhang from center of tower measured in yaw-aligned c.s. (parallel to ground)
        # TODO: Add in distance from hub flange?
        outputs['overhang'] = L_drive*np.cos(tilt) + L_bedplate

        # Total height (to ensure correct hub height can be enforced)
        outputs['drive_height'] = L_drive*np.sin(tilt) + H_bedplate

        # Discretize the drivetrain from bedplate to hub
        s_drive  = np.r_[np.cumsum(s_drive), L_drive-L_grs, L_gsn]
        s_mb1    = s_drive[4]
        s_mb2    = s_drive[2]
        s_rotor  = s_drive[-2]
        s_stator = s_drive[-1]
        s_nose   = np.r_[s_drive[:5], s_drive[-1]]
        s_shaft  = s_drive[2:-1]

        # Store outputs
        outputs['L_generator'] = L_drive - L_grs - L_gsn
        #outputs['s_drive']   = np.sort(s_drive)
        outputs['s_rotor']  = s_rotor
        outputs['s_stator'] = s_stator
        outputs['generator_cm'] = 0.5*(s_rotor + s_stator)
        outputs['s_mb1']    = s_mb1
        outputs['s_mb2']    = s_mb2
        # ------------------------------------
        
        # ------- Constraints ----------------
        outputs['constr_L_gsn'] = L_2n - L_gsn # Must be > 0
        outputs['constr_L_grs'] = L_shaft - L_grs # Must be > 0
        outputs['constr_access'] = D_nose - t_nose - D_access
        # ------------------------------------
        
        # ------- Nose, shaft, and bearing properties ----------------
        # Now is a good time to set bearing diameters
        outputs['D_bearing1'] = D_shaft[-1] - t_shaft[-1] - D_nose[0]
        outputs['D_bearing2'] = D_shaft[-1] - t_shaft[-1] - D_nose[-1]

        # Compute center of mass based on area
        m_nose, cm_nose, I_nose = rod_prop(s_nose[:-1], D_nose, t_nose, rho)
        outputs['nose_mass'] = m_nose
        outputs['nose_cm']   = cm_nose
        outputs['nose_I']    = I_nose

        m_shaft, cm_shaft, I_shaft = rod_prop(s_shaft[:-1], D_shaft, t_shaft, rho)
        outputs['lss_mass'] = m_shaft
        outputs['lss_cm']   = cm_shaft
        outputs['lss_I']    = I_shaft

        # Add in generator rotor and stator attachment points here because otherwise it is more difficult after points are sorted
        D_nose = np.r_[D_nose, np.interp(s_nose[-1], s_nose[:-1], D_nose)]
        t_nose = np.r_[t_nose, np.interp(t_nose[-1], s_nose[:-1], t_nose)]

        D_shaft = np.r_[D_shaft, np.interp(s_shaft[-1], s_shaft[:-1], D_shaft)]
        t_shaft = np.r_[t_shaft, np.interp(t_shaft[-1], s_shaft[:-1], t_shaft)]

        # Sort everything before storing it for easier use in Frame3DD
        ind = np.argsort(s_nose)
        outputs['s_nose'] = s_nose[ind]
        outputs['D_nose'] = D_nose[ind]
        outputs['t_nose'] = t_nose[ind]

        ind = np.argsort(s_shaft)
        outputs['s_shaft'] = s_shaft[ind]
        outputs['D_shaft'] = D_shaft[ind]
        outputs['t_shaft'] = t_shaft[ind]