#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from scipy.special import ellipeinc
import openmdao.api as om

import wisdem.pyframe3dd.pyframe3dd as frame3dd
import wisdem.commonse.UtilizationSupplement as Util
import wisdem.commonse.tube as tube
from wisdem.commonse.utilities import nodal2sectional
from wisdem.commonse import gravity
from wisdem.commonse.csystem import DirectionVector


RIGID = 1
FREE  = 0

class Nose_Stator_Bedplate_Frame(om.ExplicitComponent):
    """Run structural analysis of nose/turret with the generator stator and bedplate"""
    
    def initialize(self):
        self.options.declare('n_points')
        self.options.declare('n_dlcs')
    
    def setup(self):
        n_points = self.options['n_points']
        n_dlcs   = self.options['n_dlcs']

        self.add_discrete_input('upwind', True, desc='Flag whether the design is upwind or downwind') 
        self.add_input('tilt', 0.0, units='deg', desc='Shaft tilt') 

        self.add_input('s_nose', val=np.zeros(6), units='m', desc='Discretized s-coordinates along drivetrain, measured from bedplate')
        self.add_input('D_nose', val=np.zeros(6), units='m', desc='Nose discretized diameter values at coordinates')
        self.add_input('t_nose', val=np.zeros(6), units='m', desc='Nose discretized thickness values at coordinates')

        self.add_input('x_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline x-coordinates')
        self.add_input('z_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline z-coordinates')
        self.add_input('x_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve x-coordinates')
        self.add_input('z_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve z-coordinates')
        self.add_input('x_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve x-coordinates')
        self.add_input('z_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve z-coordinates')
        self.add_input('D_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate diameters')
        self.add_input('t_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate wall thickness (mirrors input)')

        self.add_input('s_mb1', val=0.0, units='m', desc='Bearing 1 s-coordinate along drivetrain, measured from bedplate')
        self.add_input('s_mb2', val=0.0, units='m', desc='Bearing 2 s-coordinate along drivetrain, measured from bedplate')
        self.add_input('mb1_mass', 0.0, units='kg', desc='component mass')
        self.add_input('mb1_I', np.zeros(3), units='kg*m**2', desc='component I')
        self.add_input('mb2_mass', 0.0, units='kg', desc='component mass')
        self.add_input('mb2_I', np.zeros(3), units='kg*m**2', desc='component I')
        
        self.add_input('s_stator', val=0.0, units='m', desc='Generator stator attachment to shaft s-coordinate measured from bedplate')
        self.add_input('m_stator', val=0.0, units='kg', desc='Generator stator mass')
        self.add_input('cm_stator', val=0.0, units='kg', desc='Generator stator center of mass (measured along drivetrain from bedplate)')
        self.add_input('I_stator', val=np.zeros(3), units='kg*m**2', desc='Generator stator moment of inertia (measured about cm)')
        
        self.add_input('F_mb1', val=np.zeros((3,n_dlcs)), units='N', desc='Force vector applied to bearing 1 in hub c.s.')
        self.add_input('F_mb2', val=np.zeros((3,n_dlcs)), units='N', desc='Force vector applied to bearing 2 in hub c.s.')
        self.add_input('M_mb1', val=np.zeros((3,n_dlcs)), units='N', desc='Moment vector applied to bearing 1 in hub c.s.')
        self.add_input('M_mb2', val=np.zeros((3,n_dlcs)), units='N', desc='Moment vector applied to bearing 2 in hub c.s.')

        self.add_input('other_mass', val=0.0, units='kg', desc='Mass of other nacelle components that rest on mainplate')

        self.add_input('E', val=0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_input('G', val=0.0, units='N/m**2', desc='shear modulus')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_input('sigma_y', val=0.0, units='N/m**2', desc='yield stress')
        self.add_input('gamma_f', val=0.0, desc='safety factor')
        self.add_input('gamma_m', 0.0, desc='safety factor on materials')
        self.add_input('gamma_n', 0.0, desc='safety factor on consequence of failure')
        
        #self.add_output('top_deflection', np.zeros(n_points), units='m', desc='Deflection of Curved_beam top in yaw-aligned +x direction')
        self.add_output('mb1_displacement', val=np.zeros(n_dlcs), units='m', desc='Total deflection distance of bearing 1')
        self.add_output('mb2_displacement', val=np.zeros(n_dlcs), units='m', desc='Total deflection distance of bearing 2')
        self.add_output('mb1_rotation', val=np.zeros(n_dlcs), units='rad', desc='Total rotation angle of bearing 1')
        self.add_output('mb2_rotation', val=np.zeros(n_dlcs), units='rad', desc='Total rotation angle of bearing 2')
        self.add_output('base_F', val=np.zeros((3,n_dlcs)), units='N', desc='Total reaction force at bedplate base')
        self.add_output('base_M', val=np.zeros((3,n_dlcs)), units='N*m', desc='Total reaction moment at bedplate base')
        self.add_output('nose_axial_stress', np.zeros((n_points+4,n_dlcs)), units='N/m**2', desc='Axial stress in Curved_beam structure')
        self.add_output('nose_shear_stress', np.zeros((n_points+4,n_dlcs)), units='N/m**2', desc='Shear stress in Curved_beam structure')
        self.add_output('nose_bending_stress', np.zeros((n_points+4,n_dlcs)), units='N/m**2', desc='Hoop stress in Curved_beam structure calculated with Roarks formulae')
        self.add_output('constr_nose_vonmises', np.zeros((n_points+4,n_dlcs)), desc='Sigma_y/Von_Mises')

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack inputs
        upwind     = discrete_inputs['upwind']
        Cup        = -1.0 if upwind else 1.0
        tiltD      = float(inputs['tilt'])
        tiltR      = np.deg2rad(tiltD)
        
        x_c        = inputs['x_bedplate']
        z_c        = inputs['z_bedplate']
        x_inner    = inputs['x_bedplate_inner']
        z_inner    = inputs['z_bedplate_inner']
        x_outer    = inputs['x_bedplate_outer']
        z_outer    = inputs['z_bedplate_outer']
        D_bed      = inputs['D_bedplate']
        t_bed      = inputs['t_bedplate']

        s_nose     = inputs['s_nose'][1:] # First point duplicated with bedplate
        D_nose     = inputs['D_nose']
        t_nose     = inputs['t_nose']
        x_nose     = s_nose.copy()
        x_nose    *= Cup
        x_nose    += x_c[-1]
        
        s_mb1      = float(inputs['s_mb1'])
        s_mb2      = float(inputs['s_mb2'])
        m_mb1      = float(inputs['mb1_mass'])
        m_mb2      = float(inputs['mb2_mass'])
        I_mb1      = inputs['mb1_I']
        I_mb2      = inputs['mb2_I']
        
        s_stator   = float(inputs['s_stator'])
        m_stator   = float(inputs['m_stator'])
        cm_stator  = float(inputs['cm_stator'])
        I_stator   = inputs['I_stator']
        
        rho        = float(inputs['rho'])
        E          = float(inputs['E'])
        G          = float(inputs['G'])
        sigma_y    = float(inputs['sigma_y'])
        gamma_f    = float(inputs['gamma_f'])
        gamma_m    = float(inputs['gamma_m'])
        gamma_n    = float(inputs['gamma_n'])

        F_mb1      = inputs['F_mb1']
        F_mb2      = inputs['F_mb2']
        M_mb1      = inputs['M_mb1']
        M_mb2      = inputs['M_mb2']

        m_other    = float(inputs['other_mass'])
        
        # ------- node data ----------------
        n     = len(x_c) + len(x_nose)
        inode = np.arange(1, n+1)
        ynode = rnode = np.zeros(n)
        xnode = np.r_[x_c, x_nose]
        znode = np.r_[z_c, z_c[-1]*np.ones(x_nose.shape)]
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        inose   = len(x_c)
        istator = inode[xnode==Cup*s_stator+x_c[-1]]
        i1      = inode[xnode==Cup*s_mb1+x_c[-1]]
        i2      = inode[xnode==Cup*s_mb2+x_c[-1]]
        # ------------------------------------
        
        # ------ reaction data ------------
        # Rigid base
        rnode = [int(inode[0])]
        rk    = np.array([RIGID])
        reactions = frame3dd.ReactionData(rnode, rk, rk, rk, rk, rk, rk, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        bedcyl   = tube.Tube(nodal2sectional(D_bed)[0], nodal2sectional(t_bed)[0])
        nosecyl  = tube.Tube(nodal2sectional(D_nose)[0], nodal2sectional(t_nose)[0])
        ielement = np.arange(1, n)
        N1       = np.arange(1, n)
        N2       = np.arange(2, n+1)
        roll     = np.zeros(n-1)
        myones   = np.ones(n-1)
        Ax = np.r_[bedcyl.Area, nosecyl.Area]
        As = np.r_[bedcyl.Asx, nosecyl.Asx]
        S  = np.r_[bedcyl.S, nosecyl.S]
        C  = np.r_[bedcyl.C, nosecyl.C]
        J0 = np.r_[bedcyl.J0, nosecyl.J0]
        Jx = np.r_[bedcyl.Jxx, nosecyl.Jxx]
        
        elements = frame3dd.ElementData(ielement, N1, N2, Ax, As, As, J0, Jx, Jx, E*myones, G*myones, roll, rho*myones)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add misc nacelle components at base and stator extra mass ------------
        myframe.changeExtraNodeMass(np.r_[inode[0], istator, i1, i2], [m_other, m_stator, m_mb1, m_mb2],
                                    [0.0, I_stator[0], I_mb1[0], I_mb2[0]],
                                    [0.0, I_stator[1], I_mb1[1], I_mb2[1]],
                                    [0.0, I_stator[2], I_mb1[2], I_mb2[2]],
                                    np.zeros(4), np.zeros(4), np.zeros(4), 
                                    [0.0, cm_stator, 0.0, 0.0], np.zeros(4), np.zeros(4), True)
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        #myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs   = self.options['n_dlcs']
        gy = 0.0
        gx = -gravity*np.sin(tiltR)
        gz = -gravity*np.cos(tiltR)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)
            
            # point loads
            # TODO: Are input loads aligned with the shaft? If so they need to be rotated.
            F_12 = np.c_[F_mb2[:,k], F_mb1[:,k]]
            M_12 = np.c_[M_mb2[:,k], M_mb1[:,k]]
            load.changePointLoads(np.r_[i2, i1], F_12[0,:], F_12[1,:], F_12[2,:], M_12[0,:], M_12[1,:], M_12[2,:])
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)
            
        #myframe.write('myframe.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # ------------ Bedplate "curved beam" geometry for post-processing -------------
        # Need to compute neutral axis, so shift points such that bedplate top is at x=0
        R_c = np.sqrt( (x_c    -x_c[-1])**2 + z_c**2)
        Ro  = np.sqrt( (x_outer-x_c[-1])**2 + z_outer**2)
        Ri  = np.sqrt( (x_inner-x_c[-1])**2 + z_inner**2)
        r_bed_o = 0.5*D_bed
        r_bed_i = r_bed_o - t_bed
        A_bed   = np.pi * (r_bed_o**2 - r_bed_i**2)

        # Radius of the neutral axis
        # http://faculty.fairfield.edu/wdornfeld/ME311/BasicStressEqns-DBWallace.pdf
        R_n  = A_bed / (2*np.pi * ( np.sqrt(R_c**2 - r_bed_i**2) - np.sqrt(R_c**2 - r_bed_o**2) ) )
        e_cn = R_c - R_n
        # ------------------------------------
        
        # Loop over DLCs and append to outputs
        outputs['mb1_deflection']       = np.zeros(n_dlcs)
        outputs['mb2_deflection']       = np.zeros(n_dlcs)
        outputs['mb1_rotation']         = np.zeros(n_dlcs)
        outputs['mb2_rotation']         = np.zeros(n_dlcs)
        outputs['base_F']               = np.zeros((3, n_dlcs))
        outputs['base_M']               = np.zeros((3, n_dlcs))
        outputs['nose_axial_stress']    = np.zeros((n-1, n_dlcs))
        outputs['nose_shear_stress']    = np.zeros((n-1, n_dlcs))
        outputs['nose_bending_stress']  = np.zeros((n-1, n_dlcs))
        outputs['constr_nose_vonmises'] = np.zeros((n-1, n_dlcs))
        for k in range(n_dlcs):
            # Deflections and rotations at bearings- how to sum up rotation angles?
            outputs['mb1_deflection'][k] = np.sqrt(displacements.dx[k,i1-1]**2 + displacements.dy[k,i1-1]**2 + displacements.dz[k,i1-1]**2)
            outputs['mb2_deflection'][k] = np.sqrt(displacements.dx[k,i2-1]**2 + displacements.dy[k,i2-1]**2 + displacements.dz[k,i2-1]**2)
            outputs['mb1_rotation'][k]   = displacements.dxrot[k,i1-1] + displacements.dyrot[k,i1-1] + displacements.dzrot[k,i1-1]
            outputs['mb2_rotation'][k]   = displacements.dxrot[k,i2-1] + displacements.dyrot[k,i2-1] + displacements.dzrot[k,i2-1]

            # shear and bending, one per element (convert from local to global c.s.)
            Fz =  forces.Nx[k, 1::2]
            Vy =  forces.Vy[k, 1::2]
            Vx = -forces.Vz[k, 1::2]
            F  =  np.sqrt(Vx**2 + Vy**2)

            Mzz =  forces.Txx[k, 1::2]
            Myy =  forces.Myy[k, 1::2]
            Mxx = -forces.Mzz[k, 1::2]
            M   =  np.sqrt(Myy**2 + Mxx**2)

            # Record total forces and moments
            F_base_k = DirectionVector(-reactions.Fx[k,:].sum(), -reactions.Fy[k,:].sum(), -reactions.Fz[k,:].sum())
            M_base_k = DirectionVector(-reactions.Mxx[k,:].sum(), -reactions.Myy[k,:].sum(), -reactions.Mzz[k,:].sum())

            # Rotate vector from tilt axes to yaw/tower axes
            outputs['base_F'][:,k] = F_base_k.hubToYaw(-tiltD).toArray()
            outputs['base_M'][:,k] = M_base_k.hubToYaw(-tiltD).toArray()

            outputs['nose_axial_stress'][:,k] = Fz/Ax + M/S
            outputs['nose_shear_stress'][:,k] = 2.0*F/As + Mzz/C

            Bending_stress_outer = M[:(inose-1)] * nodal2sectional( (Ro-R_n) / (A_bed*e_cn*Ro) )[0]
            Bending_stress_inner = M[:(inose-1)] * nodal2sectional( (R_n-Ri) / (A_bed*e_cn*Ri) )[0]
            outputs['nose_bending_stress'][:(inose-1),k] = Bending_stress_outer
        
            outputs['constr_nose_vonmises'][:,k] = Util.vonMisesStressUtilization(outputs['nose_axial_stress'][:,k],
                                                                                  outputs['nose_bending_stress'][:,k],
                                                                                  outputs['nose_shear_stress'][:,k],
                                                                                  gamma_f*gamma_m*gamma_n, sigma_y)







class Hub_Rotor_Shaft_Frame(om.ExplicitComponent):
    """Run structural analysis of hub system with the generator rotor and main shaft."""
    
    def initialize(self):
        self.options.declare('n_points')
        self.options.declare('n_dlcs')
    
    def setup(self):
        n_points = self.options['n_points']
        n_dlcs   = self.options['n_dlcs']

        self.add_input('tilt', 0.0, units='deg', desc='Shaft tilt') 

        self.add_input('s_shaft', val=np.zeros(6), units='m', desc='Discretized s-coordinates along drivetrain, measured from bedplate')
        self.add_input('D_shaft', val=np.zeros(6), units='m', desc='Shaft discretized diameter values at coordinates')
        self.add_input('t_shaft', val=np.zeros(6), units='m', desc='Shaft discretized thickness values at coordinates')

        self.add_input('hub_system_mass', 0.0, units='kg', desc='Hub system mass') 
        self.add_input('hub_system_cm', 0.0, units='m', desc='Hub system center of mass distance from hub flange') 
        self.add_input('hub_system_I', np.zeros(3), units='kg*m**2', desc='Hub system moment of inertia') 
        self.add_input('F_hub', val=np.zeros((3,n_dlcs)), units='N', desc='Force vector applied to the hub (WITH WEIGHT???)')
        self.add_input('M_hub', val=np.zeros((3,n_dlcs)), units='N', desc='Moment vector applied to the hub')

        self.add_input('s_mb1', val=0.0, units='m', desc='Bearing 1 s-coordinate along drivetrain, measured from bedplate')
        self.add_input('s_mb2', val=0.0, units='m', desc='Bearing 2 s-coordinate along drivetrain, measured from bedplate')
        
        self.add_input('s_rotor', val=0.0, units='m', desc='Generator rotor attachment to shaft s-coordinate measured from bedplate')
        self.add_input('m_rotor', val=0.0, units='kg', desc='Generator rotor mass')
        self.add_input('cm_rotor', val=0.0, units='kg', desc='Generator rotor center of mass (measured along nose from bedplate)')
        self.add_input('I_rotor', val=np.zeros(3), units='kg*m**2', desc='Generator rotor moment of inertia (measured about its cm)')
        
        self.add_input('E', val=0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_input('G', val=0.0, units='N/m**2', desc='shear modulus')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_input('sigma_y', val=0.0, units='N/m**2', desc='yield stress')
        self.add_input('gamma_f', val=0.0, desc='safety factor')
        self.add_input('gamma_m', 0.0, desc='safety factor on materials')
        self.add_input('gamma_n', 0.0, desc='safety factor on consequence of failure')
        
        #self.add_output('top_deflection', np.zeros(n_points), units='m', desc='Deflection of Curved_beam top in yaw-aligned +x direction')
        self.add_output('rotor_axial_stress', np.zeros((5,n_dlcs)), units='N/m**2', desc='Axial stress in Curved_beam structure')
        self.add_output('rotor_shear_stress', np.zeros((5,n_dlcs)), units='N/m**2', desc='Shear stress in Curved_beam structure')
        self.add_output('rotor_bending_stress', np.zeros((5,n_dlcs)), units='N/m**2', desc='Hoop stress in Curved_beam structure calculated with Roarks formulae')
        self.add_output('constr_rotor_vonmises', np.zeros((5,n_dlcs)), desc='Sigma_y/Von_Mises')
        self.add_output('F_mb1', val=np.zeros((3,n_dlcs)), units='N', desc='Force vector applied to bearing 1 in hub c.s.')
        self.add_output('F_mb2', val=np.zeros((3,n_dlcs)), units='N', desc='Force vector applied to bearing 2 in hub c.s.')
        self.add_output('M_mb1', val=np.zeros((3,n_dlcs)), units='N', desc='Moment vector applied to bearing 1 in hub c.s.')
        self.add_output('M_mb2', val=np.zeros((3,n_dlcs)), units='N', desc='Moment vector applied to bearing 2 in hub c.s.')


        
    def compute(self, inputs, outputs):

        # Unpack inputs
        tilt       = float(np.deg2rad(inputs['tilt']))
        
        s_shaft    = inputs['s_shaft']
        D_shaft    = inputs['D_shaft']
        t_shaft    = inputs['t_shaft']

        s_mb1      = float(inputs['s_mb1'])
        s_mb2      = float(inputs['s_mb2'])
        
        s_rotor   = float(inputs['s_rotor'])
        m_rotor   = float(inputs['m_rotor'])
        cm_rotor  = float(inputs['cm_rotor'])
        I_rotor   = inputs['I_rotor']
        
        rho        = float(inputs['rho'])
        E          = float(inputs['E'])
        G          = float(inputs['G'])
        sigma_y    = float(inputs['sigma_y'])
        gamma_f    = float(inputs['gamma_f'])
        gamma_m    = float(inputs['gamma_m'])
        gamma_n    = float(inputs['gamma_n'])

        m_hub      = float(inputs['hub_system_mass'])
        cm_hub     = float(inputs['hub_system_cm'])
        I_hub      = inputs['hub_system_I']
        F_hub      = inputs['F_hub']
        M_hub      = inputs['M_hub']

        # ------- node data ----------------
        n     = len(s_shaft)
        inode = np.arange(1, n+1)
        ynode = znode = rnode = np.zeros(n)
        xnode = s_shaft.copy()
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        irotor = inode[xnode==s_rotor]
        i1 = inode[xnode==s_mb1]
        i2 = inode[xnode==s_mb2]
        # ------------------------------------
        
        # ------ reaction data ------------
        # Reactions at main bearings
        rnode = np.r_[i1, i2, irotor]
        Rx  = np.array([RIGID, FREE, FREE]) # Upwind bearing restricts translational
        Ry  = np.array([RIGID, FREE, FREE]) # Upwind bearing restricts translational
        Rz  = np.array([RIGID, FREE, FREE]) # Upwind bearing restricts translational
        Rxx = np.array([FREE,  FREE, RIGID]) # Torque is absorbed by stator, so this is the best way to capture that
        Ryy = np.array([FREE,  RIGID, FREE]) # downwind bearing carry moments
        Rzz = np.array([FREE,  RIGID, FREE]) # downwind bearing carry moments
        # George's way
        #rnode = np.r_[irotor, i1, i2]
        #Rx  = np.array([FREE,  RIGID, FREE]) # WHY?
        #Ry  = np.array([FREE,  RIGID, RIGID]) # WHY?
        #Rz  = np.array([FREE,  RIGID, RIGID]) # WHY?
        #Rxx = np.array([RIGID, FREE,  FREE]) # pass the torque to the generator
        #Ryy = np.array([FREE,  RIGID,  FREE]) # upwind tapered bearing carry Ryy
        #Rzz = np.array([FREE,  RIGID,  FREE]) # upwind tapered bearing carry Rzz
        reactions = frame3dd.ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        shaftcyl = tube.Tube(nodal2sectional(D_shaft)[0], nodal2sectional(t_shaft)[0])
        ielement = np.arange(1, n)
        N1       = np.arange(1, n)
        N2       = np.arange(2, n+1)
        roll     = np.zeros(n-1)
        myones   = np.ones(n-1)
        Ax = shaftcyl.Area
        As = shaftcyl.Asx
        S  = shaftcyl.S
        C  = shaftcyl.C
        J0 = shaftcyl.J0
        Jx = shaftcyl.Jxx
        
        elements = frame3dd.ElementData(ielement, N1, N2, Ax, As, As, J0, Jx, Jx, E*myones, G*myones, roll, rho*myones)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add hub and generator rotor extra mass ------------
        myframe.changeExtraNodeMass(np.r_[1, irotor], [m_hub, m_rotor],
                                    [I_hub[0], I_rotor[0]], [I_hub[1], I_rotor[1]], [I_hub[2], I_rotor[2]], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                                    [cm_hub, cm_rotor], [0.0, 0.0], [0.0, 0.0], True)
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        #myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs   = self.options['n_dlcs']
        gy = 0.0
        gx = -gravity*np.sin(tilt)
        gz = -gravity*np.cos(tilt)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)
            
            # point loads
            # TODO: Are input loads aligned with the shaft? If so they need to be rotated.
            load.changePointLoads([1], F_hub[0], F_hub[1], F_hub[2], M_hub[0], M_hub[1], M_hub[2])
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)
            
        #myframe.write('myframe.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # Loop over DLCs and append to outputs
        outputs['F_mb1'] = np.zeros((3, n_dlcs))
        outputs['F_mb2'] = np.zeros((3, n_dlcs))
        outputs['M_mb1'] = np.zeros((3, n_dlcs))
        outputs['M_mb2'] = np.zeros((3, n_dlcs))
        outputs['rotor_axial_stress'] = np.zeros((n-1, n_dlcs))
        outputs['rotor_shear_stress'] = np.zeros((n-1, n_dlcs))
        outputs['rotor_bending_stress'] = np.zeros((n-1, n_dlcs))
        outputs['constr_rotor_vonmises'] = np.zeros((n-1, n_dlcs))
        for k in range(n_dlcs):
            # natural frequncies
            #outputs['f1'] = modal.freq[0]
            #outputs['f2'] = modal.freq[1]
            
            # deflections - at bearings?
            #outputs['top_deflection'] = np.sqrt(displacements.dx[k, n-1]**2 +
            #                                    displacements.dy[k, n-1]**2 +
            #                                    displacements.dz[k, n-1]**2)

            # shear and bending, one per element (convert from local to global c.s.)
            Fz =  forces.Nx[k, 1::2]
            Vy =  forces.Vy[k, 1::2]
            Vx = -forces.Vz[k, 1::2]
            F  =  np.sqrt(Vx**2 + Vy**2)

            Mzz =  forces.Txx[k, 1::2]
            Myy =  forces.Myy[k, 1::2]
            Mxx = -forces.Mzz[k, 1::2]
            M   =  np.sqrt(Myy**2 + Mxx**2)

            # Record total forces and moments
            outputs['F_mb1'][:,k] = -1.0 * np.array([reactions.Fx[k,0], reactions.Fy[k,0], reactions.Fz[k,0]])
            outputs['F_mb2'][:,k] = -1.0 * np.array([reactions.Fx[k,1], reactions.Fy[k,1], reactions.Fz[k,1]])
            outputs['M_mb1'][:,k] = -1.0 * np.array([reactions.Mxx[k,0], reactions.Myy[k,0], reactions.Mzz[k,0]])
            outputs['M_mb2'][:,k] = -1.0 * np.array([reactions.Mxx[k,1], reactions.Myy[k,1], reactions.Mzz[k,1]])

            outputs['rotor_axial_stress'][:,k] = Fz/Ax + M/S
            outputs['rotor_shear_stress'][:,k] = 2.0*F/As + Mzz/C
            hoop = np.zeros(F.shape)
        
            outputs['constr_rotor_vonmises'][:,k] = Util.vonMisesStressUtilization(outputs['rotor_axial_stress'][:,k],
                                                                                   hoop,
                                                                                   outputs['rotor_shear_stress'][:,k],
                                                                                   gamma_f*gamma_m*gamma_n, sigma_y)


            
if __name__ == '__main__':

    n_points = 15

    # --- geometry ----
    
    prob = om.Problem()
    prob.model.add_subsystem('bed', Curved_Bedplate(n_points = n_points), promotes=['*'])
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('L_bedplate',0.0, units='m')
    ivc.add_output('H_bedplate', 0.0, units='m')
    ivc.add_output('tilt', 0.0, units='deg')
    ivc.add_output('D_top', 0.0, units='m')
    ivc.add_output('D_nose', np.zeros(n_points), units='m')
    ivc.add_output('bedplate_wall_thickness', np.zeros(n_points), units='m')

    # --- Setup Pptimizer ---
    prob.driver = om.pyOptSparseDriver() # om.ScipyOptimizeDriver() #
    prob.driver.options['optimizer'] = 'SNOPT'

    # --- Objective ---
    prob.model.add_objective('mass', scaler=1e-4)
    # ----------------------
    prob.model.add_constraint('constr_vonmises', upper = 1.0)
    #prob.model.add_constraint('con_cmp1.con1', lower=0. )
    # prob.model.add_constraint('con_cmp2.con2', lower=0. )
    # --- Design Variables ---
    prob.model.add_design_var('bedplate_wall_thickness', lower=0.05, upper=0.09 )

    prob.setup()

    # # --- loading case 1: max Thrust ---
    prob['F'] = np.array([2409.750e3, -1716.429e3, 74.3529e3])
    prob['M'] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3])
    # # ---------------
    
    prob['L_bedplate'] = 5
    prob['H_bedplate'] = 4.875
    prob['D_top'] = 6.5
    prob['D_nose'] = 2.2*np.ones(n_points)
    prob['bedplate_wall_thickness'] = 0.06*np.ones(n_points)
    prob['tilt'] = 4.0
    prob['upwind'] = False #True
    
    # --- material props ---
    prob['E'] = 210e9
    prob['G'] = 80.8e9
    prob['rho'] = 7850.
    prob['sigma_y'] = 250e6
    # -----------

    # --- safety factors ---
    prob['gamma_f'] = 1.35
    prob['gamma_m'] = 1.3
    prob['gamma_n'] = 1.0
    # --- safety factors ---
    
    #prob.model.approx_totals()
    #prob.run_driver()
    prob.run_model()

    # ------------
    print('==================================')
    print('x_c (m) =', prob['x_c'])
    print('z_c (m) =', prob['z_c'])
    print('x_inner (m) =', prob['x_inner'])
    print('z_inner (m) =', prob['z_inner'])
    print('x_outer (m) =', prob['x_outer'])
    print('z_outer (m) =', prob['z_outer'])
    print('mass (kg) =', prob['mass'])
    print('cg (m) =', prob['center_of_mass'])
    print('wall_thickness (m)=',prob['bedplate_wall_thickness'])
    print('top_deflection1 (m) =', prob['top_deflection'])
    print('Reaction forces F =', prob['base_F'] )
    print('Reaction Moments M =', prob['base_M'] )
    print('Axial stresses =', prob['axial_stress'])
    print('Bending stresses =', prob['bending_stress'])
    print('Shear stresses =', prob['shear_stress'])
    print('Safety factor limit =', prob['constr_vonmises'])
   
    
