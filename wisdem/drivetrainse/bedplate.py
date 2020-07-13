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

RIGID = 1e30


class Curved_Bedplate(om.ExplicitComponent):
    """discretize geometry into finite element nodes"""
    
    def initialize(self):
        self.options.declare('n_points')
        self.options.declare('n_dlcs')
    
    def setup(self):
        n_points = self.options['n_points']
        n_dlcs   = self.options['n_dlcs']

        self.add_input('x_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized x-coordinates')
        self.add_input('z_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized z-coordinates')
        self.add_input('D_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized diameter values at coordinates')
        self.add_input('t_nose', val=np.zeros(n_points+2), units='m', desc='Nose discretized thickness values at coordinates')

        self.add_input('x_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline x-coordinates')
        self.add_input('z_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate centerline z-coordinates')
        self.add_input('x_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve x-coordinates')
        self.add_input('z_bedplate_inner', val=np.zeros(n_points), units='m', desc='Bedplate lower curve z-coordinates')
        self.add_input('x_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve x-coordinates')
        self.add_input('z_bedplate_outer', val=np.zeros(n_points), units='m', desc='Bedplate outer curve z-coordinates')
        self.add_input('D_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate diameters')
        self.add_input('t_bedplate', val=np.zeros(n_points), units='m', desc='Bedplate wall thickness (mirrors input)')

        self.add_input('x_mb1', val=0.0, units='m', desc='Bearing 1 x-coordinate')
        self.add_input('z_mb1', val=0.0, units='m', desc='Bearing 1 z-coordinate')
        self.add_input('x_mb2', val=0.0, units='m', desc='Bearing 2 x-coordinate')
        self.add_input('z_mb2', val=0.0, units='m', desc='Bearing 2 z-coordinate')
        
        self.add_input('x_stator', val=0.0, units='m', desc='Generator stator attachment to nose x-coordinate')
        self.add_input('z_stator', val=0.0, units='m', desc='Generator stator attachment to nose z-coordinate')
        self.add_input('m_stator', val=0.0, units='kg', desc='Generator stator mass')
        self.add_input('cm_stator', val=0.0, units='kg', desc='Generator stator center of mass (measured from attachment)')
        self.add_input('I_stator', val=0.0, units='kg', desc='Generator stator moment of inertia (measured from attachment)')
        
        self.add_input('F_mb1', val=np.zeros((3,n_dlcs)), units='N', desc='Force vector applied to bearing 1')
        self.add_input('F_mb2', val=np.zeros((3,n_dlcs)), units='N', desc='Force vector applied to bearing 2')
        self.add_input('M_mb1', val=np.zeros((3,n_dlcs)), units='N', desc='Moment vector applied to bearing 1')
        self.add_input('M_mb2', val=np.zeros((3,n_dlcs)), units='N', desc='Moment vector applied to bearing 2')

        self.add_input('E', val=0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_input('G', val=0.0, units='N/m**2', desc='shear modulus')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_input('sigma_y', val=0.0, units='N/m**2', desc='yield stress')
        self.add_input('gamma_f', val=0.0, units='m', desc='safety factor')
        self.add_input('gamma_m', 0.0, desc='safety factor on materials')
        self.add_input('gamma_n', 0.0, desc='safety factor on consequence of failure')
        
        self.add_output('top_deflection', np.zeros(n_points), units='m', desc='Deflection of Curved_beam top in yaw-aligned +x direction')
        self.add_output('base_F', val=np.zeros((3,n_dlcs)), units='N', desc='Total force on Curved_beam')
        self.add_output('base_M', val=np.zeros((3,n_dlcs)), units='N*m', desc='Total moment on Curved_beam measured at base')
        self.add_output('axial_stress', np.zeros((2*n_points+1,n_dlcs)), units='N/m**2', desc='Axial stress in Curved_beam structure')
        self.add_output('shear_stress', np.zeros((2*n_points+1,n_dlcs)), units='N/m**2', desc='Shear stress in Curved_beam structure')
        self.add_output('bending_stress', np.zeros((2*n_points+1,n_dlcs)), units='N/m**2', desc='Hoop stress in Curved_beam structure calculated with Roarks formulae')
        self.add_output('constr_vonmises', np.zeros((2*n_points+1,n_dlcs)), desc='Sigma_y/Von_Mises')

        
    def compute(self, inputs, outputs):

        # Unpack inputs
        x_c        = inputs['x_bedplate']
        z_c        = inputs['z_bedplate']
        x_inner    = inputs['x_bedplate_inner']
        z_inner    = inputs['z_bedplate_inner']
        x_outer    = inputs['x_bedplate_outer']
        z_outer    = inputs['z_bedplate_outer']
        D_bed      = inputs['D_bedplate']
        t_bed      = inputs['t_bedplate']

        x_nose     = inputs['x_nose'][:-1] # Last point duplicated with bedplate
        z_nose     = inputs['z_nose'][:-1] # Last point duplicated with bedplate
        D_nose     = inputs['D_nose']
        t_nose     = inputs['t_nose']

        x_mb1      = float(inputs['x_mb1'])
        z_mb1      = float(inputs['z_mb1'])
        x_mb2      = float(inputs['x_mb2'])
        z_mb2      = float(inputs['z_mb2'])
        
        x_stator   = float(inputs['x_stator'])
        z_stator   = float(inputs['z_stator'])
        m_stator   = float(inputs['m_stator'])
        cm_stator  = inputs['cm_stator']
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

        # ------- node data ----------------
        n     = len(x_nose) + len(x_c)
        inode = np.arange(1, n+1)
        ynode = rnode = np.zeros(n)
        xnode = np.r_[x_nose, x_c]
        znode = np.r_[z_nose, z_c]
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        ibed  = len(x_nose)
        istator = inode[np.logical_and(xnode==x_stator, znode==z_stator)]
        i1 = inode[np.logical_and(xnode==x_mb1, znode==z_mb1)]
        i2 = inode[np.logical_and(xnode==x_mb2, znode==z_mb2)]
        # ------------------------------------
        
        # ------ reaction data ------------
        # Rigid base
        rnode = [int(inode[-1])]
        rk = np.array([RIGID])
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

        # ------ add stator extra mass ------------
        myframe.changeExtraNodeMass(istator, [m_stator], [I_stator[0]], [I_stator[1]], [I_stator[2]], [I_stator[3]], [I_stator[4]], [I_stator[5]],
                                    [cm_stator[0]], [cm_stator[1]], [cm_stator[2]], True)
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        #myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs   = self.options['n_dlcs']
        gx = gy = 0.0
        gz = -gravity
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)
            
            # point loads
            # TODO: Are input loads aligned with the shaft? If so they need to be rotated.
            F_12 = np.c_[F_mb1[:,k], F_mb2[:,k]]
            M_12 = np.c_[M_mb1[:,k], M_mb2[:,k]]
            load.changePointLoads(np.r_[i1, i2], F_12[0,:], F_12[1,:], F_12[2,:], M_12[0,:], M_12[1,:], M_12[2,:])
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)
            
        myframe.write('myframe.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()
        
        # ------------ Bedplate "curved beam" geometry for post-processing -------------
        # Need to compute neutral axis, so shift points such that bedplate top is at x=0
        R_c = np.sqrt( (x_c    -x_c[0])**2 + z_c**2)
        Ro  = np.sqrt( (x_outer-x_c[0])**2 + z_outer**2)
        Ri  = np.sqrt( (x_inner-x_c[0])**2 + z_inner**2)
        r_bed_o = 0.5*D_bed
        r_bed_i = r_bed_o - t_bed
        A_bed   = np.pi * (r_bed_o**2 - r_bed_i**2)

        # Radius of the neutral axis
        # http://faculty.fairfield.edu/wdornfeld/ME311/BasicStressEqns-DBWallace.pdf
        R_n  = A_bed / (2*np.pi * ( np.sqrt(R_c**2 - r_bed_i**2) - np.sqrt(R_c**2 - r_bed_o**2) ) )
        e_cn = R_c - R_n
        # ------------------------------------
        
        # Loop over DLCs and append to outputs
        outputs['base_F'] = np.zeros((3, n_dlcs))
        outputs['base_M'] = np.zeros((3, n_dlcs))
        outputs['axial_stress'] = np.zeros((n-1, n_dlcs))
        outputs['shear_stress'] = np.zeros((n-1, n_dlcs))
        outputs['bending_stress'] = np.zeros((n-1, n_dlcs))
        outputs['constr_vonmises'] = np.zeros((n-1, n_dlcs))
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
            outputs['base_F'][:,k] = -1.0 * np.array([reactions.Fx[k,:].sum(), reactions.Fy[k,:].sum(), reactions.Fz[k,:].sum()])
            outputs['base_M'][:,k] = -1.0 * np.array([reactions.Mxx[k,:].sum(), reactions.Myy[k,:].sum(), reactions.Mzz[k,:].sum()])

            outputs['axial_stress'][:,k] = Fz/Ax + M/S
            outputs['shear_stress'][:,k] = 2.0*F/As + Mzz/C

            Bending_stress_outer = M[ibed:] * nodal2sectional( (Ro-R_n) / (A_bed*e_cn*Ro) )[0]
            Bending_stress_inner = M[ibed:] * nodal2sectional( (R_n-Ri) / (A_bed*e_cn*Ri) )[0]
            outputs['bending_stress'][ibed:,k] = Bending_stress_outer
        
            outputs['constr_vonmises'][:,k] = Util.vonMisesStressUtilization(outputs['axial_stress'][:,k],
                                                                             outputs['bending_stress'][:,k],
                                                                             outputs['shear_stress'][:,k],
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
   
    
