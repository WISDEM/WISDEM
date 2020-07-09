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
    
    def setup(self):
        n_points = self.options['n_points']

        self.add_input('L_bedplate',0.0, units='m', desc='Location of tower top') 
        self.add_input('H_bedplate', 0.0, units='m', desc='height of nose')
        self.add_input('D_top',0.0, units='m', desc='Tower diameter at top')
        self.add_input('D_nose', 0.0, units='m', desc='nose diameter')
        self.add_input('wall_thickness', np.zeros(n_points), units='m', desc='Beam thickness at corresponding locations')
        
        self.add_input('F', val=np.zeros(3), units='N', desc='force vector applied to the bedplate')
        self.add_input('M', val=np.zeros(3), units='N', desc='force vector applied to the bedplate')

        self.add_input('E', val=0.0, units='N/m**2', desc='modulus of elasticity')
        self.add_input('G', val=0.0, units='N/m**2', desc='shear modulus')
        self.add_input('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_input('sigma_y', val=0.0, units='N/m**2', desc='yield stress')
        self.add_input('gamma_f', val=0.0, units='m', desc='safety factor')
        self.add_input('gamma_m', 0.0, desc='safety factor on materials')
        self.add_input('gamma_n', 0.0, desc='safety factor on consequence of failure')
        
        self.add_output('x_c', val=np.zeros(n_points), units='m', desc='Centerline x-coordinate')
        self.add_output('z_c', val=np.zeros(n_points), units='m', desc='Centerline z-coordinate')
        self.add_output('x_inner', val=np.zeros(n_points), units='m', desc='Lower bedplate curve x-coordinate')
        self.add_output('z_inner', val=np.zeros(n_points), units='m', desc='Lower bedplate curve z-coordinate')
        self.add_output('x_outer', val=np.zeros(n_points), units='m', desc='Outer bedplate curve x-coordinate')
        self.add_output('z_outer', val=np.zeros(n_points), units='m', desc='Outer bedplate curve z-coordinate')
        self.add_output('mass', val=0.0, units='kg', desc='Total curved beam mass')
        self.add_output('center_of_mass', val=np.zeros(3), units='m', desc='center of mass of 2-D curved beam')
        self.add_output('top_deflection', np.zeros(n_points), units='m', desc='Deflection of Curved_beam top in yaw-aligned +x direction')
        self.add_output('base_F', val=np.zeros(3), units='N', desc='Total force on Curved_beam')
        self.add_output('base_M', val=np.zeros(3), units='N*m', desc='Total moment on Curved_beam measured at base')
        self.add_output('axial_stress', np.zeros(n_points-1), units='N/m**2', desc='Axial stress in Curved_beam structure')
        self.add_output('shear_stress', np.zeros(n_points-1), units='N/m**2', desc='Shear stress in Curved_beam structure')
        self.add_output('bending_stress', np.zeros(n_points-1), units='N/m**2', desc='Hoop stress in Curved_beam structure calculated with Roarks formulae')
        self.add_output('constr_vonmises', np.zeros(n_points-1), desc='Sigma_y/Von_Mises')

    def compute(self, inputs, outputs):

        # Unpack inputs
        L_bedplate = float(inputs['L_bedplate'])
        H_bedplate = float(inputs['H_bedplate'])
        D_nose     = float(inputs['D_nose'])
        D_top      = float(inputs['D_top'])
        t          = inputs['wall_thickness']
        rho        = float(inputs['rho'])
        E          = float(inputs['E'])
        G          = float(inputs['G'])
        sigma_y    = float(inputs['sigma_y'])
        gamma_f    = float(inputs['gamma_f'])
        gamma_m    = float(inputs['gamma_m'])
        gamma_n    = float(inputs['gamma_n'])
        F_stator   = inputs['F']
        M_stator   = inputs['M']

        # Define reference/centroidal axis
        rad = np.linspace(0, 0.5*np.pi, n_points)
        x_c = L_bedplate*np.cos(rad)
        z_c = H_bedplate*np.sin(rad)
        R_c = np.sqrt(x_c**2 + z_c**2)

        # Eccentricity of the centroidal ellipse
        ecc = np.sqrt(1 - (H_bedplate/L_bedplate)**2)
        
        # Points on the outermost ellipse
        x_outer = (L_bedplate+0.5*D_top)*np.cos(rad)
        z_outer = (H_bedplate+0.5*D_nose)*np.sin(rad)
        Ro      = np.sqrt(x_outer**2 + z_outer**2)

        # Points on the innermost ellipse
        x_inner = (L_bedplate-0.5*D_top)*np.cos(rad)
        z_inner = (H_bedplate-0.5*D_nose)*np.sin(rad)
        Ri      = np.sqrt(x_inner**2 + z_inner**2)

        # Outer radius of the ring
        r_outer = np.sqrt( (z_outer-z_c)**2 + (x_outer-x_c)**2 )

        # Inner radius of the ring
        r_inner = r_outer - t

        # Area of each ring and shear area (Frame3DD documentation)
        bedcyl = tube.Tube(2*r_outer, t)
        Ax,_   = nodal2sectional( bedcyl.Area )
            
        # Radius of the neutral axis
        R_n = 2*np.pi*( np.sqrt(R_c**2 - r_inner**2) - np.sqrt(R_c**2 - r_outer**2) )

        # Neutral and centroidal axis distance
        e_cn = R_c - R_n
            
        rad2 = np.arctan( (L_bedplate + 0.5*D_top) / (H_bedplate + 0.5*D_nose) * np.tan(np.diff(rad)) )
            
        # arc length
        arc = L_bedplate*np.abs(ellipeinc(rad2,ecc))    # Volume of each section Legendre complete elliptic integral of the second kind

        # Mass and MoI properties
        mass = Ax * arc * rho

        # ------- node data ----------------
        n = len(x_c)
        inode = np.arange(1, n+1)
        y     = r = np.zeros(n)
        nodes = frame3dd.NodeData(inode, x_c, y, z_c, r)
        
        # ------ reaction data ------------
        # Rigid base
        rnode = [int(inode[0])]
        rk = np.array([RIGID])
        reactions = frame3dd.ReactionData(rnode, rk, rk, rk, rk, rk, rk, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        ielement = np.arange(1, n)
        N1       = np.arange(1, n)
        N2       = np.arange(2, n+1)
        roll     = np.zeros(n-1)
        myones   = np.ones(n-1)
        elements = frame3dd.ElementData(ielement, N1, N2, Ax, bedcyl.Asx, bedcyl.Asy, bedcyl.J0, bedcyl.Jxx, bedcyl.Jyy,
                                        E*myones, G*myones, roll, rho*myones)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        Curved_beam = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add extra mass ------------
        # TODO: Stator mass
        # extra node inertia data
        #Curved_beam.changeExtraNodeMass([inodes[-1]], m_stator, I_stator[0], I_stator[1], I_stator[2], I_stator[3], I_stator[4], I_stator[5],
        #                                cm_stator[0], cm_stator[1], cm_stator[2], True)
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        #Curved_beam.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load case 1 ------------
        # gravity in the X, Y, Z, directions (global)
        gx = gy = 0.0
        gz = -gravity
        load = frame3dd.StaticLoadCase(gx, gy, gz)

        # point loads
        load.changePointLoads([inode[-1]], F_stator[0], F_stator[1], F_stator[2], M_stator[0], M_stator[1], M_stator[2])

        
        # -----------------------------------
        # Put all together and run
        Curved_beam.addLoadCase(load)
        #Curved_beam.write('Curved_beam.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = Curved_beam.run()
        iCase = 0

        # mass
        outputs['mass'] = mass.sum()
        x_c_sec,_ = nodal2sectional( x_c )
        z_c_sec,_ = nodal2sectional( z_c )
        outputs['center_of_mass'] = np.array([np.sum(mass*x_c_sec), 0.0, np.sum(mass*z_c_sec)]) / mass.sum()

        # natural frequncies
        #outputs['f1'] = modal.freq[0]
        #outputs['f2'] = modal.freq[1]

        # deflections due to loading (from Curved_beam top and wind/wave loads)
        outputs['top_deflection'] = np.sqrt(displacements.dx[iCase, n-1]**2 +
                                            displacements.dy[iCase, n-1]**2 +
                                            displacements.dz[iCase, n-1]**2)

        # shear and bending, one per element (convert from local to global c.s.)
        Fz = forces.Nx[iCase, 1::2]
        Vy = forces.Vy[iCase, 1::2]
        Vx = -forces.Vz[iCase, 1::2]
        F   = np.sqrt(Vx**2 + Vy**2)

        Mzz = forces.Txx[iCase, 1::2]
        Myy = forces.Myy[iCase, 1::2]
        Mxx = -forces.Mzz[iCase, 1::2]
        M   = np.sqrt(Myy**2 + Mxx**2)

        # Record total forces and moments
        outputs['base_F'] = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        outputs['base_M'] = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])

        As,_ = nodal2sectional( bedcyl.Asy )
        S,_  = nodal2sectional( bedcyl.S )
        C,_  = nodal2sectional( bedcyl.C )
        outputs['axial_stress'] = Fz/Ax + M/S
        outputs['shear_stress'] = 2.0*F/As + Mzz/C
        
        Bending_stress_outer = M * nodal2sectional( (Ro-R_n) / (bedcyl.Area*e_cn*Ro) )[0]
        Bending_stress_inner = M * nodal2sectional( (R_n-Ri) / (bedcyl.Area*e_cn*Ri) )[0]
        outputs['bending_stress'] = Bending_stress_outer
        
        outputs['constr_vonmises'] = Util.vonMisesStressUtilization(outputs['axial_stress'], Bending_stress_outer, outputs['shear_stress'],
                                                                    gamma_f*gamma_m*gamma_n, sigma_y)

        # Geometry outputs
        outputs['x_c'] = x_c
        outputs['z_c'] = z_c
        outputs['x_inner'] = x_inner
        outputs['z_inner'] = z_inner
        outputs['x_outer'] = x_outer
        outputs['z_outer'] = z_outer


        
if __name__ == '__main__':

        
    
    n_points = 18

    # --- geometry ----
    
    prob = om.Problem()
    prob.model.add_subsystem('bed', Curved_Bedplate(n_points = n_points), promotes=['*'])
    ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
    ivc.add_output('L_bedplate',0.0, units='m')
    ivc.add_output('H_bedplate', 0.0, units='m')
    ivc.add_output('D_top',0.0, units='m')
    ivc.add_output('D_nose',0.0, units='m')
    ivc.add_output('wall_thickness', np.zeros(n_points), units='m')

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
    prob.model.add_design_var('wall_thickness', lower=0.05, upper=0.09 )

    prob.setup()

    # # --- loading case 1: max Thrust ---
    prob['F'] = np.array([2409.750e3, -1716.429e3, 74.3529e3])
    prob['M'] = np.array([-1.83291e7, 6171.7324e3, 5785.82946e3])
    # # ---------------
    
    prob['L_bedplate'] = 5
    prob['H_bedplate'] = 4.875
    prob['D_top'] = 6.5
    prob['D_nose'] = 2.2
    prob['wall_thickness'] = 0.06*np.ones(n_points)
    
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
    print('mass (kg) =', prob['mass'])
    print('cg (m) =', prob['center_of_mass'])
    print('wall_thickness (m)=',prob['wall_thickness'])
    print('top_deflection1 (m) =', prob['top_deflection'])
    print('Reaction forces F =', prob['base_F'] )
    print('Reaction Moments M =', prob['base_M'] )
    print('Axial stresses =', prob['axial_stress'])
    print('Bending stresses =', prob['bending_stress'])
    print('Shear stresses =', prob['shear_stress'])
    print('Safety factor limit =', prob['constr_vonmises'])
   
    
