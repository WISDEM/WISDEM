#!/usr/bin/env python
# encoding: utf-8
"""
WindWaveDrag.py

Created by RRD on 2015-07-13.
Copyright (c) NREL. All rights reserved.
"""

#-------------------------------------------------------------------------------
# Name:        WindWaveDrag.py
# Purpose:     It contains OpenMDAO's Components to calculate wind or wave drag
#              on cylinders.
#
# Author:      ANing/RRD
#
# Created:     13/07/2015 - It is based on load function calculations developed for tower and jacket.
#                             Reestablished elements needed by jacketSE that were removed. Changed names to vartrees.
# Copyright:   (c) rdamiani 2015
# Licence:     <Apache 2015>
#-------------------------------------------------------------------------------
from __future__ import print_function
import math
import numpy as np

from openmdao.api import ExplicitComponent, Problem, Group
from wisdem.commonse.utilities import sind, cosd  # , linspace_with_deriv, interp_with_deriv, hstack, vstack
from wisdem.commonse.csystem import DirectionVector

from wisdem.commonse.akima import Akima

#TODO CHECK

# -----------------
#  Helper Functions
# -----------------
# "Experiments on the Flow Past a Circular Cylinder at Very High Reynolds Numbers", Roshko
Re_pt = [0.00001, 0.0001, 0.0010, 0.0100, 0.0200, 0.1220, 0.2000, 0.3000, 0.4000,
         0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 5.0000, 10.0000]
cd_pt = [4.0000,  2.0000, 1.1100, 1.1100, 1.2000, 1.2000, 1.1700, 0.9000, 0.5400,
         0.3100, 0.3800, 0.4600, 0.5300, 0.5700, 0.6100, 0.6400, 0.6700, 0.7000, 0.7000]

drag_spline = Akima(np.log10(Re_pt), cd_pt, delta_x=0.0)  # exact akima because control points do not change

def cylinderDrag(Re):
    """Drag coefficient for a smooth circular cylinder.

    Parameters
    ----------
    Re : array_like
        Reynolds number

    Returns
    -------
    cd : array_like
        drag coefficient (normalized by cylinder diameter)

    """

    ReN = Re / 1.0e6

    cd = np.zeros_like(Re)
    dcd_dRe = np.zeros_like(Re)
    idx = ReN > 0
    cd[idx], dcd_dRe[idx], _, _ = drag_spline.interp(np.log10(ReN[idx]))
    dcd_dRe[idx] /= (Re[idx]*math.log(10))  # chain rule

    return cd, dcd_dRe

# -----------------
#  Components
# -----------------

class AeroHydroLoads(ExplicitComponent):
    def initialize(self):
        self.options.declare('nPoints')
        
    def setup(self):
        nPoints = self.options['nPoints']

        #inputs

        self.add_input('windLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_input('windLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_input('windLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_input('windLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_input('windLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_input('windLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_input('windLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')
        #self.add_input('windLoads_Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_input('windLoads_Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_input('windLoads_Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_input('windLoads_qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        #self.add_input('windLoads_beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')


        self.add_input('waveLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_input('waveLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_input('waveLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_input('waveLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_input('waveLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_input('waveLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_input('waveLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')
        #self.add_input('waveLoads_Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_input('waveLoads_Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_input('waveLoads_Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_input('waveLoads_qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        #self.add_input('waveLoads_beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')

        self.add_input('z', np.zeros(nPoints), units='m', desc='locations along cylinder')
        self.add_input('yaw', 0.0, units='deg', desc='yaw angle')

        #outputs
        self.add_output('Px', np.zeros(nPoints), units='N/m', desc='force per unit length in x-direction')
        self.add_output('Py', np.zeros(nPoints), units='N/m', desc='force per unit length in y-direction')
        self.add_output('Pz', np.zeros(nPoints), units='N/m', desc='force per unit length in z-direction')
        self.add_output('qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')

    def compute(self, inputs, outputs):
        # aero/hydro loads
        # wind = inputs['windLoads']
        # wave = inputs['waveLoads']
        # outloads = inputs['outloads']
        z = inputs['z']
        hubHt = z[-1]  # top of cylinder
        windLoads = DirectionVector(inputs['windLoads_Px'], inputs['windLoads_Py'], inputs['windLoads_Pz']).inertialToWind(inputs['windLoads_beta']).windToYaw(inputs['yaw'])
        waveLoads = DirectionVector(inputs['waveLoads_Px'], inputs['waveLoads_Py'], inputs['waveLoads_Pz']).inertialToWind(inputs['waveLoads_beta']).windToYaw(inputs['yaw'])

        Px = np.interp(z, inputs['windLoads_z'], windLoads.x) + np.interp(z, inputs['waveLoads_z'], waveLoads.x)
        Py = np.interp(z, inputs['windLoads_z'], windLoads.y) + np.interp(z, inputs['waveLoads_z'], waveLoads.y)
        Pz = np.interp(z, inputs['windLoads_z'], windLoads.z) + np.interp(z, inputs['waveLoads_z'], waveLoads.z)
        qdyn = np.interp(z, inputs['windLoads_z'], inputs['windLoads_qdyn']) + np.interp(z, inputs['waveLoads_z'], inputs['waveLoads_qdyn'])
        # outloads.z = z

        #The following are redundant, at one point we will consolidate them to something that works for both cylinder (not using vartrees) and jacket (still using vartrees)
        outputs['Px'] = Px
        outputs['Py'] = Py
        outputs['Pz'] = Pz
        outputs['qdyn'] = qdyn

# -----------------

class CylinderWindDrag(ExplicitComponent):
    """drag forces on a cylindrical cylinder due to wind"""

    def initialize(self):
        self.options.declare('nPoints')
        
    def setup(self):
        nPoints = self.options['nPoints']

        # variables
        self.add_input('U', np.zeros(nPoints), units='m/s', desc='magnitude of wind speed')
        self.add_input('z', np.zeros(nPoints), units='m', desc='heights where wind speed was computed')
        self.add_input('d', np.zeros(nPoints), units='m', desc='corresponding diameter of cylinder section')

        # parameters
        self.add_input('beta_wind', 0.0, units='deg', desc='corresponding wind angles relative to inertial coordinate system')
        self.add_input('rho_air', 0.0, units='kg/m**3', desc='air density')
        self.add_input('mu_air', 0.0, units='kg/(m*s)', desc='dynamic viscosity of air')
        #TODO not sure what to do here?
        self.add_input('cd_usr', -1., desc='User input drag coefficient to override Reynolds number based one')

        # out
        self.add_output('windLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_output('windLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_output('windLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_output('windLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_output('windLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_output('windLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_output('windLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')
        #self.add_output('windLoads_Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_output('windLoads_Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_output('windLoads_Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_output('windLoads_qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        #self.add_output('windLoads_beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        rho = inputs['rho_air']
        U = inputs['U']
        d = inputs['d']
        mu = inputs['mu_air']
        beta = inputs['beta_wind']

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if float(inputs['cd_usr']) < 0.:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)
        else:
            cd = inputs['cd_usr']
            Re = 1.0
            dcd_dRe = 0.0
        Fp = q*cd*d

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0*Fp

        # pack data
        outputs['windLoads_Px'] = Px
        outputs['windLoads_Py'] = Py
        outputs['windLoads_Pz'] = Pz
        outputs['windLoads_qdyn'] = q
        outputs['windLoads_z'] = inputs['z']
        outputs['windLoads_beta'] = beta


    def compute_partials(self, inputs, J):

        # rename
        rho = inputs['rho_air']
        U = inputs['U']
        d = inputs['d']
        mu = inputs['mu_air']
        beta = inputs['beta_wind']

        # dynamic pressure
        q = 0.5*rho*U**2

        # Reynolds number and drag
        if float(inputs['cd_usr']) < 0.:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)
        else:
            cd = inputs['cd_usr']
            Re = 1.0
            dcd_dRe = 0.0

        # derivatives
        dq_dU = rho*U
        const = (dq_dU*cd + q*dcd_dRe*rho*d/mu)*d
        dPx_dU = const*cosd(beta)
        dPy_dU = const*sind(beta)

        const = (cd + dcd_dRe*Re)*q
        dPx_dd = const*cosd(beta)
        dPy_dd = const*sind(beta)

        n = len(inputs['z'])

        zeron = np.zeros((n, n))
        
        J['windLoads_Px', 'U'] = np.diag(dPx_dU)
        J['windLoads_Px', 'z'] = zeron
        J['windLoads_Px', 'd'] = np.diag(dPx_dd)

        J['windLoads_Py', 'U'] = np.diag(dPy_dU)
        J['windLoads_Py', 'z'] = zeron
        J['windLoads_Py', 'd'] = np.diag(dPy_dd)

        J['windLoads_Pz', 'U'] = zeron
        J['windLoads_Pz', 'z'] = zeron
        J['windLoads_Pz', 'd'] = zeron

        J['windLoads_qdyn', 'U'] = np.diag(dq_dU)
        J['windLoads_qdyn', 'z'] = zeron
        J['windLoads_qdyn', 'd'] = zeron

        J['windLoads_z', 'U'] = zeron
        J['windLoads_z', 'z'] = np.eye(n)
        J['windLoads_z', 'd'] = zeron

        

# -----------------

class CylinderWaveDrag(ExplicitComponent):
    """drag forces on a cylindrical cylinder due to waves"""


    def initialize(self):
        self.options.declare('nPoints')
        
    def setup(self):
        nPoints = self.options['nPoints']

        # variables
        self.add_input('U', np.zeros(nPoints), units='m/s', desc='magnitude of wave speed')
        #self.add_input('U0', 0.0, units='m/s', desc='magnitude of wave speed at z=0 MSL')
        self.add_input('A', np.zeros(nPoints), units='m/s**2', desc='magnitude of wave acceleration')
        self.add_input('p', np.zeros(nPoints), units='N/m**2', desc='pressure oscillation')
        #self.add_input('A0', 0.0, units='m/s**2', desc='magnitude of wave acceleration at z=0 MSL')
        self.add_input('z', np.zeros(nPoints), units='m', desc='heights where wave speed was computed')
        self.add_input('d', np.zeros(nPoints), units='m', desc='corresponding diameter of cylinder section')

        # parameters
        #self.add_input('wlevel', 0.0, units='m', desc='Water Level, to assess z w.r.t. MSL')
        self.add_input('beta_wave', 0.0, units='deg', desc='corresponding wave angles relative to inertial coordinate system')
        #self.add_input('beta0', 0.0, units='deg', desc='corresponding wave angles relative to inertial coordinate system at z=0 MSL')
        self.add_input('rho_water', 0.0, units='kg/m**3', desc='water density')
        self.add_input('mu_water', 0.0, units='kg/(m*s)', desc='dynamic viscosity of water')
        self.add_input('cm', 0.0, desc='mass coefficient')
        #TODO not sure what to do here?
        self.add_input('cd_usr', -1., desc='User input drag coefficient to override Reynolds number based one')

        # out
        self.add_output('waveLoads_Px', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in x-direction')
        self.add_output('waveLoads_Py', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in y-direction')
        self.add_output('waveLoads_Pz', np.zeros(nPoints), units='N/m', desc='distributed loads, force per unit length in z-direction')
        self.add_output('waveLoads_qdyn', np.zeros(nPoints), units='N/m**2', desc='dynamic pressure')
        self.add_output('waveLoads_pt', np.zeros(nPoints), units='N/m**2', desc='total (static+dynamic) pressure')
        self.add_output('waveLoads_z', np.zeros(nPoints), units='m', desc='corresponding heights')
        self.add_output('waveLoads_d', np.zeros(nPoints), units='m', desc='corresponding diameters')
        self.add_output('waveLoads_beta', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')
        #self.add_output('waveLoads_Px0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_output('waveLoads_Py0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_output('waveLoads_Pz0', 0.0, units='N/m', desc='Distributed load at z=0 MSL')
        #self.add_output('waveLoads_qdyn0', 0.0, units='N/m**2', desc='dynamic pressure at z=0 MSL')
        #self.add_output('waveLoads_beta0', 0.0, units='deg', desc='wind/wave angle relative to inertia c.s.')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        #wlevel = inputs['wlevel']
        #if wlevel > 0.0: wlevel *= -1.0
        
        rho = inputs['rho_water']
        U = inputs['U']
        #U0 = inputs['U0']
        d = inputs['d']
        #zrel= inputs['z']-wlevel
        mu = inputs['mu_water']
        beta = inputs['beta_wave']
        #beta0 = inputs['beta0']

        # dynamic pressure
        q = 0.5*rho*U**2
        #q0= 0.5*rho*U0**2

        # Reynolds number and drag
        if float(inputs['cd_usr']) < 0.:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)
        else:
            cd = inputs['cd_usr']*np.ones_like(d)
            Re = 1.0
            dcd_dRe = 0.0

        # inertial and drag forces
        Fi = rho*inputs['cm']*math.pi/4.0*d**2*inputs['A']  # Morrison's equation
        Fd = q*cd*d
        Fp = Fi + Fd

        # components of distributed loads
        Px = Fp*cosd(beta)
        Py = Fp*sind(beta)
        Pz = 0.*Fp

        #FORCES [N/m] AT z=0 m
        #idx0 = np.abs(zrel).argmin()  # closest index to z=0, used to find d at z=0
        #d0 = d[idx0]  # initialize
        #cd0 = cd[idx0]  # initialize
        #if (zrel[idx0]<0.) and (idx0< (zrel.size-1)):       # point below water
        #    d0 = np.mean(d[idx0:idx0+2])
        #    cd0 = np.mean(cd[idx0:idx0+2])
        #elif (zrel[idx0]>0.) and (idx0>0):     # point above water
        #    d0 = np.mean(d[idx0-1:idx0+1])
        #    cd0 = np.mean(cd[idx0-1:idx0+1])
        #Fi0 = rho*inputs['cm']*math.pi/4.0*d0**2*inputs['A0']  # Morrison's equation
        #Fd0 = q0*cd0*d0
        #Fp0 = Fi0 + Fd0

        #Px0 = Fp0*cosd(beta0)
        #Py0 = Fp0*sind(beta0)
        #Pz0 = 0.*Fp0

        #Store qties at z=0 MSL
        #outputs['waveLoads_Px0'] = Px0
        #outputs['waveLoads_Py0'] = Py0
        #outputs['waveLoads_Pz0'] = Pz0
        #outputs['waveLoads_qdyn0'] = q0
        #outputs['waveLoads_beta0'] = beta0

        # pack data
        outputs['waveLoads_Px'] = Px
        outputs['waveLoads_Py'] = Py
        outputs['waveLoads_Pz'] = Pz
        outputs['waveLoads_qdyn'] = q
        outputs['waveLoads_pt'] = q + inputs['p']
        outputs['waveLoads_z'] = inputs['z']
        outputs['waveLoads_beta'] = beta
        outputs['waveLoads_d'] = d


    def compute_partials(self, inputs, J):

        #wlevel = inputs['wlevel']
        #if wlevel > 0.0: wlevel *= -1.0
        
        rho = inputs['rho_water']
        U = inputs['U']
        #U0 = inputs['U0']
        d = inputs['d']
        #zrel= inputs['z']-wlevel
        mu = inputs['mu_water']
        beta = inputs['beta_wave']
        #beta0 = inputs['beta0']

        # dynamic pressure
        q = 0.5*rho*U**2
        #q0= 0.5*rho*U0**2

        # Reynolds number and drag
        if float(inputs['cd_usr']) < 0.:
            cd = inputs['cd_usr']*np.ones_like(d)
            Re = 1.0
            dcd_dRe = 0.0
        else:
            Re = rho*U*d/mu
            cd, dcd_dRe = cylinderDrag(Re)

        # derivatives
        dq_dU = rho*U
        const = (dq_dU*cd + q*dcd_dRe*rho*d/mu)*d
        dPx_dU = const*cosd(beta)
        dPy_dU = const*sind(beta)

        const = (cd + dcd_dRe*Re)*q + rho*inputs['cm']*math.pi/4.0*2*d*inputs['A']
        dPx_dd = const*cosd(beta)
        dPy_dd = const*sind(beta)

        const = rho*inputs['cm']*math.pi/4.0*d**2
        dPx_dA = const*cosd(beta)
        dPy_dA = const*sind(beta)

        n = len(inputs['z'])

        zeron = np.zeros((n, n))
        
        J['waveLoads.Px', 'U'] = np.diag(dPx_dU)
        J['waveLoads.Px', 'A'] = np.diag(dPx_dA)
        J['waveLoads.Px', 'z'] = zeron
        J['waveLoads.Px', 'd'] = np.diag(dPx_dd)
        J['waveLoads.Px', 'p'] = zeron

        J['waveLoads.Py', 'U'] = np.diag(dPy_dU)
        J['waveLoads.Py', 'A'] = np.diag(dPy_dA)
        J['waveLoads.Py', 'z'] = zeron
        J['waveLoads.Py', 'd'] = np.diag(dPy_dd)
        J['waveLoads.Py', 'p'] = zeron

        J['waveLoads.Pz', 'U'] = zeron
        J['waveLoads.Pz', 'A'] = zeron
        J['waveLoads.Pz', 'z'] = zeron
        J['waveLoads.Pz', 'd'] = zeron
        J['waveLoads.Pz', 'p'] = zeron

        J['waveLoads.qdyn', 'U'] = np.diag(dq_dU)
        J['waveLoads.qdyn', 'A'] = zeron
        J['waveLoads.qdyn', 'z'] = zeron
        J['waveLoads.qdyn', 'd'] = zeron
        J['waveLoads.qdyn', 'p'] = zeron

        J['waveLoads.pt', 'U'] = np.diag(dq_dU)
        J['waveLoads.pt', 'A'] = zeron
        J['waveLoads.pt', 'z'] = zeron
        J['waveLoads.pt', 'd'] = zeron
        J['waveLoads.pt', 'p'] = 1.0

        J['waveLoads.z', 'U'] = zeron
        J['waveLoads.z', 'A'] = zeron
        J['waveLoads.z', 'z'] = np.eye(n)
        J['waveLoads.z', 'd'] = zeron
        J['waveLoads.z', 'p'] = zeron

        

#___________________________________________#

def main():
    # initialize problem
    U = np.array([20., 25., 30.])
    z = np.array([10., 30., 80.])
    d = np.array([5.5, 4., 3.])

    beta = np.array([45., 45., 45.])
    rho = 1.225
    mu = 1.7934e-5
    #cd_usr = 0.7

    nPoints = len(z)


    prob = Problem()

    root = prob.model = Group()

    root.add('p1', CylinderWindDrag(nPoints))

    prob.setup()

    prob['p1.U'] = U
    prob['p1.z'] = z
    prob['p1.d'] = d
    prob['p1.beta'] = beta
    prob['p1.rho'] = rho
    prob['p1.mu'] = mu
    #prob['p1.cd_usr'] = cd_usr

    #run
    prob.run_once()

    # out
    Re = prob['p1.rho']*prob['p1.U']*prob['p1.d']/prob['p1.mu']
    cd, dcd_dRe = cylinderDrag(Re)
    print(cd)
    import matplotlib.pyplot as plt

    plt.plot(prob['p1.windLoads_Px'], prob['p1.windLoads_z'])
    plt.plot(prob['p1.windLoads_Py'], prob['p1.windLoads_z'])
    plt.plot(prob['p1.windLoads_qdyn'], prob['p1.windLoads_z'])
    plt.show()

if __name__ == '__main__':
    main()
