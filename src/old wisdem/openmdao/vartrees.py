#!/usr/bin/env python
# encoding: utf-8
"""
vartrees.py

Created by Andrew Ning on 2013-05-29.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.api import VariableTree
from openmdao.main.datatypes.api import Array, Float, Bool


class BladeDistributedLoads(VariableTree):
    """aerodynamic loads along blade"""

    r = Array(units='m', desc='radial locations where loads are defined')
    Px = Array(units='N/m', desc='force per unit length in x-direction')
    Py = Array(units='N/m', desc='force per unit length in y-direction')
    Pz = Array(units='N/m', desc='force per unit length in z-direction')
    pitch = Float(units='deg', desc='blade pitch angle for loading condition')
    azimuth = Float(units='deg', desc='blade azimuth angle for loading condition')


class Forces(VariableTree):
    """point forces/moments"""

    F = Array(units='N', desc='point forces')
    M = Array(units='N*m', desc='point moments')


class WindWaveDistributedLoads(VariableTree):
    """wind and wave loads along tower"""

    z = Array(units='m', desc='vertical locations where wind/wave loads are defined')
    Px = Array(units='N/m', desc='force per unit length in x-direction')
    Py = Array(units='N/m', desc='force per unit length in y-direction')
    Pz = Array(units='N/m', desc='force per unit length in z-direction')
    q = Array(units='N/m**2', desc='dynamic pressure')


class MassProperties(VariableTree):
    """mass and mass moments of inertia of a component"""

    mass = Float(units='kg', desc='mass of object')
    Ixx = Float(units='kg*m**2', desc='mass moment of inertia about x-axis')
    Iyy = Float(units='kg*m**2', desc='mass moment of inertia about y-axis')
    Izz = Float(units='kg*m**2', desc='mass moment of inertia about z-axis')
    Ixy = Float(units='kg*m**2', desc='mass x-y product of inertia')
    Ixz = Float(units='kg*m**2', desc='mass x-z product of inertia')
    Iyz = Float(units='kg*m**2', desc='mass y-z product of inertia')


class ConditionsAtRated(VariableTree):
    """operating conditions at rated speed"""

    V = Float(units='m/s', desc='rated speed')
    Omega = Float(units='rpm', desc='rotor rotation speed at rated')
    T = Float(units='N', desc='rotor thrust at rated')
    Q = Float(units='N*m', desc='rotor torque at rated')



class Atmosphere(VariableTree):

    rho = Float(1.225, iotype='in', units='kg/m**3', desc='density of air')
    mu = Float(1.81206e-5, iotype='in', units='kg/m/s', desc='dynamic viscosity of air')
    shearExp = Float(0.2, iotype='in', desc='shear exponent')


class SoilProperties(VariableTree):

    G = Float(140e6, units='N/m**2', desc='shear modulus of soil')
    nu = Float(0.4, desc='Poissons ratio of soil')
    depth = Float(units='m', desc='depth of soil')
    rigid_x = Bool(True, 'true if base should be considered rigid in the x-direction')
    rigid_y = Bool(True, 'true if base should be considered rigid in the y-direction')
    rigid_z = Bool(True, 'true if base should be considered rigid in the z-direction')
    rigid_theta_x = Bool(True, 'true if base should be considered rigid in the theta_x-direction')
    rigid_theta_y = Bool(True, 'true if base should be considered rigid in the theta_y-direction')
    rigid_theta_z = Bool(True, 'true if base should be considered rigid in the theta_z-direction')

