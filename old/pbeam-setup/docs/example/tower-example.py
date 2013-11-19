#!/usr/bin/env python
# encoding: utf-8
"""
tower-example.py

Created by Andrew Ning on 2013-04-30.
Copyright (c) NREL. All rights reserved.
"""

# 1 ---------

import numpy as np
from math import pi
import matplotlib.pyplot as plt

import _pBEAM

# 1 ---------


# 2 ---------

# tower geometry defintion
z0 = [0, 30.0, 73.8, 117.6]             # heights starting from bottom (m)
d0 = [6.0, 6.0, 4.935, 3.87]            # corresponding diameters (m)
t0 = [0.0351, 0.0351, 0.0299, 0.0247]   # corresponding shell thicknesses (m)
n0 = [5, 5, 5]                          # number of finite elements per section

# 2 ---------

# 3 ---------

# discretize
nodes = int(np.sum(n0)) + 1  # C++ interface requires int

z = np.zeros(nodes)
start = 0
for i in range(len(n0)):
    z[start:start+n0[i]+1] = np.linspace(z0[i], z0[i+1], n0[i]+1)
    start += n0[i]

d = np.interp(z, z0, d0)
t = np.interp(z, z0, t0)

# 3 ---------


# 4 ---------
# material properties

E = 210e9               # elastic modulus (Pa)
G = 80.8e9              # shear modulus (Pa)
rho = 8500.0            # material density (kg/m^3)

material = _pBEAM.Material(E, G, rho)
# 4 ---------

# 5 ---------
# distributed loads

g = 9.81  # gravity

# wind loading in x-direction
Px = np.array([0.0, 133.18, 167.41, 191.71, 211.09, 227.42, 236.04, 240.30,
               241.36, 239.92, 236.47, 231.37, 224.95, 217.64, 209.33, 200.16])
Py = np.zeros_like(Px)
Pz = -rho*g*(pi*d*t)  # self-weight


loads = _pBEAM.Loads(nodes, Px, Py, Pz)
# 5 ---------

# 6 ---------
# RNA contribution

m = 300000.0  # mass
cm = np.array([-5.0, 0.0, 0.0])  # center of mass relative to tip
I = np.array([2960437.0, 3253223.0, 3264220.0, 0.0, -18400.0, 0.0])  # moments of inertia
F = np.array([750000.0, 0.0, -m*g])  # force
M = np.array([0.0, 0.0, 0.0])  # moment

tip = _pBEAM.TipData(m, cm, I, F, M)
# 6 ---------

# 7 ---------
# rigid base

inf = float('inf')
k = np.array([inf]*6)

base = _pBEAM.BaseData(k, inf)
# 7 ---------

# 8 ---------
# create tower object

tower = _pBEAM.Beam(nodes, z, d, t, loads, material, tip, base)
# 8 ---------

# 9 ---------
# compute mass and natural frequncies
print 'mass =', tower.mass()
print 'natural freq =', tower.naturalFrequencies(5)

# plot displacement in wind direction
disp = tower.displacement()
dx = disp[0]
plt.plot(z, dx)
plt.xlabel('z (m)')
plt.ylabel('$\delta x$ (m)')
plt.show()
# 9 ---------

