#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd

# 0 ------- node data ----------------
inode = np.array([1, 2, 3, 4, 5])
x = np.array([0.0, -1200, 1200, 1200, -1200])
y = np.array([0.0, -900, -900, 900, 900])
z = np.array([1000.0, 0.0, 0.0, 0.0, 0.0])
r = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

nodes = pyframe3dd.NodeData(inode, x, y, z, r)
# 0 -----------------------------------

# 1 ------ reaction data ------------
rnode = np.array([2, 3, 4, 5])
Rx = np.ones(4)
Ry = np.ones(4)
Rz = np.ones(4)
Rxx = np.ones(4)
Ryy = np.ones(4)
Rzz = np.ones(4)

reactions = pyframe3dd.ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=1)
# 1 -----------------------------------

# 2 ------ frame element data ------------
ielement = np.array([1, 2, 3, 4])
N1 = np.array([2, 1, 1, 5])
N2 = np.array([1, 3, 4, 1])
Ax = 36.0 * np.ones(4)
Asy = 20.0 * np.ones(4)
Asz = 20.0 * np.ones(4)
Jx = 1000.0 * np.ones(4)
Iy = 492.0 * np.ones(4)
Iz = 492.0 * np.ones(4)
E = 200000.0 * np.ones(4)
G = 79300.0 * np.ones(4)
roll = np.zeros(4)
density = 7.85e-9 * np.ones(4)


elements = pyframe3dd.ElementData(ielement, N1, N2, Ax, Asy, Asz, Jx, Iy, Iz, E, G, roll, density)
# 2 -----------------------------------


# 3 ------ other data ------------
shear = True  # 1: include shear deformation
geom = True  # 1: include geometric stiffness
dx = 20.0  # x-axis increment for internal forces
other = pyframe3dd.Options(shear, geom, dx)
# 3 -----------------------------------


# 4 ------ initialize frame3dd object
frame = pyframe3dd.Frame(nodes, reactions, elements, other)
# 4 -----------------------------------


# 5 ------ static load case 1 ------------
# gravity in the X, Y, Z, directions (global)
gx = 0.0
gy = 0.0
gz = -9806.33
load = pyframe3dd.StaticLoadCase(gx, gy, gz)

# point load
nF = np.array([1])
Fx = np.array([100.0])
Fy = np.array([-200.0])
Fz = np.array([-100.0])
Mxx = np.array([0.0])
Myy = np.array([0.0])
Mzz = np.array([0.0])
load.changePointLoads(nF, Fx, Fy, Fz, Mxx, Myy, Mzz)

frame.addLoadCase(load)
# 5 -----------------------------------


# 6 ------ static load case 2 ------------
# gravity in the X, Y, Z, directions (global)
gx = 0.0
gy = 0.0
gz = -9806.33
load = pyframe3dd.StaticLoadCase(gx, gy, gz)

# uniform loads
EL = np.array([2, 1])
Ux = np.array([0.0, 0.0])
Uy = np.array([0.1, 0.0])
Uz = np.array([0.0, 0.1])
load.changeUniformLoads(EL, Ux, Uy, Uz)

# trapezoidally distributed loads
EL = np.array([3, 4])
xx1 = np.array([20.0, 0.0])
xx2 = np.array([80.0, 0.0])
wx1 = np.array([0.01, 0.0])
wx2 = np.array([0.05, 0.0])
xy1 = np.array([0.0, 68.0])
xy2 = np.array([0.0, 330.0])
wy1 = np.array([0.0, 0.05])
wy2 = np.array([0.0, 0.0])
xz1 = np.array([80.0, 80.0])
xz2 = np.array([830.0, 830.0])
wz1 = np.array([-0.05, -0.05])
wz2 = np.array([0.07, 0.07])
load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

EL = np.array([1])
a = np.array([12e-6])
hy = np.array([10.0])
hz = np.array([10.0])
Typ = np.array([20.0])
Tym = np.array([10.0])
Tzp = np.array([10.0])
Tzm = np.array([-10.0])
load.changeTemperatureLoads(EL, a, hy, hz, Typ, Tym, Tzp, Tzm)

frame.addLoadCase(load)
# 6 -----------------------------------


# 7 ------ static load case 3 ------------
gx = 0.0
gy = 0.0
gz = -9806.33
load = pyframe3dd.StaticLoadCase(gx, gy, gz)

# concentrated interior point loads
EL = np.array([1, 2])
Px = np.array([0.0, 0.0])
Py = np.array([100.0, -200.0])
Pz = np.array([-900.0, 200.0])
x = np.array([600.0, 800.0])
load.changeElementLoads(EL, Px, Py, Pz, x)

frame.addLoadCase(load)
# 7 -----------------------------------


# 8 ------ dyamic analysis data ------------
nM = 6  # number of desired dynamic modes of vibration
Mmethod = 1  # 1: subspace Jacobi     2: Stodola
lump = 0  # 0: consistent mass ... 1: lumped mass matrix
tol = 1e-9  # mode shape tolerance
shift = 0.0  # shift value ... for unrestrained structures
frame.enableDynamics(nM, Mmethod, lump, tol, shift)
# 8 -----------------------------------

# 9 ---- extra node inertia data
N = np.array([1])
EMs = np.array([0.1])
EMx = np.array([0.0])
EMy = np.array([0.0])
EMz = np.array([0.0])

# frame.changeExtraInertia(N, EMs, EMx, EMy, EMz)
frame.changeExtraNodeMass(N, EMs, EMx, EMy, EMz, [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], False)
# 9 ------------------------------------

# 10 ----- run the analysis
displacements, forces, reactions, internalForces, mass, modal = frame.run()
# 10 ------------------------------------

# 11 ----- analysis output
nC = len(frame.loadCases)  # number of load cases
nN = len(nodes.node)  # number of nodes
nE = len(elements.element)  # number of elements

# mass data
print("total_mass =", mass.total_mass)
print("struct_mass =", mass.struct_mass)
print("node =", mass.node)
print("xmass =", mass.xmass)
print("ymass =", mass.ymass)
print("zmass =", mass.zmass)
print("xinrta =", mass.xinrta)
print("yinrta =", mass.yinrta)
print("zinrta =", mass.zinrta)
print()
print()

# node displacements
for iCase in range(nC):

    print("case_idx:", iCase)

    print("node:", displacements.node[iCase, :])
    print("dx:", displacements.dx[iCase, :])
    print("dy:", displacements.dy[iCase, :])
    print("dz:", displacements.dz[iCase, :])
    print("dxrot:", displacements.dxrot[iCase, :])
    print("dyrot:", displacements.dyrot[iCase, :])
    print("dzrot:", displacements.dzrot[iCase, :])
    print()
    print("element =", forces.element[iCase, :])
    print("node =", forces.node[iCase, :])
    print("Nx =", forces.Nx[iCase, :])
    print("Vy =", forces.Vy[iCase, :])
    print("Vz =", forces.Vz[iCase, :])
    print("Txx =", forces.Txx[iCase, :])
    print("Myy =", forces.Myy[iCase, :])
    print("Mzz =", forces.Mzz[iCase, :])
    print()
    print("nodesR =", reactions.node[iCase, :])
    print("RFx =", reactions.Fx[iCase, :])
    print("RFy =", reactions.Fy[iCase, :])
    print("RFz =", reactions.Fz[iCase, :])
    print("RMxx =", reactions.Mxx[iCase, :])
    print("RMyy =", reactions.Myy[iCase, :])
    print("RMzz =", reactions.Mzz[iCase, :])
    print()

print()
print()

# internal forces
iE = 3  # note just showing for one element
for iCase in range(nC):

    print("case_idx:", iCase)
    print("element_idx:", iE)

    print("x =", internalForces[iE].x[iCase, :])
    print("Nx =", internalForces[iE].Nx[iCase, :])
    print("Vy =", internalForces[iE].Vy[iCase, :])
    print("Vz =", internalForces[iE].Vz[iCase, :])
    print("Tx =", internalForces[iE].Tx[iCase, :])
    print("My =", internalForces[iE].My[iCase, :])
    print("Mz =", internalForces[iE].Mz[iCase, :])
    print("Dx =", internalForces[iE].Dx[iCase, :])
    print("Dy =", internalForces[iE].Dy[iCase, :])
    print("Dz =", internalForces[iE].Dz[iCase, :])
    print("Rx =", internalForces[iE].Rx[iCase, :])
    print()

print()
print()

# Modal data
for iM in range(nM):

    print("mode_idx", iM)

    print("freq =", modal.freq[iM])
    print("xmpf =", modal.xmpf[iM])
    print("ympf =", modal.ympf[iM])
    print("zmpf =", modal.zmpf[iM])
    print("node =", modal.node[iM, :])
    print("xdsp =", modal.xdsp[iM, :])
    print("ydsp =", modal.ydsp[iM, :])
    print("zdsp =", modal.zdsp[iM, :])
    print("xrot =", modal.xrot[iM, :])
    print("yrot =", modal.yrot[iM, :])
    print("zrot =", modal.zrot[iM, :])
    print()
# 11 ------------------------------------

# 12 ------------------------------------
# frame.write("pyramid_frame_B.3dd")
# 12 ------------------------------------
