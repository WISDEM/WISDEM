# just to hide path details from user in docs
import os
basepath = os.path.join(os.path.expanduser('~'), 'Dropbox', 'NREL', '5MW_files', '5MW_AFFiles')
os.chdir(basepath)

# just to temporarily change PYTHONPATH without installing
import sys
sys.path.append(os.path.expanduser('~') + '/Dropbox/NREL/SysEng/TWISTER/src/twister/rotoraero')


# 1 ---------

import numpy as np
from math import pi
import matplotlib.pyplot as plt

from ccblade import CCAirfoil, CCBlade


# geometry
Rhub = 1.5
Rtip = 63.0

r = np.array([2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
              28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
              56.1667, 58.9000, 61.6333])
chord = np.array([3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
                  3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419])
theta = np.array([13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
                  6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106])
B = 3  # number of blades

# atmosphere
rho = 1.225
mu = 1.81206e-5

# 1 ----------


# 2 ----------

afinit = CCAirfoil.initFromAerodynFile  # just for shorthand

# load all airfoils
airfoil_types = [0]*8
airfoil_types[0] = afinit('Cylinder1.dat')
airfoil_types[1] = afinit('Cylinder2.dat')
airfoil_types[2] = afinit('DU40_A17.dat')
airfoil_types[3] = afinit('DU35_A17.dat')
airfoil_types[4] = afinit('DU30_A17.dat')
airfoil_types[5] = afinit('DU25_A17.dat')
airfoil_types[6] = afinit('DU21_A17.dat')
airfoil_types[7] = afinit('NACA64_A17.dat')

# place at appropriate radial stations
af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

af = [0]*len(r)
for i in range(len(r)):
    af[i] = airfoil_types[af_idx[i]]

# 2 ----------

# 3 ----------

# create CCBlade object
rotor = CCBlade(r, chord, theta, af, Rhub, Rtip, B, rho, mu)

# 3 ----------

# 4 ----------

# set conditions
Uinf = 10.0
tsr = 7.55
pitch = 0.0
Omega = Uinf*tsr/Rtip * 30.0/pi  # convert to RPM

# evaluate distributed loads
r, theta, Tp, Np = rotor.distributedAeroLoads(Uinf, Omega, pitch)

# 4 ----------

# 5 ----------

# plot
rstar = (r - Rhub) / (Rtip - Rhub)
plt.plot(rstar, Tp/1e3, 'k', label='lead-lag')
plt.plot(rstar, Np/1e3, 'r', label='flapwise')
plt.xlabel('blade fraction')
plt.ylabel('distributed aerodynamic loads (kN)')
plt.legend(loc='upper left')
plt.show()
# 5 ----------
# plt.savefig('/Users/sning/Dropbox/NREL/SysEng/TWISTER/ccblade-setup/docs/images/distributedAeroLoads.pdf')


# 6 ----------

P, T, Q = rotor.evaluate([Uinf], [Omega], [pitch])

CP, CT, CQ = rotor.evaluate([Uinf], [Omega], [pitch], coefficient=True)

print CP, CT, CQ

# 6 ----------

# 7 ----------

# velocity has a small amount of Reynolds number dependence
tsr = np.linspace(2, 14, 50)
Omega = 10.0 * np.ones_like(tsr)
Uinf = Omega*pi/30.0 * Rtip/tsr
pitch = np.zeros_like(tsr)

CP, CT, CQ = rotor.evaluate(Uinf, Omega, pitch, coefficient=True)

plt.figure()
plt.plot(tsr, CP, 'k')
plt.xlabel('$\lambda$')
plt.ylabel('$c_p$')
plt.show()
# 7 ----------

# plt.savefig('/Users/sning/Dropbox/NREL/SysEng/TWISTER/ccblade-setup/docs/images/cp.pdf')
