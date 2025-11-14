# 1 ---------

from math import pi

import numpy as np
import matplotlib.pyplot as plt

from wisdem.ccblade.ccblade import CCBlade, CCAirfoil

# geometry
Rhub = 1.5
Rtip = 63.0

r = np.array(
    [
        2.8667,
        5.6000,
        8.3333,
        11.7500,
        15.8500,
        19.9500,
        24.0500,
        28.1500,
        32.2500,
        36.3500,
        40.4500,
        44.5500,
        48.6500,
        52.7500,
        56.1667,
        58.9000,
        61.6333,
    ]
)
chord = np.array(
    [
        3.542,
        3.854,
        4.167,
        4.557,
        4.652,
        4.458,
        4.249,
        4.007,
        3.748,
        3.502,
        3.256,
        3.010,
        2.764,
        2.518,
        2.313,
        2.086,
        1.419,
    ]
)
theta = np.array(
    [
        13.308,
        13.308,
        13.308,
        13.308,
        11.480,
        10.162,
        9.011,
        7.795,
        6.544,
        5.361,
        4.188,
        3.125,
        2.319,
        1.526,
        0.863,
        0.370,
        0.106,
    ]
)
B = 3  # number of blades

tilt = 5.0
precone = 2.5
yaw = 0.0

nSector = 8  # azimuthal discretization

# atmosphere
rho = 1.225
mu = 1.81206e-5

# power-law wind shear profile
shearExp = 0.2
hubHt = 90.0

# 1 ----------


# 2 ----------
import os
from wisdem.inputs.validation import load_geometry_yaml
baseyaml = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "02_reference_turbines", "nrel5mw.yaml")
data = load_geometry_yaml(baseyaml)
af = data['airfoils']
af_names = ["Cylinder1", "Cylinder2", "DU40_A17", "DU35_A17", "DU30_A17", "DU25_A17", "DU21_A17", "NACA64_A17"]
airfoil_types = [0] * len(af_names)
for i in range(len(af_names)):
    for j in range(len(af)):
        if af[j]["name"] == af_names[i]:
            polars = af[j]['polars'][0]
            airfoil_types[i] = CCAirfoil(
                polars["re_sets"][0]["cl"]["grid"],
                [polars["re_sets"][0]["re"]],
                polars["re_sets"][0]["cl"]["values"],
                polars["re_sets"][0]["cd"]["values"],
                polars["re_sets"][0]["cm"]["values"],
            )

# place at appropriate radial stations
af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

af = [0] * len(r)
for i in range(len(r)):
    af[i] = airfoil_types[af_idx[i]]

# 2 ----------

# 3 ----------

# create CCBlade object
rotor = CCBlade(
    r, chord, theta, af, Rhub, Rtip, B, rho, mu, precone, tilt, yaw, shearExp, hubHt, nSector, derivatives=True
)

# 3 ----------

# 4 ----------
# set conditions
Uinf = 10.0
tsr = 7.55
pitch = 0.0
Omega = Uinf * tsr / Rtip * 30.0 / pi  # convert to RPM
azimuth = 0.0

# # evaluate distributed loads
# Np, Tp = rotor.distributedAeroLoads(Uinf, Omega, pitch, azimuth)

# 4 ----------

# 5 ----------

loads, derivs = rotor.distributedAeroLoads(Uinf, Omega, pitch, azimuth)
Np = loads["Np"]
Tp = loads["Tp"]
dNp = derivs["dNp"]
dTp = derivs["dTp"]

n = len(r)

# n x n (diagonal)
dNp_dr = dNp["dr"]
dNp_dchord = dNp["dchord"]
dNp_dtheta = dNp["dtheta"]
dNp_dpresweep = dNp["dpresweep"]

# n x n (tridiagonal)
dNp_dprecurve = dNp["dprecurve"]

# n x 1
dNp_dRhub = dNp["dRhub"]
dNp_dRtip = dNp["dRtip"]
dNp_dprecone = dNp["dprecone"]
dNp_dtilt = dNp["dtilt"]
dNp_dhubHt = dNp["dhubHt"]
dNp_dyaw = dNp["dyaw"]
dNp_dazimuth = dNp["dazimuth"]
dNp_dUinf = dNp["dUinf"]
dNp_dOmega = dNp["dOmega"]
dNp_dpitch = dNp["dpitch"]

# 5 ----------


# 6 ----------


loads, derivs = rotor.evaluate([Uinf], [Omega], [pitch])
P = loads["P"]
T = loads["T"]
Q = loads["Q"]

dP = derivs["dP"]
dT = derivs["dT"]
dQ = derivs["dQ"]

npts = len(P)

# npts x 1
dP_dprecone = dP["dprecone"]
dP_dtilt = dP["dtilt"]
dP_dhubHt = dP["dhubHt"]
dP_dRhub = dP["dRhub"]
dP_dRtip = dP["dRtip"]
dP_dprecurveTip = dP["dprecurveTip"]
dP_dpresweepTip = dP["dpresweepTip"]
dP_dyaw = dP["dyaw"]

# npts x npts
dP_dUinf = dP["dUinf"]
dP_dOmega = dP["dOmega"]
dP_dpitch = dP["dpitch"]

# npts x n
dP_dr = dP["dr"]
dP_dchord = dP["dchord"]
dP_dtheta = dP["dtheta"]
dP_dprecurve = dP["dprecurve"]
dP_dpresweep = dP["dpresweep"]


# 6 ----------


# 7 ----------

idx = 8
delta = 1e-6 * r[idx]
r[idx] += delta

rotor_fd = CCBlade(
    r, chord, theta, af, Rhub, Rtip, B, rho, mu, precone, tilt, yaw, shearExp, hubHt, nSector, derivatives=False
)

r[idx] -= delta

loads, derivs = rotor_fd.distributedAeroLoads(Uinf, Omega, pitch, azimuth)
Npd = loads["Np"]
Tpd = loads["Tp"]

dNp_dr_fd = (Npd - Np) / delta
dTp_dr_fd = (Tpd - Tp) / delta

print("(analytic) dNp_i/dr_i =", dNp_dr[idx, idx])
print("(fin diff) dNp_i/dr_i =", dNp_dr_fd[idx])
print()
# 7 ----------


# 8 ----------
delta = 1e-6 * precone
precone += delta

rotor_fd = CCBlade(
    r, chord, theta, af, Rhub, Rtip, B, rho, mu, precone, tilt, yaw, shearExp, hubHt, nSector, derivatives=False
)

precone -= delta

loads, derivs = rotor_fd.evaluate([Uinf], [Omega], [pitch], coefficients=False)
Pd = loads["P"]
Td = loads["T"]
Qd = loads["Q"]

dT_dprecone_fd = (Td - T) / delta
dQ_dprecone_fd = (Qd - Q) / delta
dP_dprecone_fd = (Pd - P) / delta

print("(analytic) dP/dprecone =", dP_dprecone[0, 0])
print("(fin diff) dP/dprecone =", dP_dprecone_fd[0])
print()
# 8 ----------


# 9 ----------
idx = 12
delta = 1e-6 * r[idx]
r[idx] += delta

rotor_fd = CCBlade(
    r, chord, theta, af, Rhub, Rtip, B, rho, mu, precone, tilt, yaw, shearExp, hubHt, nSector, derivatives=False
)

r[idx] -= delta

loads, derivs = rotor_fd.evaluate([Uinf], [Omega], [pitch], coefficients=False)
Pd = loads["P"]
Td = loads["T"]
Qd = loads["Q"]

dT_dr_fd = (Td - T) / delta
dQ_dr_fd = (Qd - Q) / delta
dP_dr_fd = (Pd - P) / delta

print("(analytic) dP/dr_i =", dP_dr[0, idx])
print("(fin diff) dP/dr_i =", dP_dr_fd[0])
print()
# 9 ----------
