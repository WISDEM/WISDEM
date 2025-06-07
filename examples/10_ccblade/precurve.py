import numpy as np
import matplotlib.pyplot as plt

from wisdem.ccblade.ccblade import CCBlade, CCAirfoil

plot_flag = False

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


import os
from wisdem.inputs.validation import load_geometry_yaml
baseyaml = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "02_reference_turbines", "nrel5mw.yaml")
data = load_geometry_yaml(baseyaml)
af = data['airfoils']
af_names = ["Cylinder", "Cylinder", "DU40_A17", "DU35_A17", "DU30_A17", "DU25_A17", "DU21_A17", "NACA64_A17"]
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


# 1 ----------

precone = 0.0
precurve = np.linspace(0, 4.9, len(r)) ** 2
precurveTip = 25.0


# create CCBlade object
rotor = CCBlade(
    r,
    chord,
    theta,
    af,
    Rhub,
    Rtip,
    B,
    rho,
    mu,
    precone,
    tilt,
    yaw,
    shearExp,
    hubHt,
    nSector,
    precurve=precurve,
    precurveTip=precurveTip,
)

# 1 ----------

if plot_flag:
    plt.plot(precurve, r, "k")
    plt.plot(precurve, -r, "k")
    plt.axis("equal")
    plt.grid()
    plt.savefig("rotorshape.pdf")
    plt.show()
