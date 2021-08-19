import numpy as np

# Set up inputs
r_foot = 20.0
r_head = 10.0
n_legs = 4
n_bays = 3  # n_x
L = 50.0  # total height of the jacket
q = 0.5  # ratio of bay heights, assumed to be constant

L_osg = 5.0
L_tp = 5.0

# Do calculations to get the rough topology
xi = r_head / r_foot  # must be <= 1.
nu = 2 * np.pi / n_legs  # angle enclosed by two legs
psi_s = np.arctan(r_foot * (1 - xi) / L)  # spatial batter angle
psi_p = np.arctan(r_foot * (1 - xi) * np.sin(nu / 2.0) / L)

tmp = q ** np.arange(n_bays)
bay_heights = (L - L_osg - L_tp) / (np.sum(tmp) / tmp)
# TODO : add test to verify np.sum(bay_heights == L)


# Take in member properties, like diameters and thicknesses


# Determine member cross-sectional properties
