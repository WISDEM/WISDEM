import numpy as np

# Set up inputs
r_foot = 20.0
r_head = 10.0
n_legs = 4
n_bays = 4  # n_x
L = 70.0  # total height of the jacket
q = 0.8  # ratio of bay heights, assumed to be constant

l_osg = 5.0
l_tp = 4.0

d_l = 0.5  # leg diameter
l_msl = 20.0
x_mb = True  # if there's a mud brace
gamma_b = 12.0
gamma_t = 18.0
beta_b = 0.5
beta_t = 0.8
tau_b = 0.3
tau_t = 0.6

E = 2.1e11
G = 8.077e10
rho = 7850.0

# Do calculations to get the rough topology
xi = r_head / r_foot  # must be <= 1.
nu = 2 * np.pi / n_legs  # angle enclosed by two legs
psi_s = np.arctan(r_foot * (1 - xi) / L)  # spatial batter angle
psi_p = np.arctan(r_foot * (1 - xi) * np.sin(nu / 2.0) / L)

tmp = q ** np.arange(n_bays)
bay_heights = l_i = (L - l_osg - l_tp) / (np.sum(tmp) / tmp)
# TODO : add test to verify np.sum(bay_heights == L)

lower_bay_heights = np.hstack((0.0, bay_heights))
lower_bay_radii = r_i = r_foot - np.tan(psi_s) * (l_osg + np.cumsum(lower_bay_heights))

# x joint layer info
l_mi = l_i * r_i[:-1] / (r_i[:-1] + r_i[1:])
# r_mi = r_foot - np.tan(psi_s) * (l_osg + np.cumsum(lower_bay_heights) + l_mi)  # I don't think we actually use this value later
# print(r_mi)


gamma_i = (gamma_t - gamma_b) * (l_osg + np.cumsum(l_i) + l_mi) / (L - l_i[-1] + l_mi[-1] - l_tp) + gamma_b
gamma_i = np.hstack((gamma_b, gamma_i))

length = L - l_i[-1] - l_osg - l_tp
beta_i = (beta_t - beta_b) / length * np.cumsum(l_i) + beta_b
tau_i = (tau_t - tau_b) / length * np.cumsum(l_i) + tau_b


leg_thicknesses = t_l = d_l / (2 * gamma_i)
brace_diameters = d_b = beta_i * d_l
brace_thicknesses = t_b = tau_i * t_l[:-1]


# Take in member properties, like diameters and thicknesses


# Determine member cross-sectional properties
