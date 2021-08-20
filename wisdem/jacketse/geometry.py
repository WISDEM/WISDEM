import numpy as np
import matplotlib.pyplot as plt

# Set up inputs
r_foot = 10.0
r_head = 6.0
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
bay_nodes = np.zeros((n_legs, n_bays + 1, 3))
bay_nodes[:, :, 2] = np.cumsum(lower_bay_heights)
bay_nodes[:, :, 2] += l_osg

for idx in range(n_legs):
    tmp = np.outer(lower_bay_radii, np.array([np.cos(nu * idx), np.sin(nu * idx)]))
    bay_nodes[idx, :, 0:2] = tmp

leg_nodes = np.zeros((n_legs, n_bays + 2, 3))
tmp = l_mi / np.cos(psi_p)
leg_nodes[:, 1:-1, 2] = bay_nodes[:, :-1, 2] + tmp
leg_nodes[:, -1, 2] = L

leg_radii = np.interp(
    leg_nodes[0, :, 2],
    np.linspace(leg_nodes[0, 0, 2], leg_nodes[0, -1, 2], n_bays + 2),
    np.linspace(r_foot, r_head, n_bays + 2),
)
for idx in range(n_legs):
    tmp = np.outer(leg_radii, np.array([np.cos(nu * idx), np.sin(nu * idx)]))
    leg_nodes[idx, :, 0:2] = tmp

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

nodes = leg_nodes
ax.scatter(nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2])

nodes = bay_nodes
ax.scatter(nodes[:, :, 0], nodes[:, :, 1], nodes[:, :, 2])

for jdx in range(n_bays):
    for idx in range(n_legs):
        n1 = bay_nodes[idx, jdx]
        n2 = bay_nodes[(idx + 1) % n_legs, (jdx + 1) % (n_bays + 1)]
        plt.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]])

        n1 = bay_nodes[(idx + 1) % n_legs, jdx]
        n2 = bay_nodes[idx, (jdx + 1) % (n_bays + 1)]
        plt.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]])

for jdx in range(n_bays + 1):
    for idx in range(n_legs):
        n1 = leg_nodes[idx, jdx]
        n2 = leg_nodes[idx, jdx + 1]
        plt.plot([n1[0], n2[0]], [n1[1], n2[1]], [n1[2], n2[2]])

plt.show()


# plt.figure()
#
# for idx in range(n_bays+1):
#     plt.plot(leg_nodes[0, idx:idx+2, 0], leg_nodes[0, idx:idx+2, 2])
#     plt.plot(leg_nodes[2, idx:idx+2, 0], leg_nodes[2, idx:idx+2, 2])
#
# for idx in range(n_bays):
#     plt.plot(bay_nodes[0, idx:idx+2, 0], bay_nodes[3, idx:idx+2, 2])
#     # plt.plot(bay_nodes[1, idx:idx+2, 0], bay_nodes[0, idx:idx+2, 2])
#
#
#
#
# plt.show()

# Determine member cross-sectional properties

# Eventually need to get to:
# elements = pyframe3dd.ElementData(element, N1, N2, Az, Asx, Asy, Jz, Ixx, Iyy, E, G, roll, rho)
