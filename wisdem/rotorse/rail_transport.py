import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import wisdem.pBeam._pBEAM as _pBEAM

# Inputs
angle_deg               = 10.
blade_length            = 100.
lateral_clearance       = 6.7056 # 22 feet
vertical_clearance      = 7.0104 # 23 feet
n_points                = 10000
max_strains             = 3500. * 1.e-6
n_carts                 = 4
max_LV                  = 0.5
weight_car              = 217724.16
root_rot_deg            = -7.
#########

def arc_length(x, y):
    arc = np.sqrt( np.diff(x)**2 + np.diff(y)**2 )
    return np.r_[0.0, np.cumsum(arc)]

angle_rad    = angle_deg / 180. * np.pi
root_rot_rad = root_rot_deg / 180. * np.pi
radius = sp.constants.foot * 100. /(2.*np.sin(angle_rad/2.))

filename = '/Users/pbortolo/work/3_projects/3_BAR/EC_rail_transport.xlsx'
data=pd.read_excel(filename,sheet_name='BAR03')
s = data._series['s [-]']
r = np.array(s * blade_length)
EIflap   = data._series['EIyy [Nm2]']
EIedge   = data._series['EIxx [Nm2]']
GJ       = data._series['GJ [Nm2]']
rhoA     = data._series['rhoA [kg/m]']
dist_pos = data._series['A to EC [m]']

M        = np.array(max_strains * EIflap / dist_pos)
SF       = np.gradient(M,r)


r_carts  = np.linspace(0., 100., n_carts)
M_carts  = np.interp(r_carts, r, M)
SF_carts = -np.gradient(M_carts,r_carts)


r_midline = radius
r_outer   = r_midline + 0.5*lateral_clearance
r_inner   = r_midline - 0.5*lateral_clearance

x_rail  = np.linspace(0., 2.*r_midline, n_points)
y_rail  = np.sqrt(r_midline**2. - (x_rail-r_midline)**2.)
arc_rail = arc_length(x_rail, y_rail)

x_outer  = np.linspace(- 0.5*lateral_clearance, 2.*r_midline + 0.5*lateral_clearance, n_points)
y_outer  = np.sqrt(r_outer**2. - (x_outer-r_midline)**2.)
arc_outer = arc_length(x_outer, y_outer)

x_inner  = np.linspace(0.5*lateral_clearance, 2.*r_midline - 0.5*lateral_clearance, n_points)
y_inner  = np.sqrt(r_inner**2. - (x_inner-r_midline)**2.)
arc_inner = arc_length(x_inner, y_inner)


def get_max_force(inputs):
    return max(inputs[:-1])

def get_constraints(inputs):

    SF_iter = inputs[0:-1]
    root_rot_rad_iter = inputs[-1]

    M_iter = np.zeros_like(r_carts)
    for i in range(len(r_carts)):
        M_iter[i] = np.trapz(SF_iter[i:], r_carts[i:])
    
    M_iter_interp = np.interp(r, r_carts, M_iter)
    eps = M * dist_pos / EIflap

    consts = eps - max_strains

    SF_iter_fine = -np.gradient(M_iter_interp,r)
    F_iter_fine  = np.gradient(SF_iter_fine, r)
    # create finite element objects
    EA       = EIedge
    rhoJ     = rhoA
    p_section       = _pBEAM.SectionData(len(r), r, EA, EIedge, EIflap, GJ, rhoA, rhoJ)
    p_tip           = _pBEAM.TipData()  # no tip mass
    p_base          = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
    # evaluate displacements
    p_loads = _pBEAM.Loads(len(r), F_iter_fine, np.zeros_like(r), np.zeros_like(r))
    blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
    dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

    x_blade_transport = dx*blade_length/arc_length(r, dx)[-1]
    y_blade_transport = r*blade_length/arc_length(r, dx)[-1]

    x_blade_transport_rot = x_blade_transport*np.cos(root_rot_rad_iter) - y_blade_transport*np.sin(root_rot_rad_iter)
    y_blade_transport_rot = y_blade_transport*np.cos(root_rot_rad_iter) + x_blade_transport*np.sin(root_rot_rad_iter)


    for i in range(len(r)):
        id_outer = np.argmin(abs(y_outer[:int(np.ceil(n_points*0.5))] - y_blade_transport_rot[i]))
        if x_blade_transport_rot[i] < x_outer[id_outer]:
            print('Blade breaks outer envelope at station ' + str(i))
        id_inner = np.argmin(abs(y_inner[:int(np.ceil(n_points*0.5))]  - y_blade_transport_rot[i]))
        if x_blade_transport_rot[i] > x_inner[id_inner]:
            print('Blade breaks inner envelope at station ' + str(i))




    return consts

x0    = np.hstack((SF_carts, root_rot_rad))
const           = {}
const['type']   = 'ineq'
const['fun']    = get_constraints
x1    = minimize(get_max_force, x0, method='SLSQP', constraints=const)

SF = x1[:-1]
F        = np.gradient(SF, r)
# create finite element objects
EA       = EIedge
rhoJ     = rhoA
p_section       = _pBEAM.SectionData(len(r), r, EA, EIedge, EIflap, GJ, rhoA, rhoJ)
p_tip           = _pBEAM.TipData()  # no tip mass
p_base          = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
# evaluate displacements
p_loads = _pBEAM.Loads(len(r), F, np.zeros_like(r), np.zeros_like(r))
blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

x_blade_transport = dx*blade_length/arc_length(r, dx)[-1]
y_blade_transport = r*blade_length/arc_length(r, dx)[-1]

x_blade_transport_rot = x_blade_transport*np.cos(root_rot_rad) - y_blade_transport*np.sin(root_rot_rad)
y_blade_transport_rot = y_blade_transport*np.cos(root_rot_rad) + x_blade_transport*np.sin(root_rot_rad)


fs=10
f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
ax.plot(x_rail, y_rail,   color=[0.8,0.8,0.8], linestyle='--', label='rail midline')
ax.plot(x_outer, y_outer, color=[0.8,0.8,0.8], linestyle=':', label='clearance envelope')
ax.plot(x_inner, y_inner, color=[0.8,0.8,0.8], linestyle=':')
ax.plot(x_blade_transport, y_blade_transport, label='blade max strains')
ax.plot(x_blade_transport_rot, y_blade_transport_rot, label='blade max strains rotated')
plt.xlim(left=-10, right=110)
plt.ylim(bottom=0, top=120)
ax.legend(fontsize=fs)
plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.grid(color=[0.8,0.8,0.8], linestyle='--')
plt.subplots_adjust(bottom = 0.15, left = 0.18)
plt.show()



# M_rec = np.zeros_like(M)
# M_rec2 = np.zeros_like(M)
# for i in range(len(r)):
#     M_rec[i] = np.trapz(F[i:] * (r[i:] - r[i]), r[i:])
#     M_rec2[i] = -np.trapz(SF[i:], r[i:])

# f, axes = plt.subplots(3,1,figsize=(5.3, 5.3))
# axes[0].plot(r, M * 1.e-6)
# axes[0].plot(r, M_rec * 1e-6, '--')
# axes[0].plot(r, M_rec2 * 1e-6, '--')
# axes[0].set_ylabel('Moment [MNm]')
# axes[1].plot(r, SF*1.e-6)
# axes[1].plot(r_temp, SF_temp*1.e-6)
# axes[1].set_ylabel('Shear Forces [MN]')
# axes[2].plot(r, F*1.e-3)
# axes[2].set_ylabel('Distributed Forces [kN/m]')
# plt.xlabel("Blade span [m]")
# plt.show()

# # outputs
# # create finite element objects
# EA       = EIedge
# rhoJ     = rhoA
# p_section       = _pBEAM.SectionData(len(r), r, EA, EIedge, EIflap, GJ, rhoA, rhoJ)
# p_tip           = _pBEAM.TipData()  # no tip mass
# p_base          = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
# # evaluate displacements
# p_loads = _pBEAM.Loads(len(r), F, np.zeros_like(r), np.zeros_like(r))
# blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
# dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

# x_blade_transport = dx*blade_length/arc_length(r, dx)[-1]
# y_blade_transport = r*blade_length/arc_length(r, dx)[-1]

# x_blade_transport_rot = x_blade_transport*np.cos(root_rot_rad) - y_blade_transport*np.sin(root_rot_rad)
# y_blade_transport_rot = y_blade_transport*np.cos(root_rot_rad) + x_blade_transport*np.sin(root_rot_rad)

# r_midline = radius
# r_outer   = r_midline + 0.5*lateral_clearance
# r_inner   = r_midline - 0.5*lateral_clearance

# x_rail  = np.linspace(0., 2.*r_midline, n_points)
# y_rail  = np.sqrt(r_midline**2. - (x_rail-r_midline)**2.)
# arc_rail = arc_length(x_rail, y_rail)

# x_outer  = np.linspace(- 0.5*lateral_clearance, 2.*r_midline + 0.5*lateral_clearance, n_points)
# y_outer  = np.sqrt(r_outer**2. - (x_outer-r_midline)**2.)
# arc_outer = arc_length(x_outer, y_outer)

# x_inner  = np.linspace(0.5*lateral_clearance, 2.*r_midline - 0.5*lateral_clearance, n_points)
# y_inner  = np.sqrt(r_inner**2. - (x_inner-r_midline)**2.)
# arc_inner = arc_length(x_inner, y_inner)

# fs=10
# f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
# ax.plot(x_rail, y_rail,   color=[0.8,0.8,0.8], linestyle='--', label='rail midline')
# ax.plot(x_outer, y_outer, color=[0.8,0.8,0.8], linestyle=':', label='clearance envelope')
# ax.plot(x_inner, y_inner, color=[0.8,0.8,0.8], linestyle=':')
# ax.plot(x_blade_transport, y_blade_transport, label='blade max strains')
# ax.plot(x_blade_transport_rot, y_blade_transport_rot, label='blade max strains rotated')
# plt.xlim(left=-10, right=110)
# plt.ylim(bottom=0, top=120)
# ax.legend(fontsize=fs)
# plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
# plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# plt.grid(color=[0.8,0.8,0.8], linestyle='--')
# plt.subplots_adjust(bottom = 0.15, left = 0.18)
# plt.show()


# for i in range(len(r)):
#     id_outer = np.argmin(abs(y_outer[:int(np.ceil(n_points*0.5))] - y_blade_transport_rot[i]))
#     if x_blade_transport_rot[i] < x_outer[id_outer]:
#         print('Blade breaks outer envelope at station ' + str(i))
#     id_inner = np.argmin(abs(y_inner[:int(np.ceil(n_points*0.5))]  - y_blade_transport_rot[i]))
#     if x_blade_transport_rot[i] > x_inner[id_inner]:
#         print('Blade breaks inner envelope at station ' + str(i))


# f0 = f

# def get_forces(inputs):

#     return None

# x1    = minimize(get_forces, x0, method='SLSQP')







# i_blade_start = 0
# i_blade_end   = np.argmin(abs(arc_rail - blade_length))
# i_blade_lock  = np.argmin(abs(arc_rail - rigid_portion * blade_length))


# x_blade_flex = x_rail[i_blade_start:i_blade_end]
# y_blade_flex = y_rail[i_blade_start:i_blade_end]

# dy=np.diff(y_blade_flex,1)
# dx=np.diff(x_blade_flex,1)
# yfirst=dy/dx
# xfirst=0.5*(x_blade_flex[:-1]+x_blade_flex[1:])
# dyfirst=np.diff(yfirst,1)
# dxfirst=np.diff(xfirst,1)
# ysecond=dyfirst/dxfirst
# xsecond=0.5*(xfirst[:-1]+xfirst[1:])

# # curvature = ysecond / (1.+yfirst[1:]**2.)**1.5
# # radius    = 1./curvature


# x_blade_transport = np.zeros_like(x_blade_flex)
# y_blade_transport = np.zeros_like(y_blade_flex)

# x_blade_transport[i_blade_start : i_blade_lock] = np.linspace(x_rail[i_blade_start], x_rail[i_blade_lock], i_blade_lock)
# y_blade_transport[i_blade_start : i_blade_lock] = np.linspace(y_rail[i_blade_start], y_rail[i_blade_lock], i_blade_lock)

# dy_blade_transport=np.diff(y_blade_transport,1)
# dx_blade_transport=np.diff(x_blade_transport,1)
# slope_rigid=dy_blade_transport[0]/dx_blade_transport[0]

# def get_circle(min_inputs):


#     for i in range(i_blade_lock, i_blade_end):
#         x_blade_transport[i] = x_blade_flex[i]
#         y_blade_transport[i] = np.sqrt(min_inputs[2]**2.-(x_blade_transport[i] - min_inputs[0])**2.)

#     arc_blade_transport = arc_length(x_blade_transport, y_blade_transport)


#     return abs(arc_blade_transport[-1] - blade_length)

# # def get_circle(min_inputs):

# #     Uhub_i  = min_inputs[1]
# #     pitch   = min_inputs[0]           
# #     return abs(P_i - inputs['rated_power'])

# x0              = [r_midline , 0., r_midline]
# bnds            = [1.e+6, 1.e+6, 1.e+6]
# # const           = {}
# # const['type']   = 'eq'
# # const['fun']    = get_Uhub_rated_II12
# x1    = minimize(get_circle, x0, method='SLSQP')

# # for i in range(i_blade_lock, i_blade_end):
# #     x_blade_transport[i] = x_blade_flex[i]
# #     y_blade_transport[i] = y_blade_transport[i-1] + slope_rigid * np.diff(x_blade_flex)[i-1]







