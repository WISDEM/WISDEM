import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import wisdem.pBeam._pBEAM as _pBEAM

# Inputs
angle_deg               = 15.
blade_length            = 100.
lateral_clearance       = 6.7056 # 22 feet
vertical_clearance      = 7.0104 # 23 feet
n_points                = 10000
max_strains             = 3500. * 1.e-6
n_carts                 = 4
n_opt                   = 10
max_LV                  = 0.5
weight_car              = 217724.16
max_root_rot_deg        = 10.
#########

def arc_length(x, y):
    arc = np.sqrt( np.diff(x)**2 + np.diff(y)**2 )
    return np.r_[0.0, np.cumsum(arc)]

angle_rad    = angle_deg / 180. * np.pi
radius = sp.constants.foot * 100. /(2.*np.sin(angle_rad/2.))

filename = '/Users/pbortolo/work/3_projects/3_BAR/EC_rail_transport.xlsx'
data=pd.read_excel(filename,sheet_name='BAR00_SNL')
s = data._series['s [-]']
r = np.array(s * blade_length)
EIflap   = data._series['EIyy [Nm2]']
EIedge   = data._series['EIxx [Nm2]']
GJ       = data._series['GJ [Nm2]']
rhoA     = data._series['rhoA [kg/m]']
dist_ss  = data._series['A to EC [m]']
dist_ps  = data._series['B to EC [m]']

M        = np.array(max_strains * EIflap / dist_ss)
V        = np.gradient(M,r)
q        = np.gradient(V,r)    

r_opt    = np.linspace(0., blade_length, n_opt)
M_opt    = np.interp(r_opt, r, M)
V_opt    = np.gradient(M_opt,r_opt)
q_opt    = np.gradient(V_opt,r_opt)   

# f, axes = plt.subplots(3,1,figsize=(5.3, 5.3))
# axes[0].plot(r, M * 1.e-6)
# axes[0].plot(r_opt, M_opt * 1.e-6)
# axes[0].set_ylabel('Moment [MNm]')
# axes[1].plot(r, V*1.e-6)
# axes[1].plot(r_opt, V_opt*1.e-6)
# axes[1].set_ylabel('Shear Forces [MN]')
# axes[2].plot(r, q*1.e-3)
# axes[2].plot(r_opt, q_opt*1.e-3)
# axes[2].set_ylabel('Distributed Forces [kN/m]')
# plt.xlabel("Blade span [m]")
# plt.show()

r_midline = radius
r_outer   = r_midline + 0.5*lateral_clearance
r_inner   = r_midline - 0.5*lateral_clearance

x_rail  = np.linspace(0., 2.*r_midline, n_points)
y_rail  = np.sqrt(r_midline**2. - (x_rail-r_midline)**2.)
arc_rail = arc_length(x_rail, y_rail)

x_outer  = np.linspace(- 0.5*lateral_clearance, 2.*r_midline + 0.5*lateral_clearance, n_points)
y_outer  = np.sqrt(r_outer**2. - (x_outer-r_midline)**2.)

x_inner  = np.linspace(0.5*lateral_clearance, 2.*r_midline - 0.5*lateral_clearance, n_points)
y_inner  = np.sqrt(r_inner**2. - (x_inner-r_midline)**2.)

r_carts = np.linspace(0., blade_length, n_carts+1)

def get_max_force(inputs):
    
    M_iter    = M_opt * inputs[:-1]
    V_iter    = np.gradient(M_iter,r_opt)
    q_iter    = np.gradient(V_iter,r_opt)
    
    fine_grid = np.unique(np.hstack((r, r_carts)))
    fine_q    = np.interp(fine_grid, r_opt, q_iter)

    cart_id = np.zeros_like(r_carts)
    for i in range(n_carts+1):
        cart_id[i]   = np.argmin(abs(fine_grid - r_carts[i]))

    reaction_carts = np.zeros(n_carts)
    for i in range(n_carts):
        reaction_carts[i] = np.trapz(fine_q[int(cart_id[i]):int(cart_id[i+1])], fine_grid[int(cart_id[i]):int(cart_id[i+1])])

    return np.max(abs(reaction_carts))*1.e-5

def get_constraints(inputs):
    
    M_iter    = M_opt * inputs[:-1]
    V_iter    = np.gradient(M_iter,r_opt)
    q_iter    = np.gradient(V_iter,r_opt)

    root_rot_rad_iter = inputs[-1]

    
    dist_ss_iter   = np.interp(r_opt, r, dist_ss)
    dist_ps_iter   = np.interp(r_opt, r, dist_ps)
    EIflap_iter    = np.interp(r_opt, r, EIflap)
    eps            = M_iter * dist_ss_iter / EIflap_iter
    consts_strains = (max_strains - abs(eps))*1.e+3

    # fs=10
    # f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
    # ax.plot(r_opt, eps)
    # ax.legend(fontsize=fs)
    # plt.xlabel('blade length [m]', fontsize=fs+2, fontweight='bold')
    # plt.ylabel('eps [-]', fontsize=fs+2, fontweight='bold')
    # plt.xticks(fontsize=fs)
    # plt.yticks(fontsize=fs)
    # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    # plt.subplots_adjust(bottom = 0.15, left = 0.18)
    # plt.show()

    V_iter = np.gradient(M_iter,r_opt)
    q_iter = np.gradient(V_iter,r_opt)
    EIedge_iter  = np.interp(r_opt, r, EIedge)
    GJ_iter      = np.interp(r_opt, r, GJ)
    rhoA_iter    = np.interp(r_opt, r, rhoA)
    EA_iter      = EIedge_iter
    rhoJ_iter    = rhoA_iter
    p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_iter, EIedge_iter, EIflap_iter, GJ_iter, rhoA_iter, rhoJ_iter)
    p_tip        = _pBEAM.TipData()  # no tip mass
    p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
    p_loads      = _pBEAM.Loads(n_opt, q_iter, np.zeros_like(r_opt), np.zeros_like(r_opt))
    blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
    dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

    x_blade_transport = dx*blade_length/arc_length(r_opt, dx)[-1]
    y_blade_transport = r_opt*blade_length/arc_length(r_opt, dx)[-1]

    ps_x = x_blade_transport + dist_ps_iter
    ss_x = x_blade_transport - dist_ss_iter
    ps_y = ss_y = y_blade_transport

    ps_x_rot  = ps_x*np.cos(root_rot_rad_iter) - ps_y*np.sin(root_rot_rad_iter)
    ps_y_rot  = ps_y*np.cos(root_rot_rad_iter) + ps_x*np.sin(root_rot_rad_iter)
    
    ss_x_rot  = ss_x*np.cos(root_rot_rad_iter) - ss_y*np.sin(root_rot_rad_iter)
    ss_y_rot  = ss_y*np.cos(root_rot_rad_iter) + ss_x*np.sin(root_rot_rad_iter)

    id_outer = np.zeros(n_opt, dtype = int)
    id_inner = np.zeros(n_opt, dtype = int)
    for i in range(n_opt):
        id_outer[i] = np.argmin(abs(y_outer[:int(np.ceil(n_points*0.5))] - ps_y_rot[i]))
        # if x_blade_transport_rot[i] < x_outer[int(id_outer[i])]:
        #     print('Blade breaks outer envelope at station ' + str(i))
        id_inner[i] = np.argmin(abs(y_inner[:int(np.ceil(n_points*0.5))]  - ss_y_rot[i]))
        # if x_blade_transport_rot[i] > x_inner[int(id_inner[i])]:
        #     print('Blade breaks inner envelope at station ' + str(i))

    consts_envelope_outer = ss_x_rot - x_outer[id_outer]
    consts_envelope_inner = x_inner[id_outer] - ps_x_rot

    consts = np.hstack((consts_strains, consts_envelope_outer, consts_envelope_inner))

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


    return consts

x0    = np.hstack((np.ones(n_opt), 0.))

bnds = ((0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(0., 1.),(-max_root_rot_deg / 180. * np.pi, max_root_rot_deg / 180. * np.pi))
const           = {}
const['type']   = 'ineq'
const['fun']    = get_constraints
res    = minimize(get_max_force, x0, method='SLSQP', bounds=bnds, constraints=const)

if res.success == False:
    exit('The optimization cannot satisfy the constraint on max strains of 3500 mu eps')
else:

    get_constraints(res.x)
    
    M_final    = M_opt * res.x[:-1]
    V_final    = np.gradient(M_final,r_opt)
    q_final    = np.gradient(V_final,r_opt)

    root_rot_rad_final = res.x[-1]

    print('The optimizer finds a solution!')
    print('Prescribed rotation angle: ' + str(root_rot_rad_final * 180. / np.pi) + ' deg')
    print('Max reaction force: ' + str(get_max_force(res.x)*10.) + ' MN')

    dist_ss_final  = np.interp(r_opt, r, dist_ss)
    dist_ps_final  = np.interp(r_opt, r, dist_ps)
    EIflap_final    = np.interp(r_opt, r, EIflap)
    eps            = M_final * dist_ss_final / EIflap_final

    V_final = np.gradient(M_final,r_opt)
    q_final = np.gradient(V_final,r_opt)
    EIedge_final  = np.interp(r_opt, r, EIedge)
    GJ_final      = np.interp(r_opt, r, GJ)
    rhoA_final    = np.interp(r_opt, r, rhoA)
    EA_final      = EIedge_final
    rhoJ_final    = rhoA_final
    p_section    = _pBEAM.SectionData(n_opt, r_opt, EA_final, EIedge_final, EIflap_final, GJ_final, rhoA_final, rhoJ_final)
    p_tip        = _pBEAM.TipData()  # no tip mass
    p_base       = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base
    p_loads      = _pBEAM.Loads(n_opt, q_final, np.zeros_like(r_opt), np.zeros_like(r_opt))
    blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
    dx, dy, dz, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

    x_blade_transport = dx*blade_length/arc_length(r_opt, dx)[-1]
    y_blade_transport = r_opt*blade_length/arc_length(r_opt, dx)[-1]


    ps_x = x_blade_transport + dist_ps_final
    ss_x = x_blade_transport - dist_ss_final
    ps_y = ss_y = y_blade_transport

    ps_x_rot  = ps_x*np.cos(root_rot_rad_final) - ps_y*np.sin(root_rot_rad_final)
    ps_y_rot  = ps_y*np.cos(root_rot_rad_final) + ps_x*np.sin(root_rot_rad_final)
    
    ss_x_rot  = ss_x*np.cos(root_rot_rad_final) - ss_y*np.sin(root_rot_rad_final)
    ss_y_rot  = ss_y*np.cos(root_rot_rad_final) + ss_x*np.sin(root_rot_rad_final)

    x_blade_transport_rot = x_blade_transport*np.cos(root_rot_rad_final) - y_blade_transport*np.sin(root_rot_rad_final)
    y_blade_transport_rot = y_blade_transport*np.cos(root_rot_rad_final) + x_blade_transport*np.sin(root_rot_rad_final)


    fs=10
    f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
    ax.plot(r_opt, eps)
    ax.legend(fontsize=fs)
    plt.xlabel('Blade span position [m]', fontsize=fs+2, fontweight='bold')
    plt.ylabel('Strains [-]', fontsize=fs+2, fontweight='bold')
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.18)

    f, ax = plt.subplots(1,1,figsize=(5.3, 5.3))
    ax.plot(x_rail, y_rail,   color=[0.8,0.8,0.8], linestyle='--', label='rail midline')
    ax.plot(x_outer, y_outer, color=[0.8,0.8,0.8], linestyle=':', label='clearance envelope')
    ax.plot(x_inner, y_inner, color=[0.8,0.8,0.8], linestyle=':')
    ax.plot(x_blade_transport_rot, y_blade_transport_rot, label='pitch axis')
    ax.plot(ps_x_rot, ps_y_rot, label='pressured side')
    ax.plot(ss_x_rot, ss_y_rot, label='suction side')
    plt.xlim(left=-10, right=110)
    plt.ylim(bottom=0, top=120)
    ax.legend(fontsize=fs)
    plt.xlabel('x [m]', fontsize=fs+2, fontweight='bold')
    plt.ylabel('y [m]', fontsize=fs+2, fontweight='bold')
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(color=[0.8,0.8,0.8], linestyle='--')
    plt.subplots_adjust(bottom = 0.15, left = 0.18)
    
    f, axes = plt.subplots(3,1,figsize=(5.3, 5.3))
    axes[0].plot(r, M * 1.e-6, 'b-')
    axes[0].plot(r_opt, M_final * 1.e-6, 'r--')
    axes[0].set_ylabel('Moment [MNm]')
    axes[1].plot(r, V*1.e-6), 'b-'
    axes[1].plot(r_opt, V_final*1.e-6, 'r--')
    axes[1].set_ylabel('Shear Forces [MN]')
    axes[2].plot(r, q*1.e-3, 'b-')
    axes[2].plot(r_opt, q_final*1.e-3, 'r--')
    axes[2].set_ylabel('Distributed Forces [kN/m]')
    plt.xlabel("Blade span [m]")
    plt.show()