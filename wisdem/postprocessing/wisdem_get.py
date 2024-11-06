import numpy as np
import pandas as pd
import wisdem.commonse.utilities as util


def is_floating(prob):
    return prob.model.options["modeling_options"]["flags"]["floating"]


def is_monopile(prob):
    return prob.model.options["modeling_options"]["flags"]["monopile"]


def get_tower_diameter(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.monopile_outer_diameter"], prob["towerse.tower_outer_diameter"][1:]]
    else:
        return prob["towerse.tower_outer_diameter"]


def get_tower_thickness(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.monopile_wall_thickness"], prob["towerse.tower_wall_thickness"]]
    else:
        return prob["towerse.tower_wall_thickness"]


def get_zpts(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.z_param"], prob["towerse.z_param"][1:]]
    else:
        return prob["towerse.z_param"]


def get_section_height(prob):
    return np.diff(get_zpts(prob))


def get_transition_height(prob):
    return prob["towerse.foundation_height"]


def get_tower_outfitting(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.member.outfitting_factor"], prob["towerse.member.outfitting_factor"][1:]]
    else:
        return prob["towerse.member.outfitting_factor"]


def get_tower_E(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.member.E"], prob["towerse.member.E"][1:]]
    else:
        return prob["towerse.member.E"]


def get_tower_G(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.member.G"], prob["towerse.member.G"][1:]]
    else:
        return prob["towerse.member.G"]


def get_tower_rho(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.member.rho"], prob["towerse.member.rho"][1:]]
    else:
        return prob["towerse.member.rho"]


def get_tower_mass(prob):
    return prob["towerse.tower_mass"]


def get_tower_cost(prob):
    return prob["towerse.tower_cost"]


def get_monopile_mass(prob):
    return prob["fixedse.monopile_mass"]


def get_monopile_cost(prob):
    return prob["fixedse.monopile_cost"]


def get_structural_mass(prob):
    if is_monopile(prob):
        return prob["fixedse.structural_mass"]
    else:
        return prob["towerse.tower_mass"]


def get_structural_cost(prob):
    if is_monopile(prob):
        return prob["fixedse.structural_cost"]
    else:
        return prob["towerse.tower_cost"]


def get_tower_freqs(prob):
    if is_monopile(prob):
        return np.r_[prob["fixedse.structural_frequencies"], prob["towerse.tower.structural_frequencies"]]
    else:
        return prob["towerse.tower.structural_frequencies"]


def get_tower_cm(prob):
    return prob["towerse.tower_center_of_mass"]


def get_tower_cg(prob):
    return get_tower_cm(prob)


def get_blade_shape(prob):
    blade_shape  = np.c_[prob.get_val('blade.outer_shape_bem.s'),
                         prob.get_val('blade.outer_shape_bem.ref_axis','m')[:,2],
                         prob.get_val('blade.outer_shape_bem.chord','m'),
                         prob.get_val('blade.outer_shape_bem.twist', 'deg'),
                         prob.get_val('blade.interp_airfoils.r_thick_interp')*100,
                         prob.get_val('blade.outer_shape_bem.pitch_axis')*100,
                         prob.get_val('blade.outer_shape_bem.ref_axis','m')[:,0],
                         prob.get_val('blade.outer_shape_bem.ref_axis','m')[:,1],
                         ]
    blade_shape_col = ['Blade Span','Rotor Coordinate [m]',
                       'Chord [m]', 'Twist [deg]',
                       'Relative Thickness [%]', 'Pitch Axis Chord Location [%]',
                       'Prebend [m]', 'Sweep [m]']
    return pd.DataFrame(data=blade_shape, columns=blade_shape_col)


def get_blade_elasticity(prob):
    blade_stiff  = np.c_[prob.get_val('rotorse.r','m'),
                         prob.get_val('rotorse.A','m**2'),
                         prob.get_val('rotorse.EA','N'),
                         prob.get_val('rotorse.EIxx','N*m**2'),
                         prob.get_val('rotorse.EIyy','N*m**2'),
                         prob.get_val('rotorse.EIxy','N*m**2'),
                         prob.get_val('rotorse.GJ','N*m**2'),
                         prob.get_val('rotorse.rhoA','kg/m'),
                         prob.get_val('rotorse.rhoJ','kg*m'),
                         prob.get_val('rotorse.x_ec','mm'),
                         prob.get_val('rotorse.y_ec','mm'),
                         prob.get_val('rotorse.re.x_tc','mm'),
                         prob.get_val('rotorse.re.y_tc','mm'),
                         prob.get_val('rotorse.re.x_sc','mm'),
                         prob.get_val('rotorse.re.y_sc','mm'),
                         prob.get_val('rotorse.re.x_cg','mm'),
                         prob.get_val('rotorse.re.y_cg','mm'),
                         prob.get_val('rotorse.re.precomp.flap_iner','kg/m'),
                         prob.get_val('rotorse.re.precomp.edge_iner','kg/m')]
    blade_stiff_col = ['Blade Span [m]',
                       'Cross-sectional area [m^2]',
                       'Axial stiffness [N]',
                       'Edgewise stiffness [Nm^2]',
                       'Flapwise stiffness [Nm^2]',
                       'Flap-edge coupled stiffness [Nm^2]',
                       'Torsional stiffness [Nm^2]',
                       'Mass density [kg/m]',
                       'Polar moment of inertia density [kg*m]',
                       'X-distance to elastic center [mm]',
                       'Y-distance to elastic center [mm]',
                       'X-distance to tension center [mm]',
                       'Y-distance to tension center [mm]',
                       'X-distance to shear center [mm]',
                       'Y-distance to shear center [mm]',
                       'X-distance to mass center [mm]',
                       'Y-distance to mass center [mm]',
                       'Section flap inertia [kg/m]',
                       'Section edge inertia [kg/m]',
                       ]
    return pd.DataFrame(data=blade_stiff, columns=blade_stiff_col)


def get_rotor_performance(prob):
    rotor_perf = np.c_[prob.get_val("rotorse.rp.powercurve.V",'m/s'),
                       prob.get_val("rotorse.rp.powercurve.pitch",'deg'),
                       prob.get_val("rotorse.rp.powercurve.P",'MW'),
                       prob.get_val("rotorse.rp.powercurve.Cp"),
                       prob.get_val("rotorse.rp.powercurve.Cp_aero"),
                       prob.get_val("rotorse.rp.powercurve.Omega",'rpm'),
                       prob.get_val("rotorse.rp.powercurve.Omega",'rad/s')*0.5*prob["configuration.rotor_diameter_user"],
                       prob.get_val("rotorse.rp.powercurve.T",'MN'),
                       prob.get_val("rotorse.rp.powercurve.Ct_aero"),
                       prob.get_val("rotorse.rp.powercurve.Q",'MN*m'),
                       prob.get_val("rotorse.rp.powercurve.Cq_aero"),
                       prob.get_val("rotorse.rp.powercurve.M",'MN*m'),
                       prob.get_val("rotorse.rp.powercurve.Cm_aero"),
                       ]
    rotor_perf_col = ['Wind [m/s]','Pitch [deg]',
                      'Power [MW]','Power Coefficient [-]','Aero Power Coefficient [-]',
                      'Rotor Speed [rpm]','Tip Speed [m/s]',
                      'Thrust [MN]','Thrust Coefficient [-]',
                      'Torque [MNm]','Torque Coefficient [-]',
                      'Blade Moment [MNm]','Blade Moment Coefficient [-]',
                      ]
    return pd.DataFrame(data=rotor_perf, columns=rotor_perf_col)


def get_nacelle_mass(prob):

    # Nacelle mass properties tabular
    # Columns are ['Mass', 'CoM_x', 'CoM_y', 'CoM_z',
    #              'MoI_cm_xx', 'MoI_cm_yy', 'MoI_cm_zz', 'MoI_cm_xy', 'MoI_cm_xz', 'MoI_cm_yz',
    #              'MoI_TT_xx', 'MoI_TT_yy', 'MoI_TT_zz', 'MoI_TT_xy', 'MoI_TT_xz', 'MoI_TT_yz']
    nacDF = prob.model.wt.wt_rna.drivese.nac._mass_table
    hub_cm = prob["drivese.hub_system_cm"][0]
    L_drive = prob["drivese.L_drive"][0]
    tilt = prob.get_val('nacelle.uptilt', 'rad')[0]
    shaft0 = prob["drivese.shaft_start"]
    Cup = -1.0
    hub_cm = R = shaft0 + (L_drive + hub_cm) * np.array([Cup * np.cos(tilt), 0.0, np.sin(tilt)])
    hub_mass = prob['drivese.hub_system_mass']
    hub_I = prob["drivese.hub_system_I"]
    hub_I_TT = util.rotateI(hub_I, -Cup * tilt, axis="y")
    hub_I_TT = util.unassembleI( util.assembleI(hub_I_TT) +
                                 hub_mass * (np.dot(R, R) * np.eye(3) - np.outer(R, R)) )
    blades_mass = prob['drivese.blades_mass']
    blades_I = prob["drivese.blades_I"]
    blades_I_TT = util.rotateI(blades_I, -Cup * tilt, axis="y")
    blades_I_TT = util.unassembleI( util.assembleI(blades_I_TT) +
                                    blades_mass * (np.dot(R, R) * np.eye(3) - np.outer(R, R)) )
    rna_mass = prob['drivese.rna_mass']
    rna_cm = R = prob['drivese.rna_cm']
    rna_I_TT = prob['drivese.rna_I_TT']
    rna_I = util.unassembleI( util.assembleI(rna_I_TT) +
                                    rna_mass * (np.dot(R, R) * np.eye(3) + np.outer(R, R)) )
    nacDF.loc['Blades'] = np.r_[blades_mass, hub_cm, blades_I, blades_I_TT].tolist()
    nacDF.loc['Hub_System'] = np.r_[hub_mass, hub_cm, hub_I, hub_I_TT].tolist()
    nacDF.loc['RNA'] = np.r_[rna_mass, rna_cm, rna_I, rna_I_TT].tolist()
    return nacDF


def get_tower_table(prob):

    # Tabular output: Tower
    float_flag = is_floating(prob)
    water_depth = prob['env.water_depth']
    h_trans = get_transition_height(prob)
    htow = get_zpts(prob) #np.cumsum(np.r_[0.0, prob['towerse.tower_section_height']]) + prob['towerse.z_start']
    t = get_tower_thickness(prob)
    towdata = np.c_[htow,
                    get_tower_diameter(prob),
                    np.r_[t[0], t]]
    rowadd = []
    for k in range(towdata.shape[0]):
        if k==0: continue
        if k+1 < towdata.shape[0]:
            rowadd.append([towdata[k,0]+1e-3, towdata[k,1], towdata[k+1,2]])
    towdata = np.vstack((towdata, rowadd))
    towdata[:,-1] *= 1e3
    towdata = np.round( towdata[towdata[:,0].argsort(),], 3)
    colstr = ['Height [m]','OD [m]', 'Thickness [mm]']
    towDF = pd.DataFrame(data=towdata, columns=colstr)
    mycomments = ['']*towdata.shape[0]
    if not float_flag:
        #breakpoint()
        mycomments[0] = 'Monopile start'
        mycomments[np.where(towdata[:,0] == -water_depth)[0][0]] = 'Mud line'
        mycomments[np.where(towdata[:,0] == 0.0)[0][0]] = 'Water line'
    idx_tow = np.where(towdata[:,0] == h_trans)[0][0]
    mycomments[idx_tow] = 'Tower start'
    mycomments[-1] = 'Tower top'
    towDF['Location'] = mycomments
    towDF = towDF[['Location']+colstr]
    A = 0.25*np.pi*(towDF['OD [m]']**2 - (towDF['OD [m]']-2*1e-3*towDF['Thickness [mm]'])**2)
    I = (1/64.)*np.pi*(towDF['OD [m]']**4 - (towDF['OD [m]']-2*1e-3*towDF['Thickness [mm]'])**4)
    outfitting = np.zeros( len(A) )
    if not float_flag:
        outfitting[:idx_tow] = prob['fixedse.outfitting_factor_in']
    outfitting[idx_tow:] = prob['towerse.outfitting_factor_in']
    towDF['Mass Density [kg/m]'] = outfitting * get_tower_rho(prob)[0] * A
    towDF['Fore-aft inertia [kg.m]'] = towDF['Side-side inertia [kg.m]'] = towDF['Mass Density [kg/m]'] * I/A
    towDF['Fore-aft stiffness [N.m^2]'] = towDF['Side-side stiffness [N.m^2]'] = get_tower_E(prob)[0] * I
    towDF['Torsional stiffness [N.m^2]'] = get_tower_G(prob)[0] * 2*I
    towDF['Axial stiffness [N]'] = get_tower_E(prob)[0] * A
    #with open('tow.tbl','w') as f:
    #    towDF.to_latex(f, index=False)
    return towDF
