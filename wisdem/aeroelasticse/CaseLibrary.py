import os
import numpy as np

from wisdem.aeroelasticse.CaseGen_General import CaseGen_General
from wisdem.aeroelasticse.CaseGen_IEC import CaseGen_IEC

# def power_curve_fit(fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, Vrated, U_init=[], Omega_init=[], pitch_init=[], Turbsim_exe='', ptfm_U_init=[], ptfm_pitch_init=[], ptfm_surge_init=[], ptfm_heave_init=[], metocean_U_init=[], metocean_Hs_init=[], metocean_Tp_init=[]):

#     # Default Runtime
#     T      = 240.
#     TStart = 120.
#     # T      = 120.
#     # TStart = 60.
    
#     # Overwrite for testing
#     if TMax < T:
#         T      = TMax
#         TStart = 0.

#     # Run conditions for points which will be used for a cubic polynomial fit
#     # U = [10.]
#     U = [4.,8.,9.,10.]
#     omega = np.interp(U, U_init, Omega_init)
#     pitch = np.interp(U, U_init, pitch_init)

#     # Check if floating
#     floating_dof = [fst_vt['ElastoDyn']['PtfmSgDOF'], fst_vt['ElastoDyn']['PtfmSwDOF'], fst_vt['ElastoDyn']['PtfmHvDOF'], fst_vt['ElastoDyn']['PtfmRDOF'], fst_vt['ElastoDyn']['PtfmPDOF'], fst_vt['ElastoDyn']['PtfmYDOF']]
#     if any(floating_dof):
#         floating = True
#         if ptfm_U_init == []:
#             ptfm_U_init     = [4., 5., 6., 7., 8., 9., 10., 10.5, 11., 12., 14., 19., 24.]
#             ptfm_surge_init = [3.8758245863838807, 5.57895688031965, 7.619719770801395, 9.974666446553552, 12.675469235464321, 16.173740623041965, 20.069526574594757, 22.141906121375552, 23.835466098954708, 22.976075549477354, 17.742743260748373, 14.464576583154068, 14.430969814391759]
#             ptfm_heave_init = [0.030777174904620515, 0.008329930604820483, -0.022973502300090893, -0.06506947653943342, -0.12101317451310406, -0.20589689839069836, -0.3169518280533253, -0.3831692055885472, -0.4409624802614755, -0.41411738171337675, -0.2375323506471747, -0.1156867221814119, -0.07029955933167854]
#             ptfm_pitch_init = [0.7519976895165884, 1.104483050851386, 1.5180416334025146, 1.9864587671004394, 2.5152769741130134, 3.1937704945765795, 3.951314212429935, 4.357929703098016, 4.693765745171944, 4.568760630312074, 3.495057478277534, 2.779958240049992, 2.69008798174216]
#         if metocean_U_init == []:
#             metocean_U_init  = [4.00, 6.00, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 24.00]
#             metocean_Hs_init = [1.908567568, 1.960162595, 2.062722244, 2.224539415, 2.489931091, 2.802984019, 3.182301485, 3.652236101, 4.182596165, 4.695439504, 5.422289377]
#             metocean_Tp_init = [12.23645701, 12.14497777, 11.90254947, 11.5196666, 11.05403739, 10.65483551, 10.27562225, 10.13693777, 10.27842325, 10.11660396, 10.96177917]

#         ptfm_heave = np.interp(U, ptfm_U_init, ptfm_heave_init)
#         ptfm_surge = np.interp(U, ptfm_U_init, ptfm_surge_init)
#         ptfm_pitch = np.interp(U, ptfm_U_init, ptfm_pitch_init)
#         metocean_Hs = np.interp(U, metocean_U_init, metocean_Hs_init)
#         metocean_Tp = np.interp(U, metocean_U_init, metocean_Tp_init)
#     else:
#         floating = False

#     case_inputs = {}
#     # simulation settings
#     # case_inputs[("ElastoDyn","PtfmSgDOF")]     = {'vals':['False'], 'group':0}
#     # case_inputs[("ElastoDyn","PtfmHvDOF")]     = {'vals':['False'], 'group':0}
#     # case_inputs[("ElastoDyn","PtfmPDOF")]     = {'vals':['False'], 'group':0}
#     case_inputs[("ElastoDyn","PtfmSwDOF")]     = {'vals':['False'], 'group':0}
#     case_inputs[("ElastoDyn","PtfmRDOF")]     = {'vals':['False'], 'group':0}
#     case_inputs[("ElastoDyn","PtfmYDOF")]     = {'vals':['False'], 'group':0}

#     case_inputs[("Fst","TMax")] = {'vals':[T], 'group':0}
#     case_inputs[("Fst","TStart")] = {'vals':[TStart], 'group':0}
#     case_inputs[("ElastoDyn","YawDOF")]      = {'vals':['True'], 'group':0}
#     case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':['True'], 'group':0}
#     case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':['True'], 'group':0}
#     case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':['True'], 'group':0}
#     case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':['False'], 'group':0}
#     case_inputs[("ElastoDyn","GenDOF")]      = {'vals':['True'], 'group':0} 
#     case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':['False'], 'group':0}
#     case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':['False'], 'group':0}
#     case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':['False'], 'group':0}
#     case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':['False'], 'group':0}
#     case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
#     case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}
#     case_inputs[("ServoDyn","YCMode")]       = {'vals':[5], 'group':0}
#     case_inputs[("AeroDyn15","WakeMod")]     = {'vals':[1], 'group':0}
#     case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[2], 'group':0}
#     case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[0], 'group':0}
#     case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':['False'], 'group':0}
#     case_inputs[("AeroDyn15","TwrAero")]     = {'vals':['False'], 'group':0}
#     case_inputs[("AeroDyn15","SkewMod")]     = {'vals':[1], 'group':0}
#     case_inputs[("AeroDyn15","TipLoss")]     = {'vals':['True'], 'group':0}
#     case_inputs[("AeroDyn15","HubLoss")]     = {'vals':['True'], 'group':0}
#     case_inputs[("AeroDyn15","TanInd")]      = {'vals':['True'], 'group':0}
#     case_inputs[("AeroDyn15","AIDrag")]      = {'vals':['True'], 'group':0}
#     case_inputs[("AeroDyn15","TIDrag")]      = {'vals':['True'], 'group':0}
#     case_inputs[("AeroDyn15","IndToler")]    = {'vals':[1.e-5], 'group':0}
#     case_inputs[("AeroDyn15","MaxIter")]     = {'vals':[5000], 'group':0}
#     case_inputs[("AeroDyn15","UseBlCm")]     = {'vals':['True'], 'group':0}
#     # inital conditions
#     case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
#     case_inputs[("InflowWind","HWindSpeed")] = {'vals':U, 'group':1}
#     case_inputs[("ElastoDyn","RotSpeed")] = {'vals':omega, 'group':1}
#     case_inputs[("ElastoDyn","BlPitch1")] = {'vals':pitch, 'group':1}
#     case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
#     case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
#     if floating == True:
#         case_inputs[("ElastoDyn","PtfmSurge")] = {'vals':ptfm_surge, 'group':1}
#         case_inputs[("ElastoDyn","PtfmHeave")] = {'vals':ptfm_heave, 'group':1}
#         case_inputs[("ElastoDyn","PtfmPitch")] = {'vals':ptfm_pitch, 'group':1}
#         case_inputs[("HydroDyn","WaveHs")] = {'vals':metocean_Hs, 'group':1}
#         case_inputs[("HydroDyn","WaveTp")] = {'vals':metocean_Tp, 'group':1}
#         case_inputs[("HydroDyn","RdtnDT")] = {'vals':[fst_vt["Fst"]["DT"]], 'group':0}
#         case_inputs[("HydroDyn","WaveMod")] = {'vals':[1], 'group':0}

#     from CaseGen_General import CaseGen_General
#     case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=runDir, namebase=namebase)

#     channels = ['Wind1VelX','GenPwr']

#     return case_list, case_name_list, channels

def power_curve(fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, Vrated, U_init=[], Omega_init=[], pitch_init=[], Turbsim_exe='', ptfm_U_init=[], ptfm_pitch_init=[], ptfm_surge_init=[], ptfm_heave_init=[], metocean_U_init=[], metocean_Hs_init=[], metocean_Tp_init=[], V_R25=0.):

    # Default Runtime
    T      = 360.
    TStart = 120.
    # T      = 120.
    # TStart = 60.
    
    # Overwrite for testing
    if TMax < T:
        T      = TMax
        TStart = 0.

    # Run conditions
    U_all = list(sorted([4., 6., 8., 9., 10., 10.5, 11., 11.5, 11.75, 12., 12.5, 13., 14., 19., 25., Vrated]))
    if V_R25 != 0.:
        U_all.append(V_R25)
        U_all = list(sorted(U_all))
    U = [Vi for Vi in U_all if Vi <= Vrated]
    # print(U)

    # dt = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    dt = [0.01]*len(U)

    # U = [4.,8.,9.,10.]
    omega = np.interp(U, U_init, Omega_init)
    pitch = np.interp(U, U_init, pitch_init)
    for i, (omegai, pitchi) in enumerate(zip(omega, pitch)):
        if pitchi > 0. and omegai < Omega_init[-1]:
            pitch[i] = 0.

    # Check if floating
    floating_dof = [fst_vt['ElastoDyn']['PtfmSgDOF'], fst_vt['ElastoDyn']['PtfmSwDOF'], fst_vt['ElastoDyn']['PtfmHvDOF'], fst_vt['ElastoDyn']['PtfmRDOF'], fst_vt['ElastoDyn']['PtfmPDOF'], fst_vt['ElastoDyn']['PtfmYDOF']]
    if any(floating_dof):
        floating = True
        if ptfm_U_init == []:
            ptfm_U_init     = [3., 5., 6., 7., 8., 9., 10., 10.5, 11., 12., 14., 19., 25.]
            ptfm_surge_init = [3.8758245863838807, 5.57895688031965, 7.619719770801395, 9.974666446553552, 12.675469235464321, 16.173740623041965, 20.069526574594757, 22.141906121375552, 23.835466098954708, 22.976075549477354, 17.742743260748373, 14.464576583154068, 14.430969814391759]
            ptfm_heave_init = [0.030777174904620515, 0.008329930604820483, -0.022973502300090893, -0.06506947653943342, -0.12101317451310406, -0.20589689839069836, -0.3169518280533253, -0.3831692055885472, -0.4409624802614755, -0.41411738171337675, -0.2375323506471747, -0.1156867221814119, -0.07029955933167854]
            ptfm_pitch_init = [0.7519976895165884, 1.104483050851386, 1.5180416334025146, 1.9864587671004394, 2.5152769741130134, 3.1937704945765795, 3.951314212429935, 4.357929703098016, 4.693765745171944, 4.568760630312074, 3.495057478277534, 2.779958240049992, 2.69008798174216]
        if metocean_U_init == []:
            metocean_U_init  = [3.00, 6.00, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 25.00]
            metocean_Hs_init = [1.908567568, 1.960162595, 2.062722244, 2.224539415, 2.489931091, 2.802984019, 3.182301485, 3.652236101, 4.182596165, 4.695439504, 5.422289377]
            metocean_Tp_init = [12.23645701, 12.14497777, 11.90254947, 11.5196666, 11.05403739, 10.65483551, 10.27562225, 10.13693777, 10.27842325, 10.11660396, 10.96177917]

        ptfm_heave = np.interp(U, ptfm_U_init, ptfm_heave_init)
        ptfm_surge = np.interp(U, ptfm_U_init, ptfm_surge_init)
        ptfm_pitch = np.interp(U, ptfm_U_init, ptfm_pitch_init)
        metocean_Hs = np.interp(U, metocean_U_init, metocean_Hs_init)
        metocean_Tp = np.interp(U, metocean_U_init, metocean_Tp_init)
    else:
        floating = False

    case_inputs = {}
    # simulation settings
    # case_inputs[("ElastoDyn","PtfmSgDOF")]     = {'vals':['False'], 'group':0}
    # case_inputs[("ElastoDyn","PtfmHvDOF")]     = {'vals':['False'], 'group':0}
    # case_inputs[("ElastoDyn","PtfmPDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmSwDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmRDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","PtfmYDOF")]     = {'vals':['False'], 'group':0}

    case_inputs[("Fst","TMax")] = {'vals':[T], 'group':0}
    case_inputs[("Fst","TStart")] = {'vals':[TStart], 'group':0}
    case_inputs[("Fst","DT")] = {'vals':dt, 'group':1}
    case_inputs[("ElastoDyn","YawDOF")]      = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","GenDOF")]      = {'vals':['True'], 'group':0} 
    case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","YCMode")]       = {'vals':[5], 'group':0}
    case_inputs[("AeroDyn15","WakeMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[2], 'group':0}
    case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[0], 'group':0}
    case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","TwrAero")]     = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","SkewMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TipLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","HubLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TanInd")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","AIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","IndToler")]    = {'vals':[1.e-5], 'group':0}
    case_inputs[("AeroDyn15","MaxIter")]     = {'vals':[5000], 'group':0}
    case_inputs[("AeroDyn15","UseBlCm")]     = {'vals':['True'], 'group':0}
    # inital conditions
    case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
    case_inputs[("InflowWind","HWindSpeed")] = {'vals':U, 'group':1}
    case_inputs[("ElastoDyn","RotSpeed")] = {'vals':omega, 'group':1}
    case_inputs[("ElastoDyn","BlPitch1")] = {'vals':pitch, 'group':1}
    case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
    case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
    if floating == True:
        case_inputs[("ElastoDyn","PtfmSurge")] = {'vals':ptfm_surge, 'group':1}
        case_inputs[("ElastoDyn","PtfmHeave")] = {'vals':ptfm_heave, 'group':1}
        case_inputs[("ElastoDyn","PtfmPitch")] = {'vals':ptfm_pitch, 'group':1}
        case_inputs[("HydroDyn","WaveHs")] = {'vals':metocean_Hs, 'group':1}
        case_inputs[("HydroDyn","WaveTp")] = {'vals':metocean_Tp, 'group':1}
        case_inputs[("HydroDyn","RdtnDT")] = {'vals':dt, 'group':1}
        case_inputs[("HydroDyn","WaveMod")] = {'vals':[1], 'group':0}

    from wisdem.aeroelasticse.CaseGen_General import CaseGen_General
    case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=runDir, namebase=namebase)

    channels = ['Wind1VelX','GenPwr',"RtAeroCp", "RotTorq", "RotThrust", "RotSpeed", "BldPitch1"]

    return case_list, case_name_list, channels

def RotorSE_rated(fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, Vrated, U_init=[], Omega_init=[], pitch_init=[], Turbsim_exe='', ptfm_U_init=[], ptfm_pitch_init=[], ptfm_surge_init=[], ptfm_heave_init=[], metocean_U_init=[], metocean_Hs_init=[], metocean_Tp_init=[]):

    # Default Runtime
    T      = 240.
    TStart = 120.

    # dt = 0.001
    dt = 0.01
    
    # Overwrite for testing
    if TMax < T:
        T      = TMax
        TStart = 0.

    omega = np.interp(Vrated, U_init, Omega_init)
    pitch = np.interp(Vrated, U_init, pitch_init)

    # Check if floating
    floating_dof = [fst_vt['ElastoDyn']['PtfmSgDOF'], fst_vt['ElastoDyn']['PtfmSwDOF'], fst_vt['ElastoDyn']['PtfmHvDOF'], fst_vt['ElastoDyn']['PtfmRDOF'], fst_vt['ElastoDyn']['PtfmPDOF'], fst_vt['ElastoDyn']['PtfmYDOF']]
    if any(floating_dof):
        floating = True
        if ptfm_U_init == []:
            ptfm_U_init     = [4., 5., 6., 7., 8., 9., 10., 10.5, 11., 12., 14., 19., 24.]
            ptfm_surge_init = [3.8758245863838807, 5.57895688031965, 7.619719770801395, 9.974666446553552, 12.675469235464321, 16.173740623041965, 20.069526574594757, 22.141906121375552, 23.835466098954708, 22.976075549477354, 17.742743260748373, 14.464576583154068, 14.430969814391759]
            ptfm_heave_init = [0.030777174904620515, 0.008329930604820483, -0.022973502300090893, -0.06506947653943342, -0.12101317451310406, -0.20589689839069836, -0.3169518280533253, -0.3831692055885472, -0.4409624802614755, -0.41411738171337675, -0.2375323506471747, -0.1156867221814119, -0.07029955933167854]
            ptfm_pitch_init = [0.7519976895165884, 1.104483050851386, 1.5180416334025146, 1.9864587671004394, 2.5152769741130134, 3.1937704945765795, 3.951314212429935, 4.357929703098016, 4.693765745171944, 4.568760630312074, 3.495057478277534, 2.779958240049992, 2.69008798174216]
        if metocean_U_init == []:
            metocean_U_init  = [4.00, 6.00, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 24.00]
            metocean_Hs_init = [1.908567568, 1.960162595, 2.062722244, 2.224539415, 2.489931091, 2.802984019, 3.182301485, 3.652236101, 4.182596165, 4.695439504, 5.422289377]
            metocean_Tp_init = [12.23645701, 12.14497777, 11.90254947, 11.5196666, 11.05403739, 10.65483551, 10.27562225, 10.13693777, 10.27842325, 10.11660396, 10.96177917]

        ptfm_heave = [np.interp(Vrated, ptfm_U_init, ptfm_heave_init)]
        ptfm_surge = [np.interp(Vrated, ptfm_U_init, ptfm_surge_init)]
        ptfm_pitch = [np.interp(Vrated, ptfm_U_init, ptfm_pitch_init)]
        metocean_Hs = [np.interp(Vrated, metocean_U_init, metocean_Hs_init)]
        metocean_Tp = [np.interp(Vrated, metocean_U_init, metocean_Tp_init)]
    else:
        floating = False

    case_inputs = {}
    case_inputs[("Fst","TMax")]              = {'vals':[T], 'group':0}
    case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
    case_inputs[("Fst","DT")]                = {'vals':[dt], 'group':0}
    case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}
    
    case_inputs[("InflowWind","WindType")]   = {'vals':[1], 'group':0}
    case_inputs[("InflowWind","HWindSpeed")] = {'vals':[Vrated], 'group':0}

    case_inputs[("ElastoDyn","RotSpeed")]    = {'vals':[omega], 'group':0}
    case_inputs[("ElastoDyn","BlPitch1")]    = {'vals':[pitch], 'group':0}
    case_inputs[("ElastoDyn","BlPitch2")]    = {'vals':[pitch], 'group':0}
    case_inputs[("ElastoDyn","BlPitch3")]    = {'vals':[pitch], 'group':0}
    case_inputs[("ElastoDyn","YawDOF")]      = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","GenDOF")]      = {'vals':['True'], 'group':0} 
    case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':['False'], 'group':0}

    case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}

    case_inputs[("AeroDyn15","WakeMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[2], 'group':0}
    case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[0], 'group':0}
    case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","TwrAero")]     = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","SkewMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TipLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","HubLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TanInd")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","AIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","IndToler")]    = {'vals':[1.e-5], 'group':0}
    case_inputs[("AeroDyn15","MaxIter")]     = {'vals':[5000], 'group':0}
    case_inputs[("AeroDyn15","UseBlCm")]     = {'vals':['True'], 'group':0}
    
    if floating == True:
        case_inputs[("ElastoDyn","PtfmSurge")] = {'vals':ptfm_surge, 'group':1}
        case_inputs[("ElastoDyn","PtfmHeave")] = {'vals':ptfm_heave, 'group':1}
        case_inputs[("ElastoDyn","PtfmPitch")] = {'vals':ptfm_pitch, 'group':1}
        case_inputs[("HydroDyn","WaveHs")] = {'vals':metocean_Hs, 'group':1}
        case_inputs[("HydroDyn","WaveTp")] = {'vals':metocean_Tp, 'group':1}
        case_inputs[("HydroDyn","RdtnDT")] = {'vals':[dt], 'group':0}
        case_inputs[("HydroDyn","WaveMod")] = {'vals':[1], 'group':0}

    namebase += '_rated'
    case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=runDir, namebase=namebase)

    channels  = ["TipDxc1", "TipDyc1"]
    channels += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxc3", "RootMyc3", "RootMzc3"]
    channels += ["RootFxc1", "RootFyc1", "RootFzc1", "RootFxc2", "RootFyc2", "RootFzc2", "RootFxc3", "RootFyc3", "RootFzc3"]
    channels += ["RtAeroCp", "RotTorq", "RotThrust", "RotSpeed"]

    return case_list, case_name_list, channels

def RotorSE_DLC_1_4_Rated(fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, Vrated, U_init=[], Omega_init=[], pitch_init=[], Turbsim_exe=''):

    # Default Runtime
    T      = 60.
    TStart = 30.
    # TStart = 0.
    
    # Overwrite for testing
    if TMax < T:
        T      = TMax
        TStart = 0.


    iec = CaseGen_IEC()
    iec.init_cond[("ElastoDyn","RotSpeed")] = {'U':  U_init}
    iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = Omega_init
    iec.init_cond[("ElastoDyn","BlPitch1")] = {'U':  U_init}
    iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = pitch_init
    iec.init_cond[("ElastoDyn","BlPitch2")] = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.init_cond[("ElastoDyn","BlPitch3")] = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.Turbine_Class = turbine_class
    iec.Turbulence_Class = turbulence_class
    iec.D = fst_vt['ElastoDyn']['TipRad']*2.
    iec.z_hub = fst_vt['InflowWind']['RefHt']

    iec.dlc_inputs = {}
    iec.dlc_inputs['DLC']   = [1.4]
    iec.dlc_inputs['U']     = [[Vrated]]
    iec.dlc_inputs['Seeds'] = [[]]
    iec.dlc_inputs['Yaw']   = [[]]
    iec.transient_dir_change        = '-'  # '+','-','both': sign for transient events in EDC, EWS
    iec.transient_shear_orientation = 'v'  # 'v','h','both': vertical or horizontal shear for EWS

    iec.wind_dir        = runDir
    iec.case_name_base  = namebase + '_gust'
    iec.Turbsim_exe     = ''
    iec.debug_level     = 0
    iec.parallel_windfile_gen = False
    iec.run_dir         = runDir

    case_inputs = {}
    case_inputs[("Fst","TMax")]              = {'vals':[T], 'group':0}
    case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
    case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}

    case_inputs[("ElastoDyn","YawDOF")]      = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","GenDOF")]      = {'vals':['True'], 'group':0} 
    case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':['False'], 'group':0}

    case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","YCMode")]       = {'vals':[5], 'group':0}

    case_inputs[("AeroDyn15","WakeMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[2], 'group':0}
    case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[0], 'group':0}
    case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","TwrAero")]     = {'vals':['False'], 'group':0}

    case_inputs[("AeroDyn15","SkewMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TipLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","HubLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TanInd")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","AIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","IndToler")]    = {'vals':[1.e-5], 'group':0}
    case_inputs[("AeroDyn15","MaxIter")]     = {'vals':[5000], 'group':0}
    case_inputs[("AeroDyn15","UseBlCm")]     = {'vals':['True'], 'group':0}
    
    case_list, case_name_list = iec.execute(case_inputs=case_inputs)

    channels  = ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxc3", "TipDyc3", "TipDzc3"]
    channels += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxc3", "RootMyc3", "RootMzc3"]
    channels += ["RootFxc1", "RootFyc1", "RootFzc1", "RootFxc2", "RootFyc2", "RootFzc2", "RootFxc3", "RootFyc3", "RootFzc3"]
    channels += ["RtAeroCp", "RotTorq", "RotThrust", "RotSpeed", "NacYaw"]

    channels += ["B1N1Fx", "B1N2Fx", "B1N3Fx", "B1N4Fx", "B1N5Fx", "B1N6Fx", "B1N7Fx", "B1N8Fx", "B1N9Fx"]
    channels += ["B1N1Fy", "B1N2Fy", "B1N3Fy", "B1N4Fy", "B1N5Fy", "B1N6Fy", "B1N7Fy", "B1N8Fy", "B1N9Fy"]

    return case_list, case_name_list, channels

def RotorSE_DLC_7_1_Steady(fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, U, U_init=[], Omega_init=[], pitch_init=[], Turbsim_exe=''):
    # Extreme 1yr return period wind speed with a power fault resulting in the blade not feathering

    # Default Runtime
    T      = 60.
    TStart = 30.
    
    # Overwrite for testing
    if TMax < T:
        T      = TMax
        TStart = 0.

    Pitch = 0.
    Omega = 0.

    case_inputs = {}
    case_inputs[("Fst","TMax")]              = {'vals':[T], 'group':0}
    case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
    case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}
    
    case_inputs[("InflowWind","WindType")]   = {'vals':[1], 'group':0}
    case_inputs[("InflowWind","HWindSpeed")] = {'vals':[U], 'group':0}
    case_inputs[("InflowWind","PLexp")] = {'vals':[0.11], 'group':0}

    case_inputs[("ElastoDyn","RotSpeed")]    = {'vals':[Omega], 'group':0}
    case_inputs[("ElastoDyn","BlPitch1")]    = {'vals':[Pitch], 'group':0}
    case_inputs[("ElastoDyn","BlPitch2")]    = {'vals':[Pitch], 'group':0}
    case_inputs[("ElastoDyn","BlPitch3")]    = {'vals':[Pitch], 'group':0}
    case_inputs[("ElastoDyn","YawDOF")]      = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","GenDOF")]      = {'vals':['False'], 'group':0} 
    case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':['False'], 'group':0}

    case_inputs[("ServoDyn","PCMode")]       = {'vals':[0], 'group':0}
    case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","YCMode")]       = {'vals':[5], 'group':0}

    case_inputs[("AeroDyn15","WakeMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[0], 'group':0}
    case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","TwrAero")]     = {'vals':['False'], 'group':0}

    case_inputs[("AeroDyn15","SkewMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TipLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","HubLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TanInd")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","AIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","TIDrag")]      = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","IndToler")]    = {'vals':[1.e-5], 'group':0}
    case_inputs[("AeroDyn15","MaxIter")]     = {'vals':[5000], 'group':0}
    case_inputs[("AeroDyn15","UseBlCm")]     = {'vals':['True'], 'group':0}
    

    namebase += '_idle50yr'
    case_list, case_name_list = CaseGen_General(case_inputs, namebase=namebase, save_matrix=False)

    channels  = ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxc3", "TipDyc3", "TipDzc3"]
    channels += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxc3", "RootMyc3", "RootMzc3"]
    channels += ["RootFxc1", "RootFyc1", "RootFzc1", "RootFxc2", "RootFyc2", "RootFzc2", "RootFxc3", "RootFyc3", "RootFzc3"]
    channels += ["RtAeroCp", "RotTorq", "RotThrust", "RotSpeed", "NacYaw"]

    channels += ["B1N1Fx", "B1N2Fx", "B1N3Fx", "B1N4Fx", "B1N5Fx", "B1N6Fx", "B1N7Fx", "B1N8Fx", "B1N9Fx"]
    channels += ["B1N1Fy", "B1N2Fy", "B1N3Fy", "B1N4Fy", "B1N5Fy", "B1N6Fy", "B1N7Fy", "B1N8Fy", "B1N9Fy"]

    return case_list, case_name_list, channels


def RotorSE_DLC_1_1_Turb(fst_vt, runDir, namebase, TMax, turbine_class, turbulence_class, U, U_init=[], Omega_init=[], pitch_init=[], Turbsim_exe='', debug_level=0, cores=0, mpi_run=False, mpi_comm_map_down=[]):
    
    # Default Runtime
    T      = 60.
    TStart = 30.
    
    # Overwrite for testing
    if TMax < T:
        T      = TMax
        TStart = 0.


    iec = CaseGen_IEC()
    iec.init_cond[("ElastoDyn","RotSpeed")] = {'U':  U_init}
    iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = [0.95*omega_i for omega_i in Omega_init]
    iec.init_cond[("ElastoDyn","BlPitch1")] = {'U':  U_init}
    iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = pitch_init
    iec.init_cond[("ElastoDyn","BlPitch2")] = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.init_cond[("ElastoDyn","BlPitch3")] = iec.init_cond[("ElastoDyn","BlPitch1")]
    iec.Turbine_Class = turbine_class
    iec.Turbulence_Class = turbulence_class
    iec.D = fst_vt['ElastoDyn']['TipRad']*2.
    iec.z_hub = fst_vt['InflowWind']['RefHt']

    iec.dlc_inputs = {}
    iec.dlc_inputs['DLC']   = [1.1]
    iec.dlc_inputs['U']     = [[U]]
    # iec.dlc_inputs['Seeds'] = [[1]]
    iec.dlc_inputs['Seeds'] = [[1]] # nothing special about these seeds, randomly generated
    iec.dlc_inputs['Yaw']   = [[]]
    iec.transient_dir_change        = '-'  # '+','-','both': sign for transient events in EDC, EWS
    iec.transient_shear_orientation = 'v'  # 'v','h','both': vertical or horizontal shear for EWS

    iec.wind_dir        = runDir
    iec.case_name_base  = namebase + '_turb'
    iec.Turbsim_exe     = Turbsim_exe
    iec.debug_level     = debug_level
    iec.cores           = cores
    iec.run_dir         = runDir
    iec.overwrite       = True
    # iec.overwrite       = False
    if cores > 1:
        iec.parallel_windfile_gen = True
    else:
        iec.parallel_windfile_gen = False

    # mpi_run = False
    if mpi_run:
        iec.mpi_run           = mpi_run
        iec.comm_map_down = mpi_comm_map_down

    case_inputs = {}
    case_inputs[("Fst","TMax")]              = {'vals':[T], 'group':0}
    case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
    case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}

    case_inputs[("ElastoDyn","YawDOF")]      = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF1")]    = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","FlapDOF2")]    = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","EdgeDOF")]     = {'vals':['True'], 'group':0}
    case_inputs[("ElastoDyn","DrTrDOF")]     = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","GenDOF")]      = {'vals':['True'], 'group':0} 
    case_inputs[("ElastoDyn","TwFADOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwFADOF2")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF1")]    = {'vals':['False'], 'group':0}
    case_inputs[("ElastoDyn","TwSSDOF2")]    = {'vals':['False'], 'group':0}

    case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}
    case_inputs[("ServoDyn","YCMode")]       = {'vals':[5], 'group':0}

    case_inputs[("AeroDyn15","WakeMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","AFAeroMod")]   = {'vals':[2], 'group':0}
    case_inputs[("AeroDyn15","TwrPotent")]   = {'vals':[0], 'group':0}
    case_inputs[("AeroDyn15","TwrShadow")]   = {'vals':['False'], 'group':0}
    case_inputs[("AeroDyn15","TwrAero")]     = {'vals':['False'], 'group':0}

    case_inputs[("AeroDyn15","SkewMod")]     = {'vals':[1], 'group':0}
    case_inputs[("AeroDyn15","TipLoss")]     = {'vals':['True'], 'group':0}
    case_inputs[("AeroDyn15","HubLoss")]     = {'vals':['True'], 'group':0}
    # case_inputs[("AeroDyn15","TanInd")]      = {'vals':['True'], 'group':0}
    # case_inputs[("AeroDyn15","AIDrag")]      = {'vals':['True'], 'group':0}
    # case_inputs[("AeroDyn15","TIDrag")]      = {'vals':['True'], 'group':0}
    # case_inputs[("AeroDyn15","IndToler")]    = {'vals':[1.e-5], 'group':0}
    # case_inputs[("AeroDyn15","MaxIter")]     = {'vals':[5000], 'group':0}
    case_inputs[("AeroDyn15","UseBlCm")]     = {'vals':['True'], 'group':0}
    
    case_list, case_name_list = iec.execute(case_inputs=case_inputs)

    channels  = ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxc2", "TipDyc2", "TipDzc2", "TipDxc3", "TipDyc3", "TipDzc3"]
    channels += ["RootMxc1", "RootMyc1", "RootMzc1", "RootMxc2", "RootMyc2", "RootMzc2", "RootMxc3", "RootMyc3", "RootMzc3"]
    channels += ["RootFxc1", "RootFyc1", "RootFzc1", "RootFxc2", "RootFyc2", "RootFzc2", "RootFxc3", "RootFyc3", "RootFzc3"]
    channels += ["RtAeroCp", "RotTorq", "RotThrust", "RotSpeed", "NacYaw"]

    channels += ["B1N1Fx", "B1N2Fx", "B1N3Fx", "B1N4Fx", "B1N5Fx", "B1N6Fx", "B1N7Fx", "B1N8Fx", "B1N9Fx"]
    channels += ["B1N1Fy", "B1N2Fy", "B1N3Fy", "B1N4Fy", "B1N5Fy", "B1N6Fy", "B1N7Fy", "B1N8Fy", "B1N9Fy"]

    return case_list, case_name_list, channels


if __name__ == "__main__":

    # power_curve()

    case_list, case_name_list = RotorSE_rated('test', 60., 11., 12.1, 0.)


