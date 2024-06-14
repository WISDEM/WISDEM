import logging

import numpy as np
from scipy.interpolate import PchipInterpolator

import wisdem.commonse.utilities as util
from wisdem.rotorse.geometry_tools.geometry import AirfoilShape

logger = logging.getLogger("wisdem/weis")


def yaml2openmdao(wt_opt, modeling_options, wt_init, opt_options):
    # Function to assign values to the openmdao group Wind_Turbine and all its components

    offshore = modeling_options["flags"]["offshore"]

    # These are the required components
    assembly = wt_init["assembly"]
    wt_opt = assign_configuration_values(wt_opt, assembly, opt_options)

    materials = wt_init["materials"]
    wt_opt = assign_material_values(wt_opt, modeling_options, materials)

    # Now all of the optional components
    if modeling_options["flags"]["environment"]:
        environment = wt_init["environment"]
        blade_flag = modeling_options["flags"]["blade"]
        wt_opt = assign_environment_values(wt_opt, environment, offshore, blade_flag)
    else:
        environment = {}

    if modeling_options["flags"]["blade"]:
        blade = wt_init["components"]["blade"]
        blade_DV = opt_options['design_variables']['blade']
        wt_opt = assign_blade_values(wt_opt, modeling_options, blade_DV, blade)
    else:
        blade = {}

    if modeling_options["flags"]["airfoils"]:
        airfoils = wt_init["airfoils"]
        wt_opt = assign_airfoil_values(wt_opt, modeling_options, airfoils)
    else:
        airfoils = {}

    if modeling_options["flags"]["control"]:
        control = wt_init["control"]
        wt_opt = assign_control_values(wt_opt, modeling_options, control)
    else:
        control = {}

    if modeling_options["flags"]["hub"] or modeling_options["flags"]["blade"]:
        hub = wt_init["components"]["hub"]
        wt_opt = assign_hub_values(wt_opt, hub, modeling_options["flags"])
    else:
        hub = {}

    if modeling_options["flags"]["nacelle"] or modeling_options["flags"]["blade"]:
        nacelle = wt_init["components"]["nacelle"]
        wt_opt = assign_nacelle_values(wt_opt, modeling_options, nacelle, modeling_options["flags"])

        if modeling_options["flags"]["generator"]:
            wt_opt = assign_generator_values(wt_opt, modeling_options, nacelle)
    else:
        nacelle = {}

    if modeling_options["flags"]["RNA"]:
        RNA = wt_init["components"]["RNA"]
    else:
        RNA = {}

    if modeling_options["flags"]["tower"]:
        tower = wt_init["components"]["tower"]
        wt_opt = assign_tower_values(wt_opt, modeling_options, tower)
    else:
        tower = {}

    if modeling_options["flags"]["monopile"]:
        monopile = wt_init["components"]["monopile"]
        wt_opt = assign_monopile_values(wt_opt, modeling_options, monopile)
    else:
        monopile = {}

    if modeling_options["flags"]["jacket"]:
        jacket = wt_init["components"]["jacket"]
        wt_opt = assign_jacket_values(wt_opt, modeling_options, jacket)
    else:
        jacket = {}

    if modeling_options["flags"]["floating_platform"]:
        floating_platform = wt_init["components"]["floating_platform"]
        wt_opt = assign_floating_values(wt_opt, modeling_options, floating_platform, opt_options)
        mooring = wt_init["components"]["mooring"]
        wt_opt = assign_mooring_values(wt_opt, modeling_options, mooring)

    if modeling_options["flags"]["bos"]:
        bos = wt_init["bos"]
        wt_opt = assign_bos_values(wt_opt, bos, offshore)
    else:
        bos = {}

    if modeling_options["flags"]["costs"]:
        costs = wt_init["costs"]
        wt_opt = assign_costs_values(wt_opt, costs)
    else:
        costs = {}

    return wt_opt


def assign_blade_values(wt_opt, modeling_options, blade_DV, blade):
    # Function to assign values to the openmdao group Blade
    blade_DV_aero = blade_DV['aero_shape']
    wt_opt = assign_outer_shape_bem_values(wt_opt, modeling_options, blade_DV_aero, blade["outer_shape_bem"])
    wt_opt = assign_internal_structure_2d_fem_values(wt_opt, modeling_options, blade["internal_structure_2d_fem"])
    wt_opt = assign_te_flaps_values(wt_opt, modeling_options, blade)

    return wt_opt


def assign_outer_shape_bem_values(wt_opt, modeling_options, blade_DV_aero, outer_shape_bem):
    # Function to assign values to the openmdao component Blade_Outer_Shape_BEM

    nd_span = modeling_options["WISDEM"]["RotorSE"]["nd_span"]

    wt_opt["blade.outer_shape_bem.af_position"] = outer_shape_bem["airfoil_position"]["grid"]
    wt_opt["blade.opt_var.af_position"] = outer_shape_bem["airfoil_position"]["grid"]

    wt_opt["blade.outer_shape_bem.s_default"] = nd_span
    wt_opt["blade.outer_shape_bem.chord_yaml"] = PchipInterpolator(
        outer_shape_bem["chord"]["grid"], outer_shape_bem["chord"]["values"]
    )(nd_span)
    wt_opt["blade.outer_shape_bem.twist_yaml"] = PchipInterpolator(
        outer_shape_bem["twist"]["grid"], outer_shape_bem["twist"]["values"]
    )(nd_span)
    wt_opt["blade.outer_shape_bem.pitch_axis_yaml"] = PchipInterpolator(
        outer_shape_bem["pitch_axis"]["grid"], outer_shape_bem["pitch_axis"]["values"]
    )(nd_span)
    af_opt_flag = blade_DV_aero['af_positions']['flag']
    if 'rthick' in outer_shape_bem and af_opt_flag == False:
        # If rthick is defined in input yaml and we are NOT optimizing airfoil positions
        wt_opt["blade.outer_shape_bem.r_thick_yaml"] = PchipInterpolator(
            outer_shape_bem["rthick"]["grid"], outer_shape_bem["rthick"]["values"]
        )(nd_span)
    elif 'rthick' in outer_shape_bem and af_opt_flag == True:
        logger.debug('rthick field in input geometry yaml is specified but neglected since you are optimizing airfoil positions')
    else:
        logger.debug('rthick field in input geometry yaml not specified. rthick is reconstructed from discrete airfoil positions')
    wt_opt["blade.outer_shape_bem.ref_axis_yaml"][:, 0] = PchipInterpolator(
        outer_shape_bem["reference_axis"]["x"]["grid"], outer_shape_bem["reference_axis"]["x"]["values"]
    )(nd_span)
    wt_opt["blade.outer_shape_bem.ref_axis_yaml"][:, 1] = PchipInterpolator(
        outer_shape_bem["reference_axis"]["y"]["grid"], outer_shape_bem["reference_axis"]["y"]["values"]
    )(nd_span)
    wt_opt["blade.outer_shape_bem.ref_axis_yaml"][:, 2] = PchipInterpolator(
        outer_shape_bem["reference_axis"]["z"]["grid"], outer_shape_bem["reference_axis"]["z"]["values"]
    )(nd_span)

    # # Smoothing of the shapes
    # # Chord
    # chord_init      = wt_opt['blade.outer_shape_bem.chord']
    # s_interp_c      = np.array([0.0, 0.05, 0.2, 0.35, 0.65, 0.9, 1.0 ])
    # f_interp1       = interp1d(nd_span,chord_init)
    # chord_int1      = f_interp1(s_interp_c)
    # f_interp2       = PchipInterpolator(s_interp_c,chord_int1)
    # chord_int2      = f_interp2(nd_span)

    # import matplotlib.pyplot as plt
    # fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
    # axc.plot(nd_span, chord_init, c='k', label='Initial')
    # axc.plot(s_interp_c, chord_int1, 'ko', label='Interp Points')
    # axc.plot(nd_span, chord_int2, c='b', label='PCHIP')
    # axc.set(xlabel='r/R' , ylabel='Chord (m)')
    # fig_name = 'interp_chord.png'
    # axc.legend()
    # # Planform
    # le_init = wt_opt['blade.outer_shape_bem.pitch_axis_yaml']*wt_opt['blade.outer_shape_bem.chord_yaml']
    # te_init = (1. - wt_opt['blade.outer_shape_bem.pitch_axis_yaml'])*wt_opt['blade.outer_shape_bem.chord_yaml']

    # s_interp_le     = np.array([0.0, 0.5, 0.8, 1.0])
    # f_interp1       = interp1d(wt_opt['blade.outer_shape_bem.s_default'],le_init)
    # le_int1         = f_interp1(s_interp_le)
    # f_interp2       = PchipInterpolator(s_interp_le,le_int1)
    # le_int2         = f_interp2(wt_opt['blade.outer_shape_bem.s_default'])

    # fpl, axpl  = plt.subplots(1,1,figsize=(5.3, 4))
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], -le_init, c='k', label='LE init')
    # axpl.plot(wt_opt['blade.outer_shape_bem.s_default'], te_init, c='k', label='TE init')
    # axpl.set(xlabel='r/R' , ylabel='Planform (m)')
    # axpl.legend()
    # plt.show()
    # # exit()
    # # np.savetxt('temp.txt', le_int2/wt_opt['blade.outer_shape_bem.chord'])

    # # # Twist
    # theta_init      = wt_opt['blade.outer_shape_bem.twist']
    # s_interp      = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0 ])
    # f_interp1       = interp1d(nd_span,theta_init)
    # theta_int1      = f_interp1(s_interp)
    # f_interp2       = PchipInterpolator(s_interp,theta_int1)
    # theta_int2      = f_interp2(nd_span)

    # import matplotlib.pyplot as plt
    # fc, axc  = plt.subplots(1,1,figsize=(5.3, 4))
    # axc.plot(nd_span, theta_init, c='k', label='Initial')
    # axc.plot(s_interp, theta_int1, 'ko', label='Interp Points')
    # axc.plot(nd_span, theta_int2, c='b', label='PCHIP')
    # axc.set(xlabel='r/R' , ylabel='Twist (deg)')
    # axc.legend()
    # plt.show()

    return wt_opt


def assign_internal_structure_2d_fem_values(wt_opt, modeling_options, internal_structure_2d_fem):
    # Function to assign values to the openmdao component Blade_Internal_Structure_2D_FEM

    n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
    n_webs = modeling_options["WISDEM"]["RotorSE"]["n_webs"]

    web_rotation = np.zeros((n_webs, n_span))
    web_offset_y_pa = np.zeros((n_webs, n_span))
    web_start_nd = np.zeros((n_webs, n_span))
    web_end_nd = np.zeros((n_webs, n_span))
    definition_web = np.zeros(n_webs)
    nd_span = wt_opt["blade.outer_shape_bem.s_default"]

    # Loop through the webs and interpolate spanwise the values
    for i in range(n_webs):
        if "rotation" in internal_structure_2d_fem["webs"][i] and "offset_y_pa" in internal_structure_2d_fem["webs"][i]:
            if "fixed" in internal_structure_2d_fem["webs"][i]["rotation"].keys():
                if internal_structure_2d_fem["webs"][i]["rotation"]["fixed"] == "twist":
                    definition_web[i] = 1
                else:
                    raise ValueError(
                        "Invalid rotation reference for web "
                        + modeling_options["WISDEM"]["RotorSE"]["web_name"][i]
                        + ". Please check the yaml input file"
                    )
            else:
                web_rotation[i, :] = np.nan_to_num(
                    PchipInterpolator(
                        internal_structure_2d_fem["webs"][i]["rotation"]["grid"],
                        internal_structure_2d_fem["webs"][i]["rotation"]["values"],
                        extrapolate=False,
                    )(nd_span)
                )
                definition_web[i] = 2
            web_offset_y_pa[i, :] = np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["webs"][i]["offset_y_pa"]["grid"],
                    internal_structure_2d_fem["webs"][i]["offset_y_pa"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
        elif "offset_plane" in internal_structure_2d_fem["webs"][i]:
            web_rotation[i, :] = np.ones_like(nd_span) * internal_structure_2d_fem["webs"][i]["offset_plane"]["blade_rotation"]
            web_offset_y_pa[i, :] = -np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["webs"][i]["offset_plane"]["offset"]["grid"],
                    internal_structure_2d_fem["webs"][i]["offset_plane"]["offset"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
            definition_web[i] = 4
        elif (
            "start_nd_arc" in internal_structure_2d_fem["webs"][i]
            and "end_nd_arc" in internal_structure_2d_fem["webs"][i]
        ):
            definition_web[i] = 3
            web_start_nd[i, :] = np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["webs"][i]["start_nd_arc"]["grid"],
                    internal_structure_2d_fem["webs"][i]["start_nd_arc"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
            web_end_nd[i, :] = np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["webs"][i]["end_nd_arc"]["grid"],
                    internal_structure_2d_fem["webs"][i]["end_nd_arc"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
        else:
            raise ValueError("Webs definition not supported. Please check the yaml input.")

    n_layers = modeling_options["WISDEM"]["RotorSE"]["n_layers"]
    layer_name = n_layers * [""]
    layer_mat = n_layers * [""]
    thickness = np.zeros((n_layers, n_span))
    orientation = np.zeros((n_layers, n_span))
    layer_rotation = np.zeros((n_layers, n_span))
    layer_offset_y_pa = np.zeros((n_layers, n_span))
    layer_width = np.zeros((n_layers, n_span))
    layer_midpoint_nd = np.zeros((n_layers, n_span))
    layer_start_nd = np.zeros((n_layers, n_span))
    layer_end_nd = np.zeros((n_layers, n_span))
    layer_web = np.zeros(n_layers)
    layer_side = n_layers * [""]
    definition_layer = np.zeros(n_layers)
    index_layer_start = np.zeros(n_layers)
    index_layer_end = np.zeros(n_layers)

    # Loop through the layers, interpolate along blade span, assign the inputs, and the definition flag
    for i in range(n_layers):
        layer_name[i] = modeling_options["WISDEM"]["RotorSE"]["layer_name"][i]
        layer_mat[i] = modeling_options["WISDEM"]["RotorSE"]["layer_mat"][i]
        thickness[i, :] = np.nan_to_num(
            PchipInterpolator(
                internal_structure_2d_fem["layers"][i]["thickness"]["grid"],
                internal_structure_2d_fem["layers"][i]["thickness"]["values"],
                extrapolate=False,
            )(nd_span)
        )
        orientation[i, :] = np.nan_to_num(
            PchipInterpolator(
                internal_structure_2d_fem["layers"][i]["fiber_orientation"]["grid"],
                internal_structure_2d_fem["layers"][i]["fiber_orientation"]["values"],
                extrapolate=False,
            )(nd_span)
        )
        if (
            "rotation" not in internal_structure_2d_fem["layers"][i]
            and "offset_y_pa" not in internal_structure_2d_fem["layers"][i]
            and "width" not in internal_structure_2d_fem["layers"][i]
            and "start_nd_arc" not in internal_structure_2d_fem["layers"][i]
            and "end_nd_arc" not in internal_structure_2d_fem["layers"][i]
            and "web" not in internal_structure_2d_fem["layers"][i]
            and "offset_plane" not in internal_structure_2d_fem["layers"][i]
        ):
            definition_layer[i] = 1

        if (
            "rotation" in internal_structure_2d_fem["layers"][i]
            and "offset_y_pa" in internal_structure_2d_fem["layers"][i]
            and "width" in internal_structure_2d_fem["layers"][i]
            and "side" in internal_structure_2d_fem["layers"][i]
        ):
            if "fixed" in internal_structure_2d_fem["layers"][i]["rotation"].keys():
                if internal_structure_2d_fem["layers"][i]["rotation"]["fixed"] == "twist":
                    definition_layer[i] = 2
                else:
                    raise ValueError(
                        "Invalid rotation reference for layer " + layer_name[i] + ". Please check the yaml input file."
                    )
            else:
                layer_rotation[i, :] = np.nan_to_num(
                    PchipInterpolator(
                        internal_structure_2d_fem["layers"][i]["rotation"]["grid"],
                        internal_structure_2d_fem["layers"][i]["rotation"]["values"],
                        extrapolate=False,
                    )(nd_span)
                )
                definition_layer[i] = 3
            layer_offset_y_pa[i, :] = np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["layers"][i]["offset_y_pa"]["grid"],
                    internal_structure_2d_fem["layers"][i]["offset_y_pa"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
            layer_width[i, :] = np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["layers"][i]["width"]["grid"],
                    internal_structure_2d_fem["layers"][i]["width"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
            layer_side[i] = internal_structure_2d_fem["layers"][i]["side"]
        if (
            "midpoint_nd_arc" in internal_structure_2d_fem["layers"][i]
            and "width" in internal_structure_2d_fem["layers"][i]
        ):
            if "fixed" in internal_structure_2d_fem["layers"][i]["midpoint_nd_arc"].keys():
                if internal_structure_2d_fem["layers"][i]["midpoint_nd_arc"]["fixed"] == "TE":
                    layer_midpoint_nd[i, :] = np.ones(n_span)
                    definition_layer[i] = 4
                elif internal_structure_2d_fem["layers"][i]["midpoint_nd_arc"]["fixed"] == "LE":
                    definition_layer[i] = 5
            else:
                layer_midpoint_nd[i, :] = np.nan_to_num(
                    PchipInterpolator(
                        internal_structure_2d_fem["layers"][i]["midpoint_nd_arc"]["grid"],
                        internal_structure_2d_fem["layers"][i]["midpoint_nd_arc"]["values"],
                        extrapolate=False,
                    )(nd_span)
                )
            layer_width[i, :] = np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["layers"][i]["width"]["grid"],
                    internal_structure_2d_fem["layers"][i]["width"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
        if "start_nd_arc" in internal_structure_2d_fem["layers"][i] and definition_layer[i] == 0:
            if "fixed" in internal_structure_2d_fem["layers"][i]["start_nd_arc"].keys():
                if internal_structure_2d_fem["layers"][i]["start_nd_arc"]["fixed"] == "TE":
                    layer_start_nd[i, :] = np.zeros(n_span)
                elif internal_structure_2d_fem["layers"][i]["start_nd_arc"]["fixed"] == "LE":
                    definition_layer[i] = 11
                else:
                    definition_layer[i] = 6
                    flag = False
                    for k in range(n_layers):
                        if layer_name[k] == internal_structure_2d_fem["layers"][i]["start_nd_arc"]["fixed"]:
                            index_layer_start[i] = k
                            flag = True
                            break
                    if flag == False:
                        raise ValueError(
                            "The start position of the layer "
                            + internal_structure_2d_fem["layers"][i]["name"]
                            + " is linked to the layer "
                            + internal_structure_2d_fem["layers"][i]["start_nd_arc"]["fixed"]
                            + " , but this layer does not exist in the yaml."
                        )
            else:
                layer_start_nd[i, :] = np.nan_to_num(
                    PchipInterpolator(
                        internal_structure_2d_fem["layers"][i]["start_nd_arc"]["grid"],
                        internal_structure_2d_fem["layers"][i]["start_nd_arc"]["values"],
                        extrapolate=False,
                    )(nd_span)
                )

                if np.min(layer_start_nd[i, :]) < 0.0 or np.max(layer_start_nd[i, :]) > 1.0:
                    raise Exception(
                        """The start_nd_arc layer cannot be smaller than 0 nor
                    larger than 1. Check the input yaml for layer """,
                        layer_name[i],
                    )

            if "end_nd_arc" in internal_structure_2d_fem["layers"][i]:
                if "fixed" in internal_structure_2d_fem["layers"][i]["end_nd_arc"].keys():
                    if internal_structure_2d_fem["layers"][i]["end_nd_arc"]["fixed"] == "TE":
                        layer_end_nd[i, :] = np.ones(n_span)
                        # raise ValueError('No need to fix element to TE, set it to 0.')
                    elif internal_structure_2d_fem["layers"][i]["end_nd_arc"]["fixed"] == "LE":
                        definition_layer[i] = 12
                    else:
                        flag = False
                        for k in range(n_layers):
                            if layer_name[k] == internal_structure_2d_fem["layers"][i]["end_nd_arc"]["fixed"]:
                                index_layer_end[i] = k
                                flag = True
                                break
                        if flag == False:
                            raise ValueError(
                                "The end position of the layer "
                                + internal_structure_2d_fem["layers"][i]["name"]
                                + " is linked to the layer "
                                + internal_structure_2d_fem["layers"][i]["start_nd_arc"]["fixed"]
                                + " , but this layer does not exist in the yaml."
                            )
            if "width" in internal_structure_2d_fem["layers"][i]:
                definition_layer[i] = 7
                layer_width[i, :] = np.nan_to_num(
                    PchipInterpolator(
                        internal_structure_2d_fem["layers"][i]["width"]["grid"],
                        internal_structure_2d_fem["layers"][i]["width"]["values"],
                        extrapolate=False,
                    )(nd_span)
                )

        if "end_nd_arc" in internal_structure_2d_fem["layers"][i] and definition_layer[i] == 0:
            if "fixed" in internal_structure_2d_fem["layers"][i]["end_nd_arc"].keys():
                if internal_structure_2d_fem["layers"][i]["end_nd_arc"]["fixed"] == "TE":
                    layer_end_nd[i, :] = np.ones(n_span)
                    # raise ValueError('No need to fix element to TE, set it to 0.')
                elif internal_structure_2d_fem["layers"][i]["end_nd_arc"]["fixed"] == "LE":
                    definition_layer[i] = 12
                else:
                    definition_layer[i] = 6
                    flag = False
                    for k in range(n_layers):
                        if layer_name[k] == internal_structure_2d_fem["layers"][i]["end_nd_arc"]["fixed"]:
                            index_layer_end[i] = k
                            flag = True
                            break
                    if flag == False:
                        raise ValueError("Error with layer " + internal_structure_2d_fem["layers"][i]["name"])
            else:
                layer_end_nd[i, :] = np.nan_to_num(
                    PchipInterpolator(
                        internal_structure_2d_fem["layers"][i]["end_nd_arc"]["grid"],
                        internal_structure_2d_fem["layers"][i]["end_nd_arc"]["values"],
                        extrapolate=False,
                    )(nd_span)
                )

            if np.min(layer_end_nd[i, :]) < 0.0 or np.max(layer_end_nd[i, :]) > 1.0:
                raise Exception(
                    """The end_nd_arc layer cannot be smaller than 0 nor
                larger than 1. Check the input yaml for layer """,
                    layer_name[i],
                )

            if "width" in internal_structure_2d_fem["layers"][i]:
                definition_layer[i] = 8
                layer_width[i, :] = np.nan_to_num(
                    PchipInterpolator(
                        internal_structure_2d_fem["layers"][i]["width"]["grid"],
                        internal_structure_2d_fem["layers"][i]["width"]["values"],
                        extrapolate=False,
                    )(nd_span)
                )
            if "start_nd_arc" in internal_structure_2d_fem["layers"][i]:
                definition_layer[i] = 9

        if "web" in internal_structure_2d_fem["layers"][i]:
            web_name_i = internal_structure_2d_fem["layers"][i]["web"]
            for j in range(modeling_options["WISDEM"]["RotorSE"]["n_webs"]):
                if web_name_i == modeling_options["WISDEM"]["RotorSE"]["web_name"][j]:
                    k = j + 1
                    break
            layer_web[i] = k
            definition_layer[i] = 10

        if "offset_plane" in internal_structure_2d_fem["layers"][i]:
            layer_rotation[i, :] = np.ones_like(nd_span) * internal_structure_2d_fem["layers"][i]["offset_plane"]["blade_rotation"]
            layer_offset_y_pa[i, :] = -np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["layers"][i]["offset_plane"]["offset"]["grid"],
                    internal_structure_2d_fem["layers"][i]["offset_plane"]["offset"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
            layer_width[i, :] = np.nan_to_num(
                PchipInterpolator(
                    internal_structure_2d_fem["layers"][i]["width"]["grid"],
                    internal_structure_2d_fem["layers"][i]["width"]["values"],
                    extrapolate=False,
                )(nd_span)
            )
            definition_layer[i] = 13
            layer_side[i] = internal_structure_2d_fem["layers"][i]["side"]

        # Fatigue params
        if layer_name[i] == modeling_options["WISDEM"]["RotorSE"]["spar_cap_ss"]:
            k = wt_opt["materials.name"].index(layer_mat[i])
            wt_opt["blade.fatigue.sparU_wohlerA"] = wt_opt["materials.wohler_intercept"][k]
            wt_opt["blade.fatigue.sparU_wohlerexp"] = wt_opt["materials.wohler_exp"][k]
            wt_opt["blade.fatigue.sparU_sigma_ult"] = wt_opt["materials.Xt"][k, :].max()

        elif layer_name[i] == modeling_options["WISDEM"]["RotorSE"]["spar_cap_ps"]:
            k = wt_opt["materials.name"].index(layer_mat[i])
            wt_opt["blade.fatigue.sparL_wohlerA"] = wt_opt["materials.wohler_intercept"][k]
            wt_opt["blade.fatigue.sparL_wohlerexp"] = wt_opt["materials.wohler_exp"][k]
            wt_opt["blade.fatigue.sparL_sigma_ult"] = wt_opt["materials.Xt"][k, :].max()

        elif layer_name[i] == modeling_options["WISDEM"]["RotorSE"]["te_ss"]:
            k = wt_opt["materials.name"].index(layer_mat[i])
            wt_opt["blade.fatigue.teU_wohlerA"] = wt_opt["materials.wohler_intercept"][k]
            wt_opt["blade.fatigue.teU_wohlerexp"] = wt_opt["materials.wohler_exp"][k]
            wt_opt["blade.fatigue.teU_sigma_ult"] = wt_opt["materials.Xt"][k, :].max()

        elif layer_name[i] == modeling_options["WISDEM"]["RotorSE"]["te_ps"]:
            k = wt_opt["materials.name"].index(layer_mat[i])
            wt_opt["blade.fatigue.teL_wohlerA"] = wt_opt["materials.wohler_intercept"][k]
            wt_opt["blade.fatigue.teL_wohlerexp"] = wt_opt["materials.wohler_exp"][k]
            wt_opt["blade.fatigue.teL_sigma_ult"] = wt_opt["materials.Xt"][k, :].max()

    # Assign the openmdao values
    wt_opt["blade.internal_structure_2d_fem.layer_side"] = layer_side
    wt_opt["blade.internal_structure_2d_fem.layer_thickness"] = thickness
    wt_opt["blade.internal_structure_2d_fem.layer_orientation"] = orientation
    wt_opt["blade.internal_structure_2d_fem.layer_midpoint_nd"] = layer_midpoint_nd
    wt_opt["blade.internal_structure_2d_fem.layer_web"] = layer_web
    wt_opt["blade.internal_structure_2d_fem.definition_web"] = definition_web
    wt_opt["blade.internal_structure_2d_fem.definition_layer"] = definition_layer
    wt_opt["blade.internal_structure_2d_fem.index_layer_start"] = index_layer_start
    wt_opt["blade.internal_structure_2d_fem.index_layer_end"] = index_layer_end

    wt_opt["blade.internal_structure_2d_fem.web_offset_y_pa_yaml"] = web_offset_y_pa
    wt_opt["blade.internal_structure_2d_fem.web_rotation_yaml"] = web_rotation
    wt_opt["blade.internal_structure_2d_fem.web_start_nd_yaml"] = web_start_nd
    wt_opt["blade.internal_structure_2d_fem.web_end_nd_yaml"] = web_end_nd
    wt_opt["blade.internal_structure_2d_fem.layer_offset_y_pa_yaml"] = layer_offset_y_pa
    wt_opt["blade.internal_structure_2d_fem.layer_width_yaml"] = layer_width
    wt_opt["blade.internal_structure_2d_fem.layer_start_nd_yaml"] = layer_start_nd
    wt_opt["blade.internal_structure_2d_fem.layer_end_nd_yaml"] = layer_end_nd
    wt_opt["blade.internal_structure_2d_fem.layer_rotation_yaml"] = layer_rotation

    # Spanwise joint
    wt_opt["blade.internal_structure_2d_fem.joint_bolt"] = internal_structure_2d_fem["joint"]["bolt"]
    wt_opt["blade.internal_structure_2d_fem.joint_position"] = internal_structure_2d_fem["joint"]["position"]
    wt_opt["blade.internal_structure_2d_fem.joint_mass"] = internal_structure_2d_fem["joint"]["mass"]
    wt_opt["blade.internal_structure_2d_fem.joint_nonmaterial_cost"] = internal_structure_2d_fem["joint"][
        "nonmaterial_cost"
    ]
    wt_opt["blade.internal_structure_2d_fem.reinforcement_layer_ss"] = internal_structure_2d_fem["joint"][
        "reinforcement_layer_ss"
    ]
    wt_opt["blade.internal_structure_2d_fem.reinforcement_layer_ps"] = internal_structure_2d_fem["joint"][
        "reinforcement_layer_ps"
    ]

    # Blade root
    wt_opt["blade.internal_structure_2d_fem.d_f"] = internal_structure_2d_fem["root"]["d_f"]
    wt_opt["blade.internal_structure_2d_fem.sigma_max"] = internal_structure_2d_fem["root"]["sigma_max"]

    return wt_opt


def assign_te_flaps_values(wt_opt, modeling_options, blade):
    # Function to assign the trailing edge flaps data to the openmdao data structure
    if modeling_options["WISDEM"]["RotorSE"]["n_te_flaps"] > 0:
        n_te_flaps = modeling_options["WISDEM"]["RotorSE"]["n_te_flaps"]
        for i in range(n_te_flaps):
            wt_opt["dac_ivc.te_flap_start"][i] = blade["aerodynamic_control"]["te_flaps"][i]["span_start"]
            wt_opt["dac_ivc.te_flap_end"][i] = blade["aerodynamic_control"]["te_flaps"][i]["span_end"]
            wt_opt["dac_ivc.chord_start"][i] = blade["aerodynamic_control"]["te_flaps"][i]["chord_start"]
            wt_opt["dac_ivc.delta_max_pos"][i] = blade["aerodynamic_control"]["te_flaps"][i]["delta_max_pos"]
            wt_opt["dac_ivc.delta_max_neg"][i] = blade["aerodynamic_control"]["te_flaps"][i]["delta_max_neg"]

            wt_opt["dac_ivc.te_flap_ext"] = (
                blade["aerodynamic_control"]["te_flaps"][i]["span_end"]
                - blade["aerodynamic_control"]["te_flaps"][i]["span_start"]
            )
            # wt_opt['dac_ivc.te_flap_end'] = blade['aerodynamic_control']['te_flaps'][i]['span_end']

            # Checks for consistency
            if blade["aerodynamic_control"]["te_flaps"][i]["span_start"] < 0.0:
                raise ValueError(
                    "Error: the start along blade span of the trailing edge flap number "
                    + str(i)
                    + " is defined smaller than 0, which corresponds to blade root. Please check the yaml input."
                )
            elif blade["aerodynamic_control"]["te_flaps"][i]["span_start"] > 1.0:
                raise ValueError(
                    "Error: the start along blade span of the trailing edge flap number "
                    + str(i)
                    + " is defined bigger than 1, which corresponds to blade tip. Please check the yaml input."
                )
            elif blade["aerodynamic_control"]["te_flaps"][i]["span_end"] < 0.0:
                raise ValueError(
                    "Error: the end along blade span of the trailing edge flap number "
                    + str(i)
                    + " is defined smaller than 0, which corresponds to blade root. Please check the yaml input."
                )
            elif blade["aerodynamic_control"]["te_flaps"][i]["span_end"] > 1.0:
                raise ValueError(
                    "Error: the end along blade span of the trailing edge flap number "
                    + str(i)
                    + " is defined bigger than 1, which corresponds to blade tip. Please check the yaml input."
                )
            elif (
                blade["aerodynamic_control"]["te_flaps"][i]["span_start"]
                == blade["aerodynamic_control"]["te_flaps"][i]["span_end"]
            ):
                raise ValueError(
                    "Error: the start and end along blade span of the trailing edge flap number "
                    + str(i)
                    + " are defined equal. Please check the yaml input."
                )
            elif i > 0:
                if (
                    blade["aerodynamic_control"]["te_flaps"][i]["span_start"]
                    < blade["aerodynamic_control"]["te_flaps"][i - 1]["span_end"]
                ):
                    raise ValueError(
                        "Error: the start along blade span of the trailing edge flap number "
                        + str(i)
                        + " is smaller than the end of the trailing edge flap number "
                        + str(i - 1)
                        + ". Please check the yaml input."
                    )
            elif blade["aerodynamic_control"]["te_flaps"][i]["chord_start"] < 0.2:
                raise ValueError(
                    "Error: the start along the chord of the trailing edge flap number "
                    + str(i)
                    + " is smaller than 0.2, which is too close to the leading edge. Please check the yaml input."
                )
            elif blade["aerodynamic_control"]["te_flaps"][i]["chord_start"] > 1.0:
                raise ValueError(
                    "Error: the end along the chord of the trailing edge flap number "
                    + str(i)
                    + " is larger than 1., which is beyond the trailing edge. Please check the yaml input."
                )
            elif blade["aerodynamic_control"]["te_flaps"][i]["delta_max_pos"] > 30.0 / 180.0 * np.pi:
                raise ValueError(
                    "Error: the max positive deflection of the trailing edge flap number "
                    + str(i)
                    + " is larger than 30 deg, which is beyond the limits of applicability of this tool. Please check the yaml input."
                )
            elif blade["aerodynamic_control"]["te_flaps"][i]["delta_max_neg"] < -30.0 / 180.0 * np.pi:
                raise ValueError(
                    "Error: the max negative deflection of the trailing edge flap number "
                    + str(i)
                    + " is smaller than -30 deg, which is beyond the limits of applicability of this tool. Please check the yaml input."
                )
            elif (
                blade["aerodynamic_control"]["te_flaps"][i]["delta_max_pos"]
                < blade["aerodynamic_control"]["te_flaps"][i]["delta_max_neg"]
            ):
                raise ValueError(
                    "Error: the max positive deflection of the trailing edge flap number "
                    + str(i)
                    + " is smaller than the max negative deflection. Please check the yaml input."
                )
            else:
                pass

    return wt_opt


def assign_hub_values(wt_opt, hub, flags):
    wt_opt["hub.diameter"] = hub["diameter"]
    wt_opt["hub.cone"] = hub["cone_angle"]
    # wt_opt['hub.drag_coeff']                  = hub['drag_coefficient'] # GB: This doesn't connect to anything
    if flags["hub"]:
        wt_opt["hub.flange_t2shell_t"] = hub["flange_t2shell_t"]
        wt_opt["hub.flange_OD2hub_D"] = hub["flange_OD2hub_D"]
        wt_opt["hub.flange_ID2flange_OD"] = hub["flange_ID2OD"]
        wt_opt["hub.hub_in2out_circ"] = hub["hub_blade_spacing_margin"]
        wt_opt["hub.hub_stress_concentration"] = hub["hub_stress_concentration"]
        wt_opt["hub.n_front_brackets"] = hub["n_front_brackets"]
        wt_opt["hub.n_rear_brackets"] = hub["n_rear_brackets"]
        wt_opt["hub.clearance_hub_spinner"] = hub["clearance_hub_spinner"]
        wt_opt["hub.spin_hole_incr"] = hub["spin_hole_incr"]
        wt_opt["hub.pitch_system_scaling_factor"] = hub["pitch_system_scaling_factor"]
        wt_opt["hub.hub_material"] = hub["hub_material"]
        wt_opt["hub.spinner_material"] = hub["spinner_material"]

    return wt_opt


def assign_nacelle_values(wt_opt, modeling_options, nacelle, flags):
    # Common direct and geared
    wt_opt["nacelle.uptilt"] = nacelle["drivetrain"]["uptilt"]
    wt_opt["nacelle.distance_tt_hub"] = nacelle["drivetrain"]["distance_tt_hub"]
    wt_opt["nacelle.overhang"] = nacelle["drivetrain"]["overhang"]
    wt_opt["nacelle.gear_ratio"] = nacelle["drivetrain"]["gear_ratio"]
    wt_opt["nacelle.gearbox_efficiency"] = nacelle["drivetrain"]["gearbox_efficiency"]
    if flags["nacelle"]:
        wt_opt["nacelle.distance_hub_mb"] = nacelle["drivetrain"]["distance_hub_mb"]
        wt_opt["nacelle.distance_mb_mb"] = nacelle["drivetrain"]["distance_mb_mb"]
        wt_opt["nacelle.L_generator"] = nacelle["drivetrain"]["generator_length"]
        wt_opt["nacelle.damping_ratio"] = nacelle["drivetrain"]["damping_ratio"]
        wt_opt["nacelle.mb1Type"] = nacelle["drivetrain"]["mb1Type"]
        wt_opt["nacelle.mb2Type"] = nacelle["drivetrain"]["mb2Type"]
        wt_opt["nacelle.uptower"] = nacelle["drivetrain"]["uptower"]
        wt_opt["nacelle.lss_material"] = nacelle["drivetrain"]["lss_material"]
        wt_opt["nacelle.bedplate_material"] = nacelle["drivetrain"]["bedplate_material"]
        wt_opt["nacelle.brake_mass_user"] = nacelle["drivetrain"]["brake_mass_user"]
        wt_opt["nacelle.hvac_mass_coeff"] = nacelle["drivetrain"]["hvac_mass_coefficient"]
        wt_opt["nacelle.converter_mass_user"] = nacelle["drivetrain"]["converter_mass_user"]
        wt_opt["nacelle.transformer_mass_user"] = nacelle["drivetrain"]["transformer_mass_user"]

        wt_opt["nacelle.lss_wall_thickness"] = nacelle["drivetrain"]["lss_wall_thickness"]
        wt_opt["nacelle.lss_diameter"] = nacelle["drivetrain"]["lss_diameter"]

        if modeling_options["WISDEM"]["DriveSE"]["direct"]:
            if wt_opt["nacelle.gear_ratio"] > 1:
                raise Exception(
                    "The gear ratio is larger than 1, but the wind turbine is marked as direct drive. Please check the input yaml file."
                )
            # Direct only
            wt_opt["nacelle.nose_wall_thickness"] = nacelle["drivetrain"]["nose_wall_thickness"]
            wt_opt["nacelle.nose_diameter"] = nacelle["drivetrain"]["nose_diameter"]

            s_bedplate = np.linspace(0.0, 1.0, len(wt_opt["nacelle.bedplate_wall_thickness"]))
            s_bed_thick_in = nacelle["drivetrain"]["bedplate_wall_thickness"]["grid"]
            v_bed_thick_in = nacelle["drivetrain"]["bedplate_wall_thickness"]["values"]
            wt_opt["nacelle.bedplate_wall_thickness"] = PchipInterpolator(s_bed_thick_in, v_bed_thick_in)(s_bedplate)
        else:
            if wt_opt["nacelle.gear_ratio"] == 1:
                raise Exception(
                    "The gear ratio is set to 1, but the wind turbine is marked as geared. Please check the input yaml file."
                )
            # Geared only
            wt_opt["nacelle.hss_wall_thickness"] = nacelle["drivetrain"]["hss_wall_thickness"]
            wt_opt["nacelle.hss_diameter"] = nacelle["drivetrain"]["hss_diameter"]

            wt_opt["nacelle.hss_length"] = nacelle["drivetrain"]["hss_length"]
            wt_opt["nacelle.bedplate_flange_width"] = nacelle["drivetrain"]["bedplate_flange_width"]
            wt_opt["nacelle.bedplate_flange_thickness"] = nacelle["drivetrain"]["bedplate_flange_thickness"]
            wt_opt["nacelle.bedplate_web_thickness"] = nacelle["drivetrain"]["bedplate_web_thickness"]
            wt_opt["nacelle.gear_configuration"] = nacelle["drivetrain"]["gear_configuration"].lower()
            wt_opt["nacelle.gearbox_mass_user"] = nacelle["drivetrain"]["gearbox_mass_user"]
            wt_opt["nacelle.gearbox_torque_density"] = nacelle["drivetrain"]["gearbox_torque_density"]
            wt_opt["nacelle.gearbox_radius_user"] = nacelle["drivetrain"]["gearbox_radius_user"]
            wt_opt["nacelle.gearbox_length_user"] = nacelle["drivetrain"]["gearbox_length_user"]
            wt_opt["nacelle.planet_numbers"] = nacelle["drivetrain"]["planet_numbers"]
            wt_opt["nacelle.hss_material"] = nacelle["drivetrain"]["hss_material"]

        if not modeling_options["flags"]["generator"]:
            wt_opt["generator.generator_radius_user"] = nacelle["drivetrain"]["generator_radius_user"]
            wt_opt["generator.generator_mass_user"] = nacelle["drivetrain"]["generator_mass_user"]

            eff_user = np.c_[
                nacelle["drivetrain"]["generator_rpm_efficiency_user"]["grid"],
                nacelle["drivetrain"]["generator_rpm_efficiency_user"]["values"],
            ]
            n_pc = modeling_options["WISDEM"]["RotorSE"]["n_pc"]
            if np.any(eff_user):
                newrpm = np.linspace(eff_user[:, 0].min(), eff_user[:, 0].max(), n_pc)
                neweff = PchipInterpolator(eff_user[:, 0], eff_user[:, 1])(newrpm)
                myeff = np.c_[newrpm, neweff]
            else:
                myeff = np.zeros((n_pc, 2))
            wt_opt["generator.generator_efficiency_user"] = myeff

    return wt_opt


def assign_generator_values(wt_opt, modeling_options, nacelle):
    wt_opt["generator.B_r"] = nacelle["generator"]["B_r"]
    wt_opt["generator.P_Fe0e"] = nacelle["generator"]["P_Fe0e"]
    wt_opt["generator.P_Fe0h"] = nacelle["generator"]["P_Fe0h"]
    wt_opt["generator.S_N"] = nacelle["generator"]["S_N"]
    wt_opt["generator.alpha_p"] = nacelle["generator"]["alpha_p"]
    wt_opt["generator.b_r_tau_r"] = nacelle["generator"]["b_r_tau_r"]
    wt_opt["generator.b_ro"] = nacelle["generator"]["b_ro"]
    wt_opt["generator.b_s_tau_s"] = nacelle["generator"]["b_s_tau_s"]
    wt_opt["generator.b_so"] = nacelle["generator"]["b_so"]
    wt_opt["generator.cofi"] = nacelle["generator"]["cofi"]
    wt_opt["generator.freq"] = nacelle["generator"]["freq"]
    wt_opt["generator.h_i"] = nacelle["generator"]["h_i"]
    wt_opt["generator.h_sy0"] = nacelle["generator"]["h_sy0"]
    wt_opt["generator.h_w"] = nacelle["generator"]["h_w"]
    wt_opt["generator.k_fes"] = nacelle["generator"]["k_fes"]
    wt_opt["generator.k_fillr"] = nacelle["generator"]["k_fillr"]
    wt_opt["generator.k_fills"] = nacelle["generator"]["k_fills"]
    wt_opt["generator.k_s"] = nacelle["generator"]["k_s"]
    wt_opt["generator.m"] = nacelle["generator"]["m"]
    wt_opt["generator.mu_0"] = nacelle["generator"]["mu_0"]
    wt_opt["generator.mu_r"] = nacelle["generator"]["mu_r"]
    wt_opt["generator.p"] = nacelle["generator"]["p"]
    wt_opt["generator.phi"] = nacelle["generator"]["phi"]
    wt_opt["generator.q1"] = nacelle["generator"]["q1"]
    wt_opt["generator.q2"] = nacelle["generator"]["q2"]
    wt_opt["generator.ratio_mw2pp"] = nacelle["generator"]["ratio_mw2pp"]
    wt_opt["generator.resist_Cu"] = nacelle["generator"]["resist_Cu"]
    wt_opt["generator.sigma"] = nacelle["generator"]["sigma"]
    wt_opt["generator.y_tau_p"] = nacelle["generator"]["y_tau_p"]
    wt_opt["generator.y_tau_pr"] = nacelle["generator"]["y_tau_pr"]

    wt_opt["generator.I_0"] = nacelle["generator"]["I_0"]
    wt_opt["generator.d_r"] = nacelle["generator"]["d_r"]
    wt_opt["generator.h_m"] = nacelle["generator"]["h_m"]
    wt_opt["generator.h_0"] = nacelle["generator"]["h_0"]
    wt_opt["generator.h_s"] = nacelle["generator"]["h_s"]
    wt_opt["generator.len_s"] = nacelle["generator"]["len_s"]
    wt_opt["generator.n_r"] = nacelle["generator"]["n_r"]
    wt_opt["generator.rad_ag"] = nacelle["generator"]["rad_ag"]
    wt_opt["generator.t_wr"] = nacelle["generator"]["t_wr"]

    wt_opt["generator.n_s"] = nacelle["generator"]["n_s"]
    wt_opt["generator.b_st"] = nacelle["generator"]["b_st"]
    wt_opt["generator.d_s"] = nacelle["generator"]["d_s"]
    wt_opt["generator.t_ws"] = nacelle["generator"]["t_ws"]

    wt_opt["generator.rho_Copper"] = nacelle["generator"]["rho_Copper"]
    wt_opt["generator.rho_Fe"] = nacelle["generator"]["rho_Fe"]
    wt_opt["generator.rho_Fes"] = nacelle["generator"]["rho_Fes"]
    wt_opt["generator.rho_PM"] = nacelle["generator"]["rho_PM"]

    wt_opt["generator.C_Cu"] = nacelle["generator"]["C_Cu"]
    wt_opt["generator.C_Fe"] = nacelle["generator"]["C_Fe"]
    wt_opt["generator.C_Fes"] = nacelle["generator"]["C_Fes"]
    wt_opt["generator.C_PM"] = nacelle["generator"]["C_PM"]

    if modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["pmsg_outer"]:
        wt_opt["generator.N_c"] = nacelle["generator"]["N_c"]
        wt_opt["generator.b"] = nacelle["generator"]["b"]
        wt_opt["generator.c"] = nacelle["generator"]["c"]
        wt_opt["generator.E_p"] = nacelle["generator"]["E_p"]
        wt_opt["generator.h_yr"] = nacelle["generator"]["h_yr"]
        wt_opt["generator.h_ys"] = nacelle["generator"]["h_ys"]
        wt_opt["generator.h_sr"] = nacelle["generator"]["h_sr"]
        wt_opt["generator.h_ss"] = nacelle["generator"]["h_ss"]
        wt_opt["generator.t_r"] = nacelle["generator"]["t_r"]
        wt_opt["generator.t_s"] = nacelle["generator"]["t_s"]

        wt_opt["generator.u_allow_pcent"] = nacelle["generator"]["u_allow_pcent"]
        wt_opt["generator.y_allow_pcent"] = nacelle["generator"]["y_allow_pcent"]
        wt_opt["generator.z_allow_deg"] = nacelle["generator"]["z_allow_deg"]
        wt_opt["generator.B_tmax"] = nacelle["generator"]["B_tmax"]

    if modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
        wt_opt["generator.tau_p"] = nacelle["generator"]["tau_p"]
        wt_opt["generator.h_ys"] = nacelle["generator"]["h_ys"]
        wt_opt["generator.h_yr"] = nacelle["generator"]["h_yr"]
        wt_opt["generator.b_arm"] = nacelle["generator"]["b_arm"]

    elif modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["scig", "dfig"]:
        wt_opt["generator.B_symax"] = nacelle["generator"]["B_symax"]
        wt_opt["generator.S_Nmax"] = nacelle["generator"]["S_Nmax"]

    return wt_opt


def assign_tower_values(wt_opt, modeling_options, tower):
    # Function to assign values to the openmdao component Tower
    n_height = modeling_options["WISDEM"]["TowerSE"]["n_height_tower"]  # Number of points along tower height
    n_layers = modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"]

    svec = np.unique(
        np.r_[
            tower["outer_shape_bem"]["outer_diameter"]["grid"],
            tower["outer_shape_bem"]["reference_axis"]["x"]["grid"],
            tower["outer_shape_bem"]["reference_axis"]["y"]["grid"],
            tower["outer_shape_bem"]["reference_axis"]["z"]["grid"],
        ]
    )

    # wt_opt['tower.s'] = svec
    wt_opt["tower.diameter"] = PchipInterpolator(
        tower["outer_shape_bem"]["outer_diameter"]["grid"], tower["outer_shape_bem"]["outer_diameter"]["values"]
    )(svec)
    wt_opt["tower.cd"] = PchipInterpolator(
        tower["outer_shape_bem"]["drag_coefficient"]["grid"],
        tower["outer_shape_bem"]["drag_coefficient"]["values"],
    )(svec)

    wt_opt["tower.ref_axis"][:, 0] = PchipInterpolator(
        tower["outer_shape_bem"]["reference_axis"]["x"]["grid"],
        tower["outer_shape_bem"]["reference_axis"]["x"]["values"],
    )(svec)
    wt_opt["tower.ref_axis"][:, 1] = PchipInterpolator(
        tower["outer_shape_bem"]["reference_axis"]["y"]["grid"],
        tower["outer_shape_bem"]["reference_axis"]["y"]["values"],
    )(svec)
    wt_opt["tower.ref_axis"][:, 2] = PchipInterpolator(
        tower["outer_shape_bem"]["reference_axis"]["z"]["grid"],
        tower["outer_shape_bem"]["reference_axis"]["z"]["values"],
    )(svec)

    layer_name = n_layers * [""]
    layer_mat = n_layers * [""]
    thickness = np.zeros((n_layers, n_height))
    for i in range(n_layers):
        layer_name[i] = tower["internal_structure_2d_fem"]["layers"][i]["name"]
        layer_mat[i] = tower["internal_structure_2d_fem"]["layers"][i]["material"]
        thickness[i] = PchipInterpolator(
            tower["internal_structure_2d_fem"]["layers"][i]["thickness"]["grid"],
            tower["internal_structure_2d_fem"]["layers"][i]["thickness"]["values"],
        )(svec)

    wt_opt["tower.layer_name"] = layer_name
    wt_opt["tower.layer_mat"] = layer_mat
    wt_opt["tower.layer_thickness"] = thickness

    wt_opt["tower.outfitting_factor"] = tower["internal_structure_2d_fem"]["outfitting_factor"]

    if "Loading" in modeling_options["WISDEM"]:
        F = []
        M = []
        n_dlc = modeling_options["WISDEM"]["n_dlc"]
        for k in range(n_dlc):
            F = np.append(F, modeling_options["WISDEM"]["Loading"]["loads"][k]["force"])
            M = np.append(M, modeling_options["WISDEM"]["Loading"]["loads"][k]["moment"])
        F = F.reshape((n_dlc, 3)).T
        M = M.reshape((n_dlc, 3)).T

        if modeling_options["flags"]["tower"]:
            wt_opt["towerse.rna_mass"] = modeling_options["WISDEM"]["Loading"]["mass"]
            wt_opt["towerse.rna_cg"] = modeling_options["WISDEM"]["Loading"]["center_of_mass"]
            wt_opt["towerse.rna_I"] = modeling_options["WISDEM"]["Loading"]["moment_of_inertia"]
            n_dlc = modeling_options["WISDEM"]["n_dlc"]
            for k in range(n_dlc):
                kstr = "" if n_dlc <= 1 else str(k + 1)
                wt_opt[f"towerse.env{kstr}.Uref"] = modeling_options["WISDEM"]["Loading"]["loads"][k]["velocity"]
            wt_opt["towerse.tower.rna_F"] = F
            wt_opt["towerse.tower.rna_M"] = M

        if modeling_options["flags"]["monopile"]:
            # Monopile has the option for joint tower-monopile analysis, so load it here too.  Not true for jackets
            wt_opt["fixedse.rna_mass"] = modeling_options["WISDEM"]["Loading"]["mass"]
            wt_opt["fixedse.rna_cg"] = modeling_options["WISDEM"]["Loading"]["center_of_mass"]
            wt_opt["fixedse.rna_I"] = modeling_options["WISDEM"]["Loading"]["moment_of_inertia"]
            wt_opt["fixedse.monopile.rna_F"] = F
            wt_opt["fixedse.monopile.rna_M"] = M

    return wt_opt


def assign_monopile_values(wt_opt, modeling_options, monopile):
    # Function to assign values to the openmdao component Monopile
    n_height = modeling_options["WISDEM"]["FixedBottomSE"]["n_height"]  # Number of points along monopile height
    n_layers = modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"]

    svec = np.unique(
        np.r_[
            monopile["outer_shape_bem"]["outer_diameter"]["grid"],
            monopile["outer_shape_bem"]["reference_axis"]["x"]["grid"],
            monopile["outer_shape_bem"]["reference_axis"]["y"]["grid"],
            monopile["outer_shape_bem"]["reference_axis"]["z"]["grid"],
        ]
    )

    wt_opt["monopile.s"] = svec
    wt_opt["monopile.diameter"] = PchipInterpolator(
        monopile["outer_shape_bem"]["outer_diameter"]["grid"],
        monopile["outer_shape_bem"]["outer_diameter"]["values"],
    )(svec)

    wt_opt["monopile.ref_axis"][:, 0] = PchipInterpolator(
        monopile["outer_shape_bem"]["reference_axis"]["x"]["grid"],
        monopile["outer_shape_bem"]["reference_axis"]["x"]["values"],
    )(svec)
    wt_opt["monopile.ref_axis"][:, 1] = PchipInterpolator(
        monopile["outer_shape_bem"]["reference_axis"]["y"]["grid"],
        monopile["outer_shape_bem"]["reference_axis"]["y"]["values"],
    )(svec)
    wt_opt["monopile.ref_axis"][:, 2] = PchipInterpolator(
        monopile["outer_shape_bem"]["reference_axis"]["z"]["grid"],
        monopile["outer_shape_bem"]["reference_axis"]["z"]["values"],
    )(svec)

    layer_name = n_layers * [""]
    layer_mat = n_layers * [""]
    thickness = np.zeros((n_layers, n_height))
    for i in range(n_layers):
        layer_name[i] = monopile["internal_structure_2d_fem"]["layers"][i]["name"]
        layer_mat[i] = monopile["internal_structure_2d_fem"]["layers"][i]["material"]
        thickness[i] = PchipInterpolator(
            monopile["internal_structure_2d_fem"]["layers"][i]["thickness"]["grid"],
            monopile["internal_structure_2d_fem"]["layers"][i]["thickness"]["values"],
        )(svec)

    wt_opt["monopile.layer_name"] = layer_name
    wt_opt["monopile.layer_mat"] = layer_mat
    wt_opt["monopile.layer_thickness"] = thickness

    wt_opt["monopile.outfitting_factor"] = monopile["internal_structure_2d_fem"]["outfitting_factor"]
    wt_opt["monopile.transition_piece_mass"] = monopile["transition_piece_mass"]
    wt_opt["monopile.transition_piece_cost"] = monopile["transition_piece_cost"]
    wt_opt["monopile.gravity_foundation_mass"] = monopile["gravity_foundation_mass"]

    return wt_opt


def assign_jacket_values(wt_opt, modeling_options, jacket):
    # Function to assign values to the openmdao component Jacket
    wt_opt["jacket.transition_piece_mass"] = jacket["transition_piece_mass"]
    wt_opt["jacket.transition_piece_cost"] = jacket["transition_piece_cost"]
    wt_opt["jacket.gravity_foundation_mass"] = jacket["gravity_foundation_mass"]
    wt_opt["jacket.r_head"] = jacket["r_head"]
    wt_opt["jacket.foot_head_ratio"] = jacket["r_foot"] / jacket["r_head"]
    wt_opt["jacket.height"] = jacket["height"]
    wt_opt["jacket.leg_diameter"] = jacket["leg_diameter"]
    wt_opt["jacket.leg_thickness"] = jacket["leg_thickness"]
    wt_opt["jacket.brace_diameters"] = jacket["brace_diameters"]
    wt_opt["jacket.brace_thicknesses"] = jacket["brace_thicknesses"]
    wt_opt["jacket.bay_spacing"] = jacket["bay_spacing"]

    return wt_opt


def assign_floating_values(wt_opt, modeling_options, floating, opt_options):
    float_opt = opt_options["design_variables"]["floating"]
    floating_init_options = modeling_options["floating"]
    n_joints = floating_init_options["joints"]["n_joints"]
    # Loop through joints and assign location values to openmdao entry
    for i in range(n_joints):
        wt_opt["floating.location_in"][i, :] = floating["joints"][i]["location"]

    # Set transition joint/node
    if modeling_options["floating"]["transition_joint"] is None:
        centroid = wt_opt["floating.location_in"][:, :2].mean(axis=0)
        zmax = wt_opt["floating.location_in"][:, 2].max()
        itrans = util.closest_node(wt_opt["floating.location_in"], np.r_[centroid, zmax])
    else:
        itrans = modeling_options["floating"]["transition_joint"]
    wt_opt["floating.transition_node"] = wt_opt["floating.location_in"][itrans, :]
    wt_opt["floating.transition_piece_mass"] = floating["transition_piece_mass"]
    wt_opt["floating.transition_piece_cost"] = floating["transition_piece_cost"]

    # Make sure IVCs are initialized too
    for k, linked_node_dict in enumerate(modeling_options["floating"]["joints"]["design_variable_data"]):
        idx = linked_node_dict["indices"]
        dim = linked_node_dict["dimension"]
        wt_opt[f"floating.jointdv_{k}"] = wt_opt["floating.location_in"][idx, dim].mean()

    # Now do members by assigning to unique member groups
    n_members = floating_init_options["members"]["n_members"]
    for i in range(n_members):
        name_member = floating_init_options["members"]["name"][i]
        grid_member = floating_init_options["members"]["grid_member_" + floating_init_options["members"]["name"][i]]
        grid_geom = floating_init_options["members"]["geom_member_" + floating_init_options["members"]["name"][i]]
        idx = floating_init_options["members"]["name2idx"][name_member]

        wt_opt[f"floating.memgrp{idx}.s"] = grid_member
        wt_opt[f"floating.memgrp{idx}.s_in"] = grid_geom

        # Check user defined coefficients inputs
        usr_defined_coeffs = ["Ca", "Cd"]
        if floating["members"][i]["outer_shape"]["shape"] == "rectangular":
            usr_defined_coeffs += ["Cay", "Cdy"]
            grid_length = len(floating["members"][i]["outer_shape"]["side_length_a"]["grid"])
        else:
            grid_length = len(floating["members"][i]["outer_shape"]["outer_diameter"]["grid"])
            
        usr_defined_flag = {}
        for coeff in usr_defined_coeffs:
            usr_defined_flag[coeff] = np.all(np.array(floating["members"][i][coeff])>0)
            coeff_length = len(floating["members"][i][coeff])
            if usr_defined_flag[coeff]:
                assert grid_length == coeff_length, f"Users define {coeff}, but the length is different from grid length. Please correct."

        diameter_assigned = False
        for j, kgrp in enumerate(float_opt["members"]["groups"]):
            memname = kgrp["names"][0]
            idx2 = floating_init_options["members"]["name2idx"][memname]
            if idx == idx2:
                # TODO: Need better ways to condition
                if "diameter" in float_opt["members"]["groups"][j]:
                    if float_opt["members"]["groups"][j]["diameter"]["constant"]:
                        wt_opt[f"floating.memgrp{idx}.outer_diameter_in"] = floating["members"][i]["outer_shape"][
                            "outer_diameter"
                        ]["values"][0]
                        wt_opt[f"floating.memgrp{idx}.ca_usr_geom"] = floating["members"][i]["Ca"][0] if floating["members"][i]["Ca"][0]>0.0 else 1
                        wt_opt[f"floating.memgrp{idx}.cd_usr_geom"] = floating["members"][i]["Cd"][0] if floating["members"][i]["Cd"][0]>0.0 else 1
                    else:
                        wt_opt[f"floating.memgrp{idx}.outer_diameter_in"][:] = floating["members"][i]["outer_shape"][
                            "outer_diameter"
                        ]["values"]
                        wt_opt[f"floating.memgrp{idx}.ca_usr_geom"] = floating["members"][i]["Ca"] if np.all(floating["members"][i]["Ca"]>0.0) else 1
                        wt_opt[f"floating.memgrp{idx}.cd_usr_geom"] = floating["members"][i]["Cd"] if np.all(floating["members"][i]["Cd"]>0.0) else 1
                    diameter_assigned = True
                if "side_length_a" in float_opt["members"]["groups"][j]:
                    if float_opt["members"]["groups"][j]["side_length_a"]["constant"]:
                        wt_opt[f"floating.memgrp{idx}.side_length_a_in"] = floating["members"][i]["outer_shape"][
                            "side_length_a"
                        ]["values"][0]
                        wt_opt[f"floating.memgrp{idx}.ca_usr_geom"] = floating["members"][i]["Ca"][0] if floating["members"][i]["Ca"][0]>0.0 else 1
                        wt_opt[f"floating.memgrp{idx}.cd_usr_geom"] = floating["members"][i]["Cd"][0] if floating["members"][i]["Ca"][0]>0.0 else 1
                    else:
                        wt_opt[f"floating.memgrp{idx}.side_length_a_in"][:] = floating["members"][i]["outer_shape"][
                            "side_length_a"
                        ]["values"]
                        wt_opt[f"floating.memgrp{idx}.ca_usr_geom"] = floating["members"][i]["Ca"] if np.all(floating["members"][i]["Ca"]>0.0) else 1
                        wt_opt[f"floating.memgrp{idx}.cd_usr_geom"] = floating["members"][i]["Cd"] if np.all(floating["members"][i]["Ca"]>0.0) else 1
                if "side_length_b" in float_opt["members"]["groups"][j]:
                    if float_opt["members"]["groups"][j]["side_length_b"]["constant"]:
                        wt_opt[f"floating.memgrp{idx}.side_length_b_in"] = floating["members"][i]["outer_shape"][
                            "side_length_b"
                        ]["values"][0]
                        wt_opt[f"floating.memgrp{idx}.cay_usr_geom"] = floating["members"][i]["Cay"][0] if floating["members"][i]["Cay"][0]>0.0 else 1
                        wt_opt[f"floating.memgrp{idx}.cdy_usr_geom"] = floating["members"][i]["Cdy"][0] if floating["members"][i]["Cay"][0]>0.0 else 1
                    else:
                        wt_opt[f"floating.memgrp{idx}.side_length_b_in"][:] = floating["members"][i]["outer_shape"][
                            "side_length_b"
                        ]["values"]
                        wt_opt[f"floating.memgrp{idx}.cay_usr_geom"] = floating["members"][i]["Cay"] if np.all(floating["members"][i]["Cay"]>0.0) else 1
                        wt_opt[f"floating.memgrp{idx}.cdy_usr_geom"] = floating["members"][i]["Cdy"] if np.all(floating["members"][i]["Cdy"]>0.0) else 1
                    diameter_assigned = True

        if not diameter_assigned:
            try:
                wt_opt[f"floating.memgrp{idx}.outer_diameter_in"] = PchipInterpolator(
                    floating["members"][i]["outer_shape"]["outer_diameter"]["grid"],
                    floating["members"][i]["outer_shape"]["outer_diameter"]["values"],
                )(grid_geom)
                for coeff in usr_defined_flag.keys():
                    if usr_defined_flag[coeff]:
                        wt_opt[f"floating.memgrp{idx}.{coeff.lower()}_usr_geom"] = PchipInterpolator(
                        floating["members"][i]["outer_shape"]["outer_diameter"]["grid"],
                        floating["members"][i][coeff],
                        )(grid_geom)
            except:
                wt_opt[f"floating.memgrp{idx}.side_length_a_in"] = PchipInterpolator(
                    floating["members"][i]["outer_shape"]["side_length_a"]["grid"],
                    floating["members"][i]["outer_shape"]["side_length_a"]["values"],
                )(grid_geom)
                wt_opt[f"floating.memgrp{idx}.side_length_b_in"] = PchipInterpolator(
                    floating["members"][i]["outer_shape"]["side_length_b"]["grid"],
                    floating["members"][i]["outer_shape"]["side_length_b"]["values"],
                )(grid_geom)
                
                for coeff in usr_defined_flag.keys():
                    if usr_defined_flag[coeff]:
                        wt_opt[f"floating.memgrp{idx}.{coeff.lower()}_usr_geom"] = PchipInterpolator(
                        floating["members"][i]["outer_shape"]["side_length_a"]["grid"],
                        floating["members"][i][coeff],
                        )(grid_geom)

        wt_opt[f"floating.memgrp{idx}.outfitting_factor"] = floating["members"][i]["internal_structure"][
            "outfitting_factor"
        ]

        wt_opt[f"floating.memgrp{idx}.outfitting_factor"] = floating["members"][i]["internal_structure"][
            "outfitting_factor"
        ]

        istruct = floating["members"][i]["internal_structure"]
        if "bulkhead" in istruct:
            wt_opt[f"floating.memgrp{idx}.bulkhead_grid"] = istruct["bulkhead"]["thickness"]["grid"]
            wt_opt[f"floating.memgrp{idx}.bulkhead_thickness"] = istruct["bulkhead"]["thickness"]["values"]

        n_layers = floating_init_options["members"]["n_layers"][i]
        layer_mat = [""] * n_layers
        for j in range(n_layers):
            layer_mat[j] = istruct["layers"][j]["material"]

            wt_opt[f"floating.memgrp{idx}.layer_thickness_in"][j, :] = PchipInterpolator(
                istruct["layers"][j]["thickness"]["grid"],
                istruct["layers"][j]["thickness"]["values"],
            )(grid_geom)
        wt_opt[f"floating.memgrp{idx}.layer_materials"] = layer_mat

        if "ring_stiffeners" in istruct:
            wt_opt[f"floating.memgrp{idx}.ring_stiffener_web_height"] = istruct["ring_stiffeners"]["web_height"]
            wt_opt[f"floating.memgrp{idx}.ring_stiffener_web_thickness"] = istruct["ring_stiffeners"]["web_thickness"]
            wt_opt[f"floating.memgrp{idx}.ring_stiffener_flange_thickness"] = istruct["ring_stiffeners"][
                "flange_thickness"
            ]
            wt_opt[f"floating.memgrp{idx}.ring_stiffener_flange_width"] = istruct["ring_stiffeners"]["flange_width"]
            wt_opt[f"floating.memgrp{idx}.ring_stiffener_spacing"] = istruct["ring_stiffeners"]["spacing"]

        if "longitudinal_stiffeners" in istruct:
            wt_opt[f"floating.memgrp{idx}.axial_stiffener_web_height"] = istruct["longitudinal_stiffeners"][
                "web_height"
            ]
            wt_opt[f"floating.memgrp{idx}.axial_stiffener_web_thickness"] = istruct["longitudinal_stiffeners"][
                "web_thickness"
            ]
            wt_opt[f"floating.memgrp{idx}.axial_stiffener_flange_thickness"] = istruct["longitudinal_stiffeners"][
                "flange_thickness"
            ]
            wt_opt[f"floating.memgrp{idx}.axial_stiffener_flange_width"] = istruct["longitudinal_stiffeners"][
                "flange_width"
            ]
            wt_opt[f"floating.memgrp{idx}.axial_stiffener_spacing"] = istruct["longitudinal_stiffeners"]["spacing"]

        n_ballasts = floating_init_options["members"]["n_ballasts"][i]
        ballast_mat = [""] * n_ballasts
        for j in range(n_ballasts):
            wt_opt[f"floating.memgrp{idx}.ballast_grid"][j, :] = istruct["ballasts"][j]["grid"]
            if floating_init_options["members"]["ballast_flag_member_" + name_member][j] == False:
                wt_opt[f"floating.memgrp{idx}.ballast_volume"][j] = istruct["ballasts"][j]["volume"]
                ballast_mat[j] = istruct["ballasts"][j]["material"]
            else:
                wt_opt[f"floating.memgrp{idx}.ballast_volume"][j] = 0.0
                ballast_mat[j] = "seawater"
        wt_opt[f"floating.memgrp{idx}.ballast_materials"] = ballast_mat

        if floating_init_options["members"]["n_axial_joints"][i] > 0:
            for j in range(floating_init_options["members"]["n_axial_joints"][i]):
                wt_opt[f"floating.memgrp{idx}.grid_axial_joints"][j] = floating["members"][i]["axial_joints"][j]["grid"]

    if "Loading" in modeling_options["WISDEM"]:
        if modeling_options["flags"]["tower"]:
            wt_opt["floatingse.rna_mass"] = modeling_options["WISDEM"]["Loading"]["mass"]
            wt_opt["floatingse.rna_cg"] = modeling_options["WISDEM"]["Loading"]["center_of_mass"]
            wt_opt["floatingse.rna_I"] = modeling_options["WISDEM"]["Loading"]["moment_of_inertia"]

    return wt_opt


def assign_mooring_values(wt_opt, modeling_options, mooring):
    # Mooring system parameters
    mooring_init_options = modeling_options["mooring"]

    n_nodes = mooring_init_options["n_nodes"]
    n_lines = mooring_init_options["n_lines"]
    n_line_types = mooring_init_options["n_line_types"]
    n_anchor_types = mooring_init_options["n_anchor_types"]
    n_design = 1 if mooring_init_options["symmetric"] else n_lines

    wt_opt["mooring.n_lines"] = n_lines  # Needed for ORBIT
    wt_opt["mooring.node_names"] = [mooring["nodes"][i]["name"] for i in range(n_nodes)]
    wt_opt["mooring.nodes_joint_name"] = ["" for i in range(n_nodes)]
    wt_opt["mooring.line_id"] = [mooring["lines"][i]["line_type"] for i in range(n_lines)]
    line_names = [mooring["line_types"][i]["name"] for i in range(n_line_types)]
    anchor_names = [mooring["anchor_types"][i]["name"] for i in range(n_anchor_types)]
    for i in range(n_nodes):
        if "location" in mooring["nodes"][i]:
            wt_opt["mooring.nodes_location"][i, :] = mooring["nodes"][i]["location"]
        else:
            wt_opt["mooring.nodes_joint_name"][i] = mooring["nodes"][i]["joint"]
        wt_opt["mooring.nodes_mass"][i] = mooring["nodes"][i]["node_mass"]
        wt_opt["mooring.nodes_volume"][i] = mooring["nodes"][i]["node_volume"]
        wt_opt["mooring.nodes_drag_area"][i] = mooring["nodes"][i]["drag_area"]
        wt_opt["mooring.nodes_added_mass"][i] = mooring["nodes"][i]["added_mass"]

    for i in range(n_design):
        wt_opt["mooring.unstretched_length_in"][i] = mooring["lines"][i]["unstretched_length"]

    for jj, jname in enumerate(wt_opt["mooring.line_id"]):
        node1 = mooring["lines"][jj]["node1"]
        node2 = mooring["lines"][jj]["node2"]
        for ii, iname in enumerate(line_names):
            if jname == iname:
                d2 = mooring["line_types"][ii]["diameter"] ** 2
                if jj < n_design:
                    wt_opt["mooring.line_diameter_in"][jj] = mooring["line_types"][ii]["diameter"]
                if mooring_init_options["line_material"][jj] == "custom":
                    wt_opt["mooring.line_mass_density_coeff"][jj] = mooring["line_types"][ii]["mass_density"] / d2
                    wt_opt["mooring.line_stiffness_coeff"][jj] = mooring["line_types"][ii]["stiffness"] / d2
                    wt_opt["mooring.line_breaking_load_coeff"][jj] = mooring["line_types"][ii]["breaking_load"] / d2
                    wt_opt["mooring.line_cost_rate_coeff"][jj] = mooring["line_types"][ii]["cost"] / d2
                wt_opt["mooring.line_transverse_added_mass_coeff"][jj] = (
                    mooring["line_types"][ii]["transverse_added_mass"] / d2
                )
                wt_opt["mooring.line_tangential_added_mass_coeff"][jj] = (
                    mooring["line_types"][ii]["tangential_added_mass"] / d2
                )
                wt_opt["mooring.line_transverse_drag_coeff"][jj] = mooring["line_types"][ii]["transverse_drag"] / d2
                wt_opt["mooring.line_tangential_drag_coeff"][jj] = mooring["line_types"][ii]["tangential_drag"] / d2
        for ii, iname in enumerate(wt_opt["mooring.node_names"]):
            if node1 == iname or node2 == iname and mooring["nodes"][ii]["node_type"] == "fixed":
                for kk, kname in enumerate(anchor_names):
                    if kname == mooring["nodes"][ii]["anchor_type"]:
                        if mooring_init_options["line_anchor"][jj] == "custom":
                            wt_opt["mooring.anchor_mass"][jj] = mooring["anchor_types"][kk]["mass"]
                            wt_opt["mooring.anchor_cost"][jj] = mooring["anchor_types"][kk]["cost"]
                            wt_opt["mooring.anchor_max_vertical_load"][jj] = mooring["anchor_types"][kk][
                                "max_vertical_load"
                            ]
                            wt_opt["mooring.anchor_max_lateral_load"][jj] = mooring["anchor_types"][kk][
                                "max_lateral_load"
                            ]

    # Give warnings if we have different types or asymmetrical lines
    if (
        np.unique(wt_opt["mooring.unstretched_length"]).size > 1
        or np.unique(wt_opt["mooring.line_diameter"]).size > 1
        or np.unique(wt_opt["mooring.line_mass_density_coeff"]).size > 1
        or np.unique(wt_opt["mooring.line_stiffness_coeff"]).size > 1
        or np.unique(wt_opt["mooring.anchor_mass"]).size > 1
    ):
        logger.debug(
            "WARNING: Multiple mooring line or anchor types entered, but can only process symmetrical arrangements for now"
        )

    return wt_opt


def assign_control_values(wt_opt, modeling_options, control):
    # Controller parameters
    wt_opt["control.V_in"] = control["supervisory"]["Vin"]
    wt_opt["control.V_out"] = control["supervisory"]["Vout"]
    wt_opt["control.minOmega"] = control["torque"]["VS_minspd"]
    wt_opt["control.maxOmega"] = control["torque"]["VS_maxspd"]
    wt_opt["control.rated_TSR"] = control["torque"]["tsr"]
    wt_opt["control.rated_pitch"] = control["pitch"]["min_pitch"]
    wt_opt["control.max_TS"] = control["supervisory"]["maxTS"]
    wt_opt["control.max_pitch_rate"] = control["pitch"]["max_pitch_rate"]
    wt_opt["control.max_torque_rate"] = control["torque"]["max_torque_rate"]

    return wt_opt


def assign_configuration_values(wt_opt, assembly, opt_options):
    class_val = assembly["turbine_class"].upper()
    if class_val in [1, "1"]:
        class_val = "I"
    elif class_val in [2, "2"]:
        class_val = "II"
    elif class_val in [3, "3"]:
        class_val = "III"
    elif class_val in [4, "4"]:
        class_val = "IV"
    wt_opt["configuration.ws_class"] = class_val
    wt_opt["configuration.turb_class"] = assembly["turbulence_class"].upper()
    wt_opt["configuration.gearbox_type"] = assembly["drivetrain"].lower()
    wt_opt["configuration.rotor_orientation"] = assembly["rotor_orientation"].lower()
    wt_opt["configuration.upwind"] = wt_opt["configuration.rotor_orientation"] == "upwind"
    wt_opt["configuration.n_blades"] = int(assembly["number_of_blades"])
    wt_opt["configuration.rotor_diameter_user"] = assembly["rotor_diameter"]
    wt_opt["configuration.hub_height_user"] = assembly["hub_height"]
    wt_opt["configuration.rated_power"] = assembly["rated_power"]
    wt_opt["configuration.lifetime"] = assembly["lifetime"]

    # Checks for errors
    if int(assembly["number_of_blades"]) - assembly["number_of_blades"] != 0:
        raise Exception("ERROR: the number of blades must be an integer")

    if assembly["rotor_diameter"] == 0.0 and opt_options["design_variables"]["rotor_diameter"]["flag"]:
        raise Exception(
            "ERROR: you activated the rotor diameter as design variable, but you have not specified the rotor diameter in the geometry yaml."
        )

    return wt_opt


def assign_environment_values(wt_opt, environment, offshore, blade_flag):
    wt_opt["env.rho_air"] = environment["air_density"]
    wt_opt["env.mu_air"] = environment["air_dyn_viscosity"]
    if offshore:
        wt_opt["env.rho_water"] = environment["water_density"]
        wt_opt["env.mu_water"] = environment["water_dyn_viscosity"]
        wt_opt["env.water_depth"] = environment["water_depth"]
        wt_opt["env.Hsig_wave"] = environment["significant_wave_height"]
        wt_opt["env.Tsig_wave"] = environment["significant_wave_period"]
    wt_opt["env.weibull_k"] = environment["weib_shape_parameter"]
    wt_opt["env.speed_sound_air"] = environment["air_speed_sound"]
    wt_opt["env.shear_exp"] = environment["shear_exp"]
    wt_opt["env.G_soil"] = environment["soil_shear_modulus"]
    wt_opt["env.nu_soil"] = environment["soil_poisson"]
    if blade_flag:
        wt_opt["rotorse.wt_class.V_mean_overwrite"] = environment["V_mean"]

    return wt_opt


def assign_bos_values(wt_opt, bos, offshore):
    wt_opt["bos.plant_turbine_spacing"] = bos["plant_turbine_spacing"]
    wt_opt["bos.plant_row_spacing"] = bos["plant_row_spacing"]
    wt_opt["bos.commissioning_pct"] = bos["commissioning_pct"]
    wt_opt["bos.decommissioning_pct"] = bos["decommissioning_pct"]
    wt_opt["bos.distance_to_substation"] = bos["distance_to_substation"]
    wt_opt["bos.distance_to_interconnection"] = bos["distance_to_interconnection"]
    if offshore:
        wt_opt["bos.site_distance"] = bos["distance_to_site"]
        wt_opt["bos.distance_to_landfall"] = bos["distance_to_landfall"]
        wt_opt["bos.port_cost_per_month"] = bos["port_cost_per_month"]
        wt_opt["bos.site_auction_price"] = bos["site_auction_price"]
        wt_opt["bos.site_assessment_plan_cost"] = bos["site_assessment_plan_cost"]
        wt_opt["bos.site_assessment_cost"] = bos["site_assessment_cost"]
        wt_opt["bos.construction_operations_plan_cost"] = bos["construction_operations_plan_cost"]
        wt_opt["bos.boem_review_cost"] = bos["boem_review_cost"]
        wt_opt["bos.design_install_plan_cost"] = bos["design_install_plan_cost"]
    else:
        wt_opt["bos.interconnect_voltage"] = bos["interconnect_voltage"]

    return wt_opt


def assign_costs_values(wt_opt, costs):
    wt_opt["costs.turbine_number"] = costs["turbine_number"]
    wt_opt["costs.opex_per_kW"] = costs["opex_per_kW"]
    wt_opt["costs.bos_per_kW"] = costs["bos_per_kW"]
    wt_opt["costs.wake_loss_factor"] = costs["wake_loss_factor"]
    wt_opt["costs.fixed_charge_rate"] = costs["fixed_charge_rate"]
    wt_opt["costs.labor_rate"] = costs["labor_rate"]
    wt_opt["costs.painting_rate"] = costs["painting_rate"]

    wt_opt["costs.blade_mass_cost_coeff"] = costs["blade_mass_cost_coeff"]
    wt_opt["costs.hub_mass_cost_coeff"] = costs["hub_mass_cost_coeff"]
    wt_opt["costs.pitch_system_mass_cost_coeff"] = costs["pitch_system_mass_cost_coeff"]
    wt_opt["costs.spinner_mass_cost_coeff"] = costs["spinner_mass_cost_coeff"]
    wt_opt["costs.lss_mass_cost_coeff"] = costs["lss_mass_cost_coeff"]
    wt_opt["costs.bearing_mass_cost_coeff"] = costs["bearing_mass_cost_coeff"]
    wt_opt["costs.gearbox_torque_cost"] = costs["gearbox_torque_cost"]
    wt_opt["costs.hss_mass_cost_coeff"] = costs["hss_mass_cost_coeff"]
    wt_opt["costs.generator_mass_cost_coeff"] = costs["generator_mass_cost_coeff"]
    wt_opt["costs.bedplate_mass_cost_coeff"] = costs["bedplate_mass_cost_coeff"]
    wt_opt["costs.yaw_mass_cost_coeff"] = costs["yaw_mass_cost_coeff"]
    wt_opt["costs.converter_mass_cost_coeff"] = costs["converter_mass_cost_coeff"]
    wt_opt["costs.transformer_mass_cost_coeff"] = costs["transformer_mass_cost_coeff"]
    wt_opt["costs.hvac_mass_cost_coeff"] = costs["hvac_mass_cost_coeff"]
    wt_opt["costs.cover_mass_cost_coeff"] = costs["cover_mass_cost_coeff"]
    wt_opt["costs.elec_connec_machine_rating_cost_coeff"] = costs["elec_connec_machine_rating_cost_coeff"]
    wt_opt["costs.platforms_mass_cost_coeff"] = costs["platforms_mass_cost_coeff"]
    wt_opt["costs.tower_mass_cost_coeff"] = costs["tower_mass_cost_coeff"]
    wt_opt["costs.controls_machine_rating_cost_coeff"] = costs["controls_machine_rating_cost_coeff"]
    wt_opt["costs.crane_cost"] = costs["crane_cost"]
    wt_opt["costs.electricity_price"] = costs["electricity_price"]
    wt_opt["costs.reserve_margin_price"] = costs["reserve_margin_price"]
    wt_opt["costs.capacity_credit"] = costs["capacity_credit"]
    wt_opt["costs.benchmark_price"] = costs["benchmark_price"]

    if "offset_tcc_per_kW" in costs:
        wt_opt["costs.offset_tcc_per_kW"] = costs["offset_tcc_per_kW"]

    return wt_opt


def assign_airfoil_values(wt_opt, modeling_options, airfoils, coordinates_only=False):
    # Function to assign values to the openmdao component Airfoils

    n_af = modeling_options["WISDEM"]["RotorSE"]["n_af"]
    n_aoa = modeling_options["WISDEM"]["RotorSE"]["n_aoa"]
    aoa = modeling_options["WISDEM"]["RotorSE"]["aoa"]
    n_Re = modeling_options["WISDEM"]["RotorSE"]["n_Re"]
    n_tab = modeling_options["WISDEM"]["RotorSE"]["n_tab"]
    n_xy = modeling_options["WISDEM"]["RotorSE"]["n_xy"]

    name = n_af * [""]
    ac = np.zeros(n_af)
    r_thick = np.zeros(n_af)
    Re_all = []
    for i in range(n_af):
        name[i] = airfoils[i]["name"]
        ac[i] = airfoils[i]["aerodynamic_center"]
        r_thick[i] = airfoils[i]["relative_thickness"]
        for j in range(len(airfoils[i]["polars"])):
            Re_all.append(airfoils[i]["polars"][j]["re"])
    Re = np.unique(Re_all)

    cl = np.zeros((n_af, n_aoa, n_Re, n_tab))
    cd = np.zeros((n_af, n_aoa, n_Re, n_tab))
    cm = np.zeros((n_af, n_aoa, n_Re, n_tab))

    coord_xy = np.zeros((n_af, n_xy, 2))

    # Interp cl-cd-cm along predefined grid of angle of attack
    for i in range(n_af):
        Re_i = np.array( [airfoils[i]["polars"][j]["re"] for j in range(len(airfoils[i]["polars"]))] )
        n_Re_i = len(np.unique(Re_i))
        Re_j = np.zeros(n_Re_i)
        j_Re = np.zeros(n_Re_i, dtype=int)
        for j in range(n_Re_i):
            Re_j[j] = airfoils[i]["polars"][j]["re"]
            j_Re[j] = np.argmin(np.abs(Re - Re_j[j]))
            for k in range(n_tab):
                cl[i, :, j_Re[j], k] = PchipInterpolator(
                    airfoils[i]["polars"][j]["c_l"]["grid"], airfoils[i]["polars"][j]["c_l"]["values"]
                )(aoa)
                cd[i, :, j_Re[j], k] = PchipInterpolator(
                    airfoils[i]["polars"][j]["c_d"]["grid"], airfoils[i]["polars"][j]["c_d"]["values"]
                )(aoa)
                cm[i, :, j_Re[j], k] = PchipInterpolator(
                    airfoils[i]["polars"][j]["c_m"]["grid"], airfoils[i]["polars"][j]["c_m"]["values"]
                )(aoa)

                if np.abs(cl[i, 0, j, k] - cl[i, -1, j, k]) > 1.0e-5:
                    cl[i, 0, j, k] = cl[i, -1, j, k]
                    logger.debug(
                        "WARNING: Airfoil "
                        + name[i]
                        + " has the lift coefficient at Re "
                        + str(Re_j[j])
                        + " different between + and - pi rad. This is fixed automatically, but please check the input data."
                    )
                if np.abs(cd[i, 0, j, k] - cd[i, -1, j, k]) > 1.0e-5:
                    cd[i, 0, j, k] = cd[i, -1, j, k]
                    logger.debug(
                        "WARNING: Airfoil "
                        + name[i]
                        + " has the drag coefficient at Re "
                        + str(Re_j[j])
                        + " different between + and - pi rad. This is fixed automatically, but please check the input data."
                    )
                if np.abs(cm[i, 0, j, k] - cm[i, -1, j, k]) > 1.0e-5:
                    cm[i, 0, j, k] = cm[i, -1, j, k]
                    logger.debug(
                        "WARNING: Airfoil "
                        + name[i]
                        + " has the moment coefficient at Re "
                        + str(Re_j[j])
                        + " different between + and - pi rad. This is fixed automatically, but please check the input data."
                    )

        # Re-interpolate cl-cd-cm along the Re dimension if less than n_Re were provided in the input yaml (common condition)
        for l in range(n_aoa):
            for k in range(n_tab):
                cl[i, l, :, k] = (
                    PchipInterpolator(Re_j, cl[i, l, j_Re, k])(Re)
                    if (len(cl[i, l, j_Re, k]) != 1)
                    else cl[i, l, j_Re, k]
                )
                cd[i, l, :, k] = (
                    PchipInterpolator(Re_j, cd[i, l, j_Re, k])(Re)
                    if (len(cd[i, l, j_Re, k]) != 1)
                    else cd[i, l, j_Re, k]
                )
                cm[i, l, :, k] = (
                    PchipInterpolator(Re_j, cm[i, l, j_Re, k])(Re)
                    if (len(cm[i, l, j_Re, k]) != 1)
                    else cm[i, l, j_Re, k]
                )

        points = np.column_stack((airfoils[i]["coordinates"]["x"], airfoils[i]["coordinates"]["y"]))
        # Check that airfoil points are declared from the TE suction side to TE pressure side
        idx_le = np.argmin(points[:, 0])
        if np.mean(points[:idx_le, 1]) > 0.0:
            points = np.flip(points, axis=0)

        # Remap points using class AirfoilShape
        af = AirfoilShape(points=points)
        af.redistribute(n_xy, even=False, dLE=True)
        s = af.s
        af_points = af.points

        # Add trailing edge point if not defined
        if [1, 0] not in af_points.tolist():
            af_points[:, 0] -= af_points[np.argmin(af_points[:, 0]), 0]
        c = max(af_points[:, 0]) - min(af_points[:, 0])
        af_points[:, :] /= c

        coord_xy[i, :, :] = af_points

        # Plotting
        # import matplotlib.pyplot as plt
        # plt.plot(af_points[:,0], af_points[:,1], '.')
        # plt.plot(af_points[:,0], af_points[:,1])
        # plt.show()

    # Assign to openmdao structure
    wt_opt["airfoils.name"] = name
    wt_opt["airfoils.r_thick"] = r_thick
    if coordinates_only == False:
        wt_opt["airfoils.aoa"] = aoa
        wt_opt["airfoils.ac"] = ac
        wt_opt["airfoils.Re"] = Re
        wt_opt["airfoils.cl"] = cl
        wt_opt["airfoils.cd"] = cd
        wt_opt["airfoils.cm"] = cm

    wt_opt["airfoils.coord_xy"] = coord_xy

    return wt_opt


def assign_material_values(wt_opt, modeling_options, materials):
    # Function to assign values to the openmdao component Materials

    n_mat = modeling_options["materials"]["n_mat"]

    name = n_mat * [""]
    orth = np.zeros(n_mat)
    component_id = -np.ones(n_mat)
    rho = np.zeros(n_mat)
    E = np.zeros([n_mat, 3])
    G = np.zeros([n_mat, 3])
    nu = np.zeros([n_mat, 3])
    Xt = np.zeros([n_mat, 3])
    Xc = np.zeros([n_mat, 3])
    S = np.zeros([n_mat, 3])
    sigma_y = np.zeros(n_mat)
    m = np.ones(n_mat)
    A = np.zeros(n_mat)
    rho_fiber = np.zeros(n_mat)
    rho_area_dry = np.zeros(n_mat)
    fvf = np.zeros(n_mat)
    fwf = np.zeros(n_mat)
    ply_t = np.zeros(n_mat)
    roll_mass = np.zeros(n_mat)
    unit_cost = np.zeros(n_mat)
    waste = np.zeros(n_mat)

    for i in range(n_mat):
        name[i] = materials[i]["name"]
        orth[i] = materials[i]["orth"]
        rho[i] = materials[i]["rho"]
        if "component_id" in materials[i]:
            component_id[i] = materials[i]["component_id"]
        if orth[i] == 0:
            if "E" in materials[i]:
                E[i, :] = np.ones(3) * materials[i]["E"]
            if "nu" in materials[i]:
                nu[i, :] = np.ones(3) * materials[i]["nu"]
            if "G" in materials[i]:
                G[i, :] = np.ones(3) * materials[i]["G"]
            elif "nu" in materials[i]:
                G[i, :] = (
                    np.ones(3) * materials[i]["E"] / (2 * (1 + materials[i]["nu"]))
                )  # If G is not provided but the material is isotropic and we have E and nu we can just estimate it
                warning_shear_modulus_isotropic = 'WARNING: NO shear modulus, G, was provided for material "%s". The code assumes 2G*(1 + nu) = E, which is only valid for isotropic materials.'%name[i]
                logger.debug(warning_shear_modulus_isotropic)
            if "Xt" in materials[i]:
                Xt[i, :] = np.ones(3) * materials[i]["Xt"]
            if "Xc" in materials[i]:
                Xc[i, :] = np.ones(3) * materials[i]["Xc"]
            if "S" in materials[i]:
                S[i, :] = np.ones(3) * materials[i]["S"]
        elif orth[i] == 1:
            E[i, :] = materials[i]["E"]
            G[i, :] = materials[i]["G"]
            nu[i, :] = materials[i]["nu"]
            Xt[i, :] = materials[i]["Xt"]
            Xc[i, :] = materials[i]["Xc"]
            if "S" in materials[i]:
                S[i, :] = materials[i]["S"]
            else:
                if modeling_options["WISDEM"]["RotorSE"]["bjs"]:
                    raise Exception(
                        "The blade joint sizer model is activated and requires the material shear strength S, which is not defined in the yaml for material "
                        + materials[i]["name"]
                    )

        else:
            raise ValueError("The flag orth must be set to either 0 or 1. Error in material " + name[i])
        if "fiber_density" in materials[i]:
            rho_fiber[i] = materials[i]["fiber_density"]
        if "area_density_dry" in materials[i]:
            rho_area_dry[i] = materials[i]["area_density_dry"]
        if "fvf" in materials[i]:
            fvf[i] = materials[i]["fvf"]
        if "fwf" in materials[i]:
            fwf[i] = materials[i]["fwf"]
        if "ply_t" in materials[i]:
            ply_t[i] = materials[i]["ply_t"]
        if "roll_mass" in materials[i]:
            roll_mass[i] = materials[i]["roll_mass"]
        if "unit_cost" in materials[i]:
            unit_cost[i] = materials[i]["unit_cost"]
            if unit_cost[i] == 0.0:
                logger.debug("The material " + name[i] + " has zero unit cost associated to it.")
        if "waste" in materials[i]:
            waste[i] = materials[i]["waste"]
        if "Xy" in materials[i]:
            sigma_y[i] = materials[i]["Xy"]
        if "m" in materials[i]:
            m[i] = materials[i]["m"]
        if "A" in materials[i]:
            A[i] = materials[i]["A"]
        if A[i] == 0.0:
            A[i] = np.r_[Xt[i, :], Xc[i, :]].max()

    wt_opt["materials.name"] = name
    wt_opt["materials.orth"] = orth
    wt_opt["materials.rho"] = rho
    wt_opt["materials.sigma_y"] = sigma_y
    wt_opt["materials.component_id"] = component_id
    wt_opt["materials.E"] = E
    wt_opt["materials.G"] = G
    wt_opt["materials.Xt"] = Xt
    wt_opt["materials.Xc"] = Xc
    wt_opt["materials.S"] = S
    wt_opt["materials.nu"] = nu
    wt_opt["materials.wohler_exp"] = m
    wt_opt["materials.wohler_intercept"] = A
    wt_opt["materials.rho_fiber"] = rho_fiber
    wt_opt["materials.rho_area_dry"] = rho_area_dry
    wt_opt["materials.fvf_from_yaml"] = fvf
    wt_opt["materials.fwf_from_yaml"] = fwf
    wt_opt["materials.ply_t_from_yaml"] = ply_t
    wt_opt["materials.roll_mass"] = roll_mass
    wt_opt["materials.unit_cost"] = unit_cost
    wt_opt["materials.waste"] = waste

    return wt_opt


if __name__ == "__main__":
    pass
