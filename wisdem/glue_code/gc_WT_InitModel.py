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
        environment = modeling_options["WISDEM"]["Environment"]
        blade_flag = modeling_options["flags"]["blade"]
        wt_opt = assign_environment_values(wt_opt, environment, offshore, blade_flag)
    else:
        environment = {}

    if modeling_options["flags"]["blade"] or modeling_options["user_elastic"]["blade"]:
        blade = wt_init["components"]["blade"]
        blade_DV = opt_options["design_variables"]["blade"]
        wt_opt = assign_blade_values(wt_opt, modeling_options, blade_DV, blade, modeling_options["user_elastic"]["blade"])
    else:
        blade = {}

    if modeling_options["flags"]["airfoils"]:
        airfoils = wt_init["airfoils"]
        airfoils_master = wt_init["components"]["blade"]["outer_shape"]["airfoils"]
        wt_opt = assign_airfoil_values(wt_opt, modeling_options, airfoils_master, airfoils)
    else:
        airfoils = {}

    if modeling_options["flags"]["control"]:
        control = wt_init["control"]
        wt_opt = assign_control_values(wt_opt, modeling_options, control)
    else:
        control = {}

    user_elastic = (modeling_options["user_elastic"]["hub"] or modeling_options["user_elastic"]["drivetrain"])
    if modeling_options["flags"]["hub"] or modeling_options["flags"]["blade"] or user_elastic or modeling_options["user_elastic"]["blade"]:
        hub = wt_init["components"]["hub"]
        wt_opt = assign_hub_values(wt_opt, hub, modeling_options["flags"], user_elastic)

    if modeling_options["flags"]["drivetrain"] or modeling_options["flags"]["blade"] or user_elastic or modeling_options["user_elastic"]["blade"]:
        drivetrain = wt_init["components"]["drivetrain"]
        wt_opt = assign_drivetrain_values(wt_opt, modeling_options, drivetrain, modeling_options["flags"], user_elastic)

        if modeling_options["flags"]["drivetrain"] or user_elastic:
            wt_opt = assign_generator_values(wt_opt, modeling_options, drivetrain, modeling_options["flags"], user_elastic)

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
        bos = modeling_options["WISDEM"]["BOS"]
        wt_opt = assign_bos_values(wt_opt, bos, offshore)
    else:
        bos = {}

    if modeling_options["flags"]["costs"]:
        costs = modeling_options["WISDEM"]["LCOE"]
        wt_opt = assign_costs_values(wt_opt, costs)
    else:
        costs = {}

    return wt_opt


def MoI_setter(wt_opt, varstr, listin):
    nn = len(listin)
    if nn == 6:
        wt_opt[varstr] = listin
    elif nn == 3:
        wt_opt[varstr][:3] = listin
    else:
        raise ValueError(f"When setting, {varstr}, expected 3 or 6 elements but found {nn}")


def assign_blade_values(wt_opt, modeling_options, blade_DV, blade, user_elastic):
    # Function to assign values to the openmdao group Blade
    
    nd_span = modeling_options["WISDEM"]["RotorSE"]["nd_span"]
    wt_opt["blade.ref_axis"][:, 0] = PchipInterpolator(
    blade["reference_axis"]["x"]["grid"], blade["reference_axis"]["x"]["values"]
    )(nd_span)
    wt_opt["blade.ref_axis"][:, 1] = PchipInterpolator(
        blade["reference_axis"]["y"]["grid"], blade["reference_axis"]["y"]["values"]
    )(nd_span)
    wt_opt["blade.ref_axis"][:, 2] = PchipInterpolator(
        blade["reference_axis"]["z"]["grid"], blade["reference_axis"]["z"]["values"]
    )(nd_span)

    
    blade_DV_aero = blade_DV["aero_shape"]
    wt_opt = assign_outer_shape_values(wt_opt, modeling_options, blade_DV_aero, blade["outer_shape"])
    if not user_elastic:
        wt_opt = assign_blade_structural_webs_values(wt_opt, modeling_options, blade["structure"])
        wt_opt = assign_blade_structural_layers_values(wt_opt, modeling_options, blade["structure"])
        wt_opt = assign_blade_root_joint_values(wt_opt, blade["structure"])
    else:
        wt_opt = assign_user_elastic(wt_opt, blade["elastic_properties"])

    return wt_opt


def assign_outer_shape_values(wt_opt, modeling_options, blade_DV_aero, outer_shape):
    # Function to assign values to the openmdao component Blade_Outer_Shape_BEM

    nd_span = modeling_options["WISDEM"]["RotorSE"]["nd_span"]
    n_af_master = modeling_options["WISDEM"]["RotorSE"]["n_af_master"]

    for i in range(n_af_master):
        wt_opt["blade.outer_shape.af_position"][i] = outer_shape["airfoils"][i]["spanwise_position"]
        wt_opt["blade.opt_var.af_position"][i] = outer_shape["airfoils"][i]["spanwise_position"]

    wt_opt["blade.outer_shape.s"] = nd_span
    wt_opt["blade.outer_shape.chord"] = PchipInterpolator(
        outer_shape["chord"]["grid"], outer_shape["chord"]["values"]
    )(nd_span)
    wt_opt["blade.outer_shape.twist"] = PchipInterpolator(
        outer_shape["twist"]["grid"], outer_shape["twist"]["values"]
    )(nd_span)
    wt_opt["blade.outer_shape.section_offset_x"] = PchipInterpolator(
        outer_shape["section_offset_x"]["grid"], outer_shape["section_offset_x"]["values"]
    )(nd_span)
    if "section_offset_y" in outer_shape:
        wt_opt["blade.outer_shape.section_offset_y"] = PchipInterpolator(
            outer_shape["section_offset_y"]["grid"], outer_shape["section_offset_y"]["values"]
        )(nd_span)
    af_opt_flag = blade_DV_aero["af_positions"]["flag"]
    if "rthick" in outer_shape and af_opt_flag == False:
        # If rthick is defined in input yaml and we are NOT optimizing airfoil positions
        wt_opt["blade.outer_shape.rthick_yaml"] = PchipInterpolator(
            outer_shape["rthick"]["grid"], outer_shape["rthick"]["values"]
        )(nd_span)
    elif "rthick" in outer_shape and af_opt_flag == True:
        logger.debug("rthick field in input geometry yaml is specified but neglected since you are optimizing airfoil positions")
    else:
        logger.debug("rthick field in input geometry yaml not specified. rthick is reconstructed from discrete airfoil positions")

    return wt_opt


def assign_blade_structural_webs_values(wt_opt, modeling_options, structure):
    # Function to assign values to the openmdao component Blade_Structure

    n_webs = modeling_options["WISDEM"]["RotorSE"]["n_webs"]
    nd_span = wt_opt["blade.outer_shape.s"]
    anchors = structure["anchors"]
    n_anchors = len(anchors)
    for i in range(n_webs):
        web_i = structure["webs"][i]
        # web_start_nd
        if "grid" in web_i["start_nd_arc"] and "values" in web_i["start_nd_arc"]:
            web_start_nd_grid = web_i["start_nd_arc"]["grid"]
            web_start_nd_values = web_i["start_nd_arc"]["values"]
        elif "anchor" in web_i["start_nd_arc"]:
            anchor_name = web_i["start_nd_arc"]["anchor"]["name"]
            anchor_handle = web_i["start_nd_arc"]["anchor"]["handle"]
            for j in range(n_anchors):
                if anchors[j]["name"] == anchor_name:
                    web_start_nd_grid = anchors[j][anchor_handle]["grid"]
                    web_start_nd_values = anchors[j][anchor_handle]["values"]
                    break
        else:
            raise Exception(
                "Blade structure web start_nd_arc must be defined by either grid/values or anchor"
            )
        
        
        web_start_nd = np.nan_to_num(
                    PchipInterpolator(
                        web_start_nd_grid,
                        web_start_nd_values,
                        extrapolate=False,
                    )(nd_span)
                )
        wt_opt["blade.structure.web_start_nd"][i, :] = web_start_nd

        # web_end_nd
        if "grid" in web_i["end_nd_arc"] and "values" in web_i["end_nd_arc"]:
            web_end_nd_grid = web_i["end_nd_arc"]["grid"]
            web_end_nd_values = web_i["end_nd_arc"]["values"]
        elif "anchor" in web_i["end_nd_arc"]:
            anchor_name = web_i["end_nd_arc"]["anchor"]["name"]
            anchor_handle = web_i["end_nd_arc"]["anchor"]["handle"]
            for j in range(n_anchors):
                if anchors[j]["name"] == anchor_name:
                    web_end_nd_grid = anchors[j][anchor_handle]["grid"]
                    web_end_nd_values = anchors[j][anchor_handle]["values"]
                    break
        else:
            raise Exception(
                "Blade structure web end_nd_arc must be defined by either grid/values or anchor"
            )
    
        web_end_nd = np.nan_to_num(
                    PchipInterpolator(
                        web_end_nd_grid,
                        web_end_nd_values,
                        extrapolate=False,
                    )(nd_span)
                )
        wt_opt["blade.structure.web_end_nd"][i, :] = web_end_nd

    return wt_opt


def assign_blade_structural_layers_values(wt_opt, modeling_options, structure):
    # Function to assign values to the openmdao component Blade_Structure
    n_layers = modeling_options["WISDEM"]["RotorSE"]["n_layers"]
    n_webs = modeling_options["WISDEM"]["RotorSE"]["n_webs"]
    nd_span = wt_opt["blade.outer_shape.s"]
    anchors = structure["anchors"]
    webs = structure["webs"]
    n_anchors = len(anchors)
    layer_location = -np.ones(n_layers)
    for i in range(n_layers):
        layer_i = structure["layers"][i]
        # layer_start_nd
        if "grid" in layer_i["start_nd_arc"] and "values" in layer_i["start_nd_arc"]:
            layer_start_nd_grid = layer_i["start_nd_arc"]["grid"]
            layer_start_nd_values = layer_i["start_nd_arc"]["values"]
        elif "anchor" in layer_i["start_nd_arc"]:
            anchor_name = layer_i["start_nd_arc"]["anchor"]["name"]
            anchor_handle = layer_i["start_nd_arc"]["anchor"]["handle"]
            anchor_start_found = False
            for j in range(n_anchors):
                if anchor_name == anchors[j]["name"]:
                    anchor_start_found = True
                    layer_start_nd_grid = anchors[j][anchor_handle]["grid"]
                    layer_start_nd_values = anchors[j][anchor_handle]["values"]
                    layer_location[i] = 0
                    break
            if not anchor_start_found:
                for j in range(n_webs):
                    if len(webs[j]["anchors"]) > 1:
                        raise Exception(
                            f"WISDEM does not support multiple anchors per web yet. Please contact the NREL developers."
                        )
                    if anchor_name == webs[j]["anchors"][0]["name"]:
                        anchor_start_found = True
                        layer_location[i] = j + 1
                        layer_start_nd_grid = np.linspace(0., 1., len(nd_span))
                        layer_start_nd_values = np.zeros(len(nd_span))
                        break

            if not anchor_start_found:
                
                raise Exception(
                    f"Blade structure layer {layer_i["name"]} start_nd_arc anchor {anchor_name} not found in anchors list"
                )
        else:
            raise Exception(
                "Blade structure layer start_nd_arc must be defined by either grid/values or anchor"
            )
        
        
        layer_start_nd = np.nan_to_num(
                    PchipInterpolator(
                        layer_start_nd_grid,
                        layer_start_nd_values,
                        extrapolate=False,
                    )(nd_span)
                )
        wt_opt["blade.structure.layer_start_nd"][i, :] = layer_start_nd

        # layer_end_nd
        if "grid" in layer_i["end_nd_arc"] and "values" in layer_i["end_nd_arc"]:
            layer_end_nd_grid = layer_i["end_nd_arc"]["grid"]
            layer_end_nd_values = layer_i["end_nd_arc"]["values"]
        elif "anchor" in layer_i["end_nd_arc"]:
            anchor_name = layer_i["end_nd_arc"]["anchor"]["name"]
            anchor_handle = layer_i["end_nd_arc"]["anchor"]["handle"]
            anchor_end_found = False
            for j in range(n_anchors):
                if anchors[j]["name"] == anchor_name:
                    anchor_end_found = True
                    layer_end_nd_grid = anchors[j][anchor_handle]["grid"]
                    layer_end_nd_values = anchors[j][anchor_handle]["values"]
                    break
            if not anchor_end_found:
                for j in range(n_webs):
                    if len(webs[j]["anchors"]) > 1:
                        raise Exception(
                            f"WISDEM does not support multiple anchors per web yet. Please contact the NREL developers."
                        )
                    if anchor_name == webs[j]["anchors"][0]["name"]:
                        anchor_end_found = True
                        layer_location[i] = j + 1
                        layer_end_nd_grid = np.linspace(0., 1., len(nd_span))
                        layer_end_nd_values = np.zeros(len(nd_span))
                        break

            if not anchor_end_found:
                raise Exception(
                    f"Blade structure layer {layer_i["name"]} end_nd_arc anchor {anchor_name} not found in anchors list"
                )
        else:
            raise Exception(
                "Blade structure layer end_nd_arc must be defined by either grid/values or anchor"
            )
    
        layer_end_nd = np.nan_to_num(
                    PchipInterpolator(
                        layer_end_nd_grid,
                        layer_end_nd_values,
                        extrapolate=False,
                    )(nd_span)
                )
        wt_opt["blade.structure.layer_end_nd"][i, :] = layer_end_nd
        
        # thickness
        if "thickness" in layer_i:
            layer_thickness_grid = layer_i["thickness"]["grid"]
            layer_thickness_values = layer_i["thickness"]["values"]
    
        layer_thickness = np.nan_to_num(
                    PchipInterpolator(
                        layer_thickness_grid,
                        layer_thickness_values,
                        extrapolate=False,
                    )(nd_span)
                )
        wt_opt["blade.structure.layer_thickness"][i, :] = layer_thickness
        
        # fiber_orientation
        if "fiber_orientation" in layer_i:
            layer_fiber_orientation_grid = layer_i["fiber_orientation"]["grid"]
            layer_fiber_orientation_values = layer_i["fiber_orientation"]["values"]
    
        layer_fiber_orientation = np.nan_to_num(
                    PchipInterpolator(
                        layer_fiber_orientation_grid,
                        layer_fiber_orientation_values,
                        extrapolate=False,
                    )(nd_span)
                )
        wt_opt["blade.structure.layer_fiber_orientation"][i, :] = layer_fiber_orientation
    wt_opt["blade.structure.layer_location"] = layer_location

    return wt_opt


def assign_blade_root_joint_values(wt_opt, structure):

    wt_opt["blade.structure.joint_position"] = structure["joint"]["position"]
    wt_opt["blade.structure.joint_mass"] = structure["joint"]["mass"]
    wt_opt["blade.structure.joint_cost"] = structure["joint"]["cost"]
    wt_opt["blade.structure.d_f"] = structure["root"]["d_f"]
    wt_opt["blade.structure.sigma_max"] = structure["root"]["sigma_max"]

    return wt_opt


def assign_user_elastic(wt_opt, user_elastic_properties):

    nd_span = wt_opt["blade.outer_shape.s"]

    stiff_grid = user_elastic_properties["stiffness_matrix"]["grid"]
    stiffness_matrix = user_elastic_properties["stiffness_matrix"]

    inertia_grid = user_elastic_properties["inertia_matrix"]["grid"]
    inertia_matrix = user_elastic_properties["inertia_matrix"]

    # Assemble stiffnees and inertia matrices
    K11 = PchipInterpolator(stiff_grid, stiffness_matrix["K11"][:])(nd_span)
    K22 = PchipInterpolator(stiff_grid, stiffness_matrix["K22"][:])(nd_span)
    K33 = PchipInterpolator(stiff_grid, stiffness_matrix["K33"][:])(nd_span)
    K44 = PchipInterpolator(stiff_grid, stiffness_matrix["K44"][:])(nd_span)
    K55 = PchipInterpolator(stiff_grid, stiffness_matrix["K55"][:])(nd_span)
    K66 = PchipInterpolator(stiff_grid, stiffness_matrix["K66"][:])(nd_span)
    K12 = PchipInterpolator(stiff_grid, stiffness_matrix["K12"][:])(nd_span)
    K13 = PchipInterpolator(stiff_grid, stiffness_matrix["K13"][:])(nd_span)
    K14 = PchipInterpolator(stiff_grid, stiffness_matrix["K14"][:])(nd_span)
    K15 = PchipInterpolator(stiff_grid, stiffness_matrix["K15"][:])(nd_span)
    K16 = PchipInterpolator(stiff_grid, stiffness_matrix["K16"][:])(nd_span)
    K23 = PchipInterpolator(stiff_grid, stiffness_matrix["K23"][:])(nd_span)
    K24 = PchipInterpolator(stiff_grid, stiffness_matrix["K24"][:])(nd_span)
    K25 = PchipInterpolator(stiff_grid, stiffness_matrix["K25"][:])(nd_span)
    K26 = PchipInterpolator(stiff_grid, stiffness_matrix["K26"][:])(nd_span)
    K34 = PchipInterpolator(stiff_grid, stiffness_matrix["K34"][:])(nd_span)
    K35 = PchipInterpolator(stiff_grid, stiffness_matrix["K35"][:])(nd_span)
    K36 = PchipInterpolator(stiff_grid, stiffness_matrix["K36"][:])(nd_span)
    K45 = PchipInterpolator(stiff_grid, stiffness_matrix["K45"][:])(nd_span)
    K46 = PchipInterpolator(stiff_grid, stiffness_matrix["K46"][:])(nd_span)
    K56 = PchipInterpolator(stiff_grid, stiffness_matrix["K56"][:])(nd_span)

    wt_opt["blade.user_KI.K33"] = K33
    wt_opt["blade.user_KI.K11"] = K11
    wt_opt["blade.user_KI.K22"] = K22
    wt_opt["blade.user_KI.K13"] = K13
    wt_opt["blade.user_KI.K14"] = K14
    wt_opt["blade.user_KI.K15"] = K15
    wt_opt["blade.user_KI.K16"] = K16
    wt_opt["blade.user_KI.K23"] = K23
    wt_opt["blade.user_KI.K24"] = K24
    wt_opt["blade.user_KI.K25"] = K25
    wt_opt["blade.user_KI.K26"] = K26
    wt_opt["blade.user_KI.K12"] = K12
    wt_opt["blade.user_KI.K34"] = K34
    wt_opt["blade.user_KI.K35"] = K35
    wt_opt["blade.user_KI.K36"] = K36
    wt_opt["blade.user_KI.K44"] = K44
    wt_opt["blade.user_KI.K45"] = K45
    wt_opt["blade.user_KI.K46"] = K46
    wt_opt["blade.user_KI.K55"] = K55
    wt_opt["blade.user_KI.K56"] = K56
    wt_opt["blade.user_KI.K66"] = K66

    wt_opt["blade.user_KI.mass"] = PchipInterpolator(inertia_grid, inertia_matrix["mass"][:])(nd_span)
    wt_opt["blade.user_KI.i_plr"] = PchipInterpolator(inertia_grid, inertia_matrix["i_plr"][:])(nd_span)
    wt_opt["blade.user_KI.cm_y"] = PchipInterpolator(inertia_grid, inertia_matrix["cm_y"][:])(nd_span)
    wt_opt["blade.user_KI.cm_x"] = PchipInterpolator(inertia_grid, inertia_matrix["cm_x"][:])(nd_span)
    wt_opt["blade.user_KI.i_flap"] = PchipInterpolator(inertia_grid, inertia_matrix["i_flap"][:])(nd_span)
    wt_opt["blade.user_KI.i_edge"] = PchipInterpolator(inertia_grid, inertia_matrix["i_edge"][:])(nd_span)
    wt_opt["blade.user_KI.i_cp"] = PchipInterpolator(inertia_grid, inertia_matrix["i_cp"][:])(nd_span)

    return wt_opt


def assign_hub_values(wt_opt, hub, flags, user_elastic):
    if flags["hub"] or flags["blade"]:
        wt_opt["hub.diameter"] = hub["diameter"]
        wt_opt["hub.radius"]   = hub["diameter"] / 2
        wt_opt["hub.cone"]     = hub["cone_angle"]
        # wt_opt["hub.drag_coeff"] = hub["drag_coefficient"] # GB: This doesn"t connect to anything

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
        if "spinner_mass_user" in hub:
            wt_opt["hub.spinner_mass_user"] = hub["spinner_mass_user"]
        if "pitch_system_mass_user" in hub:
            wt_opt["hub.pitch_system_mass_user"] = hub["pitch_system_mass_user"]
        if "hub_shell_mass_user" in hub:
            wt_opt["hub.hub_shell_mass_user"] = hub["hub_shell_mass_user"]
        if "hub_system_mass_user" in hub:
            wt_opt["hub.hub_system_mass_user"] = hub["hub_system_mass_user"]

    elif user_elastic:
        # Note that this is stored in the drivese namespace per gc_WT_DataStruct to mimic DrivetrainSE
        # windio v2
        #wt_opt["drivese.hub_system_mass"]         = hub["elastic_properties"]["mass"]
        #wt_opt["drivese.hub_system_cm"]           = hub["elastic_properties"]["location"]
        #MoI_setter(wt_opt, "drivese.hub_system_I", hub["elastic_properties"]["inertia"])
        # windio v1
        wt_opt["drivese.hub_system_mass"] = hub["elastic_properties_mb"]["system_mass"]
        wt_opt["drivese.hub_system_cm"] = hub["elastic_properties_mb"]["system_center_mass"][0]
        MoI_setter(wt_opt, "drivese.hub_system_I", hub["elastic_properties_mb"]["system_inertia"])
        # TODO: This cm isn"t right.  OpenFAST CM is measured from rotor apex.  WISDEM CM is measured from hub flange.

    return wt_opt


def assign_drivetrain_values(wt_opt, modeling_options, drivetrain, flags, user_elastic):
    # Common direct and geared
    wt_opt["drivetrain.uptilt"] = drivetrain["outer_shape"]["uptilt"]
    wt_opt["drivetrain.distance_tt_hub"] = drivetrain["outer_shape"]["distance_tt_hub"]
    wt_opt["drivetrain.overhang"] = drivetrain["outer_shape"]["overhang"]
    wt_opt["drivetrain.gear_ratio"] = drivetrain["gearbox"]["gear_ratio"]
    wt_opt["drivetrain.gearbox_efficiency"] = drivetrain["gearbox"]["efficiency"]

    if flags["drivetrain"]:
        wt_opt["drivetrain.distance_hub_mb"] = drivetrain["outer_shape"]["distance_hub_mb"]
        wt_opt["drivetrain.distance_mb_mb"] = drivetrain["outer_shape"]["distance_mb_mb"]
        wt_opt["drivetrain.damping_ratio"] = drivetrain["gearbox"]["damping_ratio"]
        wt_opt["drivetrain.lss_wall_thickness"] = drivetrain["lss"]["wall_thickness"]
        wt_opt["drivetrain.lss_diameter"] = drivetrain["lss"]["diameter"]
        wt_opt["drivetrain.lss_material"] = drivetrain["lss"]["material"]
        wt_opt["drivetrain.mb1Type"] = drivetrain["other_components"]["mb1Type"]
        wt_opt["drivetrain.mb2Type"] = drivetrain["other_components"]["mb2Type"]
        wt_opt["drivetrain.uptower"] = drivetrain["other_components"]["uptower"]
        wt_opt["drivetrain.hvac_mass_coeff"] = drivetrain["other_components"]["hvac_mass_coefficient"]
        wt_opt["drivetrain.bedplate_material"] = drivetrain["bedplate"]["material"]
        if "mass_user" in drivetrain["bedplate"]:
            wt_opt["drivetrain.bedplate_mass_user"] = drivetrain["bedplate"]["mass_user"]
        if "brake_mass_user" in drivetrain["other_components"]:
            wt_opt["drivetrain.brake_mass_user"] = drivetrain["other_components"]["brake_mass_user"]
        if "mb1_mass_user" in drivetrain["other_components"]:
            wt_opt["drivetrain.mb1_mass_user"] = drivetrain["other_components"]["mb1_mass_user"]
        if "mb2_mass_user" in drivetrain["other_components"]:
            wt_opt["drivetrain.mb2_mass_user"] = drivetrain["mb2_mass_user"]
        if "converter_mass_user" in drivetrain["other_components"]:
            wt_opt["drivetrain.converter_mass_user"] = drivetrain["other_components"]["converter_mass_user"]
        if "transformer_mass_user" in drivetrain["other_components"]:
            wt_opt["drivetrain.transformer_mass_user"] = drivetrain["other_components"]["transformer_mass_user"]
        if "spring_constant_user" in drivetrain:
            wt_opt["drivetrain.drivetrain_spring_constant_user"]     = drivetrain["spring_constant_user"]
        if "damping_coefficient_user" in drivetrain:
            wt_opt["drivetrain.drivetrain_damping_coefficient_user"] = drivetrain["damping_coefficient_user"]

        if modeling_options["WISDEM"]["DriveSE"]["direct"]:
            if wt_opt["drivetrain.gear_ratio"] > 1:
                raise Exception(
                    "The gear ratio is larger than 1, but the wind turbine is marked as direct drive. Please check the input yaml file."
                )
            # Direct only
            wt_opt["drivetrain.nose_wall_thickness"] = drivetrain["nose"]["wall_thickness"]
            wt_opt["drivetrain.nose_diameter"] = drivetrain["nose"]["diameter"]

            s_bedplate = np.linspace(0.0, 1.0, len(wt_opt["drivetrain.bedplate_wall_thickness"]))
            s_bed_thick_in = drivetrain["bedplate"]["wall_thickness"]["grid"]
            v_bed_thick_in = drivetrain["bedplate"]["wall_thickness"]["values"]
            wt_opt["drivetrain.bedplate_wall_thickness"] = PchipInterpolator(s_bed_thick_in, v_bed_thick_in)(s_bedplate)
        else:
            if wt_opt["drivetrain.gear_ratio"] == 1:
                raise Exception(
                    "The gear ratio is set to 1, but the wind turbine is marked as geared. Please check the input yaml file."
                )
            # Geared only
            wt_opt["drivetrain.hss_wall_thickness"] = drivetrain["hss"]["wall_thickness"]
            wt_opt["drivetrain.hss_diameter"] = drivetrain["hss"]["diameter"]
            wt_opt["drivetrain.hss_length"] = drivetrain["hss"]["length"]
            wt_opt["drivetrain.hss_material"] = drivetrain["hss"]["material"]
            wt_opt["drivetrain.bedplate_flange_width"] = drivetrain["bedplate"]["flange_width"]
            wt_opt["drivetrain.bedplate_flange_thickness"] = drivetrain["bedplate"]["flange_thickness"]
            wt_opt["drivetrain.bedplate_web_thickness"] = drivetrain["bedplate"]["web_thickness"]
            wt_opt["drivetrain.gear_configuration"] = drivetrain["gearbox"]["gear_configuration"].lower()
            wt_opt["drivetrain.planet_numbers"] = drivetrain["gearbox"]["planet_numbers"]
            if "mass_user" in drivetrain["gearbox"]:
                wt_opt["drivetrain.gearbox_mass_user"] = drivetrain["gearbox"]["mass_user"]
            if "gearbox_radius_user" in drivetrain["gearbox"]:
                wt_opt["drivetrain.gearbox_radius_user"] = drivetrain["gearbox_radius_user"]
            if "gearbox_length_user" in drivetrain["gearbox"]:
                wt_opt["drivetrain.gearbox_length_user"] = drivetrain["gearbox_length_user"]


    elif user_elastic:
        # windio v2
        #wt_opt["drivese.yaw_mass"]          = drivetrain["yaw"]["elastic_properties"]["mass"]
        #wt_opt["drivese.above_yaw_mass"]    = drivetrain["elastic_properties"]["mass"]
        #wt_opt["drivese.above_yaw_cm"]      = drivetrain["elastic_properties"]["location"]
        #wt_opt["drivese.drivetrain_spring_constant"]     = drivetrain["elastic_properties"]["spring_constant"]
        #wt_opt["drivese.drivetrain_damping_coefficient"] = drivetrain["elastic_properties"]["damping_coefficient"]
        #MoI_setter(wt_opt, "drivese.above_yaw_I_TT", drivetrain["elastic_properties"]["inertia"])
        #MoI_setter(wt_opt, "drivese.above_yaw_I", drivetrain["elastic_properties"]["inertia"])
        #if wt_opt["drivetrain.gear_ratio"] > 1:
        #    wt_opt["drivese.gearbox_mass"]  = drivetrain["gearbox"]["elastic_properties"]["mass"]
        #    wt_opt["drivese.gearbox_I"]     = drivetrain["gearbox"]["elastic_properties"]["inertia"]
        #    #wt_opt["drivese.gearbox_cm"]    = drivetrain["gearbox"]["elastic_properties"]["location"]
        #    #wt_opt["drivese.gearbox_stiffness"] = drivetrain["gearbox"]["elastic_properties"]["torsional_stiffness"]
        #    #wt_opt["drivese.gearbox_damping"] = drivetrain["gearbox"]["elastic_properties"]["torsional_damping"]
        # windio v1
        wt_opt["drivese.yaw_mass"]          = drivetrain["elastic_properties_mb"]["yaw_mass"]
        wt_opt["drivese.above_yaw_mass"]    = drivetrain["elastic_properties_mb"]["system_mass"]
        wt_opt["drivese.above_yaw_cm"]      = drivetrain["elastic_properties_mb"]["system_center_mass"]
        wt_opt["drivese.drivetrain_spring_constant"]     = drivetrain["elastic_properties_mb"]["spring_constant"]
        wt_opt["drivese.drivetrain_damping_coefficient"] = drivetrain["elastic_properties_mb"]["damping_coefficient"]
        MoI_setter(wt_opt, "drivese.above_yaw_I_TT", drivetrain["elastic_properties_mb"]["system_inertia_tt"])
        MoI_setter(wt_opt, "drivese.above_yaw_I", drivetrain["elastic_properties_mb"]["system_inertia"])
        wt_opt["drivese.rna_mass"] = wt_opt["drivese.above_yaw_mass"] + wt_opt["drivese.yaw_mass"]
        wt_opt["drivese.rna_cm"]   = wt_opt["drivese.above_yaw_cm"]
        wt_opt["drivese.rna_I_TT"] = wt_opt["drivese.above_yaw_I_TT"]

    return wt_opt


def assign_generator_values(wt_opt, modeling_options, drivetrain, flags, user_elastic):
    if user_elastic:
        #MoI_setter(wt_opt, "drivese.generator_rotor_I", drivetrain["generator"]["elastic_properties"]["rotor_inertia"])
        MoI_setter(wt_opt, "drivese.generator_rotor_I", drivetrain["generator"]["elastic_properties_mb"]["rotor_inertia"])
    else:
        wt_opt["generator.L_generator"] = drivetrain["generator"]["length"]
        if "mass_user" in drivetrain["generator"]:
            wt_opt["generator.generator_mass_user"] = drivetrain["generator"]["mass_user"]

        if not flags["generator"]:
            if "radius_user" in drivetrain["generator"]:
                wt_opt["generator.generator_radius_user"] = drivetrain["generator"]["radius_user"]

            if "rpm_efficiency_user" in drivetrain["generator"]:
                eff_user = np.c_[
                    drivetrain["generator"]["rpm_efficiency_user"]["grid"],
                    drivetrain["generator"]["rpm_efficiency_user"]["values"],
                ]
                n_pc = modeling_options["WISDEM"]["RotorSE"]["n_pc"]
                if np.any(eff_user):
                    newrpm = np.linspace(eff_user[:, 0].min(), eff_user[:, 0].max(), n_pc)
                    neweff = PchipInterpolator(eff_user[:, 0], eff_user[:, 1])(newrpm)
                    myeff = np.c_[newrpm, neweff]
                else:
                    myeff = np.zeros((n_pc, 2))
                wt_opt["generator.generator_efficiency_user"] = myeff

        else:
            wt_opt["generator.B_r"] = drivetrain["generator"]["B_r"]
            wt_opt["generator.P_Fe0e"] = drivetrain["generator"]["P_Fe0e"]
            wt_opt["generator.P_Fe0h"] = drivetrain["generator"]["P_Fe0h"]
            wt_opt["generator.S_N"] = drivetrain["generator"]["S_N"]
            wt_opt["generator.alpha_p"] = drivetrain["generator"]["alpha_p"]
            wt_opt["generator.b_r_tau_r"] = drivetrain["generator"]["b_r_tau_r"]
            wt_opt["generator.b_ro"] = drivetrain["generator"]["b_ro"]
            wt_opt["generator.b_s_tau_s"] = drivetrain["generator"]["b_s_tau_s"]
            wt_opt["generator.b_so"] = drivetrain["generator"]["b_so"]
            wt_opt["generator.cofi"] = drivetrain["generator"]["cofi"]
            wt_opt["generator.freq"] = drivetrain["generator"]["freq"]
            wt_opt["generator.h_i"] = drivetrain["generator"]["h_i"]
            wt_opt["generator.h_sy0"] = drivetrain["generator"]["h_sy0"]
            wt_opt["generator.h_w"] = drivetrain["generator"]["h_w"]
            wt_opt["generator.k_fes"] = drivetrain["generator"]["k_fes"]
            wt_opt["generator.k_fillr"] = drivetrain["generator"]["k_fillr"]
            wt_opt["generator.k_fills"] = drivetrain["generator"]["k_fills"]
            wt_opt["generator.k_s"] = drivetrain["generator"]["k_s"]
            wt_opt["generator.m"] = drivetrain["generator"]["m"]
            wt_opt["generator.mu_0"] = drivetrain["generator"]["mu_0"]
            wt_opt["generator.mu_r"] = drivetrain["generator"]["mu_r"]
            wt_opt["generator.p"] = drivetrain["generator"]["p"]
            wt_opt["generator.phi"] = drivetrain["generator"]["phi"]
            wt_opt["generator.q1"] = drivetrain["generator"]["q1"]
            wt_opt["generator.q2"] = drivetrain["generator"]["q2"]
            wt_opt["generator.ratio_mw2pp"] = drivetrain["generator"]["ratio_mw2pp"]
            wt_opt["generator.resist_Cu"] = drivetrain["generator"]["resist_Cu"]
            wt_opt["generator.sigma"] = drivetrain["generator"]["sigma"]
            wt_opt["generator.y_tau_p"] = drivetrain["generator"]["y_tau_p"]
            wt_opt["generator.y_tau_pr"] = drivetrain["generator"]["y_tau_pr"]

            wt_opt["generator.I_0"] = drivetrain["generator"]["I_0"]
            wt_opt["generator.d_r"] = drivetrain["generator"]["d_r"]
            wt_opt["generator.h_m"] = drivetrain["generator"]["h_m"]
            wt_opt["generator.h_0"] = drivetrain["generator"]["h_0"]
            wt_opt["generator.h_s"] = drivetrain["generator"]["h_s"]
            wt_opt["generator.len_s"] = drivetrain["generator"]["len_s"]
            wt_opt["generator.n_r"] = drivetrain["generator"]["n_r"]
            wt_opt["generator.rad_ag"] = drivetrain["generator"]["rad_ag"]
            wt_opt["generator.t_wr"] = drivetrain["generator"]["t_wr"]

            wt_opt["generator.n_s"] = drivetrain["generator"]["n_s"]
            wt_opt["generator.b_st"] = drivetrain["generator"]["b_st"]
            wt_opt["generator.d_s"] = drivetrain["generator"]["d_s"]
            wt_opt["generator.t_ws"] = drivetrain["generator"]["t_ws"]

            wt_opt["generator.rho_Copper"] = drivetrain["generator"]["rho_Copper"]
            wt_opt["generator.rho_Fe"] = drivetrain["generator"]["rho_Fe"]
            wt_opt["generator.rho_Fes"] = drivetrain["generator"]["rho_Fes"]
            wt_opt["generator.rho_PM"] = drivetrain["generator"]["rho_PM"]

            wt_opt["generator.C_Cu"] = drivetrain["generator"]["C_Cu"]
            wt_opt["generator.C_Fe"] = drivetrain["generator"]["C_Fe"]
            wt_opt["generator.C_Fes"] = drivetrain["generator"]["C_Fes"]
            wt_opt["generator.C_PM"] = drivetrain["generator"]["C_PM"]

            if modeling_options["WISDEM"]["DriveSE"]["generator"]["type"] in ["pmsg_outer"]:
                wt_opt["generator.N_c"] = drivetrain["generator"]["N_c"]
                wt_opt["generator.b"] = drivetrain["generator"]["b"]
                wt_opt["generator.c"] = drivetrain["generator"]["c"]
                wt_opt["generator.E_p"] = drivetrain["generator"]["E_p"]
                wt_opt["generator.h_yr"] = drivetrain["generator"]["h_yr"]
                wt_opt["generator.h_ys"] = drivetrain["generator"]["h_ys"]
                wt_opt["generator.h_sr"] = drivetrain["generator"]["h_sr"]
                wt_opt["generator.h_ss"] = drivetrain["generator"]["h_ss"]
                wt_opt["generator.t_r"] = drivetrain["generator"]["t_r"]
                wt_opt["generator.t_s"] = drivetrain["generator"]["t_s"]

                wt_opt["generator.u_allow_pcent"] = drivetrain["generator"]["u_allow_pcent"]
                wt_opt["generator.y_allow_pcent"] = drivetrain["generator"]["y_allow_pcent"]
                wt_opt["generator.z_allow_deg"] = drivetrain["generator"]["z_allow_deg"]
                wt_opt["generator.B_tmax"] = drivetrain["generator"]["B_tmax"]

            if modeling_options["WISDEM"]["DriveSE"]["generator"]["type"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
                wt_opt["generator.tau_p"] = drivetrain["generator"]["tau_p"]
                wt_opt["generator.h_ys"] = drivetrain["generator"]["h_ys"]
                wt_opt["generator.h_yr"] = drivetrain["generator"]["h_yr"]
                wt_opt["generator.b_arm"] = drivetrain["generator"]["b_arm"]

            elif modeling_options["WISDEM"]["DriveSE"]["generator"]["type"] in ["scig", "dfig"]:
                wt_opt["generator.B_symax"] = drivetrain["generator"]["B_symax"]
                wt_opt["generator.S_Nmax"] = drivetrain["generator"]["S_Nmax"]

    return wt_opt


def assign_tower_values(wt_opt, modeling_options, tower):
    # Function to assign values to the openmdao component Tower
    n_height = modeling_options["WISDEM"]["TowerSE"]["n_height_tower"]  # Number of points along tower height
    n_layers = modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"]

    svec = np.unique(
        np.r_[
            tower["outer_shape"]["outer_diameter"]["grid"],
            tower["reference_axis"]["x"]["grid"],
            tower["reference_axis"]["y"]["grid"],
            tower["reference_axis"]["z"]["grid"],
        ]
    )

    # wt_opt["tower.s"] = svec
    wt_opt["tower.diameter"] = PchipInterpolator(
        tower["outer_shape"]["outer_diameter"]["grid"], tower["outer_shape"]["outer_diameter"]["values"]
    )(svec)
    wt_opt["tower.cd"] = PchipInterpolator(
        tower["outer_shape"]["cd"]["grid"],
        tower["outer_shape"]["cd"]["values"],
    )(svec)

    wt_opt["tower.ref_axis"][:, 0] = PchipInterpolator(
        tower["reference_axis"]["x"]["grid"],
        tower["reference_axis"]["x"]["values"],
    )(svec)
    wt_opt["tower.ref_axis"][:, 1] = PchipInterpolator(
        tower["reference_axis"]["y"]["grid"],
        tower["reference_axis"]["y"]["values"],
    )(svec)
    wt_opt["tower.ref_axis"][:, 2] = PchipInterpolator(
        tower["reference_axis"]["z"]["grid"],
        tower["reference_axis"]["z"]["values"],
    )(svec)

    layer_name = n_layers * [""]
    layer_mat = n_layers * [""]
    thickness = np.zeros((n_layers, n_height))
    for i in range(n_layers):
        layer_name[i] = tower["structure"]["layers"][i]["name"]
        layer_mat[i] = tower["structure"]["layers"][i]["material"]
        thickness[i] = PchipInterpolator(
            tower["structure"]["layers"][i]["thickness"]["grid"],
            tower["structure"]["layers"][i]["thickness"]["values"],
        )(svec)

    wt_opt["tower.layer_name"] = layer_name
    wt_opt["tower.layer_mat"] = layer_mat
    wt_opt["tower.layer_thickness"] = thickness

    wt_opt["tower.outfitting_factor"] = tower["structure"]["outfitting_factor"]
    if "tower_mass_user" in tower:
        wt_opt["tower.tower_mass_user"] = tower["tower_mass_user"]

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
            MoI_setter(wt_opt, "towerse.rna_I", modeling_options["WISDEM"]["Loading"]["moment_of_inertia"])
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
            MoI_setter(wt_opt, "fixedse.rna_I", modeling_options["WISDEM"]["Loading"]["moment_of_inertia"])
            wt_opt["fixedse.monopile.rna_F"] = F
            wt_opt["fixedse.monopile.rna_M"] = M

    return wt_opt


def assign_monopile_values(wt_opt, modeling_options, monopile):
    # Function to assign values to the openmdao component Monopile
    n_height = modeling_options["WISDEM"]["FixedBottomSE"]["n_height"]  # Number of points along monopile height
    n_layers = modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"]

    svec = np.unique(
        np.r_[
            monopile["outer_shape"]["outer_diameter"]["grid"],
            monopile["reference_axis"]["x"]["grid"],
            monopile["reference_axis"]["y"]["grid"],
            monopile["reference_axis"]["z"]["grid"],
        ]
    )

    wt_opt["monopile.s"] = svec
    wt_opt["monopile.diameter"] = PchipInterpolator(
        monopile["outer_shape"]["outer_diameter"]["grid"],
        monopile["outer_shape"]["outer_diameter"]["values"],
    )(svec)

    wt_opt["monopile.ref_axis"][:, 0] = PchipInterpolator(
        monopile["reference_axis"]["x"]["grid"],
        monopile["reference_axis"]["x"]["values"],
    )(svec)
    wt_opt["monopile.ref_axis"][:, 1] = PchipInterpolator(
        monopile["reference_axis"]["y"]["grid"],
        monopile["reference_axis"]["y"]["values"],
    )(svec)
    wt_opt["monopile.ref_axis"][:, 2] = PchipInterpolator(
        monopile["reference_axis"]["z"]["grid"],
        monopile["reference_axis"]["z"]["values"],
    )(svec)

    layer_name = n_layers * [""]
    layer_mat = n_layers * [""]
    thickness = np.zeros((n_layers, n_height))
    for i in range(n_layers):
        layer_name[i] = monopile["structure"]["layers"][i]["name"]
        layer_mat[i] = monopile["structure"]["layers"][i]["material"]
        thickness[i] = PchipInterpolator(
            monopile["structure"]["layers"][i]["thickness"]["grid"],
            monopile["structure"]["layers"][i]["thickness"]["values"],
        )(svec)

    wt_opt["monopile.layer_name"] = layer_name
    wt_opt["monopile.layer_mat"] = layer_mat
    wt_opt["monopile.layer_thickness"] = thickness

    wt_opt["monopile.outfitting_factor"] = monopile["structure"]["outfitting_factor"]
    wt_opt["monopile.transition_piece_mass"] = monopile["transition_piece_mass"]
    wt_opt["monopile.transition_piece_cost"] = monopile["transition_piece_cost"]
    wt_opt["monopile.gravity_foundation_mass"] = monopile["gravity_foundation_mass"]
    if "monopile_mass_user" in monopile:
        wt_opt["monopile.monopile_mass_user"] = monopile["monopile_mass_user"]

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
    wt_opt["jacket.jacket_mass_user"] = jacket["jacket_mass_user"]

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

    # Assign rigid body info
    for k, rb in enumerate(floating["rigid_bodies"]):
        rb_joint_index = floating_init_options["joints"]["name2idx"][rb["joint1"]]
        wt_opt[f"floating.rigid_body_{k}_node"] = floating["joints"][rb_joint_index]["location"]
        wt_opt[f"floating.rigid_body_{k}_mass"] = rb["mass"]
        wt_opt[f"floating.rigid_body_{k}_inertia"] = rb["moments_of_inertia"]

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
            if isinstance(floating["members"][i][coeff], list):
                coeff_length = len(floating["members"][i][coeff])
                if usr_defined_flag[coeff]:
                        assert grid_length == coeff_length, f"Users define {coeff} array along member {name_member} for different sectitions, but the coefficient array length is different from grid length. Please correct them to consistent or you can also define {coeff} as a scalar constant."
            else:
            # If the coefficient is a constant, make it a list with one constant. Just for each of operation and simplicity, so the we can uniformlly treat it as list later and no need for extra conditionals.
                floating["members"][i][coeff] = [floating["members"][i][coeff]]*grid_length


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

        wt_opt[f"floating.memgrp{idx}.member_mass_user"] = floating["members"][i]["member_mass_user"]

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
    wt_opt["control.ps_percent"] = control["pitch"]["ps_percent"]
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
    wt_opt["bos.commissioning_cost_kW"] = bos["commissioning_cost_kW"]
    wt_opt["bos.decommissioning_cost_kW"] = bos["decommissioning_cost_kW"]
    wt_opt["bos.distance_to_substation"] = bos["distance_to_substation"]
    wt_opt["bos.distance_to_interconnection"] = bos["distance_to_interconnection"]
    if offshore:
        wt_opt["bos.site_distance"] = bos["distance_to_site"]
        wt_opt["bos.distance_to_landfall"] = bos["distance_to_landfall"]
        wt_opt["bos.port_cost_per_month"] = bos["port_cost_per_month"]
        wt_opt["bos.site_auction_price"] = bos["site_auction_price"]
        wt_opt["bos.site_assessment_cost"] = bos["site_assessment_cost"]
        wt_opt["bos.construction_insurance"] = bos["construction_insurance"]
        wt_opt["bos.construction_financing"] = bos["construction_financing"]
        wt_opt["bos.contingency"] = bos["contingency"]
        wt_opt["bos.construction_plan_cost"] = bos["construction_plan_cost"]
        wt_opt["bos.installation_plan_cost"] = bos["installation_plan_cost"]
        wt_opt["bos.boem_review_cost"] = bos["review_cost"]
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


def assign_airfoil_values(wt_opt, modeling_options, airfoils_master, airfoils, coordinates_only=False):
    # Function to assign values to the openmdao component Airfoils

    
    n_af_database = modeling_options["WISDEM"]["RotorSE"]["n_af_database"]
    n_af_master = modeling_options["WISDEM"]["RotorSE"]["n_af_master"]
    af_master = modeling_options["WISDEM"]["RotorSE"]["af_master"]
    n_aoa = modeling_options["WISDEM"]["RotorSE"]["n_aoa"]
    aoa = modeling_options["WISDEM"]["RotorSE"]["aoa"]
    n_Re = modeling_options["WISDEM"]["RotorSE"]["n_Re"]
    Re = modeling_options["WISDEM"]["RotorSE"]["Re"]
    n_xy = modeling_options["WISDEM"]["RotorSE"]["n_xy"]

    coord_xy_master = np.zeros((n_af_master, n_xy, 2))

    ac_master = np.zeros(n_af_master)
    rthick_master = np.zeros(n_af_master)
    cl_master = np.zeros((n_af_master, n_aoa, n_Re))
    cd_master = np.zeros((n_af_master, n_aoa, n_Re))
    cm_master = np.zeros((n_af_master, n_aoa, n_Re))
    
    
    for i in range(n_af_master):
        airfoil_exists = False
        for j in range(n_af_database):
            if af_master[i] == airfoils[j]["name"]:
                airfoil_exists = True
                ac_master[i] = airfoils[j]["aerodynamic_center"]
                rthick_master[i] = airfoils[j]["rthick"]
                points = np.column_stack((airfoils[j]["coordinates"]["x"], airfoils[j]["coordinates"]["y"]))
                # Check that airfoil points are declared from the TE suction side to TE pressure side
                idx_le = np.argmin(points[:, 0])
                if np.mean(points[:idx_le, 1]) > 0.0:
                    points = np.flip(points, axis=0)

                # Remap points using class AirfoilShape
                af = AirfoilShape(points=points)
                af.redistribute(n_xy, even=False, dLE=True)
                af_points = af.points

                # Add trailing edge point if not defined
                if [1, 0] not in af_points.tolist():
                    af_points[:, 0] -= af_points[np.argmin(af_points[:, 0]), 0]
                c = max(af_points[:, 0]) - min(af_points[:, 0])
                af_points[:, :] /= c

                coord_xy_master[i, :, :] = af_points

                if coordinates_only:
                    break
                
                # now move on to the polars, first combining polars across configurations
                n_configs = len(airfoils_master[i]["configuration"])
                configuration = [''] * n_configs
                weights = np.zeros(n_configs)
                for k in range(n_configs):
                    configuration[k] = airfoils_master[i]["configuration"][k]
                    weights[k] = airfoils_master[i]["weight"][k]
                    config_exist = False
                    for l in range(len(airfoils[j]["polars"])):
                        if configuration[k] == airfoils[j]["polars"][l]["configuration"]:
                            config_exist = True
                            n_re_config = len(airfoils[j]["polars"][l]["re_sets"])
                            re_config = np.zeros(n_re_config)
                            cl_config = np.zeros((n_aoa, n_re_config, n_configs))
                            cd_config = np.zeros((n_aoa, n_re_config, n_configs))
                            cm_config = np.zeros((n_aoa, n_re_config, n_configs))
                            for re_i in range(len(airfoils[j]["polars"][l]["re_sets"])):
                                cl_config[:, re_i, k] = PchipInterpolator(
                                    airfoils[j]["polars"][l]["re_sets"][re_i]["cl"]["grid"], airfoils[j]["polars"][l]["re_sets"][re_i]["cl"]["values"]
                                )(aoa)
                            
                                cd_config[:, re_i, k] = PchipInterpolator(
                                    airfoils[j]["polars"][l]["re_sets"][re_i]["cd"]["grid"], airfoils[j]["polars"][l]["re_sets"][re_i]["cd"]["values"]
                                )(aoa)
                                cm_config[:, re_i, k] = PchipInterpolator(
                                    airfoils[j]["polars"][l]["re_sets"][re_i]["cm"]["grid"], airfoils[j]["polars"][l]["re_sets"][re_i]["cm"]["values"]
                                )(aoa)

                                re_config = airfoils[j]["polars"][l]["re_sets"][re_i]["re"]

                                break
                    # Check if the configuration exists
                    if not config_exist:
                        raise ValueError(
                            f"Configuration {configuration[k]} not found for airfoil {af_master[i]}. Please check the configuration names for airfoil polars."
                        )

                # Perform weighted average across configurations
                if abs(sum(weights) - 1.0) > 1e-6:
                    raise ValueError(
                        f"Configuration weights for airfoil {af_master[i]} do not sum to 1.0. Please check the configuration weights."
                    )

                cl_master_i = np.average(cl_config[:, :, :], axis=2, weights=weights)
                cd_master_i = np.average(cd_config[:, :, :], axis=2, weights=weights)
                cm_master_i = np.average(cm_config[:, :, :], axis=2, weights=weights)

                # Interpolate across Re sets
                if n_re_config == 1:
                    for j in range(n_Re):
                        cl_master[i, :, j] = cl_master_i[:, 0]
                        cd_master[i, :, j] = cd_master_i[:, 0]
                        cm_master[i, :, j] = cm_master_i[:, 0]
                else:
                    for j in range(n_aoa):
                        cl_master[i, j, :] = PchipInterpolator(
                                            re_config, cl_master_i[j, :]
                                        )(Re)
                        cd_master[i, j, :] = PchipInterpolator(
                                            re_config, cd_master_i[j, :]
                                        )(Re)
                        cm_master[i, j, :] = PchipInterpolator(
                                            re_config, cm_master_i[j, :]
                                        )(Re)
                            
                break

        if not airfoil_exists:
            raise ValueError(
                f"Airfoil {af_master[i]} not found in airfoil database. Please check the airfoil names."
            )
    
    # Assign to openmdao structure
    wt_opt["airfoils.coord_xy"] = coord_xy_master
    wt_opt["airfoils.rthick_master"] = rthick_master
    if coordinates_only == False:
        wt_opt["airfoils.aoa"] = aoa
        wt_opt["airfoils.ac"] = ac_master
        wt_opt["airfoils.Re"] = Re
        wt_opt["airfoils.cl"] = cl_master
        wt_opt["airfoils.cd"] = cd_master
        wt_opt["airfoils.cm"] = cm_master


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
                warning_shear_modulus_isotropic = f"WARNING: NO shear modulus, G, was provided for material {name[i]}. The code assumes 2G*(1 + nu) = E, which is only valid for isotropic materials."
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
