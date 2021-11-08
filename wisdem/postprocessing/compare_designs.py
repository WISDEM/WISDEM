"""
This file postprocesses results from WISDEM. This script is installed
as a console script when WISDEM is installed.

This script allows you to compare wind turbine designs via the
outputted yaml files. This script runs an instance of WISDEM for each yaml
file then produces plots and text output comparing the designs and performance.
You can modify the yaml files being compared, as well as which values are plotted
and printed to screen.
See the bottom of this file for some user customization options.

At a terminal, example usage is:

compare_designs blade.yaml optimized_blade.yaml

This will print results to screen, then create and save plots.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import wisdem.postprocessing.wisdem_get as getter
from wisdem.glue_code.runWISDEM import run_wisdem, load_wisdem

this_dir = os.path.dirname(os.path.realpath(__file__))

# These are the modeling and analysis options used to evaluate the designs.
# Both of these yaml files are used for all yamls to make a fair comparison.
fname_modeling_options_default = this_dir + os.sep + "default_modeling_options.yaml"
fname_analysis_options_default = this_dir + os.sep + "default_analysis_options.yaml"


def create_all_plots(
    list_of_sims,
    list_of_labels,
    modeling_options,
    analysis_options,
    folder_output,
    show_plots,
    font_size,
    extension,
):
    mult_flag = len(list_of_sims) > 1

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if not mult_flag:
        colors = ["k"] + colors

    # Twist
    try:
        ftw, axtw = plt.subplots(1, 1, figsize=(5.3, 4))

        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            s_opt_twist = yaml_data["blade.opt_var.s_opt_twist"]
            twist_opt = yaml_data["blade.opt_var.twist_opt"]
            axtw.plot(
                yaml_data["blade.outer_shape_bem.s"],
                np.rad2deg(yaml_data["blade.pa.twist_param"]),
                "-",
                color=colors[idx],
                label=label,
            )
            axtw.plot(s_opt_twist, np.rad2deg(twist_opt), "o", color=colors[idx], markersize=3)

        s_opt_twist = list_of_sims[0]["blade.outer_shape_bem.s"]
        twist_opt = list_of_sims[0]["blade.pa.twist_param"]
        axtw.plot(
            s_opt_twist,
            np.rad2deg(
                twist_opt - analysis_options["design_variables"]["blade"]["aero_shape"]["twist"]["max_decrease"]
            ),
            ":o",
            color=colors[idx + 1],
            markersize=3,
            label="Bounds",
        )
        axtw.plot(
            s_opt_twist,
            np.rad2deg(
                twist_opt + analysis_options["design_variables"]["blade"]["aero_shape"]["twist"]["max_increase"]
            ),
            ":o",
            color=colors[idx + 1],
            markersize=3,
        )

        if mult_flag:
            axtw.legend(fontsize=font_size)

        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Twist [deg]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "twist_opt" + extension
        ftw.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    try:
        # Chord
        fc, axc = plt.subplots(1, 1, figsize=(5.3, 4))

        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            s_opt_chord = yaml_data["blade.opt_var.s_opt_chord"]
            chord_opt = yaml_data["blade.opt_var.chord_opt"]
            axc.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["blade.pa.chord_param"],
                "-",
                color=colors[idx],
                label=label,
            )
            axc.plot(s_opt_chord, chord_opt, "o", color=colors[idx], markersize=3)

        s_opt_chord = list_of_sims[0]["blade.outer_shape_bem.s"]
        chord_opt = list_of_sims[0]["blade.pa.chord_param"]
        axc.plot(
            s_opt_chord,
            np.array(analysis_options["design_variables"]["blade"]["aero_shape"]["chord"]["max_decrease"]) * chord_opt,
            ":o",
            color=colors[idx + 1],
            markersize=3,
            label="Bounds",
        )
        axc.plot(
            s_opt_chord,
            np.array(analysis_options["design_variables"]["blade"]["aero_shape"]["chord"]["max_increase"]) * chord_opt,
            ":o",
            color=colors[idx + 1],
            markersize=3,
        )

        if mult_flag:
            axc.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Chord [m]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "chord" + extension
        fc.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Spar caps
    try:
        fsc, axsc = plt.subplots(1, 1, figsize=(5.3, 4))

        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            s_opt_sc = yaml_data["blade.opt_var.s_opt_spar_cap_ss"]
            sc_opt = yaml_data["blade.opt_var.spar_cap_ss_opt"] * 1e3
            n_layers = yaml_data["blade.ps.layer_thickness_param"].shape[0]
            ilayer = None
            if ilayer is None:
                for i in range(n_layers):
                    layer_name = modeling_options["WISDEM"]["RotorSE"]["layer_name"][i]
                    if modeling_options["WISDEM"]["RotorSE"]["spar_cap_ss"] == layer_name:
                        ilayer = i
            axsc.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["blade.ps.layer_thickness_param"][ilayer, :] * 1e3,
                "-",
                color=colors[idx],
                label=label,
            )
            axsc.plot(s_opt_sc, sc_opt, "o", color=colors[idx], markersize=3)

        s_opt_sc = list_of_sims[0]["blade.outer_shape_bem.s"]
        sc_opt = list_of_sims[0]["blade.ps.layer_thickness_param"][ilayer, :] * 1e3
        axsc.plot(
            s_opt_sc,
            np.array(analysis_options["design_variables"]["blade"]["structure"]["spar_cap_ss"]["max_decrease"])
            * sc_opt,
            ":o",
            color=colors[idx + 1],
            markersize=3,
            label="Bounds",
        )
        axsc.plot(
            s_opt_sc,
            np.array(analysis_options["design_variables"]["blade"]["structure"]["spar_cap_ss"]["max_increase"])
            * sc_opt,
            ":o",
            color=colors[idx + 1],
            markersize=3,
        )

        if mult_flag:
            axsc.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Spar Caps Thickness [mm]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "sc_opt" + extension
        fsc.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Trailing edge reinforcements
    try:
        fte, axte = plt.subplots(1, 1, figsize=(5.3, 4))

        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            s_opt_te = yaml_data["blade.opt_var.s_opt_te_ss"]
            te_opt = yaml_data["blade.opt_var.te_ss_opt"] * 1e3
            n_layers = yaml_data["blade.ps.layer_thickness_param"].shape[0]
            ilayer = None
            if ilayer is None:
                for i in range(n_layers):
                    layer_name = modeling_options["WISDEM"]["RotorSE"]["layer_name"][i]
                    if modeling_options["WISDEM"]["RotorSE"]["te_ss"] == layer_name:
                        ilayer = i
            axte.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["blade.ps.layer_thickness_param"][ilayer, :] * 1e3,
                "-",
                color=colors[idx],
                label=label,
            )
            axte.plot(s_opt_te, te_opt, "o", color=colors[idx], markersize=3)

        s_opt_te = list_of_sims[0]["blade.outer_shape_bem.s"]
        te_opt = list_of_sims[0]["blade.ps.layer_thickness_param"][ilayer, :] * 1e3
        axte.plot(
            s_opt_te,
            np.array(analysis_options["design_variables"]["blade"]["structure"]["te_ss"]["max_decrease"]) * te_opt,
            ":o",
            color=colors[idx + 1],
            markersize=3,
            label="Bounds",
        )
        axte.plot(
            s_opt_te,
            np.array(analysis_options["design_variables"]["blade"]["structure"]["te_ss"]["max_increase"]) * te_opt,
            ":o",
            color=colors[idx + 1],
            markersize=3,
        )

        if mult_flag:
            axte.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("TE Reinforcement Thickness [mm]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "te_opt" + extension
        fte.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Skins
    try:
        f, ax = plt.subplots(1, 1, figsize=(5.3, 4))
        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            ax.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["blade.internal_structure_2d_fem.layer_thickness"][1, :] * 1e3,
                "-",
                color=colors[idx],
                label=label,
            )
        if mult_flag:
            ax.legend(fontsize=font_size)
        # plt.ylim([0., 120])
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Outer Shell Skin Thickness [mm]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "skin_opt" + extension
        f.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Strains spar caps
    try:
        feps, axeps = plt.subplots(1, 1, figsize=(5.3, 4))
        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            axeps.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["rotorse.rs.frame.strainU_spar"] * 1.0e6,
                "-",
                color=colors[idx],
                label=label,
            )
            axeps.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["rotorse.rs.frame.strainL_spar"] * 1.0e6,
                "-",
                color=colors[idx],
            )

        plt.ylim([-5e3, 5e3])
        if mult_flag:
            axeps.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Spar Caps Strains [mu eps]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.2)
        fig_name = "strains_sc_opt" + extension
        feps.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Strains trailing edge
    try:
        fete, axete = plt.subplots(1, 1, figsize=(5.3, 4))
        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            axete.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["rotorse.rs.frame.strainU_te"] * 1.0e6,
                "-",
                color=colors[idx],
                label=label,
            )
            axete.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["rotorse.rs.frame.strainL_te"] * 1.0e6,
                "-",
                color=colors[idx],
            )

        plt.ylim([-5e3, 5e3])
        if mult_flag:
            axete.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Trailing Edge Strains [mu eps]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.2)
        fig_name = "strains_te_opt" + extension
        fete.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Angle of attack and stall angle
    try:
        faoa, axaoa = plt.subplots(1, 1, figsize=(5.3, 4))
        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            axaoa.plot(
                yaml_data["rotorse.s"],
                yaml_data["rotorse.stall_check.aoa_along_span"],
                "-",
                color=colors[idx],
                label=label,
            )
        axaoa.plot(
            yaml_data["rotorse.s"],
            yaml_data["rotorse.stall_check.stall_angle_along_span"],
            ":",
            color=colors[idx + 1],
            label="Stall",
        )
        if mult_flag:
            axaoa.legend(fontsize=font_size)
        axaoa.set_ylim([0, 20])
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Angle of Attack [deg]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "aoa" + extension
        faoa.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Airfoil efficiency
    try:
        feff, axeff = plt.subplots(1, 1, figsize=(5.3, 4))
        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            axeff.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["rotorse.rp.powercurve.cl_regII"] / yaml_data["rotorse.rp.powercurve.cd_regII"],
                "-",
                color=colors[idx],
                label=label,
            )
        if mult_flag:
            axeff.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Airfoil Efficiency [-]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "af_efficiency" + extension
        feff.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Prebend and sweep
    try:
        fbend, axbend = plt.subplots(1, 1, figsize=(5.3, 4))
        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            axbend.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["blade.outer_shape_bem.ref_axis"][:, 0],
                "-",
                color=colors[idx],
                label=label,
            )
        if mult_flag:
            axbend.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Prebend [m]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "prebend" + extension
        fbend.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")

        fsweep, axsweep = plt.subplots(1, 1, figsize=(5.3, 4))
        for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
            axsweep.plot(
                yaml_data["blade.outer_shape_bem.s"],
                yaml_data["blade.outer_shape_bem.ref_axis"][:, 1],
                "-",
                color=colors[idx],
                label=label,
            )
        if mult_flag:
            axsweep.legend(fontsize=font_size)
        plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
        plt.ylabel("Sweep [m]", fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = "sweep" + extension
        fsweep.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    except KeyError:
        pass

    # Tower geometry
    # try:
    brown = np.array([150.0, 75.0, 0.0]) / 256.0
    ftow = plt.figure(figsize=(11, 4))
    ax1 = ftow.add_subplot(121)
    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        ax1.plot(
            getter.get_tower_diameter(yaml_data),
            getter.get_zpts(yaml_data),
            "-",
            color=colors[idx],
            label=label,
        )
    vx = ax1.get_xlim()
    zs = getter.get_zpts(list_of_sims[0])
    if zs.min() < -5.0:
        water_depth = list_of_sims[0]["env.water_depth"]
        h_trans = getter.get_transition_height(list_of_sims[0])
        ax1.plot(vx, np.zeros(2), color="b", linestyle="--")
        ax1.plot(vx, -water_depth * np.ones(2), color=brown, linestyle="--")
        ax1.plot(vx, h_trans * np.ones(2), color="g", linestyle="--")
        ax1.text(vx[0] + 0.02 * np.diff(vx), 2, "Water line", color="b", fontsize=12)
        ax1.text(vx[0] + 0.02 * np.diff(vx), -water_depth + 2, "Mud line", color=brown, fontsize=12)
        ax1.text(vx[0] + 0.02 * np.diff(vx), h_trans + 2, "Tower transition", color="g", fontsize=12)
    ax1.set_xlim(vx)
    plt.xlabel("Outer Diameter [m]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Tower Height [m]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")

    ax2 = ftow.add_subplot(122)
    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        y = 1e3 * getter.get_tower_thickness(yaml_data)
        ax2.step(
            np.r_[y, y[-1]],
            getter.get_zpts(yaml_data),
            "-",
            color=colors[idx],
            label=label,
            where="post",
        )
    vx = ax2.get_xlim()
    if zs.min() < -5.0:
        ax2.plot(vx, np.zeros(2), color="b", linestyle="--")
        ax2.plot(vx, -water_depth * np.ones(2), color=brown, linestyle="--")
        ax2.plot(vx, 20 * np.ones(2), color="g", linestyle="--")
    ax2.set_xlim(vx)
    if mult_flag:
        ax2.legend(fontsize=font_size)
    plt.xlabel("Wall Thickness [mm]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    fig_name = "tower-monopile_geometry" + extension
    ftow.subplots_adjust(hspace=0.02, wspace=0.02, bottom=0.15, left=0.15)
    ftow.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
    # except KeyError:
    #    pass

    def simple_plot_results(x_axis_label, y_axis_label, x_axis_data_name, y_axis_data_name, plot_filename):
        f, ax = plt.subplots(1, 1, figsize=(5.3, 4))
        for i_yaml, yaml_data in enumerate(list_of_sims):
            ax.plot(
                yaml_data[x_axis_data_name],
                yaml_data[y_axis_data_name],
                "-",
                color=colors[i_yaml],
                label=list_of_labels[i_yaml],
            )
        if mult_flag:
            ax.legend(fontsize=font_size)
        plt.xlabel(x_axis_label, fontsize=font_size + 2, fontweight="bold")
        plt.ylabel(y_axis_label, fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = plot_filename + extension
        f.savefig(os.path.join(folder_output, fig_name), pad_inches=0.1, bbox_inches="tight")
        plt.close()

    try:
        # Edgewise stiffness
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Edgewise Stiffness [Nm2]",
            "blade.outer_shape_bem.s",
            "rotorse.EIxx",
            "edge",
        )

        # Torsional stiffness
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Torsional Stiffness [Nm2]",
            "blade.outer_shape_bem.s",
            "rotorse.GJ",
            "torsion",
        )

        # Flapwise stiffness
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Flapwise Stiffness [Nm2]",
            "blade.outer_shape_bem.s",
            "rotorse.EIyy",
            "flap",
        )

        # Mass
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Unit Mass [kg/m]",
            "blade.outer_shape_bem.s",
            "rotorse.rhoA",
            "mass",
        )
    except KeyError:
        pass

    try:
        # Relative thickness
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Relative Thickness [-]",
            "blade.outer_shape_bem.s",
            "blade.interp_airfoils.r_thick_interp",
            "r_thick",
        )
    except KeyError:
        pass

    try:
        # Induction
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Axial Induction [-]",
            "blade.outer_shape_bem.s",
            "rotorse.rp.powercurve.ax_induct_regII",
            "induction",
        )

        # Lift coefficient
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Lift Coefficient [-]",
            "blade.outer_shape_bem.s",
            "rotorse.rp.powercurve.cl_regII",
            "lift_coeff",
        )

        # Drag coefficient
        simple_plot_results(
            "Blade Nondimensional Span [-]",
            "Drag Coefficient [-]",
            "blade.outer_shape_bem.s",
            "rotorse.rp.powercurve.cd_regII",
            "drag_coeff",
        )

        # Power curve pitch
        simple_plot_results(
            "Wind velocity [m/s]",
            "Pitch angle [deg]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.pitch",
            "pitch",
        )

        # Power curve power
        simple_plot_results(
            "Wind velocity [m/s]",
            "Electrical Power [W]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.P",
            "power_elec",
        )

        # Power curve power (mechanical)
        simple_plot_results(
            "Wind velocity [m/s]",
            "Mechanical Power [W]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.P_aero",
            "power_aero",
        )

        # Power curve power coeff
        simple_plot_results(
            "Wind velocity [m/s]",
            "Electrical Power Coefficient [-]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.Cp",
            "cp_elec",
        )

        # Power curve power coeff (mechanical)
        simple_plot_results(
            "Wind velocity [m/s]",
            "Mechanical Power Coefficient [-]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.Cp_aero",
            "cp_aero",
        )

        # Power curve rpm
        simple_plot_results(
            "Wind velocity [m/s]",
            "Rotor speed [rpm]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.Omega",
            "omega",
        )

        # Power curve thrust
        simple_plot_results(
            "Wind velocity [m/s]",
            "Thrust [N]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.T",
            "thrust",
        )

        # Power curve thrust coeff
        simple_plot_results(
            "Wind velocity [m/s]",
            "Thrust Coefficient [-]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.Ct_aero",
            "ct_aero",
        )

        # Power curve torque
        simple_plot_results(
            "Wind velocity [m/s]",
            "Torque [Nm]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.Q",
            "torque",
        )

        # Power curve torque coeff
        simple_plot_results(
            "Wind velocity [m/s]",
            "Torque Coefficient [-]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.Cq_aero",
            "cq_aero",
        )

        # Power curve moment
        simple_plot_results(
            "Wind velocity [m/s]",
            "Blade moment [Nm]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.M",
            "moment",
        )

        # Power curve moment coeff
        simple_plot_results(
            "Wind velocity [m/s]",
            "Blade moment coefficient [-]",
            "rotorse.rp.powercurve.V",
            "rotorse.rp.powercurve.Cm_aero",
            "cm_aero",
        )
    except KeyError:
        pass

    if show_plots:
        plt.show()


def print_results_to_screen(list_of_sims, list_of_labels, values_to_print):
    list_of_augmented_labels = []
    max_label_length = 1
    for label in list_of_labels:
        list_of_augmented_labels.append(f"{label:15.15}")

    case_headers = "| Data name       | " + " | ".join(list_of_augmented_labels) + " | Units          |"
    # Header describing what we are printing:
    title_string = "Comparison between WISDEM results from yaml files"
    spacing = (len(case_headers) - len(title_string) - 2) // 2

    print("+" + "-" * (len(case_headers) - 2) + "+")
    print("|" + " " * spacing + title_string + " " * spacing + " |")
    print("+" + "-" * (len(case_headers) - 2) + "+")

    print(case_headers)
    print("+" + "-" * (len(case_headers) - 2) + "+")

    for key in values_to_print:
        value_name = values_to_print[key][0]
        units = values_to_print[key][1]
        units_str = f" | {units}" + (15 - len(str(units))) * " " + "|"

        try:
            value_sizer = list_of_sims[0].get_val(value_name, units)
        except KeyError:
            continue
        size_of_variable = len(value_sizer)

        for idx in range(size_of_variable):

            if size_of_variable > 1:
                augmented_key = f"{key}_{idx}"
            else:
                augmented_key = key

            name_str = f"| {augmented_key}" + (16 - len(augmented_key)) * " " + "| "

            list_of_values = []
            for yaml_data in list_of_sims:
                value = yaml_data.get_val(value_name, units).copy()

                if len(values_to_print[key]) > 2:
                    value *= values_to_print[key][2]

                list_of_values.append(f"{float(value[idx]):15.5f}")

            values_str = " | ".join(list_of_values)
            print(name_str + values_str + units_str)

    print("+" + "-" * (len(case_headers) - 2) + "+")
    print()


def save_lcoe_data_to_file(list_of_sims, folder_output):
    lcoe_data = np.zeros((8, len(list_of_sims)))
    try:
        for idx, yaml_data in enumerate(list_of_sims):
            lcoe_data[0, idx] = yaml_data["financese.turbine_number"]
            lcoe_data[1, idx] = yaml_data["financese.machine_rating"][0]
            lcoe_data[2, idx] = yaml_data["financese.tcc_per_kW"][0]
            lcoe_data[3, idx] = yaml_data["financese.bos_per_kW"][0]
            lcoe_data[4, idx] = yaml_data["financese.opex_per_kW"][0]
            lcoe_data[5, idx] = yaml_data["financese.turbine_aep"][0] * 1.0e-6
            lcoe_data[6, idx] = yaml_data["financese.fixed_charge_rate"][0] * 100.0
            lcoe_data[7, idx] = yaml_data["financese.lcoe"][0] * 1000.0

        np.savetxt(os.path.join(folder_output, "lcoe.dat"), lcoe_data)
    except:
        pass


def run(list_of_sims, list_of_labels, modeling_options, analysis_options):

    # These are options for the plotting and saving
    show_plots = False  # if True, print plots to screen in addition to saving files
    font_size = 12
    extension = ".png"  # '.pdf'
    folder_output = "outputs"

    # These are the values to print to screen for text-based output.
    # The dictionary keys are the value names.
    # The first string in the list is where that value exists in the WISDEM problem,
    # the second string is the units to print the value in,
    # and the optional third string is the multiplicative scalar on the value to be printed.
    values_to_print = {
        "Rotor Diameter": ["blade.high_level_blade_props.rotor_diameter", "m"],
        "TSR": ["control.rated_TSR", None],
        "AEP": ["rotorse.rp.AEP", "GW*h"],
        "LCOE": ["financese.lcoe", "USD/(MW*h)"],
        # "Cp": ["rotorse.rp.powercurve.Cp_aero", None],
        "Rated velocity": ["rotorse.rp.powercurve.rated_V", "m/s"],
        "Rated rpm": ["rotorse.rp.powercurve.rated_Omega", "rpm"],
        "Rated pitch": ["control.rated_pitch", "deg"],
        "Rated thrust": ["rotorse.rp.powercurve.rated_T", "kN"],
        "Rated torque": ["rotorse.rp.powercurve.rated_Q", "kN*m"],
        "Blade mass": ["rotorse.re.precomp.blade_mass", "kg"],
        "Blade cost": ["rotorse.re.precomp.total_blade_cost", "USD"],
        "Tip defl": ["tcons.tip_deflection", "m"],
        "Tip defl ratio": ["tcons.tip_deflection_ratio", None],
        "Flap freqs": ["rotorse.rs.frame.flap_mode_freqs", "Hz"],
        "Edge freqs": ["rotorse.rs.frame.edge_mode_freqs", "Hz"],
        "3P freq": ["rotorse.rp.powercurve.rated_Omega", None, 3.0 / 60],
        "6P freq": ["rotorse.rp.powercurve.rated_Omega", None, 6.0 / 60],
        "Hub forces": ["rotorse.rs.aero_hub_loads.Fxyz_hub_aero", "kN"],
        "Hub moments": ["rotorse.rs.aero_hub_loads.Mxyz_hub_aero", "kN*m"],
        "Nacelle mass": ["drivese.nacelle_mass", "kg"],
        "RNA mass": ["drivese.rna_mass", "kg"],
        "Tower mass": ["towerse.tower_mass", "kg"],
        "Floating Tower mass": ["floatingse.tower_mass", "kg"],
        "Tower cost": ["towerse.tower_cost", "USD"],
        "Floating Tower cost": ["floatingse.tower_cost", "USD"],
        "Monopile mass": ["towerse.monopile_mass", "kg"],
        "Monopile cost": ["towerse.monopile_cost", "USD"],
        "Tower-Monopile freqs": ["towerse.tower.structural_frequencies", "Hz"],
        "Floating Tower freqs": ["floatingse.tower_freqs", "Hz"],
    }

    # Generally it's not necessary to change the code below here, unless you
    # want to plot additional values

    # Create the output folder if it doesn't exist
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    # Call the functions to print, save, and plot results
    print_results_to_screen(list_of_sims, list_of_labels, values_to_print)
    save_lcoe_data_to_file(list_of_sims, folder_output)
    create_all_plots(
        list_of_sims,
        list_of_labels,
        modeling_options,
        analysis_options,
        folder_output,
        show_plots,
        font_size,
        extension,
    )


def main():
    # Called only if this script is run as main.

    # ======================================================================
    # Input Information
    # ======================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_files",
        nargs="*",
        type=str,
        help="Specify the yaml filenames and pickled output filenames to be compared.",
    )
    parser.add_argument(
        "--modeling_options",
        nargs="?",
        type=str,
        default=fname_modeling_options_default,
        help="Specify the modeling options yaml.",
    )
    parser.add_argument(
        "--analysis_options",
        nargs="?",
        type=str,
        default=fname_analysis_options_default,
        help="Specify the analysis options yaml.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        type=str,
        default=None,
        help="Specify the labels for the yaml files (use spaces for separation, no brackets).",
    )

    args = parser.parse_args()
    input_filenames = args.input_files
    fname_modeling_options = args.modeling_options
    fname_analysis_options = args.analysis_options
    list_of_labels = args.labels

    if len(input_filenames) == 0:
        print("ERROR: Must specify either a set of yaml files or pkl files\n")
        parser.print_help(sys.stderr)
        sys.exit(1)

    if list_of_labels is None:
        list_of_labels = [f"run_{idx}" for idx in range(len(input_filenames))]

    list_of_sims = []
    for input_filename in input_filenames:
        froot = os.path.splitext(input_filename)[0]

        print()
        if os.path.exists(froot + ".yaml") and os.path.exists(froot + ".pkl"):
            # Load in saved pickle dictionary if already ran WISDEM
            print(f"Loading WISDEM data for {input_filename}.")
            wt_opt, modeling_options, analysis_options = load_wisdem(froot)

        else:
            # Run WISDEM for each yaml file to compare using the modeling and analysis options set above
            print(f"Running WISDEM for {input_filename}.")
            wt_opt, modeling_options, analysis_options = run_wisdem(
                input_filename,
                fname_modeling_options,
                fname_analysis_options,
                run_only=True,
            )

        list_of_sims.append(wt_opt)

    run(list_of_sims, list_of_labels, modeling_options, analysis_options)
    sys.exit(0)


if __name__ == "__main__":
    main()
