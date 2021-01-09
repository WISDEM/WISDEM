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
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Twist
    ftw, axtw = plt.subplots(1, 1, figsize=(5.3, 4))

    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        axtw.plot(
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["blade.outer_shape_bem.twist"] * 180.0 / np.pi,
            "-",
            color=colors[idx],
            label=label,
        )
        s_opt_twist = np.linspace(0.0, 1.0, 8)
        twist_opt = np.interp(
            s_opt_twist,
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["ccblade.theta"],
        )
        axtw.plot(s_opt_twist, twist_opt * 180.0 / np.pi, "o", color=colors[idx], markersize=3)

    axtw.plot(
        s_opt_twist,
        np.array(analysis_options["design_variables"]["blade"]["aero_shape"]["twist"]["lower_bound"]) * 180.0 / np.pi,
        ":o",
        color=colors[idx + 1],
        markersize=3,
        label="Bounds",
    )
    axtw.plot(
        s_opt_twist,
        np.array(analysis_options["design_variables"]["blade"]["aero_shape"]["twist"]["upper_bound"]) * 180.0 / np.pi,
        ":o",
        color=colors[idx + 1],
        markersize=3,
    )
    axtw.legend(fontsize=font_size)

    axtw.set_ylim([-5, 20])
    plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Twist [deg]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    fig_name = "twist_opt" + extension
    ftw.savefig(os.path.join(folder_output, fig_name))

    # Chord
    fc, axc = plt.subplots(1, 1, figsize=(5.3, 4))

    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        axc.plot(
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["blade.outer_shape_bem.chord"],
            "-",
            color=colors[idx],
            label=label,
        )
        s_opt_chord = np.linspace(0.0, 1.0, 8)
        chord_opt = np.interp(
            s_opt_chord,
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["ccblade.chord"],
        )
        axc.plot(s_opt_chord, chord_opt, "o", color=colors[idx], markersize=3)

    chord_init = np.interp(
        s_opt_chord,
        list_of_sims[0]["blade.outer_shape_bem.s"],
        yaml_data["ccblade.chord"],
    )
    axc.plot(
        s_opt_chord,
        np.array(analysis_options["design_variables"]["blade"]["aero_shape"]["chord"]["min_gain"]) * chord_init,
        ":o",
        color=colors[idx + 1],
        markersize=3,
        label="Bounds",
    )
    axc.plot(
        s_opt_chord,
        np.array(analysis_options["design_variables"]["blade"]["aero_shape"]["chord"]["max_gain"]) * chord_init,
        ":o",
        color=colors[idx + 1],
        markersize=3,
    )

    axc.legend(fontsize=font_size)
    plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Chord [m]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    fig_name = "chord" + extension
    fc.savefig(os.path.join(folder_output, fig_name))

    # Spar caps
    fsc, axsc = plt.subplots(1, 1, figsize=(5.3, 4))

    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        n_layers = len(yaml_data["blade.internal_structure_2d_fem.layer_thickness"][:, 0])
        for i in range(n_layers):
            layer_name = modeling_options["WISDEM"]["RotorSE"]["layer_name"][i]
            if modeling_options["WISDEM"]["RotorSE"]["spar_cap_ss"] == layer_name:
                axsc.plot(
                    yaml_data["blade.outer_shape_bem.s"],
                    yaml_data["blade.internal_structure_2d_fem.layer_thickness"][i, :] * 1.0e3,
                    "-",
                    color=colors[idx],
                    label=label,
                )

                s_opt_sc = np.linspace(0.0, 1.0, 8)
                sc_opt = np.interp(
                    s_opt_sc,
                    yaml_data["blade.outer_shape_bem.s"],
                    yaml_data["blade.internal_structure_2d_fem.layer_thickness"][i, :] * 1.0e3,
                )
                axsc.plot(s_opt_sc, sc_opt, "o", color=colors[idx], markersize=3)

    for i in range(n_layers):
        layer_name = modeling_options["WISDEM"]["RotorSE"]["layer_name"][i]
        if modeling_options["WISDEM"]["RotorSE"]["spar_cap_ss"] == layer_name:
            sc_init = np.interp(
                s_opt_sc,
                list_of_sims[0]["blade.outer_shape_bem.s"],
                list_of_sims[0]["blade.internal_structure_2d_fem.layer_thickness"][i, :] * 1.0e3,
            )
            axsc.plot(
                s_opt_sc,
                np.array(analysis_options["design_variables"]["blade"]["structure"]["spar_cap_ss"]["min_gain"])
                * sc_init,
                ":o",
                color=colors[idx + 1],
                markersize=3,
                label="Bounds",
            )
            axsc.plot(
                s_opt_sc,
                np.array(analysis_options["design_variables"]["blade"]["structure"]["spar_cap_ss"]["max_gain"])
                * sc_init,
                ":o",
                color=colors[idx + 1],
                markersize=3,
            )

    axsc.legend(fontsize=font_size)
    plt.ylim([0.0, 200])
    plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Spar Caps Thickness [mm]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    fig_name = "sc_opt" + extension
    fsc.savefig(os.path.join(folder_output, fig_name))

    # Skins
    f, ax = plt.subplots(1, 1, figsize=(5.3, 4))
    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        ax.plot(
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["blade.internal_structure_2d_fem.layer_thickness"][1, :] * 1.0e3,
            "--",
            color=colors[idx],
            label=label,
        )
    ax.legend(fontsize=font_size)
    # plt.ylim([0., 120])
    plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Outer Shell Skin Thickness [mm]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    fig_name = "skin_opt" + extension
    f.savefig(os.path.join(folder_output, fig_name))

    # Strains spar caps
    feps, axeps = plt.subplots(1, 1, figsize=(5.3, 4))
    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        axeps.plot(
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["rs.frame.strainU_spar"] * 1.0e6,
            "--",
            color=colors[idx],
            label=label,
        )
        axeps.plot(
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["rs.frame.strainL_spar"] * 1.0e6,
            "--",
            color=colors[idx],
        )

    plt.ylim([-5e3, 5e3])
    axeps.legend(fontsize=font_size)
    plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Spar Caps Strains [mu eps]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.2)
    fig_name = "strains_opt" + extension
    feps.savefig(os.path.join(folder_output, fig_name))

    # Angle of attack and stall angle
    faoa, axaoa = plt.subplots(1, 1, figsize=(5.3, 4))
    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        axaoa.plot(
            yaml_data["stall_check.s"],
            yaml_data["stall_check.aoa_along_span"],
            "--",
            color=colors[idx],
            label=label,
        )
    axaoa.plot(
        yaml_data["stall_check.s"],
        yaml_data["stall_check.stall_angle_along_span"],
        ":",
        color=colors[idx + 1],
        label="Stall",
    )
    axaoa.legend(fontsize=font_size)
    axaoa.set_ylim([0, 20])
    plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Angle of Attack [deg]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    fig_name = "aoa" + extension
    faoa.savefig(os.path.join(folder_output, fig_name))

    # Airfoil efficiency
    feff, axeff = plt.subplots(1, 1, figsize=(5.3, 4))
    for idx, (yaml_data, label) in enumerate(zip(list_of_sims, list_of_labels)):
        axeff.plot(
            yaml_data["blade.outer_shape_bem.s"],
            yaml_data["rp.powercurve.cl_regII"] / yaml_data["rp.powercurve.cd_regII"],
            "--",
            color=colors[idx],
            label=label,
        )
    axeff.legend(fontsize=font_size)
    plt.xlabel("Blade Nondimensional Span [-]", fontsize=font_size + 2, fontweight="bold")
    plt.ylabel("Airfoil Efficiency [-]", fontsize=font_size + 2, fontweight="bold")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    fig_name = "af_efficiency" + extension
    feff.savefig(os.path.join(folder_output, fig_name))

    def simple_plot_results(x_axis_label, y_axis_label, x_axis_data_name, y_axis_data_name, plot_filename):
        f, ax = plt.subplots(1, 1, figsize=(5.3, 4))
        for i_yaml, yaml_data in enumerate(list_of_sims):
            ax.plot(
                yaml_data[x_axis_data_name],
                yaml_data[y_axis_data_name],
                "--",
                color=colors[i_yaml],
                label=list_of_labels[i_yaml],
            )
        ax.legend(fontsize=font_size)
        plt.xlabel(x_axis_label, fontsize=font_size + 2, fontweight="bold")
        plt.ylabel(y_axis_label, fontsize=font_size + 2, fontweight="bold")
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.grid(color=[0.8, 0.8, 0.8], linestyle="--")
        plt.subplots_adjust(bottom=0.15, left=0.15)
        fig_name = plot_filename + extension
        f.savefig(os.path.join(folder_output, fig_name))

    # Edgewise stiffness
    simple_plot_results(
        "Blade Nondimensional Span [-]",
        "Edgewise Stiffness [Nm2]",
        "blade.outer_shape_bem.s",
        "re.EIxx",
        "edge",
    )

    # Torsional stiffness
    simple_plot_results(
        "Blade Nondimensional Span [-]",
        "Torsional Stiffness [Nm2]",
        "blade.outer_shape_bem.s",
        "re.GJ",
        "torsion",
    )

    # Flapwise stiffness
    simple_plot_results(
        "Blade Nondimensional Span [-]",
        "Flapwise Stiffness [Nm2]",
        "blade.outer_shape_bem.s",
        "re.EIyy",
        "flap",
    )

    # Mass
    simple_plot_results(
        "Blade Nondimensional Span [-]",
        "Unit Mass [kg/m]",
        "blade.outer_shape_bem.s",
        "re.rhoA",
        "mass",
    )

    # Induction
    simple_plot_results(
        "Blade Nondimensional Span [-]",
        "Axial Induction [-]",
        "blade.outer_shape_bem.s",
        "rp.powercurve.ax_induct_regII",
        "induction",
    )

    # Lift coefficient
    simple_plot_results(
        "Blade Nondimensional Span [-]",
        "Lift Coefficient [-]",
        "blade.outer_shape_bem.s",
        "rp.powercurve.cl_regII",
        "lift_coeff",
    )

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

        value_sizer = list_of_sims[0].get_val(value_name, units)
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
        "AEP": ["rp.AEP", "GW*h"],
        "Blade mass": ["re.precomp.blade_mass", "kg"],
        "LCOE": ["financese.lcoe", "USD/(MW*h)"],
        "Cp": ["rp.powercurve.Cp_aero", None],
        "Blade cost": ["re.precomp.total_blade_cost", "USD"],
        "Tip defl ratio": ["tcons.tip_deflection_ratio", None],
        "Flap freqs": ["rs.frame.flap_mode_freqs", "Hz"],
        "Edge freqs": ["rs.frame.edge_mode_freqs", "Hz"],
        "3P freq": ["rp.powercurve.rated_Omega", None, 3.0 / 60],
        "6P freq": ["rp.powercurve.rated_Omega", None, 6.0 / 60],
        "Hub forces": ["rs.aero_hub_loads.Fxyz_hub_aero", "kN"],
        "Hub moments": ["rs.aero_hub_loads.Mxyz_hub_aero", "kN*m"],
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
    parser.add_argument("--labels", nargs="*", type=str, default=None, help="Specify the labels for the yaml files.")

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
                input_filename, fname_modeling_options, fname_analysis_options
            )

        list_of_sims.append(wt_opt)

    run(list_of_sims, list_of_labels, modeling_options, analysis_options)
    sys.exit(0)


if __name__ == "__main__":
    main()
