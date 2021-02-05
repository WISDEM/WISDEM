import os

import numpy as np
import openmdao.api as om


class PoseOptimization(object):
    def __init__(self, modeling_options, analysis_options):
        self.modeling = modeling_options
        self.opt = analysis_options

    def get_number_design_variables(self):
        # Determine the number of design variables
        n_DV = 0

        rotorD_opt = self.opt["design_variables"]["rotor_diameter"]
        blade_opt = self.opt["design_variables"]["blade"]
        tower_opt = self.opt["design_variables"]["tower"]
        mono_opt = self.opt["design_variables"]["monopile"]
        hub_opt = self.opt["design_variables"]["hub"]
        drive_opt = self.opt["design_variables"]["drivetrain"]
        float_opt = self.opt["design_variables"]["floating"]

        if rotorD_opt["flag"]:
            n_DV += 1
        if blade_opt["aero_shape"]["twist"]["flag"]:
            n_DV += blade_opt["aero_shape"]["twist"]["n_opt"] - (
                blade_opt["aero_shape"]["twist"]["lock_root"] + blade_opt["aero_shape"]["twist"]["lock_tip"]
            )
        if blade_opt["aero_shape"]["chord"]["flag"]:
            n_DV += blade_opt["aero_shape"]["chord"]["n_opt"] - (
                blade_opt["aero_shape"]["chord"]["lock_root"] + blade_opt["aero_shape"]["chord"]["lock_tip"]
            )
        if blade_opt["aero_shape"]["af_positions"]["flag"]:
            n_DV += (
                self.modeling["WISDEM"]["RotorSE"]["n_af_span"]
                - blade_opt["aero_shape"]["af_positions"]["af_start"]
                - 1
            )
        if blade_opt["structure"]["spar_cap_ss"]["flag"]:
            n_DV += blade_opt["structure"]["spar_cap_ss"]["n_opt"] - (
                blade_opt["structure"]["spar_cap_ss"]["lock_root"] + blade_opt["structure"]["spar_cap_ss"]["lock_tip"]
            )
        if (
            blade_opt["structure"]["spar_cap_ps"]["flag"]
            and not blade_opt["structure"]["spar_cap_ps"]["equal_to_suction"]
        ):
            n_DV += blade_opt["structure"]["spar_cap_ps"]["n_opt"] - (
                blade_opt["structure"]["spar_cap_ps"]["lock_root"] + blade_opt["structure"]["spar_cap_ps"]["lock_tip"]
            )
        if self.opt["design_variables"]["control"]["tsr"]["flag"]:
            n_DV += 1

        if tower_opt["outer_diameter"]["flag"]:
            n_DV += self.modeling["WISDEM"]["TowerSE"]["n_height_tower"]
        if tower_opt["layer_thickness"]["flag"]:
            n_DV += (
                self.modeling["WISDEM"]["TowerSE"]["n_height_tower"]
                * self.modeling["WISDEM"]["TowerSE"]["n_layers_tower"]
            )
        if mono_opt["outer_diameter"]["flag"]:
            n_DV += self.modeling["WISDEM"]["TowerSE"]["n_height_monopile"]
        if mono_opt["layer_thickness"]["flag"]:
            n_DV += (
                self.modeling["WISDEM"]["TowerSE"]["n_height_monopile"]
                * self.modeling["WISDEM"]["TowerSE"]["n_layers_monopile"]
            )
        if hub_opt["cone"]["flag"]:
            n_DV += 1
        if hub_opt["hub_diameter"]["flag"]:
            n_DV += 1
        for k in [
            "uptilt",
            "overhang",
            "distance_tt_hub",
            "distance_hub_mb",
            "distance_mb_mb",
            "generator_length",
            "gear_ratio",
            "generator_length",
            "bedplate_web_thickness",
            "bedplate_flange_thickness",
            "bedplate_flange_width",
        ]:
            if drive_opt[k]["flag"]:
                n_DV += 1
        for k in [
            "lss_diameter",
            "lss_wall_thickness",
            "hss_diameter",
            "hss_wall_thickness",
            "nose_diameter",
            "nose_wall_thickness",
        ]:
            if drive_opt[k]["flag"]:
                n_DV += 2
        if drive_opt["bedplate_wall_thickness"]["flag"]:
            n_DV += 4

        if float_opt["joints"]["flag"]:
            n_DV += len(float_opt["joints"]["z_coordinate"]) + len(float_opt["joints"]["r_coordinate"])

        if float_opt["members"]["flag"]:
            for k, kgrp in enumerate(float_opt["members"]["groups"]):
                memname = kgrp["names"][0]
                memidx = self.modeling["floating"]["members"]["name"].index(memname)
                n_grid = len(self.modeling["floating"]["members"]["grid_member_" + memname])
                n_layers = self.modeling["floating"]["members"]["n_layers"][memidx]
                if "diameter" in kgrp:
                    n_DV += n_grid
                if "thickness" in kgrp:
                    n_DV += n_grid * n_layers
                if "ballast" in kgrp:
                    n_DV += self.modeling["floating"]["members"]["ballast_flag_member_" + memname].count(False)
                if "stiffeners" in kgrp:
                    if "ring" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["ring"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["ring"]:
                            n_DV += 1
                    if "longitudinal" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["longitudinal"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["longitudinal"]:
                            n_DV += 1
                if "axial_joints" in kgrp:
                    n_DV += len(kgrp["axial_joints"])

        # Wrap-up at end with multiplier for finite differencing
        if self.opt["driver"]["optimization"]["form"] == "central":
            n_DV *= 2

        return n_DV

    def _get_step_size(self):
        # If a step size for the driver-level finite differencing is provided, use that step size. Otherwise use a default value.
        return (
            1.0e-6
            if not "step_size" in self.opt["driver"]["optimization"]
            else self.opt["driver"]["optimization"]["step_size"]
        )

    def set_driver(self, wt_opt):
        folder_output = self.opt["general"]["folder_output"]

        if self.opt["driver"]["optimization"]["flag"]:
            step_size = self._get_step_size()

            # Solver has specific meaning in OpenMDAO
            wt_opt.model.approx_totals(method="fd", step=step_size, form=self.opt["driver"]["optimization"]["form"])

            # Set optimization solver and options. First, Scipy's SLSQP
            if self.opt["driver"]["optimization"]["solver"] == "SLSQP":
                wt_opt.driver = om.ScipyOptimizeDriver()
                wt_opt.driver.options["optimizer"] = self.opt["driver"]["optimization"]["solver"]
                wt_opt.driver.options["tol"] = self.opt["driver"]["optimization"]["tol"]
                wt_opt.driver.options["maxiter"] = self.opt["driver"]["optimization"]["max_iter"]

            # The next two optimization methods require pyOptSparse.
            elif self.opt["driver"]["optimization"]["solver"] == "CONMIN":
                try:
                    from openmdao.api import pyOptSparseDriver
                except:
                    raise ImportError(
                        "You requested the optimization solver CONMIN, but you have not installed the pyOptSparseDriver. Please do so and rerun."
                    )
                wt_opt.driver = pyOptSparseDriver()
                wt_opt.driver.options["optimizer"] = self.opt["driver"]["optimization"]["solver"]
                wt_opt.driver.opt_settings["ITMAX"] = self.opt["driver"]["optimization"]["max_iter"]

            elif self.opt["driver"]["optimization"]["solver"] == "SNOPT":
                try:
                    from openmdao.api import pyOptSparseDriver
                except:
                    raise ImportError(
                        "You requested the optimization solver SNOPT, but you have not installed the pyOptSparseDriver. Please do so and rerun."
                    )
                wt_opt.driver = pyOptSparseDriver()
                try:
                    wt_opt.driver.options["optimizer"] = self.opt["driver"]["optimization"]["solver"]
                except:
                    raise ImportError(
                        "You requested the optimization solver SNOPT, but you have not installed it within the pyOptSparseDriver. Please do so and rerun."
                    )
                wt_opt.driver.opt_settings["Major optimality tolerance"] = float(
                    self.opt["driver"]["optimization"]["tol"]
                )
                wt_opt.driver.opt_settings["Major iterations limit"] = int(
                    self.opt["driver"]["optimization"]["max_major_iter"]
                )
                wt_opt.driver.opt_settings["Iterations limit"] = int(
                    self.opt["driver"]["optimization"]["max_minor_iter"]
                )
                wt_opt.driver.opt_settings["Major feasibility tolerance"] = float(
                    self.opt["driver"]["optimization"]["tol"]
                )
                wt_opt.driver.opt_settings["Summary file"] = os.path.join(folder_output, "SNOPT_Summary_file.txt")
                wt_opt.driver.opt_settings["Print file"] = os.path.join(folder_output, "SNOPT_Print_file.txt")
                if "hist_file_name" in self.opt["driver"]["optimization"]:
                    wt_opt.driver.hist_file = self.opt["driver"]["optimization"]["hist_file_name"]
                if "verify_level" in self.opt["driver"]["optimization"]:
                    wt_opt.driver.opt_settings["Verify level"] = self.opt["driver"]["optimization"]["verify_level"]
                else:
                    wt_opt.driver.opt_settings["Verify level"] = -1
                # wt_opt.driver.declare_coloring()
                if "hotstart_file" in self.opt["driver"]["optimization"]:
                    wt_opt.driver.hotstart_file = self.opt["driver"]["optimization"]["hotstart_file"]

            else:
                raise ValueError(
                    "The optimizer " + self.opt["driver"]["optimization"]["solver"] + "is not yet supported!"
                )

        elif self.opt["driver"]["design_of_experiments"]["flag"]:
            if self.opt["driver"]["design_of_experiments"]["generator"].lower() == "uniform":
                generator = om.UniformGenerator(
                    num_samples=self.opt["driver"]["design_of_experiments"]["num_samples"],
                    seed=self.opt["driver"]["design_of_experiments"]["seed"],
                )
            elif self.opt["driver"]["design_of_experiments"]["generator"].lower() == "fullfact":
                generator = om.FullFactorialGenerator(levels=self.opt["driver"]["design_of_experiments"]["num_samples"])
            elif self.opt["driver"]["design_of_experiments"]["generator"].lower() == "plackettburman":
                generator = om.PlackettBurmanGenerator()
            elif self.opt["driver"]["design_of_experiments"]["generator"].lower() == "boxbehnken":
                generator = om.BoxBehnkenGenerator()
            elif self.opt["driver"]["design_of_experiments"]["generator"].lower() == "latinhypercube":
                generator = om.LatinHypercubeGenerator(
                    samples=self.opt["driver"]["design_of_experiments"]["num_samples"],
                    criterion=self.opt["driver"]["design_of_experiments"]["criterion"],
                    seed=self.opt["driver"]["design_of_experiments"]["seed"],
                )
            else:
                raise Exception(
                    "The generator type {} is unsupported.".format(
                        self.opt["driver"]["design_of_experiments"]["generator"]
                    )
                )

            # Initialize driver
            wt_opt.driver = om.DOEDriver(generator)

            # options
            wt_opt.driver.options["run_parallel"] = self.opt["driver"]["design_of_experiments"]["run_parallel"]

        else:
            raise Exception(
                "Design variables are set to be optimized or studied, but no driver is selected. Please enable a driver."
            )

        return wt_opt

    def set_objective(self, wt_opt):

        # Set merit figure. Each objective has its own scaling.
        if self.opt["merit_figure"] == "AEP":
            wt_opt.model.add_objective("rp.AEP", ref=-1.0e6)

        elif self.opt["merit_figure"] == "blade_mass":
            wt_opt.model.add_objective("re.precomp.blade_mass", ref=1.0e4)

        elif self.opt["merit_figure"] == "LCOE":
            wt_opt.model.add_objective("financese.lcoe", ref=0.1)

        elif self.opt["merit_figure"] == "blade_tip_deflection":
            wt_opt.model.add_objective("tcons.tip_deflection_ratio")

        elif self.opt["merit_figure"] == "tower_mass":
            wt_opt.model.add_objective("towerse.tower_mass", ref=1e6)

        elif self.opt["merit_figure"] == "mononpile_mass":
            wt_opt.model.add_objective("towerse.mononpile_mass", ref=1e6)

        elif self.opt["merit_figure"] == "structural_mass":
            wt_opt.model.add_objective("towerse.structural_mass", ref=1e6)

        elif self.opt["merit_figure"] == "tower_cost":
            wt_opt.model.add_objective("tcc.tower_cost", ref=1e6)

        elif self.opt["merit_figure"] == "hub_mass":
            wt_opt.model.add_objective("drivese.hub_system_mass", ref=1e5)

        elif self.opt["merit_figure"] == "nacelle_mass":
            wt_opt.model.add_objective("drivese.nacelle_mass", ref=1e6)

        elif self.opt["merit_figure"] == "nacelle_cost":
            wt_opt.model.add_objective("tcc.nacelle_cost", ref=1e6)

        elif self.opt["merit_figure"] == "platform_mass":
            wt_opt.model.add_objective("floatingse.platform_mass", ref=1e6)

        elif self.opt["merit_figure"] == "platform_cost":
            wt_opt.model.add_objective("floatingse.platform_cost", ref=1e6)

        elif self.opt["merit_figure"] == "Cp":
            if self.modeling["flags"]["blade"]:
                wt_opt.model.add_objective("rp.powercurve.Cp_regII", ref=-1.0)
            else:
                wt_opt.model.add_objective("ccblade.CP", ref=-1.0)
        else:
            raise ValueError("The merit figure " + self.opt["merit_figure"] + " is unknown or not supported.")

        return wt_opt

    def set_design_variables(self, wt_opt, wt_init):

        # Set optimization design variables.
        rotorD_opt = self.opt["design_variables"]["rotor_diameter"]
        blade_opt = self.opt["design_variables"]["blade"]
        tower_opt = self.opt["design_variables"]["tower"]
        monopile_opt = self.opt["design_variables"]["monopile"]
        control_opt = self.opt["design_variables"]["control"]
        hub_opt = self.opt["design_variables"]["hub"]
        drive_opt = self.opt["design_variables"]["drivetrain"]
        float_opt = self.opt["design_variables"]["floating"]

        # -- Rotor & Blade --
        if rotorD_opt["flag"]:
            wt_opt.model.add_design_var(
                "configuration.rotor_diameter_user", lower=rotorD_opt["minimum"], upper=rotorD_opt["minimum"], ref=1.0e2
            )

        twist_options = blade_opt["aero_shape"]["twist"]
        if twist_options["flag"]:
            indices = range(twist_options["lock_root"], twist_options["n_opt"] - twist_options["lock_tip"])
            wt_opt.model.add_design_var("blade.opt_var.twist_opt_gain", indices=indices, lower=0.0, upper=1.0)

        chord_options = blade_opt["aero_shape"]["chord"]
        if chord_options["flag"]:
            indices = range(chord_options["lock_root"], chord_options["n_opt"] - chord_options["lock_tip"])
            wt_opt.model.add_design_var(
                "blade.opt_var.chord_opt_gain",
                indices=indices,
                lower=chord_options["min_gain"],
                upper=chord_options["max_gain"],
            )

        if blade_opt["aero_shape"]["af_positions"]["flag"]:
            n_af = self.modeling["WISDEM"]["RotorSE"]["n_af_span"]
            indices = range(blade_opt["aero_shape"]["af_positions"]["af_start"], n_af - 1)
            af_pos_init = wt_init["components"]["blade"]["outer_shape_bem"]["airfoil_position"]["grid"]
            step_size = self._get_step_size()
            lb_af = np.zeros(n_af)
            ub_af = np.zeros(n_af)
            for i in range(1, indices[0]):
                lb_af[i] = ub_af[i] = af_pos_init[i]
            for i in indices:
                lb_af[i] = 0.5 * (af_pos_init[i - 1] + af_pos_init[i]) + step_size
                ub_af[i] = 0.5 * (af_pos_init[i + 1] + af_pos_init[i]) - step_size
            lb_af[-1] = ub_af[-1] = 1.0
            wt_opt.model.add_design_var(
                "blade.opt_var.af_position", indices=indices, lower=lb_af[indices], upper=ub_af[indices]
            )

        spar_cap_ss_options = blade_opt["structure"]["spar_cap_ss"]
        if spar_cap_ss_options["flag"]:
            indices = range(
                spar_cap_ss_options["lock_root"], spar_cap_ss_options["n_opt"] - spar_cap_ss_options["lock_tip"]
            )
            wt_opt.model.add_design_var(
                "blade.opt_var.spar_cap_ss_opt_gain",
                indices=indices,
                lower=spar_cap_ss_options["min_gain"],
                upper=spar_cap_ss_options["max_gain"],
            )

        # Only add the pressure side design variables if we do set
        # `equal_to_suction` as False in the optimization yaml.
        spar_cap_ps_options = blade_opt["structure"]["spar_cap_ps"]
        if spar_cap_ps_options["flag"] and not spar_cap_ps_options["equal_to_suction"]:
            indices = range(
                spar_cap_ps_options["lock_root"], spar_cap_ps_options["n_opt"] - spar_cap_ps_options["lock_tip"]
            )
            wt_opt.model.add_design_var(
                "blade.opt_var.spar_cap_ps_opt_gain",
                indices=indices,
                lower=spar_cap_ps_options["min_gain"],
                upper=spar_cap_ps_options["max_gain"],
            )

        # -- Tower & Monopile --
        if tower_opt["outer_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "tower.diameter",
                lower=tower_opt["outer_diameter"]["lower_bound"],
                upper=tower_opt["outer_diameter"]["upper_bound"],
                ref=5.0,
            )

        if tower_opt["layer_thickness"]["flag"]:
            wt_opt.model.add_design_var(
                "tower.layer_thickness",
                lower=tower_opt["layer_thickness"]["lower_bound"],
                upper=tower_opt["layer_thickness"]["upper_bound"],
                ref=1e-2,
            )

        if monopile_opt["outer_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "monopile.diameter",
                lower=monopile_opt["outer_diameter"]["lower_bound"],
                upper=monopile_opt["outer_diameter"]["upper_bound"],
                ref=5.0,
            )

        if monopile_opt["layer_thickness"]["flag"]:
            wt_opt.model.add_design_var(
                "monopile.layer_thickness",
                lower=monopile_opt["layer_thickness"]["lower_bound"],
                upper=monopile_opt["layer_thickness"]["upper_bound"],
                ref=1e-2,
            )

        # -- Control --
        if control_opt["tsr"]["flag"]:
            wt_opt.model.add_design_var(
                "control.rated_TSR", lower=control_opt["tsr"]["minimum"], upper=control_opt["tsr"]["maximum"], ref=1e1
            )
        # -- Hub & Drivetrain --
        if hub_opt["cone"]["flag"]:
            wt_opt.model.add_design_var(
                "hub.cone", lower=hub_opt["cone"]["lower_bound"], upper=hub_opt["cone"]["upper_bound"], ref=1e-2
            )
        if hub_opt["hub_diameter"]["flag"]:
            wt_opt.model.add_design_var(
                "hub.diameter",
                lower=hub_opt["hub_diameter"]["lower_bound"],
                upper=hub_opt["hub_diameter"]["upper_bound"],
            )
        if drive_opt["uptilt"]["flag"]:
            wt_opt.model.add_design_var(
                "nacelle.uptilt",
                lower=drive_opt["uptilt"]["lower_bound"],
                upper=drive_opt["uptilt"]["upper_bound"],
                ref=1e-2,
            )

        if drive_opt["generator_length"]["flag"]:
            wt_opt.model.add_design_var(
                "nacelle.L_generator",
                lower=drive_opt["generator_length"]["lower_bound"],
                upper=drive_opt["generator_length"]["upper_bound"],
            )

        for k in [
            "overhang",
            "distance_tt_hub",
            "distance_hub_mb",
            "distance_mb_mb",
            "gear_ratio",
            "bedplate_flange_width",
            "lss_diameter",
            "hss_diameter",
            "nose_diameter",
        ]:
            if drive_opt[k]["flag"]:
                wt_opt.model.add_design_var(
                    "nacelle." + k, lower=drive_opt[k]["lower_bound"], upper=drive_opt[k]["upper_bound"]
                )

        for k in [
            "bedplate_wall_thickness",
            "bedplate_web_thickness",
            "bedplate_flange_thickness",
            "lss_wall_thickness",
            "hss_wall_thickness",
            "nose_wall_thickness",
        ]:
            if drive_opt[k]["flag"]:
                wt_opt.model.add_design_var(
                    "nacelle." + k, lower=drive_opt[k]["lower_bound"], upper=drive_opt[k]["upper_bound"], ref=1e-2
                )

        # -- Floating & Mooring --
        if float_opt["joints"]["flag"]:
            jointz = float_opt["joints"]["z_coordinate"]
            jointr = float_opt["joints"]["r_coordinate"]

            count = 0
            for k in range(len(jointz)):
                wt_opt.model.add_design_var(
                    f"floating.jointdv_{count}",
                    lower=jointz[k]["lower_bound"],
                    upper=jointz[k]["upper_bound"],
                )
                count += 1

            for k in range(len(jointr)):
                wt_opt.model.add_design_var(
                    f"floating.jointdv_{count}",
                    lower=jointr[k]["lower_bound"],
                    upper=jointr[k]["upper_bound"],
                )
                count += 1

        if float_opt["members"]["flag"]:
            for kgrp in float_opt["members"]["groups"]:
                memname = kgrp["names"][0]
                idx = self.modeling["floating"]["members"]["name2idx"][memname]
                imem = self.modeling["floating"]["members"]["name"].index(memname)
                istruct = wt_init["components"]["floating_platform"]["members"][imem]["internal_structure"]

                if "diameter" in kgrp:
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.outer_diameter",
                        lower=kgrp["diameter"]["lower_bound"],
                        upper=kgrp["diameter"]["upper_bound"],
                    )
                if "thickness" in kgrp:
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.layer_thickness",
                        lower=kgrp["thickness"]["lower_bound"],
                        upper=kgrp["thickness"]["upper_bound"],
                    )
                if "ballast" in kgrp and len(istruct["ballasts"]) > 0:
                    V_ballast = np.zeros(len(istruct["ballasts"]))
                    for j in range(V_ballast.size):
                        if "volume" in istruct["ballasts"][j]:
                            V_ballast[j] = istruct["ballasts"][j]["volume"]
                    iball = np.where(V_ballast > 0.0)[0]
                    if iball.size > 0:
                        wt_opt.model.add_design_var(
                            f"floating.memgrp{idx}.ballast_volume",
                            lower=kgrp["ballast"]["lower_bound"],
                            upper=kgrp["ballast"]["upper_bound"],
                            indices=iball,
                            ref=1e3,
                        )
                if "stiffeners" in kgrp:
                    if "ring" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["ring"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["ring"]:
                            wt_opt.model.add_design_var(
                                f"floating.memgrp{idx}.ring_stiffener_spacing",
                                lower=kgrp["stiffeners"]["ring"]["spacing"]["lower_bound"],
                                upper=kgrp["stiffeners"]["ring"]["spacing"]["upper_bound"],
                            )
                    if "longitudinal" in kgrp["stiffeners"]:
                        if "size" in kgrp["stiffeners"]["longitudinal"]:
                            pass
                        if "spacing" in kgrp["stiffeners"]["longitudinal"]:
                            wt_opt.model.add_design_var(
                                f"floating.memgrp{idx}.axial_stiffener_spacing",
                                lower=kgrp["stiffeners"]["longitudinal"]["spacing"]["lower_bound"],
                                upper=kgrp["stiffeners"]["longitudinal"]["spacing"]["upper_bound"],
                            )
                if "axial_joints" in kgrp:
                    aidx = []
                    lower = []
                    upper = []
                    mem_axial_names = self.modeling["floating"]["members"]["axial_joint_name_member_" + memname]
                    for agrp in kgrp["axial_joints"]:
                        tryidx = None
                        agrp_names = agrp["names"]
                        lower.append(agrp["lower_bound"])
                        upper.append(agrp["upper_bound"])
                        for iname in agrp_names:
                            try:
                                tryidx = mem_axial_names.index(iname)
                                aidx.append(tryidx)
                                break
                            except:
                                continue
                        if tryidx is None:
                            raise ValueError(
                                f"None of these axial joint names were found in member, {memname}:{agrp_names}"
                            )
                    wt_opt.model.add_design_var(
                        f"floating.memgrp{idx}.grid_axial_joints", lower=lower, upper=upper, indices=aidx
                    )

        return wt_opt

    def set_constraints(self, wt_opt):
        blade_opt = self.opt["design_variables"]["blade"]

        # Set non-linear blade constraints
        blade_constr = self.opt["constraints"]["blade"]
        if blade_constr["strains_spar_cap_ss"]["flag"]:
            if blade_opt["structure"]["spar_cap_ss"]["flag"]:
                wt_opt.model.add_constraint("rs.constr.constr_max_strainU_spar", upper=1.0)
            else:
                print(
                    "WARNING: the strains of the suction-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced."
                )

        if blade_constr["strains_spar_cap_ps"]["flag"]:
            if (
                blade_opt["structure"]["spar_cap_ps"]["flag"]
                or blade_opt["structure"]["spar_cap_ps"]["equal_to_suction"]
            ):
                wt_opt.model.add_constraint("rs.constr.constr_max_strainL_spar", upper=1.0)
            else:
                print(
                    "WARNING: the strains of the pressure-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced."
                )

        if blade_constr["stall"]["flag"]:
            if blade_opt["aero_shape"]["twist"]["flag"]:
                wt_opt.model.add_constraint("stall_check.no_stall_constraint", upper=1.0)
            else:
                print(
                    "WARNING: the margin to stall is set to be constrained, but twist is not an active design variable. The constraint is not enforced."
                )

        if blade_constr["tip_deflection"]["flag"]:
            if blade_opt["structure"]["spar_cap_ss"]["flag"] or blade_opt["structure"]["spar_cap_ps"]["flag"]:
                wt_opt.model.add_constraint("tcons.tip_deflection_ratio", upper=1.0)
            else:
                print(
                    "WARNING: the tip deflection is set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced."
                )

        if blade_constr["chord"]["flag"]:
            if blade_opt["aero_shape"]["chord"]["flag"]:
                wt_opt.model.add_constraint("blade.pa.max_chord_constr", upper=1.0)
            else:
                print(
                    "WARNING: the max chord is set to be constrained, but chord is not an active design variable. The constraint is not enforced."
                )

        if blade_constr["frequency"]["flap_3P"]:
            if blade_opt["structure"]["spar_cap_ss"]["flag"] or blade_opt["structure"]["spar_cap_ps"]["flag"]:
                wt_opt.model.add_constraint("rs.constr.constr_flap_f_margin", upper=0.0)
            else:
                print(
                    "WARNING: the blade flap frequencies are set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced."
                )

        if blade_constr["frequency"]["edge_3P"]:
            wt_opt.model.add_constraint("rs.constr.constr_edge_f_margin", upper=0.0)

        if blade_constr["rail_transport"]["8_axle"]:
            wt_opt.model.add_constraint("re.rail.constr_LV_8axle_horiz", lower=0.8, upper=1.0)
            wt_opt.model.add_constraint("re.rail.constr_strainPS", upper=1.0)
            wt_opt.model.add_constraint("re.rail.constr_strainSS", upper=1.0)
        elif blade_constr["rail_transport"]["4_axle"]:
            wt_opt.model.add_constraint("re.rail.constr_LV_4axle_horiz", upper=1.0)
            wt_opt.model.add_constraint("re.rail.constr_strainPS", upper=1.0)
            wt_opt.model.add_constraint("re.rail.constr_strainSS", upper=1.0)

        if self.opt["constraints"]["blade"]["moment_coefficient"]["flag"]:
            wt_opt.model.add_constraint(
                "ccblade.CM",
                lower=self.opt["constraints"]["blade"]["moment_coefficient"]["min"],
                upper=self.opt["constraints"]["blade"]["moment_coefficient"]["max"],
            )
        if (
            self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cl"]
            or self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cd"]
        ):
            data_target = np.loadtxt(self.opt["constraints"]["blade"]["match_cl_cd"]["filename"])
            eta_opt = np.linspace(0.0, 1.0, self.opt["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"])
            target_cl = np.interp(eta_opt, data_target[:, 0], data_target[:, 3])
            target_cd = np.interp(eta_opt, data_target[:, 0], data_target[:, 4])
            eps_cl = 1.0e-2
            if self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cl"]:
                wt_opt.model.add_constraint("ccblade.cl_n_opt", lower=target_cl - eps_cl, upper=target_cl + eps_cl)
            if self.opt["constraints"]["blade"]["match_cl_cd"]["flag_cd"]:
                wt_opt.model.add_constraint("ccblade.cd_n_opt", lower=target_cd - eps_cl, upper=target_cd + eps_cl)
        if (
            self.opt["constraints"]["blade"]["match_L_D"]["flag_L"]
            or self.opt["constraints"]["blade"]["match_L_D"]["flag_D"]
        ):
            data_target = np.loadtxt(self.opt["constraints"]["blade"]["match_L_D"]["filename"])
            eta_opt = np.linspace(0.0, 1.0, self.opt["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"])
            target_L = np.interp(eta_opt, data_target[:, 0], data_target[:, 7])
            target_D = np.interp(eta_opt, data_target[:, 0], data_target[:, 8])
        eps_L = 1.0e2
        if self.opt["constraints"]["blade"]["match_L_D"]["flag_L"]:
            wt_opt.model.add_constraint("ccblade.L_n_opt", lower=target_L - eps_L, upper=target_L + eps_L)
        if self.opt["constraints"]["blade"]["match_L_D"]["flag_D"]:
            wt_opt.model.add_constraint("ccblade.D_n_opt", lower=target_D - eps_L, upper=target_D + eps_L)

        # Tower and monopile contraints
        tower_constr = self.opt["constraints"]["tower"]
        monopile_constr = self.opt["constraints"]["monopile"]
        if tower_constr["height_constraint"]["flag"]:
            wt_opt.model.add_constraint(
                "towerse.height_constraint",
                lower=tower_constr["height_constraint"]["lower_bound"],
                upper=tower_constr["height_constraint"]["upper_bound"],
            )

        if tower_constr["stress"]["flag"] or monopile_constr["stress"]["flag"]:
            for k in range(self.modeling["WISDEM"]["TowerSE"]["nLC"]):
                kstr = "" if self.modeling["WISDEM"]["TowerSE"]["nLC"] <= 1 else str(k + 1)
                wt_opt.model.add_constraint("towerse.post" + kstr + ".stress", upper=1.0)

        if tower_constr["global_buckling"]["flag"] or monopile_constr["global_buckling"]["flag"]:
            for k in range(self.modeling["WISDEM"]["TowerSE"]["nLC"]):
                kstr = "" if self.modeling["WISDEM"]["TowerSE"]["nLC"] <= 1 else str(k + 1)
                wt_opt.model.add_constraint("towerse.post" + kstr + ".global_buckling", upper=1.0)

        if tower_constr["shell_buckling"]["flag"] or monopile_constr["shell_buckling"]["flag"]:
            for k in range(self.modeling["WISDEM"]["TowerSE"]["nLC"]):
                kstr = "" if self.modeling["WISDEM"]["TowerSE"]["nLC"] <= 1 else str(k + 1)
                wt_opt.model.add_constraint("towerse.post" + kstr + ".shell_buckling", upper=1.0)

        if tower_constr["d_to_t"]["flag"] or monopile_constr["d_to_t"]["flag"]:
            wt_opt.model.add_constraint(
                "towerse.constr_d_to_t",
                lower=tower_constr["d_to_t"]["lower_bound"],
                upper=tower_constr["d_to_t"]["upper_bound"],
            )

        if tower_constr["taper"]["flag"] or monopile_constr["taper"]["flag"]:
            wt_opt.model.add_constraint("towerse.constr_taper", lower=tower_constr["taper"]["lower_bound"])

        if tower_constr["slope"]["flag"] or monopile_constr["slope"]["flag"]:
            wt_opt.model.add_constraint("towerse.slope", upper=1.0)

        if monopile_constr["pile_depth"]["flag"]:
            wt_opt.model.add_constraint("towerse.suctionpile_depth", lower=monopile_constr["pile_depth"]["lower_bound"])

        if tower_constr["frequency"]["flag"]:
            wt_opt.model.add_constraint("tcons.constr_tower_f_1Pmargin", upper=0.0)
            wt_opt.model.add_constraint("tcons.constr_tower_f_NPmargin", upper=0.0)

        elif tower_constr["frequency_1"]["flag"] or monopile_constr["frequency_1"]["flag"]:
            for k in range(self.modeling["WISDEM"]["TowerSE"]["nLC"]):
                kstr = "" if self.modeling["WISDEM"]["TowerSE"]["nLC"] <= 1 else str(k + 1)
                wt_opt.model.add_constraint(
                    "towerse.post" + kstr + ".structural_frequencies",
                    indices=[0],
                    lower=tower_constr["frequency_1"]["lower_bound"],
                    upper=tower_constr["frequency_1"]["upper_bound"],
                )

        # Hub and drivetrain constraints
        hub_constr = self.opt["constraints"]["hub"]
        drive_constr = self.opt["constraints"]["drivetrain"]

        if hub_constr["hub_diameter"]["flag"]:
            wt_opt.model.add_constraint("drivese.constr_hub_diameter", lower=0.0)

        for k in ["lss", "hss", "bedplate", "mb1", "mb2"]:
            if drive_constr[k]["flag"]:
                mystr = k + "_vonmises" if k in ["lss", "hss", "bedplate"] else k + "_defl"
                wt_opt.model.add_constraint("drivese.constr_" + mystr, upper=1.0)

        for k in ["length", "height", "access", "ecc"]:
            if drive_constr[k]["flag"]:
                wt_opt.model.add_constraint("drivese.constr_" + k, lower=0.0)

        # Floating platform and mooring constraints
        float_constr = self.opt["constraints"]["floating"]

        if float_constr["buoyancy"]["flag"]:
            wt_opt.model.add_constraint("floatingse.variable_ballast_mass", lower=0.0)

        if float_constr["fixed_ballast_capacity"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_fixed_margin", upper=1.0)

        if float_constr["variable_ballast_capacity"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_variable_margin", upper=1.0)

        if float_constr["metacentric_height"]["flag"]:
            wt_opt.model.add_constraint("floatingse.metacentric_height", lower=0.0)

        if float_constr["freeboard_margin"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_freeboard_heel_margin", upper=1.0)

        if float_constr["draft_margin"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_draft_heel_margin", upper=1.0)

        if float_constr["fairlead_depth"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_fairlead_wave", upper=1.0)

        if float_constr["mooring_heel"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_mooring_heel", upper=1.0)

        if float_constr["mooring_surge"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_mooring_surge", upper=1.0)

        if float_constr["mooring_tension"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_axial_load", upper=1.0)

        if float_constr["mooring_length"]["flag"]:
            wt_opt.model.add_constraint("floatingse.constr_mooring_length", upper=1.0)

        return wt_opt

    def set_recorders(self, wt_opt):
        folder_output = self.opt["general"]["folder_output"]

        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        if self.opt["recorder"]["flag"]:
            recorder = om.SqliteRecorder(os.path.join(folder_output, self.opt["recorder"]["file_name"]))
            wt_opt.driver.add_recorder(recorder)
            wt_opt.add_recorder(recorder)

            wt_opt.driver.recording_options["excludes"] = ["*_df"]
            wt_opt.driver.recording_options["record_constraints"] = True
            wt_opt.driver.recording_options["record_desvars"] = True
            wt_opt.driver.recording_options["record_objectives"] = True

        return wt_opt

    def set_initial(self, wt_opt, wt_init):
        blade_opt = self.opt["design_variables"]["blade"]

        if self.modeling["flags"]["blade"]:
            wt_opt["blade.opt_var.s_opt_twist"] = np.linspace(0.0, 1.0, blade_opt["aero_shape"]["twist"]["n_opt"])
            if blade_opt["aero_shape"]["twist"]["flag"]:
                init_twist_opt = np.interp(
                    wt_opt["blade.opt_var.s_opt_twist"],
                    wt_init["components"]["blade"]["outer_shape_bem"]["twist"]["grid"],
                    wt_init["components"]["blade"]["outer_shape_bem"]["twist"]["values"],
                )
                lb_twist = np.array(blade_opt["aero_shape"]["twist"]["lower_bound"])
                ub_twist = np.array(blade_opt["aero_shape"]["twist"]["upper_bound"])
                wt_opt["blade.opt_var.twist_opt_gain"] = (init_twist_opt - lb_twist) / (ub_twist - lb_twist)
                if (
                    max(wt_opt["blade.opt_var.twist_opt_gain"]) > 1.0
                    or min(wt_opt["blade.opt_var.twist_opt_gain"]) < 0.0
                ):
                    print(
                        "Warning: the initial twist violates the upper or lower bounds of the twist design variables."
                    )

            blade_constr = self.opt["constraints"]["blade"]
            wt_opt["blade.opt_var.s_opt_chord"] = np.linspace(0.0, 1.0, blade_opt["aero_shape"]["chord"]["n_opt"])
            wt_opt["blade.ps.s_opt_spar_cap_ss"] = np.linspace(0.0, 1.0, blade_opt["structure"]["spar_cap_ss"]["n_opt"])
            wt_opt["blade.ps.s_opt_spar_cap_ps"] = np.linspace(0.0, 1.0, blade_opt["structure"]["spar_cap_ps"]["n_opt"])
            wt_opt["rs.constr.max_strainU_spar"] = blade_constr["strains_spar_cap_ss"]["max"]
            wt_opt["rs.constr.max_strainL_spar"] = blade_constr["strains_spar_cap_ps"]["max"]
            wt_opt["stall_check.stall_margin"] = blade_constr["stall"]["margin"] * 180.0 / np.pi
            wt_opt["tcons.max_allowable_td_ratio"] = blade_constr["tip_deflection"]["margin"]

        if self.modeling["flags"]["nacelle"] and self.modeling["WISDEM"]["DriveSE"]["direct"]:
            drive_constr = self.opt["constraints"]["drivetrain"]
            wt_opt["drivese.access_diameter"] = drive_constr["access"]["lower_bound"]

        if self.modeling["flags"]["floating"]:
            float_constr = self.opt["constraints"]["floating"]
            wt_opt["floatingse.max_surge_fraction"] = float_constr["max_surge"]["upper_bound"]
            wt_opt.set_val("floatingse.operational_heel", float_constr["operational_heel"]["upper_bound"], units="rad")
            wt_opt.set_val("floatingse.survival_heel", float_constr["survival_heel"]["upper_bound"], units="rad")

        return wt_opt

    def set_restart(self, wt_opt):
        if "warmstart_file" in self.opt["driver"]["optimization"]:

            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict

            db = SqliteDict(self.opt["driver"]["optimization"]["warmstart_file"])

            # Grab the last iteration's design variables
            last_key = db["last"]
            desvars = db[last_key]["xuser"]

            # Obtain the already-setup OM problem's design variables
            if wt_opt.model._static_mode:
                design_vars = wt_opt.model._static_design_vars
            else:
                design_vars = wt_opt.model._design_vars

            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = wt_opt.model._var_allprocs_prom2abs_list["output"]
            abs2prom = {}
            for key in design_vars:
                abs2prom[prom2abs[key][0]] = key

            # Loop through each design variable
            for key in desvars:
                prom_key = abs2prom[key]

                # Scale each DV based on the OM scaling from the problem.
                # This assumes we're running the same problem with the same scaling
                scaler = design_vars[prom_key]["scaler"]
                adder = design_vars[prom_key]["adder"]

                if scaler is None:
                    scaler = 1.0
                if adder is None:
                    adder = 0.0

                scaled_dv = desvars[key] / scaler - adder

                # Special handling for blade twist as we only have the
                # last few control points as design variables
                if "twist_opt_gain" in key:
                    wt_opt[key][2:] = scaled_dv
                else:
                    wt_opt[key][:] = scaled_dv

        return wt_opt
