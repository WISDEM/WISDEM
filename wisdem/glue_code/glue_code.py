import numpy as np
import openmdao.api as om

from wisdem.rotorse.rotor import RotorSE
from wisdem.towerse.tower import TowerSE
from wisdem.floatingse.floating import FloatingSE
from wisdem.fixed_bottomse.jacket import JacketSE
from wisdem.glue_code.gc_RunTools import Outputs_2_Screen
from wisdem.drivetrainse.drivetrain import DrivetrainSE
from wisdem.fixed_bottomse.monopile import MonopileSE
from wisdem.glue_code.gc_WT_DataStruc import WindTurbineOntologyOpenMDAO
from wisdem.nrelcsm.nrel_csm_cost_2015 import Turbine_CostsSE_2015
from wisdem.commonse.turbine_constraints import TurbineConstraints
from wisdem.plant_financese.plant_finance import PlantFinance
from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE

try:
    from wisdem.orbit.api.wisdem import Orbit
except ImportError:
    print("WARNING: Be sure to pip install simpy and marmot-agents for offshore BOS runs")


class WT_RNTA(om.Group):
    # Openmdao group to run the analysis of the wind turbine

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        nLC = modeling_options["WISDEM"]["n_dlc"]
        opt_options = self.options["opt_options"]

        if modeling_options["flags"]["blade"] and modeling_options["flags"]["nacelle"]:
            self.linear_solver = lbgs = om.LinearBlockGS()
            self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
            nlbgs.options["maxiter"] = modeling_options["General"]["solver_maxiter"]
            nlbgs.options["atol"] = 1e-2
            nlbgs.options["rtol"] = 1e-8
            nlbgs.options["iprint"] = 2

        # Analysis components
        self.add_subsystem(
            "wt_init",
            WindTurbineOntologyOpenMDAO(modeling_options=modeling_options, opt_options=opt_options),
            promotes=["*"],
        )

        if modeling_options["flags"]["blade"]:
            self.add_subsystem("rotorse", RotorSE(modeling_options=modeling_options, opt_options=opt_options))

        if modeling_options["flags"]["nacelle"]:
            self.add_subsystem("drivese", DrivetrainSE(modeling_options=modeling_options))

        if modeling_options["flags"]["tower"]:
            self.add_subsystem("towerse", TowerSE(modeling_options=modeling_options))

        if modeling_options["flags"]["monopile"]:
            self.add_subsystem("fixedse", MonopileSE(modeling_options=modeling_options))

        elif modeling_options["flags"]["jacket"]:
            self.add_subsystem("fixedse", JacketSE(modeling_options=modeling_options))

        elif modeling_options["flags"]["floating"]:
            self.add_subsystem("floatingse", FloatingSE(modeling_options=modeling_options))

        if modeling_options["flags"]["blade"] and modeling_options["flags"]["tower"]:
            self.add_subsystem("tcons", TurbineConstraints(modeling_options=modeling_options))

        self.add_subsystem("tcc", Turbine_CostsSE_2015(verbosity=modeling_options["General"]["verbosity"]))

        if modeling_options["flags"]["blade"]:
            n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]

            self.connect("rotorse.ccblade.local_airfoil_velocities", "blade.compute_reynolds.local_airfoil_velocities")
            self.connect("blade.pa.chord_param", "blade.compute_reynolds.chord")
            self.connect("env.rho_air", "blade.compute_reynolds.rho")
            self.connect("env.mu_air", "blade.compute_reynolds.mu")

            # Conncetions to ccblade
            self.connect("blade.pa.chord_param", "rotorse.chord")
            self.connect("blade.pa.twist_param", "rotorse.ccblade.theta_in")
            self.connect("blade.opt_var.s_opt_chord", "rotorse.ccblade.s_opt_chord")
            self.connect("blade.opt_var.s_opt_twist", "rotorse.ccblade.s_opt_theta")
            self.connect("blade.outer_shape_bem.s", "rotorse.s")
            self.connect("blade.high_level_blade_props.r_blade", "rotorse.r")
            self.connect("blade.high_level_blade_props.rotor_radius", "rotorse.Rtip")
            self.connect("hub.radius", "rotorse.Rhub")
            self.connect("blade.interp_airfoils.r_thick_interp", "rotorse.ccblade.rthick")
            self.connect("airfoils.aoa", "rotorse.airfoils_aoa")
            self.connect("airfoils.Re", "rotorse.airfoils_Re")
            self.connect("af_3d.cl_corrected", "rotorse.airfoils_cl")
            self.connect("af_3d.cd_corrected", "rotorse.airfoils_cd")
            self.connect("af_3d.cm_corrected", "rotorse.airfoils_cm")
            if modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                self.connect("blade.run_inn_af.aoa_inn", "rotorse.ccblade.aoa_op")
            self.connect("high_level_tower_props.hub_height", "rotorse.hub_height")
            self.connect("hub.cone", "rotorse.precone")
            self.connect("nacelle.uptilt", "rotorse.tilt")

            self.connect("blade.high_level_blade_props.prebend", "rotorse.precurve")
            self.connect("blade.high_level_blade_props.prebendTip", "rotorse.precurveTip")
            self.connect("blade.high_level_blade_props.presweep", "rotorse.presweep")
            self.connect("blade.high_level_blade_props.presweepTip", "rotorse.presweepTip")

            if modeling_options["flags"]["control"]:
                self.connect("control.rated_pitch", "rotorse.pitch")
            self.connect("control.rated_TSR", "rotorse.tsr")
            self.connect("env.rho_air", "rotorse.rho_air")
            self.connect("env.mu_air", "rotorse.mu_air")
            self.connect("env.shear_exp", "rotorse.shearExp")
            self.connect(
                "configuration.n_blades",
                ["rotorse.nBlades", "rotorse.re.precomp.n_blades", "rotorse.rs.constr.blade_number"],
            )
            self.connect("configuration.ws_class", "rotorse.wt_class.turbine_class")
            self.connect("blade.ps.layer_thickness_param", "rotorse.re.precomp.layer_thickness")

            # Connections to rotor elastic and frequency analysis
            self.connect("nacelle.uptilt", "rotorse.re.precomp.uptilt")
            self.connect("blade.outer_shape_bem.pitch_axis", "rotorse.re.pitch_axis")
            if modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                self.connect("blade.run_inn_af.coord_xy_interp", "rotorse.re.coord_xy_interp")
            else:
                self.connect("blade.interp_airfoils.coord_xy_interp", "rotorse.re.coord_xy_interp")
            self.connect("blade.internal_structure_2d_fem.layer_start_nd", "rotorse.re.precomp.layer_start_nd")
            self.connect("blade.internal_structure_2d_fem.layer_end_nd", "rotorse.re.precomp.layer_end_nd")
            self.connect("blade.internal_structure_2d_fem.layer_web", "rotorse.re.precomp.layer_web")
            self.connect("blade.internal_structure_2d_fem.definition_layer", "rotorse.re.precomp.definition_layer")
            self.connect("blade.internal_structure_2d_fem.web_start_nd", "rotorse.re.precomp.web_start_nd")
            self.connect("blade.internal_structure_2d_fem.web_end_nd", "rotorse.re.precomp.web_end_nd")
            self.connect("blade.internal_structure_2d_fem.joint_position", "rotorse.re.precomp.joint_position")
            if modeling_options["WISDEM"]["RotorSE"]["bjs"]:
                self.connect("blade.internal_structure_2d_fem.joint_bolt", "rotorse.rs.bjs.joint_bolt")
                # Let wisdem estimate the joint mass, although 
                # this generates an implicit loop since the bjs modules requires loads among the inputs
                self.connect("rotorse.rs.bjs.joint_mass", "rotorse.re.precomp.joint_mass") 
            else:
                # joint mass as user input from yaml
                self.connect("blade.internal_structure_2d_fem.joint_mass", "rotorse.re.precomp.joint_mass") 
            self.connect("materials.name", "rotorse.re.precomp.mat_name")
            self.connect("materials.orth", "rotorse.re.precomp.orth")
            self.connect("materials.E", "rotorse.re.precomp.E")
            self.connect("materials.G", "rotorse.re.precomp.G")
            self.connect("materials.nu", "rotorse.re.precomp.nu")
            self.connect("materials.rho", "rotorse.re.precomp.rho")

            # Conncetions to rail transport module
            if (
                modeling_options["WISDEM"]["RotorSE"]["rail_transport"]
                or opt_options["constraints"]["blade"]["rail_transport"]["flag"]
            ):
                self.connect("blade.high_level_blade_props.blade_ref_axis", "rotorse.re.rail.blade_ref_axis")
            # Connections from blade struct parametrization to rotor load anlysis
            spars_tereinf = modeling_options["WISDEM"]["RotorSE"]["spars_tereinf"]
            self.connect("blade.opt_var.s_opt_layer_%d"%spars_tereinf[0], "rotorse.rs.constr.s_opt_spar_cap_ss")
            self.connect("blade.opt_var.s_opt_layer_%d"%spars_tereinf[1], "rotorse.rs.constr.s_opt_spar_cap_ps")
            self.connect("blade.opt_var.s_opt_layer_%d"%spars_tereinf[2], "rotorse.rs.constr.s_opt_te_ss")
            self.connect("blade.opt_var.s_opt_layer_%d"%spars_tereinf[3], "rotorse.rs.constr.s_opt_te_ps")

            # Connections to RotorPower
            self.connect("control.V_in", "rotorse.rp.v_min")
            self.connect("control.V_out", "rotorse.rp.v_max")
            self.connect("configuration.rated_power", "rotorse.rp.rated_power")
            self.connect("control.minOmega", "rotorse.rp.omega_min")
            self.connect("control.maxOmega", "rotorse.rp.omega_max")
            self.connect("control.max_TS", "rotorse.rp.control_maxTS")
            self.connect("configuration.gearbox_type", "rotorse.rp.drivetrainType")
            self.connect("nacelle.gearbox_efficiency", "rotorse.rp.powercurve.gearbox_efficiency")
            if modeling_options["flags"]["nacelle"]:
                self.connect("drivese.lss_rpm", "rotorse.rp.powercurve.lss_rpm")
                self.connect("drivese.generator_efficiency", "rotorse.rp.powercurve.generator_efficiency")
            self.connect("env.weibull_k", "rotorse.rp.cdf.k")
            self.connect("configuration.turb_class", "rotorse.rp.gust.turbulence_class")

            # Connections to RotorStructure
            self.connect("blade.internal_structure_2d_fem.d_f", "rotorse.rs.brs.d_f")
            self.connect("blade.internal_structure_2d_fem.sigma_max", "rotorse.rs.brs.sigma_max")
            self.connect("blade.pa.chord_param", "rotorse.rs.brs.rootD", src_indices=[0])
            self.connect("blade.ps.layer_thickness_param", "rotorse.rs.brs.layer_thickness")
            self.connect("blade.internal_structure_2d_fem.layer_start_nd", "rotorse.rs.brs.layer_start_nd")
            self.connect("blade.internal_structure_2d_fem.layer_end_nd", "rotorse.rs.brs.layer_end_nd")
            if modeling_options["WISDEM"]["RotorSE"]["bjs"]:
                self.connect("materials.name", "rotorse.rs.bjs.name_mat")
                self.connect("materials.rho", "rotorse.rs.bjs.rho_mat")
                self.connect("materials.Xt", "rotorse.rs.bjs.Xt_mat")
                self.connect("materials.sigma_y", "rotorse.rs.bjs.Xy_mat")
                self.connect("materials.E", "rotorse.rs.bjs.E_mat")
                self.connect("materials.S", "rotorse.rs.bjs.S_mat")
                self.connect("blade.pa.chord_param", "rotorse.rs.bjs.chord")
                self.connect("blade.high_level_blade_props.r_blade", "rotorse.rs.bjs.blade_length")
                self.connect("blade.ps.layer_thickness_param", "rotorse.rs.bjs.layer_thickness")
                self.connect("blade.internal_structure_2d_fem.layer_width", "rotorse.rs.bjs.layer_width")
                self.connect("blade.internal_structure_2d_fem.layer_start_nd", "rotorse.rs.bjs.layer_start_nd")
                self.connect("blade.internal_structure_2d_fem.layer_end_nd", "rotorse.rs.bjs.layer_end_nd")
                self.connect("blade.interp_airfoils.coord_xy_interp", "rotorse.rs.bjs.coord_xy_interp")
                self.connect("blade.interp_airfoils.r_thick_interp", "rotorse.rs.bjs.rthick")
                self.connect("blade.internal_structure_2d_fem.joint_position", "rotorse.rs.bjs.joint_position")
                self.connect(
                    "blade.internal_structure_2d_fem.joint_nonmaterial_cost", "rotorse.rs.bjs.joint_nonmaterial_cost"
                )
                self.connect(
                    "blade.internal_structure_2d_fem.reinforcement_layer_ss", "rotorse.rs.bjs.reinforcement_layer_ss"
                )
                self.connect(
                    "blade.internal_structure_2d_fem.reinforcement_layer_ps", "rotorse.rs.bjs.reinforcement_layer_ps"
                )
                # self.connect("blade.outer_shape_bem.thickness", "rotorse.rs.bjs.blade_thickness")
                self.connect("blade.internal_structure_2d_fem.layer_offset_y_pa", "rotorse.rs.bjs.layer_offset_y_pa")
                self.connect("blade.compute_coord_xy_dim.coord_xy_dim", "rotorse.rs.bjs.coord_xy_dim")
                self.connect("blade.internal_structure_2d_fem.layer_side", "rotorse.rs.bjs.layer_side")
                self.connect("blade.pa.twist_param", "rotorse.rs.bjs.twist")
                self.connect("blade.outer_shape_bem.pitch_axis", "rotorse.rs.bjs.pitch_axis")
                self.connect("materials.unit_cost", "rotorse.rs.bjs.unit_cost")
                # Connections to RotorCost
                # Inputs to be split between inner and outer blade portions
                self.connect("blade.high_level_blade_props.blade_length", "rotorse.split.blade_length")
                self.connect("blade.outer_shape_bem.s", "rotorse.split.s")
                self.connect("blade.pa.chord_param", "rotorse.split.chord")
                if modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                    self.connect("blade.run_inn_af.coord_xy_interp", "rotorse.split.coord_xy_interp")
                else:
                    self.connect("blade.interp_airfoils.coord_xy_interp", "rotorse.split.coord_xy_interp")
                self.connect("blade.ps.layer_thickness_param", "rotorse.split.layer_thickness")
                self.connect("blade.internal_structure_2d_fem.layer_start_nd", "rotorse.split.layer_start_nd")
                self.connect("blade.internal_structure_2d_fem.layer_end_nd", "rotorse.split.layer_end_nd")
                self.connect("blade.internal_structure_2d_fem.web_start_nd", "rotorse.split.web_start_nd")
                self.connect("blade.internal_structure_2d_fem.web_end_nd", "rotorse.split.web_end_nd")
                self.connect("blade.internal_structure_2d_fem.joint_position", "rotorse.split.joint_position")

                # Common inputs to blade cost model
                self.connect("materials.name", ["rotorse.rc_in.mat_name", "rotorse.rc_out.mat_name"])
                self.connect("materials.orth", ["rotorse.rc_in.orth", "rotorse.rc_out.orth"])
                self.connect("materials.rho", ["rotorse.rc_in.rho", "rotorse.rc_out.rho"])
                self.connect("materials.component_id", ["rotorse.rc_in.component_id", "rotorse.rc_out.component_id"])
                self.connect("materials.unit_cost", ["rotorse.rc_in.unit_cost", "rotorse.rc_out.unit_cost"])
                self.connect("materials.waste", ["rotorse.rc_in.waste", "rotorse.rc_out.waste"])
                self.connect("materials.rho_fiber", ["rotorse.rc_in.rho_fiber", "rotorse.rc_out.rho_fiber"])
                self.connect("materials.ply_t", ["rotorse.rc_in.ply_t", "rotorse.rc_out.ply_t"])
                self.connect("materials.fwf", ["rotorse.rc_in.fwf", "rotorse.rc_out.fwf"])
                self.connect("materials.fvf", ["rotorse.rc_in.fvf", "rotorse.rc_out.fvf"])
                self.connect("materials.roll_mass", ["rotorse.rc_in.roll_mass", "rotorse.rc_out.roll_mass"])
                self.connect(
                    "blade.internal_structure_2d_fem.definition_layer",
                    ["rotorse.rc_in.definition_layer", "rotorse.rc_out.definition_layer"],
                )
                self.connect(
                    "blade.internal_structure_2d_fem.layer_web", ["rotorse.rc_in.layer_web", "rotorse.rc_out.layer_web"]
                )

            else:
                self.connect("blade.high_level_blade_props.blade_length", "rotorse.rc.blade_length")
                self.connect("blade.outer_shape_bem.s", "rotorse.rc.s")
                self.connect("blade.pa.chord_param", "rotorse.rc.chord")
                if modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                    self.connect("blade.run_inn_af.coord_xy_interp", "rotorse.rc.coord_xy_interp")
                else:
                    self.connect("blade.interp_airfoils.coord_xy_interp", "rotorse.rc.coord_xy_interp")
                self.connect("blade.ps.layer_thickness_param", "rotorse.rc.layer_thickness")
                self.connect("blade.internal_structure_2d_fem.layer_start_nd", "rotorse.rc.layer_start_nd")
                self.connect("blade.internal_structure_2d_fem.layer_end_nd", "rotorse.rc.layer_end_nd")
                self.connect("blade.internal_structure_2d_fem.layer_web", "rotorse.rc.layer_web")
                self.connect("blade.internal_structure_2d_fem.definition_layer", "rotorse.rc.definition_layer")
                self.connect("blade.internal_structure_2d_fem.web_start_nd", "rotorse.rc.web_start_nd")
                self.connect("blade.internal_structure_2d_fem.web_end_nd", "rotorse.rc.web_end_nd")
                self.connect("materials.name", "rotorse.rc.mat_name")
                self.connect("materials.orth", "rotorse.rc.orth")
                self.connect("materials.rho", "rotorse.rc.rho")
                self.connect("materials.component_id", "rotorse.rc.component_id")
                self.connect("materials.unit_cost", "rotorse.rc.unit_cost")
                self.connect("materials.waste", "rotorse.rc.waste")
                self.connect("materials.rho_fiber", "rotorse.rc.rho_fiber")
                self.connect("materials.ply_t", "rotorse.rc.ply_t")
                self.connect("materials.fwf", "rotorse.rc.fwf")
                self.connect("materials.fvf", "rotorse.rc.fvf")
                self.connect("materials.roll_mass", "rotorse.rc.roll_mass")

        # Connections to DriveSE
        if modeling_options["flags"]["nacelle"]:
            self.connect("hub.diameter", "drivese.hub_diameter")
            self.connect("hub.hub_in2out_circ", "drivese.hub_in2out_circ")
            self.connect("hub.flange_t2shell_t", "drivese.flange_t2shell_t")
            self.connect("hub.flange_OD2hub_D", "drivese.flange_OD2hub_D")
            self.connect("hub.flange_ID2flange_OD", "drivese.flange_ID2flange_OD")
            self.connect("hub.hub_stress_concentration", "drivese.hub_stress_concentration")
            self.connect("hub.n_front_brackets", "drivese.n_front_brackets")
            self.connect("hub.n_rear_brackets", "drivese.n_rear_brackets")
            self.connect("hub.clearance_hub_spinner", "drivese.clearance_hub_spinner")
            self.connect("hub.spin_hole_incr", "drivese.spin_hole_incr")
            self.connect("hub.pitch_system_scaling_factor", "drivese.pitch_system_scaling_factor")
            self.connect("rotorse.wt_class.V_extreme50", "drivese.spinner_gust_ws")

            self.connect("configuration.n_blades", "drivese.n_blades")

            self.connect("blade.high_level_blade_props.rotor_diameter", "drivese.rotor_diameter")
            self.connect("configuration.upwind", "drivese.upwind")
            self.connect("control.minOmega", "drivese.minimum_rpm")
            self.connect("rotorse.rp.powercurve.rated_Omega", "drivese.rated_rpm")
            self.connect("rotorse.rp.powercurve.rated_Q", "drivese.rated_torque")
            self.connect("configuration.rated_power", "drivese.machine_rating")
            if modeling_options["flags"]["tower"]:
                self.connect("tower.diameter", "drivese.D_top", src_indices=[-1])

            self.connect("rotorse.rs.aero_hub_loads.Fhub", "drivese.F_hub")
            self.connect("rotorse.rs.aero_hub_loads.Mhub", "drivese.M_hub")
            self.connect("rotorse.rs.frame.root_M", "drivese.pitch_system.BRFM", src_indices=[1])

            self.connect("blade.pa.chord_param", "drivese.blade_root_diameter", src_indices=[0])
            self.connect("rotorse.rs.curvature.blades_cg_hubcc", "drivese.blades_cm")
            self.connect("rotorse.blade_mass", "drivese.blade_mass")
            self.connect("rotorse.mass_all_blades", "drivese.blades_mass")
            self.connect("rotorse.I_all_blades", "drivese.blades_I")

            self.connect("nacelle.distance_hub_mb", "drivese.L_h1")
            self.connect("nacelle.distance_mb_mb", "drivese.L_12")
            self.connect("nacelle.L_generator", "drivese.L_generator")
            self.connect("nacelle.overhang", "drivese.overhang")
            self.connect("nacelle.distance_tt_hub", "drivese.drive_height")
            self.connect("nacelle.uptilt", "drivese.tilt")
            self.connect("nacelle.gear_ratio", "drivese.gear_ratio")
            self.connect("nacelle.damping_ratio", "drivese.damping_ratio")
            self.connect("nacelle.mb1Type", "drivese.bear1.bearing_type")
            self.connect("nacelle.mb2Type", "drivese.bear2.bearing_type")
            self.connect("nacelle.lss_diameter", "drivese.lss_diameter")
            self.connect("nacelle.lss_wall_thickness", "drivese.lss_wall_thickness")
            if modeling_options["WISDEM"]["DriveSE"]["direct"]:
                self.connect("nacelle.nose_diameter", "drivese.bear1.D_shaft", src_indices=[0])
                self.connect("nacelle.nose_diameter", "drivese.bear2.D_shaft", src_indices=[-1])
            else:
                self.connect("nacelle.lss_diameter", "drivese.bear1.D_shaft", src_indices=[0])
                self.connect("nacelle.lss_diameter", "drivese.bear2.D_shaft", src_indices=[-1])
            self.connect("nacelle.uptower", "drivese.uptower")
            self.connect("nacelle.brake_mass_user", "drivese.brake_mass_user")
            self.connect("nacelle.hvac_mass_coeff", "drivese.hvac_mass_coeff")
            self.connect("nacelle.converter_mass_user", "drivese.converter_mass_user")
            self.connect("nacelle.transformer_mass_user", "drivese.transformer_mass_user")

            if modeling_options["WISDEM"]["DriveSE"]["direct"]:
                self.connect("nacelle.nose_diameter", "drivese.nose_diameter")
                self.connect("nacelle.nose_wall_thickness", "drivese.nose_wall_thickness")
                self.connect("nacelle.bedplate_wall_thickness", "drivese.bedplate_wall_thickness")
            else:
                self.connect("nacelle.hss_length", "drivese.L_hss")
                self.connect("nacelle.hss_diameter", "drivese.hss_diameter")
                self.connect("nacelle.hss_wall_thickness", "drivese.hss_wall_thickness")
                self.connect("nacelle.hss_material", "drivese.hss_material")
                self.connect("nacelle.planet_numbers", "drivese.planet_numbers")
                self.connect("nacelle.gear_configuration", "drivese.gear_configuration")
                self.connect("nacelle.gearbox_mass_user", "drivese.gearbox_mass_user")
                self.connect("nacelle.gearbox_torque_density", "drivese.gearbox_torque_density")
                self.connect("nacelle.gearbox_radius_user", "drivese.gearbox_radius_user")
                self.connect("nacelle.gearbox_length_user", "drivese.gearbox_length_user")
                self.connect("nacelle.bedplate_flange_width", "drivese.bedplate_flange_width")
                self.connect("nacelle.bedplate_flange_thickness", "drivese.bedplate_flange_thickness")
                self.connect("nacelle.bedplate_web_thickness", "drivese.bedplate_web_thickness")

            self.connect("hub.hub_material", "drivese.hub_material")
            self.connect("hub.spinner_material", "drivese.spinner_material")
            self.connect("nacelle.lss_material", "drivese.lss_material")
            self.connect("nacelle.bedplate_material", "drivese.bedplate_material")
            self.connect("materials.name", "drivese.material_names")
            self.connect("materials.E", "drivese.E_mat")
            self.connect("materials.G", "drivese.G_mat")
            self.connect("materials.rho", "drivese.rho_mat")
            self.connect("materials.sigma_y", "drivese.Xy_mat")
            self.connect("materials.Xt", "drivese.Xt_mat")
            self.connect("materials.wohler_exp", "drivese.wohler_exp_mat")
            self.connect("materials.wohler_intercept", "drivese.wohler_A_mat")
            self.connect("materials.unit_cost", "drivese.unit_cost_mat")

            if modeling_options["flags"]["generator"]:
                self.connect("generator.B_r", "drivese.generator.B_r")
                self.connect("generator.P_Fe0e", "drivese.generator.P_Fe0e")
                self.connect("generator.P_Fe0h", "drivese.generator.P_Fe0h")
                self.connect("generator.S_N", "drivese.generator.S_N")
                self.connect("generator.alpha_p", "drivese.generator.alpha_p")
                self.connect("generator.b_r_tau_r", "drivese.generator.b_r_tau_r")
                self.connect("generator.b_ro", "drivese.generator.b_ro")
                self.connect("generator.b_s_tau_s", "drivese.generator.b_s_tau_s")
                self.connect("generator.b_so", "drivese.generator.b_so")
                self.connect("generator.cofi", "drivese.generator.cofi")
                self.connect("generator.freq", "drivese.generator.freq")
                self.connect("generator.h_i", "drivese.generator.h_i")
                self.connect("generator.h_sy0", "drivese.generator.h_sy0")
                self.connect("generator.h_w", "drivese.generator.h_w")
                self.connect("generator.k_fes", "drivese.generator.k_fes")
                self.connect("generator.k_fillr", "drivese.generator.k_fillr")
                self.connect("generator.k_fills", "drivese.generator.k_fills")
                self.connect("generator.k_s", "drivese.generator.k_s")
                self.connect("generator.m", "drivese.generator.m")
                self.connect("generator.mu_0", "drivese.generator.mu_0")
                self.connect("generator.mu_r", "drivese.generator.mu_r")
                self.connect("generator.p", "drivese.generator.p")
                self.connect("generator.phi", "drivese.generator.phi")
                self.connect("generator.q1", "drivese.generator.q1")
                self.connect("generator.q2", "drivese.generator.q2")
                self.connect("generator.ratio_mw2pp", "drivese.generator.ratio_mw2pp")
                self.connect("generator.resist_Cu", "drivese.generator.resist_Cu")
                self.connect("generator.sigma", "drivese.generator.sigma")
                self.connect("generator.y_tau_p", "drivese.generator.y_tau_p")
                self.connect("generator.y_tau_pr", "drivese.generator.y_tau_pr")

                self.connect("generator.I_0", "drivese.generator.I_0")
                self.connect("generator.d_r", "drivese.generator.d_r")
                self.connect("generator.h_m", "drivese.generator.h_m")
                self.connect("generator.h_0", "drivese.generator.h_0")
                self.connect("generator.h_s", "drivese.generator.h_s")
                self.connect("generator.len_s", "drivese.generator.len_s")
                self.connect("generator.n_r", "drivese.generator.n_r")
                self.connect("generator.rad_ag", "drivese.generator.rad_ag")
                self.connect("generator.t_wr", "drivese.generator.t_wr")

                self.connect("generator.n_s", "drivese.generator.n_s")
                self.connect("generator.b_st", "drivese.generator.b_st")
                self.connect("generator.d_s", "drivese.generator.d_s")
                self.connect("generator.t_ws", "drivese.generator.t_ws")

                self.connect("generator.rho_Copper", "drivese.generator.rho_Copper")
                self.connect("generator.rho_Fe", "drivese.generator.rho_Fe")
                self.connect("generator.rho_Fes", "drivese.generator.rho_Fes")
                self.connect("generator.rho_PM", "drivese.generator.rho_PM")

                self.connect("generator.C_Cu", "drivese.generator.C_Cu")
                self.connect("generator.C_Fe", "drivese.generator.C_Fe")
                self.connect("generator.C_Fes", "drivese.generator.C_Fes")
                self.connect("generator.C_PM", "drivese.generator.C_PM")

                if modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["pmsg_outer"]:
                    self.connect("generator.N_c", "drivese.generator.N_c")
                    self.connect("generator.b", "drivese.generator.b")
                    self.connect("generator.c", "drivese.generator.c")
                    self.connect("generator.E_p", "drivese.generator.E_p")
                    self.connect("generator.h_yr", "drivese.generator.h_yr")
                    self.connect("generator.h_ys", "drivese.generator.h_ys")
                    self.connect("generator.h_sr", "drivese.generator.h_sr")
                    self.connect("generator.h_ss", "drivese.generator.h_ss")
                    self.connect("generator.t_r", "drivese.generator.t_r")
                    self.connect("generator.t_s", "drivese.generator.t_s")

                    self.connect("generator.u_allow_pcent", "drivese.generator.u_allow_pcent")
                    self.connect("generator.y_allow_pcent", "drivese.generator.y_allow_pcent")
                    self.connect("generator.z_allow_deg", "drivese.generator.z_allow_deg")
                    self.connect("generator.B_tmax", "drivese.generator.B_tmax")
                    self.connect("rotorse.rp.powercurve.rated_mech", "drivese.generator.P_mech")

                if modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
                    self.connect("generator.tau_p", "drivese.generator.tau_p")
                    self.connect("generator.h_ys", "drivese.generator.h_ys")
                    self.connect("generator.h_yr", "drivese.generator.h_yr")
                    self.connect("generator.b_arm", "drivese.generator.b_arm")

                elif modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["scig", "dfig"]:
                    self.connect("generator.B_symax", "drivese.generator.B_symax")
                    self.connect("generator.S_Nmax", "drivese.generator.S_Nmax")

                if modeling_options["WISDEM"]["DriveSE"]["direct"]:
                    self.connect("nacelle.nose_diameter", "drivese.generator.D_nose", src_indices=[-1])
                    self.connect("nacelle.lss_diameter", "drivese.generator.D_shaft", src_indices=[0])
                else:
                    self.connect("nacelle.hss_diameter", "drivese.generator.D_shaft", src_indices=[-1])

            else:
                self.connect("generator.generator_radius_user", "drivese.generator_radius_user")
                self.connect("generator.generator_mass_user", "drivese.generator_mass_user")
                self.connect("generator.generator_efficiency_user", "drivese.generator_efficiency_user")

        # Connections to TowerSE
        if modeling_options["flags"]["tower"]:
            if modeling_options["flags"]["nacelle"]:
                self.connect("drivese.base_F", "towerse.tower.rna_F")
                self.connect("drivese.base_M", "towerse.tower.rna_M")
                self.connect("drivese.rna_I_TT", "towerse.rna_I")
                self.connect("drivese.rna_cm", "towerse.rna_cg")
                self.connect("drivese.rna_mass", "towerse.rna_mass")
            if modeling_options["flags"]["blade"]:
                self.connect("rotorse.rp.gust.V_gust", "towerse.env.Uref")
            self.connect("high_level_tower_props.hub_height", "towerse.wind_reference_height")
            self.connect("high_level_tower_props.hub_height", "towerse.hub_height")
            self.connect("env.rho_air", "towerse.rho_air")
            self.connect("env.mu_air", "towerse.mu_air")
            self.connect("env.shear_exp", "towerse.shearExp")
            self.connect("tower_grid.foundation_height", "towerse.foundation_height")
            self.connect("tower.diameter", "towerse.tower_outer_diameter_in")
            self.connect("tower_grid.height", "towerse.tower_height")
            self.connect("tower_grid.s", "towerse.tower_s")
            self.connect("tower.layer_thickness", "towerse.tower_layer_thickness")
            self.connect("tower.outfitting_factor", "towerse.outfitting_factor_in")
            self.connect("tower.layer_mat", "towerse.tower_layer_materials")
            self.connect("materials.name", "towerse.material_names")
            self.connect("materials.E", "towerse.E_mat")
            self.connect("materials.G", "towerse.G_mat")
            self.connect("materials.rho", "towerse.rho_mat")
            self.connect("materials.sigma_y", "towerse.sigma_y_mat")
            self.connect("materials.Xt", "towerse.sigma_ult_mat")
            self.connect("materials.wohler_exp", "towerse.wohler_exp_mat")
            self.connect("materials.wohler_intercept", "towerse.wohler_A_mat")
            self.connect("materials.unit_cost", "towerse.unit_cost_mat")
            self.connect("costs.labor_rate", "towerse.labor_cost_rate")
            self.connect("costs.painting_rate", "towerse.painting_cost_rate")

        if modeling_options["flags"]["monopile"] or modeling_options["flags"]["jacket"]:
            self.connect("materials.E", "fixedse.E_mat")
            self.connect("materials.G", "fixedse.G_mat")
            self.connect("materials.rho", "fixedse.rho_mat")
            self.connect("materials.name", "fixedse.material_names")
            self.connect("materials.unit_cost", "fixedse.unit_cost_mat")
            self.connect("costs.labor_rate", "fixedse.labor_cost_rate")
            self.connect("costs.painting_rate", "fixedse.painting_cost_rate")
            self.connect("materials.sigma_y", "fixedse.sigma_y_mat")
            if modeling_options["flags"]["tower"]:
                self.connect("towerse.tower_mass", "fixedse.tower_mass")
                self.connect("towerse.tower_cost", "fixedse.tower_cost")
                self.connect("towerse.turbine_mass", "fixedse.turbine_mass")
                self.connect("towerse.turbine_center_of_mass", "fixedse.turbine_cg")
                self.connect("towerse.turbine_I_base", "fixedse.turbine_I")
                self.connect("towerse.tower.turbine_F", "fixedse.turbine_F")
                self.connect("towerse.tower.turbine_M", "fixedse.turbine_M")
                self.connect("tower.diameter", "fixedse.tower_base_diameter", src_indices=[0])
                self.connect("tower_grid.foundation_height", "fixedse.tower_foundation_height")

        if modeling_options["flags"]["monopile"]:
            if modeling_options["flags"]["blade"]:
                self.connect("rotorse.rp.gust.V_gust", "fixedse.env.Uref")
            self.connect("high_level_tower_props.hub_height", "fixedse.wind_reference_height")
            self.connect("env.rho_air", "fixedse.rho_air")
            self.connect("env.mu_air", "fixedse.mu_air")
            self.connect("env.shear_exp", "fixedse.shearExp")
            self.connect("env.water_depth", "fixedse.water_depth")
            self.connect("env.rho_water", "fixedse.rho_water")
            self.connect("env.mu_water", "fixedse.mu_water")
            if modeling_options["WISDEM"]["FixedBottomSE"]["soil_springs"]:
                self.connect("env.G_soil", "fixedse.G_soil")
                self.connect("env.nu_soil", "fixedse.nu_soil")
            self.connect("env.Hsig_wave", "fixedse.Hsig_wave")
            self.connect("env.Tsig_wave", "fixedse.Tsig_wave")
            self.connect("monopile.diameter", "fixedse.monopile_outer_diameter_in")
            self.connect("monopile.diameter", "fixedse.monopile_top_diameter", src_indices=[-1])
            self.connect("monopile.foundation_height", "fixedse.monopile_foundation_height")
            self.connect("monopile.outfitting_factor", "fixedse.outfitting_factor_in")
            self.connect("monopile.height", "fixedse.monopile_height")
            self.connect("monopile.s", "fixedse.monopile_s")
            self.connect("monopile.layer_thickness", "fixedse.monopile_layer_thickness")
            self.connect("monopile.layer_mat", "fixedse.monopile_layer_materials")
            self.connect("materials.Xt", "fixedse.sigma_ult_mat")
            self.connect("materials.wohler_exp", "fixedse.wohler_exp_mat")
            self.connect("materials.wohler_intercept", "fixedse.wohler_A_mat")
            self.connect("monopile.transition_piece_cost", "fixedse.transition_piece_cost")
            self.connect("monopile.transition_piece_mass", "fixedse.transition_piece_mass")
            self.connect("monopile.gravity_foundation_mass", "fixedse.gravity_foundation_mass")
            if modeling_options["flags"]["tower"]:
                self.connect("towerse.nodes_xyz", "fixedse.tower_xyz")
                self.connect("towerse.d_full", "fixedse.tower_d_full")
                self.connect("towerse.t_full", "fixedse.tower_t_full")
                self.connect("towerse.sigma_y_full", "fixedse.tower_sigma_y_full")
                self.connect("towerse.qdyn", "fixedse.tower_qdyn")
                self.connect("tower_grid.height", "fixedse.tower_bending_height")

                for var in ["A", "Asx", "Asy", "Ixx", "Iyy", "J0", "rho", "E", "G"]:
                    self.connect(f"towerse.section_{var}", f"fixedse.tower_{var}")
                for var in ["Px", "Py", "Pz"]:
                    self.connect(f"towerse.{var}", f"fixedse.tower_{var}")
            if modeling_options["flags"]["nacelle"]:
                self.connect("drivese.base_F", "fixedse.monopile.rna_F")
                self.connect("drivese.base_M", "fixedse.monopile.rna_M")
                self.connect("drivese.rna_I_TT", "fixedse.rna_I")
                self.connect("drivese.rna_cm", "fixedse.rna_cg")
                self.connect("drivese.rna_mass", "fixedse.rna_mass")

        if modeling_options["flags"]["jacket"]:
            self.connect("jacket.transition_piece_cost", "fixedse.transition_piece_cost")
            self.connect("jacket.transition_piece_mass", "fixedse.transition_piece_mass")
            self.connect("jacket.foot_head_ratio", "fixedse.foot_head_ratio")
            self.connect("jacket.r_head", "fixedse.r_head")
            self.connect("jacket.height", "fixedse.height")
            self.connect("jacket.leg_diameter", "fixedse.leg_diameter")
            self.connect("jacket.leg_thickness", "fixedse.leg_thickness")
            self.connect("jacket.brace_diameters", "fixedse.brace_diameters")
            self.connect("jacket.brace_thicknesses", "fixedse.brace_thicknesses")
            self.connect("jacket.bay_spacing", "fixedse.bay_spacing")

        if modeling_options["flags"]["floating"]:
            self.connect("env.rho_water", "floatingse.rho_water")
            self.connect("env.water_depth", "floatingse.water_depth")
            self.connect("env.mu_water", "floatingse.mu_water")
            self.connect("env.Hsig_wave", "floatingse.Hsig_wave")
            self.connect("env.Tsig_wave", "floatingse.Tsig_wave")
            self.connect("env.rho_air", "floatingse.rho_air")
            self.connect("env.mu_air", "floatingse.mu_air")
            self.connect("env.shear_exp", "floatingse.shearExp")
            self.connect("high_level_tower_props.hub_height", "floatingse.wind_reference_height")
            if modeling_options["flags"]["blade"]:
                self.connect("rotorse.rp.gust.V_gust", "floatingse.env.Uref")
            self.connect("materials.name", "floatingse.material_names")
            self.connect("materials.E", "floatingse.E_mat")
            self.connect("materials.G", "floatingse.G_mat")
            self.connect("materials.rho", "floatingse.rho_mat")
            self.connect("materials.sigma_y", "floatingse.sigma_y_mat")
            self.connect("materials.Xt", "floatingse.sigma_ult_mat")
            self.connect("materials.wohler_exp", "floatingse.wohler_exp_mat")
            self.connect("materials.wohler_intercept", "floatingse.wohler_A_mat")
            self.connect("materials.unit_cost", "floatingse.unit_cost_mat")
            self.connect("costs.labor_rate", "floatingse.labor_cost_rate")
            self.connect("costs.painting_rate", "floatingse.painting_cost_rate")
            self.connect("floating.transition_node", "floatingse.transition_node")
            self.connect("floating.transition_piece_mass", "floatingse.transition_piece_mass")
            self.connect("floating.transition_piece_cost", "floatingse.transition_piece_cost")
            if modeling_options["flags"]["tower"]:
                self.connect("towerse.turbine_mass", "floatingse.turbine_mass")
                self.connect("towerse.turbine_center_of_mass", "floatingse.turbine_cg")
                self.connect("towerse.turbine_I_base", "floatingse.turbine_I")
                self.connect("towerse.tower.turbine_F", "floatingse.turbine_F")
                self.connect("towerse.tower.turbine_M", "floatingse.turbine_M")
                self.connect("towerse.nodes_xyz", "floatingse.tower_xyz")
                for var in ["A", "Asx", "Asy", "Ixx", "Iyy", "J0", "rho", "E", "G"]:
                    self.connect(f"towerse.section_{var}", f"floatingse.tower_{var}")
            if modeling_options["flags"]["nacelle"]:
                self.connect("drivese.rna_I_TT", "floatingse.rna_I")
                self.connect("drivese.rna_cm", "floatingse.rna_cg")
                self.connect("drivese.rna_mass", "floatingse.rna_mass")

            # Individual member connections
            for k, kname in enumerate(modeling_options["floating"]["members"]["name"]):
                idx = modeling_options["floating"]["members"]["name2idx"][kname]
                self.connect(f"floating.memgrid{idx}.outer_diameter", f"floatingse.member{k}.outer_diameter_in")
                self.connect(f"floating.memgrid{idx}.layer_thickness", f"floatingse.member{k}.layer_thickness")
                self.connect(f"floating.memgrp{idx}.outfitting_factor", f"floatingse.member{k}.outfitting_factor_in")
                self.connect(f"floating.memgrp{idx}.s", f"floatingse.member{k}.s_in")

                for var in [
                    "layer_materials",
                    "bulkhead_grid",
                    "bulkhead_thickness",
                    "ballast_grid",
                    "ballast_volume",
                    "ballast_materials",
                    "grid_axial_joints",
                    "ring_stiffener_web_height",
                    "ring_stiffener_web_thickness",
                    "ring_stiffener_flange_width",
                    "ring_stiffener_flange_thickness",
                    "ring_stiffener_spacing",
                    "axial_stiffener_web_height",
                    "axial_stiffener_web_thickness",
                    "axial_stiffener_flange_width",
                    "axial_stiffener_flange_thickness",
                    "axial_stiffener_spacing",
                ]:
                    self.connect(f"floating.memgrp{idx}.{var}", f"floatingse.member{k}.{var}")

                for var in ["joint1", "joint2"]:
                    self.connect(f"floating.member_{kname}:{var}", f"floatingse.member{k}:{var}")

                for var in ["s_ghost1", "s_ghost2"]:
                    self.connect(f"floating.member_{kname}:{var}", f"floatingse.member{k}.{var}")

            # Mooring connections
            self.connect("mooring.unstretched_length", "floatingse.line_length", src_indices=[0])
            for var in [
                "fairlead",
                "fairlead_radius",
                "anchor_radius",
                "anchor_mass",
                "anchor_cost",
                "anchor_max_vertical_load",
                "anchor_max_lateral_load",
                "line_diameter",
                "line_mass_density_coeff",
                "line_stiffness_coeff",
                "line_breaking_load_coeff",
                "line_cost_rate_coeff",
            ]:
                self.connect(f"mooring.{var}", f"floatingse.{var}", src_indices=[0])

        # Connections to turbine constraints
        if modeling_options["flags"]["blade"] and modeling_options["flags"]["tower"]:
            self.connect("configuration.rotor_orientation", "tcons.rotor_orientation")
            self.connect("rotorse.rs.tip_pos.tip_deflection", "tcons.tip_deflection")
            self.connect("blade.high_level_blade_props.rotor_radius", "tcons.Rtip")
            self.connect("blade.high_level_blade_props.blade_ref_axis", "tcons.ref_axis_blade")
            self.connect("hub.cone", "tcons.precone")
            self.connect("nacelle.uptilt", "tcons.tilt")
            self.connect("nacelle.overhang", "tcons.overhang")
            self.connect("high_level_tower_props.tower_ref_axis", "tcons.ref_axis_tower")
            self.connect("tower.diameter", "tcons.d_full")
            if modeling_options["flags"]["floating"]:
                self.connect("floatingse.structural_frequencies", "tcons.tower_freq", src_indices=[0])
            else:
                self.connect("towerse.tower.structural_frequencies", "tcons.tower_freq", src_indices=[0])
            self.connect("configuration.n_blades", "tcons.blade_number")
            self.connect("rotorse.rp.powercurve.rated_Omega", "tcons.rated_Omega")

        # Connections to turbine capital cost
        self.connect("configuration.n_blades", "tcc.blade_number")
        self.connect("configuration.rated_power", "tcc.machine_rating")
        if modeling_options["flags"]["blade"]:
            self.connect("rotorse.blade_mass", "tcc.blade_mass")
            self.connect("rotorse.total_bc.total_blade_cost", "tcc.blade_cost_external")

        if modeling_options["flags"]["nacelle"]:
            self.connect("drivese.hub_mass", "tcc.hub_mass")
            self.connect("drivese.pitch_mass", "tcc.pitch_system_mass")
            self.connect("drivese.spinner_mass", "tcc.spinner_mass")
            self.connect("drivese.lss_mass", "tcc.lss_mass")
            self.connect("drivese.mean_bearing_mass", "tcc.main_bearing_mass")
            self.connect("drivese.gearbox_mass", "tcc.gearbox_mass")
            self.connect("drivese.hss_mass", "tcc.hss_mass")
            self.connect("drivese.brake_mass", "tcc.brake_mass")
            self.connect("drivese.generator_mass", "tcc.generator_mass")
            self.connect("drivese.total_bedplate_mass", "tcc.bedplate_mass")
            self.connect("drivese.yaw_mass", "tcc.yaw_mass")
            self.connect("drivese.converter_mass", "tcc.converter_mass")
            self.connect("drivese.transformer_mass", "tcc.transformer_mass")
            self.connect("drivese.hvac_mass", "tcc.hvac_mass")
            self.connect("drivese.cover_mass", "tcc.cover_mass")
            self.connect("drivese.platform_mass", "tcc.platforms_mass")

            if modeling_options["flags"]["generator"]:
                self.connect("drivese.generator_cost", "tcc.generator_cost_external")

        if modeling_options["flags"]["tower"]:
            self.connect("towerse.tower_mass", "tcc.tower_mass")
            self.connect("towerse.tower_cost", "tcc.tower_cost_external")

        self.connect("costs.blade_mass_cost_coeff", "tcc.blade_mass_cost_coeff")
        self.connect("costs.hub_mass_cost_coeff", "tcc.hub_mass_cost_coeff")
        self.connect("costs.pitch_system_mass_cost_coeff", "tcc.pitch_system_mass_cost_coeff")
        self.connect("costs.spinner_mass_cost_coeff", "tcc.spinner_mass_cost_coeff")
        self.connect("costs.lss_mass_cost_coeff", "tcc.lss_mass_cost_coeff")
        self.connect("costs.bearing_mass_cost_coeff", "tcc.bearing_mass_cost_coeff")
        self.connect("costs.gearbox_torque_cost", "tcc.gearbox_torque_cost")
        self.connect("costs.hss_mass_cost_coeff", "tcc.hss_mass_cost_coeff")
        self.connect("costs.generator_mass_cost_coeff", "tcc.generator_mass_cost_coeff")
        self.connect("costs.bedplate_mass_cost_coeff", "tcc.bedplate_mass_cost_coeff")
        self.connect("costs.yaw_mass_cost_coeff", "tcc.yaw_mass_cost_coeff")
        self.connect("costs.converter_mass_cost_coeff", "tcc.converter_mass_cost_coeff")
        self.connect("costs.transformer_mass_cost_coeff", "tcc.transformer_mass_cost_coeff")
        self.connect("costs.hvac_mass_cost_coeff", "tcc.hvac_mass_cost_coeff")
        self.connect("costs.cover_mass_cost_coeff", "tcc.cover_mass_cost_coeff")
        self.connect("costs.elec_connec_machine_rating_cost_coeff", "tcc.elec_connec_machine_rating_cost_coeff")
        self.connect("costs.platforms_mass_cost_coeff", "tcc.platforms_mass_cost_coeff")
        self.connect("costs.tower_mass_cost_coeff", "tcc.tower_mass_cost_coeff")
        self.connect("costs.controls_machine_rating_cost_coeff", "tcc.controls_machine_rating_cost_coeff")
        self.connect("costs.crane_cost", "tcc.crane_cost")

        # Final component for inverse design objective
        if opt_options["inverse_design"]:
            self.add_subsystem("inverse_design", InverseDesign(opt_options=opt_options))

            for name in opt_options["inverse_design"]:
                indices = opt_options["inverse_design"][name]["indices"]
                short_name = name.replace(".", "_")
                self.connect(name, f"inverse_design.{short_name}", src_indices=indices)


class InverseDesign(om.ExplicitComponent):
    """
    Component that takes in an arbitrary set of user-defined inputs and computes
    the root-mean-square (RMS) difference between the values in the model and
    a set of reference values.

    This is useful for inverse design problems where we are trying to design a
    wind turbine system that has a certain set of properties. Specifically, we
    might be trying to match performance values from a report by allowing the
    optimizer to select the design variable values that most closely produce a
    system that has those properties.

    """

    def initialize(self):
        self.options.declare("opt_options")

    def setup(self):
        opt_options = self.options["opt_options"]

        # Loop through all of the keys in the inverse_design definition
        for name in opt_options["inverse_design"]:
            item = opt_options["inverse_design"][name]

            indices = item["indices"]

            # Grab the short name for each parameter to match
            short_name = name.replace(".", "_")

            # Only apply units if they're provided by the user
            if "units" in item:
                units = item["units"]
            else:
                units = None

            self.add_input(
                short_name,
                val=np.zeros(len(indices)),
                units=units,
            )

        # Create a singular output called objective
        self.add_output(
            "objective",
            val=0.0,
        )

    def compute(self, inputs, outputs):
        opt_options = self.options["opt_options"]

        total = 0.0
        # Loop through all of the keys in the inverse_design definition
        for name in opt_options["inverse_design"]:
            item = opt_options["inverse_design"][name]

            # Grab the short name for each parameter to match
            short_name = name.replace(".", "_")

            # Grab the reference value provided by the user
            ref_value = item["ref_value"]

            # Compute the mean square difference between the parameter
            # value outputted from the model and the reference value. Sum this
            # to `total` to get the total across all parameters
            total += np.sum(((inputs[short_name] - ref_value) / (np.abs(ref_value) + 1.0)) ** 2)

        # Take the square root of the total
        rms_total = np.sqrt(total)
        outputs["objective"] = rms_total


class WindPark(om.Group):
    # Openmdao group to run the cost analysis of a wind park

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]

        self.add_subsystem("wt", WT_RNTA(modeling_options=modeling_options, opt_options=opt_options), promotes=["*"])
        if modeling_options["WISDEM"]["BOS"]["flag"]:
            if modeling_options["flags"]["offshore"]:
                self.add_subsystem(
                    "orbit",
                    Orbit(
                        floating=modeling_options["flags"]["floating"],
                        jacket=modeling_options["flags"]["jacket"],
                        jacket_legs=modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"],
                    ),
                )
            else:
                self.add_subsystem("landbosse", LandBOSSE())

        if modeling_options["flags"]["blade"]:
            self.add_subsystem("financese", PlantFinance(verbosity=modeling_options["General"]["verbosity"]))
            self.add_subsystem(
                "outputs_2_screen", Outputs_2_Screen(modeling_options=modeling_options, opt_options=opt_options)
            )

        # BOS inputs
        if modeling_options["WISDEM"]["BOS"]["flag"]:
            if modeling_options["flags"]["offshore"]:
                # Inputs into ORBIT
                self.connect("configuration.rated_power", "orbit.turbine_rating")
                self.connect("env.water_depth", "orbit.site_depth")
                self.connect("costs.turbine_number", "orbit.number_of_turbines")
                self.connect("configuration.n_blades", "orbit.number_of_blades")
                self.connect("high_level_tower_props.hub_height", "orbit.hub_height")
                self.connect("blade.high_level_blade_props.rotor_diameter", "orbit.turbine_rotor_diameter")
                self.connect("tower_grid.height", "orbit.tower_length")
                if modeling_options["flags"]["tower"]:
                    self.connect("towerse.tower_mass", "orbit.tower_mass")
                if modeling_options["flags"]["monopile"]:
                    self.connect("fixedse.monopile_mass", "orbit.monopile_mass")
                    self.connect("fixedse.monopile_cost", "orbit.monopile_cost")
                    self.connect("monopile.height", "orbit.monopile_length")
                    self.connect("monopile.transition_piece_mass", "orbit.transition_piece_mass")
                    self.connect("monopile.transition_piece_cost", "orbit.transition_piece_cost")
                    self.connect("monopile.diameter", "orbit.monopile_diameter", src_indices=[0])
                elif modeling_options["flags"]["jacket"]:
                    self.connect("fixedse.r_foot", "orbit.jacket_r_foot")
                    self.connect("jacket.height", "orbit.jacket_length")
                    self.connect("fixedse.jacket_mass", "orbit.jacket_mass")
                    self.connect("fixedse.jacket_cost", "orbit.jacket_cost")
                    self.connect("jacket.transition_piece_mass", "orbit.transition_piece_mass")
                    self.connect("jacket.transition_piece_cost", "orbit.transition_piece_cost")
                elif modeling_options["flags"]["floating"]:
                    self.connect("mooring.n_lines", "orbit.num_mooring_lines")
                    self.connect("floatingse.line_mass", "orbit.mooring_line_mass", src_indices=[0])
                    self.connect("mooring.line_diameter", "orbit.mooring_line_diameter", src_indices=[0])
                    self.connect("mooring.unstretched_length", "orbit.mooring_line_length", src_indices=[0])
                    self.connect("mooring.anchor_mass", "orbit.anchor_mass", src_indices=[0])
                    self.connect("floating.transition_piece_mass", "orbit.transition_piece_mass")
                    self.connect("floating.transition_piece_cost", "orbit.transition_piece_cost")
                    self.connect("floatingse.platform_cost", "orbit.floating_substructure_cost")
                if modeling_options["flags"]["nacelle"]:
                    self.connect("drivese.nacelle_mass", "orbit.nacelle_mass")
                self.connect("rotorse.blade_mass", "orbit.blade_mass")
                self.connect("tcc.turbine_cost_kW", "orbit.turbine_capex")
                self.connect("rotorse.wt_class.V_mean", "orbit.site_mean_windspeed")
                self.connect("rotorse.rp.powercurve.rated_V", "orbit.turbine_rated_windspeed")
                self.connect("bos.plant_turbine_spacing", "orbit.plant_turbine_spacing")
                self.connect("bos.plant_row_spacing", "orbit.plant_row_spacing")
                self.connect("bos.commissioning_pct", "orbit.commissioning_pct")
                self.connect("bos.decommissioning_pct", "orbit.decommissioning_pct")
                self.connect("bos.distance_to_substation", "orbit.plant_substation_distance")
                self.connect("bos.distance_to_interconnection", "orbit.interconnection_distance")
                self.connect("bos.site_distance", "orbit.site_distance")
                self.connect("bos.distance_to_landfall", "orbit.site_distance_to_landfall")
                self.connect("bos.port_cost_per_month", "orbit.port_cost_per_month")
                self.connect("bos.site_auction_price", "orbit.site_auction_price")
                self.connect("bos.site_assessment_plan_cost", "orbit.site_assessment_plan_cost")
                self.connect("bos.site_assessment_cost", "orbit.site_assessment_cost")
                self.connect("bos.construction_operations_plan_cost", "orbit.construction_operations_plan_cost")
                self.connect("bos.boem_review_cost", "orbit.boem_review_cost")
                self.connect("bos.design_install_plan_cost", "orbit.design_install_plan_cost")
            else:
                # Inputs into LandBOSSE
                self.connect("high_level_tower_props.hub_height", "landbosse.hub_height_meters")
                self.connect("costs.turbine_number", "landbosse.num_turbines")
                self.connect("configuration.rated_power", "landbosse.turbine_rating_MW")
                self.connect("env.shear_exp", "landbosse.wind_shear_exponent")
                self.connect("blade.high_level_blade_props.rotor_diameter", "landbosse.rotor_diameter_m")
                self.connect("configuration.n_blades", "landbosse.number_of_blades")
                if modeling_options["flags"]["blade"]:
                    self.connect("rotorse.rp.powercurve.rated_T", "landbosse.rated_thrust_N")
                    self.connect("rotorse.wt_class.V_extreme50", "landbosse.gust_velocity_m_per_s")
                    self.connect("blade.compute_coord_xy_dim.projected_area", "landbosse.blade_surface_area")
                self.connect("towerse.tower_mass", "landbosse.tower_mass")
                if modeling_options["flags"]["nacelle"]:
                    self.connect("drivese.nacelle_mass", "landbosse.nacelle_mass")
                    self.connect("drivese.hub_system_mass", "landbosse.hub_mass")
                self.connect("rotorse.blade_mass", "landbosse.blade_mass")
                self.connect("tower_grid.foundation_height", "landbosse.foundation_height")
                self.connect("bos.plant_turbine_spacing", "landbosse.turbine_spacing_rotor_diameters")
                self.connect("bos.plant_row_spacing", "landbosse.row_spacing_rotor_diameters")
                self.connect("bos.commissioning_pct", "landbosse.commissioning_pct")
                self.connect("bos.decommissioning_pct", "landbosse.decommissioning_pct")
                self.connect("bos.distance_to_substation", "landbosse.trench_len_to_substation_km")
                self.connect("bos.distance_to_interconnection", "landbosse.distance_to_interconnect_mi")
                self.connect("bos.interconnect_voltage", "landbosse.interconnect_voltage_kV")

        # Inputs to plantfinancese from wt group
        if modeling_options["flags"]["blade"]:
            self.connect("rotorse.rp.AEP", "financese.turbine_aep")
            self.connect("tcc.turbine_cost_kW", "financese.tcc_per_kW")

            if modeling_options["flags"]["bos"]:
                if modeling_options["flags"]["offshore"]:
                    self.connect("orbit.total_capex_kW", "financese.bos_per_kW")
                else:
                    self.connect("landbosse.bos_capex_kW", "financese.bos_per_kW")
            else:
                self.connect("costs.bos_per_kW", "financese.bos_per_kW")

            # Inputs to plantfinancese from input yaml
            if modeling_options["flags"]["control"]:
                self.connect("configuration.rated_power", "financese.machine_rating")
            self.connect("costs.turbine_number", "financese.turbine_number")
            self.connect("costs.opex_per_kW", "financese.opex_per_kW")
            self.connect("costs.offset_tcc_per_kW", "financese.offset_tcc_per_kW")
            self.connect("costs.wake_loss_factor", "financese.wake_loss_factor")
            self.connect("costs.fixed_charge_rate", "financese.fixed_charge_rate")
            self.connect("costs.electricity_price", "financese.electricity_price")
            self.connect("costs.reserve_margin_price", "financese.reserve_margin_price")
            self.connect("costs.capacity_credit", "financese.capacity_credit")
            self.connect("costs.benchmark_price", "financese.benchmark_price")

        # Connections to outputs to screen
        if modeling_options["flags"]["blade"]:
            self.connect("rotorse.rp.AEP", "outputs_2_screen.aep")
            self.connect("financese.lcoe", "outputs_2_screen.lcoe")
            self.connect("rotorse.blade_mass", "outputs_2_screen.blade_mass")
            self.connect("rotorse.rs.tip_pos.tip_deflection", "outputs_2_screen.tip_deflection")
