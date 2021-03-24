import numpy as np
import openmdao.api as om
from wisdem.towerse.tower import TowerSE
from wisdem.floatingse.floating import FloatingSE
from wisdem.rotorse.rotor_power import RotorPower, NoStallConstraint
from wisdem.glue_code.gc_RunTools import Outputs_2_Screen, Convergence_Trends_Opt
from wisdem.commonse.turbine_class import TurbineClass
from wisdem.drivetrainse.drivetrain import DrivetrainSE
from wisdem.rotorse.rotor_structure import RotorStructure
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from wisdem.ccblade.ccblade_component import CCBladeTwist
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
        opt_options = self.options["opt_options"]

        if modeling_options["flags"]["blade"] and modeling_options["flags"]["nacelle"]:
            self.linear_solver = lbgs = om.LinearBlockGS()
            self.nonlinear_solver = nlbgs = om.NonlinearBlockGS()
            nlbgs.options["maxiter"] = 5
            nlbgs.options["atol"] = 1e-2
            nlbgs.options["rtol"] = 1e-8
            nlbgs.options["iprint"] = 0

        # Analysis components
        self.add_subsystem(
            "wt_init",
            WindTurbineOntologyOpenMDAO(modeling_options=modeling_options, opt_options=opt_options),
            promotes=["*"],
        )
        if modeling_options["flags"]["blade"]:
            self.add_subsystem(
                "ccblade", CCBladeTwist(modeling_options=modeling_options, opt_options=opt_options)
            )  # Run standalong CCBlade and possibly determine optimal twist from user-defined margin to stall
            self.add_subsystem("wt_class", TurbineClass())
            self.add_subsystem("re", RotorElasticity(modeling_options=modeling_options, opt_options=opt_options))
            self.add_subsystem("rp", RotorPower(modeling_options=modeling_options))  # Aero analysis
            self.add_subsystem("stall_check", NoStallConstraint(modeling_options=modeling_options))
            self.add_subsystem(
                "rs", RotorStructure(modeling_options=modeling_options, opt_options=opt_options, freq_run=False)
            )
        if modeling_options["flags"]["nacelle"]:
            self.add_subsystem("drivese", DrivetrainSE(modeling_options=modeling_options, n_dlcs=1))
        if modeling_options["flags"]["tower"] and not modeling_options["flags"]["floating"]:
            self.add_subsystem("towerse", TowerSE(modeling_options=modeling_options))
        if modeling_options["flags"]["floating"]:
            self.add_subsystem("floatingse", FloatingSE(modeling_options=modeling_options))
        if modeling_options["flags"]["blade"] and modeling_options["flags"]["tower"]:
            self.add_subsystem("tcons", TurbineConstraints(modeling_options=modeling_options))
        self.add_subsystem("tcc", Turbine_CostsSE_2015(verbosity=modeling_options["General"]["verbosity"]))

        if modeling_options["flags"]["blade"]:
            n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]

            # Conncetions to ccblade
            self.connect("blade.pa.chord_param", "ccblade.chord")
            self.connect("blade.pa.twist_param", "ccblade.twist")
            self.connect("blade.opt_var.s_opt_chord", "ccblade.s_opt_chord")
            self.connect("blade.opt_var.s_opt_twist", "ccblade.s_opt_twist")
            self.connect("assembly.r_blade", "ccblade.r")
            self.connect("assembly.rotor_radius", "ccblade.Rtip")
            self.connect("hub.radius", "ccblade.Rhub")
            self.connect("blade.interp_airfoils.r_thick_interp", "ccblade.rthick")
            self.connect("airfoils.aoa", "ccblade.airfoils_aoa")
            self.connect("airfoils.Re", "ccblade.airfoils_Re")
            self.connect("blade.interp_airfoils.cl_interp", "ccblade.airfoils_cl")
            self.connect("blade.interp_airfoils.cd_interp", "ccblade.airfoils_cd")
            self.connect("blade.interp_airfoils.cm_interp", "ccblade.airfoils_cm")
            self.connect("assembly.hub_height", "ccblade.hub_height")
            self.connect("hub.cone", "ccblade.precone")
            self.connect("nacelle.uptilt", "ccblade.tilt")
            self.connect("assembly.blade_ref_axis", "ccblade.precurve", src_indices=[(i, 0) for i in np.arange(n_span)])
            self.connect("assembly.blade_ref_axis", "ccblade.precurveTip", src_indices=[(-1, 0)])
            self.connect("assembly.blade_ref_axis", "ccblade.presweep", src_indices=[(i, 1) for i in np.arange(n_span)])
            self.connect("assembly.blade_ref_axis", "ccblade.presweepTip", src_indices=[(-1, 1)])
            self.connect("configuration.n_blades", "ccblade.nBlades")
            if modeling_options["flags"]["control"]:
                self.connect("control.rated_pitch", "ccblade.pitch")
            self.connect("control.rated_TSR", "ccblade.tsr")
            self.connect("env.rho_air", "ccblade.rho")
            self.connect("env.mu_air", "ccblade.mu")
            self.connect("env.shear_exp", "ccblade.shearExp")

            # Connections to wind turbine class
            self.connect("configuration.ws_class", "wt_class.turbine_class")

            # Connections from blade aero parametrization to other modules
            self.connect("ccblade.theta", ["re.theta", "rs.theta"])
            self.connect("blade.pa.chord_param", "re.chord")
            self.connect("blade.pa.chord_param", ["rs.chord"])
            if modeling_options["flags"]["blade"]:
                self.connect("ccblade.theta", "rp.theta")
                self.connect("blade.pa.chord_param", "rp.chord")
            self.connect("configuration.n_blades", "rs.constr.blade_number")

            # Connections from blade struct parametrization to rotor elasticity
            self.connect("blade.ps.layer_thickness_param", "re.precomp.layer_thickness")

            # Connections to rotor elastic and frequency analysis
            self.connect("nacelle.uptilt", "re.precomp.uptilt")
            self.connect("configuration.n_blades", "re.precomp.n_blades")
            self.connect("assembly.r_blade", "re.r")
            self.connect("blade.outer_shape_bem.pitch_axis", "re.precomp.pitch_axis")
            self.connect("blade.interp_airfoils.coord_xy_interp", "re.precomp.coord_xy_interp")
            self.connect("blade.internal_structure_2d_fem.layer_start_nd", "re.precomp.layer_start_nd")
            self.connect("blade.internal_structure_2d_fem.layer_end_nd", "re.precomp.layer_end_nd")
            self.connect("blade.internal_structure_2d_fem.layer_web", "re.precomp.layer_web")
            self.connect("blade.internal_structure_2d_fem.definition_layer", "re.precomp.definition_layer")
            self.connect("blade.internal_structure_2d_fem.web_start_nd", "re.precomp.web_start_nd")
            self.connect("blade.internal_structure_2d_fem.web_end_nd", "re.precomp.web_end_nd")
            self.connect("blade.internal_structure_2d_fem.joint_position", "re.precomp.joint_position")
            self.connect("blade.internal_structure_2d_fem.joint_mass", "re.precomp.joint_mass")
            self.connect("blade.internal_structure_2d_fem.joint_cost", "re.precomp.joint_cost")
            self.connect("materials.name", "re.precomp.mat_name")
            self.connect("materials.orth", "re.precomp.orth")
            self.connect("materials.E", "re.precomp.E")
            self.connect("materials.G", "re.precomp.G")
            self.connect("materials.nu", "re.precomp.nu")
            self.connect("materials.rho", "re.precomp.rho")
            self.connect("materials.component_id", "re.precomp.component_id")
            self.connect("materials.unit_cost", "re.precomp.unit_cost")
            self.connect("materials.waste", "re.precomp.waste")
            self.connect("materials.rho_fiber", "re.precomp.rho_fiber")
            self.connect("materials.rho_area_dry", "re.precomp.rho_area_dry")
            self.connect("materials.ply_t", "re.precomp.ply_t")
            self.connect("materials.fvf", "re.precomp.fvf")
            self.connect("materials.fwf", "re.precomp.fwf")
            self.connect("materials.roll_mass", "re.precomp.roll_mass")

            # Conncetions to rail transport module
            if opt_options["constraints"]["blade"]["rail_transport"]["flag"]:
                self.connect("blade.outer_shape_bem.pitch_axis", "re.rail.pitch_axis")
                self.connect("assembly.blade_ref_axis", "re.rail.blade_ref_axis")
                self.connect("blade.interp_airfoils.coord_xy_interp", "re.rail.coord_xy_interp")

            # Connections from blade struct parametrization to rotor load anlysis
            self.connect("blade.opt_var.s_opt_spar_cap_ss", "rs.constr.s_opt_spar_cap_ss")
            self.connect("blade.opt_var.s_opt_spar_cap_ps", "rs.constr.s_opt_spar_cap_ps")

            # Connection from ra to rs for the rated conditions
            # self.connect('rp.powercurve.rated_V',        'rs.aero_rated.V_load')
            self.connect("rp.gust.V_gust", ["rs.aero_gust.V_load", "rs.aero_hub_loads.V_load"])
            self.connect("env.shear_exp", ["rp.powercurve.shearExp", "rs.aero_gust.shearExp"])
            self.connect(
                "rp.powercurve.rated_Omega",
                ["rs.Omega_load", "rs.tot_loads_gust.aeroloads_Omega", "rs.constr.rated_Omega"],
            )
            self.connect("rp.powercurve.rated_pitch", ["rs.pitch_load", "rs.tot_loads_gust.aeroloads_pitch"])

            # Connections to RotorPower
            self.connect("control.V_in", "rp.v_min")
            self.connect("control.V_out", "rp.v_max")
            self.connect("configuration.rated_power", "rp.rated_power")
            self.connect("control.minOmega", "rp.omega_min")
            self.connect("control.maxOmega", "rp.omega_max")
            self.connect("control.max_TS", "rp.control_maxTS")
            self.connect("control.rated_TSR", "rp.tsr_operational")
            self.connect("control.rated_pitch", "rp.control_pitch")
            self.connect("configuration.gearbox_type", "rp.drivetrainType")
            self.connect("nacelle.gearbox_efficiency", "rp.powercurve.gearbox_efficiency")
            if modeling_options["flags"]["nacelle"]:
                self.connect("drivese.lss_rpm", "rp.powercurve.lss_rpm")
                self.connect("drivese.generator_efficiency", "rp.powercurve.generator_efficiency")
            self.connect("assembly.r_blade", "rp.r")
            self.connect("hub.radius", "rp.Rhub")
            self.connect("assembly.rotor_radius", "rp.Rtip")
            self.connect("assembly.hub_height", "rp.hub_height")
            self.connect("hub.cone", "rp.precone")
            self.connect("nacelle.uptilt", "rp.tilt")
            self.connect("assembly.blade_ref_axis", "rp.precurve", src_indices=[(i, 0) for i in np.arange(n_span)])
            self.connect("assembly.blade_ref_axis", "rp.precurveTip", src_indices=[(-1, 0)])
            self.connect("assembly.blade_ref_axis", "rp.presweep", src_indices=[(i, 1) for i in np.arange(n_span)])
            self.connect("assembly.blade_ref_axis", "rp.presweepTip", src_indices=[(-1, 1)])
            self.connect("airfoils.aoa", "rp.airfoils_aoa")
            self.connect("airfoils.Re", "rp.airfoils_Re")
            self.connect("blade.interp_airfoils.cl_interp", "rp.airfoils_cl")
            self.connect("blade.interp_airfoils.cd_interp", "rp.airfoils_cd")
            self.connect("blade.interp_airfoils.cm_interp", "rp.airfoils_cm")
            self.connect("configuration.n_blades", "rp.nBlades")
            self.connect("env.rho_air", "rp.rho")
            self.connect("env.mu_air", "rp.mu")
            self.connect("wt_class.V_mean", "rp.cdf.xbar")
            self.connect("env.weibull_k", "rp.cdf.k")
            # Connections to rotorse-rs-gustetm
            self.connect("wt_class.V_mean", "rp.gust.V_mean")
            self.connect("configuration.turb_class", "rp.gust.turbulence_class")

            # Connections to the stall check
            self.connect("blade.outer_shape_bem.s", "stall_check.s")
            self.connect("airfoils.aoa", "stall_check.airfoils_aoa")
            self.connect("blade.interp_airfoils.cl_interp", "stall_check.airfoils_cl")
            self.connect("blade.interp_airfoils.cd_interp", "stall_check.airfoils_cd")
            self.connect("blade.interp_airfoils.cm_interp", "stall_check.airfoils_cm")
            if modeling_options["flags"]["blade"]:
                self.connect("rp.powercurve.aoa_regII", "stall_check.aoa_along_span")
            else:
                self.connect("ccblade.alpha", "stall_check.aoa_along_span")

            # Connections to rotor load analysis
            self.connect("blade.interp_airfoils.cl_interp", "rs.airfoils_cl")
            self.connect("blade.interp_airfoils.cd_interp", "rs.airfoils_cd")
            self.connect("blade.interp_airfoils.cm_interp", "rs.airfoils_cm")
            self.connect("airfoils.aoa", "rs.airfoils_aoa")
            self.connect("airfoils.Re", "rs.airfoils_Re")
            self.connect("assembly.rotor_radius", "rs.Rtip")
            self.connect("hub.radius", "rs.Rhub")
            self.connect("env.rho_air", "rs.rho")
            self.connect("env.mu_air", "rs.mu")
            self.connect("env.shear_exp", "rs.aero_hub_loads.shearExp")
            self.connect("assembly.hub_height", "rs.hub_height")
            self.connect("configuration.n_blades", "rs.nBlades")
            self.connect("assembly.r_blade", "rs.r")
            self.connect("hub.cone", "rs.precone")
            self.connect("nacelle.uptilt", "rs.tilt")

            self.connect("re.A", "rs.A")
            self.connect("re.EA", "rs.EA")
            self.connect("re.EIxx", "rs.EIxx")
            self.connect("re.EIyy", "rs.EIyy")
            self.connect("re.EIxy", "rs.EIxy")
            self.connect("re.GJ", "rs.GJ")
            self.connect("re.rhoA", "rs.rhoA")
            self.connect("re.rhoJ", "rs.rhoJ")
            self.connect("re.x_ec", "rs.x_ec")
            self.connect("re.y_ec", "rs.y_ec")
            self.connect("re.precomp.xu_strain_spar", "rs.xu_strain_spar")
            self.connect("re.precomp.xl_strain_spar", "rs.xl_strain_spar")
            self.connect("re.precomp.yu_strain_spar", "rs.yu_strain_spar")
            self.connect("re.precomp.yl_strain_spar", "rs.yl_strain_spar")
            self.connect("re.precomp.xu_strain_te", "rs.xu_strain_te")
            self.connect("re.precomp.xl_strain_te", "rs.xl_strain_te")
            self.connect("re.precomp.yu_strain_te", "rs.yu_strain_te")
            self.connect("re.precomp.yl_strain_te", "rs.yl_strain_te")
            self.connect("blade.outer_shape_bem.s", "rs.constr.s")

            self.connect("blade.internal_structure_2d_fem.d_f", "rs.brs.d_f")
            self.connect("blade.internal_structure_2d_fem.sigma_max", "rs.brs.sigma_max")
            self.connect("blade.pa.chord_param", "rs.brs.rootD", src_indices=[0])
            self.connect("blade.ps.layer_thickness_param", "rs.brs.layer_thickness")
            self.connect("blade.internal_structure_2d_fem.layer_start_nd", "rs.brs.layer_start_nd")
            self.connect("blade.internal_structure_2d_fem.layer_end_nd", "rs.brs.layer_end_nd")

            # Connections to rotorse-rc
            # self.connect('blade.length',                                    'rotorse.rc.blade_length')
            # self.connect('blade.outer_shape_bem.s',                         'rotorse.rc.s')
            # self.connect('blade.outer_shape_bem.pitch_axis',                'rotorse.rc.pitch_axis')
            # self.connect('blade.interp_airfoils.coord_xy_interp',           'rotorse.rc.coord_xy_interp')
            # self.connect('blade.internal_structure_2d_fem.layer_start_nd',  'rotorse.rc.layer_start_nd')
            # self.connect('blade.internal_structure_2d_fem.layer_end_nd',    'rotorse.rc.layer_end_nd')
            # self.connect('blade.internal_structure_2d_fem.layer_web',       'rotorse.rc.layer_web')
            # self.connect('blade.internal_structure_2d_fem.web_start_nd',    'rotorse.rc.web_start_nd')
            # self.connect('blade.internal_structure_2d_fem.web_end_nd',      'rotorse.rc.web_end_nd')
            # self.connect('materials.name',          'rotorse.rc.mat_name')
            # self.connect('materials.rho',           'rotorse.rc.rho')

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
            self.connect("hub.spinner_gust_ws", "drivese.spinner_gust_ws")

            self.connect("configuration.n_blades", "drivese.n_blades")

            self.connect("assembly.rotor_diameter", "drivese.rotor_diameter")
            self.connect("configuration.upwind", "drivese.upwind")
            self.connect("control.minOmega", "drivese.minimum_rpm")
            self.connect("rp.powercurve.rated_Omega", "drivese.rated_rpm")
            self.connect("rp.powercurve.rated_Q", "drivese.rated_torque")
            self.connect("configuration.rated_power", "drivese.machine_rating")
            if modeling_options["flags"]["tower"]:
                self.connect("tower.diameter", "drivese.D_top", src_indices=[-1])

            self.connect("rs.aero_hub_loads.Fxyz_hub_aero", "drivese.F_hub")
            self.connect("rs.aero_hub_loads.Mxyz_hub_aero", "drivese.M_hub")
            self.connect("rs.frame.root_M", "drivese.pitch_system.BRFM", src_indices=[1])

            self.connect("blade.pa.chord_param", "drivese.blade_root_diameter", src_indices=[0])
            self.connect("re.precomp.blade_mass", "drivese.blade_mass")
            self.connect("re.precomp.mass_all_blades", "drivese.blades_mass")
            self.connect("re.precomp.I_all_blades", "drivese.blades_I")

            self.connect("nacelle.distance_hub2mb", "drivese.L_h1")
            self.connect("nacelle.distance_mb2mb", "drivese.L_12")
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
                self.connect("nacelle.nose_diameter", "drivese.nose_diameter")  # only used in direct
                self.connect("nacelle.nose_wall_thickness", "drivese.nose_wall_thickness")  # only used in direct
                self.connect(
                    "nacelle.bedplate_wall_thickness", "drivese.bedplate_wall_thickness"
                )  # only used in direct
            else:
                self.connect("nacelle.hss_length", "drivese.L_hss")  # only used in geared
                self.connect("nacelle.hss_diameter", "drivese.hss_diameter")  # only used in geared
                self.connect("nacelle.hss_wall_thickness", "drivese.hss_wall_thickness")  # only used in geared
                self.connect("nacelle.hss_material", "drivese.hss_material")
                self.connect("nacelle.planet_numbers", "drivese.planet_numbers")  # only used in geared
                self.connect("nacelle.gear_configuration", "drivese.gear_configuration")  # only used in geared
                self.connect("nacelle.bedplate_flange_width", "drivese.bedplate_flange_width")  # only used in geared
                self.connect(
                    "nacelle.bedplate_flange_thickness", "drivese.bedplate_flange_thickness"
                )  # only used in geared
                self.connect("nacelle.bedplate_web_thickness", "drivese.bedplate_web_thickness")  # only used in geared

            self.connect("hub.hub_material", "drivese.hub_material")
            self.connect("hub.spinner_material", "drivese.spinner_material")
            self.connect("nacelle.lss_material", "drivese.lss_material")
            self.connect("nacelle.bedplate_material", "drivese.bedplate_material")
            self.connect("materials.name", "drivese.material_names")
            self.connect("materials.E", "drivese.E_mat")
            self.connect("materials.G", "drivese.G_mat")
            self.connect("materials.rho", "drivese.rho_mat")
            self.connect("materials.sigma_y", "drivese.sigma_y_mat")
            self.connect("materials.Xt", "drivese.Xt_mat")
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
                    self.connect("rp.powercurve.rated_mech", "drivese.generator.P_mech")

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
        if modeling_options["flags"]["tower"] and not modeling_options["flags"]["floating"]:
            if modeling_options["flags"]["nacelle"]:
                self.connect("drivese.base_F", "towerse.pre.rna_F")
                self.connect("drivese.base_M", "towerse.pre.rna_M")
                self.connect("drivese.rna_I_TT", "towerse.rna_I")
                self.connect("drivese.rna_cm", "towerse.rna_cg")
                self.connect("drivese.rna_mass", "towerse.rna_mass")
            if modeling_options["flags"]["blade"]:
                self.connect("rp.gust.V_gust", "towerse.wind.Uref")
            self.connect("assembly.hub_height", "towerse.wind_reference_height")
            self.connect("assembly.hub_height", "towerse.hub_height")
            self.connect("env.rho_air", "towerse.rho_air")
            self.connect("env.mu_air", "towerse.mu_air")
            self.connect("env.shear_exp", "towerse.shearExp")
            self.connect("tower_grid.foundation_height", "towerse.tower_foundation_height")
            self.connect("tower.diameter", "towerse.tower_outer_diameter_in")
            self.connect("tower_grid.height", "towerse.tower_height")
            self.connect("tower_grid.s", "towerse.tower_s")
            self.connect("tower.layer_thickness", "towerse.tower_layer_thickness")
            self.connect("tower.outfitting_factor", "towerse.tower_outfitting_factor")
            self.connect("tower.layer_mat", "towerse.tower_layer_materials")
            self.connect("materials.name", "towerse.material_names")
            self.connect("materials.E", "towerse.E_mat")
            self.connect("materials.G", "towerse.G_mat")
            self.connect("materials.rho", "towerse.rho_mat")
            self.connect("materials.sigma_y", "towerse.sigma_y_mat")
            self.connect("materials.unit_cost", "towerse.unit_cost_mat")
            self.connect("costs.labor_rate", "towerse.labor_cost_rate")
            self.connect("costs.painting_rate", "towerse.painting_cost_rate")
            if modeling_options["flags"]["monopile"]:
                self.connect("env.water_depth", "towerse.water_depth")
                self.connect("env.rho_water", "towerse.rho_water")
                self.connect("env.mu_water", "towerse.mu_water")
                if modeling_options["WISDEM"]["TowerSE"]["soil_springs"]:
                    self.connect("env.G_soil", "towerse.G_soil")
                    self.connect("env.nu_soil", "towerse.nu_soil")
                self.connect("env.Hsig_wave", "towerse.Hsig_wave")
                self.connect("env.Tsig_wave", "towerse.Tsig_wave")
                self.connect("monopile.diameter", "towerse.monopile_outer_diameter_in")
                self.connect("monopile.foundation_height", "towerse.monopile_foundation_height")
                self.connect("monopile.height", "towerse.monopile_height")
                self.connect("monopile.s", "towerse.monopile_s")
                self.connect("monopile.layer_thickness", "towerse.monopile_layer_thickness")
                self.connect("monopile.layer_mat", "towerse.monopile_layer_materials")
                self.connect("monopile.outfitting_factor", "towerse.monopile_outfitting_factor")
                self.connect("monopile.transition_piece_cost", "towerse.transition_piece_cost")
                self.connect("monopile.transition_piece_mass", "towerse.transition_piece_mass")
                self.connect("monopile.gravity_foundation_mass", "towerse.gravity_foundation_mass")

        if modeling_options["flags"]["floating"]:
            self.connect("env.rho_water", "floatingse.rho_water")
            self.connect("env.water_depth", "floatingse.water_depth")
            self.connect("env.mu_water", "floatingse.mu_water")
            self.connect("env.Hsig_wave", "floatingse.Hsig_wave")
            self.connect("env.Tsig_wave", "floatingse.Tsig_wave")
            self.connect("env.rho_air", "floatingse.rho_air")
            self.connect("env.mu_air", "floatingse.mu_air")
            self.connect("env.shear_exp", "floatingse.shearExp")
            self.connect("assembly.hub_height", "floatingse.zref")
            if modeling_options["flags"]["blade"]:
                self.connect("rp.gust.V_gust", "floatingse.Uref")
            self.connect("materials.name", "floatingse.material_names")
            self.connect("materials.E", "floatingse.E_mat")
            self.connect("materials.G", "floatingse.G_mat")
            self.connect("materials.rho", "floatingse.rho_mat")
            self.connect("materials.sigma_y", "floatingse.sigma_y_mat")
            self.connect("materials.unit_cost", "floatingse.unit_cost_mat")
            self.connect("costs.labor_rate", "floatingse.labor_cost_rate")
            self.connect("costs.painting_rate", "floatingse.painting_cost_rate")
            self.connect("tower.diameter", "floatingse.tower.outer_diameter_in")
            self.connect("tower_grid.s", "floatingse.tower.s")
            self.connect("tower.layer_thickness", "floatingse.tower.layer_thickness")
            self.connect("tower.outfitting_factor", "floatingse.tower.outfitting_factor_in")
            self.connect("tower.layer_mat", "floatingse.tower.layer_materials")
            self.connect("floating.transition_node", "floatingse.transition_node")
            if modeling_options["flags"]["tower"]:
                self.connect("tower_grid.height", "floatingse.tower_height")
            if modeling_options["flags"]["nacelle"]:
                self.connect("drivese.base_F", "floatingse.rna_F")
                self.connect("drivese.base_M", "floatingse.rna_M")
                self.connect("drivese.rna_I_TT", "floatingse.rna_I")
                self.connect("drivese.rna_cm", "floatingse.rna_cg")
                self.connect("drivese.rna_mass", "floatingse.rna_mass")

            # Individual member connections
            for k, kname in enumerate(modeling_options["floating"]["members"]["name"]):
                idx = modeling_options["floating"]["members"]["name2idx"][kname]
                self.connect(f"floating.memgrp{idx}.outer_diameter", f"floatingse.member{k}.outer_diameter_in")
                self.connect(f"floating.memgrp{idx}.outfitting_factor", f"floatingse.member{k}.outfitting_factor_in")

                for var in [
                    "s",
                    "layer_thickness",
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
                    "axial_stiffener_web_height",
                    "axial_stiffener_web_thickness",
                    "axial_stiffener_flange_width",
                    "axial_stiffener_flange_thickness",
                    "axial_stiffener_spacing",
                ]:
                    self.connect(f"floating.memgrp{idx}.{var}", f"floatingse.member{k}.{var}")

                for var in ["joint1", "joint2", "s_ghost1", "s_ghost2"]:
                    self.connect(f"floating.member_{kname}:{var}", f"floatingse.member{k}.{var}")

            # Mooring connections
            self.connect("mooring.unstretched_length", "floatingse.line_length", src_indices=[0])
            for var in [
                "fairlead",
                "fairlead_radius",
                "anchor_radius",
                "anchor_cost",
                "line_diameter",
                "line_mass_density_coeff",
                "line_stiffness_coeff",
                "line_breaking_load_coeff",
                "line_cost_rate_coeff",
            ]:
                self.connect("mooring." + var, "floatingse." + var, src_indices=[0])

        # Connections to turbine constraints
        if modeling_options["flags"]["blade"] and modeling_options["flags"]["tower"]:
            self.connect("configuration.rotor_orientation", "tcons.rotor_orientation")
            self.connect("rs.tip_pos.tip_deflection", "tcons.tip_deflection")
            self.connect("assembly.rotor_radius", "tcons.Rtip")
            self.connect("assembly.blade_ref_axis", "tcons.ref_axis_blade")
            self.connect("hub.cone", "tcons.precone")
            self.connect("nacelle.uptilt", "tcons.tilt")
            self.connect("nacelle.overhang", "tcons.overhang")
            self.connect("assembly.tower_ref_axis", "tcons.ref_axis_tower")
            self.connect("tower.diameter", "tcons.d_full")
            if modeling_options["flags"]["floating"]:
                self.connect("floatingse.tower_freqs", "tcons.tower_freq", src_indices=[0])
            else:
                self.connect("towerse.tower.freqs", "tcons.tower_freq", src_indices=[0])
            self.connect("configuration.n_blades", "tcons.blade_number")
            self.connect("rp.powercurve.rated_Omega", "tcons.rated_Omega")

        # Connections to turbine capital cost
        self.connect("configuration.n_blades", "tcc.blade_number")
        self.connect("configuration.rated_power", "tcc.machine_rating")
        if modeling_options["flags"]["blade"]:
            self.connect("re.precomp.blade_mass", "tcc.blade_mass")
            self.connect("re.precomp.total_blade_cost", "tcc.blade_cost_external")

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

        if modeling_options["flags"]["tower"] and not modeling_options["flags"]["floating"]:
            self.connect("towerse.structural_mass", "tcc.tower_mass")
            self.connect("towerse.structural_cost", "tcc.tower_cost_external")
        elif modeling_options["flags"]["floating"]:
            self.connect("floatingse.tower_mass", "tcc.tower_mass")
            self.connect("floatingse.tower_cost", "tcc.tower_cost_external")

        self.connect("costs.blade_mass_cost_coeff", "tcc.blade_mass_cost_coeff")
        self.connect("costs.hub_mass_cost_coeff", "tcc.hub_mass_cost_coeff")
        self.connect("costs.pitch_system_mass_cost_coeff", "tcc.pitch_system_mass_cost_coeff")
        self.connect("costs.spinner_mass_cost_coeff", "tcc.spinner_mass_cost_coeff")
        self.connect("costs.lss_mass_cost_coeff", "tcc.lss_mass_cost_coeff")
        self.connect("costs.bearing_mass_cost_coeff", "tcc.bearing_mass_cost_coeff")
        self.connect("costs.gearbox_mass_cost_coeff", "tcc.gearbox_mass_cost_coeff")
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
                self.add_subsystem("orbit", Orbit(floating=modeling_options["flags"]["floating_platform"]))
            else:
                self.add_subsystem("landbosse", LandBOSSE())

        if modeling_options["flags"]["blade"]:
            self.add_subsystem("financese", PlantFinance(verbosity=modeling_options["General"]["verbosity"]))
            self.add_subsystem(
                "outputs_2_screen", Outputs_2_Screen(modeling_options=modeling_options, opt_options=opt_options)
            )

        if opt_options["opt_flag"] and opt_options["recorder"]["flag"]:
            self.add_subsystem("conv_plots", Convergence_Trends_Opt(opt_options=opt_options))

        # BOS inputs
        if modeling_options["WISDEM"]["BOS"]["flag"]:
            if modeling_options["flags"]["offshore"]:
                # Inputs into ORBIT
                self.connect("configuration.rated_power", "orbit.turbine_rating")
                self.connect("env.water_depth", "orbit.site_depth")
                self.connect("costs.turbine_number", "orbit.number_of_turbines")
                self.connect("configuration.n_blades", "orbit.number_of_blades")
                self.connect("assembly.hub_height", "orbit.hub_height")
                self.connect("assembly.rotor_diameter", "orbit.turbine_rotor_diameter")
                self.connect("tower_grid.height", "orbit.tower_length")
                if modeling_options["flags"]["monopile"]:
                    self.connect("towerse.tower_mass", "orbit.tower_mass")
                    self.connect("towerse.monopile_mass", "orbit.monopile_mass")
                    self.connect("towerse.monopile_cost", "orbit.monopile_cost")
                    self.connect("monopile.height", "orbit.monopile_length")
                    self.connect("monopile.transition_piece_mass", "orbit.transition_piece_mass")
                    self.connect("monopile.transition_piece_cost", "orbit.transition_piece_cost")
                    self.connect("monopile.diameter", "orbit.monopile_diameter", src_indices=[0])
                else:
                    self.connect("floatingse.tower_mass", "orbit.tower_mass")
                    self.connect("mooring.n_lines", "orbit.num_mooring_lines")
                    self.connect("floatingse.line_mass", "orbit.mooring_line_mass", src_indices=[0])
                    self.connect("mooring.line_diameter", "orbit.mooring_line_diameter", src_indices=[0])
                    self.connect("mooring.unstretched_length", "orbit.mooring_line_length", src_indices=[0])
                    self.connect("mooring.anchor_mass", "orbit.anchor_mass", src_indices=[0])
                self.connect("re.precomp.blade_mass", "orbit.blade_mass")
                self.connect("tcc.turbine_cost_kW", "orbit.turbine_capex")
                if modeling_options["flags"]["nacelle"]:
                    self.connect("drivese.nacelle_mass", "orbit.nacelle_mass")
                self.connect("wt_class.V_mean", "orbit.site_mean_windspeed")
                self.connect("rp.powercurve.rated_V", "orbit.turbine_rated_windspeed")
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
                self.connect("assembly.hub_height", "landbosse.hub_height_meters")
                self.connect("costs.turbine_number", "landbosse.num_turbines")
                self.connect("configuration.rated_power", "landbosse.turbine_rating_MW")
                self.connect("env.shear_exp", "landbosse.wind_shear_exponent")
                self.connect("assembly.rotor_diameter", "landbosse.rotor_diameter_m")
                self.connect("configuration.n_blades", "landbosse.number_of_blades")
                if modeling_options["flags"]["blade"]:
                    self.connect("rp.powercurve.rated_T", "landbosse.rated_thrust_N")
                self.connect("towerse.tower_mass", "landbosse.tower_mass")
                if modeling_options["flags"]["nacelle"]:
                    self.connect("drivese.nacelle_mass", "landbosse.nacelle_mass")
                    self.connect("drivese.hub_system_mass", "landbosse.hub_mass")
                self.connect("re.precomp.blade_mass", "landbosse.blade_mass")
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
            self.connect("rp.AEP", "financese.turbine_aep")
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

        # Connections to outputs to screen
        if modeling_options["flags"]["blade"]:
            self.connect("rp.AEP", "outputs_2_screen.aep")
            self.connect("financese.lcoe", "outputs_2_screen.lcoe")
            self.connect("re.precomp.blade_mass", "outputs_2_screen.blade_mass")
            self.connect("rs.tip_pos.tip_deflection", "outputs_2_screen.tip_deflection")
