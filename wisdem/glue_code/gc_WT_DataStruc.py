import copy
import logging

import numpy as np
import openmdao.api as om
from scipy.interpolate import PchipInterpolator, interp1d

import moorpy.MoorProps as mp
from wisdem.ccblade.Polar import Polar
from wisdem.commonse.utilities import arc_length, arc_length_deriv
from wisdem.rotorse.parametrize_rotor import ComputeReynolds, ParametrizeBladeAero, ParametrizeBladeStruct
from wisdem.rotorse.geometry_tools.geometry import AirfoilShape, remap2grid, trailing_edge_smoothing

try:
    from INN_interface.INN import INN

    INN_loaded = True
except:
    INN_loaded = False
logger = logging.getLogger("wisdem/weis")


class WindTurbineOntologyOpenMDAO(om.Group):
    # Openmdao group with all wind turbine data

    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]

        # Material dictionary inputs
        self.add_subsystem(
            "materials",
            Materials(mat_init_options=modeling_options["materials"], composites=modeling_options["flags"]["blade"]),
        )

        # Environment inputs
        if modeling_options["flags"]["environment"]:
            env_ivc = self.add_subsystem("env", om.IndepVarComp())
            env_ivc.add_output("rho_air", val=1.225, units="kg/m**3", desc="Density of air")
            env_ivc.add_output("mu_air", val=1.81e-5, units="kg/(m*s)", desc="Dynamic viscosity of air")
            env_ivc.add_output("shear_exp", val=0.2, desc="Shear exponent of the wind.")
            env_ivc.add_output("speed_sound_air", val=340.0, units="m/s", desc="Speed of sound in air.")
            env_ivc.add_output(
                "weibull_k", val=2.0, desc="Shape parameter of the Weibull probability density function of the wind."
            )
            env_ivc.add_output("rho_water", val=1025.0, units="kg/m**3", desc="Density of ocean water")
            env_ivc.add_output("mu_water", val=1.3351e-3, units="kg/(m*s)", desc="Dynamic viscosity of ocean water")
            env_ivc.add_output(
                "water_depth", val=0.0, units="m", desc="Water depth for analysis.  Values > 0 mean offshore"
            )
            env_ivc.add_output("Hsig_wave", val=0.0, units="m", desc="Significant wave height")
            env_ivc.add_output("Tsig_wave", val=0.0, units="s", desc="Significant wave period")
            env_ivc.add_output("G_soil", val=140e6, units="N/m**2", desc="Shear stress of soil")
            env_ivc.add_output("nu_soil", val=0.4, desc="Poisson ratio of soil")
        
        # Airfoil dictionary inputs
        if modeling_options["flags"]["airfoils"]:
            airfoils = om.IndepVarComp()
            rotorse_options = modeling_options["WISDEM"]["RotorSE"]
            n_af = rotorse_options["n_af"]  # Number of airfoils
            n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
            n_Re = rotorse_options["n_Re"]  # Number of Reynolds, so far hard set at 1
            n_tab = rotorse_options[
                "n_tab"
            ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1
            n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
            airfoils.add_discrete_output("name", val=n_af * [""], desc="1D array of names of airfoils.")
            airfoils.add_output("ac", val=np.zeros(n_af), desc="1D array of the aerodynamic centers of each airfoil.")
            airfoils.add_output(
                "r_thick", val=np.zeros(n_af), desc="1D array of the relative thicknesses of each airfoil."
            )
            airfoils.add_output(
                "aoa",
                val=np.zeros(n_aoa),
                units="rad",
                desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
            )
            airfoils.add_output(
                "Re",
                val=np.zeros(n_Re),
                desc="1D array of the Reynolds numbers used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
            )
            airfoils.add_output(
                "cl",
                val=np.zeros((n_af, n_aoa, n_Re, n_tab)),
                desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
            )
            airfoils.add_output(
                "cd",
                val=np.zeros((n_af, n_aoa, n_Re, n_tab)),
                desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
            )
            airfoils.add_output(
                "cm",
                val=np.zeros((n_af, n_aoa, n_Re, n_tab)),
                desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
            )
            # Airfoil coordinates
            airfoils.add_output(
                "coord_xy",
                val=np.zeros((n_af, n_xy, 2)),
                desc="3D array of the x and y airfoil coordinates of the n_af airfoils.",
            )
            self.add_subsystem("airfoils", airfoils)

            if modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                if not INN_loaded:
                    raise Exception("The INN framework for airfoil design is activated, but not installed correctly")

                inn_af = om.IndepVarComp()
                inn_af.add_output(
                    "s_opt_r_thick", val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["rthick"]["n_opt"])
                )
                inn_af.add_output(
                    "r_thick_opt",
                    val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["rthick"]["n_opt"]),
                )
                inn_af.add_output(
                    "s_opt_L_D", val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["L/D"]["n_opt"])
                )
                inn_af.add_output(
                    "L_D_opt",
                    val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["L/D"]["n_opt"]),
                )
                inn_af.add_output(
                    "s_opt_c_d", val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["c_d"]["n_opt"])
                )
                inn_af.add_output(
                    "c_d_opt",
                    val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["c_d"]["n_opt"]),
                )
                inn_af.add_output(
                    "s_opt_stall_margin",
                    val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["stall_margin"]["n_opt"]),
                )
                inn_af.add_output(
                    "stall_margin_opt",
                    val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["stall_margin"]["n_opt"]),
                    units="rad",
                )
                inn_af.add_output(
                    "z",
                    val=np.zeros(opt_options["design_variables"]["blade"]["aero_shape"]["z"]["n_opt"]),
                )
                self.add_subsystem("inn_af", inn_af)

        # Wind turbine configuration inputs
        conf_ivc = self.add_subsystem("configuration", om.IndepVarComp())
        conf_ivc.add_discrete_output(
            "ws_class",
            val="",
            desc="IEC wind turbine class. I - offshore, II coastal, III - land-based, IV - low wind speed site.",
        )
        conf_ivc.add_discrete_output(
            "turb_class",
            val="",
            desc="IEC wind turbine category. A - high turbulence intensity (land-based), B - mid turbulence, C - low turbulence (offshore).",
        )
        conf_ivc.add_discrete_output(
            "gearbox_type", val="geared", desc="Gearbox configuration (geared, direct-drive, etc.)."
        )
        conf_ivc.add_discrete_output(
            "rotor_orientation", val="upwind", desc="Rotor orientation, either upwind or downwind."
        )
        conf_ivc.add_discrete_output(
            "upwind", val=True, desc="Convenient boolean for upwind (True) or downwind (False)."
        )
        conf_ivc.add_discrete_output("n_blades", val=3, desc="Number of blades of the rotor.")
        conf_ivc.add_output("rated_power", val=0.0, units="W", desc="Electrical rated power of the generator.")
        conf_ivc.add_output("lifetime", val=25.0, units="yr", desc="Turbine design lifetime.")

        conf_ivc.add_output(
            "rotor_diameter_user",
            val=0.0,
            units="m",
            desc="Diameter of the rotor specified by the user. It is defined as two times the blade length plus the hub diameter.",
        )
        conf_ivc.add_output(
            "hub_height_user",
            val=0.0,
            units="m",
            desc="Height of the hub center over the ground (land-based) or the mean sea level (offshore) specified by the user.",
        )

        # Hub inputs
        if modeling_options["flags"]["hub"] or modeling_options["flags"]["blade"]:
            self.add_subsystem("hub", Hub(flags=modeling_options["flags"]))

        # Control inputs
        if modeling_options["flags"]["control"]:
            ctrl_ivc = self.add_subsystem("control", om.IndepVarComp())
            ctrl_ivc.add_output(
                "V_in", val=0.0, units="m/s", desc="Cut in wind speed. This is the wind speed where region II begins."
            )
            ctrl_ivc.add_output(
                "V_out", val=0.0, units="m/s", desc="Cut out wind speed. This is the wind speed where region III ends."
            )
            ctrl_ivc.add_output("minOmega", val=0.0, units="rad/s", desc="Minimum allowed rotor speed.")
            ctrl_ivc.add_output("maxOmega", val=0.0, units="rad/s", desc="Maximum allowed rotor speed.")
            ctrl_ivc.add_output("max_TS", val=0.0, units="m/s", desc="Maximum allowed blade tip speed.")
            ctrl_ivc.add_output("max_pitch_rate", val=0.0, units="rad/s", desc="Maximum allowed blade pitch rate")
            ctrl_ivc.add_output("max_torque_rate", val=0.0, units="N*m/s", desc="Maximum allowed generator torque rate")
            ctrl_ivc.add_output("rated_TSR", val=0.0, desc="Constant tip speed ratio in region II.")
            ctrl_ivc.add_output("rated_pitch", val=0.0, units="rad", desc="Constant pitch angle in region II.")

        # Blade inputs and connections from airfoils
        if modeling_options["flags"]["blade"]:
            self.add_subsystem(
                "blade",
                Blade(
                    rotorse_options=modeling_options["WISDEM"]["RotorSE"],
                    opt_options=opt_options,
                ),
            )
            self.connect("airfoils.name", "blade.interp_airfoils.name")
            self.connect("airfoils.r_thick", "blade.interp_airfoils.r_thick_discrete")
            self.connect("airfoils.ac", "blade.interp_airfoils.ac")
            self.connect("airfoils.coord_xy", "blade.interp_airfoils.coord_xy")
            self.connect("airfoils.aoa", "blade.interp_airfoils.aoa")
            self.connect("airfoils.cl", "blade.interp_airfoils.cl")
            self.connect("airfoils.cd", "blade.interp_airfoils.cd")
            self.connect("airfoils.cm", "blade.interp_airfoils.cm")

            self.connect("hub.radius", "blade.high_level_blade_props.hub_radius")
            self.connect("configuration.rotor_diameter_user", "blade.high_level_blade_props.rotor_diameter_user")

            if modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                self.connect("airfoils.aoa", "blade.run_inn_af.aoa")
                self.connect("inn_af.s_opt_r_thick", "blade.run_inn_af.s_opt_r_thick")
                self.connect("inn_af.s_opt_L_D", "blade.run_inn_af.s_opt_L_D")
                self.connect("inn_af.s_opt_c_d", "blade.run_inn_af.s_opt_c_d")
                self.connect("inn_af.s_opt_stall_margin", "blade.run_inn_af.s_opt_stall_margin")
                self.connect("inn_af.r_thick_opt", "blade.run_inn_af.r_thick_opt")
                self.connect("inn_af.L_D_opt", "blade.run_inn_af.L_D_opt")
                self.connect("inn_af.c_d_opt", "blade.run_inn_af.c_d_opt")
                self.connect("inn_af.stall_margin_opt", "blade.run_inn_af.stall_margin_opt")
                self.connect("inn_af.z", "blade.run_inn_af.z")
                self.connect("control.rated_TSR", "blade.run_inn_af.rated_TSR")
                self.connect("hub.radius", "blade.run_inn_af.hub_radius")

        # Nacelle inputs
        if modeling_options["flags"]["nacelle"] or modeling_options["flags"]["blade"]:
            nacelle_ivc = om.IndepVarComp()
            # Common direct and geared
            nacelle_ivc.add_output(
                "uptilt", val=0.0, units="rad", desc="Nacelle uptilt angle. A standard machine has positive values."
            )
            nacelle_ivc.add_output(
                "distance_tt_hub", val=0.0, units="m", desc="Vertical distance from tower top plane to hub flange"
            )
            nacelle_ivc.add_output(
                "overhang", val=0.0, units="m", desc="Horizontal distance from tower top edge to hub flange"
            )
            nacelle_ivc.add_output(
                "gearbox_efficiency", val=1.0, desc="Efficiency of the gearbox. Set to 1.0 for direct-drive"
            )
            nacelle_ivc.add_output("gearbox_mass_user", val=0.0, units="kg", desc="User override of gearbox mass.")
            nacelle_ivc.add_output("gearbox_torque_density", val=0.0, units="N*m/kg", desc="Torque density of the gearbox.")
            nacelle_ivc.add_output(
                "gearbox_radius_user",
                val=0.0,
                units="m",
                desc="User override of gearbox radius (only used if gearbox_mass_user is > 0).",
            )
            nacelle_ivc.add_output(
                "gearbox_length_user",
                val=0.0,
                units="m",
                desc="User override of gearbox length (only used if gearbox_mass_user is > 0).",
            )

            nacelle_ivc.add_output("gear_ratio", val=1.0, desc="Total gear ratio of drivetrain (use 1.0 for direct)")
            if modeling_options["flags"]["nacelle"]:
                nacelle_ivc.add_output(
                    "distance_hub_mb",
                    val=0.0,
                    units="m",
                    desc="Distance from hub flange to first main bearing along shaft",
                )
                nacelle_ivc.add_output(
                    "distance_mb_mb", val=0.0, units="m", desc="Distance from first to second main bearing along shaft"
                )
                nacelle_ivc.add_output("L_generator", val=0.0, units="m", desc="Generator length along shaft")
                nacelle_ivc.add_output("lss_diameter", val=np.zeros(2), units="m", desc="Diameter of low speed shaft")
                nacelle_ivc.add_output(
                    "lss_wall_thickness", val=np.zeros(2), units="m", desc="Thickness of low speed shaft"
                )
                nacelle_ivc.add_output("damping_ratio", val=0.0, desc="Damping ratio for the drivetrain system")
                nacelle_ivc.add_output(
                    "brake_mass_user",
                    val=0.0,
                    units="kg",
                    desc="Override regular regression-based calculation of brake mass with this value",
                )
                nacelle_ivc.add_output(
                    "hvac_mass_coeff",
                    val=0.025,
                    units="kg/kW/m",
                    desc="Regression-based scaling coefficient on machine rating to get HVAC system mass",
                )
                nacelle_ivc.add_output(
                    "converter_mass_user",
                    val=0.0,
                    units="kg",
                    desc="Override regular regression-based calculation of converter mass with this value",
                )
                nacelle_ivc.add_output(
                    "transformer_mass_user",
                    val=0.0,
                    units="kg",
                    desc="Override regular regression-based calculation of transformer mass with this value",
                )
                nacelle_ivc.add_discrete_output(
                    "mb1Type", val="CARB", desc="Type of main bearing: CARB / CRB / SRB / TRB"
                )
                nacelle_ivc.add_discrete_output(
                    "mb2Type", val="SRB", desc="Type of main bearing: CARB / CRB / SRB / TRB"
                )
                nacelle_ivc.add_discrete_output(
                    "uptower", val=True, desc="If power electronics are located uptower (True) or at tower base (False)"
                )
                nacelle_ivc.add_discrete_output(
                    "lss_material", val="steel", desc="Material name identifier for the low speed shaft"
                )
                nacelle_ivc.add_discrete_output(
                    "hss_material", val="steel", desc="Material name identifier for the high speed shaft"
                )
                nacelle_ivc.add_discrete_output(
                    "bedplate_material", val="steel", desc="Material name identifier for the bedplate"
                )

                if modeling_options["WISDEM"]["DriveSE"]["direct"]:
                    # Direct only
                    nacelle_ivc.add_output(
                        "nose_diameter",
                        val=np.zeros(2),
                        units="m",
                        desc="Diameter of nose (also called turret or spindle)",
                    )
                    nacelle_ivc.add_output(
                        "nose_wall_thickness",
                        val=np.zeros(2),
                        units="m",
                        desc="Thickness of nose (also called turret or spindle)",
                    )
                    nacelle_ivc.add_output(
                        "bedplate_wall_thickness",
                        val=np.zeros(4),
                        units="m",
                        desc="Thickness of hollow elliptical bedplate",
                    )
                else:
                    # Geared only
                    nacelle_ivc.add_output("hss_length", val=0.0, units="m", desc="Length of high speed shaft")
                    nacelle_ivc.add_output(
                        "hss_diameter", val=np.zeros(2), units="m", desc="Diameter of high speed shaft"
                    )
                    nacelle_ivc.add_output(
                        "hss_wall_thickness", val=np.zeros(2), units="m", desc="Wall thickness of high speed shaft"
                    )
                    nacelle_ivc.add_output(
                        "bedplate_flange_width", val=0.0, units="m", desc="Bedplate I-beam flange width"
                    )
                    nacelle_ivc.add_output(
                        "bedplate_flange_thickness", val=0.0, units="m", desc="Bedplate I-beam flange thickness"
                    )
                    nacelle_ivc.add_output(
                        "bedplate_web_thickness", val=0.0, units="m", desc="Bedplate I-beam web thickness"
                    )
                    nacelle_ivc.add_discrete_output(
                        "gear_configuration",
                        val="eep",
                        desc="3-letter string of Es or Ps to denote epicyclic or parallel gear configuration",
                    )
                    nacelle_ivc.add_discrete_output(
                        "planet_numbers",
                        val=[3, 3, 0],
                        desc="Number of planets for epicyclic stages (use 0 for parallel)",
                    )

            # Mulit-body properties
            # GB: I understand these will need to be in there for OpenFAST, but if running DrivetrainSE & OpenFAST this might cause problems?
            # nacelle_ivc.add_output('above_yaw_mass',   val=0.0, units='kg', desc='Mass of the nacelle above the yaw system')
            # nacelle_ivc.add_output('yaw_mass',         val=0.0, units='kg', desc='Mass of yaw system')
            # nacelle_ivc.add_output('nacelle_cm',       val=np.zeros(3), units='m', desc='Center of mass of the component in [x,y,z] for an arbitrary coordinate system')
            # nacelle_ivc.add_output('nacelle_I',        val=np.zeros(6), units='kg*m**2', desc=' moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
            self.add_subsystem("nacelle", nacelle_ivc)

            # Generator inputs
            generator_ivc = om.IndepVarComp()
            if modeling_options["flags"]["generator"]:
                generator_ivc.add_output("B_r", val=1.2, units="T")
                generator_ivc.add_output("P_Fe0e", val=1.0, units="W/kg")
                generator_ivc.add_output("P_Fe0h", val=4.0, units="W/kg")
                generator_ivc.add_output("S_N", val=-0.002)
                generator_ivc.add_output("alpha_p", val=0.5 * np.pi * 0.7)
                generator_ivc.add_output("b_r_tau_r", val=0.45)
                generator_ivc.add_output("b_ro", val=0.004, units="m")
                generator_ivc.add_output("b_s_tau_s", val=0.45)
                generator_ivc.add_output("b_so", val=0.004, units="m")
                generator_ivc.add_output("cofi", val=0.85)
                generator_ivc.add_output("freq", val=60, units="Hz")
                generator_ivc.add_output("h_i", val=0.001, units="m")
                generator_ivc.add_output("h_sy0", val=0.0)
                generator_ivc.add_output("h_w", val=0.005, units="m")
                generator_ivc.add_output("k_fes", val=0.9)
                generator_ivc.add_output("k_fillr", val=0.7)
                generator_ivc.add_output("k_fills", val=0.65)
                generator_ivc.add_output("k_s", val=0.2)
                generator_ivc.add_discrete_output("m", val=3)
                generator_ivc.add_output("mu_0", val=np.pi * 4e-7, units="m*kg/s**2/A**2")
                generator_ivc.add_output("mu_r", val=1.06, units="m*kg/s**2/A**2")
                generator_ivc.add_output("p", val=3.0)
                generator_ivc.add_output("phi", val=np.deg2rad(90), units="rad")
                generator_ivc.add_discrete_output("q1", val=6)
                generator_ivc.add_discrete_output("q2", val=4)
                generator_ivc.add_output("ratio_mw2pp", val=0.7)
                generator_ivc.add_output("resist_Cu", val=1.8e-8 * 1.4, units="ohm/m")
                generator_ivc.add_output("sigma", val=40e3, units="Pa")
                generator_ivc.add_output("y_tau_p", val=1.0)
                generator_ivc.add_output("y_tau_pr", val=10.0 / 12)

                generator_ivc.add_output("I_0", val=0.0, units="A")
                generator_ivc.add_output("d_r", val=0.0, units="m")
                generator_ivc.add_output("h_m", val=0.0, units="m")
                generator_ivc.add_output("h_0", val=0.0, units="m")
                generator_ivc.add_output("h_s", val=0.0, units="m")
                generator_ivc.add_output("len_s", val=0.0, units="m")
                generator_ivc.add_output("n_r", val=0.0)
                generator_ivc.add_output("rad_ag", val=0.0, units="m")
                generator_ivc.add_output("t_wr", val=0.0, units="m")

                generator_ivc.add_output("n_s", val=0.0)
                generator_ivc.add_output("b_st", val=0.0, units="m")
                generator_ivc.add_output("d_s", val=0.0, units="m")
                generator_ivc.add_output("t_ws", val=0.0, units="m")

                generator_ivc.add_output("rho_Copper", val=0.0, units="kg*m**-3")
                generator_ivc.add_output("rho_Fe", val=0.0, units="kg*m**-3")
                generator_ivc.add_output("rho_Fes", val=0.0, units="kg*m**-3")
                generator_ivc.add_output("rho_PM", val=0.0, units="kg*m**-3")

                generator_ivc.add_output("C_Cu", val=0.0, units="USD/kg")
                generator_ivc.add_output("C_Fe", val=0.0, units="USD/kg")
                generator_ivc.add_output("C_Fes", val=0.0, units="USD/kg")
                generator_ivc.add_output("C_PM", val=0.0, units="USD/kg")

                if modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["pmsg_outer"]:
                    generator_ivc.add_output("N_c", 0.0)
                    generator_ivc.add_output("b", 0.0)
                    generator_ivc.add_output("c", 0.0)
                    generator_ivc.add_output("E_p", 0.0, units="V")
                    generator_ivc.add_output("h_yr", val=0.0, units="m")
                    generator_ivc.add_output("h_ys", val=0.0, units="m")
                    generator_ivc.add_output("h_sr", 0.0, units="m", desc="Structural Mass")
                    generator_ivc.add_output("h_ss", 0.0, units="m")
                    generator_ivc.add_output("t_r", 0.0, units="m")
                    generator_ivc.add_output("t_s", 0.0, units="m")

                    generator_ivc.add_output("u_allow_pcent", 0.0)
                    generator_ivc.add_output("y_allow_pcent", 0.0)
                    generator_ivc.add_output("z_allow_deg", 0.0, units="deg")
                    generator_ivc.add_output("B_tmax", 0.0, units="T")

                if modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
                    generator_ivc.add_output("tau_p", val=0.0, units="m")
                    generator_ivc.add_output("h_ys", val=0.0, units="m")
                    generator_ivc.add_output("h_yr", val=0.0, units="m")
                    generator_ivc.add_output("b_arm", val=0.0, units="m")

                elif modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["scig", "dfig"]:
                    generator_ivc.add_output("B_symax", val=0.0, units="T")
                    generator_ivc.add_output("S_Nmax", val=-0.2)

            else:
                # If using simple (regression) generator scaling, this is an optional input to override default values
                n_pc = modeling_options["WISDEM"]["RotorSE"]["n_pc"]
                generator_ivc.add_output("generator_radius_user", val=0.0, units="m")
                generator_ivc.add_output("generator_mass_user", val=0.0, units="kg")
                generator_ivc.add_output("generator_efficiency_user", val=np.zeros((n_pc, 2)))

            self.add_subsystem("generator", generator_ivc)

        # Tower inputs
        if modeling_options["flags"]["tower"]:
            tower_init_options = modeling_options["WISDEM"]["TowerSE"]
            n_height_tower = tower_init_options["n_height"]
            n_layers_tower = tower_init_options["n_layers"]
            ivc = self.add_subsystem("tower", om.IndepVarComp())
            ivc.add_output(
                "ref_axis",
                val=np.zeros((n_height_tower, 3)),
                units="m",
                desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
            )
            ivc.add_output(
                "diameter",
                val=np.zeros(n_height_tower),
                units="m",
                desc="1D array of the outer diameter values defined along the tower axis.",
            )
            ivc.add_output(
                "cd",
                val=np.zeros(n_height_tower),
                desc="1D array of the drag coefficients defined along the tower height.",
            )
            ivc.add_output(
                "layer_thickness",
                val=np.zeros((n_layers_tower, n_height_tower)),
                units="m",
                desc="2D array of the thickness of the layers of the tower structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the tower sections.",
            )
            ivc.add_output(
                "outfitting_factor",
                val=0.0,
                desc="Multiplier that accounts for secondary structure mass inside of tower",
            )
            ivc.add_discrete_output(
                "layer_name", val=[], desc="1D array of the names of the layers modeled in the tower structure."
            )
            ivc.add_discrete_output(
                "layer_mat",
                val=[],
                desc="1D array of the names of the materials of each layer modeled in the tower structure.",
            )

        # Monopile inputs
        if modeling_options["flags"]["monopile"]:
            self.add_subsystem("monopile", Monopile(fixedbottomse_options=modeling_options["WISDEM"]["FixedBottomSE"]))

        # Jacket inputs
        if modeling_options["flags"]["jacket"]:
            self.add_subsystem("jacket", Jacket(fixedbottomse_options=modeling_options["WISDEM"]["FixedBottomSE"]))

        # Floating substructure inputs
        if modeling_options["flags"]["floating_platform"]:
            self.add_subsystem(
                "floating",
                Floating(floating_init_options=modeling_options["floating"], opt_options=self.options["opt_options"]),
            )
            self.add_subsystem("mooring", Mooring(options=modeling_options))
            self.connect("floating.joints_xyz", "mooring.joints_xyz")

        # Balance of station inputs
        if modeling_options["flags"]["bos"]:
            bos_ivc = self.add_subsystem("bos", om.IndepVarComp())
            bos_ivc.add_output("plant_turbine_spacing", 7, desc="Distance between turbines in rotor diameters")
            bos_ivc.add_output("plant_row_spacing", 7, desc="Distance between turbine rows in rotor diameters")
            bos_ivc.add_output("commissioning_pct", 0.01)
            bos_ivc.add_output("decommissioning_pct", 0.15)
            bos_ivc.add_output("distance_to_substation", 50.0, units="km")
            bos_ivc.add_output("distance_to_interconnection", 5.0, units="km")
            if modeling_options["flags"]["offshore"]:
                bos_ivc.add_output("site_distance", 40.0, units="km")
                bos_ivc.add_output("distance_to_landfall", 40.0, units="km")
                bos_ivc.add_output("port_cost_per_month", 2e6, units="USD/mo")
                bos_ivc.add_output("site_auction_price", 100e6, units="USD")
                bos_ivc.add_output("site_assessment_plan_cost", 1e6, units="USD")
                bos_ivc.add_output("site_assessment_cost", 25e6, units="USD")
                bos_ivc.add_output("construction_operations_plan_cost", 2.5e6, units="USD")
                bos_ivc.add_output("boem_review_cost", 0.0, units="USD")
                bos_ivc.add_output("design_install_plan_cost", 2.5e6, units="USD")
            else:
                bos_ivc.add_output("interconnect_voltage", 130.0, units="kV")

        # Cost analysis inputs
        if modeling_options["flags"]["costs"]:
            costs_ivc = self.add_subsystem("costs", om.IndepVarComp())
            costs_ivc.add_discrete_output("turbine_number", val=0, desc="Number of turbines at plant")
            costs_ivc.add_output("offset_tcc_per_kW", val=0.0, units="USD/kW", desc="Offset to turbine capital cost")
            costs_ivc.add_output("bos_per_kW", val=0.0, units="USD/kW", desc="Balance of station/plant capital cost")
            costs_ivc.add_output(
                "opex_per_kW", val=0.0, units="USD/kW/yr", desc="Average annual operational expenditures of the turbine"
            )
            costs_ivc.add_output("wake_loss_factor", val=0.0, desc="The losses in AEP due to waked conditions")
            costs_ivc.add_output("fixed_charge_rate", val=0.0, desc="Fixed charge rate for coe calculation")
            costs_ivc.add_output("labor_rate", 0.0, units="USD/h")
            costs_ivc.add_output("painting_rate", 0.0, units="USD/m**2")

            costs_ivc.add_output("blade_mass_cost_coeff", units="USD/kg", val=14.6)
            costs_ivc.add_output("hub_mass_cost_coeff", units="USD/kg", val=3.9)
            costs_ivc.add_output("pitch_system_mass_cost_coeff", units="USD/kg", val=22.1)
            costs_ivc.add_output("spinner_mass_cost_coeff", units="USD/kg", val=11.1)
            costs_ivc.add_output("lss_mass_cost_coeff", units="USD/kg", val=11.9)
            costs_ivc.add_output("bearing_mass_cost_coeff", units="USD/kg", val=4.5)
            costs_ivc.add_output("gearbox_torque_cost", units="USD/kN/m", val=50.)
            costs_ivc.add_output("hss_mass_cost_coeff", units="USD/kg", val=6.8)
            costs_ivc.add_output("generator_mass_cost_coeff", units="USD/kg", val=12.4)
            costs_ivc.add_output("bedplate_mass_cost_coeff", units="USD/kg", val=2.9)
            costs_ivc.add_output("yaw_mass_cost_coeff", units="USD/kg", val=8.3)
            costs_ivc.add_output("converter_mass_cost_coeff", units="USD/kg", val=18.8)
            costs_ivc.add_output("transformer_mass_cost_coeff", units="USD/kg", val=18.8)
            costs_ivc.add_output("hvac_mass_cost_coeff", units="USD/kg", val=124.0)
            costs_ivc.add_output("cover_mass_cost_coeff", units="USD/kg", val=5.7)
            costs_ivc.add_output("elec_connec_machine_rating_cost_coeff", units="USD/kW", val=41.85)
            costs_ivc.add_output("platforms_mass_cost_coeff", units="USD/kg", val=17.1)
            costs_ivc.add_output("tower_mass_cost_coeff", units="USD/kg", val=2.9)
            costs_ivc.add_output("controls_machine_rating_cost_coeff", units="USD/kW", val=21.15)
            costs_ivc.add_output("crane_cost", units="USD", val=12e3)
            costs_ivc.add_output("electricity_price", val=0.04, units="USD/kW/h")
            costs_ivc.add_output("reserve_margin_price", val=120.0, units="USD/kW/yr")
            costs_ivc.add_output("capacity_credit", val=1.0)
            costs_ivc.add_output("benchmark_price", val=0.071, units="USD/kW/h")

        # Assembly setup
        self.add_subsystem("high_level_tower_props", ComputeHighLevelTowerProperties(modeling_options=modeling_options))
        self.connect("configuration.hub_height_user", "high_level_tower_props.hub_height_user")
        if modeling_options["flags"]["blade"]:
            self.connect("blade.high_level_blade_props.rotor_diameter", "high_level_tower_props.rotor_diameter")
            self.add_subsystem("af_3d", Airfoil3DCorrection(rotorse_options=modeling_options["WISDEM"]["RotorSE"]))
            self.connect("airfoils.aoa", "af_3d.aoa")
            self.connect("airfoils.Re", "af_3d.Re")
            if modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                self.connect("blade.run_inn_af.cl_interp", "af_3d.cl")
                self.connect("blade.run_inn_af.cd_interp", "af_3d.cd")
            else:
                self.connect("blade.interp_airfoils.cl_interp", "af_3d.cl")
                self.connect("blade.interp_airfoils.cd_interp", "af_3d.cd")
            self.connect("blade.interp_airfoils.cm_interp", "af_3d.cm")
            self.connect("blade.high_level_blade_props.rotor_radius", "af_3d.rotor_radius")
            self.connect("blade.high_level_blade_props.r_blade", "af_3d.r_blade")
            self.connect("blade.interp_airfoils.r_thick_interp", "af_3d.r_thick")
            self.connect("blade.pa.chord_param", "af_3d.chord")
            self.connect("control.rated_TSR", "af_3d.rated_TSR")
            self.connect("control.maxOmega", "blade.compute_reynolds.maxOmega")
            self.connect("control.max_TS", "blade.compute_reynolds.max_TS")
            self.connect("control.V_out", "blade.compute_reynolds.V_out")
        if modeling_options["flags"]["tower"]:
            self.connect("tower.ref_axis", "high_level_tower_props.tower_ref_axis_user")
            self.add_subsystem("tower_grid", Compute_Grid(n_height=n_height_tower))
            self.connect("high_level_tower_props.tower_ref_axis", "tower_grid.ref_axis")
        if modeling_options["flags"]["nacelle"]:
            self.connect("nacelle.distance_tt_hub", "high_level_tower_props.distance_tt_hub")


class Blade(om.Group):
    # Openmdao group with components with the blade data coming from the input yaml file.
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("opt_options")

    def setup(self):
        # Options
        rotorse_options = self.options["rotorse_options"]
        opt_options = self.options["opt_options"]

        # Optimization parameters initialized as indipendent variable component
        opt_var = om.IndepVarComp()
        opt_var.add_output(
            "s_opt_twist", val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"])
        )
        opt_var.add_output(
            "s_opt_chord", val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"])
        )
        opt_var.add_output(
            "twist_opt",
            val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]),
            units="rad",
        )
        opt_var.add_output(
            "chord_opt",
            units="m",
            val=np.ones(opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"]),
        )
        opt_var.add_output("af_position", val=np.ones(rotorse_options["n_af_span"]))
        for i in range(rotorse_options["n_layers"]):
            opt_var.add_output(
                "s_opt_layer_%d"%i,
                val=np.ones(opt_options["design_variables"]["blade"]["n_opt_struct"][i]),
            )
            opt_var.add_output(
                "layer_%d_opt"%i,
                units="m",
                val=np.ones(opt_options["design_variables"]["blade"]["n_opt_struct"][i]),
            )
        self.add_subsystem("opt_var", opt_var)

        # Import outer shape BEM
        self.add_subsystem("outer_shape_bem", Blade_Outer_Shape_BEM(rotorse_options=rotorse_options))

        # Parametrize blade outer shape
        self.add_subsystem(
            "pa", ParametrizeBladeAero(rotorse_options=rotorse_options, opt_options=opt_options)
        )  # Parameterize aero (chord and twist)
        # Connections to blade aero parametrization
        self.connect("opt_var.s_opt_twist", "pa.s_opt_twist")
        self.connect("opt_var.s_opt_chord", "pa.s_opt_chord")
        self.connect("opt_var.twist_opt", "pa.twist_opt")
        self.connect("opt_var.chord_opt", "pa.chord_opt")
        self.connect("outer_shape_bem.s", "pa.s")

        # Interpolate airfoil profiles and coordinates
        self.add_subsystem(
            "interp_airfoils",
            Blade_Interp_Airfoils(rotorse_options=rotorse_options),
        )

        # Connections from oute_shape_bem to interp_airfoils
        self.connect("outer_shape_bem.s", "interp_airfoils.s")
        self.connect("outer_shape_bem.r_thick_yaml_interp", "interp_airfoils.r_thick_yaml")
        self.connect("pa.chord_param", ["interp_airfoils.chord", "compute_coord_xy_dim.chord"])
        self.connect("outer_shape_bem.pitch_axis", ["interp_airfoils.pitch_axis", "compute_coord_xy_dim.pitch_axis"])
        self.connect("opt_var.af_position", "interp_airfoils.af_position")

        self.add_subsystem("high_level_blade_props", ComputeHighLevelBladeProperties(rotorse_options=rotorse_options))
        self.connect("outer_shape_bem.ref_axis", "high_level_blade_props.blade_ref_axis_user")

        # TODO : Compute Reynolds here
        self.add_subsystem("compute_reynolds", ComputeReynolds(n_span=rotorse_options["n_span"]))
        self.connect("high_level_blade_props.r_blade", "compute_reynolds.r_blade")
        self.connect("high_level_blade_props.rotor_radius", "compute_reynolds.rotor_radius")
        
        if rotorse_options["inn_af"]:
            self.add_subsystem(
                "run_inn_af",
                INN_Airfoils(
                    rotorse_options=rotorse_options,
                    aero_shape_opt_options=opt_options["design_variables"]["blade"]["aero_shape"],
                ),
            )
            self.connect("outer_shape_bem.s", "run_inn_af.s")
            self.connect("pa.chord_param", "run_inn_af.chord")
            self.connect("interp_airfoils.r_thick_interp", "run_inn_af.r_thick")
            self.connect("interp_airfoils.cl_interp", "run_inn_af.cl_interp_yaml")
            self.connect("interp_airfoils.cd_interp", "run_inn_af.cd_interp_yaml")
            self.connect("interp_airfoils.cm_interp", "run_inn_af.cm_interp_yaml")
            self.connect("interp_airfoils.coord_xy_interp", "run_inn_af.coord_xy_interp_yaml")
            self.connect("high_level_blade_props.rotor_diameter", "run_inn_af.rotor_diameter")
            self.connect("compute_reynolds.Re", "run_inn_af.Re")

        self.add_subsystem(
            "compute_coord_xy_dim",
            Compute_Coord_XY_Dim(rotorse_options=rotorse_options),
        )
        self.connect("pa.twist_param", "compute_coord_xy_dim.twist")
        self.connect("high_level_blade_props.blade_ref_axis", "compute_coord_xy_dim.ref_axis")

        if rotorse_options["inn_af"]:
            self.connect("run_inn_af.coord_xy_interp", "compute_coord_xy_dim.coord_xy_interp")
        else:
            self.connect("interp_airfoils.coord_xy_interp", "compute_coord_xy_dim.coord_xy_interp")

        # If the flag is true, generate the 3D x,y,z points of the outer blade shape
        if rotorse_options["lofted_output"] == True:
            self.add_subsystem(
                "blade_lofted",
                Blade_Lofted_Shape(rotorse_options=rotorse_options),
            )
            self.connect("compute_coord_xy_dim.coord_xy_dim_twisted", "blade_lofted.coord_xy_dim_twisted")
            self.connect("high_level_blade_props.blade_ref_axis", "blade_lofted.ref_axis")

        # Import blade internal structure data and remap composites on the outer blade shape
        self.add_subsystem(
            "internal_structure_2d_fem",
            Blade_Internal_Structure_2D_FEM(rotorse_options=rotorse_options),
        )
        self.connect("outer_shape_bem.s", "internal_structure_2d_fem.s")
        self.connect("pa.twist_param", "internal_structure_2d_fem.twist")
        self.connect("pa.chord_param", "internal_structure_2d_fem.chord")
        self.connect("outer_shape_bem.pitch_axis", "internal_structure_2d_fem.pitch_axis")

        self.connect("compute_coord_xy_dim.coord_xy_dim", "internal_structure_2d_fem.coord_xy_dim")

        self.add_subsystem(
            "ps", ParametrizeBladeStruct(rotorse_options=rotorse_options, opt_options=opt_options)
        )  # Parameterize struct (spar caps ss and ps)

        # Connections to blade struct parametrization
        for i in range(rotorse_options["n_layers"]):
            self.connect("opt_var.layer_%d_opt"%i, "ps.layer_%d_opt"%i)
            self.connect("opt_var.s_opt_layer_%d"%i, "ps.s_opt_layer_%d"%i)

        self.connect("outer_shape_bem.s", "ps.s")
        # self.connect('internal_structure_2d_fem.layer_name',      'ps.layer_name')
        self.connect("internal_structure_2d_fem.layer_thickness", "ps.layer_thickness_original")

        # Fatigue specific parameters
        fat_var = om.IndepVarComp()
        fat_var.add_output("sparU_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("sparU_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("sparU_wohlerexp", val=1.0)
        fat_var.add_output("sparL_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("sparL_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("sparL_wohlerexp", val=1.0)
        fat_var.add_output("teU_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("teU_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("teU_wohlerexp", val=1.0)
        fat_var.add_output("teL_sigma_ult", val=1.0, units="Pa")
        fat_var.add_output("teL_wohlerA", val=1.0, units="Pa")
        fat_var.add_output("teL_wohlerexp", val=1.0)
        self.add_subsystem("fatigue", fat_var)


class Blade_Outer_Shape_BEM(om.Group):
    # Openmdao group with the blade outer shape data coming from the input yaml file.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        n_af_span = rotorse_options["n_af_span"]
        self.n_span = n_span = rotorse_options["n_span"]

        ivc = self.add_subsystem("blade_outer_shape_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "af_position",
            val=np.zeros(n_af_span),
            desc="1D array of the non dimensional positions of the airfoils af_used defined along blade span.",
        )
        ivc.add_output(
            "s_default",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        ivc.add_output(
            "chord_yaml", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        ivc.add_output(
            "twist_yaml",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        ivc.add_output(
            "pitch_axis_yaml",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )
        ivc.add_output(
            "ref_axis_yaml",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )
        ivc.add_output(
            "r_thick_yaml", val=np.zeros(n_span), desc="1D array of the relative thickness values defined along blade span."
        )

        self.add_subsystem(
            "compute_blade_outer_shape_bem",
            Compute_Blade_Outer_Shape_BEM(rotorse_options=rotorse_options),
            promotes=["*"],
        )


class Compute_Blade_Outer_Shape_BEM(om.ExplicitComponent):
    # Openmdao group with the blade outer shape data coming from the input yaml file.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        n_af_span = rotorse_options["n_af_span"]
        self.n_span = n_span = rotorse_options["n_span"]
        if "n_te_flaps" in rotorse_options.keys():
            n_te_flaps = rotorse_options["n_te_flaps"]
        else:
            n_te_flaps = 0

        self.add_input(
            "s_default",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "chord_yaml", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        self.add_input(
            "twist_yaml",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        self.add_input(
            "r_thick_yaml",
            val=np.zeros(n_span),
            desc="1D array of the relative thickness values defined along blade span.",
        )
        self.add_input(
            "pitch_axis_yaml",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )
        self.add_input(
            "ref_axis_yaml",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

        self.add_input(
            "span_end",
            val=np.zeros(n_te_flaps),
            desc="1D array of the positions along blade span where something (a DAC device?) starts and we want a grid point. Only values between 0 and 1 are meaningful.",
        )
        self.add_input(
            "span_ext",
            val=np.zeros(n_te_flaps),
            desc="1D array of the extensions along blade span where something (a DAC device?) lives and we want a grid point. Only values between 0 and 1 are meaningful.",
        )

        self.add_output(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_output(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        self.add_output(
            "twist",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        self.add_output(
            "pitch_axis",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )
        self.add_output(
            "r_thick_yaml_interp",
            val=np.zeros(n_span),
            desc="1D array of the relative thickness values defined along blade span.",
        )
        self.add_output(
            "ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

    def compute(self, inputs, outputs):
        # If devices are defined along span, manipulate the grid s to always have a grid point where it is needed, and reinterpolate the blade quantities, namely chord, twist, pitch axis, and reference axis
        if len(inputs["span_end"]) > 0:
            nd_span_orig = np.linspace(0.0, 1.0, self.n_span)

            chord_orig = PchipInterpolator(inputs["s_default"], inputs["chord_yaml"])(nd_span_orig)
            twist_orig = PchipInterpolator(inputs["s_default"], inputs["twist_yaml"])(nd_span_orig)
            pitch_axis_orig = PchipInterpolator(inputs["s_default"], inputs["pitch_axis_yaml"])(nd_span_orig)
            r_thick_orig = PchipInterpolator(inputs["s_default"], inputs["r_thick_yaml"])(nd_span_orig)
            ref_axis_orig = np.zeros((self.n_span, 3))
            ref_axis_orig[:, 0] = PchipInterpolator(inputs["s_default"], inputs["ref_axis_yaml"][:, 0])(nd_span_orig)
            ref_axis_orig[:, 1] = PchipInterpolator(inputs["s_default"], inputs["ref_axis_yaml"][:, 1])(nd_span_orig)
            ref_axis_orig[:, 2] = PchipInterpolator(inputs["s_default"], inputs["ref_axis_yaml"][:, 2])(nd_span_orig)

            outputs["s"] = copy.copy(nd_span_orig)

            # Account for start and end positions
            if inputs["span_end"] >= 0.98:
                flap_start = 0.98 - inputs["span_ext"]
                flap_end = 0.98
                # print("WARNING: span_end point reached limits and was set to r/R = 0.98")
            else:
                flap_start = inputs["span_end"] - inputs["span_ext"]
                flap_end = inputs["span_end"]

            idx_flap_start = np.where(np.abs(nd_span_orig - flap_start) == (np.abs(nd_span_orig - flap_start)).min())[
                0
            ][0]
            idx_flap_end = np.where(np.abs(nd_span_orig - flap_end) == (np.abs(nd_span_orig - flap_end)).min())[0][0]
            if idx_flap_start == idx_flap_end:
                idx_flap_end += 1
            outputs["s"][idx_flap_start] = flap_start
            outputs["s"][idx_flap_end] = flap_end
            outputs["chord"] = PchipInterpolator(nd_span_orig, chord_orig)(outputs["s"])
            outputs["twist"] = PchipInterpolator(nd_span_orig, twist_orig)(outputs["s"])
            outputs["pitch_axis"] = PchipInterpolator(nd_span_orig, pitch_axis_orig)(outputs["s"])
            outputs["r_thick_yaml_interp"] = PchipInterpolator(nd_span_orig, r_thick_orig)(outputs["s"])
            outputs["ref_axis"][:, 0] = PchipInterpolator(nd_span_orig, ref_axis_orig[:, 0])(outputs["s"])
            outputs["ref_axis"][:, 1] = PchipInterpolator(nd_span_orig, ref_axis_orig[:, 1])(outputs["s"])
            outputs["ref_axis"][:, 2] = PchipInterpolator(nd_span_orig, ref_axis_orig[:, 2])(outputs["s"])
        else:
            outputs["s"] = inputs["s_default"]
            outputs["chord"] = inputs["chord_yaml"]
            outputs["twist"] = inputs["twist_yaml"]
            outputs["pitch_axis"] = inputs["pitch_axis_yaml"]
            outputs["r_thick_yaml_interp"] = inputs["r_thick_yaml"]
            outputs["ref_axis"] = inputs["ref_axis_yaml"]


class Blade_Interp_Airfoils(om.ExplicitComponent):
    # Openmdao component to interpolate airfoil coordinates and airfoil polars along the span of the blade for a predefined set of airfoils coming from component Airfoils.
    # JPJ: can split this up into multiple components to ease derivative computation
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_af_span = n_af_span = rotorse_options["n_af_span"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_af = n_af = rotorse_options["n_af"]  # Number of airfoils
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.n_tab = n_tab = rotorse_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        self.af_used = rotorse_options["af_used"]  # Names of the airfoils adopted along blade span

        self.add_input(
            "af_position",
            val=np.zeros(n_af_span),
            desc="1D array of the non dimensional positions of the airfoils af_used defined along blade span.",
        )
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "pitch_axis",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )
        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )

        # Airfoil properties
        self.add_discrete_input("name", val=n_af * [""], desc="1D array of names of airfoils.")
        self.add_input("ac", val=np.zeros(n_af), desc="1D array of the aerodynamic centers of each airfoil.")
        self.add_input("r_thick_discrete", val=np.zeros(n_af), desc="1D array of the relative thicknesses of each airfoil.")
        self.add_input(
            "aoa",
            val=np.zeros(n_aoa),
            units="rad",
            desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
        )
        self.add_input(
            "cl",
            val=np.zeros((n_af, n_aoa, n_Re, n_tab)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input(
            "cd",
            val=np.zeros((n_af, n_aoa, n_Re, n_tab)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input(
            "cm",
            val=np.zeros((n_af, n_aoa, n_Re, n_tab)),
            desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the different airfoils defined in the yaml, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )

        # Airfoil coordinates
        self.add_input(
            "coord_xy",
            val=np.zeros((n_af, n_xy, 2)),
            desc="3D array of the x and y airfoil coordinates of the n_af airfoils.",
        )
        self.add_input(
            "r_thick_yaml",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )

        # Polars and coordinates interpolated along span
        self.add_output(
            "r_thick_interp",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )
        self.add_output(
            "ac_interp",
            val=np.zeros(n_span),
            desc="1D array of the aerodynamic center of the blade defined along span.",
        )
        self.add_output(
            "cl_interp",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_output(
            "cd_interp",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_output(
            "cm_interp",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_output(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Reconstruct the blade relative thickness along span with a pchip
        r_thick_used = np.zeros(self.n_af_span)
        ac_used = np.zeros(self.n_af_span)
        coord_xy_used = np.zeros((self.n_af_span, self.n_xy, 2))
        coord_xy_interp = np.zeros((self.n_span, self.n_xy, 2))
        cl_used = np.zeros((self.n_af_span, self.n_aoa, self.n_Re, self.n_tab))
        cl_interp = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        cd_used = np.zeros((self.n_af_span, self.n_aoa, self.n_Re, self.n_tab))
        cd_interp = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        cm_used = np.zeros((self.n_af_span, self.n_aoa, self.n_Re, self.n_tab))
        cm_interp = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))

        for i in range(self.n_af_span):
            for j in range(self.n_af):
                if self.af_used[i] == discrete_inputs["name"][j]:
                    r_thick_used[i] = inputs["r_thick_discrete"][j]
                    ac_used[i] = inputs["ac"][j]
                    coord_xy_used[i, :, :] = inputs["coord_xy"][j]
                    cl_used[i, :, :, :] = inputs["cl"][j, :, :, :]
                    cd_used[i, :, :, :] = inputs["cd"][j, :, :, :]
                    cm_used[i, :, :, :] = inputs["cm"][j, :, :, :]
                    break

        # Pchip does have an associated derivative method built-in:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.derivative.html#scipy.interpolate.PchipInterpolator.derivative
        spline = PchipInterpolator
        rthick_spline = spline(inputs["af_position"], r_thick_used)
        ac_spline = spline(inputs["af_position"], ac_used)
        if np.max(inputs["r_thick_yaml"]) < 1.e-6:
            rthick_spline = spline(inputs["af_position"], r_thick_used)
            outputs["r_thick_interp"] = rthick_spline(inputs["s"])
        else:
            outputs["r_thick_interp"] = inputs["r_thick_yaml"]
        ac_spline = spline(inputs["af_position"], ac_used)
        outputs["ac_interp"] = ac_spline(inputs["s"])

        # Spanwise interpolation of the profile coordinates with a pchip
        r_thick_unique, indices = np.unique(r_thick_used, return_index=True)
        profile_spline = spline(r_thick_unique, coord_xy_used[indices, :, :])
        coord_xy_interp = np.flip(profile_spline(np.flip(outputs["r_thick_interp"])), axis=0)

        for i in range(self.n_span):
            # Correction to move the leading edge (min x point) to (0,0)
            af_le = coord_xy_interp[i, np.argmin(coord_xy_interp[i, :, 0]), :]
            coord_xy_interp[i, :, 0] -= af_le[0]
            coord_xy_interp[i, :, 1] -= af_le[1]
            c = max(coord_xy_interp[i, :, 0]) - min(coord_xy_interp[i, :, 0])
            coord_xy_interp[i, :, :] /= c
            # If the rel thickness is smaller than 0.4 apply a trailing ege smoothing step
            if outputs["r_thick_interp"][i] < 0.4:
                coord_xy_interp[i, :, :] = trailing_edge_smoothing(coord_xy_interp[i, :, :])

        pitch_axis = inputs["pitch_axis"]
        chord = inputs["chord"]

        # Spanwise interpolation of the airfoil polars with a pchip
        cl_spline = spline(r_thick_unique, cl_used[indices, :, :, :])
        cl_interp = np.flip(cl_spline(np.flip(outputs["r_thick_interp"])), axis=0)
        cd_spline = spline(r_thick_unique, cd_used[indices, :, :, :])
        cd_interp = np.flip(cd_spline(np.flip(outputs["r_thick_interp"])), axis=0)
        cm_spline = spline(r_thick_unique, cm_used[indices, :, :, :])
        cm_interp = np.flip(cm_spline(np.flip(outputs["r_thick_interp"])), axis=0)

        outputs["coord_xy_interp"] = coord_xy_interp
        outputs["cl_interp"] = cl_interp
        outputs["cd_interp"] = cd_interp
        outputs["cm_interp"] = cm_interp


class Compute_Coord_XY_Dim(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        self.add_input(
            "pitch_axis",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )
        self.add_input(
            "twist",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        self.add_input(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.",
        )
        self.add_input(
            "ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

        self.add_output(
            "coord_xy_dim",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_output(
            "coord_xy_dim_twisted",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_output("wetted_area", val=0.0, units="m**2", desc="The wetted (painted) surface area of the blade")
        self.add_output("projected_area", val=0.0, units="m**2", desc="The projected surface area of the blade")

    def compute(self, inputs, outputs):
        pitch_axis = inputs["pitch_axis"]
        chord = inputs["chord"]
        twist = inputs["twist"]
        coord_xy_interp = inputs["coord_xy_interp"]

        coord_xy_dim = copy.copy(coord_xy_interp)
        coord_xy_dim[:, :, 0] -= pitch_axis[:, np.newaxis]
        coord_xy_dim = coord_xy_dim * chord[:, np.newaxis, np.newaxis]

        outputs["coord_xy_dim"] = coord_xy_dim

        coord_xy_twist = copy.copy(coord_xy_interp)
        x = coord_xy_dim[:, :, 0]
        y = coord_xy_dim[:, :, 1]
        coord_xy_twist[:, :, 0] = x * np.cos(twist[:,np.newaxis]) - y * np.sin(twist[:,np.newaxis])
        coord_xy_twist[:, :, 1] = y * np.cos(twist[:,np.newaxis]) + x * np.sin(twist[:,np.newaxis])
        outputs["coord_xy_dim_twisted"] = coord_xy_twist

        # Integrate along span for surface area
        wetted_chord = coord_xy_dim[:,:,1].max(axis=1) - coord_xy_dim[:,:,1].min(axis=1)
        outputs["wetted_area"] = np.trapz(wetted_chord, inputs["ref_axis"][:,2])

        projected_chord = coord_xy_twist[:,:,1].max(axis=1) - coord_xy_twist[:,:,1].min(axis=1)
        outputs["projected_area"] = np.trapz(projected_chord, inputs["ref_axis"][:,2])
        
            

class INN_Airfoils(om.ExplicitComponent):
    # Openmdao component to run the inverted neural network framework for airfoil design
    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("aero_shape_opt_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        aero_shape_opt_options = self.options["aero_shape_opt_options"]
        self.n_af_span = n_af_span = rotorse_options["n_af_span"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_af = n_af = rotorse_options["n_af"]  # Number of airfoils
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.n_tab = n_tab = rotorse_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        self.af_used = rotorse_options["af_used"]  # Names of the airfoils adopted along blade span

        # Polars and coordinates interpolated along span
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "r_thick",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )
        self.add_input(
            "ac_interp_yaml",
            val=np.zeros(n_span),
            desc="1D array of the aerodynamic center of the blade defined along span.",
        )
        self.add_input(
            "aoa",
            val=np.zeros(n_aoa),
            units="rad",
            desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
        )
        self.add_input(
            "cl_interp_yaml",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input(
            "cd_interp_yaml",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input(
            "cm_interp_yaml",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input(
            "coord_xy_interp_yaml",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.",
        )
        self.add_input("s_opt_r_thick", val=np.ones(aero_shape_opt_options["rthick"]["n_opt"]))
        self.add_input(
            "r_thick_opt",
            val=np.ones(aero_shape_opt_options["rthick"]["n_opt"]),
        )
        self.add_input("s_opt_L_D", val=np.ones(aero_shape_opt_options["L/D"]["n_opt"]))
        self.add_input(
            "L_D_opt",
            val=np.ones(aero_shape_opt_options["L/D"]["n_opt"]),
        )
        self.add_input("s_opt_c_d", val=np.ones(aero_shape_opt_options["c_d"]["n_opt"]))
        self.add_input(
            "c_d_opt",
            val=np.ones(aero_shape_opt_options["c_d"]["n_opt"]),
        )
        self.add_input("s_opt_stall_margin", val=np.ones(aero_shape_opt_options["stall_margin"]["n_opt"]))
        self.add_input(
            "stall_margin_opt",
            val=np.ones(aero_shape_opt_options["stall_margin"]["n_opt"]),
            units="rad",
        )
        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        self.add_input("Re", val=np.zeros(n_span), desc="Reynolds number at each blade airfoil location.")
        self.add_input("rated_TSR", val=0.0, desc="Constant tip speed ratio in region II.")

        self.add_input(
            "hub_radius",
            val=0.0,
            units="m",
            desc="Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line.",
        )
        self.add_input(
            "rotor_diameter",
            val=0.0,
            units="m",
            desc="Diameter of the rotor specified by the user. It is defined as two times the blade length plus the hub diameter.",
        )
        self.add_input(
            "z",
            val=np.ones(aero_shape_opt_options["z"]["n_opt"]),
            desc="Design parameter for INN, z.",
        )

        # Airfoil coordinates
        self.add_output(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the x and y airfoil coordinates of the n_af airfoils.",
        )
        self.add_output(
            "cl_interp",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_output(
            "cd_interp",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_output(
            "aoa_inn",
            val=np.pi * np.ones(n_span),
            desc="1D array with the operational angles of attack prescribed by the INN for the airfoils along blade span.",
            units="rad",
        )

        self.inn = INN()

    def compute(self, inputs, outputs):
        # Interpolate rthick and L/D from opt grid to full grid
        spline = PchipInterpolator
        r_thick_spline = spline(inputs["s_opt_r_thick"], inputs["r_thick_opt"])
        r_thick = r_thick_spline(inputs["s"])
        L_D_spline = spline(inputs["s_opt_L_D"], inputs["L_D_opt"])
        L_D = L_D_spline(inputs["s"])
        c_d_spline = spline(inputs["s_opt_c_d"], inputs["c_d_opt"])
        c_d = c_d_spline(inputs["s"])
        stall_margin_spline = spline(inputs["s_opt_stall_margin"], inputs["stall_margin_opt"])
        stall_margin = stall_margin_spline(inputs["s"])

        # Find indices for start and end of the optimization
        max_t_c = self.options["rotorse_options"]["inn_af_max_rthick"]
        min_t_c = self.options["rotorse_options"]["inn_af_min_rthick"]
        indices = np.argwhere(np.logical_and(r_thick > min_t_c, r_thick < max_t_c))
        indices = list(np.squeeze(indices))

        # Copy in all airfoil coordinates across the span as a starting point.
        # Some of these will be overwritten by the INN.
        outputs["coord_xy_interp"] = inputs["coord_xy_interp_yaml"]

        outputs["cl_interp"] = inputs["cl_interp_yaml"]
        outputs["cd_interp"] = inputs["cd_interp_yaml"]

        # print()
        # print("Performing INN analysis for these indices:")
        # print(indices)

        for i in indices:
            Re = inputs["Re"][i]
            if Re < 100.0:
                Re = 9.0e6
            # print(f"Querying INN at L/D {L_D[i]} and Reynolds {Re} with Cd {c_d[i]}")
            if self.options["aero_shape_opt_options"]["z"]["flag"]:
                z = inputs["z"]
            else:
                z = 314
            try:
                all_coords, alpha_inn, y_inv, z_val = self.inn.inverse_design(
                    c_d[i],
                    L_D[i],
                    np.rad2deg(stall_margin[i]),
                    r_thick[i],
                    Re,
                    N=1,
                    process_samples=True,
                    z=z,
                    return_z=True,
                    data_format="XY",
                )
            except:
                raise Exception("The INN for airfoil design failed in the inverse mode")
            alpha = np.arange(-4, 20, 0.25)
            try:
                cd, cl = self.inn.generate_polars(all_coords, Re, alpha=alpha)
            except:
                raise Exception("The INN for airfoil design failed in the forward mode")

            # print(f"inverse design completed for index {i} with a thickness of {r_thick[i]}")

            for coords in all_coords:
                # Check that airfoil points are declared from the TE suction side to TE pressure side
                idx_le = np.argmin(coords[:, 0])
                if np.mean(coords[:idx_le, 1]) > 0.0:
                    coords = np.flip(coords, axis=0)

                # Remap points using class AirfoilShape
                af = AirfoilShape(points=coords)
                af.redistribute(self.n_xy, even=False, dLE=True)
                s = af.s
                af_points = af.points

                # Add trailing edge point if not defined
                if [1, 0] not in af_points.tolist():
                    af_points[:, 0] -= af_points[np.argmin(af_points[:, 0]), 0]
                c = max(af_points[:, 0]) - min(af_points[:, 0])
                af_points[:, :] /= c

                lower = af_points[:99, 1]
                upper = af_points[101:, 1][::-1]
                diff = upper - lower
                diffo = np.min(diff)
                if diffo < 0:
                    idxs = np.where(diff < 0)[0]
                    for idx in idxs:
                        af_points[idx, 1] = upper[idx] - 1.0e-6

                outputs["coord_xy_interp"][i, :, :] = af_points

            inn_polar = Polar(Re, alpha, cl[0, :], cd[0, :], np.zeros_like(cl[0, :]))
            cdmax = 1.5
            polar = inn_polar.extrapolate(cdmax)  # Extrapolate polars for alpha between -180 deg and 180 deg

            cl_interp = PchipInterpolator(polar.alpha, polar.cl)(np.degrees(inputs["aoa"]))
            cd_interp = PchipInterpolator(polar.alpha, polar.cd)(np.degrees(inputs["aoa"]))

            for j in range(self.n_Re):
                outputs["cl_interp"][i, :, j, 0] = cl_interp
                outputs["cd_interp"][i, :, j, 0] = cd_interp

            outputs["aoa_inn"][i] = np.deg2rad(alpha_inn)


class Blade_Lofted_Shape(om.ExplicitComponent):
    # Openmdao component to generate the x, y, z coordinates of the points describing the blade outer shape.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        self.add_input(
            "ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

        self.add_input(
            "coord_xy_dim_twisted",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_output(
            "3D_shape",
            val=np.zeros((n_span * n_xy, 4)),
            units="m",
            desc="4D array of the s, and x, y, and z coordinates of the points describing the outer shape of the blade. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )

    def compute(self, inputs, outputs):
        k = 0
        for i in range(self.n_span):
            for j in range(self.n_xy):
                outputs["3D_shape"][k, :] = np.array(
                    [k, inputs["coord_xy_dim_twisted"][i, j, 1], inputs["coord_xy_dim_twisted"][i, j, 0], 0.0]
                ) + np.hstack([0, inputs["ref_axis"][i, :]])
                k = k + 1

        # Debug output
            np.savetxt(
                "3d_xyz_blade_lofted.dat",
                outputs["3D_shape"],
                header="\t point number [-]\t\t\t\t x [m] \t\t\t\t\t y [m]  \t\t\t\t z [m] \t\t\t\t The coordinate system follows the BeamDyn one.",
            )


class Blade_Internal_Structure_2D_FEM(om.Group):
    # Openmdao group with the blade internal structure data coming from the input yaml file.
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_webs = n_webs = rotorse_options["n_webs"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        ivc = self.add_subsystem("blade_2dfem_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "layer_web",
            val=np.zeros(n_layers),
            desc="1D array of the web id the layer is associated to. If the layer is on the outer profile, this entry can simply stay equal to zero.",
        )
        ivc.add_output(
            "layer_thickness",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "layer_orientation",
            val=np.zeros((n_layers, n_span)),
            units="rad",
            desc="Fiber orientation of the composite layer with 0-value meaning alignment with reference axis. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "layer_midpoint_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional midpoint defined along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        ivc.add_discrete_output(
            "layer_side",
            val=n_layers * [""],
            desc="1D array setting whether the layer is on the suction or pressure side. This entry is only used if definition_layer is equal to 1 or 2.",
        )
        ivc.add_discrete_output(
            "definition_web",
            val=np.zeros(n_webs),
            desc="1D array of flags identifying how webs are specified in the yaml. 1) offset+rotation=twist 2) offset+rotation",
        )
        ivc.add_discrete_output(
            "definition_layer",
            val=np.zeros(n_layers),
            desc="1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer",
        )
        ivc.add_discrete_output(
            "index_layer_start", val=np.zeros(n_layers), desc="Index used to fix a layer to another"
        )
        ivc.add_discrete_output("index_layer_end", val=np.zeros(n_layers), desc="Index used to fix a layer to another")

        ivc.add_output(
            "web_start_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "web_end_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "web_rotation_yaml",
            val=np.zeros((n_webs, n_span)),
            units="rad",
            desc="2D array of the rotation angle of the shear webs in respect to the chord line. The first dimension represents each shear web, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the web is built straight.",
        )
        ivc.add_output(
            "web_offset_y_pa_yaml",
            val=np.zeros((n_webs, n_span)),
            units="m",
            desc="2D array of the offset along the y axis to set the position of the shear webs. Positive values move the web towards the trailing edge, negative values towards the leading edge. The first dimension represents each shear web, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "layer_rotation_yaml",
            val=np.zeros((n_layers, n_span)),
            units="rad",
            desc="2D array of the rotation angle of a layer in respect to the chord line. The first dimension represents each layer, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the layer is built straight.",
        )
        ivc.add_output(
            "layer_start_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "layer_end_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "layer_offset_y_pa_yaml",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        ivc.add_output(
            "layer_width_yaml",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the width along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )

        ivc.add_output(
            "joint_position",
            val=0.0,
            desc="Spanwise position of the segmentation joint.",
        )
        ivc.add_output("joint_mass", val=0.0, units="kg", desc="Mass of the joint.")
        ivc.add_output("joint_nonmaterial_cost", val=0.0, units="USD", desc="Cost of the joint.")
        ivc.add_discrete_output("joint_bolt", val="M48", desc="Type of bolt: M30, M36, or M48")
        ivc.add_discrete_output(
            "reinforcement_layer_ss",
            val="joint_reinf_ss",
            desc="Layer identifier for the reinforcement layer at the join where bolts are inserted, suction side",
        )
        ivc.add_discrete_output(
            "reinforcement_layer_ps",
            val="joint_reinf_ps",
            desc="Layer identifier for the reinforcement layer at the join where bolts are inserted, pressure side",
        )

        ivc.add_output("d_f", val=0.0, units="m", desc="Diameter of the fastener")
        ivc.add_output("sigma_max", val=0.0, units="Pa", desc="Max stress on bolt")

        self.add_subsystem(
            "compute_internal_structure_2d_fem",
            Compute_Blade_Internal_Structure_2D_FEM(rotorse_options=rotorse_options),
            promotes=["*"],
        )


class Compute_Blade_Internal_Structure_2D_FEM(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_webs = n_webs = rotorse_options["n_webs"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        # From user defined yaml
        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "web_rotation_yaml",
            val=np.zeros((n_webs, n_span)),
            units="rad",
            desc="2D array of the rotation angle of the shear webs in respect to the chord line. The first dimension represents each shear web, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the web is built straight.",
        )
        self.add_input(
            "web_offset_y_pa_yaml",
            val=np.zeros((n_webs, n_span)),
            units="m",
            desc="2D array of the offset along the y axis to set the position of the shear webs. Positive values move the web towards the trailing edge, negative values towards the leading edge. The first dimension represents each shear web, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "web_start_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "web_end_nd_yaml",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )

        self.add_input(
            "layer_web",
            val=np.zeros(n_layers),
            desc="1D array of the web id the layer is associated to. If the layer is on the outer profile, this entry can simply stay equal to zero.",
        )
        self.add_input(
            "layer_thickness",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_orientation",
            val=np.zeros((n_layers, n_span)),
            units="rad",
            desc="Fiber orientation of the composite layer with 0-value meaning alignment with reference axis. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_rotation_yaml",
            val=np.zeros((n_layers, n_span)),
            units="rad",
            desc="2D array of the rotation angle of a layer in respect to the chord line. The first dimension represents each layer, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the layer is built straight.",
        )
        self.add_input(
            "layer_offset_y_pa_yaml",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_width_yaml",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the width along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_midpoint_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional midpoint defined along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_discrete_input(
            "layer_side",
            val=n_layers * [""],
            desc="1D array setting whether the layer is on the suction or pressure side. This entry is only used if definition_layer is equal to 1 or 2.",
        )
        self.add_input(
            "layer_start_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_end_nd_yaml",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_discrete_input(
            "definition_web",
            val=np.zeros(n_webs),
            desc="1D array of flags identifying how webs are specified in the yaml. 1) offset+rotation=twist 2) offset+rotation",
        )
        self.add_discrete_input(
            "definition_layer",
            val=np.zeros(n_layers),
            desc="1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer",
        )
        self.add_discrete_input(
            "index_layer_start", val=np.zeros(n_layers), desc="Index used to fix a layer to another"
        )
        self.add_discrete_input("index_layer_end", val=np.zeros(n_layers), desc="Index used to fix a layer to another")

        # From blade outer shape
        self.add_input(
            "coord_xy_dim",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_input(
            "twist",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        self.add_input(
            "pitch_axis",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )

        self.add_output(
            "web_rotation",
            val=np.zeros((n_webs, n_span)),
            units="rad",
            desc="2D array of the rotation angle of the shear webs in respect to the chord line. The first dimension represents each shear web, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the web is built straight.",
        )
        self.add_output(
            "web_start_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "web_end_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each web, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "web_offset_y_pa",
            val=np.zeros((n_webs, n_span)),
            units="m",
            desc="2D array of the offset along the y axis to set the position of the shear webs. Positive values move the web towards the trailing edge, negative values towards the leading edge. The first dimension represents each shear web, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "layer_rotation",
            val=np.zeros((n_layers, n_span)),
            units="rad",
            desc="2D array of the rotation angle of a layer in respect to the chord line. The first dimension represents each layer, the second dimension represents each entry along blade span. If the rotation is equal to negative twist +- a constant, then the layer is built straight.",
        )
        self.add_output(
            "layer_start_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "layer_end_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "layer_offset_y_pa",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_output(
            "layer_width",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the width along the outer profile of a layer. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )

        # # These outputs don't depend on anything and should be refactored to be
        # # outputs that come from an om.IndepVarComp.
        # self.declare_partials('definition_layer', '*', dependent=False)
        # self.declare_partials('layer_offset_y_pa', '*', dependent=False)
        # self.declare_partials('layer_thickness', '*', dependent=False)
        # self.declare_partials('layer_web', '*', dependent=False)
        # self.declare_partials('layer_width', '*', dependent=False)
        # self.declare_partials('s', '*', dependent=False)
        # self.declare_partials('web_offset_y_pa', '*', dependent=False)

        # self.declare_partials('layer_end_nd', ['coord_xy_dim', 'twist'], method='fd')
        # self.declare_partials('layer_midpoint_nd', ['coord_xy_dim'], method='fd')
        # self.declare_partials('layer_rotation', ['twist'], method='fd')
        # self.declare_partials('layer_start_nd', ['coord_xy_dim', 'twist'], method='fd')
        # self.declare_partials('web_end_nd', ['coord_xy_dim', 'twist'], method='fd')
        # self.declare_partials('web_rotation', ['twist'], method='fd')
        # self.declare_partials('web_start_nd', ['coord_xy_dim', 'twist'], method='fd')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Initialize temporary arrays for the outputs
        web_rotation = np.zeros((self.n_webs, self.n_span))
        layer_rotation = np.zeros((self.n_layers, self.n_span))
        web_start_nd = np.zeros((self.n_webs, self.n_span))
        web_end_nd = np.zeros((self.n_webs, self.n_span))
        layer_start_nd = np.zeros((self.n_layers, self.n_span))
        layer_end_nd = np.zeros((self.n_layers, self.n_span))

        layer_name = self.options["rotorse_options"]["layer_name"]
        layer_mat = self.options["rotorse_options"]["layer_mat"]
        web_name = self.options["rotorse_options"]["web_name"]

        # Loop through spanwise stations
        for i in range(self.n_span):
            # Compute the arc length (arc_L_i), the non-dimensional arc coordinates (xy_arc_i), and the non dimensional position of the leading edge of the profile at position i
            xy_coord_i = inputs["coord_xy_dim"][i, :, :]
            xy_arc_i = arc_length(xy_coord_i)
            arc_L_i = xy_arc_i[-1]
            xy_arc_i /= arc_L_i
            idx_le = np.argmin(xy_coord_i[:, 0])
            LE_loc = xy_arc_i[idx_le]
            chord = inputs["chord"][i]
            p_le_i = inputs["pitch_axis"][i]
            ratio_SCmax = 0.8
            ratio_Websmax = 0.75

            # Loop through the webs and compute non-dimensional start and end positions along the profile
            for j in range(self.n_webs):
                offset = inputs["web_offset_y_pa_yaml"][j, i]
                # Geometry checks on webs
                if offset < ratio_Websmax * (-chord * p_le_i) or offset > ratio_Websmax * (chord * (1.0 - p_le_i)):
                    offset_old = copy.copy(offset)
                    if offset_old <= 0.0:
                        offset = ratio_Websmax * (-chord * p_le_i)
                    else:
                        offset = ratio_Websmax * (chord * (1.0 - p_le_i))

                    outputs["web_offset_y_pa"][j, i] = copy.copy(offset)
                    layer_resize_warning = (
                        'WARNING: Web "%s" may be too large to fit within chord. "offset_x_pa" changed from %f to %f at R=%f (i=%d)'
                        % (web_name[j], offset_old, offset, inputs["s"][i], i)
                    )
                    # print(layer_resize_warning)
                else:
                    outputs["web_offset_y_pa"][j, i] = copy.copy(offset)

                if discrete_inputs["definition_web"][j] == 1:
                    web_rotation[j, i] = -inputs["twist"][i]
                    web_start_nd[j, i], web_end_nd[j, i] = calc_axis_intersection(
                        inputs["coord_xy_dim"][i, :, :],
                        -web_rotation[j, i],
                        outputs["web_offset_y_pa"][j, i],
                        [0.0, 0.0],
                        ["suction", "pressure"],
                    )
                elif discrete_inputs["definition_web"][j] == 2:
                    web_rotation[j, i] = -inputs["web_rotation_yaml"][j, i]
                    web_start_nd[j, i], web_end_nd[j, i] = calc_axis_intersection(
                        inputs["coord_xy_dim"][i, :, :],
                        -web_rotation[j, i],
                        outputs["web_offset_y_pa"][j, i],
                        [0.0, 0.0],
                        ["suction", "pressure"],
                    )
                elif discrete_inputs["definition_web"][j] == 3:
                    web_start_nd[j, i] = inputs["web_start_nd_yaml"][j, i]
                    web_end_nd[j, i] = inputs["web_end_nd_yaml"][j, i]
                elif discrete_inputs["definition_web"][j] == 4:
                    web_rotation[j, i] = -inputs["web_rotation_yaml"][j, i]
                    web_rotation[j, i] -= inputs["twist"][i]
                    web_start_nd[j, i], web_end_nd[j, i] = calc_axis_intersection(
                        inputs["coord_xy_dim"][i, :, :],
                        -web_rotation[j, i],
                        outputs["web_offset_y_pa"][j, i],
                        [0.0, 0.0],
                        ["suction", "pressure"],
                    )
                else:
                    raise ValueError(
                        "Blade web " + web_name[j] + " not described correctly. Please check the yaml input file."
                    )

            # Loop through the layers and compute non-dimensional start and end positions along the profile for the different layer definitions
            for j in range(self.n_layers):
                if discrete_inputs["definition_layer"][j] == 1:  # All around
                    layer_start_nd[j, i] = 0.0
                    layer_end_nd[j, i] = 1.0
                elif (
                    discrete_inputs["definition_layer"][j] == 2 
                    or discrete_inputs["definition_layer"][j] == 3
                    or discrete_inputs["definition_layer"][j] == 13
                ):  # Midpoint and width
                    if discrete_inputs["definition_layer"][j] == 2:
                        layer_rotation[j, i] = -inputs["twist"][i]
                    elif discrete_inputs["definition_layer"][j] == 3:
                        layer_rotation[j, i] = -inputs["layer_rotation_yaml"][j, i]
                    else:
                        layer_rotation[j, i] = -inputs["layer_rotation_yaml"][j, i]
                        layer_rotation[j, i] -= inputs["twist"][i]
                    midpoint = calc_axis_intersection(
                        inputs["coord_xy_dim"][i, :, :],
                        -layer_rotation[j, i],
                        inputs["layer_offset_y_pa_yaml"][j, i],
                        [0.0, 0.0],
                        [discrete_inputs["layer_side"][j]],
                    )[0]

                    # Geometry check to make sure the spar caps does not exceed 80% of the chord
                    width = inputs["layer_width_yaml"][j, i]
                    offset = inputs["layer_offset_y_pa_yaml"][j, i]
                    if (
                        offset + 0.5 * width > ratio_SCmax * chord * (1.0 - p_le_i)
                        or offset - 0.5 * width < -ratio_SCmax * chord * p_le_i
                    ):  # hitting TE or LE
                        width_old = copy.copy(width)
                        width = 2.0 * min([ratio_SCmax * (chord * p_le_i), ratio_SCmax * (chord * (1.0 - p_le_i))])
                        offset = 0.0
                        outputs["layer_width"][j, i] = copy.copy(width)
                        outputs["layer_offset_y_pa"][j, i] = copy.copy(offset)
                        layer_resize_warning = (
                            'WARNING: Layer "%s" may be too large to fit within chord. "offset_y_pa" changed from %f to 0.0 and "width" changed from %f to %f at s=%f (i=%d)'
                            % (layer_name[j], offset, width_old, width, inputs["s"][i], i)
                        )
                    else:
                        outputs["layer_width"][j, i] = copy.copy(width)
                        outputs["layer_offset_y_pa"][j, i] = copy.copy(offset)

                    layer_start_nd[j, i] = midpoint - width / arc_L_i / 2.0
                    layer_end_nd[j, i] = midpoint + width / arc_L_i / 2.0
                elif discrete_inputs["definition_layer"][j] == 4:  # Midpoint and width
                    midpoint = 1.0
                    inputs["layer_midpoint_nd"][j, i] = midpoint
                    width = inputs["layer_width_yaml"][j, i]
                    outputs["layer_width"][j, i] = copy.copy(width)
                    layer_start_nd[j, i] = midpoint - width / arc_L_i / 2.0
                    layer_end_nd[j, i] = width / arc_L_i / 2.0
                elif discrete_inputs["definition_layer"][j] == 5:  # Midpoint and width
                    midpoint = LE_loc
                    inputs["layer_midpoint_nd"][j, i] = midpoint
                    width = inputs["layer_width_yaml"][j, i]
                    outputs["layer_width"][j, i] = copy.copy(width)
                    layer_start_nd[j, i] = midpoint - width / arc_L_i / 2.0
                    layer_end_nd[j, i] = midpoint + width / arc_L_i / 2.0
                elif discrete_inputs["definition_layer"][j] == 6:  # Start and end locked to other element
                    layer_start_nd[j, i] = layer_end_nd[int(discrete_inputs["index_layer_start"][j]), i]
                    layer_end_nd[j, i] = layer_start_nd[int(discrete_inputs["index_layer_end"][j]), i]
                elif discrete_inputs["definition_layer"][j] == 7:  # Start nd and width
                    width = inputs["layer_width_yaml"][j, i]
                    outputs["layer_width"][j, i] = copy.copy(width)
                    layer_start_nd[j, i] = inputs["layer_start_nd_yaml"][j, i]
                    layer_end_nd[j, i] = layer_start_nd[j, i] + width / arc_L_i
                elif discrete_inputs["definition_layer"][j] == 8:  # End nd and width
                    width = inputs["layer_width_yaml"][j, i]
                    outputs["layer_width"][j, i] = copy.copy(width)
                    layer_end_nd[j, i] = inputs["layer_end_nd_yaml"][j, i]
                    layer_start_nd[j, i] = layer_end_nd[j, i] - width / arc_L_i
                elif discrete_inputs["definition_layer"][j] == 9:  # Start and end nd positions
                    layer_start_nd[j, i] = inputs["layer_start_nd_yaml"][j, i]
                    layer_end_nd[j, i] = inputs["layer_end_nd_yaml"][j, i]
                elif discrete_inputs["definition_layer"][j] == 10:  # Web layer
                    pass
                elif discrete_inputs["definition_layer"][j] == 11:  # Start nd arc locked to LE
                    layer_start_nd[j, i] = LE_loc + 1.0e-6
                    layer_end_nd[j, i] = layer_start_nd[int(discrete_inputs["index_layer_end"][j]), i]
                elif discrete_inputs["definition_layer"][j] == 12:  # End nd arc locked to LE
                    layer_end_nd[j, i] = LE_loc - 1.0e-6
                    layer_start_nd[j, i] = layer_end_nd[int(discrete_inputs["index_layer_start"][j]), i]
                else:
                    raise ValueError(
                        "Blade layer "
                        + str(layer_name[j])
                        + " not described correctly. Please check the yaml input file."
                    )

        # Assign openmdao outputs
        outputs["web_rotation"] = web_rotation
        outputs["web_start_nd"] = web_start_nd
        outputs["web_end_nd"] = web_end_nd
        outputs["layer_rotation"] = layer_rotation
        outputs["layer_start_nd"] = layer_start_nd
        outputs["layer_end_nd"] = layer_end_nd


def calc_axis_intersection(xy_coord, rotation, offset, p_le_d, side, thk=0.0):
    # dimentional analysis that takes a rotation and offset from the pitch axis and calculates the airfoil intersection
    # rotation
    offset_x = offset * np.cos(rotation) + p_le_d[0]
    offset_y = offset * np.sin(rotation) + p_le_d[1]

    m_rot = np.sin(rotation) / np.cos(rotation)  # slope of rotated axis
    plane_rot = [m_rot, -1 * m_rot * p_le_d[0] + p_le_d[1]]  # coefficients for rotated axis line: a1*x + a0

    m_intersection = np.sin(rotation + np.pi / 2.0) / np.cos(
        rotation + np.pi / 2.0
    )  # slope perpendicular to rotated axis
    plane_intersection = [
        m_intersection,
        -1 * m_intersection * offset_x + offset_y,
    ]  # coefficients for line perpendicular to rotated axis line at the offset: a1*x + a0

    # intersection between airfoil surface and the line perpendicular to the rotated/offset axis
    y_intersection = np.polyval(plane_intersection, xy_coord[:, 0])

    idx_le = np.argmin(xy_coord[:, 0])
    xy_coord_arc = arc_length(xy_coord)
    arc_L = xy_coord_arc[-1]
    xy_coord_arc /= arc_L

    idx_inter = np.argwhere(
        np.diff(np.sign(xy_coord[:, 1] - y_intersection))
    ).flatten()  # find closest airfoil surface points to intersection

    midpoint_arc = []
    for sidei in side:
        if sidei.lower() == "suction":
            tangent_line = np.polyfit(
                xy_coord[idx_inter[0] : idx_inter[0] + 2, 0], xy_coord[idx_inter[0] : idx_inter[0] + 2, 1], 1
            )
        elif sidei.lower() == "pressure":
            tangent_line = np.polyfit(
                xy_coord[idx_inter[1] : idx_inter[1] + 2, 0], xy_coord[idx_inter[1] : idx_inter[1] + 2, 1], 1
            )

        midpoint_x = (tangent_line[1] - plane_intersection[1]) / (plane_intersection[0] - tangent_line[0])
        midpoint_y = (
            plane_intersection[0]
            * (tangent_line[1] - plane_intersection[1])
            / (plane_intersection[0] - tangent_line[0])
            + plane_intersection[1]
        )

        # convert to arc position
        if sidei.lower() == "suction":
            x_half = xy_coord[: idx_le + 1, 0]
            arc_half = xy_coord_arc[: idx_le + 1]

        elif sidei.lower() == "pressure":
            x_half = xy_coord[idx_le:, 0]
            arc_half = xy_coord_arc[idx_le:]

        midpoint_arc.append(remap2grid(x_half, arc_half, midpoint_x, spline=interp1d))

    return midpoint_arc


class Hub(om.Group):
    # Openmdao group with the hub data coming from the input yaml file.
    def initialize(self):
        self.options.declare("flags")

    def setup(self):
        ivc = self.add_subsystem("hub_indep_vars", om.IndepVarComp(), promotes=["*"])

        ivc.add_output(
            "cone",
            val=0.0,
            units="rad",
            desc="Cone angle of the rotor. It defines the angle between the rotor plane and the blade pitch axis. A standard machine has positive values.",
        )
        # ivc.add_output('drag_coeff',   val=0.0,                desc='Drag coefficient to estimate the aerodynamic forces generated by the hub.') # GB: this doesn't connect to anything
        ivc.add_output("diameter", val=0.0, units="m")
        if self.options["flags"]["hub"]:
            ivc.add_output("flange_t2shell_t", val=0.0)
            ivc.add_output("flange_OD2hub_D", val=0.0)
            ivc.add_output("flange_ID2flange_OD", val=0.0)
            ivc.add_output("hub_stress_concentration", val=0.0)
            ivc.add_discrete_output("n_front_brackets", val=0)
            ivc.add_discrete_output("n_rear_brackets", val=0)
            ivc.add_output("clearance_hub_spinner", val=0.0, units="m")
            ivc.add_output("spin_hole_incr", val=0.0)
            ivc.add_output("pitch_system_scaling_factor", val=0.54)
            ivc.add_output("hub_in2out_circ", val=1.2)
            ivc.add_discrete_output("hub_material", val="steel")
            ivc.add_discrete_output("spinner_material", val="carbon")

        exec_comp = om.ExecComp(
            "radius = 0.5 * diameter",
            units="m",
            radius={
                "desc": "Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line."
            },
        )
        self.add_subsystem("compute_radius", exec_comp, promotes=["*"])


class Compute_Grid(om.ExplicitComponent):
    """
    Compute the non-dimensional grid for a tower or monopile.

    Using the dimensional `ref_axis` array, this component computes the
    non-dimensional grid, height (vertical distance) and length (curve distance)
    of a tower or monopile.
    """

    def initialize(self):
        self.options.declare("n_height")

    def setup(self):
        n_height = self.options["n_height"]

        self.add_input(
            "ref_axis",
            val=np.zeros((n_height, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
        )

        self.add_output(
            "s",
            val=np.zeros(n_height),
            desc="1D array of the non-dimensional grid defined along the tower axis (0-tower base, 1-tower top)",
        )
        self.add_output("height", val=0.0, units="m", desc="Scalar of the tower height computed along the z axis.")
        self.add_output(
            "length",
            val=0.0,
            units="m",
            desc="Scalar of the tower length computed along its curved axis. A standard straight tower will be as high as long.",
        )
        self.add_output(
            "foundation_height",
            val=0.0,
            units="m",
            desc="Foundation height in respect to the ground level.",
        )

        # Declare all partial derivatives.
        self.declare_partials("height", "ref_axis")
        self.declare_partials("length", "ref_axis")
        self.declare_partials("s", "ref_axis")
        self.declare_partials("foundation_height", "ref_axis")

    def compute(self, inputs, outputs):
        # Compute tower height and tower length (a straight tower will be high as long)
        outputs["foundation_height"] = inputs["ref_axis"][0, 2]
        outputs["height"] = inputs["ref_axis"][-1, 2] - inputs["ref_axis"][0, 2]
        myarc = arc_length(inputs["ref_axis"])
        outputs["length"] = myarc[-1]

        if myarc[-1] > 0.0:
            outputs["s"] = myarc / myarc[-1]

    def compute_partials(self, inputs, partials):
        n_height = self.options["n_height"]
        partials["height", "ref_axis"] = np.zeros((1, n_height * 3))
        partials["height", "ref_axis"][0, -1] = 1.0
        partials["height", "ref_axis"][0, 2] = -1.0
        partials["foundation_height", "ref_axis"] = np.zeros((1, n_height * 3))
        partials["foundation_height", "ref_axis"][0, 2] = 1.0
        arc_distances, d_arc_distances_d_points = arc_length_deriv(inputs["ref_axis"])

        # The length is based on only the final point in the arc,
        # but that final point has sensitivity to all ref_axis points
        partials["length", "ref_axis"] = d_arc_distances_d_points[-1, :]

        # Do quotient rule to get the non-dimensional grid derivatives
        low_d_high = arc_distances[-1] * d_arc_distances_d_points
        high_d_low = np.outer(arc_distances, d_arc_distances_d_points[-1, :])
        partials["s", "ref_axis"] = (low_d_high - high_d_low) / arc_distances[-1] ** 2


class Monopile(om.Group):
    def initialize(self):
        self.options.declare("fixedbottomse_options")

    def setup(self):
        fixedbottomse_options = self.options["fixedbottomse_options"]
        n_height = fixedbottomse_options["n_height"]
        n_layers = fixedbottomse_options["n_layers"]

        ivc = self.add_subsystem("monopile_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "diameter",
            val=np.zeros(n_height),
            units="m",
            desc="1D array of the outer diameter values defined along the tower axis.",
        )
        ivc.add_discrete_output(
            "layer_name",
            val=n_layers * [""],
            desc="1D array of the names of the layers modeled in the tower structure.",
        )
        ivc.add_discrete_output(
            "layer_mat",
            val=n_layers * [""],
            desc="1D array of the names of the materials of each layer modeled in the tower structure.",
        )
        ivc.add_output(
            "layer_thickness",
            val=np.zeros((n_layers, n_height)),
            units="m",
            desc="2D array of the thickness of the layers of the tower structure. The first dimension represents each layer, the second dimension represents each piecewise-constant entry of the tower sections.",
        )
        ivc.add_output(
            "outfitting_factor", val=0.0, desc="Multiplier that accounts for secondary structure mass inside of tower"
        )
        ivc.add_output("transition_piece_mass", val=0.0, units="kg", desc="point mass of transition piece")
        ivc.add_output("transition_piece_cost", val=0.0, units="USD", desc="cost of transition piece")
        ivc.add_output("gravity_foundation_mass", val=0.0, units="kg", desc="extra mass of gravity foundation")

        self.add_subsystem("compute_monopile_grid", Compute_Grid(n_height=n_height), promotes=["*"])


class Jacket(om.Group):
    def initialize(self):
        self.options.declare("fixedbottomse_options")

    def setup(self):
        fixedbottomse_options = self.options["fixedbottomse_options"]
        n_bays = fixedbottomse_options["n_bays"]
        n_legs = fixedbottomse_options["n_legs"]

        ivc = self.add_subsystem("jacket_indep_vars", om.IndepVarComp(), promotes=["*"])
        ivc.add_output(
            "foot_head_ratio",
            val=1.5,
            desc="Ratio of radius of foot (bottom) of jacket to head.",
        )
        ivc.add_output(
            "r_head",
            val=0.0,
            units="m",
            desc="Radius of head (top) of jacket, in meters.",
        )
        ivc.add_output(
            "height",
            val=0.0,
            units="m",
            desc="Overall jacket height, meters.",
        )
        ivc.add_output(
            "leg_diameter",
            val=0.0,
            units="m",
            desc="Leg diameter, meters. Constant throughout each leg.",
        )
        ivc.add_output(
            "leg_thickness",
            val=0.0,
            units="m",
            desc="Leg thickness, meters. Constant throughout each leg.",
        )
        ivc.add_output(
            "brace_diameters",
            val=np.zeros((n_bays)),
            units="m",
            desc="Brace diameter, meters. Array starts at the bottom of the jacket.",
        )
        ivc.add_output(
            "brace_thicknesses",
            val=np.zeros((n_bays)),
            units="m",
            desc="Brace thickness, meters. Array starts at the bottom of the jacket.",
        )
        ivc.add_output(
            "bay_spacing",
            val=np.zeros((n_bays + 1)),
            desc="Bay nodal spacing. Array starts at the bottom of the jacket.",
        )
        ivc.add_output(
            "outfitting_factor", val=0.0, desc="Multiplier that accounts for secondary structure mass inside of jacket"
        )
        ivc.add_output("transition_piece_mass", val=0.0, units="kg", desc="point mass of transition piece")
        ivc.add_output("transition_piece_cost", val=0.0, units="USD", desc="cost of transition piece")
        ivc.add_output("gravity_foundation_mass", val=0.0, units="kg", desc="extra mass of gravity foundation")


class Floating(om.Group):
    def initialize(self):
        self.options.declare("floating_init_options")
        self.options.declare("opt_options")

    def setup(self):
        floating_init_options = self.options["floating_init_options"]
        float_opt = self.options["opt_options"]["design_variables"]["floating"]
        n_joints = floating_init_options["joints"]["n_joints"]
        n_members = floating_init_options["members"]["n_members"]

        jivc = self.add_subsystem("joints", om.IndepVarComp(), promotes=["*"])
        jivc.add_output("location_in", val=np.zeros((n_joints, 3)), units="m")
        jivc.add_output("transition_node", val=np.zeros(3), units="m")
        jivc.add_output("transition_piece_mass", val=0.0, units="kg", desc="point mass of transition piece")
        jivc.add_output("transition_piece_cost", val=0.0, units="USD", desc="cost of transition piece")

        # Additions for optimizing individual nodes or multiple nodes concurrently
        self.add_subsystem("nodedv", NodeDVs(options=floating_init_options["joints"]), promotes=["*"])
        for k in range(len(floating_init_options["joints"]["design_variable_data"])):
            jivc.add_output(f"jointdv_{k}", val=0.0, units="m")

        # Members added in groups to allow for symmetry
        member_link_data = floating_init_options["members"]["linked_members"]
        for k in range(len(member_link_data)):
            name_member = member_link_data[k][0]
            memidx = floating_init_options["members"]["name"].index(name_member)
            n_geom = floating_init_options["members"]["n_geom"][memidx]
            n_height = floating_init_options["members"]["n_height"][memidx]
            n_layers = floating_init_options["members"]["n_layers"][memidx]
            n_ballasts = floating_init_options["members"]["n_ballasts"][memidx]
            n_bulkheads = floating_init_options["members"]["n_bulkheads"][memidx]
            n_axial_joints = floating_init_options["members"]["n_axial_joints"][memidx]

            ivc = self.add_subsystem(f"memgrp{k}", om.IndepVarComp())
            ivc.add_output("s_in", val=np.zeros(n_geom))
            ivc.add_output("s", val=np.zeros(n_height))

            member_shape_assigned = False
            for i, kgrp in enumerate(float_opt["members"]["groups"]):
                memname = kgrp["names"][0]
                idx = floating_init_options["members"]["name2idx"][memname]
                if idx == k:
                    if "diameter" in float_opt["members"]["groups"][i]:
                        if float_opt["members"]["groups"][i]["diameter"]["constant"]:
                            ivc.add_output("outer_diameter_in", val=0.0, units="m")
                        else:
                            ivc.add_output("outer_diameter_in", val=np.zeros(n_geom), units="m")
                        ivc.add_output("ca_usr_geom", val=-1.0*np.ones(n_geom))
                        ivc.add_output("cd_usr_geom", val=-1.0*np.ones(n_geom))
                        member_shape_assigned = True
                    if "side_length_a" in float_opt["members"]["groups"][i]:
                        if float_opt["members"]["groups"][i]["side_length_a"]["constant"]:
                            ivc.add_output("side_length_a_in", val=0.0, units="m")
                        else:
                            ivc.add_output("side_length_a_in", val=np.zeros(n_geom), units="m")
                        member_shape_assigned = True
                        ivc.add_output("ca_usr_geom", val=-1.0*np.ones(n_geom))
                        ivc.add_output("cd_usr_geom", val=-1.0*np.ones(n_geom))
                    if "side_length_b" in float_opt["members"]["groups"][i]:
                        if float_opt["members"]["groups"][i]["side_length_b"]["constant"]:
                            ivc.add_output("side_length_b_in", val=0.0, units="m")
                        else:
                            ivc.add_output("side_length_b_in", val=np.zeros(n_geom), units="m")
                        ivc.add_output("cay_usr_geom", val=-1.0*np.ones(n_geom))
                        ivc.add_output("cdy_usr_geom", val=-1.0*np.ones(n_geom))
                        member_shape_assigned = True

            if not member_shape_assigned:
                # Use the memidx to query the correct member_shape
                if floating_init_options["members"]["outer_shape"][memidx] == "circular":
                    ivc.add_output("outer_diameter_in", val=np.zeros(n_geom), units="m")
                    ivc.add_output("ca_usr_geom", val=-1.0*np.ones(n_geom))
                    ivc.add_output("cd_usr_geom", val=-1.0*np.ones(n_geom))
                elif floating_init_options["members"]["outer_shape"][memidx] == "rectangular":
                    ivc.add_output("side_length_a_in", val=np.zeros(n_geom), units="m")
                    ivc.add_output("side_length_b_in", val=np.zeros(n_geom), units="m")
                    ivc.add_output("ca_usr_geom", val=-1.0*np.ones(n_geom))
                    ivc.add_output("cd_usr_geom", val=-1.0*np.ones(n_geom))
                    ivc.add_output("cay_usr_geom", val=-1.0*np.ones(n_geom))
                    ivc.add_output("cdy_usr_geom", val=-1.0*np.ones(n_geom))

            ivc.add_discrete_output("layer_materials", val=[""] * n_layers)
            ivc.add_output("layer_thickness_in", val=np.zeros((n_layers, n_geom)), units="m")
            ivc.add_output("bulkhead_grid", val=np.zeros(n_bulkheads))
            ivc.add_output("bulkhead_thickness", val=np.zeros(n_bulkheads), units="m")
            ivc.add_output("ballast_grid", val=np.zeros((n_ballasts, 2)))
            ivc.add_output("ballast_volume", val=np.zeros(n_ballasts), units="m**3")
            ivc.add_discrete_output("ballast_materials", val=[""] * n_ballasts)
            ivc.add_output("grid_axial_joints", val=np.zeros(n_axial_joints))
            ivc.add_output("outfitting_factor", 0.0)
            ivc.add_output("ring_stiffener_web_height", 0.0, units="m")
            ivc.add_output("ring_stiffener_web_thickness", 0.0, units="m")
            ivc.add_output("ring_stiffener_flange_width", 0.0, units="m")
            ivc.add_output("ring_stiffener_flange_thickness", 0.0, units="m")
            ivc.add_output("ring_stiffener_spacing", 0.0)
            ivc.add_output("axial_stiffener_web_height", 0.0, units="m")
            ivc.add_output("axial_stiffener_web_thickness", 0.0, units="m")
            ivc.add_output("axial_stiffener_flange_width", 0.0, units="m")
            ivc.add_output("axial_stiffener_flange_thickness", 0.0, units="m")
            ivc.add_output("axial_stiffener_spacing", 0.0, units="rad")

            # Use the memidx to query the correct member_shape
            self.add_subsystem(f"memgrid{k}", MemberGrid(n_height=n_height, n_geom=n_geom, n_layers=n_layers, member_shape=floating_init_options["members"]["outer_shape"][memidx]))
            self.connect(f"memgrp{k}.s_in", f"memgrid{k}.s_in")
            self.connect(f"memgrp{k}.s", f"memgrid{k}.s_grid")
            # Here looping all dv member groups
            if floating_init_options["members"]["outer_shape"][memidx] == "circular":
                self.connect(f"memgrp{k}.outer_diameter_in", f"memgrid{k}.outer_diameter_in")
                self.connect(f"memgrp{k}.ca_usr_geom", f"memgrid{k}.ca_usr_geom")
                self.connect(f"memgrp{k}.cd_usr_geom", f"memgrid{k}.cd_usr_geom")
            elif floating_init_options["members"]["outer_shape"][memidx] == "rectangular":
                self.connect(f"memgrp{k}.side_length_a_in", f"memgrid{k}.side_length_a_in")
                self.connect(f"memgrp{k}.side_length_b_in", f"memgrid{k}.side_length_b_in")
                self.connect(f"memgrp{k}.ca_usr_geom", f"memgrid{k}.ca_usr_geom")
                self.connect(f"memgrp{k}.cd_usr_geom", f"memgrid{k}.cd_usr_geom")
                self.connect(f"memgrp{k}.cay_usr_geom", f"memgrid{k}.cay_usr_geom")
                self.connect(f"memgrp{k}.cdy_usr_geom", f"memgrid{k}.cdy_usr_geom")

            self.connect(f"memgrp{k}.layer_thickness_in", f"memgrid{k}.layer_thickness_in")

        self.add_subsystem("alljoints", AggregateJoints(floating_init_options=floating_init_options), promotes=["*"])

        for i in range(n_members):
            name_member = floating_init_options["members"]["name"][i]
            idx = floating_init_options["members"]["name2idx"][name_member]
            self.connect(f"memgrp{idx}.grid_axial_joints", "member_" + name_member + ":grid_axial_joints")
            if floating_init_options["members"]["outer_shape"][i] == "circular":
                self.connect(f"memgrid{idx}.outer_diameter", "member_" + name_member + ":outer_diameter")
            elif floating_init_options["members"]["outer_shape"][i] == "rectangular":
                # TODO: AggregatedJoints hasn't included rectangular yet, so no connection now
                print("WARNING: AggregatedJoints hasn't included rectangular yet")
                # self.connect(f"memgrid{idx}.side_length_a", "member_" + name_member + ":side_length_a")
                # self.connect(f"memgrid{idx}.side_length_b", "member_" + name_member + ":side_length_b")
            self.connect(f"memgrp{idx}.s", "member_" + name_member + ":s")


# Component that links certain nodes together in a specific dimension for optimization
class NodeDVs(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_joints = opt["n_joints"]
        self.add_input("location_in", val=np.zeros((n_joints, 3)), units="m")

        for k in range(len(opt["design_variable_data"])):
            self.add_input(f"jointdv_{k}", val=0.0, units="m")

        self.add_output("location", val=np.zeros((n_joints, 3)), units="m")

    def compute(self, inputs, outputs):
        opt = self.options["options"]

        xyz = inputs["location_in"]
        for i, linked_node_dict in enumerate(opt["design_variable_data"]):
            idx = linked_node_dict["indices"]
            dim = linked_node_dict["dimension"]
            xyz[idx, dim] = inputs[f"jointdv_{i}"]

        outputs["location"] = xyz


# Component that interpolates the diameter/thickness nodes to all of the other points needed in the member discretization
# TODO: This can be cleaned by generalizing the variables, set variables as options and loop all required variables to interpolate
class MemberGrid(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_layers")
        self.options.declare("n_height")
        self.options.declare("n_geom")
        self.options.declare("member_shape")

    def setup(self):
        n_layers = self.options["n_layers"]
        n_height = self.options["n_height"]
        n_geom = self.options["n_geom"]
        member_shape = self.options["member_shape"]

        self.add_input("s_in", val=np.zeros(n_geom))
        self.add_input("s_grid", val=np.zeros(n_height))
        if member_shape == "circular":
            self.add_input("outer_diameter_in", shape_by_conn=True, units="m")
            self.add_input("ca_usr_geom", shape_by_conn=True)  
            self.add_input("cd_usr_geom", shape_by_conn=True)            
        elif member_shape == "rectangular":
            self.add_input("side_length_a_in", shape_by_conn=True, units="m")
            self.add_input("side_length_b_in", shape_by_conn=True, units="m")
            self.add_input("ca_usr_geom", shape_by_conn=True)  
            self.add_input("cd_usr_geom", shape_by_conn=True)    
            self.add_input("cay_usr_geom", shape_by_conn=True)  
            self.add_input("cdy_usr_geom", shape_by_conn=True)    

        self.add_input("layer_thickness_in", val=np.zeros((n_layers, n_geom)), units="m")

        if member_shape == "circular":
            self.add_output("outer_diameter", val=np.zeros(n_height), units="m")
            self.add_output("ca_usr_grid", val=-1.0*np.ones(n_height))
            self.add_output("cd_usr_grid", val=-1.0*np.ones(n_height))
        elif member_shape == "rectangular":
            self.add_output("side_length_a", val=np.zeros(n_height), units="m")
            self.add_output("side_length_b", val=np.zeros(n_height), units="m")
            self.add_output("ca_usr_grid", val=-1.0*np.ones(n_height))
            self.add_output("cd_usr_grid", val=-1.0*np.ones(n_height))
            self.add_output("cay_usr_grid", val=-1.0*np.ones(n_height))
            self.add_output("cdy_usr_grid", val=-1.0*np.ones(n_height))

        self.add_output("layer_thickness", val=np.zeros((n_layers, n_height)), units="m")

    def compute(self, inputs, outputs):
        n_layers = self.options["n_layers"]
        member_shape = self.options["member_shape"]

        s_in = inputs["s_in"]
        s_grid = inputs["s_grid"]

        if member_shape == "circular":
            if len(inputs["outer_diameter_in"]) > 1:
                outputs["outer_diameter"] = PchipInterpolator(s_in, inputs["outer_diameter_in"])(s_grid)
            else:
                outputs["outer_diameter"][:] = inputs["outer_diameter_in"]
            outputs["ca_usr_grid"] = PchipInterpolator(s_in, inputs["ca_usr_geom"])(s_grid)
            outputs["cd_usr_grid"] = PchipInterpolator(s_in, inputs["cd_usr_geom"])(s_grid)
        elif member_shape == "rectangular":
            if len(inputs["side_length_a_in"]) > 1:
                outputs["side_length_a"] = PchipInterpolator(s_in, inputs["side_length_a_in"])(s_grid)
                outputs["side_length_b"] = PchipInterpolator(s_in, inputs["side_length_b_in"])(s_grid)
            else:
                outputs["side_length_a"][:] = inputs["side_length_a_in"]
                outputs["side_length_b"][:] = inputs["side_length_b_in"]
            outputs["ca_usr_grid"] = PchipInterpolator(s_in, inputs["ca_usr_geom"])(s_grid)
            outputs["cd_usr_grid"] = PchipInterpolator(s_in, inputs["cd_usr_geom"])(s_grid)
            outputs["cay_usr_grid"] = PchipInterpolator(s_in, inputs["cay_usr_geom"])(s_grid)
            outputs["cdy_usr_grid"] = PchipInterpolator(s_in, inputs["cdy_usr_geom"])(s_grid)
        
        for k in range(n_layers):
            outputs["layer_thickness"][k, :] = PchipInterpolator(s_in, inputs["layer_thickness_in"][k, :])(s_grid)


class AggregateJoints(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("floating_init_options")

    def setup(self):
        floating_init_options = self.options["floating_init_options"]
        n_joints = floating_init_options["joints"]["n_joints"]
        n_joints_tot = len(floating_init_options["joints"]["name2idx"])
        n_members = floating_init_options["members"]["n_members"]

        self.add_input("location", val=np.zeros((n_joints, 3)), units="m")

        for i in range(n_members):
            iname = floating_init_options["members"]["name"][i]
            i_axial_joints = floating_init_options["members"]["n_axial_joints"][i]
            i_grid = len(floating_init_options["members"]["grid_member_" + iname])

            self.add_input("member_" + iname + ":s", val=np.zeros(i_grid))
            self.add_input("member_" + iname + ":outer_diameter", val=np.zeros(i_grid), units="m")
            self.add_input("member_" + iname + ":grid_axial_joints", val=np.zeros(i_axial_joints))

            self.add_output("member_" + iname + ":joint1", val=np.zeros(3), units="m")
            self.add_output("member_" + iname + ":joint2", val=np.zeros(3), units="m")
            self.add_output("member_" + iname + ":height", val=0.0, units="m")
            self.add_output("member_" + iname + ":s_ghost1", val=0.0)
            self.add_output("member_" + iname + ":s_ghost2", val=1.0)

        self.add_output("joints_xyz", val=np.zeros((n_joints_tot, 3)), units="m")

    def compute(self, inputs, outputs):
        # Unpack options
        floating_init_options = self.options["floating_init_options"]
        memopt = floating_init_options["members"]
        n_joints = floating_init_options["joints"]["n_joints"]
        n_members = memopt["n_members"]
        name2idx = floating_init_options["joints"]["name2idx"]
        n_joint_tot = len(name2idx)
        NULL = -9999.0

        # Unpack inputs
        locations = inputs["location"]
        joints_xyz = NULL * np.ones(outputs["joints_xyz"].shape)

        # Handle cylindrical coordinate joints
        icyl = floating_init_options["joints"]["cylindrical"]
        locations_xyz = locations.copy()
        locations_xyz[icyl, 0] = locations[icyl, 0] * np.cos(locations[icyl, 1])
        locations_xyz[icyl, 1] = locations[icyl, 0] * np.sin(locations[icyl, 1])
        joints_xyz[:n_joints, :] = locations_xyz.copy()

        # Initial biggest radius at each node
        node_r = np.zeros(n_joint_tot)
        intersects = np.zeros(n_joint_tot)

        # Now add axial joints
        member_list = list(range(n_members))
        count = n_joints
        while count < n_joint_tot:
            for k in member_list[:]:
                # Get the end joint locations for members and then compute the axial joint loc
                joint1xyz = joints_xyz[name2idx[memopt["joint1"][k]], :]
                joint2xyz = joints_xyz[name2idx[memopt["joint2"][k]], :]

                # Check if we are ready to compute xyz position of axial joints in this member
                if np.all(joint1xyz != NULL) and np.all(joint2xyz != NULL):
                    member_list.remove(k)
                else:
                    continue

                i_axial_joints = memopt["n_axial_joints"][k]
                if i_axial_joints == 0:
                    continue

                iname = memopt["name"][k]
                s = 0.5 * inputs["member_" + iname + ":s"]
                Rk = 0.5 * inputs["member_" + iname + ":outer_diameter"]
                dxyz = joint2xyz - joint1xyz

                for a in range(i_axial_joints):
                    s_axial = inputs["member_" + iname + ":grid_axial_joints"][a]
                    joints_xyz[count, :] = joint1xyz + s_axial * dxyz

                    Ra = PchipInterpolator(s, Rk)(s_axial)
                    node_r[count] = max(node_r[count], Ra)
                    intersects[count] += 1

                    count += 1

        # Record starting and ending location for each member now.
        # Also log biggest radius at each node intersection to compute ghost nodes
        for k in range(n_members):
            iname = memopt["name"][k]
            joint1id = name2idx[memopt["joint1"][k]]
            joint2id = name2idx[memopt["joint2"][k]]
            joint1xyz = joints_xyz[joint1id, :]
            joint2xyz = joints_xyz[joint2id, :]
            hk = np.sqrt(np.sum((joint2xyz - joint1xyz) ** 2))
            outputs["member_" + iname + ":joint1"] = joint1xyz
            outputs["member_" + iname + ":joint2"] = joint2xyz
            outputs["member_" + iname + ":height"] = hk

            # Largest radius at connection points for this member
            Rk = 0.5 * inputs["member_" + iname + ":outer_diameter"]
            node_r[joint1id] = max(node_r[joint1id], Rk[0])
            node_r[joint2id] = max(node_r[joint2id], Rk[-1])
            intersects[joint1id] += 1
            intersects[joint2id] += 1

        # Store the ghost node non-dimensional locations
        for k in range(n_members):
            iname = memopt["name"][k]
            joint1id = name2idx[memopt["joint1"][k]]
            joint2id = name2idx[memopt["joint2"][k]]
            hk = outputs["member_" + iname + ":height"]
            Rk = 0.5 * inputs["member_" + iname + ":outer_diameter"]
            s_ghost1 = 0.0
            s_ghost2 = 1.0
            if intersects[joint1id] > 1 and node_r[joint1id] > Rk[0]:
                s_ghost1 = node_r[joint1id] / hk
            if intersects[joint2id] > 1 and node_r[joint2id] > Rk[-1]:
                s_ghost2 = 1.0 - node_r[joint2id] / hk
            outputs["member_" + iname + ":s_ghost1"] = s_ghost1
            outputs["member_" + iname + ":s_ghost2"] = s_ghost2

        # Store outputs
        outputs["joints_xyz"] = joints_xyz


class Mooring(om.Group):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        mooring_init_options = self.options["options"]["mooring"]

        n_nodes = mooring_init_options["n_nodes"]
        n_lines = mooring_init_options["n_lines"]

        n_design = 1 if mooring_init_options["symmetric"] else n_lines

        ivc = self.add_subsystem("mooring", om.IndepVarComp(), promotes=["*"])

        ivc.add_discrete_output("node_names", val=[""] * n_nodes)
        ivc.add_discrete_output("n_lines", val=0)  # Needed for ORBIT
        ivc.add_output("nodes_location", val=np.zeros((n_nodes, 3)), units="m")
        ivc.add_output("nodes_mass", val=np.zeros(n_nodes), units="kg")
        ivc.add_output("nodes_volume", val=np.zeros(n_nodes), units="m**3")
        ivc.add_output("nodes_added_mass", val=np.zeros(n_nodes))
        ivc.add_output("nodes_drag_area", val=np.zeros(n_nodes), units="m**2")
        ivc.add_discrete_output("nodes_joint_name", val=[""] * n_nodes)
        ivc.add_output("unstretched_length_in", val=np.zeros(n_design), units="m")
        ivc.add_discrete_output("line_id", val=[""] * n_lines)
        ivc.add_output("line_diameter_in", val=np.zeros(n_design), units="m")
        ivc.add_output("line_mass_density_coeff", val=np.zeros(n_lines), units="kg/m**3")
        ivc.add_output("line_stiffness_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("line_breaking_load_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("line_cost_rate_coeff", val=np.zeros(n_lines), units="USD/m**3")
        ivc.add_output("line_transverse_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        ivc.add_output("line_tangential_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        ivc.add_output("line_transverse_drag_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("line_tangential_drag_coeff", val=np.zeros(n_lines), units="N/m**2")
        ivc.add_output("anchor_mass", val=np.zeros(n_lines), units="kg")
        ivc.add_output("anchor_cost", val=np.zeros(n_lines), units="USD")
        ivc.add_output("anchor_max_vertical_load", val=1e30 * np.ones(n_lines), units="N")
        ivc.add_output("anchor_max_lateral_load", val=1e30 * np.ones(n_lines), units="N")

        self.add_subsystem("moorprop", MooringProperties(mooring_init_options=mooring_init_options), promotes=["*"])
        self.add_subsystem("moorjoint", MooringJoints(options=self.options["options"]), promotes=["*"])


class MooringProperties(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("mooring_init_options")

    def setup(self):
        mooring_init_options = self.options["mooring_init_options"]
        n_lines = mooring_init_options["n_lines"]
        n_design = 1 if mooring_init_options["symmetric"] else n_lines

        self.add_input("unstretched_length_in", val=np.zeros(n_design), units="m")
        self.add_input("line_diameter_in", val=np.zeros(n_design), units="m")
        self.add_input("line_mass_density_coeff", val=np.zeros(n_lines), units="kg/m**3")
        self.add_input("line_stiffness_coeff", val=np.zeros(n_lines), units="N/m**2")
        self.add_input("line_breaking_load_coeff", val=np.zeros(n_lines), units="N/m**2")
        self.add_input("line_cost_rate_coeff", val=np.zeros(n_lines), units="USD/m**3")
        self.add_input("line_transverse_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        self.add_input("line_tangential_added_mass_coeff", val=np.zeros(n_lines), units="kg/m**3")
        self.add_input("line_transverse_drag_coeff", val=np.zeros(n_lines), units="N/m**2")
        self.add_input("line_tangential_drag_coeff", val=np.zeros(n_lines), units="N/m**2")

        self.add_output("unstretched_length", val=np.zeros(n_lines), units="m")
        self.add_output("line_diameter", val=np.zeros(n_lines), units="m")
        self.add_output("line_mass_density", val=np.zeros(n_lines), units="kg/m")
        self.add_output("line_stiffness", val=np.zeros(n_lines), units="N")
        self.add_output("line_breaking_load", val=np.zeros(n_lines), units="N")
        self.add_output("line_cost_rate", val=np.zeros(n_lines), units="USD/m")
        self.add_output("line_transverse_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_output("line_tangential_added_mass", val=np.zeros(n_lines), units="kg/m")
        self.add_output("line_transverse_drag", val=np.zeros(n_lines))
        self.add_output("line_tangential_drag", val=np.zeros(n_lines))

    def compute(self, inputs, outputs):
        n_lines = self.options["mooring_init_options"]["n_lines"]
        line_mat = self.options["mooring_init_options"]["line_material"]
        outputs["line_diameter"] = d = inputs["line_diameter_in"] * np.ones(n_lines)
        outputs["unstretched_length"] = inputs["unstretched_length_in"] * np.ones(n_lines)
        d2 = d * d

        line_obj = None
        if line_mat[0] == "custom":
            varlist = [
                "line_mass_density",
                "line_stiffness",
                "line_breaking_load",
                "line_cost_rate",
                "line_transverse_added_mass",
                "line_tangential_added_mass",
                "line_transverse_drag",
                "line_tangential_drag",
            ]
            for var in varlist:
                outputs[var] = d2 * inputs[var + "_coeff"]

        elif line_mat[0] == "chain_stud":
            line_obj = mp.getLineProps(1e3 * d[0], type="chain", stud="stud")
        else:
            line_obj = mp.getLineProps(1e3 * d[0], type=line_mat[0])

        if not line_obj is None:
            outputs["line_mass_density"] = line_obj.m
            outputs["line_stiffness"] = line_obj.EA
            outputs["line_breaking_load"] = line_obj.MBL
            outputs["line_cost_rate"] = line_obj.cost
            varlist = [
                "line_transverse_added_mass",
                "line_tangential_added_mass",
                "line_transverse_drag",
                "line_tangential_drag",
            ]
            for var in varlist:
                outputs[var] = d2 * inputs[var + "_coeff"]


class MooringJoints(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        mooring_init_options = self.options["options"]["mooring"]
        n_nodes = mooring_init_options["n_nodes"]
        n_attach = mooring_init_options["n_attach"]
        n_lines = mooring_init_options["n_lines"]

        self.add_discrete_input("nodes_joint_name", val=[""] * n_nodes)
        self.add_input("nodes_location", val=np.zeros((n_nodes, 3)), units="m")
        self.add_input("joints_xyz", shape_by_conn=True, units="m")

        self.add_output("mooring_nodes", val=np.zeros((n_nodes, 3)), units="m")
        self.add_output("fairlead_nodes", val=np.zeros((n_attach, 3)), units="m")
        self.add_output("fairlead", val=np.zeros(n_lines), units="m")
        self.add_output("fairlead_radius", val=np.zeros(n_attach), units="m")
        self.add_output("anchor_nodes", val=np.zeros((n_lines, 3)), units="m")
        self.add_output("anchor_radius", val=np.zeros(n_lines), units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        mooring_init_options = self.options["options"]["mooring"]
        n_nodes = mooring_init_options["n_nodes"]

        node_joints = discrete_inputs["nodes_joint_name"]
        node_loc = inputs["nodes_location"]
        joints_loc = inputs["joints_xyz"]
        idx_map = self.options["options"]["floating"]["joints"]["name2idx"]
        for k in range(n_nodes):
            if node_joints[k] == "":
                continue
            idx = idx_map[node_joints[k]]
            node_loc[k, :] = joints_loc[idx, :]
        outputs["mooring_nodes"] = node_loc

        node_loc = np.unique(node_loc, axis=0)
        tol = 0.5
        z_fair = node_loc[:, 2].max()
        z_anch = node_loc[:, 2].min()
        ifair = np.where(np.abs(node_loc[:, 2] - z_fair) < tol)[0]
        ianch = np.where(np.abs(node_loc[:, 2] - z_anch) < tol)[0]

        node_fair = node_loc[ifair, :]
        node_anch = node_loc[ianch, :]
        ang_fair = np.arctan2(node_fair[:, 1], node_fair[:, 0])
        ang_anch = np.arctan2(node_anch[:, 1], node_anch[:, 0])
        node_fair = node_fair[np.argsort(ang_fair), :]
        node_anch = node_anch[np.argsort(ang_anch), :]

        outputs["fairlead_nodes"] = node_fair
        outputs["anchor_nodes"] = node_anch
        outputs["fairlead"] = -z_fair  # Positive is defined below the waterline here
        outputs["fairlead_radius"] = np.sqrt(np.sum(node_loc[ifair, :2] ** 2, axis=1))
        outputs["anchor_radius"] = np.sqrt(np.sum(node_loc[ianch, :2] ** 2, axis=1))


class ComputeMaterialsProperties(om.ExplicitComponent):
    # Openmdao component with the wind turbine materials coming from the input yaml file. The inputs and outputs are arrays where each entry represents a material

    def initialize(self):
        self.options.declare("mat_init_options")
        self.options.declare("composites", default=True)

    def setup(self):
        mat_init_options = self.options["mat_init_options"]
        self.n_mat = n_mat = mat_init_options["n_mat"]

        self.add_discrete_input("name", val=n_mat * [""], desc="1D array of names of materials.")
        self.add_discrete_input(
            "component_id",
            val=-np.ones(n_mat),
            desc="1D array of flags to set whether a material is used in a blade: 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE/LE reinf.",
        )
        self.add_input(
            "rho_fiber",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the fibers of the materials.",
        )
        self.add_input(
            "rho",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the materials. For composites, this is the density of the laminate.",
        )
        self.add_input(
            "rho_area_dry",
            val=np.zeros(n_mat),
            units="kg/m**2",
            desc="1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "ply_t_from_yaml",
            val=np.zeros(n_mat),
            units="m",
            desc="1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "fvf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.",
        )
        self.add_input(
            "fwf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.",
        )

        self.add_output(
            "ply_t",
            val=np.zeros(n_mat),
            units="m",
            desc="1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.",
        )
        self.add_output(
            "fvf",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.",
        )
        self.add_output(
            "fwf",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        density_resin = 0.0
        for i in range(self.n_mat):
            if discrete_inputs["name"][i] == "resin":
                density_resin = inputs["rho"][i]
                id_resin = i
        if self.options["composites"] and density_resin == 0.0:
            raise Exception(
                "Warning: a material named resin is not defined in the input yaml.  This is required for blade composite analysis"
            )

        fvf = np.zeros(self.n_mat)
        fwf = np.zeros(self.n_mat)
        ply_t = np.zeros(self.n_mat)

        for i in range(self.n_mat):
            if discrete_inputs["component_id"][i] > 1:  # It's a composite
                # Formula to estimate the fiber volume fraction fvf from the laminate and the fiber densities
                fvf[i] = (inputs["rho"][i] - density_resin) / (inputs["rho_fiber"][i] - density_resin)
                if inputs["fvf_from_yaml"][i] > 0.0:
                    if abs(fvf[i] - inputs["fvf_from_yaml"][i]) > 1e-3:
                        raise ValueError(
                            "Error: the fvf of composite "
                            + discrete_inputs["name"][i]
                            + " specified in the yaml is equal to "
                            + str(inputs["fvf_from_yaml"][i] * 100)
                            + "%, but this value is not compatible to the other values provided. Given the fiber, laminate and resin densities, it should instead be equal to "
                            + str(fvf[i] * 100.0)
                            + "%."
                        )
                    else:
                        outputs["fvf"][i] = inputs["fvf_from_yaml"][i]
                else:
                    outputs["fvf"][i] = fvf[i]

                # Formula to estimate the fiber weight fraction fwf from the fiber volume fraction and the fiber densities
                fwf[i] = (
                    inputs["rho_fiber"][i]
                    * outputs["fvf"][i]
                    / (density_resin + ((inputs["rho_fiber"][i] - density_resin) * outputs["fvf"][i]))
                )
                if inputs["fwf_from_yaml"][i] > 0.0:
                    if abs(fwf[i] - inputs["fwf_from_yaml"][i]) > 1e-3:
                        raise ValueError(
                            "Error: the fwf of composite "
                            + discrete_inputs["name"][i]
                            + " specified in the yaml is equal to "
                            + str(inputs["fwf_from_yaml"][i] * 100)
                            + "%, but this value is not compatible to the other values provided. It should instead be equal to "
                            + str(fwf[i] * 100.0)
                            + "%"
                        )
                    else:
                        outputs["fwf"][i] = inputs["fwf_from_yaml"][i]
                else:
                    outputs["fwf"][i] = fwf[i]

                # Formula to estimate the plyt thickness ply_t of a laminate from the aerial density, the laminate density and the fiber weight fraction
                ply_t[i] = inputs["rho_area_dry"][i] / inputs["rho"][i] / outputs["fwf"][i]
                if inputs["ply_t_from_yaml"][i] > 0.0:
                    if abs(ply_t[i] - inputs["ply_t_from_yaml"][i]) > 1.0e-4:
                        raise ValueError(
                            "Error: the ply_t of composite "
                            + discrete_inputs["name"][i]
                            + " specified in the yaml is equal to "
                            + str(inputs["ply_t_from_yaml"][i])
                            + "m, but this value is not compatible to the other values provided. It should instead be equal to "
                            + str(ply_t[i])
                            + "m. Alternatively, adjust the aerial density to "
                            + str(outputs["ply_t"][i] * inputs["rho"][i] * outputs["fwf"][i])
                            + " kg/m2."
                        )
                    else:
                        outputs["ply_t"][i] = inputs["ply_t_from_yaml"][i]
                else:
                    outputs["ply_t"][i] = ply_t[i]


class Materials(om.Group):
    # Openmdao group with the wind turbine materials coming from the input yaml file.
    # The inputs and outputs are arrays where each entry represents a material

    def initialize(self):
        self.options.declare("mat_init_options")
        self.options.declare("composites", default=True)

    def setup(self):
        mat_init_options = self.options["mat_init_options"]
        self.n_mat = n_mat = mat_init_options["n_mat"]

        ivc = self.add_subsystem("materials_indep_vars", om.IndepVarComp(), promotes=["*"])

        ivc.add_discrete_output(
            "orth",
            val=np.zeros(n_mat),
            desc="1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.",
        )
        ivc.add_output(
            "E",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.",
        )
        ivc.add_output(
            "G",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.",
        )
        ivc.add_output(
            "nu",
            val=np.zeros([n_mat, 3]),
            desc="2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.",
        )
        ivc.add_output(
            "Xt",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.",
        )
        ivc.add_output(
            "Xc",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.",
        )
        ivc.add_output(
            "S",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Ultimate Shear Strength (USS) of the materials. Each row represents a material, the three columns represent S12, S13 and S23.",
        )
        ivc.add_output(
            "sigma_y",
            val=np.zeros(n_mat),
            units="Pa",
            desc="Yield stress of the material (in the principle direction for composites).",
        )
        ivc.add_output(
            "wohler_exp",
            val=np.zeros(n_mat),
            desc="Exponent of S-N Wohler fatigue curve in the form of S = A*N^-(1/m).",
        )
        ivc.add_output(
            "wohler_intercept",
            val=np.zeros(n_mat),
            desc="Stress-intercept (A) of S-N Wohler fatigue curve in the form of S = A*N^-(1/m), taken as ultimate stress unless otherwise specified.",
        )
        ivc.add_output(
            "unit_cost", val=np.zeros(n_mat), units="USD/kg", desc="1D array of the unit costs of the materials."
        )
        ivc.add_output(
            "waste", val=np.zeros(n_mat), desc="1D array of the non-dimensional waste fraction of the materials."
        )
        ivc.add_output(
            "roll_mass",
            val=np.zeros(n_mat),
            units="kg",
            desc="1D array of the roll mass of the composite fabrics. Non-composite materials are kept at 0.",
        )

        ivc.add_discrete_output("name", val=n_mat * [""], desc="1D array of names of materials.")
        ivc.add_discrete_output(
            "component_id",
            val=-np.ones(n_mat),
            desc="1D array of flags to set whether a material is used in a blade: 0 - coating, 1 - sandwich filler , 2 - shell skin, 3 - shear webs, 4 - spar caps, 5 - TE reinf.isotropic.",
        )
        ivc.add_output(
            "rho_fiber",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the fibers of the materials.",
        )
        ivc.add_output(
            "rho",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the materials. For composites, this is the density of the laminate.",
        )
        ivc.add_output(
            "rho_area_dry",
            val=np.zeros(n_mat),
            units="kg/m**2",
            desc="1D array of the dry aerial density of the composite fabrics. Non-composite materials are kept at 0.",
        )
        ivc.add_output(
            "ply_t_from_yaml",
            val=np.zeros(n_mat),
            units="m",
            desc="1D array of the ply thicknesses of the materials. Non-composite materials are kept at 0.",
        )
        ivc.add_output(
            "fvf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber volume fraction of the composite materials. Non-composite materials are kept at 0.",
        )
        ivc.add_output(
            "fwf_from_yaml",
            val=np.zeros(n_mat),
            desc="1D array of the non-dimensional fiber weight- fraction of the composite materials. Non-composite materials are kept at 0.",
        )

        self.add_subsystem(
            "compute_materials_properties",
            ComputeMaterialsProperties(mat_init_options=mat_init_options, composites=self.options["composites"]),
            promotes=["*"],
        )


class ComputeHighLevelBladeProperties(om.ExplicitComponent):
    # Openmdao component that computes rotor quantities, such as the rotor coordinate of the blade stations
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]

        n_span = rotorse_options["n_span"]

        self.add_input(
            "blade_ref_axis_user",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )
        self.add_input(
            "rotor_diameter_user",
            val=0.0,
            units="m",
            desc="Diameter of the rotor specified by the user. It is defined as two times the blade length plus the hub diameter.",
        )
        self.add_input(
            "hub_radius",
            val=0.0,
            units="m",
            desc="Radius of the hub. It defines the distance of the blade root from the rotor center along the coned line.",
        )

        self.add_output(
            "rotor_diameter",
            val=0.0,
            units="m",
            desc="Diameter of the rotor used in WISDEM. It is defined as two times the blade length plus the hub diameter.",
        )
        self.add_output(
            "r_blade",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the dimensional spanwise grid defined along the rotor (hub radius to blade tip projected on the plane)",
        )
        self.add_output(
            "rotor_radius",
            val=0.0,
            units="m",
            desc="Scalar of the rotor radius, defined ignoring prebend and sweep curvatures, and cone and uptilt angles.",
        )
        self.add_output(
            "blade_ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis scaled based on rotor diameter, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )
        self.add_output("prebend", val=np.zeros(n_span), units="m", desc="Blade prebend at each section")
        self.add_output("prebendTip", val=0.0, units="m", desc="Blade prebend at tip")
        self.add_output("presweep", val=np.zeros(n_span), units="m", desc="Blade presweep at each section")
        self.add_output("presweepTip", val=0.0, units="m", desc="Blade presweep at tip")
        self.add_output(
            "blade_length",
            val=0.0,
            units="m",
            desc="Scalar of the 3D blade length computed along its axis, scaled based on the user defined rotor diameter.",
        )

    def compute(self, inputs, outputs):
        outputs["blade_ref_axis"][:, 0] = inputs["blade_ref_axis_user"][:, 0]
        outputs["blade_ref_axis"][:, 1] = inputs["blade_ref_axis_user"][:, 1]
        # Scale z if the blade length provided by the user does not match the rotor diameter. D = (blade length + hub radius) * 2
        if inputs["rotor_diameter_user"] != 0.0:
            outputs["rotor_diameter"] = inputs["rotor_diameter_user"]
            outputs["blade_ref_axis"][:, 2] = (
                inputs["blade_ref_axis_user"][:, 2]
                * inputs["rotor_diameter_user"]
                / ((inputs["blade_ref_axis_user"][-1,2] + inputs["hub_radius"]) * 2.0)
            )
        # If the user does not provide a rotor diameter, this is computed from the hub diameter and the blade length
        else:
            outputs["rotor_diameter"] = (inputs["blade_ref_axis_user"][-1,2] + inputs["hub_radius"]) * 2.0
            outputs["blade_ref_axis"][:, 2] = inputs["blade_ref_axis_user"][:, 2]
        outputs["r_blade"] = outputs["blade_ref_axis"][:, 2] + inputs["hub_radius"]
        outputs["rotor_radius"] = outputs["r_blade"][-1]
        outputs["blade_length"] = arc_length(outputs["blade_ref_axis"])[-1]
        outputs["prebend"] = outputs["blade_ref_axis"][:, 0]
        outputs["prebendTip"] = outputs["blade_ref_axis"][-1, 0]
        outputs["presweep"] = outputs["blade_ref_axis"][:, 1]
        outputs["presweepTip"] = outputs["blade_ref_axis"][-1, 1]


class ComputeHighLevelTowerProperties(om.ExplicitComponent):
    # Openmdao component that computes tower quantities, such as the hub height, and the blade-tower clearance
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        if modeling_options["flags"]["tower"]:
            n_height_tower = modeling_options["WISDEM"]["TowerSE"]["n_height_tower"]
        else:
            n_height_tower = 0

        self.add_input(
            "tower_ref_axis_user",
            val=np.zeros((n_height_tower, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
        )
        self.add_input("distance_tt_hub", val=0.0, units="m", desc="Vertical distance from tower top to hub center.")
        self.add_input("hub_height_user", val=0.0, units="m", desc="Height of the hub specified by the user.")
        self.add_input(
            "rotor_diameter",
            val=0.0,
            units="m",
            desc="Diameter of the rotor used in WISDEM. It is defined as two times the blade length plus the hub diameter.",
        )

        self.add_output(
            "tower_ref_axis",
            val=np.zeros((n_height_tower, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate system is the global coordinate system of OpenFAST: it is placed at tower base with x pointing downwind, y pointing on the side and z pointing vertically upwards. A standard tower configuration will have zero x and y values and positive z values.",
        )
        self.add_output(
            "hub_height",
            val=0.0,
            units="m",
            desc="Height of the hub in the global reference system, i.e. distance rotor center to ground.",
        )

    def compute(self, inputs, outputs):
        modeling_options = self.options["modeling_options"]
        if inputs["hub_height_user"][0] != 0.0:
            outputs["hub_height"] = inputs["hub_height_user"]
            
        if modeling_options["flags"]["tower"]:
            if inputs["hub_height_user"][0] != 0.0:
                z_base = inputs["tower_ref_axis_user"][0, 2]
                z_current = inputs["tower_ref_axis_user"][:, 2] - z_base
                h_needed = inputs["hub_height_user"] - inputs["distance_tt_hub"] - z_base
                z_new = z_current * h_needed / z_current[-1]
                outputs["tower_ref_axis"][:, 2] = z_new + z_base
            else:
                outputs["hub_height"] = inputs["tower_ref_axis_user"][-1, 2] + inputs["distance_tt_hub"]
                outputs["tower_ref_axis"] = inputs["tower_ref_axis_user"]

        if outputs["hub_height"][0] == 0.0:
            raise Exception("The hub height cannot be set.  Please set it in the top level 'assembly' section in the yaml file and/or define the tower reference axis")
        
        if modeling_options["flags"]["blade"] and inputs["rotor_diameter"] > 2.0 * outputs["hub_height"]:
            raise Exception(
                "The rotor blade extends past the ground or water line. Please adjust hub height and/or rotor diameter."
            )


class Airfoil3DCorrection(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.af_correction = rotorse_options["3d_af_correction"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.n_tab = n_tab = rotorse_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.add_input(
            "aoa",
            val=np.zeros(n_aoa),
            units="rad",
            desc="1D array of the angles of attack used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
        )
        self.add_input(
            "Re",
            val=np.zeros(n_Re),
            desc="1D array of the Reynolds numbers used to define the polars of the airfoils. All airfoils defined in openmdao share this grid.",
        )
        self.add_input(
            "cl",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the lift coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input(
            "cd",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the drag coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input(
            "cm",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="4D array with the moment coefficients of the airfoils. Dimension 0 is along the blade span for n_span stations, dimension 1 is along the angles of attack, dimension 2 is along the Reynolds number, dimension 3 is along the number of tabs, which may describe multiple sets at the same station, for example in presence of a flap.",
        )
        self.add_input("rated_TSR", val=0.0, desc="Constant tip speed ratio in region II.")
        self.add_input(
            "r_blade",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the dimensional spanwise grid defined along the rotor (hub radius to blade tip projected on the plane)",
        )
        self.add_input(
            "rotor_radius",
            val=0.0,
            units="m",
            desc="Scalar of the rotor radius, defined ignoring prebend and sweep curvatures, and cone and uptilt angles.",
        )
        self.add_input(
            "r_thick",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span.",
        )
        self.add_input(
            "chord", val=np.zeros(n_span), units="m", desc="1D array of the chord values defined along blade span."
        )
        # Outputs
        self.add_output(
            "cl_corrected",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="Lift coefficient corrected with CCBlade.Polar.",
        )
        self.add_output(
            "cd_corrected",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="Drag coefficient corrected with CCBlade.Polar.",
        )
        self.add_output(
            "cm_corrected",
            val=np.zeros((n_span, n_aoa, n_Re, n_tab)),
            desc="Moment coefficient corrected with CCblade.Polar.",
        )

    def compute(self, inputs, outputs):
        cl_corrected = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        cd_corrected = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        cm_corrected = np.zeros((self.n_span, self.n_aoa, self.n_Re, self.n_tab))
        for i in range(self.n_span):
            if (
                inputs["r_thick"][i] < 0.7 and self.af_correction
            ):  # Only apply 3D correction to airfoils thinner than 70% to avoid numerical problems at blade root
                logger.info("3D correction applied to airfoil polars for section " + str(i))
                for j in range(self.n_Re):
                    for k in range(self.n_tab):
                        inn_polar = Polar(
                            Re=inputs["Re"][j],
                            alpha=np.degrees(inputs["aoa"]),
                            cl=inputs["cl"][i, :, j, k],
                            cd=inputs["cd"][i, :, j, k],
                            cm=inputs["cm"][i, :, j, k],
                        )
                        polar3d = inn_polar.correction3D(
                            inputs["r_blade"][i] / inputs["rotor_radius"],
                            inputs["chord"][i] / inputs["r_blade"][i],
                            inputs["rated_TSR"],
                        )
                        cl_corrected[i, :, j, k] = PchipInterpolator(polar3d.alpha, polar3d.cl)(
                            np.degrees(inputs["aoa"])
                        )
                        cd_corrected[i, :, j, k] = PchipInterpolator(polar3d.alpha, polar3d.cd)(
                            np.degrees(inputs["aoa"])
                        )
                        cm_corrected[i, :, j, k] = PchipInterpolator(polar3d.alpha, polar3d.cm)(
                            np.degrees(inputs["aoa"])
                        )
            else:
                cl_corrected[i, :, :, :] = inputs["cl"][i, :, :, :]
                cd_corrected[i, :, :, :] = inputs["cd"][i, :, :, :]
                cm_corrected[i, :, :, :] = inputs["cm"][i, :, :, :]
        outputs["cl_corrected"] = cl_corrected
        outputs["cd_corrected"] = cd_corrected
        outputs["cm_corrected"] = cm_corrected
