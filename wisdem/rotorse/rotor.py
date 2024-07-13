import openmdao.api as om

from wisdem.rotorse.blade_cost import BladeCost, BladeSplit, TotalBladeCosts
from wisdem.rotorse.rotor_power import RotorPower, NoStallConstraint
from wisdem.commonse.turbine_class import TurbineClass
from wisdem.rotorse.rotor_structure import RotorStructure
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from wisdem.ccblade.ccblade_component import CCBladeTwist


class RotorSEProp(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]

        ivc = om.IndepVarComp()
        ivc.add_discrete_output("hubloss", val=modeling_options["WISDEM"]["RotorSE"]["hubloss"])
        ivc.add_discrete_output("tiploss", val=modeling_options["WISDEM"]["RotorSE"]["tiploss"])
        ivc.add_discrete_output("wakerotation", val=modeling_options["WISDEM"]["RotorSE"]["wakerotation"])
        ivc.add_discrete_output("usecd", val=modeling_options["WISDEM"]["RotorSE"]["usecd"])
        ivc.add_discrete_output("nSector", val=modeling_options["WISDEM"]["RotorSE"]["n_sector"])
        self.add_subsystem("ivc", ivc, promotes=["*"])
        
        promoteGeom = [
            "A",
            "EA",
            "EIxx",
            "EIyy",
            "EIxy",
            "GJ",
            "rhoA",
            "rhoJ",
            "x_ec",
            "y_ec",
            "xu_spar",
            "xl_spar",
            "yu_spar",
            "yl_spar",
            "xu_te",
            "xl_te",
            "yu_te",
            "yl_te",
        ]

        promoteCC = [
            "chord",
            "theta",
            "r",
            "Rtip",
            "Rhub",
            "hub_height",
            "precone",
            "tilt",
            "precurve",
            "presweep",
            "airfoils_aoa",
            "airfoils_Re",
            "airfoils_cl",
            "airfoils_cd",
            "airfoils_cm",
            "nBlades",
            ("rho", "rho_air"),
            ("mu", "mu_air"),
            "shearExp",
            "hubloss",
            "tiploss",
            "wakerotation",
            "usecd",
            "nSector",
            "yaw",
        ]

        self.add_subsystem(
            "ccblade",
            CCBladeTwist(modeling_options=modeling_options, opt_options=opt_options),
            promotes=promoteCC + ["pitch", "tsr", "precurveTip", "presweepTip"],
        )  # Run standalone CCBlade and possibly determine optimal twist from user-defined margin to stall

        self.add_subsystem("wt_class", TurbineClass())

        re_promote_add = ["chord", "theta", "r", "precurve", "presweep",
                          "blade_mass", "blade_span_cg", "blade_moment_of_inertia",
                          "mass_all_blades", "I_all_blades"]
        self.add_subsystem(
            "re",
            RotorElasticity(modeling_options=modeling_options, opt_options=opt_options),
            promotes=promoteGeom + re_promote_add,
        )

        if not modeling_options["WISDEM"]["RotorSE"]["bjs"]:
            n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
            self.add_subsystem(
                "rc", BladeCost(mod_options=modeling_options, opt_options=opt_options, n_span=n_span, root=True)
            )

            self.add_subsystem("total_bc", TotalBladeCosts())

            self.connect("rc.total_blade_cost", "total_bc.inner_blade_cost")



class RotorSEPerf(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]

        promoteGeom = [
            "A",
            "EA",
            "EIxx",
            "EIyy",
            "EIxy",
            "GJ",
            "rhoA",
            "rhoJ",
            "x_ec",
            "y_ec",
            "xu_spar",
            "xl_spar",
            "yu_spar",
            "yl_spar",
            "xu_te",
            "xl_te",
            "yu_te",
            "yl_te",
        ]

        promoteCC = [
            "chord",
            "theta",
            "r",
            "Rtip",
            "Rhub",
            "hub_height",
            "precone",
            "tilt",
            "precurve",
            "presweep",
            "airfoils_aoa",
            "airfoils_Re",
            "airfoils_cl",
            "airfoils_cd",
            "airfoils_cm",
            "nBlades",
            ("rho", "rho_air"),
            ("mu", "mu_air"),
            "shearExp",
            "hubloss",
            "tiploss",
            "wakerotation",
            "usecd",
            "nSector",
            "yaw",
        ]

        self.add_subsystem(
            "rp",
            RotorPower(modeling_options=modeling_options),
            promotes=promoteCC + ["precurveTip", "presweepTip", ("tsr_operational", "tsr"), ("control_pitch", "pitch")],
        )

        self.add_subsystem(
            "stall_check",
            NoStallConstraint(modeling_options=modeling_options),
            promotes=["s", "airfoils_aoa", "airfoils_cl", "airfoils_cd", "airfoils_cm"],
        )

        self.add_subsystem(
            "rs",
            RotorStructure(modeling_options=modeling_options, opt_options=opt_options, freq_run=False),
            promotes=promoteGeom + promoteCC + ["s", "precurveTip", "presweepTip", "blade_span_cg"],
        )

        if modeling_options["WISDEM"]["RotorSE"]["bjs"]:
            self.add_subsystem("split", BladeSplit(mod_options=modeling_options, opt_options=opt_options))
            n_span_in = modeling_options["WISDEM"]["RotorSE"]["id_joint_position"] + 1
            n_span_out = (
                modeling_options["WISDEM"]["RotorSE"]["n_span"]
                - modeling_options["WISDEM"]["RotorSE"]["id_joint_position"]
            )
            self.add_subsystem(
                "rc_in", BladeCost(mod_options=modeling_options, opt_options=opt_options, n_span=n_span_in, root=True)
            )
            self.add_subsystem(
                "rc_out",
                BladeCost(mod_options=modeling_options, opt_options=opt_options, n_span=n_span_out, root=False),
            )
            # Inner blade portion inputs
            self.connect("split.blade_length_inner", "rc_in.blade_length")
            self.connect("split.s_inner", "rc_in.s")
            self.connect("split.chord_inner", "rc_in.chord")
            self.connect("split.coord_xy_interp_inner", "rc_in.coord_xy_interp")
            self.connect("split.layer_thickness_inner", "rc_in.layer_thickness")
            self.connect("split.layer_start_nd_inner", "rc_in.layer_start_nd")
            self.connect("split.layer_end_nd_inner", "rc_in.layer_end_nd")
            self.connect("split.web_start_nd_inner", "rc_in.web_start_nd")
            self.connect("split.web_end_nd_inner", "rc_in.web_end_nd")
            # Outer blade portion inputs
            self.connect("split.blade_length_outer", "rc_out.blade_length")
            self.connect("split.s_outer", "rc_out.s")
            self.connect("split.chord_outer", "rc_out.chord")
            self.connect("split.coord_xy_interp_outer", "rc_out.coord_xy_interp")
            self.connect("split.layer_thickness_outer", "rc_out.layer_thickness")
            self.connect("split.layer_start_nd_outer", "rc_out.layer_start_nd")
            self.connect("split.layer_end_nd_outer", "rc_out.layer_end_nd")
            self.connect("split.web_start_nd_outer", "rc_out.web_start_nd")
            self.connect("split.web_end_nd_outer", "rc_out.web_end_nd")

            self.add_subsystem("total_bc", TotalBladeCosts())

            self.connect("rc_in.total_blade_cost", "total_bc.inner_blade_cost")
            self.connect("rc_out.total_blade_cost", "total_bc.outer_blade_cost")
            self.connect("rs.bjs.joint_total_cost", "total_bc.joint_cost")

        # Connection from ra to rs for the rated conditions
        self.connect("rp.gust.V_gust", ["rs.aero_gust.V_load", "rs.aero_hub_loads.V_load"])
        self.connect(
            "rp.powercurve.rated_Omega", ["rs.Omega_load", "rs.tot_loads_gust.aeroloads_Omega", "rs.constr.rated_Omega"]
        )
        self.connect("rp.powercurve.rated_pitch", ["rs.pitch_load", "rs.tot_loads_gust.aeroloads_pitch"])

        # Connections to the stall check
        if modeling_options["flags"]["blade"]:
            self.connect("rp.powercurve.aoa_regII", "stall_check.aoa_along_span")



class RotorSE(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        
        self.add_subsystem("prop", RotorSEProp(modeling_options=modeling_options), promotes=["*"])
        self.add_subsystem("perf", RotorSEPerf(modeling_options=modeling_options), promotes=["*"])

        # Connections to RotorPower
        self.connect("wt_class.V_mean", "rp.cdf.xbar")
        self.connect("wt_class.V_mean", "rp.gust.V_mean")

        # Connections to the stall check
        if not modeling_options["flags"]["blade"]:
            self.connect("ccblade.alpha", "stall_check.aoa_along_span")
            
