import openmdao.api as om
from wisdem.rotorse.rotor_power import RotorPower, NoStallConstraint
from wisdem.commonse.turbine_class import TurbineClass
from wisdem.rotorse.rotor_structure import RotorStructure
from wisdem.rotorse.rotor_elasticity import RotorElasticity
from wisdem.rotorse.rotor_cost import RotorCost
from wisdem.ccblade.ccblade_component import CCBladeTwist


class RotorSE(om.Group):
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

        ivc = om.IndepVarComp()
        ivc.add_discrete_output("hubloss", val=modeling_options["WISDEM"]["RotorSE"]["hubloss"])
        ivc.add_discrete_output("tiploss", val=modeling_options["WISDEM"]["RotorSE"]["tiploss"])
        ivc.add_discrete_output("wakerotation", val=modeling_options["WISDEM"]["RotorSE"]["wakerotation"])
        ivc.add_discrete_output("usecd", val=modeling_options["WISDEM"]["RotorSE"]["usecd"])
        ivc.add_discrete_output("nSector", val=modeling_options["WISDEM"]["RotorSE"]["n_sector"])
        self.add_subsystem("ivc", ivc, promotes=["*"])

        self.add_subsystem(
            "ccblade",
            CCBladeTwist(modeling_options=modeling_options, opt_options=opt_options),
            promotes=promoteCC + ["pitch", "tsr", "precurveTip", "presweepTip"],
        )  # Run standalone CCBlade and possibly determine optimal twist from user-defined margin to stall

        self.add_subsystem("wt_class", TurbineClass())

        self.add_subsystem(
            "re",
            RotorElasticity(modeling_options=modeling_options, opt_options=opt_options),
            promotes=promoteGeom + ["chord", "theta", "r", "precurve", "presweep"],
        )

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
            promotes=promoteGeom + promoteCC + ["s", "precurveTip", "presweepTip"],
        )

        self.add_subsystem(
            "rc",
            RotorCost(mod_options=modeling_options, opt_options=opt_options))

        # Connection from ra to rs for the rated conditions
        self.connect("rp.gust.V_gust", ["rs.aero_gust.V_load", "rs.aero_hub_loads.V_load"])
        self.connect(
            "rp.powercurve.rated_Omega", ["rs.Omega_load", "rs.tot_loads_gust.aeroloads_Omega", "rs.constr.rated_Omega"]
        )
        self.connect("rp.powercurve.rated_pitch", ["rs.pitch_load", "rs.tot_loads_gust.aeroloads_pitch"])
        self.connect("re.precomp.blade_mass", "rs.bjs.blade_mass_re")
        self.connect("rs.bjs.joint_material_cost", "rc.joint_material_cost")
        # TODO pass joint cost to rotor cost

        # Connections to RotorPower
        self.connect("wt_class.V_mean", "rp.cdf.xbar")
        self.connect("wt_class.V_mean", "rp.gust.V_mean")

        # Connections to the stall check
        if modeling_options["flags"]["blade"]:
            self.connect("rp.powercurve.aoa_regII", "stall_check.aoa_along_span")
        else:
            self.connect("ccblade.alpha", "stall_check.aoa_along_span")
