"""
Script that computes the regulation trajectory of the rotor and the annual energy production

Nikhar J. Abbas, Pietro Bortolotti
January 2020
"""

import numpy as np
from openmdao.api import Group, ExplicitComponent
from scipy.optimize import brentq, minimize, minimize_scalar
from wisdem.ccblade.ccblade import CCBlade, CCAirfoil
from scipy.interpolate import PchipInterpolator
from wisdem.commonse.utilities import smooth_abs, smooth_min, linspace_with_deriv
from wisdem.commonse.distribution import RayleighCDF, WeibullWithMeanCDF


class RotorPower(Group):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem(
            "powercurve",
            RegulatedPowerCurve(modeling_options=modeling_options),
            promotes=[
                "v_min",
                "v_max",
                "rated_power",
                "omega_min",
                "omega_max",
                "control_maxTS",
                "tsr_operational",
                "control_pitch",
                "drivetrainType",
                "r",
                "chord",
                "theta",
                "Rhub",
                "Rtip",
                "hub_height",
                "precone",
                "tilt",
                "yaw",
                "precurve",
                "precurveTip",
                "presweep",
                "presweepTip",
                "airfoils_aoa",
                "airfoils_Re",
                "airfoils_cl",
                "airfoils_cd",
                "airfoils_cm",
                "nBlades",
                "rho",
                "mu",
            ],
        )
        self.add_subsystem("gust", GustETM())
        self.add_subsystem("cdf", WeibullWithMeanCDF(nspline=modeling_options["RotorSE"]["n_pc_spline"]))
        self.add_subsystem("aep", AEP(nspline=modeling_options["RotorSE"]["n_pc_spline"]), promotes=["AEP"])

        # Connections to the Weibull CDF
        self.connect("powercurve.V_spline", "cdf.x")

        # Connections to the aep computation component
        self.connect("cdf.F", "aep.CDF_V")
        self.connect("powercurve.P_spline", "aep.P")


class GustETM(ExplicitComponent):
    # OpenMDAO component that generates an "equivalent gust" wind speed by summing an user-defined wind speed at hub height with 3 times sigma. sigma is the turbulent wind speed standard deviation for the extreme turbulence model, see IEC-61400-1 Eq. 19 paragraph 6.3.2.3

    def setup(self):
        # Inputs
        self.add_input("V_mean", val=0.0, units="m/s", desc="IEC average wind speed for turbine class")
        self.add_input("V_hub", val=0.0, units="m/s", desc="hub height wind speed")
        self.add_discrete_input("turbulence_class", val="A", desc="IEC turbulence class")
        self.add_input("std", val=3.0, desc="number of standard deviations for strength of gust")

        # Output
        self.add_output("V_gust", val=0.0, units="m/s", desc="gust wind speed")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        V_mean = inputs["V_mean"]
        V_hub = inputs["V_hub"]
        std = inputs["std"]
        turbulence_class = discrete_inputs["turbulence_class"]

        if turbulence_class.upper() == "A":
            Iref = 0.16
        elif turbulence_class.upper() == "B":
            Iref = 0.14
        elif turbulence_class.upper() == "C":
            Iref = 0.12
        else:
            raise ValueError("Unknown Turbulence Class: " + str(turbulence_class) + " . Permitted values are A / B / C")

        c = 2.0
        sigma = c * Iref * (0.072 * (V_mean / c + 3) * (V_hub / c - 4) + 10)
        V_gust = V_hub + std * sigma
        outputs["V_gust"] = V_gust


class RegulatedPowerCurve(Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem("compute_power_curve", ComputePowerCurve(modeling_options=modeling_options), promotes=["*"])

        self.add_subsystem("compute_splines", ComputeSplines(modeling_options=modeling_options), promotes=["*"])


class ComputePowerCurve(ExplicitComponent):
    """
    Iteratively call CCBlade to compute the power curve.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        self.n_span = n_span = modeling_options["RotorSE"]["n_span"]
        self.n_aoa = n_aoa = modeling_options["RotorSE"]["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = modeling_options["RotorSE"]["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.n_tab = n_tab = modeling_options["RotorSE"][
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1
        self.regulation_reg_III = modeling_options["RotorSE"]["regulation_reg_III"]
        self.n_pc = modeling_options["RotorSE"]["n_pc"]
        self.n_pc_spline = modeling_options["RotorSE"]["n_pc_spline"]

        # parameters
        self.add_input("v_min", val=0.0, units="m/s", desc="cut-in wind speed")
        self.add_input("v_max", val=0.0, units="m/s", desc="cut-out wind speed")
        self.add_input("rated_power", val=0.0, units="W", desc="electrical rated power")
        self.add_input("omega_min", val=0.0, units="rpm", desc="minimum allowed rotor rotation speed")
        self.add_input("omega_max", val=0.0, units="rpm", desc="maximum allowed rotor rotation speed")
        self.add_input("control_maxTS", val=0.0, units="m/s", desc="maximum allowed blade tip speed")
        self.add_input("tsr_operational", val=0.0, desc="tip-speed ratio in Region 2 (should be optimized externally)")
        self.add_input(
            "control_pitch",
            val=0.0,
            units="deg",
            desc="pitch angle in region 2 (and region 3 for fixed pitch machines)",
        )
        self.add_discrete_input("drivetrainType", val="GEARED")
        self.add_input("gearbox_efficiency", val=1.0)
        self.add_input(
            "generator_efficiency",
            val=np.ones(self.n_pc),
            desc="Generator efficiency at various rpm values to support table lookup",
        )
        self.add_input(
            "lss_rpm",
            val=np.zeros(self.n_pc),
            units="rpm",
            desc="Low speed shaft RPM values at which the generator efficiency values are given",
        )

        self.add_input(
            "r",
            val=np.zeros(n_span),
            units="m",
            desc="radial locations where blade is defined (should be increasing and not go all the way to hub or tip)",
        )
        self.add_input("chord", val=np.zeros(n_span), units="m", desc="chord length at each section")
        self.add_input(
            "theta",
            val=np.zeros(n_span),
            units="deg",
            desc="twist angle at each section (positive decreases angle of attack)",
        )
        self.add_input("Rhub", val=0.0, units="m", desc="hub radius")
        self.add_input("Rtip", val=0.0, units="m", desc="tip radius")
        self.add_input("hub_height", val=0.0, units="m", desc="hub height")
        self.add_input(
            "precone",
            val=0.0,
            units="deg",
            desc="precone angle",
        )
        self.add_input(
            "tilt",
            val=0.0,
            units="deg",
            desc="shaft tilt",
        )
        self.add_input(
            "yaw",
            val=0.0,
            units="deg",
            desc="yaw error",
        )
        self.add_input("precurve", val=np.zeros(n_span), units="m", desc="precurve at each section")
        self.add_input("precurveTip", val=0.0, units="m", desc="precurve at tip")
        self.add_input("presweep", val=np.zeros(n_span), units="m", desc="presweep at each section")
        self.add_input("presweepTip", val=0.0, units="m", desc="presweep at tip")

        # self.add_discrete_input('airfoils',  val=[0]*n_span,                      desc='CCAirfoil instances')
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="lift coefficients, spanwise")
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="drag coefficients, spanwise")
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="moment coefficients, spanwise")
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg", desc="angle of attack grid for polars")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)), desc="Reynolds numbers of polars")
        self.add_discrete_input("nBlades", val=0, desc="number of blades")
        self.add_input("rho", val=1.225, units="kg/m**3", desc="density of air")
        self.add_input("mu", val=1.81e-5, units="kg/(m*s)", desc="dynamic viscosity of air")
        self.add_input("shearExp", val=0.0, desc="shear exponent")
        self.add_discrete_input(
            "nSector", val=4, desc="number of sectors to divide rotor face into in computing thrust and power"
        )
        self.add_discrete_input("tiploss", val=True, desc="include Prandtl tip loss model")
        self.add_discrete_input("hubloss", val=True, desc="include Prandtl hub loss model")
        self.add_discrete_input(
            "wakerotation",
            val=True,
            desc="include effect of wake rotation (i.e., tangential induction factor is nonzero)",
        )
        self.add_discrete_input("usecd", val=True, desc="use drag coefficient in computing induction factors")

        # outputs
        self.add_output("V", val=np.zeros(self.n_pc), units="m/s", desc="wind vector")
        self.add_output("Omega", val=np.zeros(self.n_pc), units="rpm", desc="rotor rotational speed")
        self.add_output("pitch", val=np.zeros(self.n_pc), units="deg", desc="rotor pitch schedule")
        self.add_output("P", val=np.zeros(self.n_pc), units="W", desc="rotor electrical power")
        self.add_output("P_aero", val=np.zeros(self.n_pc), units="W", desc="rotor mechanical power")
        self.add_output("T", val=np.zeros(self.n_pc), units="N", desc="rotor aerodynamic thrust")
        self.add_output("Q", val=np.zeros(self.n_pc), units="N*m", desc="rotor aerodynamic torque")
        self.add_output("M", val=np.zeros(self.n_pc), units="N*m", desc="blade root moment")
        self.add_output("Cp", val=np.zeros(self.n_pc), desc="rotor electrical power coefficient")
        self.add_output("Cp_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic power coefficient")
        self.add_output("Ct_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic thrust coefficient")
        self.add_output("Cq_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic torque coefficient")
        self.add_output("Cm_aero", val=np.zeros(self.n_pc), desc="rotor aerodynamic moment coefficient")

        self.add_output("V_R25", val=0.0, units="m/s", desc="region 2.5 transition wind speed")
        self.add_output("rated_V", val=0.0, units="m/s", desc="rated wind speed")
        self.add_output("rated_Omega", val=0.0, units="rpm", desc="rotor rotation speed at rated")
        self.add_output("rated_pitch", val=0.0, units="deg", desc="pitch setting at rated")
        self.add_output("rated_T", val=0.0, units="N", desc="rotor aerodynamic thrust at rated")
        self.add_output("rated_Q", val=0.0, units="N*m", desc="rotor aerodynamic torque at rated")
        self.add_output("rated_mech", val=0.0, units="W", desc="Mechanical shaft power at rated")
        self.add_output(
            "ax_induct_regII", val=np.zeros(n_span), desc="rotor axial induction at cut-in wind speed along blade span"
        )
        self.add_output(
            "tang_induct_regII",
            val=np.zeros(n_span),
            desc="rotor tangential induction at cut-in wind speed along blade span",
        )
        self.add_output(
            "aoa_regII",
            val=np.zeros(n_span),
            units="deg",
            desc="angle of attack distribution along blade span at cut-in wind speed",
        )
        self.add_output("Cp_regII", val=0.0, desc="power coefficient at cut-in wind speed")
        self.add_output(
            "cl_regII", val=np.zeros(n_span), desc="lift coefficient distribution along blade span at cut-in wind speed"
        )
        self.add_output(
            "cd_regII", val=np.zeros(n_span), desc="drag coefficient distribution along blade span at cut-in wind speed"
        )
        self.add_output("rated_efficiency", val=1.0, desc="Efficiency at rated conditions")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Create Airfoil class instances
        af = [None] * self.n_span
        for i in range(self.n_span):
            if self.n_tab > 1:
                ref_tab = int(np.floor(self.n_tab / 2))
                af[i] = CCAirfoil(
                    inputs["airfoils_aoa"],
                    inputs["airfoils_Re"],
                    inputs["airfoils_cl"][i, :, :, ref_tab],
                    inputs["airfoils_cd"][i, :, :, ref_tab],
                    inputs["airfoils_cm"][i, :, :, ref_tab],
                )
            else:
                af[i] = CCAirfoil(
                    inputs["airfoils_aoa"],
                    inputs["airfoils_Re"],
                    inputs["airfoils_cl"][i, :, :, 0],
                    inputs["airfoils_cd"][i, :, :, 0],
                    inputs["airfoils_cm"][i, :, :, 0],
                )

        self.ccblade = CCBlade(
            inputs["r"],
            inputs["chord"],
            inputs["theta"],
            af,
            inputs["Rhub"],
            inputs["Rtip"],
            discrete_inputs["nBlades"],
            inputs["rho"],
            inputs["mu"],
            inputs["precone"],
            inputs["tilt"],
            inputs["yaw"],
            inputs["shearExp"],
            inputs["hub_height"],
            discrete_inputs["nSector"],
            inputs["precurve"],
            inputs["precurveTip"],
            inputs["presweep"],
            inputs["presweepTip"],
            discrete_inputs["tiploss"],
            discrete_inputs["hubloss"],
            discrete_inputs["wakerotation"],
            discrete_inputs["usecd"],
        )

        # JPJ: what is this grid for? Seems to be a special distribution of velocities
        # for the hub
        grid0 = np.cumsum(np.abs(np.diff(np.cos(np.linspace(-np.pi / 4.0, np.pi / 2.0, self.n_pc + 1)))))
        grid1 = (grid0 - grid0[0]) / (grid0[-1] - grid0[0])
        Uhub = grid1 * (inputs["v_max"] - inputs["v_min"]) + inputs["v_min"]

        P_aero = np.zeros(Uhub.shape)
        Cp_aero = np.zeros(Uhub.shape)
        Ct_aero = np.zeros(Uhub.shape)
        Cq_aero = np.zeros(Uhub.shape)
        Cm_aero = np.zeros(Uhub.shape)
        P = np.zeros(Uhub.shape)
        Cp = np.zeros(Uhub.shape)
        T = np.zeros(Uhub.shape)
        Q = np.zeros(Uhub.shape)
        M = np.zeros(Uhub.shape)
        pitch = np.zeros(Uhub.shape) + inputs["control_pitch"]

        # Unpack variables
        P_rated = inputs["rated_power"]
        R_tip = inputs["Rtip"]
        tsr = inputs["tsr_operational"]
        driveType = discrete_inputs["drivetrainType"]

        # Set rotor speed based on TSR
        Omega_tsr = Uhub * tsr / R_tip

        # Determine maximum rotor speed (rad/s)- either by TS or by control input
        Omega_max = min([inputs["control_maxTS"] / R_tip, inputs["omega_max"] * np.pi / 30.0])

        # Apply maximum and minimum rotor speed limits
        Omega = np.maximum(np.minimum(Omega_tsr, Omega_max), inputs["omega_min"] * np.pi / 30.0)
        Omega_rpm = Omega * 30.0 / np.pi

        # Create table lookup of total drivetrain efficiency, where rpm is first column and second column is gearbox*generator
        lss_rpm = inputs["lss_rpm"]
        gen_eff = inputs["generator_efficiency"]
        if not np.any(lss_rpm):
            lss_rpm = np.linspace(np.maximum(0.1, Omega_rpm[0]), Omega_rpm[-1], self.n_pc)
            _, gen_eff = compute_P_and_eff(
                P_rated * lss_rpm / lss_rpm[-1], P_rated, np.zeros(self.n_pc), driveType, np.zeros((self.n_pc, 2))
            )

        # driveEta  = np.c_[lss_rpm, float(inputs['gearbox_efficiency'])*gen_eff]
        driveEta = float(inputs["gearbox_efficiency"]) * gen_eff

        # Set baseline power production
        myout, derivs = self.ccblade.evaluate(Uhub, Omega_rpm, pitch, coefficients=True)
        P_aero, T, Q, M, Cp_aero, Ct_aero, Cq_aero, Cm_aero = [
            myout[key] for key in ["P", "T", "Q", "M", "CP", "CT", "CQ", "CM"]
        ]
        # P, eff  = compute_P_and_eff(P_aero, P_rated, Omega_rpm, driveType, driveEta)
        eff = np.interp(Omega_rpm, lss_rpm, driveEta)
        P = P_aero * eff
        Cp = Cp_aero * eff

        # Find Region 3 index
        region_bool = np.nonzero(P >= P_rated)[0]
        if len(region_bool) == 0:
            i_3 = self.n_pc
            region3 = False
        else:
            i_3 = region_bool[0] + 1
            region3 = True

        # Guess at Region 2.5, but we will do a more rigorous search below
        if Omega_max < Omega_tsr[-1]:
            U_2p5 = np.interp(Omega[-1], Omega_tsr, Uhub)
            outputs["V_R25"] = U_2p5
        else:
            U_2p5 = Uhub[-1]
        i_2p5 = np.nonzero(U_2p5 <= Uhub)[0][0]

        # Find rated index and guess at rated speed
        if P_aero[-1] > P_rated:
            U_rated = np.interp(P_rated, P_aero * eff, Uhub)
        else:
            U_rated = Uhub[-1]
        i_rated = np.nonzero(U_rated <= Uhub)[0][0]

        # Function to be used inside of power maximization until Region 3
        def maximizePower(pitch, Uhub, Omega_rpm):
            myout, _ = self.ccblade.evaluate([Uhub], [Omega_rpm], [pitch], coefficients=False)
            return -myout["P"]

        # Maximize power until Region 3
        region2p5 = False
        for i in range(i_3):
            # No need to optimize if already doing well
            if Omega[i] == Omega_tsr[i]:
                continue

            # Find pitch value that gives highest power rating
            pitch0 = pitch[i] if i == 0 else pitch[i - 1]
            bnds = [pitch0 - 10.0, pitch0 + 10.0]
            pitch[i] = minimize_scalar(
                lambda x: maximizePower(x, Uhub[i], Omega_rpm[i]),
                bounds=bnds,
                method="bounded",
                options={"disp": False, "xatol": 1e-2, "maxiter": 40},
            )["x"]

            # Find associated power
            myout, _ = self.ccblade.evaluate([Uhub[i]], [Omega_rpm[i]], [pitch[i]], coefficients=True)
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [
                myout[key] for key in ["P", "T", "Q", "M", "CP", "CT", "CQ", "CM"]
            ]
            # P[i], eff[i] = compute_P_and_eff(P_aero[i], P_rated, Omega_rpm[i], driveType, driveEta)
            eff[i] = np.interp(Omega_rpm[i], lss_rpm, driveEta)
            P[i] = P_aero[i] * eff[i]
            Cp[i] = Cp_aero[i] * eff[i]

            # Note if we find Region 2.5
            if (not region2p5) and (Omega[i] == Omega_max) and (P[i] < P_rated):
                region2p5 = True
                i_2p5 = i

            # Stop if we find Region 3 early
            if P[i] > P_rated:
                i_3 = i + 1
                i_rated = i
                break

        # Solve for rated velocity
        # JPJ: why rename i_rated to i here? It removes clarity in the following 50 lines that we're looking at the rated properties
        i = i_rated
        if i < self.n_pc - 1:

            def const_Urated(x):
                pitch_i = x[0]
                Uhub_i = x[1]
                Omega_i = min([Uhub_i * tsr / R_tip, Omega_max])
                Omega_i_rpm = Omega_i * 30.0 / np.pi
                myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_i_rpm], [pitch_i], coefficients=False)
                P_aero_i = float(myout["P"])
                # P_i,_  = compute_P_and_eff(P_aero_i.flatten(), P_rated, Omega_i_rpm, driveType, driveEta)
                eff_i = np.interp(Omega_i_rpm, lss_rpm, driveEta)
                P_i = float(P_aero_i * eff_i)
                return P_i - P_rated

            if region2p5:
                # Have to search over both pitch and speed
                x0 = [0.0, Uhub[i]]
                bnds = [np.sort([pitch[i - 1], pitch[i + 1]]), [Uhub[i - 1], Uhub[i + 1]]]
                const = {}
                const["type"] = "eq"
                const["fun"] = const_Urated
                params_rated = minimize(lambda x: x[1], x0, method="slsqp", bounds=bnds, constraints=const, tol=1e-3)

                if params_rated.success and not np.isnan(params_rated.x[1]):
                    U_rated = params_rated.x[1]
                    pitch[i] = params_rated.x[0]
                else:
                    U_rated = U_rated  # Use guessed value earlier
                    pitch[i] = 0.0
            else:
                # Just search over speed
                pitch[i] = 0.0
                try:
                    U_rated = brentq(
                        lambda x: const_Urated([0.0, x]),
                        Uhub[i - 1],
                        Uhub[i + 1],
                        xtol=1e-4,
                        rtol=1e-5,
                        maxiter=40,
                        disp=False,
                    )
                except ValueError:
                    U_rated = minimize_scalar(
                        lambda x: np.abs(const_Urated([0.0, x])),
                        bounds=[Uhub[i - 1], Uhub[i + 1]],
                        method="bounded",
                        options={"disp": False, "xatol": 1e-3, "maxiter": 40},
                    )["x"]

            Omega_rated = min([U_rated * tsr / R_tip, Omega_max])
            Omega[i:] = np.minimum(Omega[i:], Omega_rated)  # Stay at this speed if hit rated too early
            Omega_rpm = Omega * 30.0 / np.pi
            myout, _ = self.ccblade.evaluate([U_rated], [Omega_rpm[i]], [pitch[i]], coefficients=True)
            P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [
                myout[key] for key in ["P", "T", "Q", "M", "CP", "CT", "CQ", "CM"]
            ]
            # P[i], eff[i] = compute_P_and_eff(P_aero[i], P_rated, Omega_rpm[i], driveType, driveEta)
            eff[i] = np.interp(Omega_rpm[i], lss_rpm, driveEta)
            P[i] = P_aero[i] * eff[i]
            Cp[i] = Cp_aero[i] * eff[i]
            P[i] = P_rated

        # Store rated speed in array
        Uhub[i_rated] = U_rated

        # Store outputs
        outputs["rated_V"] = np.float64(U_rated)
        outputs["rated_Omega"] = Omega_rpm[i]
        outputs["rated_pitch"] = pitch[i]
        outputs["rated_T"] = T[i]
        outputs["rated_Q"] = Q[i]
        outputs["rated_mech"] = P_aero[i]
        outputs["rated_efficiency"] = eff[i]

        # JPJ: this part can be converted into a BalanceComp with a solver.
        # This will be less expensive and allow us to get derivatives through the process.
        if region3:
            # Function to be used to stay at rated power in Region 3
            def rated_power_dist(pitch_i, Uhub_i, Omega_rpm_i):
                myout, _ = self.ccblade.evaluate([Uhub_i], [Omega_rpm_i], [pitch_i], coefficients=False)
                P_aero_i = myout["P"]
                # P_i, _   = compute_P_and_eff(P_aero_i, P_rated, Omega_rpm_i, driveType, driveEta)
                eff_i = np.interp(Omega_rpm_i, lss_rpm, driveEta)
                P_i = P_aero_i * eff_i
                return P_i - P_rated

            # Solve for Region 3 pitch
            options = {"disp": False}
            if self.regulation_reg_III:
                for i in range(i_3, self.n_pc):
                    pitch0 = pitch[i - 1]
                    try:
                        pitch[i] = brentq(
                            lambda x: rated_power_dist(x, Uhub[i], Omega_rpm[i]),
                            pitch0,
                            pitch0 + 10.0,
                            xtol=1e-4,
                            rtol=1e-5,
                            maxiter=40,
                            disp=False,
                        )
                    except ValueError:
                        pitch[i] = minimize_scalar(
                            lambda x: np.abs(rated_power_dist(x, Uhub[i], Omega_rpm[i])),
                            bounds=[pitch0 - 5.0, pitch0 + 15.0],
                            method="bounded",
                            options={"disp": False, "xatol": 1e-3, "maxiter": 40},
                        )["x"]

                    myout, _ = self.ccblade.evaluate([Uhub[i]], [Omega_rpm[i]], [pitch[i]], coefficients=True)
                    P_aero[i], T[i], Q[i], M[i], Cp_aero[i], Ct_aero[i], Cq_aero[i], Cm_aero[i] = [
                        myout[key] for key in ["P", "T", "Q", "M", "CP", "CT", "CQ", "CM"]
                    ]
                    # P[i], eff[i] = compute_P_and_eff(P_aero[i], P_rated, Omega_rpm[i], driveType, driveEta)
                    eff[i] = np.interp(Omega_rpm[i], lss_rpm, driveEta)
                    P[i] = P_aero[i] * eff[i]
                    Cp[i] = Cp_aero[i] * eff[i]
                    # P[i]        = P_rated

            else:
                P[i_3:] = P_rated
                T[i_3:] = 0
                Q[i_3:] = P[i_3:] / Omega[i_3:]
                M[i_3:] = 0
                pitch[i_3:] = 0
                Cp[i_3:] = P[i_3:] / (0.5 * inputs["rho"] * np.pi * R_tip ** 2 * Uhub[i_3:] ** 3)
                Ct_aero[i_3:] = 0
                Cq_aero[i_3:] = 0
                Cm_aero[i_3:] = 0

        outputs["T"] = T
        outputs["Q"] = Q
        outputs["Omega"] = Omega_rpm

        outputs["P"] = P
        outputs["Cp"] = Cp
        outputs["P_aero"] = P_aero
        outputs["Cp_aero"] = Cp_aero
        outputs["Ct_aero"] = Ct_aero
        outputs["Cq_aero"] = Cq_aero
        outputs["Cm_aero"] = Cm_aero
        outputs["V"] = Uhub
        outputs["M"] = M
        outputs["pitch"] = pitch

        self.ccblade.induction_inflow = True
        tsr_vec = Omega_rpm / 30.0 * np.pi * R_tip / Uhub
        id_regII = np.argmin(abs(tsr_vec - inputs["tsr_operational"]))
        loads, derivs = self.ccblade.distributedAeroLoads(Uhub[id_regII], Omega_rpm[id_regII], pitch[id_regII], 0.0)

        # outputs
        outputs["ax_induct_regII"] = loads["a"]
        outputs["tang_induct_regII"] = loads["ap"]
        outputs["aoa_regII"] = loads["alpha"]
        outputs["cl_regII"] = loads["Cl"]
        outputs["cd_regII"] = loads["Cd"]
        outputs["Cp_regII"] = Cp_aero[id_regII]


class ComputeSplines(ExplicitComponent):
    """
    Compute splined quantities for V, P, and Omega.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        self.n_pc = modeling_options["RotorSE"]["n_pc"]
        self.n_pc_spline = modeling_options["RotorSE"]["n_pc_spline"]

        self.add_input("v_min", val=0.0, units="m/s", desc="cut-in wind speed")
        self.add_input("v_max", val=0.0, units="m/s", desc="cut-out wind speed")
        self.add_input("V", val=np.zeros(self.n_pc), units="m/s", desc="wind vector")
        self.add_input("Omega", val=np.zeros(self.n_pc), units="rpm", desc="rotor rotational speed")
        self.add_input("P", val=np.zeros(self.n_pc), units="W", desc="rotor electrical power")

        self.add_output("V_spline", val=np.zeros(self.n_pc_spline), units="m/s", desc="wind vector")
        self.add_output("P_spline", val=np.zeros(self.n_pc_spline), units="W", desc="rotor electrical power")
        self.add_output("Omega_spline", val=np.zeros(self.n_pc_spline), units="rpm", desc="omega")

        self.declare_partials(of="V_spline", wrt="v_min")
        self.declare_partials(of="V_spline", wrt="v_max")

        self.declare_partials(of="P_spline", wrt="v_min", method="fd")
        self.declare_partials(of="P_spline", wrt="v_max", method="fd")
        self.declare_partials(of="P_spline", wrt="V", method="fd")
        self.declare_partials(of="P_spline", wrt="P", method="fd")

        self.declare_partials(of="Omega_spline", wrt="v_min", method="fd")
        self.declare_partials(of="Omega_spline", wrt="v_max", method="fd")
        self.declare_partials(of="Omega_spline", wrt="V", method="fd")
        self.declare_partials(of="Omega_spline", wrt="Omega", method="fd")

    def compute(self, inputs, outputs):
        # Fit spline to powercurve for higher grid density
        V_spline = np.linspace(inputs["v_min"], inputs["v_max"], self.n_pc_spline)
        spline = PchipInterpolator(inputs["V"], inputs["P"])
        P_spline = spline(V_spline)
        spline = PchipInterpolator(inputs["V"], inputs["Omega"])
        Omega_spline = spline(V_spline)

        # outputs
        outputs["V_spline"] = V_spline.flatten()
        outputs["P_spline"] = P_spline.flatten()
        outputs["Omega_spline"] = Omega_spline.flatten()

    def compute_partials(self, inputs, partials):
        linspace_with_deriv
        V_spline, dy_dstart, dy_dstop = linspace_with_deriv(inputs["v_min"], inputs["v_max"], self.n_pc_spline)
        partials["V_spline", "v_min"] = dy_dstart
        partials["V_spline", "v_max"] = dy_dstop


# Class to define a constraint so that the blade cannot operate in stall conditions
class NoStallConstraint(ExplicitComponent):
    def initialize(self):

        self.options.declare("modeling_options")

    def setup(self):

        modeling_options = self.options["modeling_options"]
        self.n_span = n_span = modeling_options["RotorSE"]["n_span"]
        self.n_aoa = n_aoa = modeling_options["RotorSE"]["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = modeling_options["RotorSE"]["n_Re"]  # Number of Reynolds, so far hard set at 1
        self.n_tab = n_tab = modeling_options["RotorSE"][
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1

        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input("aoa_along_span", val=np.zeros(n_span), units="deg", desc="Angle of attack along blade span")
        self.add_input("stall_margin", val=3.0, units="deg", desc="Minimum margin from the stall angle")
        self.add_input(
            "min_s",
            val=0.25,
            desc="Minimum nondimensional coordinate along blade span where to define the constraint (blade root typically stalls)",
        )
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="lift coefficients, spanwise")
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="drag coefficients, spanwise")
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="moment coefficients, spanwise")
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg", desc="angle of attack grid for polars")

        self.add_output(
            "no_stall_constraint",
            val=np.zeros(n_span),
            desc="Constraint, ratio between angle of attack plus a margin and stall angle",
        )
        self.add_output(
            "stall_angle_along_span", val=np.zeros(n_span), units="deg", desc="Stall angle along blade span"
        )

    def compute(self, inputs, outputs):

        verbosity = True

        i_min = np.argmin(abs(inputs["min_s"] - inputs["s"]))

        for i in range(self.n_span):
            unsteady = eval_unsteady(
                inputs["airfoils_aoa"],
                inputs["airfoils_cl"][i, :, 0, 0],
                inputs["airfoils_cd"][i, :, 0, 0],
                inputs["airfoils_cm"][i, :, 0, 0],
            )
            outputs["stall_angle_along_span"][i] = unsteady["alpha1"]
            if outputs["stall_angle_along_span"][i] == 0:
                outputs["stall_angle_along_span"][i] = 1e-6  # To avoid nan

        for i in range(i_min, self.n_span):
            outputs["no_stall_constraint"][i] = (inputs["aoa_along_span"][i] + inputs["stall_margin"]) / outputs[
                "stall_angle_along_span"
            ][i]

            # if verbosity == True:
            #     if outputs['no_stall_constraint'][i] > 1:
            #         print('Blade is violating the minimum margin to stall at span location %.2f %%' % (inputs['s'][i]*100.))


class AEP(ExplicitComponent):
    def initialize(self):

        self.options.declare("nspline")

    def setup(self):
        n_pc_spline = self.options["nspline"]
        """integrate to find annual energy production"""

        # inputs
        self.add_input(
            "CDF_V",
            val=np.zeros(n_pc_spline),
            units="m/s",
            desc="cumulative distribution function evaluated at each wind speed",
        )
        self.add_input("P", val=np.zeros(n_pc_spline), units="W", desc="power curve (power)")
        self.add_input(
            "lossFactor", val=1.0, desc="multiplicative factor for availability and other losses (soiling, array, etc.)"
        )

        # outputs
        self.add_output("AEP", val=0.0, units="kW*h", desc="annual energy production")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):

        lossFactor = inputs["lossFactor"]
        P = inputs["P"]
        CDF_V = inputs["CDF_V"]

        factor = lossFactor / 1e3 * 365.0 * 24.0
        outputs["AEP"] = factor * np.trapz(P, CDF_V)  # in kWh
        """
        dAEP_dP, dAEP_dCDF = trapz_deriv(P, CDF_V)
        dAEP_dP *= factor
        dAEP_dCDF *= factor

        dAEP_dlossFactor = np.array([outputs['AEP']/lossFactor])
        self.J = {}
        self.J['AEP', 'CDF_V'] = np.reshape(dAEP_dCDF, (1, len(dAEP_dCDF)))
        self.J['AEP', 'P'] = np.reshape(dAEP_dP, (1, len(dAEP_dP)))
        self.J['AEP', 'lossFactor'] = dAEP_dlossFactor

    def compute_partials(self, inputs, J):
        J = self.J
        """


def compute_P_and_eff(aeroPower, ratedPower, Omega_rpm, drivetrainType, drivetrainEff):

    if not np.any(drivetrainEff):
        drivetrainType = drivetrainType.upper()
        if drivetrainType == "GEARED":
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif drivetrainType == "SINGLE_STAGE":
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif drivetrainType == "MULTI_DRIVE":
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif drivetrainType in ["PM_DIRECT_DRIVE", "DIRECT_DRIVE", "DIRECT DRIVE"]:
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899

        elif drivetrainType == "CONSTANT_EFF":
            constant = 0.00
            linear = 0.07
            quadratic = 0.0
        else:
            raise ValueError("The drivetrain model is not supported! Please check rotor_power.py")

        Pbar0 = aeroPower / ratedPower

        # handle negative power case (with absolute value)
        Pbar1, dPbar1_dPbar0 = smooth_abs(Pbar0, dx=0.01)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar, dPbar_dPbar1, _ = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant / Pbar + linear + quadratic * Pbar)
        eff = np.maximum(eff, 1e-3)
    else:
        # Use table lookup from rpm to calculate total efficiency
        eff = np.interp(Omega_rpm, drivetrainEff[:, 0], drivetrainEff[:, 1])

    return aeroPower * eff, eff


def eval_unsteady(alpha, cl, cd, cm):
    # calculate unsteady coefficients from polars for OpenFAST's Aerodyn

    unsteady = {}

    alpha_rad = np.radians(alpha)
    cn = cl * np.cos(alpha_rad) + cd * np.sin(alpha_rad)

    # alpha0, Cd0, Cm0
    aoa_l = [-30.0]
    aoa_h = [30.0]
    idx_low = np.argmin(abs(alpha - aoa_l))
    idx_high = np.argmin(abs(alpha - aoa_h))

    if max(np.abs(np.gradient(cl))) > 0.0:
        unsteady["alpha0"] = np.interp(0.0, cl[idx_low:idx_high], alpha[idx_low:idx_high])
        unsteady["Cd0"] = np.interp(0.0, cl[idx_low:idx_high], cd[idx_low:idx_high])
        unsteady["Cm0"] = np.interp(0.0, cl[idx_low:idx_high], cm[idx_low:idx_high])
    else:
        unsteady["alpha0"] = 0.0
        unsteady["Cd0"] = cd[np.argmin(abs(alpha - 0.0))]
        unsteady["Cm0"] = 0.0

    unsteady["eta_e"] = 1
    unsteady["T_f0"] = "Default"
    unsteady["T_V0"] = "Default"
    unsteady["T_p"] = "Default"
    unsteady["T_VL"] = "Default"
    unsteady["b1"] = "Default"
    unsteady["b2"] = "Default"
    unsteady["b5"] = "Default"
    unsteady["A1"] = "Default"
    unsteady["A2"] = "Default"
    unsteady["A5"] = "Default"
    unsteady["S1"] = 0
    unsteady["S2"] = 0
    unsteady["S3"] = 0
    unsteady["S4"] = 0

    def find_breakpoint(x, y, idx_low, idx_high, multi=1.0):
        lin_fit = np.interp(x[idx_low:idx_high], [x[idx_low], x[idx_high]], [y[idx_low], y[idx_high]])
        idx_break = 0
        lin_diff = 0
        for i, (fit, yi) in enumerate(zip(lin_fit, y[idx_low:idx_high])):
            if multi == 0:
                diff_i = np.abs(yi - fit)
            else:
                diff_i = multi * (yi - fit)
            if diff_i > lin_diff:
                lin_diff = diff_i
                idx_break = i
        idx_break += idx_low
        return idx_break

    # Cn1
    idx_alpha0 = np.argmin(abs(alpha - unsteady["alpha0"]))

    if max(np.abs(np.gradient(cm))) > 1.0e-10:
        aoa_h = alpha[idx_alpha0] + 35.0
        idx_high = np.argmin(abs(alpha - aoa_h))

        cm_temp = cm[idx_low:idx_high]
        idx_cm_min = [
            i
            for i, local_min in enumerate(
                np.r_[True, cm_temp[1:] < cm_temp[:-1]] & np.r_[cm_temp[:-1] < cm_temp[1:], True]
            )
            if local_min
        ] + idx_low
        idx_high = idx_cm_min[-1]

        idx_Cn1 = find_breakpoint(alpha, cm, idx_alpha0, idx_high)
        unsteady["Cn1"] = cn[idx_Cn1]
    else:
        idx_Cn1 = np.argmin(abs(alpha - 0.0))
        unsteady["Cn1"] = 0.0

    # Cn2
    if max(np.abs(np.gradient(cm))) > 1.0e-10:
        aoa_l = np.mean([alpha[idx_alpha0], alpha[idx_Cn1]]) - 30.0
        idx_low = np.argmin(abs(alpha - aoa_l))

        cm_temp = cm[idx_low:idx_high]
        idx_cm_min = [
            i
            for i, local_min in enumerate(
                np.r_[True, cm_temp[1:] < cm_temp[:-1]] & np.r_[cm_temp[:-1] < cm_temp[1:], True]
            )
            if local_min
        ] + idx_low
        idx_high = idx_cm_min[-1]

        idx_Cn2 = find_breakpoint(alpha, cm, idx_low, idx_alpha0, multi=0.0)
        unsteady["Cn2"] = cn[idx_Cn2]
    else:
        idx_Cn2 = np.argmin(abs(alpha - 0.0))
        unsteady["Cn2"] = 0.0

    # C_nalpha
    if max(np.abs(np.gradient(cm))) > 1.0e-10:
        # unsteady['C_nalpha'] = np.gradient(cn, alpha_rad)[idx_alpha0]
        unsteady["C_nalpha"] = max(np.gradient(cn[idx_alpha0:idx_Cn1], alpha_rad[idx_alpha0:idx_Cn1]))

    else:
        unsteady["C_nalpha"] = 0.0

    # alpha1, alpha2
    # finding the break point in drag as a proxy for Trailing Edge separation, f=0.7
    # 3d stall corrections cause erroneous f calculations
    if max(np.abs(np.gradient(cm))) > 1.0e-10:
        aoa_l = [0.0]
        idx_low = np.argmin(abs(alpha - aoa_l))
        idx_alpha1 = find_breakpoint(alpha, cd, idx_low, idx_Cn1, multi=-1.0)
        unsteady["alpha1"] = alpha[idx_alpha1]
    else:
        idx_alpha1 = np.argmin(abs(alpha - 0.0))
        unsteady["alpha1"] = 0.0
    unsteady["alpha2"] = -1.0 * unsteady["alpha1"]

    unsteady["St_sh"] = "Default"
    unsteady["k0"] = 0
    unsteady["k1"] = 0
    unsteady["k2"] = 0
    unsteady["k3"] = 0
    unsteady["k1_hat"] = 0
    unsteady["x_cp_bar"] = "Default"
    unsteady["UACutout"] = "Default"
    unsteady["filtCutOff"] = "Default"

    unsteady["Alpha"] = alpha
    unsteady["Cl"] = cl
    unsteady["Cd"] = cd
    unsteady["Cm"] = cm

    return unsteady
