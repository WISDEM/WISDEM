import numpy as np
from openmdao.api import ExplicitComponent
from scipy.interpolate import PchipInterpolator

from wisdem.ccblade.ccblade import CCBlade, CCAirfoil
from wisdem.commonse.csystem import DirectionVector

class CCBladeLoads(ExplicitComponent):
    """
    Compute the aerodynamic forces along the blade span given a rotor speed,
    pitch angle, and wind speed.

    This component instantiates and calls a CCBlade instance to compute the loads.
    Analytic derivatives are provided for all inptus except all airfoils*,
    mu, rho, and shearExp.

    Parameters
    ----------
    V_load : float
        Hub height wind speed.
    Omega_load : float
        Rotor rotation speed.
    pitch_load : float
        Blade pitch setting.
    azimuth_load : float
        Blade azimuthal location.
    r : numpy array[n_span]
        Radial locations where blade is defined. Should be increasing and not
        go all the way to hub or tip.
    chord : numpy array[n_span]
        Chord length at each section.
    theta : numpy array[n_span]
        Twist angle at each section (positive decreases angle of attack).
    Rhub : float
        Hub radius.
    Rtip : float
        Distance between rotor center and blade tip along z axis of blade root c.s.
    hub_height : float
        Hub height.
    precone : float
        Precone angle.
    tilt : float
        Shaft tilt.
    yaw : float
        Yaw error.
    precurve : numpy array[n_span]
        Precurve at each section.
    precurveTip : float
        Precurve at tip.
    airfoils_cl : numpy array[n_span, n_aoa, n_Re]
        Lift coefficients, spanwise.
    airfoils_cd : numpy array[n_span, n_aoa, n_Re]
        Drag coefficients, spanwise.
    airfoils_cm : numpy array[n_span, n_aoa, n_Re]
        Moment coefficients, spanwise.
    airfoils_aoa : numpy array[n_aoa]
        Angle of attack grid for polars.
    airfoils_Re : numpy array[n_Re]
        Reynolds numbers of polars.
    nBlades : int
        Number of blades
    rho : float
        Density of air
    mu : float
        Dynamic viscosity of air
    shearExp : float
        Shear exponent.
    nSector : int
        Number of sectors to divide rotor face into in computing thrust and power.
    tiploss : boolean
        Include Prandtl tip loss model.
    hubloss : boolean
        Include Prandtl hub loss model.
    wakerotation : boolean
        Include effect of wake rotation (i.e., tangential induction factor is nonzero).
    usecd : boolean
        Use drag coefficient in computing induction factors.

    Returns
    -------
    loads_r : numpy array[n_span]
        Radial positions along blade going toward tip.
    loads_Px : numpy array[n_span]
         Distributed loads in blade-aligned x-direction.
    loads_Py : numpy array[n_span]
         Distributed loads in blade-aligned y-direction.
    loads_Pz : numpy array[n_span]
         Distributed loads in blade-aligned z-direction.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds

        # inputs
        self.add_input("V_load", val=20.0, units="m/s")
        self.add_input("Omega_load", val=0.0, units="rpm")
        self.add_input("pitch_load", val=0.0, units="deg")
        self.add_input("azimuth_load", val=0.0, units="deg")

        self.add_input("r", val=np.zeros(n_span), units="m")
        self.add_input("chord", val=np.zeros(n_span), units="m")
        self.add_input("theta", val=np.zeros(n_span), units="deg")
        self.add_input("Rhub", val=0.0, units="m")
        self.add_input("Rtip", val=0.0, units="m")
        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("precone", val=0.0, units="deg")
        self.add_input("tilt", val=0.0, units="deg")
        self.add_input("yaw", val=0.0, units="deg")
        self.add_input("precurve", val=np.zeros(n_span), units="m")
        self.add_input("precurveTip", val=0.0, units="m")

        # parameters
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re)))
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re)))
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re)))
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)))

        self.add_discrete_input("nBlades", val=0)
        self.add_input("rho", val=0.0, units="kg/m**3")
        self.add_input("mu", val=0.0, units="kg/(m*s)")
        self.add_input("shearExp", val=0.0)
        self.add_discrete_input("nSector", val=4)
        self.add_discrete_input("tiploss", val=True)
        self.add_discrete_input("hubloss", val=True)
        self.add_discrete_input("wakerotation", val=True)
        self.add_discrete_input("usecd", val=True)

        # outputs
        self.add_output("loads_r", val=np.zeros(n_span), units="m")
        self.add_output("loads_Px", val=np.zeros(n_span), units="N/m")
        self.add_output("loads_Py", val=np.zeros(n_span), units="N/m")
        self.add_output("loads_Pz", val=np.zeros(n_span), units="N/m")

        arange = np.arange(n_span)
        self.declare_partials(
            "loads_Px",
            [
                "Omega_load",
                "Rhub",
                "Rtip",
                "V_load",
                "azimuth_load",
                "chord",
                "hub_height",
                "pitch_load",
                "precone",
                "precurve",
                "r",
                "theta",
                "tilt",
                "yaw",
                "shearExp",
            ],
        )
        self.declare_partials(
            "loads_Py",
            [
                "Omega_load",
                "Rhub",
                "Rtip",
                "V_load",
                "azimuth_load",
                "chord",
                "hub_height",
                "pitch_load",
                "precone",
                "precurve",
                "r",
                "theta",
                "tilt",
                "yaw",
                "shearExp",
            ],
        )
        self.declare_partials("loads_Pz", "*", dependent=False)
        self.declare_partials("loads_r", "r", val=1.0, rows=arange, cols=arange)
        self.declare_partials("*", "airfoils*", dependent=False)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"][0]
        Rtip = inputs["Rtip"][0]
        hub_height = inputs["hub_height"][0]
        precone = inputs["precone"][0]
        tilt = inputs["tilt"][0]
        yaw = inputs["yaw"][0]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"][0]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"][0]
        mu = inputs["mu"][0]
        shearExp = inputs["shearExp"][0]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"][0]
        Omega_load = inputs["Omega_load"][0]
        pitch_load = inputs["pitch_load"][0]
        azimuth_load = inputs["azimuth_load"][0]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :],
                inputs["airfoils_cd"][i, :, :],
                inputs["airfoils_cm"][i, :, :],
            )

        ccblade = CCBlade(
            r,
            chord,
            theta,
            af,
            Rhub,
            Rtip,
            B,
            rho,
            mu,
            precone,
            tilt,
            yaw,
            shearExp,
            hub_height,
            nSector,
            precurve,
            precurveTip,
            tiploss=tiploss,
            hubloss=hubloss,
            wakerotation=wakerotation,
            usecd=usecd,
            derivatives=True,
        )

        # distributed loads
        loads, self.derivs = ccblade.distributedAeroLoads(V_load, Omega_load, pitch_load, azimuth_load)
        Np = loads["Np"]
        Tp = loads["Tp"]

        # unclear why we need this output at all
        outputs["loads_r"] = r

        # conform to blade-aligned coordinate system
        outputs["loads_Px"] = Np
        outputs["loads_Py"] = -Tp
        outputs["loads_Pz"][:] = 0.0

    def compute_partials(self, inputs, J, discrete_inputs):
        dNp = self.derivs["dNp"]
        dTp = self.derivs["dTp"]

        J["loads_Px", "r"] = dNp["dr"]
        J["loads_Px", "chord"] = dNp["dchord"]
        J["loads_Px", "theta"] = dNp["dtheta"]
        J["loads_Px", "Rhub"] = np.squeeze(dNp["dRhub"])
        J["loads_Px", "Rtip"] = np.squeeze(dNp["dRtip"])
        J["loads_Px", "hub_height"] = np.squeeze(dNp["dhubHt"])
        J["loads_Px", "precone"] = np.squeeze(dNp["dprecone"])
        J["loads_Px", "tilt"] = np.squeeze(dNp["dtilt"])
        J["loads_Px", "yaw"] = np.squeeze(dNp["dyaw"])
        J["loads_Px", "shearExp"] = np.squeeze(dNp["dshear"])
        J["loads_Px", "V_load"] = np.squeeze(dNp["dUinf"])
        J["loads_Px", "Omega_load"] = np.squeeze(dNp["dOmega"])
        J["loads_Px", "pitch_load"] = np.squeeze(dNp["dpitch"])
        J["loads_Px", "azimuth_load"] = np.squeeze(dNp["dazimuth"])
        J["loads_Px", "precurve"] = dNp["dprecurve"]

        J["loads_Py", "r"] = -dTp["dr"]
        J["loads_Py", "chord"] = -dTp["dchord"]
        J["loads_Py", "theta"] = -dTp["dtheta"]
        J["loads_Py", "Rhub"] = -np.squeeze(dTp["dRhub"])
        J["loads_Py", "Rtip"] = -np.squeeze(dTp["dRtip"])
        J["loads_Py", "hub_height"] = -np.squeeze(dTp["dhubHt"])
        J["loads_Py", "precone"] = -np.squeeze(dTp["dprecone"])
        J["loads_Py", "tilt"] = -np.squeeze(dTp["dtilt"])
        J["loads_Py", "yaw"] = -np.squeeze(dTp["dyaw"])
        J["loads_Py", "shearExp"] = -np.squeeze(dTp["dshear"])
        J["loads_Py", "V_load"] = -np.squeeze(dTp["dUinf"])
        J["loads_Py", "Omega_load"] = -np.squeeze(dTp["dOmega"])
        J["loads_Py", "pitch_load"] = -np.squeeze(dTp["dpitch"])
        J["loads_Py", "azimuth_load"] = -np.squeeze(dTp["dazimuth"])
        J["loads_Py", "precurve"] = -dTp["dprecurve"]


class CCBladeTwist(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]
        self.n_span = n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
        # self.n_af          = n_af      = af_init_options['n_af'] # Number of airfoils
        self.n_aoa = n_aoa = modeling_options["WISDEM"]["RotorSE"]["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = modeling_options["WISDEM"]["RotorSE"]["n_Re"]  # Number of Reynolds, so far hard set at 1
        n_opt_chord = opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"]
        n_opt_twist = opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]

        # Inputs
        self.add_input("Uhub", val=5.0, units="m/s", desc="Undisturbed wind speed")

        self.add_input("tsr", val=0.0, desc="Tip speed ratio")
        self.add_input("pitch", val=0.0, units="deg", desc="Pitch angle")
        self.add_input(
            "r",
            val=np.zeros(n_span),
            units="m",
            desc="radial locations where blade is defined (should be increasing and not go all the way to hub or tip)",
        )
        self.add_input(
            "s_opt_chord",
            val=np.zeros(n_opt_chord),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade chord",
        )
        self.add_input(
            "s_opt_theta",
            val=np.zeros(n_opt_twist),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist",
        )
        self.add_input("chord", val=np.zeros(n_span), units="m", desc="chord length at each section")
        self.add_input(
            "theta_in",
            val=np.zeros(n_span),
            units="rad",
            desc="twist angle at each section (positive decreases angle of attack)",
        )
        self.add_input(
            "aoa_op",
            val=np.pi * np.ones(n_span),
            desc="1D array with the operational angles of attack for the airfoils along blade span.",
            units="rad",
        )
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg", desc="angle of attack grid for polars")
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re)), desc="lift coefficients, spanwise")
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re)), desc="drag coefficients, spanwise")
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re)), desc="moment coefficients, spanwise")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)), desc="Reynolds numbers of polars")
        self.add_input("Rhub", val=0.0, units="m", desc="hub radius")
        self.add_input("Rtip", val=0.0, units="m", desc="Distance between rotor center and blade tip along z axis of blade root c.s.")
        self.add_input(
            "rthick", val=np.zeros(n_span), desc="1D array of the relative thicknesses of the blade defined along span."
        )
        self.add_input("precurve", val=np.zeros(n_span), units="m", desc="precurve at each section")
        self.add_input("precurveTip", val=0.0, units="m", desc="precurve at tip")
        self.add_input("presweep", val=np.zeros(n_span), units="m", desc="presweep at each section")
        self.add_input("presweepTip", val=0.0, units="m", desc="presweep at tip")
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

        # Outputs
        self.add_output(
            "theta",
            val=np.zeros(n_span),
            units="rad",
            desc="Twist angle at each section (positive decreases angle of attack)",
        )
        self.add_output("CP", val=0.0, desc="Rotor power coefficient")
        self.add_output("CM", val=0.0, desc="Blade flapwise moment coefficient")

        self.add_output(
            "local_airfoil_velocities",
            val=np.zeros(n_span),
            desc="Local relative velocities for the airfoils",
            units="m/s",
        )

        self.add_output("P", val=0.0, units="W", desc="Rotor aerodynamic power")
        self.add_output("T", val=0.0, units="N*m", desc="Rotor aerodynamic thrust")
        self.add_output("Q", val=0.0, units="N*m", desc="Rotor aerodynamic torque")
        self.add_output("M", val=0.0, units="N*m", desc="Blade root flapwise moment")

        self.add_output("a", val=np.zeros(n_span), desc="Axial induction  along blade span")
        self.add_output("ap", val=np.zeros(n_span), desc="Tangential induction along blade span")
        self.add_output("alpha", val=np.zeros(n_span), units="deg", desc="Angles of attack along blade span")
        self.add_output("cl", val=np.zeros(n_span), desc="Lift coefficients along blade span")
        self.add_output("cd", val=np.zeros(n_span), desc="Drag coefficients along blade span")
        n_opt = opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]
        self.add_output("cl_n_opt", val=np.zeros(n_opt), desc="Lift coefficients along blade span")
        self.add_output("cd_n_opt", val=np.zeros(n_opt), desc="Drag coefficients along blade span")
        self.add_output(
            "Px_b", val=np.zeros(n_span), units="N/m", desc="Distributed loads in blade-aligned x-direction"
        )
        self.add_output(
            "Py_b", val=np.zeros(n_span), units="N/m", desc="Distributed loads in blade-aligned y-direction"
        )
        self.add_output(
            "Pz_b", val=np.zeros(n_span), units="N/m", desc="Distributed loads in blade-aligned z-direction"
        )
        self.add_output("Px_af", val=np.zeros(n_span), units="N/m", desc="Distributed loads in airfoil x-direction")
        self.add_output("Py_af", val=np.zeros(n_span), units="N/m", desc="Distributed loads in airfoil y-direction")
        self.add_output("Pz_af", val=np.zeros(n_span), units="N/m", desc="Distributed loads in airfoil z-direction")
        self.add_output("LiftF", val=np.zeros(n_span), units="N/m", desc="Distributed lift force")
        self.add_output("DragF", val=np.zeros(n_span), units="N/m", desc="Distributed drag force")
        self.add_output("L_n_opt", val=np.zeros(n_opt), units="N/m", desc="Distributed lift force")
        self.add_output("D_n_opt", val=np.zeros(n_opt), units="N/m", desc="Distributed drag force")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Create Airfoil class instances
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :],
                inputs["airfoils_cd"][i, :, :],
                inputs["airfoils_cm"][i, :, :],
            )

        # Create the CCBlade class instance
        ccblade = CCBlade(
            inputs["r"],
            inputs["chord"],
            np.zeros_like(inputs["chord"]),
            af,
            inputs["Rhub"][0],
            inputs["Rtip"][0],
            discrete_inputs["nBlades"],
            inputs["rho"][0],
            inputs["mu"][0],
            inputs["precone"][0],
            inputs["tilt"][0],
            inputs["yaw"][0],
            inputs["shearExp"][0],
            inputs["hub_height"][0],
            discrete_inputs["nSector"],
            inputs["precurve"],
            inputs["precurveTip"][0],
            inputs["presweep"],
            inputs["presweepTip"][0],
            discrete_inputs["tiploss"],
            discrete_inputs["hubloss"],
            discrete_inputs["wakerotation"],
            discrete_inputs["usecd"],
        )

        Omega = inputs["tsr"][0] * inputs["Uhub"][0] / (
            inputs["Rtip"][0] * np.cos(np.deg2rad(inputs["precone"][0]))) 
        Omega_rpm = Omega * 30.0 / np.pi

        if self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["inverse"]:
            if self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["flag"]:
                raise Exception(
                    "Twist cannot be simultaneously optimized and set to be defined inverting the BEM equations. Please check your analysis options yaml."
                )
            # Find cl and cd along blade span. Mix inputs from the airfoil INN (if active) and the airfoil for max efficiency
            cl = np.zeros(self.n_span)
            cd = np.zeros(self.n_span)
            alpha = np.zeros(self.n_span)
            Emax = np.zeros(self.n_span)
            margin2stall = self.options["opt_options"]["constraints"]["blade"]["stall"]["margin"] * 180.0 / np.pi
            Re = np.array(Omega * inputs["r"] * inputs["chord"] * inputs["rho"][0] / inputs["mu"][0])
            aoa_op = inputs["aoa_op"]
            for i in range(self.n_span):
                # Use the required angle of attack if defined. If it isn't defined (==pi), then take the stall point minus the margin
                if abs(aoa_op[i] - np.pi) > 1.0e-4:
                    alpha[i] = aoa_op[i]
                elif self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["inverse_target"] == 'stall_margin':
                    af[i].eval_unsteady(
                        inputs["airfoils_aoa"],
                        inputs["airfoils_cl"][i, :, 0],
                        inputs["airfoils_cd"][i, :, 0],
                        inputs["airfoils_cm"][i, :, 0],
                    )
                    alpha[i] = (af[i].unsteady["alpha1"] - margin2stall) / 180.0 * np.pi
                elif self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["inverse_target"] == 'max_efficiency':
                    af[i].eval_unsteady(
                        inputs["airfoils_aoa"],
                        inputs["airfoils_cl"][i, :, 0],
                        inputs["airfoils_cd"][i, :, 0],
                        inputs["airfoils_cm"][i, :, 0],
                    )
                    Emax[i], alpha[i], _, _ = af[i].max_eff(Re[i])
                else:
                    raise Exception('The flags for the twist inverse design are not set appropriately. Please check documentation for the available analysis options.')
                cl[i], cd[i] = af[i].evaluate(alpha[i], Re[i])

            # Overwrite aoa of high thickness airfoils at blade root
            idx_min = [i for i, thk in enumerate(inputs["rthick"]) if thk < 95.0][0]
            alpha[0:idx_min] = alpha[idx_min]

            # Call ccblade in inverse mode for desired alpha, cl, and cd along blade span
            ccblade.inverse_analysis = True
            ccblade.alpha = alpha
            ccblade.cl = cl
            ccblade.cd = cd
            _, _ = ccblade.evaluate([inputs["Uhub"]], [Omega_rpm], [inputs["pitch"]], coefficients=False)

            # Cap twist root region to 20 degrees
            for i in range(len(ccblade.theta)):
                cap_twist_root = self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["cap_twist_root"]
                if ccblade.theta[-i - 1] > cap_twist_root:
                    ccblade.theta[0 : len(ccblade.theta) - i] = cap_twist_root
                    break
        else:
            ccblade.theta = inputs["theta_in"]

        # Smooth out twist profile if we're doing inverse design for twist
        if self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["inverse"]:
            training_theta = ccblade.theta
            s = (inputs["r"] - inputs["r"][0]) / (inputs["r"][-1] - inputs["r"][0])

            twist_spline = PchipInterpolator(s, training_theta)
            theta_opt = twist_spline(inputs["s_opt_theta"])

            twist_spline = PchipInterpolator(inputs["s_opt_theta"], theta_opt)
            theta_full = twist_spline(s)
            ccblade.theta = theta_full

        # Turn off the inverse analysis
        ccblade.inverse_analysis = False

        # Call ccblade at azimuth 0 deg
        loads, _ = ccblade.distributedAeroLoads(inputs["Uhub"][0], Omega_rpm, inputs["pitch"][0], 0.0)

        # Call ccblade evaluate (averaging across azimuth)
        myout, _ = ccblade.evaluate([inputs["Uhub"]], [Omega_rpm], [inputs["pitch"]], coefficients=True)
        CP, CMb, W = [myout[key] for key in ["CP", "CMb", "W"]]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(inputs['r'], np.rad2deg(alpha), '-')
        # plt.plot(inputs['r'], loads["alpha"], ':')
        # plt.xlabel('blade fraction')
        # plt.ylabel('aoa (deg)')
        # plt.legend()
        # plt.show()
        # exit()

        # Return twist angle
        outputs["theta"] = ccblade.theta
        outputs["CP"] = CP[0]
        outputs["CM"] = CMb[0]
        outputs["local_airfoil_velocities"] = np.nan_to_num(W)
        outputs["a"] = loads["a"]
        outputs["ap"] = loads["ap"]
        outputs["alpha"] = loads["alpha"]
        outputs["cl"] = loads["Cl"]
        outputs["cd"] = loads["Cd"]
        s = (inputs["r"] - inputs["r"][0]) / (inputs["r"][-1] - inputs["r"][0])
        outputs["cl_n_opt"] = np.interp(inputs["s_opt_theta"], s, loads["Cl"])
        outputs["cd_n_opt"] = np.interp(inputs["s_opt_theta"], s, loads["Cd"])
        # Forces in the blade coordinate system, pag 21 of https://www.nrel.gov/docs/fy13osti/58819.pdf
        outputs["Px_b"] = loads["Np"]
        outputs["Py_b"] = -loads["Tp"]
        outputs["Pz_b"] = 0 * loads["Np"]
        # Forces in the airfoil coordinate system, pag 21 of https://www.nrel.gov/docs/fy13osti/58819.pdf
        P_b = DirectionVector(loads["Np"], -loads["Tp"], 0)
        P_af = P_b.bladeToAirfoil(ccblade.theta * 180.0 / np.pi)
        outputs["Px_af"] = P_af.x
        outputs["Py_af"] = P_af.y
        outputs["Pz_af"] = P_af.z
        # Lift and drag forces
        F = P_b.bladeToAirfoil(ccblade.theta * 180.0 / np.pi + loads["alpha"] + inputs["pitch"])
        outputs["LiftF"] = F.x
        outputs["DragF"] = F.y
        outputs["L_n_opt"] = np.interp(inputs["s_opt_theta"], s, F.x)
        outputs["D_n_opt"] = np.interp(inputs["s_opt_theta"], s, F.y)
        # print(CP[0])


class CCBladeEvaluate(ExplicitComponent):
    """
    Standalone component for CCBlade that is only a light wrapper on CCBlade()
    to run the instance evaluate and compute aerodynamic hub forces and moments, blade
    root flapwise moment, and power. The coefficients are also computed.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_init_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_init_options["n_span"]
        self.n_aoa = n_aoa = rotorse_init_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_init_options["n_Re"]  # Number of Reynolds

        # inputs
        self.add_input("V_load", val=20.0, units="m/s")
        self.add_input("Omega_load", val=9.0, units="rpm")
        self.add_input("pitch_load", val=0.0, units="deg")

        self.add_input("r", val=np.zeros(n_span), units="m")
        self.add_input("chord", val=np.zeros(n_span), units="m")
        self.add_input("theta", val=np.zeros(n_span), units="deg")
        self.add_input("Rhub", val=0.0, units="m")
        self.add_input("Rtip", val=0.0, units="m")
        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("precone", val=0.0, units="deg")
        self.add_input("tilt", val=0.0, units="deg")
        self.add_input("yaw", val=0.0, units="deg")
        self.add_input("precurve", val=np.zeros(n_span), units="m")
        self.add_input("precurveTip", val=0.0, units="m")
        self.add_input("presweep", val=np.zeros(n_span), units="m")
        self.add_input("presweepTip", val=0.0, units="m")

        # parameters
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re)))
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re)))
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re)))
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)))

        self.add_discrete_input("nBlades", val=0)
        self.add_input("rho", val=0.0, units="kg/m**3")
        self.add_input("mu", val=0.0, units="kg/(m*s)")
        self.add_input("shearExp", val=0.0)
        self.add_discrete_input("nSector", val=4)
        self.add_discrete_input("tiploss", val=True)
        self.add_discrete_input("hubloss", val=True)
        self.add_discrete_input("wakerotation", val=True)
        self.add_discrete_input("usecd", val=True)

        # outputs
        self.add_output("P", val=0.0, units="W", desc="Rotor aerodynamic power")
        self.add_output("Mb", val=0.0, units="N/m", desc="Aerodynamic blade root flapwise moment")
        self.add_output("Fhub", val=np.zeros(3), units="N", desc="Aerodynamic forces at hub center in the hub c.s.")
        self.add_output("Mhub", val=np.zeros(3), units="N*m", desc="Aerodynamic moments at hub center in the hub c.s.")
        self.add_output("CP", val=0.0, desc="Rotor aerodynamic power coefficient")
        self.add_output("CMb", val=0.0, desc="Aerodynamic blade root flapwise moment coefficient")
        self.add_output("CFhub", val=np.zeros(3), desc="Aerodynamic force coefficients at hub center in the hub c.s.")
        self.add_output("CMhub", val=np.zeros(3), desc="Aerodynamic moment coefficients at hub center in the hub c.s.")

        self.declare_partials("*", "*")
        self.declare_partials("*", "airfoils*", dependent=False)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"][0]
        Rtip = inputs["Rtip"][0]
        hub_height = inputs["hub_height"][0]
        precone = inputs["precone"][0]
        tilt = inputs["tilt"][0]
        yaw = inputs["yaw"][0]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"][0]
        presweep = inputs["presweep"]
        presweepTip = inputs["presweepTip"][0]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"][0]
        mu = inputs["mu"][0]
        shearExp = inputs["shearExp"][0]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"][0]
        Omega_load = inputs["Omega_load"][0]
        pitch_load = inputs["pitch_load"][0]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :],
                inputs["airfoils_cd"][i, :, :],
                inputs["airfoils_cm"][i, :, :],
            )

        ccblade = CCBlade(
            r,
            chord,
            theta,
            af,
            Rhub,
            Rtip,
            B,
            rho,
            mu,
            precone,
            tilt,
            yaw,
            shearExp,
            hub_height,
            nSector,
            precurve,
            precurveTip,
            presweep,
            presweepTip,
            tiploss=tiploss,
            hubloss=hubloss,
            wakerotation=wakerotation,
            usecd=usecd,
            derivatives=False,
        )

        loads, _ = ccblade.evaluate(V_load, Omega_load, pitch_load, coefficients=True)
        outputs["P"] = loads["P"]
        outputs["Mb"] = loads["Mb"]
        outputs["CP"] = loads["CP"]
        outputs["CMb"] = loads["CMb"]
        outputs["Fhub"] = np.array([loads["T"], loads["Y"], loads["Z"]])
        outputs["Mhub"] = np.array([loads["Q"], loads["My"], loads["Mz"]])
        outputs["CFhub"] = np.array([loads["CT"], loads["CY"], loads["CZ"]])
        outputs["CMhub"] = np.array([loads["CQ"], loads["CMy"], loads["CMz"]])

    def compute_partials(self, inputs, J, discrete_inputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"][0]
        Rtip = inputs["Rtip"][0]
        hub_height = inputs["hub_height"][0]
        precone = inputs["precone"][0]
        tilt = inputs["tilt"][0]
        yaw = inputs["yaw"][0]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"][0]
        presweep = inputs["presweep"]
        presweepTip = inputs["presweepTip"][0]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"][0]
        mu = inputs["mu"][0]
        shearExp = inputs["shearExp"][0]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"][0]
        Omega_load = inputs["Omega_load"][0]
        pitch_load = inputs["pitch_load"][0]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :],
                inputs["airfoils_cd"][i, :, :],
                inputs["airfoils_cm"][i, :, :],
            )

        ccblade = CCBlade(
            r,
            chord,
            theta,
            af,
            Rhub,
            Rtip,
            B,
            rho,
            mu,
            precone,
            tilt,
            yaw,
            shearExp,
            hub_height,
            nSector,
            precurve,
            precurveTip,
            presweep,
            presweepTip,
            tiploss=tiploss,
            hubloss=hubloss,
            wakerotation=wakerotation,
            usecd=usecd,
            derivatives=True,
        )

        loads, derivs = ccblade.evaluate(V_load, Omega_load, pitch_load, coefficients=True)

        dP = derivs["dP"]
        J["P", "r"] = dP["dr"]
        J["P", "chord"] = dP["dchord"]
        J["P", "theta"] = dP["dtheta"]
        J["P", "Rhub"] = np.squeeze(dP["dRhub"])
        J["P", "Rtip"] = np.squeeze(dP["dRtip"])
        J["P", "hub_height"] = np.squeeze(dP["dhubHt"])
        J["P", "precone"] = np.squeeze(dP["dprecone"])
        J["P", "tilt"] = np.squeeze(dP["dtilt"])
        J["P", "yaw"] = np.squeeze(dP["dyaw"])
        J["P", "shearExp"] = np.squeeze(dP["dshear"])
        J["P", "V_load"] = np.squeeze(dP["dUinf"])
        J["P", "Omega_load"] = np.squeeze(dP["dOmega"])
        J["P", "pitch_load"] = np.squeeze(dP["dpitch"])
        J["P", "precurve"] = dP["dprecurve"]
        J["P", "precurveTip"] = dP["dprecurveTip"]
        J["P", "presweep"] = dP["dpresweep"]
        J["P", "presweepTip"] = dP["dpresweepTip"]

        dT = derivs["dT"]
        J["Fhub", "r"][0, :] = dT["dr"]
        J["Fhub", "chord"][0, :] = dT["dchord"]
        J["Fhub", "theta"][0, :] = dT["dtheta"]
        J["Fhub", "Rhub"][0, :] = np.squeeze(dT["dRhub"])
        J["Fhub", "Rtip"][0, :] = np.squeeze(dT["dRtip"])
        J["Fhub", "hub_height"][0, :] = np.squeeze(dT["dhubHt"])
        J["Fhub", "precone"][0, :] = np.squeeze(dT["dprecone"])
        J["Fhub", "tilt"][0, :] = np.squeeze(dT["dtilt"])
        J["Fhub", "yaw"][0, :] = np.squeeze(dT["dyaw"])
        J["Fhub", "shearExp"][0, :] = np.squeeze(dT["dshear"])
        J["Fhub", "V_load"][0, :] = np.squeeze(dT["dUinf"])
        J["Fhub", "Omega_load"][0, :] = np.squeeze(dT["dOmega"])
        J["Fhub", "pitch_load"][0, :] = np.squeeze(dT["dpitch"])
        J["Fhub", "precurve"][0, :] = dT["dprecurve"]
        J["Fhub", "precurveTip"][0, :] = dT["dprecurveTip"]
        J["Fhub", "presweep"][0, :] = dT["dpresweep"]
        J["Fhub", "presweepTip"][0, :] = dT["dpresweepTip"]

        dY = derivs["dY"]
        J["Fhub", "r"][1, :] = dY["dr"]
        J["Fhub", "chord"][1, :] = dY["dchord"]
        J["Fhub", "theta"][1, :] = dY["dtheta"]
        J["Fhub", "Rhub"][1, :] = np.squeeze(dY["dRhub"])
        J["Fhub", "Rtip"][1, :] = np.squeeze(dY["dRtip"])
        J["Fhub", "hub_height"][1, :] = np.squeeze(dY["dhubHt"])
        J["Fhub", "precone"][1, :] = np.squeeze(dY["dprecone"])
        J["Fhub", "tilt"][1, :] = np.squeeze(dY["dtilt"])
        J["Fhub", "yaw"][1, :] = np.squeeze(dY["dyaw"])
        J["Fhub", "shearExp"][1, :] = np.squeeze(dY["dshear"])
        J["Fhub", "V_load"][1, :] = np.squeeze(dY["dUinf"])
        J["Fhub", "Omega_load"][1, :] = np.squeeze(dY["dOmega"])
        J["Fhub", "pitch_load"][1, :] = np.squeeze(dY["dpitch"])
        J["Fhub", "precurve"][1, :] = dY["dprecurve"]
        J["Fhub", "precurveTip"][1, :] = dY["dprecurveTip"]
        J["Fhub", "presweep"][1, :] = dY["dpresweep"]
        J["Fhub", "presweepTip"][1, :] = dY["dpresweepTip"]

        dZ = derivs["dZ"]
        J["Fhub", "r"][2, :] = dZ["dr"]
        J["Fhub", "chord"][2, :] = dZ["dchord"]
        J["Fhub", "theta"][2, :] = dZ["dtheta"]
        J["Fhub", "Rhub"][2, :] = np.squeeze(dZ["dRhub"])
        J["Fhub", "Rtip"][2, :] = np.squeeze(dZ["dRtip"])
        J["Fhub", "hub_height"][2, :] = np.squeeze(dZ["dhubHt"])
        J["Fhub", "precone"][2, :] = np.squeeze(dZ["dprecone"])
        J["Fhub", "tilt"][2, :] = np.squeeze(dZ["dtilt"])
        J["Fhub", "yaw"][2, :] = np.squeeze(dZ["dyaw"])
        J["Fhub", "shearExp"][2, :] = np.squeeze(dZ["dshear"])
        J["Fhub", "V_load"][2, :] = np.squeeze(dZ["dUinf"])
        J["Fhub", "Omega_load"][2, :] = np.squeeze(dZ["dOmega"])
        J["Fhub", "pitch_load"][2, :] = np.squeeze(dZ["dpitch"])
        J["Fhub", "precurve"][2, :] = dZ["dprecurve"]
        J["Fhub", "precurveTip"][2, :] = dZ["dprecurveTip"]
        J["Fhub", "presweep"][2, :] = dZ["dpresweep"]
        J["Fhub", "presweepTip"][2, :] = dZ["dpresweepTip"]

        dQ = derivs["dQ"]
        J["Mhub", "r"][0, :] = dQ["dr"]
        J["Mhub", "chord"][0, :] = dQ["dchord"]
        J["Mhub", "theta"][0, :] = dQ["dtheta"]
        J["Mhub", "Rhub"][0, :] = np.squeeze(dQ["dRhub"])
        J["Mhub", "Rtip"][0, :] = np.squeeze(dQ["dRtip"])
        J["Mhub", "hub_height"][0, :] = np.squeeze(dQ["dhubHt"])
        J["Mhub", "precone"][0, :] = np.squeeze(dQ["dprecone"])
        J["Mhub", "tilt"][0, :] = np.squeeze(dQ["dtilt"])
        J["Mhub", "yaw"][0, :] = np.squeeze(dQ["dyaw"])
        J["Mhub", "shearExp"][0, :] = np.squeeze(dQ["dshear"])
        J["Mhub", "V_load"][0, :] = np.squeeze(dQ["dUinf"])
        J["Mhub", "Omega_load"][0, :] = np.squeeze(dQ["dOmega"])
        J["Mhub", "pitch_load"][0, :] = np.squeeze(dQ["dpitch"])
        J["Mhub", "precurve"][0, :] = dQ["dprecurve"]
        J["Mhub", "precurveTip"][0, :] = dQ["dprecurveTip"]
        J["Mhub", "presweep"][0, :] = dQ["dpresweep"]
        J["Mhub", "presweepTip"][0, :] = dQ["dpresweepTip"]

        dMy = derivs["dMy"]
        J["Mhub", "r"][1, :] = dMy["dr"]
        J["Mhub", "chord"][1, :] = dMy["dchord"]
        J["Mhub", "theta"][1, :] = dMy["dtheta"]
        J["Mhub", "Rhub"][1, :] = np.squeeze(dMy["dRhub"])
        J["Mhub", "Rtip"][1, :] = np.squeeze(dMy["dRtip"])
        J["Mhub", "hub_height"][1, :] = np.squeeze(dMy["dhubHt"])
        J["Mhub", "precone"][1, :] = np.squeeze(dMy["dprecone"])
        J["Mhub", "tilt"][1, :] = np.squeeze(dMy["dtilt"])
        J["Mhub", "yaw"][1, :] = np.squeeze(dMy["dyaw"])
        J["Mhub", "shearExp"][1, :] = np.squeeze(dMy["dshear"])
        J["Mhub", "V_load"][1, :] = np.squeeze(dMy["dUinf"])
        J["Mhub", "Omega_load"][1, :] = np.squeeze(dMy["dOmega"])
        J["Mhub", "pitch_load"][1, :] = np.squeeze(dMy["dpitch"])
        J["Mhub", "precurve"][1, :] = dMy["dprecurve"]
        J["Mhub", "precurveTip"][1, :] = dMy["dprecurveTip"]
        J["Mhub", "presweep"][1, :] = dMy["dpresweep"]
        J["Mhub", "presweepTip"][1, :] = dMy["dpresweepTip"]

        dMz = derivs["dMz"]
        J["Mhub", "r"][2, :] = dMz["dr"]
        J["Mhub", "chord"][2, :] = dMz["dchord"]
        J["Mhub", "theta"][2, :] = dMz["dtheta"]
        J["Mhub", "Rhub"][2, :] = np.squeeze(dMz["dRhub"])
        J["Mhub", "Rtip"][2, :] = np.squeeze(dMz["dRtip"])
        J["Mhub", "hub_height"][2, :] = np.squeeze(dMz["dhubHt"])
        J["Mhub", "precone"][2, :] = np.squeeze(dMz["dprecone"])
        J["Mhub", "tilt"][2, :] = np.squeeze(dMz["dtilt"])
        J["Mhub", "yaw"][2, :] = np.squeeze(dMz["dyaw"])
        J["Mhub", "shearExp"][2, :] = np.squeeze(dMz["dshear"])
        J["Mhub", "V_load"][2, :] = np.squeeze(dMz["dUinf"])
        J["Mhub", "Omega_load"][2, :] = np.squeeze(dMz["dOmega"])
        J["Mhub", "pitch_load"][2, :] = np.squeeze(dMz["dpitch"])
        J["Mhub", "precurve"][2, :] = dMz["dprecurve"]
        J["Mhub", "precurveTip"][2, :] = dMz["dprecurveTip"]
        J["Mhub", "presweep"][2, :] = dMz["dpresweep"]
        J["Mhub", "presweepTip"][2, :] = dMz["dpresweepTip"]

        dMb = derivs["dMb"]
        J["Mb", "r"] = dMb["dr"]
        J["Mb", "chord"] = dMb["dchord"]
        J["Mb", "theta"] = dMb["dtheta"]
        J["Mb", "Rhub"] = np.squeeze(dMb["dRhub"])
        J["Mb", "Rtip"] = np.squeeze(dMb["dRtip"])
        J["Mb", "hub_height"] = np.squeeze(dMb["dhubHt"])
        J["Mb", "precone"] = np.squeeze(dMb["dprecone"])
        J["Mb", "tilt"] = np.squeeze(dMb["dtilt"])
        J["Mb", "yaw"] = np.squeeze(dMb["dyaw"])
        J["Mb", "shearExp"] = np.squeeze(dMb["dshear"])
        J["Mb", "V_load"] = np.squeeze(dMb["dUinf"])
        J["Mb", "Omega_load"] = np.squeeze(dMb["dOmega"])
        J["Mb", "pitch_load"] = np.squeeze(dMb["dpitch"])
        J["Mb", "precurve"] = dMb["dprecurve"]
        J["Mb", "precurveTip"] = dMb["dprecurveTip"]
        J["Mb", "presweep"] = dMb["dpresweep"]
        J["Mb", "presweepTip"] = dMb["dpresweepTip"]

        dCP = derivs["dCP"]
        J["CP", "r"] = dCP["dr"]
        J["CP", "chord"] = dCP["dchord"]
        J["CP", "theta"] = dCP["dtheta"]
        J["CP", "Rhub"] = np.squeeze(dCP["dRhub"])
        J["CP", "Rtip"] = np.squeeze(dCP["dRtip"])
        J["CP", "hub_height"] = np.squeeze(dCP["dhubHt"])
        J["CP", "precone"] = np.squeeze(dCP["dprecone"])
        J["CP", "tilt"] = np.squeeze(dCP["dtilt"])
        J["CP", "yaw"] = np.squeeze(dCP["dyaw"])
        J["CP", "shearExp"] = np.squeeze(dCP["dshear"])
        J["CP", "V_load"] = np.squeeze(dCP["dUinf"])
        J["CP", "Omega_load"] = np.squeeze(dCP["dOmega"])
        J["CP", "pitch_load"] = np.squeeze(dCP["dpitch"])
        J["CP", "precurve"] = dCP["dprecurve"]
        J["CP", "precurveTip"] = dCP["dprecurveTip"]
        J["CP", "presweep"] = dCP["dpresweep"]
        J["CP", "presweepTip"] = dCP["dpresweepTip"]

        dCT = derivs["dCT"]
        J["CFhub", "r"][0, :] = dCT["dr"]
        J["CFhub", "chord"][0, :] = dCT["dchord"]
        J["CFhub", "theta"][0, :] = dCT["dtheta"]
        J["CFhub", "Rhub"][0, :] = np.squeeze(dCT["dRhub"])
        J["CFhub", "Rtip"][0, :] = np.squeeze(dCT["dRtip"])
        J["CFhub", "hub_height"][0, :] = np.squeeze(dCT["dhubHt"])
        J["CFhub", "precone"][0, :] = np.squeeze(dCT["dprecone"])
        J["CFhub", "tilt"][0, :] = np.squeeze(dCT["dtilt"])
        J["CFhub", "yaw"][0, :] = np.squeeze(dCT["dyaw"])
        J["CFhub", "shearExp"][0, :] = np.squeeze(dCT["dshear"])
        J["CFhub", "V_load"][0, :] = np.squeeze(dCT["dUinf"])
        J["CFhub", "Omega_load"][0, :] = np.squeeze(dCT["dOmega"])
        J["CFhub", "pitch_load"][0, :] = np.squeeze(dCT["dpitch"])
        J["CFhub", "precurve"][0, :] = dCT["dprecurve"]
        J["CFhub", "precurveTip"][0, :] = dCT["dprecurveTip"]
        J["CFhub", "presweep"][0, :] = dCT["dpresweep"]
        J["CFhub", "presweepTip"][0, :] = dCT["dpresweepTip"]

        dCY = derivs["dCY"]
        J["CFhub", "r"][1, :] = dCY["dr"]
        J["CFhub", "chord"][1, :] = dCY["dchord"]
        J["CFhub", "theta"][1, :] = dCY["dtheta"]
        J["CFhub", "Rhub"][1, :] = np.squeeze(dCY["dRhub"])
        J["CFhub", "Rtip"][1, :] = np.squeeze(dCY["dRtip"])
        J["CFhub", "hub_height"][1, :] = np.squeeze(dCY["dhubHt"])
        J["CFhub", "precone"][1, :] = np.squeeze(dCY["dprecone"])
        J["CFhub", "tilt"][1, :] = np.squeeze(dCY["dtilt"])
        J["CFhub", "yaw"][1, :] = np.squeeze(dCY["dyaw"])
        J["CFhub", "shearExp"][1, :] = np.squeeze(dCY["dshear"])
        J["CFhub", "V_load"][1, :] = np.squeeze(dCY["dUinf"])
        J["CFhub", "Omega_load"][1, :] = np.squeeze(dCY["dOmega"])
        J["CFhub", "pitch_load"][1, :] = np.squeeze(dCY["dpitch"])
        J["CFhub", "precurve"][1, :] = dCY["dprecurve"]
        J["CFhub", "precurveTip"][1, :] = dCY["dprecurveTip"]
        J["CFhub", "presweep"][1, :] = dCY["dpresweep"]
        J["CFhub", "presweepTip"][1, :] = dCY["dpresweepTip"]

        dCZ = derivs["dCZ"]
        J["CFhub", "r"][2, :] = dCZ["dr"]
        J["CFhub", "chord"][2, :] = dCZ["dchord"]
        J["CFhub", "theta"][2, :] = dCZ["dtheta"]
        J["CFhub", "Rhub"][2, :] = np.squeeze(dCZ["dRhub"])
        J["CFhub", "Rtip"][2, :] = np.squeeze(dCZ["dRtip"])
        J["CFhub", "hub_height"][2, :] = np.squeeze(dCZ["dhubHt"])
        J["CFhub", "precone"][2, :] = np.squeeze(dCZ["dprecone"])
        J["CFhub", "tilt"][2, :] = np.squeeze(dCZ["dtilt"])
        J["CFhub", "yaw"][2, :] = np.squeeze(dCZ["dyaw"])
        J["CFhub", "shearExp"][2, :] = np.squeeze(dCZ["dshear"])
        J["CFhub", "V_load"][2, :] = np.squeeze(dCZ["dUinf"])
        J["CFhub", "Omega_load"][2, :] = np.squeeze(dCZ["dOmega"])
        J["CFhub", "pitch_load"][2, :] = np.squeeze(dCZ["dpitch"])
        J["CFhub", "precurve"][2, :] = dCZ["dprecurve"]
        J["CFhub", "precurveTip"][2, :] = dCZ["dprecurveTip"]
        J["CFhub", "presweep"][2, :] = dCZ["dpresweep"]
        J["CFhub", "presweepTip"][2, :] = dCZ["dpresweepTip"]

        dCQ = derivs["dCQ"]
        J["CMhub", "r"][0, :] = dCQ["dr"]
        J["CMhub", "chord"][0, :] = dCQ["dchord"]
        J["CMhub", "theta"][0, :] = dCQ["dtheta"]
        J["CMhub", "Rhub"][0, :] = np.squeeze(dCQ["dRhub"])
        J["CMhub", "Rtip"][0, :] = np.squeeze(dCQ["dRtip"])
        J["CMhub", "hub_height"][0, :] = np.squeeze(dCQ["dhubHt"])
        J["CMhub", "precone"][0, :] = np.squeeze(dCQ["dprecone"])
        J["CMhub", "tilt"][0, :] = np.squeeze(dCQ["dtilt"])
        J["CMhub", "yaw"][0, :] = np.squeeze(dCQ["dyaw"])
        J["CMhub", "shearExp"][0, :] = np.squeeze(dCQ["dshear"])
        J["CMhub", "V_load"][0, :] = np.squeeze(dCQ["dUinf"])
        J["CMhub", "Omega_load"][0, :] = np.squeeze(dCQ["dOmega"])
        J["CMhub", "pitch_load"][0, :] = np.squeeze(dCQ["dpitch"])
        J["CMhub", "precurve"][0, :] = dCQ["dprecurve"]
        J["CMhub", "precurveTip"][0, :] = dCQ["dprecurveTip"]
        J["CMhub", "presweep"][0, :] = dCQ["dpresweep"]
        J["CMhub", "presweepTip"][0, :] = dCQ["dpresweepTip"]

        dCMy = derivs["dCMy"]
        J["CMhub", "r"][1, :] = dCMy["dr"]
        J["CMhub", "chord"][1, :] = dCMy["dchord"]
        J["CMhub", "theta"][1, :] = dCMy["dtheta"]
        J["CMhub", "Rhub"][1, :] = np.squeeze(dCMy["dRhub"])
        J["CMhub", "Rtip"][1, :] = np.squeeze(dCMy["dRtip"])
        J["CMhub", "hub_height"][1, :] = np.squeeze(dCMy["dhubHt"])
        J["CMhub", "precone"][1, :] = np.squeeze(dCMy["dprecone"])
        J["CMhub", "tilt"][1, :] = np.squeeze(dCMy["dtilt"])
        J["CMhub", "yaw"][1, :] = np.squeeze(dCMy["dyaw"])
        J["CMhub", "shearExp"][1, :] = np.squeeze(dCMy["dshear"])
        J["CMhub", "V_load"][1, :] = np.squeeze(dCMy["dUinf"])
        J["CMhub", "Omega_load"][1, :] = np.squeeze(dCMy["dOmega"])
        J["CMhub", "pitch_load"][1, :] = np.squeeze(dCMy["dpitch"])
        J["CMhub", "precurve"][1, :] = dCMy["dprecurve"]
        J["CMhub", "precurveTip"][1, :] = dCMy["dprecurveTip"]
        J["CMhub", "presweep"][1, :] = dCMy["dpresweep"]
        J["CMhub", "presweepTip"][1, :] = dCMy["dpresweepTip"]

        dCMz = derivs["dCMz"]
        J["CMhub", "r"][2, :] = dCMz["dr"]
        J["CMhub", "chord"][2, :] = dCMz["dchord"]
        J["CMhub", "theta"][2, :] = dCMz["dtheta"]
        J["CMhub", "Rhub"][2, :] = np.squeeze(dCMz["dRhub"])
        J["CMhub", "Rtip"][2, :] = np.squeeze(dCMz["dRtip"])
        J["CMhub", "hub_height"][2, :] = np.squeeze(dCMz["dhubHt"])
        J["CMhub", "precone"][2, :] = np.squeeze(dCMz["dprecone"])
        J["CMhub", "tilt"][2, :] = np.squeeze(dCMz["dtilt"])
        J["CMhub", "yaw"][2, :] = np.squeeze(dCMz["dyaw"])
        J["CMhub", "shearExp"][2, :] = np.squeeze(dCMz["dshear"])
        J["CMhub", "V_load"][2, :] = np.squeeze(dCMz["dUinf"])
        J["CMhub", "Omega_load"][2, :] = np.squeeze(dCMz["dOmega"])
        J["CMhub", "pitch_load"][2, :] = np.squeeze(dCMz["dpitch"])
        J["CMhub", "precurve"][2, :] = dCMz["dprecurve"]
        J["CMhub", "precurveTip"][2, :] = dCMz["dprecurveTip"]
        J["CMhub", "presweep"][2, :] = dCMz["dpresweep"]
        J["CMhub", "presweepTip"][2, :] = dCMz["dpresweepTip"]

        dCMb = derivs["dCMb"]
        J["CMb", "r"] = dCMb["dr"]
        J["CMb", "chord"] = dCMb["dchord"]
        J["CMb", "theta"] = dCMb["dtheta"]
        J["CMb", "Rhub"] = np.squeeze(dCMb["dRhub"])
        J["CMb", "Rtip"] = np.squeeze(dCMb["dRtip"])
        J["CMb", "hub_height"] = np.squeeze(dCMb["dhubHt"])
        J["CMb", "precone"] = np.squeeze(dCMb["dprecone"])
        J["CMb", "tilt"] = np.squeeze(dCMb["dtilt"])
        J["CMb", "yaw"] = np.squeeze(dCMb["dyaw"])
        J["CMb", "shearExp"] = np.squeeze(dCMb["dshear"])
        J["CMb", "V_load"] = np.squeeze(dCMb["dUinf"])
        J["CMb", "Omega_load"] = np.squeeze(dCMb["dOmega"])
        J["CMb", "pitch_load"] = np.squeeze(dCMb["dpitch"])
        J["CMb", "precurve"] = dCMb["dprecurve"]
        J["CMb", "precurveTip"] = dCMb["dprecurveTip"]
        J["CMb", "presweep"] = dCMb["dpresweep"]
        J["CMb", "presweepTip"] = dCMb["dpresweepTip"]
