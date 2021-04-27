import numpy as np
import wisdem.ccblade._bem as _bem
from openmdao.api import ExplicitComponent
from scipy.interpolate import PchipInterpolator
from wisdem.ccblade.ccblade import CCBlade, CCAirfoil
from wisdem.commonse.csystem import DirectionVector

cosd = lambda x: np.cos(np.deg2rad(x))
sind = lambda x: np.sin(np.deg2rad(x))


class CCBladeGeometry(ExplicitComponent):
    """
    Compute some geometric properties of the turbine based on the tip radius,
    precurve, presweep, and precone.

    Parameters
    ----------
    Rtip : float
        Rotor tip radius.
    precurve_in : numpy array[n_span]
        Prebend distribution along the span.
    presweep_in : numpy array[n_span]
        Presweep distribution along the span.
    precone : float
        Precone angle.

    Returns
    -------
    R : float
        Rotor radius.
    diameter : float
        Rotor diameter.
    precurveTip : float
        Precurve value at the rotor tip.
    presweepTip : float
        Presweep value at the rotor tip.
    """

    def initialize(self):
        self.options.declare("n_span")

    def setup(self):
        n_span = self.options["n_span"]

        self.add_input("Rtip", val=0.0, units="m")
        self.add_input("precurve_in", val=np.zeros(n_span), units="m")
        self.add_input("presweep_in", val=np.zeros(n_span), units="m")
        self.add_input("precone", val=0.0, units="deg")

        self.add_output("R", val=0.0, units="m")
        self.add_output("diameter", val=0.0, units="m")
        self.add_output("precurveTip", val=0.0, units="m")
        self.add_output("presweepTip", val=0.0, units="m")

        self.declare_partials("R", ["Rtip", "precone"])
        self.declare_partials("diameter", ["Rtip", "precone"])

        self.declare_partials(["R", "diameter"], "precurve_in", rows=[0], cols=[n_span - 1])

        self.declare_partials("precurveTip", "precurve_in", val=1.0, rows=[0], cols=[n_span - 1])
        self.declare_partials("presweepTip", "presweep_in", val=1.0, rows=[0], cols=[n_span - 1])

    def compute(self, inputs, outputs):
        Rtip = inputs["Rtip"]
        precone = inputs["precone"]

        outputs["precurveTip"] = inputs["precurve_in"][-1]
        outputs["presweepTip"] = inputs["presweep_in"][-1]

        outputs["R"] = Rtip * cosd(precone) + outputs["precurveTip"] * sind(precone)
        outputs["diameter"] = outputs["R"] * 2

    def compute_partials(self, inputs, J):
        Rtip = inputs["Rtip"]
        precone = inputs["precone"]
        precurveTip = inputs["precurve_in"][-1]

        J["R", "precurve_in"] = sind(precone)
        J["R", "Rtip"] = cosd(precone)
        J["R", "precone"] = (-Rtip * sind(precone) + precurveTip * cosd(precone)) * np.pi / 180.0

        J["diameter", "precurve_in"] = 2.0 * J["R", "precurve_in"]
        J["diameter", "Rtip"] = 2.0 * J["R", "Rtip"]
        J["diameter", "precone"] = 2.0 * J["R", "precone"]


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
        Tip radius.
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
    airfoils_cl : numpy array[n_span, n_aoa, n_Re, n_tab]
        Lift coefficients, spanwise.
    airfoils_cd : numpy array[n_span, n_aoa, n_Re, n_tab]
        Drag coefficients, spanwise.
    airfoils_cm : numpy array[n_span, n_aoa, n_Re, n_tab]
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
        self.n_tab = n_tab = rotorse_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1

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
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
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
        Rhub = inputs["Rhub"]
        Rtip = inputs["Rtip"]
        hub_height = inputs["hub_height"]
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        yaw = inputs["yaw"]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        shearExp = inputs["shearExp"]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"]
        Omega_load = inputs["Omega_load"]
        pitch_load = inputs["pitch_load"]
        azimuth_load = inputs["azimuth_load"]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :, 0],
                inputs["airfoils_cd"][i, :, :, 0],
                inputs["airfoils_cm"][i, :, :, 0],
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
        self.n_tab = n_tab = modeling_options["WISDEM"]["RotorSE"][
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1
        n_opt_chord = opt_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"]
        n_opt_twist = opt_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"]

        # Inputs
        self.add_input("Uhub", val=9.0, units="m/s", desc="Undisturbed wind speed")

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
            "s_opt_twist",
            val=np.zeros(n_opt_twist),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade twist",
        )
        self.add_input("chord", val=np.zeros(n_span), units="m", desc="chord length at each section")
        self.add_input(
            "twist",
            val=np.zeros(n_span),
            units="rad",
            desc="twist angle at each section (positive decreases angle of attack)",
        )
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg", desc="angle of attack grid for polars")
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="lift coefficients, spanwise")
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="drag coefficients, spanwise")
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)), desc="moment coefficients, spanwise")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)), desc="Reynolds numbers of polars")
        self.add_input("Rhub", val=0.0, units="m", desc="hub radius")
        self.add_input("Rtip", val=0.0, units="m", desc="tip radius")
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

        if self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["inverse"]:
            if self.options["opt_options"]["design_variables"]["blade"]["aero_shape"]["twist"]["flag"]:
                raise Exception(
                    "Twist cannot be simultaneously optimized and set to be defined inverting the BEM equations. Please check your analysis options yaml."
                )
            # Find cl and cd for max efficiency
            cl = np.zeros(self.n_span)
            cd = np.zeros(self.n_span)
            alpha = np.zeros(self.n_span)
            Eff = np.zeros(self.n_span)

            Omega = inputs["tsr"] * inputs["Uhub"] / inputs["r"][-1]

            margin2stall = self.options["opt_options"]["constraints"]["blade"]["stall"]["margin"] * 180.0 / np.pi
            Re = np.array(Omega * inputs["r"] * inputs["chord"] * inputs["rho"] / inputs["mu"])
            for i in range(self.n_span):
                af[i].eval_unsteady(
                    inputs["airfoils_aoa"],
                    inputs["airfoils_cl"][i, :, 0, 0],
                    inputs["airfoils_cd"][i, :, 0, 0],
                    inputs["airfoils_cm"][i, :, 0, 0],
                )
                alpha[i] = (af[i].unsteady["alpha1"] - margin2stall) / 180.0 * np.pi
                cl[i], cd[i] = af[i].evaluate(alpha[i], Re[i])
            Eff = cl / cd

            # overwrite aoa of high thickness airfoils at root
            idx_min = [i for i, thk in enumerate(inputs["rthick"]) if thk < 95.0][0]
            alpha[0:idx_min] = alpha[idx_min]

            eta = inputs["r"] / inputs["r"][-1]
            n_points = 30
            r_interp_alpha = np.linspace(eta[0], eta[-1], n_points)
            # r_interp_alpha   = np.array([prob['eta'][0],0.2,0.45, 0.6, prob['eta'][-1]])
            alpha_control_p = np.interp(r_interp_alpha, eta, alpha)
            alpha_spline = PchipInterpolator(r_interp_alpha, alpha_control_p)
            alphafit = alpha_spline(eta)

            # find cl/cd for smooth alpha
            for i, (aoa, afi) in enumerate(zip(alphafit, af)):
                cl[i], cd[i] = afi.evaluate(aoa, Re[i])
                Eff[i] = cl[i] / cd[i]

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.plot(inputs['r'], alpha*180./np.pi, 'k')
            # plt.plot(inputs['r'], alphafit*180./np.pi, 'r')
            # plt.xlabel('blade fraction')
            # plt.ylabel('aoa (deg)')
            # plt.legend(loc='upper left')
            # plt.figure()
            # plt.plot(inputs['r'], cl, 'k')
            # plt.plot(inputs['r'], cd, 'r')
            # plt.xlabel('blade fraction')
            # plt.ylabel('cl and cd (-)')
            # plt.legend(loc='upper left')
            # plt.figure()
            # plt.plot(inputs['r'], Eff, 'k')
            # plt.xlabel('blade fraction')
            # plt.ylabel('Eff (-)')
            # plt.legend(loc='upper left')
            # plt.show()

            get_twist = CCBlade(
                inputs["r"],
                inputs["chord"],
                np.zeros_like(inputs["chord"]),
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

            get_twist.inverse_analysis = True
            get_twist.alpha = alphafit
            get_twist.cl = cl
            get_twist.cd = cd

            # Compute omega given TSR
            Omega = inputs["Uhub"] * inputs["tsr"] / inputs["Rtip"] * 30.0 / np.pi

            _, _ = get_twist.evaluate([inputs["Uhub"]], [Omega], [inputs["pitch"]], coefficients=False)

            # Cap twist root region
            for i in range(len(get_twist.theta)):
                if get_twist.theta[-i - 1] > 20.0 / 180.0 * np.pi:
                    get_twist.theta[0 : len(get_twist.theta) - i] = 20.0 / 180.0 * np.pi
                    break

            twist = get_twist.theta
        else:
            twist = inputs["twist"]

        get_cp_cm = CCBlade(
            inputs["r"],
            inputs["chord"],
            twist * 180.0 / np.pi,
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
        get_cp_cm.inverse_analysis = False
        get_cp_cm.induction = True
        # get_cp_cm.alpha            = alpha
        # get_cp_cm.cl               = cl
        # get_cp_cm.cd               = cd

        # Compute omega given TSR
        Omega = inputs["Uhub"] * inputs["tsr"] / inputs["Rtip"] * 30.0 / np.pi

        myout, derivs = get_cp_cm.evaluate([inputs["Uhub"]], [Omega], [inputs["pitch"]], coefficients=True)
        CP, CT, CY, CZ, CQ, CM, CMb = [myout[key] for key in ["CP", "CT", "CY", "CZ", "CQ", "CMy", "CMb"]]

        # if self.options['opt_options']['design_variables']['blade']['aero_shape']['twist']['flag']:
        get_cp_cm.induction = False
        get_cp_cm.induction_inflow = True
        loads, deriv = get_cp_cm.distributedAeroLoads(inputs["Uhub"][0], Omega[0], inputs["pitch"][0], 0.0)
        # get_cp_cm.induction_inflow = False
        # Np, Tp = get_cp_cm.distributedAeroLoads(inputs['Uhub'][0], Omega[0], inputs['pitch'][0], 0.0)

        # Return twist angle
        outputs["theta"] = twist
        outputs["CP"] = CP[0]
        outputs["CM"] = CMb[0]
        outputs["a"] = loads["a"]
        outputs["ap"] = loads["ap"]
        outputs["alpha"] = loads["alpha"]
        outputs["cl"] = loads["Cl"]
        outputs["cd"] = loads["Cd"]
        s = (inputs["r"] - inputs["r"][0]) / (inputs["r"][-1] - inputs["r"][0])
        outputs["cl_n_opt"] = np.interp(inputs["s_opt_twist"], s, loads["Cl"])
        outputs["cd_n_opt"] = np.interp(inputs["s_opt_twist"], s, loads["Cd"])
        # Forces in the blade coordinate system, pag 21 of https://www.nrel.gov/docs/fy13osti/58819.pdf
        outputs["Px_b"] = loads["Np"]
        outputs["Py_b"] = -loads["Tp"]
        outputs["Pz_b"] = 0 * loads["Np"]
        # Forces in the airfoil coordinate system, pag 21 of https://www.nrel.gov/docs/fy13osti/58819.pdf
        P_b = DirectionVector(loads["Np"], -loads["Tp"], 0)
        P_af = P_b.bladeToAirfoil(twist * 180.0 / np.pi)
        outputs["Px_af"] = P_af.x
        outputs["Py_af"] = P_af.y
        outputs["Pz_af"] = P_af.z
        # Lift and drag forces
        F = P_b.bladeToAirfoil(twist * 180.0 / np.pi + loads["alpha"] + inputs["pitch"])
        outputs["LiftF"] = F.x
        outputs["DragF"] = F.y
        outputs["L_n_opt"] = np.interp(inputs["s_opt_twist"], s, F.x)
        outputs["D_n_opt"] = np.interp(inputs["s_opt_twist"], s, F.y)
        # print(CP[0])


class AeroHubLoads(ExplicitComponent):
    """
    Estimate the aerodynamic loading at hub center 
    by running three instances of CCBlade at azimuth 0/120/240 degs


    Parameters
    ----------
    V_load : float
        Hub height wind speed.
    Omega_load : float
        Rotor rotation speed.
    pitch_load : float
        Blade pitch setting.
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
        Tip radius.
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
    presweep : numpy array[n_span]
        Presweep at each section.
    presweepTip : float
        Presweep at tip.
    airfoils_cl : numpy array[n_span, n_aoa, n_Re, n_tab]
        Lift coefficients, spanwise.
    airfoils_cd : numpy array[n_span, n_aoa, n_Re, n_tab]
        Drag coefficients, spanwise.
    airfoils_cm : numpy array[n_span, n_aoa, n_Re, n_tab]
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
    tiploss : boolean
        Include Prandtl tip loss model.
    hubloss : boolean
        Include Prandtl hub loss model.
    wakerotation : boolean
        Iclude effect of wake rotation (i.e., tangential induction factor is nonzero).
    usecd : boolean
        Use drag coefficient in computing induction factors.

    Returns
    -------
    Fxyz_hub_aero : numpy array [6]
        Aerodynamic forces at hub center in the hub aligned coordinate system
    Mxyz_hub_aero : numpy array [6]
        Aerodynamic moments at hub center in the hub aligned coordinate system
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        n_blades = self.options["modeling_options"]["assembly"]["number_of_blades"]

        self.n_span = n_span = rotorse_options["n_span"]
        self.n_aoa = n_aoa = rotorse_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_options["n_Re"]  # Number of Reynolds
        self.n_tab = n_tab = rotorse_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1

        # inputs
        self.add_input("V_load", val=0.0, units="m/s")
        self.add_input("Omega_load", val=0.0, units="rpm")
        self.add_input("pitch_load", val=0.0, units="deg")

        # (potential) variables
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
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_aoa", val=np.zeros((n_aoa)), units="deg")
        self.add_input("airfoils_Re", val=np.zeros((n_Re)))

        self.add_discrete_input("nBlades", val=0)
        self.add_input("rho", val=0.0, units="kg/m**3")
        self.add_input("mu", val=0.0, units="kg/(m*s)")
        self.add_input("shearExp", val=0.0)
        self.add_discrete_input("tiploss", val=True)
        self.add_discrete_input("hubloss", val=True)
        self.add_discrete_input("wakerotation", val=True)
        self.add_discrete_input("usecd", val=True)

        # outputs
        self.add_output("Fxyz_hub_aero", val=np.zeros(3), units="N")
        self.add_output("Mxyz_hub_aero", val=np.zeros(3), units="N*m")

        # Just finite difference over the relevant derivatives for now
        self.declare_partials(
            ["Fxyz_hub_aero", "Mxyz_hub_aero"],
            [
                "Omega_load",
                "Rhub",
                "Rtip",
                "V_load",
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
                "Omega_load",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"]
        Rtip = inputs["Rtip"]
        hub_height = inputs["hub_height"]
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        yaw = inputs["yaw"]
        precurve = inputs["precurve"]
        presweep = inputs["presweep"]
        precurveTip = inputs["precurveTip"]
        presweepTip = inputs["presweepTip"]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        shearExp = inputs["shearExp"]
        nSector = 1
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"]
        Omega_load = inputs["Omega_load"]
        pitch_load = inputs["pitch_load"]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :, 0],
                inputs["airfoils_cd"][i, :, :, 0],
                inputs["airfoils_cm"][i, :, :, 0],
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
            derivatives=False,
        )

        azimuth_blades = np.linspace(0, 360, B + 1)

        # distributed loads
        args = (
                r,
                precurve,
                presweep,
                np.deg2rad(precone),
                Rhub,
                Rtip,
                precurveTip,
                presweepTip,
            )
            
        F_blade_hub_cs = np.zeros((B,3))
        M_blade_hub_cs = np.zeros((B,3))
        
        for i_blade in range(B):

            # Get normal and tangential loads along the blade
            loads, _ = ccblade.distributedAeroLoads(V_load, Omega_load, pitch_load, azimuth_blades[i_blade])
            Np = loads["Np"]
            Tp = loads["Tp"]

            # Integrate blade loads along span
            Tsub, Ysub, Zsub, Qsub, Msub = _bem.thrusttorque(Np, Tp, *args)

            # Rotate forces and moments from azimuth c.s. to hub c.s. 
            myF = DirectionVector.fromArray([Tsub, Ysub, Zsub]).azimuthToHub(
                azimuth_blades[i_blade]
            )
            myM = DirectionVector.fromArray([Qsub, Msub, 0.]).azimuthToHub(
                azimuth_blades[i_blade]
            )

            F_blade_hub_cs[i_blade,:] = np.array([myF.x, myF.y, myF.z])
            M_blade_hub_cs[i_blade,:] = np.array([myM.x, myM.y, myM.z])

        # Vector sum of the contributions from the three blades
        outputs["Fxyz_hub_aero"] = np.sum(F_blade_hub_cs, axis=0)
        outputs["Mxyz_hub_aero"] = np.sum(M_blade_hub_cs, axis=0)

    '''
    def compute_partials(self, inputs, J, discrete_inputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"]
        Rtip = inputs["Rtip"]
        hub_height = inputs["hub_height"]
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        yaw = inputs["yaw"]
        precurve = inputs["precurve"]
        presweep = inputs["presweep"]
        precurveTip = inputs["precurveTip"]
        presweepTip = inputs["presweepTip"]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        shearExp = inputs["shearExp"]
        nSector = 1
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"]
        Omega_load = inputs["Omega_load"]
        pitch_load = inputs["pitch_load"]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :, 0],
                inputs["airfoils_cd"][i, :, :, 0],
                inputs["airfoils_cm"][i, :, :, 0],
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
            derivatives=False,
        )

        azimuth_blades = np.linspace(0, 360, B + 1)

        # distributed loads
        args = (
                r,
                precurve,
                presweep,
                np.deg2rad(precone),
                Rhub,
                Rtip,
                precurveTip,
                presweepTip,
            )
            
        F_blade_hub_cs = np.zeros((B,3))
        M_blade_hub_cs = np.zeros((B,3))
        
        for i_blade in range(B):

            # Get normal and tangential loads along the blade
            _, derivs = ccblade.distributedAeroLoads(V_load, Omega_load, pitch_load, azimuth_blades[i_blade])
            
            dNp = derivs["dNp"]
            dTp = derivs["dTp"]
            
            # We need more outputs, Tapenade on _bem.thrusttorque needed
            (dT_ds_sub, dY_ds_sub, dZ_ds_sub, dQ_ds_sub, dM_ds_sub,
             dT_dv_sub, dY_dv_sub, dZ_dv_sub, dQ_dv_sub, dM_dv_sub) = self.__thrustTorqueDeriv(
                Np, Tp, self._dNp_dX, self._dTp_dX, self._dNp_dprecurve, self._dTp_dprecurve, *args
            )

        #     # Rotate forces and moments from azimuth c.s. to hub c.s. 
        #     myF = DirectionVector.fromArray([Tsub, Ssub, Vsub]).azimuthToHub(
        #         azimuth_blades[i_blade]
        #     )
        #     myM = DirectionVector.fromArray([Qsub, Msub, 0.]).azimuthToHub(
        #         azimuth_blades[i_blade]
        #     )

        #     F_blade_hub_cs[i_blade,:] = np.array([myF.x, myF.y, myF.z])
        #     M_blade_hub_cs[i_blade,:] = np.array([myM.x, myM.y, myM.z])

        # # Vector sum of the contributions from the three blades
        # outputs["Fxyz_hub_aero"] = np.sum(F_blade_hub_cs, axis=0)
        # outputs["Mxyz_hub_aero"] = np.sum(M_blade_hub_cs, axis=0)
    '''



class CCBladeEvaluate(ExplicitComponent):
    """
    Standalone component for CCBlade that is only a light wrapper on CCBlade().

    Currently, this component is not used in any workflow, but it is a
    convenient way to test the derivatives coming out of CCBlade using OpenMDAO's
    check_partials method.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_init_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_init_options["n_span"]
        self.n_aoa = n_aoa = rotorse_init_options["n_aoa"]  # Number of angle of attacks
        self.n_Re = n_Re = rotorse_init_options["n_Re"]  # Number of Reynolds
        self.n_tab = n_tab = rotorse_init_options[
            "n_tab"
        ]  # Number of tabulated data. For distributed aerodynamic control this could be > 1

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

        # parameters
        self.add_input("airfoils_cl", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cd", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
        self.add_input("airfoils_cm", val=np.zeros((n_span, n_aoa, n_Re, n_tab)))
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
        self.add_output("P", val=0.0, units="W")
        self.add_output("T", val=0.0, units="N")
        self.add_output("Y", val=0.0, units="N")
        self.add_output("Z", val=0.0, units="N")
        self.add_output("Q", val=0.0, units="N/m")
        self.add_output("My", val=0.0, units="N/m")
        self.add_output("Mz", val=0.0, units="N/m")
        self.add_output("Mb", val=0.0, units="N/m")

        self.add_output("CP", val=0.0)
        self.add_output("CT", val=0.0)
        self.add_output("CY", val=0.0)
        self.add_output("CZ", val=0.0)
        self.add_output("CQ", val=0.0)
        self.add_output("CMy", val=0.0)
        self.add_output("CMz", val=0.0)
        self.add_output("CMb", val=0.0)

        self.declare_partials("*", "*")
        self.declare_partials("*", "airfoils*", dependent=False)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"]
        Rtip = inputs["Rtip"]
        hub_height = inputs["hub_height"]
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        yaw = inputs["yaw"]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        shearExp = inputs["shearExp"]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"]
        Omega_load = inputs["Omega_load"]
        pitch_load = inputs["pitch_load"]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :, 0],
                inputs["airfoils_cd"][i, :, :, 0],
                inputs["airfoils_cm"][i, :, :, 0],
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
            derivatives=False,
        )

        loads, _ = ccblade.evaluate(V_load, Omega_load, pitch_load, coefficients=True)
        outputs["P"] = loads["P"]
        outputs["T"] = loads["T"]
        outputs["Y"] = loads["Y"]
        outputs["Z"] = loads["Z"]
        outputs["Q"] = loads["Q"]
        outputs["My"] = loads["My"]
        outputs["Mz"] = loads["Mz"]
        outputs["Mb"] = loads["Mb"]
        outputs["CP"] = loads["CP"]
        outputs["CT"] = loads["CT"]
        outputs["CY"] = loads["CY"]
        outputs["CZ"] = loads["CZ"]
        outputs["CQ"] = loads["CQ"]
        outputs["CMy"] = loads["CMy"]
        outputs["CMz"] = loads["CMz"]
        outputs["CMb"] = loads["CMb"]

    def compute_partials(self, inputs, J, discrete_inputs):
        r = inputs["r"]
        chord = inputs["chord"]
        theta = inputs["theta"]
        Rhub = inputs["Rhub"]
        Rtip = inputs["Rtip"]
        hub_height = inputs["hub_height"]
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        yaw = inputs["yaw"]
        precurve = inputs["precurve"]
        precurveTip = inputs["precurveTip"]
        B = discrete_inputs["nBlades"]
        rho = inputs["rho"]
        mu = inputs["mu"]
        shearExp = inputs["shearExp"]
        nSector = discrete_inputs["nSector"]
        tiploss = discrete_inputs["tiploss"]
        hubloss = discrete_inputs["hubloss"]
        wakerotation = discrete_inputs["wakerotation"]
        usecd = discrete_inputs["usecd"]
        V_load = inputs["V_load"]
        Omega_load = inputs["Omega_load"]
        pitch_load = inputs["pitch_load"]

        if len(precurve) == 0:
            precurve = np.zeros_like(r)

        # airfoil files
        af = [None] * self.n_span
        for i in range(self.n_span):
            af[i] = CCAirfoil(
                inputs["airfoils_aoa"],
                inputs["airfoils_Re"],
                inputs["airfoils_cl"][i, :, :, 0],
                inputs["airfoils_cd"][i, :, :, 0],
                inputs["airfoils_cm"][i, :, :, 0],
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

        dT = derivs["dT"]
        J["T", "r"] = dT["dr"]
        J["T", "chord"] = dT["dchord"]
        J["T", "theta"] = dT["dtheta"]
        J["T", "Rhub"] = np.squeeze(dT["dRhub"])
        J["T", "Rtip"] = np.squeeze(dT["dRtip"])
        J["T", "hub_height"] = np.squeeze(dT["dhubHt"])
        J["T", "precone"] = np.squeeze(dT["dprecone"])
        J["T", "tilt"] = np.squeeze(dT["dtilt"])
        J["T", "yaw"] = np.squeeze(dT["dyaw"])
        J["T", "shearExp"] = np.squeeze(dT["dshear"])
        J["T", "V_load"] = np.squeeze(dT["dUinf"])
        J["T", "Omega_load"] = np.squeeze(dT["dOmega"])
        J["T", "pitch_load"] = np.squeeze(dT["dpitch"])
        J["T", "precurve"] = dT["dprecurve"]
        J["T", "precurveTip"] = dT["dprecurveTip"]

        dY = derivs["dY"]
        J["Y", "r"] = dY["dr"]
        J["Y", "chord"] = dY["dchord"]
        J["Y", "theta"] = dY["dtheta"]
        J["Y", "Rhub"] = np.squeeze(dY["dRhub"])
        J["Y", "Rtip"] = np.squeeze(dY["dRtip"])
        J["Y", "hub_height"] = np.squeeze(dY["dhubHt"])
        J["Y", "precone"] = np.squeeze(dY["dprecone"])
        J["Y", "tilt"] = np.squeeze(dY["dtilt"])
        J["Y", "yaw"] = np.squeeze(dY["dyaw"])
        J["Y", "shearExp"] = np.squeeze(dY["dshear"])
        J["Y", "V_load"] = np.squeeze(dY["dUinf"])
        J["Y", "Omega_load"] = np.squeeze(dY["dOmega"])
        J["Y", "pitch_load"] = np.squeeze(dY["dpitch"])
        J["Y", "precurve"] = dY["dprecurve"]
        J["Y", "precurveTip"] = dY["dprecurveTip"]

        dZ = derivs["dZ"]
        J["Z", "r"] = dZ["dr"]
        J["Z", "chord"] = dZ["dchord"]
        J["Z", "theta"] = dZ["dtheta"]
        J["Z", "Rhub"] = np.squeeze(dZ["dRhub"])
        J["Z", "Rtip"] = np.squeeze(dZ["dRtip"])
        J["Z", "hub_height"] = np.squeeze(dZ["dhubHt"])
        J["Z", "precone"] = np.squeeze(dZ["dprecone"])
        J["Z", "tilt"] = np.squeeze(dZ["dtilt"])
        J["Z", "yaw"] = np.squeeze(dZ["dyaw"])
        J["Z", "shearExp"] = np.squeeze(dZ["dshear"])
        J["Z", "V_load"] = np.squeeze(dZ["dUinf"])
        J["Z", "Omega_load"] = np.squeeze(dZ["dOmega"])
        J["Z", "pitch_load"] = np.squeeze(dZ["dpitch"])
        J["Z", "precurve"] = dZ["dprecurve"]
        J["Z", "precurveTip"] = dZ["dprecurveTip"]

        dQ = derivs["dQ"]
        J["Q", "r"] = dQ["dr"]
        J["Q", "chord"] = dQ["dchord"]
        J["Q", "theta"] = dQ["dtheta"]
        J["Q", "Rhub"] = np.squeeze(dQ["dRhub"])
        J["Q", "Rtip"] = np.squeeze(dQ["dRtip"])
        J["Q", "hub_height"] = np.squeeze(dQ["dhubHt"])
        J["Q", "precone"] = np.squeeze(dQ["dprecone"])
        J["Q", "tilt"] = np.squeeze(dQ["dtilt"])
        J["Q", "yaw"] = np.squeeze(dQ["dyaw"])
        J["Q", "shearExp"] = np.squeeze(dQ["dshear"])
        J["Q", "V_load"] = np.squeeze(dQ["dUinf"])
        J["Q", "Omega_load"] = np.squeeze(dQ["dOmega"])
        J["Q", "pitch_load"] = np.squeeze(dQ["dpitch"])
        J["Q", "precurve"] = dQ["dprecurve"]
        J["Q", "precurveTip"] = dQ["dprecurveTip"]

        dMy = derivs["dMy"]
        J["My", "r"] = dMy["dr"]
        J["My", "chord"] = dMy["dchord"]
        J["My", "theta"] = dMy["dtheta"]
        J["My", "Rhub"] = np.squeeze(dMy["dRhub"])
        J["My", "Rtip"] = np.squeeze(dMy["dRtip"])
        J["My", "hub_height"] = np.squeeze(dMy["dhubHt"])
        J["My", "precone"] = np.squeeze(dMy["dprecone"])
        J["My", "tilt"] = np.squeeze(dMy["dtilt"])
        J["My", "yaw"] = np.squeeze(dMy["dyaw"])
        J["My", "shearExp"] = np.squeeze(dMy["dshear"])
        J["My", "V_load"] = np.squeeze(dMy["dUinf"])
        J["My", "Omega_load"] = np.squeeze(dMy["dOmega"])
        J["My", "pitch_load"] = np.squeeze(dMy["dpitch"])
        J["My", "precurve"] = dMy["dprecurve"]
        J["My", "precurveTip"] = dMy["dprecurveTip"]

        dMz = derivs["dMz"]
        J["Mz", "r"] = dMz["dr"]
        J["Mz", "chord"] = dMz["dchord"]
        J["Mz", "theta"] = dMz["dtheta"]
        J["Mz", "Rhub"] = np.squeeze(dMz["dRhub"])
        J["Mz", "Rtip"] = np.squeeze(dMz["dRtip"])
        J["Mz", "hub_height"] = np.squeeze(dMz["dhubHt"])
        J["Mz", "precone"] = np.squeeze(dMz["dprecone"])
        J["Mz", "tilt"] = np.squeeze(dMz["dtilt"])
        J["Mz", "yaw"] = np.squeeze(dMz["dyaw"])
        J["Mz", "shearExp"] = np.squeeze(dMz["dshear"])
        J["Mz", "V_load"] = np.squeeze(dMz["dUinf"])
        J["Mz", "Omega_load"] = np.squeeze(dMz["dOmega"])
        J["Mz", "pitch_load"] = np.squeeze(dMz["dpitch"])
        J["Mz", "precurve"] = dMz["dprecurve"]
        J["Mz", "precurveTip"] = dMz["dprecurveTip"]

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

        dCT = derivs["dCT"]
        J["CT", "r"] = dCT["dr"]
        J["CT", "chord"] = dCT["dchord"]
        J["CT", "theta"] = dCT["dtheta"]
        J["CT", "Rhub"] = np.squeeze(dCT["dRhub"])
        J["CT", "Rtip"] = np.squeeze(dCT["dRtip"])
        J["CT", "hub_height"] = np.squeeze(dCT["dhubHt"])
        J["CT", "precone"] = np.squeeze(dCT["dprecone"])
        J["CT", "tilt"] = np.squeeze(dCT["dtilt"])
        J["CT", "yaw"] = np.squeeze(dCT["dyaw"])
        J["CT", "shearExp"] = np.squeeze(dCT["dshear"])
        J["CT", "V_load"] = np.squeeze(dCT["dUinf"])
        J["CT", "Omega_load"] = np.squeeze(dCT["dOmega"])
        J["CT", "pitch_load"] = np.squeeze(dCT["dpitch"])
        J["CT", "precurve"] = dCT["dprecurve"]
        J["CT", "precurveTip"] = dCT["dprecurveTip"]

        dCY = derivs["dCY"]
        J["CY", "r"] = dCY["dr"]
        J["CY", "chord"] = dCY["dchord"]
        J["CY", "theta"] = dCY["dtheta"]
        J["CY", "Rhub"] = np.squeeze(dCY["dRhub"])
        J["CY", "Rtip"] = np.squeeze(dCY["dRtip"])
        J["CY", "hub_height"] = np.squeeze(dCY["dhubHt"])
        J["CY", "precone"] = np.squeeze(dCY["dprecone"])
        J["CY", "tilt"] = np.squeeze(dCY["dtilt"])
        J["CY", "yaw"] = np.squeeze(dCY["dyaw"])
        J["CY", "shearExp"] = np.squeeze(dCY["dshear"])
        J["CY", "V_load"] = np.squeeze(dCY["dUinf"])
        J["CY", "Omega_load"] = np.squeeze(dCY["dOmega"])
        J["CY", "pitch_load"] = np.squeeze(dCY["dpitch"])
        J["CY", "precurve"] = dCY["dprecurve"]
        J["CY", "precurveTip"] = dCY["dprecurveTip"]

        dCZ = derivs["dCZ"]
        J["CZ", "r"] = dCZ["dr"]
        J["CZ", "chord"] = dCZ["dchord"]
        J["CZ", "theta"] = dCZ["dtheta"]
        J["CZ", "Rhub"] = np.squeeze(dCZ["dRhub"])
        J["CZ", "Rtip"] = np.squeeze(dCZ["dRtip"])
        J["CZ", "hub_height"] = np.squeeze(dCZ["dhubHt"])
        J["CZ", "precone"] = np.squeeze(dCZ["dprecone"])
        J["CZ", "tilt"] = np.squeeze(dCZ["dtilt"])
        J["CZ", "yaw"] = np.squeeze(dCZ["dyaw"])
        J["CZ", "shearExp"] = np.squeeze(dCZ["dshear"])
        J["CZ", "V_load"] = np.squeeze(dCZ["dUinf"])
        J["CZ", "Omega_load"] = np.squeeze(dCZ["dOmega"])
        J["CZ", "pitch_load"] = np.squeeze(dCZ["dpitch"])
        J["CZ", "precurve"] = dCZ["dprecurve"]
        J["CZ", "precurveTip"] = dCZ["dprecurveTip"]

        dCQ = derivs["dCQ"]
        J["CQ", "r"] = dCQ["dr"]
        J["CQ", "chord"] = dCQ["dchord"]
        J["CQ", "theta"] = dCQ["dtheta"]
        J["CQ", "Rhub"] = np.squeeze(dCQ["dRhub"])
        J["CQ", "Rtip"] = np.squeeze(dCQ["dRtip"])
        J["CQ", "hub_height"] = np.squeeze(dCQ["dhubHt"])
        J["CQ", "precone"] = np.squeeze(dCQ["dprecone"])
        J["CQ", "tilt"] = np.squeeze(dCQ["dtilt"])
        J["CQ", "yaw"] = np.squeeze(dCQ["dyaw"])
        J["CQ", "shearExp"] = np.squeeze(dCQ["dshear"])
        J["CQ", "V_load"] = np.squeeze(dCQ["dUinf"])
        J["CQ", "Omega_load"] = np.squeeze(dCQ["dOmega"])
        J["CQ", "pitch_load"] = np.squeeze(dCQ["dpitch"])
        J["CQ", "precurve"] = dCQ["dprecurve"]
        J["CQ", "precurveTip"] = dCQ["dprecurveTip"]

        dCMy = derivs["dCMy"]
        J["CMy", "r"] = dCMy["dr"]
        J["CMy", "chord"] = dCMy["dchord"]
        J["CMy", "theta"] = dCMy["dtheta"]
        J["CMy", "Rhub"] = np.squeeze(dCMy["dRhub"])
        J["CMy", "Rtip"] = np.squeeze(dCMy["dRtip"])
        J["CMy", "hub_height"] = np.squeeze(dCMy["dhubHt"])
        J["CMy", "precone"] = np.squeeze(dCMy["dprecone"])
        J["CMy", "tilt"] = np.squeeze(dCMy["dtilt"])
        J["CMy", "yaw"] = np.squeeze(dCMy["dyaw"])
        J["CMy", "shearExp"] = np.squeeze(dCMy["dshear"])
        J["CMy", "V_load"] = np.squeeze(dCMy["dUinf"])
        J["CMy", "Omega_load"] = np.squeeze(dCMy["dOmega"])
        J["CMy", "pitch_load"] = np.squeeze(dCMy["dpitch"])
        J["CMy", "precurve"] = dCMy["dprecurve"]
        J["CMy", "precurveTip"] = dCMy["dprecurveTip"]

        dCMz = derivs["dCMz"]
        J["CMz", "r"] = dCMz["dr"]
        J["CMz", "chord"] = dCMz["dchord"]
        J["CMz", "theta"] = dCMz["dtheta"]
        J["CMz", "Rhub"] = np.squeeze(dCMz["dRhub"])
        J["CMz", "Rtip"] = np.squeeze(dCMz["dRtip"])
        J["CMz", "hub_height"] = np.squeeze(dCMz["dhubHt"])
        J["CMz", "precone"] = np.squeeze(dCMz["dprecone"])
        J["CMz", "tilt"] = np.squeeze(dCMz["dtilt"])
        J["CMz", "yaw"] = np.squeeze(dCMz["dyaw"])
        J["CMz", "shearExp"] = np.squeeze(dCMz["dshear"])
        J["CMz", "V_load"] = np.squeeze(dCMz["dUinf"])
        J["CMz", "Omega_load"] = np.squeeze(dCMz["dOmega"])
        J["CMz", "pitch_load"] = np.squeeze(dCMz["dpitch"])
        J["CMz", "precurve"] = dCMz["dprecurve"]
        J["CMz", "precurveTip"] = dCMz["dprecurveTip"]

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
