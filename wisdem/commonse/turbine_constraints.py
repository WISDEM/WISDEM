import numpy as np
import openmdao.api as om

from wisdem.commonse.csystem import DirectionVector
import wisdem.commonse.utilities as util




class TowerModes(om.ExplicitComponent):
    """
    Compute tower frequency constraints

    Parameters
    ----------
    tower_freq : numpy array[2], [Hz]
        First natural frequencies of tower (and substructure)
    rotor_omega : float, [rpm]
        rated rotor rotation speed
    gamma_freq : float
        partial safety factor for fatigue
    blade_number : int
        number of rotor blades

    Returns
    -------
    frequencyNP_margin : numpy array[2]
        constraint on tower/structure frequency to blade passing frequency with margin
    frequency1P_margin : numpy array[2]
        constraint on tower/structure frequency to rotor frequency with margin

    """

    def initialize(self):
        self.options.declare("gamma", default=1.1)

    def setup(self):
        self.add_input("rated_Omega", val=0.0, units="rpm", desc="rotor rotation speed at rated")
        self.add_input("tower_freq", val=0.0, units="Hz")  # np.zeros(NFREQ),
        self.add_discrete_input("blade_number", 3)

        self.add_output(
            "constr_tower_f_NPmargin",
            val=0.0,  # np.zeros(NFREQ),
            desc="constraint on tower frequency such that ratio of 3P/f is above or below gamma with constraint <= 0",
        )
        self.add_output(
            "constr_tower_f_1Pmargin",
            val=0.0,  # np.zeros(NFREQ),
            desc="constraint on tower frequency such that ratio of 1P/f is above or below gamma with constraint <= 0",
        )

        # self.declare_partials("*", "*", method="fd", form="central", step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        freq_struct = inputs["tower_freq"]
        gamma = self.options["gamma"]
        oneP = inputs["rated_Omega"] / 60.0
        threeP = oneP * discrete_inputs["blade_number"]

        outputs["constr_tower_f_NPmargin"] = np.array(
            [min([threeP - (2 - gamma) * f, gamma * f - threeP]) for f in freq_struct]
        ).flatten()
        outputs["constr_tower_f_1Pmargin"] = np.array(
            [min([oneP - (2 - gamma) * f, gamma * f - oneP]) for f in freq_struct]
        ).flatten()


class TipDeflectionConstraint(om.ExplicitComponent):
    """
    Compute the undeflected tip-tower clearance and the ratio between the two
    including a safety factor (typically equal to 1.3)

    Parameters
    ----------
    rotor_orientation : string
        Rotor orientation, either upwind or downwind.
    tip_deflection : float, [m]
        Blade tip deflection in yaw x-direction
    Rtip : float, [m]
        Blade tip location in z_b
    ref_axis_blade : numpy array[n_span, 3], [m]
        2D array of the coordinates (x,y,z) of the blade reference axis, defined along
        blade span. The coordinate system is the one of BeamDyn: it is placed at blade
        root with x pointing the suction side of the blade, y pointing the trailing edge
        and z along the blade span. A standard configuration will have negative x values
        (prebend), if swept positive y values, and positive z values.
    precone : float, [deg]
        Rotor precone angle
    tilt : float, [deg]
        Nacelle uptilt angle
    overhang : float, [m]
        Horizontal distance between hub and tower-top axis
    ref_axis_tower : numpy array[n_height_tow, 3], [m]
        2D array of the coordinates (x,y,z) of the tower reference axis. The coordinate
        system is the global coordinate system of OpenFAST: it is placed at tower base
        with x pointing downwind, y pointing on the side and z pointing vertically
        upwards. A standard tower configuration will have zero x and y values and
        positive z values.
    outer_diameter_full : numpy array[n_height_tow], [m]
        Diameter of tower at fine-section nodes
    max_allowable_td_ratio : float
        Safety factor of the tip deflection to stay within the tower clearance

    Returns
    -------
    tip_deflection_ratio : float
        Ratio of blade tip deflection towards the tower and clearance between
        undeflected blade tip and tower
    blade_tip_tower_clearance : float, [m]
        Clearance between undeflected blade tip and tower in x-direction of yaw c.s.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        n_span = modeling_options["WISDEM"]["RotorSE"]["n_span"]
        n_height_tow = modeling_options["WISDEM"]["TowerSE"]["n_height_tower"]

        self.add_discrete_input("rotor_orientation", val="upwind")
        self.add_input("tip_deflection", val=0.0, units="m")
        self.add_input("Rtip", val=0.0, units="m")
        self.add_input("ref_axis_blade", val=np.zeros((n_span, 3)), units="m")
        self.add_input("precone", val=0.0, units="deg")
        self.add_input("tilt", val=0.0, units="deg")
        self.add_input("overhang", val=0.0, units="m")
        self.add_input("ref_axis_tower", val=np.zeros((n_height_tow, 3)), units="m")
        self.add_input("outer_diameter_full", val=np.zeros(n_height_tow), units="m")
        self.add_input("max_allowable_td_ratio", val=1.35 * 1.05)

        self.add_output("tip_deflection_ratio", val=0.0)
        self.add_output("blade_tip_tower_clearance", val=0.0, units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack variables
        z_tower = inputs["ref_axis_tower"][:, 2]
        d_tower = inputs["outer_diameter_full"]
        overhang = inputs["overhang"]
        tt2hub = overhang * np.sin(inputs["tilt"] / 180.0 * np.pi)
        precone = inputs["precone"]
        tilt = inputs["tilt"]
        delta = inputs["tip_deflection"]
        prebend_tip = inputs["ref_axis_blade"][-1, 0]  # Defined negative for a standard upwind blade
        presweep_tip = inputs["ref_axis_blade"][-1, 1]  # Defined positive for a standard blade
        # Coordinates of blade tip in yaw c.s.
        if discrete_inputs["rotor_orientation"] == "upwind":
            blade_yaw = (
                DirectionVector(prebend_tip, presweep_tip, inputs["Rtip"])
                .bladeToAzimuth(precone)
                .azimuthToHub(180.0)
                .hubToYaw(tilt)
            )
        else:
            blade_yaw = (
                DirectionVector(prebend_tip, presweep_tip, inputs["Rtip"])
                .bladeToAzimuth(-precone)
                .azimuthToHub(180.0)
                .hubToYaw(-tilt)
            )

        # Find the radius of tower where blade passes
        z_interp = z_tower[-1] + tt2hub + blade_yaw.z

        if np.mean(d_tower) == 0.0:
            print(
                "Warning: turbine_constraints.py : TipDeflectionConstraint.compute : No tower data for blade tip tower clearnace calculation.  Assuming 0m for tower radius, tip clearance estimates will be too conservative."
            )
            r_interp = 0.0
        else:
            d_interp, ddinterp_dzinterp, ddinterp_dtowerz, ddinterp_dtowerd = util.interp_with_deriv(
                z_interp, z_tower, d_tower
            )
            r_interp = 0.5 * d_interp
            drinterp_dzinterp = 0.5 * ddinterp_dzinterp
            drinterp_dtowerz = 0.5 * ddinterp_dtowerz
            drinterp_dtowerd = 0.5 * ddinterp_dtowerd

        # Max deflection before strike
        if discrete_inputs["rotor_orientation"] == "upwind":
            parked_margin = overhang - blade_yaw.x - r_interp
        else:
            parked_margin = overhang + blade_yaw.x - r_interp
        outputs["blade_tip_tower_clearance"] = parked_margin
        outputs["tip_deflection_ratio"] = delta * inputs["max_allowable_td_ratio"] / parked_margin


class TurbineConstraints(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem(
            "modes", TowerModes(gamma=modeling_options["WISDEM"]["TowerSE"]["gamma_freq"]), promotes=["*"]
        )
        self.add_subsystem("tipd", TipDeflectionConstraint(modeling_options=modeling_options), promotes=["*"])
