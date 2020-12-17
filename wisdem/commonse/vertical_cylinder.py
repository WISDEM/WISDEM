import numpy as np
import openmdao.api as om
import wisdem.commonse.frustum as frustum
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.manufacturing as manufacture
from wisdem.commonse import NFREQ, eps, gravity
from wisdem.commonse.utilization_constraints import hoopStress, hoopStressEurocode

RIGID = 1e30
NREFINE = 3


def get_nfull(npts):
    nFull = int(1 + NREFINE * (npts - 1))
    return nFull


def get_npts(nFull):
    npts = int(1 + (nFull - 1) / NREFINE)
    return npts


# -----------------
#  Components
# -----------------

# TODO need to check the length of each array
class CylinderDiscretization(om.ExplicitComponent):
    """
    Discretize geometry into finite element nodes

    Parameters
    ----------
    foundation_height : float, [m]
        starting height of tower
    section_height : numpy array[nPoints-1], [m]
        parameterized section heights along cylinder
    diameter : numpy array[nPoints], [m]
        cylinder diameter at corresponding locations
    wall_thickness : numpy array[nPoints-1], [m]
        shell thickness at corresponding locations

    Returns
    -------
    z_param : numpy array[nPoints], [m]
        parameterized locations along cylinder, linear lofting between
    z_full : numpy array[nFull], [m]
        locations along cylinder
    d_full : numpy array[nFull], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[nFull-1], [m]
        shell thickness at corresponding locations

    """

    """discretize geometry into finite element nodes"""

    def initialize(self):
        self.options.declare("nPoints")
        self.options.declare("nRefine", default=NREFINE)
        self.options.declare("nPin", default=0)

    def setup(self):
        nPoints = self.options["nPoints"]
        nRefine = np.round(self.options["nRefine"])
        nFull = int(nRefine * (nPoints - 1) + 1)

        self.add_input("foundation_height", val=0.0, units="m")
        self.add_input("section_height", np.zeros(nPoints - 1), units="m")
        self.add_input("diameter", np.zeros(nPoints), units="m")
        self.add_input("wall_thickness", np.zeros(nPoints - 1), units="m")

        self.add_output("z_param", np.zeros(nPoints), units="m")
        self.add_output("z_full", np.zeros(nFull), units="m")
        self.add_output("d_full", np.zeros(nFull), units="m")
        self.add_output("t_full", np.zeros(nFull - 1), units="m")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        # Check to make sure we have good values
        if np.any(inputs["section_height"] <= 0.0):
            raise ValueError("Section height values must be greater than zero, " + str(inputs["section_height"]))
        if np.any(inputs["wall_thickness"] <= 0.0):
            raise ValueError("Wall thickness values must be greater than zero, " + str(inputs["wall_thickness"]))
        if np.any(inputs["diameter"] <= 0.0):
            raise ValueError("Diameter values must be greater than zero, " + str(inputs["diameter"]))

        nRefine = int(np.round(self.options["nRefine"]))
        z_param = float(inputs["foundation_height"]) + np.r_[0.0, np.cumsum(inputs["section_height"].flatten())]

        # Have to regine each element one at a time so that we preserve input nodes
        z_full = np.array([])
        for k in range(z_param.size - 1):
            zref = np.linspace(z_param[k], z_param[k + 1], nRefine + 1)
            z_full = np.append(z_full, zref)
        z_full = np.unique(z_full)

        outputs["z_full"] = z_full
        outputs["d_full"] = np.interp(z_full, z_param, inputs["diameter"])
        z_section = 0.5 * (z_full[:-1] + z_full[1:])
        outputs["t_full"] = util.sectionalInterp(z_section, z_param, inputs["wall_thickness"])
        outputs["z_param"] = z_param


class CylinderMass(om.ExplicitComponent):
    """
    Compute cylinder cost and mass properties

    Parameters
    ----------
    d_full : numpy array[nPoints], [m]
        cylinder diameter at corresponding locations
    t_full : numpy array[nPoints-1], [m]
        shell thickness at corresponding locations
    z_full : numpy array[nPoints], [m]
        parameterized locations along cylinder, linear lofting between
    rho : numpy array[nPoints-1], [kg/m**3]
        material density
    outfitting_factor : numpy array[nPoints-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    material_cost_rate : numpy array[nPoints-1], [USD/kg]
        Raw material cost rate: steel $1.1/kg, aluminum $3.5/kg
    labor_cost_rate : float, [USD/min]
        Labor cost rate
    painting_cost_rate : float, [USD/m/m]
        Painting / surface finishing cost rate

    Returns
    -------
    cost : float, [USD]
        Total cylinder cost
    mass : numpy array[nPoints-1], [kg]
        Total cylinder mass
    center_of_mass : float, [m]
        z-position of center of mass of cylinder
    section_center_of_mass : numpy array[nPoints-1], [m]
        z position of center of mass of each can in the cylinder
    I_base : numpy array[6], [kg*m**2]
        mass moment of inertia of cylinder about base [xx yy zz xy xz yz]

    """

    def initialize(self):
        self.options.declare("nPoints")

    def setup(self):
        nPoints = self.options["nPoints"]

        self.add_input("d_full", val=np.zeros(nPoints), units="m")
        self.add_input("t_full", val=np.zeros(nPoints - 1), units="m")
        self.add_input("z_full", val=np.zeros(nPoints), units="m")
        self.add_input("rho", val=np.zeros(nPoints - 1), units="kg/m**3")
        self.add_input("outfitting_factor", val=np.zeros(nPoints - 1))
        self.add_input("material_cost_rate", val=np.zeros(nPoints - 1), units="USD/kg")
        self.add_input("labor_cost_rate", 0.0, units="USD/min")
        self.add_input("painting_cost_rate", 0.0, units="USD/m/m")

        self.add_output("cost", val=0.0, units="USD")
        self.add_output("mass", val=np.zeros(nPoints - 1), units="kg")
        self.add_output("center_of_mass", val=0.0, units="m")
        self.add_output("section_center_of_mass", val=np.zeros(nPoints - 1), units="m")
        self.add_output("I_base", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack variables for thickness and average radius at each can interface
        twall = inputs["t_full"]
        Rb = 0.5 * inputs["d_full"][:-1]
        Rt = 0.5 * inputs["d_full"][1:]
        zz = inputs["z_full"]
        H = np.diff(zz)
        rho = inputs["rho"]
        coeff = inputs["outfitting_factor"]
        coeff = coeff + np.where(coeff < 1.0, 1.0, 0.0)

        # Total mass of cylinder
        V_shell = frustum.frustumShellVol(Rb, Rt, twall, H)
        mass = outputs["mass"] = coeff * rho * V_shell

        # Center of mass of each can/section
        cm_section = zz[:-1] + frustum.frustumShellCG(Rb, Rt, twall, H)
        outputs["section_center_of_mass"] = cm_section

        # Center of mass of cylinder
        V_shell += eps
        outputs["center_of_mass"] = np.dot(V_shell, cm_section) / V_shell.sum()

        # Moments of inertia
        Izz_section = coeff * rho * frustum.frustumShellIzz(Rb, Rt, twall, H)
        Ixx_section = Iyy_section = coeff * rho * frustum.frustumShellIxx(Rb, Rt, twall, H)

        # Sum up each cylinder section using parallel axis theorem
        I_base = np.zeros((3, 3))
        for k in range(Izz_section.size):
            R = np.array([0.0, 0.0, cm_section[k] - zz[0]])
            Icg = util.assembleI([Ixx_section[k], Iyy_section[k], Izz_section[k], 0.0, 0.0, 0.0])

            I_base += Icg + mass[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        outputs["I_base"] = util.unassembleI(I_base)

        # Compute costs based on "Optimum Design of Steel Structures" by Farkas and Jarmai
        # All dimensions for correlations based on mm, not meters.
        R_ave = 0.5 * (Rb + Rt)
        taper = np.minimum(Rb / Rt, Rt / Rb)
        nsec = twall.size
        mshell = rho * V_shell
        mshell_tot = np.sum(rho * V_shell)
        k_m = inputs["material_cost_rate"]  # 1.1 # USD / kg carbon steel plate
        k_f = inputs["labor_cost_rate"]  # 1.0 # USD / min labor
        k_p = inputs["painting_cost_rate"]  # USD / m^2 painting
        k_e = 0.064  # Industrial electricity rate $/kWh https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
        e_f = 15.9  # Electricity usage kWh/kg for steel
        e_fo = 26.9  # Electricity usage kWh/kg for stainless steel

        # Cost Step 1) Cutting flat plates for taper using plasma cutter
        cutLengths = 2.0 * np.sqrt((Rt - Rb) ** 2.0 + H ** 2.0)  # Factor of 2 for both sides
        # Cost Step 2) Rolling plates
        # Cost Step 3) Welding rolled plates into shells (set difficulty factor based on tapering with logistic function)
        theta_F = 4.0 - 3.0 / (1 + np.exp(-5.0 * (taper - 0.75)))
        # Cost Step 4) Circumferential welds to join cans together
        theta_A = 2.0

        # Labor-based expenses
        K_f = k_f * (
            manufacture.steel_cutting_plasma_time(cutLengths, twall)
            + manufacture.steel_rolling_time(theta_F, R_ave, twall)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, cutLengths, twall)
            + manufacture.steel_butt_welding_time(theta_A, nsec, mshell_tot, 2 * np.pi * Rb[1:], twall[1:])
        )

        # Cost step 5) Painting- outside and inside
        theta_p = 2
        K_p = k_p * theta_p * 2 * (2 * np.pi * R_ave * H).sum()

        # Cost step 6) Outfitting with electricity usage
        K_o = np.sum(1.5 * k_m * (coeff - 1.0) * mshell)

        # Material cost with waste fraction, but without outfitting,
        K_m = 1.21 * np.sum(k_m * mshell)

        # Electricity usage
        K_e = np.sum(k_e * (e_f * mshell + e_fo * (coeff - 1.0) * mshell))

        # Assemble all costs for now
        tempSum = K_m + K_e + K_o + K_p + K_f

        # Capital cost share from BLS MFP by NAICS
        K_c = 0.118 * tempSum / (1.0 - 0.118)

        outputs["cost"] = tempSum + K_c


# @implement_base(CylinderFromCSProps)
class CylinderFrame3DD(om.ExplicitComponent):
    """
    Run Frame3DD on the cylinder geometry

    Parameters
    ----------
    z : numpy array[npts], [m]
        location along cylinder. start at bottom and go to top
    Az : numpy array[npts-1], [m**2]
        cross-sectional area
    Asx : numpy array[npts-1], [m**2]
        x shear area
    Asy : numpy array[npts-1], [m**2]
        y shear area
    Jz : numpy array[npts-1], [m**4]
        polar moment of inertia
    Ixx : numpy array[npts-1], [m**4]
        area moment of inertia about x-axis
    Iyy : numpy array[npts-1], [m**4]
        area moment of inertia about y-axis
    E : numpy array[npts-1], [N/m**2]
        modulus of elasticity
    G : numpy array[npts-1], [N/m**2]
        shear modulus
    rho : numpy array[npts-1], [kg/m**3]
        material density
    d : numpy array[npts], [m]
        effective cylinder diameter for section
    t : numpy array[npts-1], [m]
        effective shell thickness for section
    kidx : numpy array[nK, dtype]
        indices of z where external stiffness reactions should be applied.
    kx : numpy array[nK], [N/m]
        spring stiffness in x-direction
    ky : numpy array[nK], [N/m]
        spring stiffness in y-direction
    kz : numpy array[nK], [N/m]
        spring stiffness in z-direction
    ktx : numpy array[nK], [N/m]
        spring stiffness in theta_x-rotation
    kty : numpy array[nK], [N/m]
        spring stiffness in theta_y-rotation
    ktz : numpy array[nK], [N/m]
        spring stiffness in theta_z-rotation
    midx : numpy array[nMass, dtype]
        indices where added mass should be applied.
    m : numpy array[nMass], [kg]
        added mass
    mIxx : numpy array[nMass], [kg*m**2]
        x mass moment of inertia about some point p
    mIyy : numpy array[nMass], [kg*m**2]
        y mass moment of inertia about some point p
    mIzz : numpy array[nMass], [kg*m**2]
        z mass moment of inertia about some point p
    mIxy : numpy array[nMass], [kg*m**2]
        xy mass moment of inertia about some point p
    mIxz : numpy array[nMass], [kg*m**2]
        xz mass moment of inertia about some point p
    mIyz : numpy array[nMass], [kg*m**2]
        yz mass moment of inertia about some point p
    mrhox : numpy array[nMass], [m]
        x-location of p relative to node
    mrhoy : numpy array[nMass], [m]
        y-location of p relative to node
    mrhoz : numpy array[nMass], [m]
        z-location of p relative to node
    plidx : numpy array[nPL, dtype]
        indices where point loads should be applied.
    Fx : numpy array[nPL], [N]
        point force in x-direction
    Fy : numpy array[nPL], [N]
        point force in y-direction
    Fz : numpy array[nPL], [N]
        point force in z-direction
    Mxx : numpy array[nPL], [N*m]
        point moment about x-axis
    Myy : numpy array[nPL], [N*m]
        point moment about y-axis
    Mzz : numpy array[nPL], [N*m]
        point moment about z-axis
    Px : numpy array[npts], [N/m]
        force per unit length in x-direction
    Py : numpy array[npts], [N/m]
        force per unit length in y-direction
    Pz : numpy array[npts], [N/m]
        force per unit length in z-direction
    qdyn : numpy array[npts], [N/m**2]
        dynamic pressure

    Returns
    -------
    mass : float, [kg]
        Structural mass computed by Frame3DD
    f1 : float, [Hz]
        First natural frequency
    f2 : float, [Hz]
        Second natural frequency
    freqs : numpy array[NFREQ], [Hz]
        Natural frequencies of the structure
    x_mode_shapes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the x-direction
    y_mode_shapes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the x-direction
    cylinder_deflection : numpy array[npts], [m]
        Deflection of cylinder nodes in yaw-aligned +x direction
    Fz_out : numpy array[npts-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    Vx_out : numpy array[npts-1], [N]
        Shear force in x-direction in cylinder structure.
    Vy_out : numpy array[npts-1], [N]
        Shear force in y-direction in cylinder structure.
    Mxx_out : numpy array[npts-1], [N*m]
        Moment about x-axis in cylinder structure.
    Myy_out : numpy array[npts-1], [N*m]
        Moment about y-axis in cylinder structure.
    Mzz_out : numpy array[npts-1], [N*m]
        Moment about z-axis in cylinder structure.
    base_F : numpy array[3], [N]
        Total force on cylinder
    base_M : numpy array[3], [N*m]
        Total moment on cylinder measured at base
    axial_stress : numpy array[npts-1], [N/m**2]
        Axial stress in cylinder structure
    shear_stress : numpy array[npts-1], [N/m**2]
        Shear stress in cylinder structure
    hoop_stress : numpy array[npts-1], [N/m**2]
        Hoop stress in cylinder structure calculated with simple method used in API
        standards
    hoop_stress_euro : numpy array[npts-1], [N/m**2]
        Hoop stress in cylinder structure calculated with Eurocode method

    """

    def initialize(self):
        self.options.declare("npts")
        self.options.declare("nK")
        self.options.declare("nMass")
        self.options.declare("nPL")
        self.options.declare("frame3dd_opt")
        self.options.declare("buckling_length")

    def setup(self):
        npts = self.options["npts"]
        nK = self.options["nK"]
        nMass = self.options["nMass"]
        nPL = self.options["nPL"]

        # cross-sectional data along cylinder.
        self.add_input("z", val=np.zeros(npts), units="m")
        self.add_input("Az", val=np.zeros(npts - 1), units="m**2")
        self.add_input("Asx", val=np.zeros(npts - 1), units="m**2")
        self.add_input("Asy", val=np.zeros(npts - 1), units="m**2")
        self.add_input("Jz", val=np.zeros(npts - 1), units="m**4")
        self.add_input("Ixx", val=np.zeros(npts - 1), units="m**4")
        self.add_input("Iyy", val=np.zeros(npts - 1), units="m**4")
        self.add_input("E", val=np.zeros(npts - 1), units="N/m**2")
        self.add_input("G", val=np.zeros(npts - 1), units="N/m**2")
        self.add_input("rho", val=np.zeros(npts - 1), units="kg/m**3")

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_input("d", val=np.zeros(npts), units="m")
        self.add_input("t", val=np.zeros(npts - 1), units="m")

        # spring reaction data.  Use global RIGID for rigid constraints.
        # JPJ: this next output should be a discrete output
        self.add_input("kidx", val=np.zeros(nK, dtype=np.int_))
        self.add_input("kx", val=np.zeros(nK), units="N/m")
        self.add_input("ky", val=np.zeros(nK), units="N/m")
        self.add_input("kz", val=np.zeros(nK), units="N/m")
        self.add_input("ktx", val=np.zeros(nK), units="N/m")
        self.add_input("kty", val=np.zeros(nK), units="N/m")
        self.add_input("ktz", val=np.zeros(nK), units="N/m")

        # extra mass
        # JPJ: this next output should be a discrete output
        self.add_input("midx", val=np.zeros(nMass, dtype=np.int_))
        self.add_input("m", val=np.zeros(nMass), units="kg")
        self.add_input("mIxx", val=np.zeros(nMass), units="kg*m**2")
        self.add_input("mIyy", val=np.zeros(nMass), units="kg*m**2")
        self.add_input("mIzz", val=np.zeros(nMass), units="kg*m**2")
        self.add_input("mIxy", val=np.zeros(nMass), units="kg*m**2")
        self.add_input("mIxz", val=np.zeros(nMass), units="kg*m**2")
        self.add_input("mIyz", val=np.zeros(nMass), units="kg*m**2")
        self.add_input("mrhox", val=np.zeros(nMass), units="m")
        self.add_input("mrhoy", val=np.zeros(nMass), units="m")
        self.add_input("mrhoz", val=np.zeros(nMass), units="m")

        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        self.add_input("plidx", val=np.zeros(nPL, dtype=np.int_))
        self.add_input("Fx", val=np.zeros(nPL), units="N")
        self.add_input("Fy", val=np.zeros(nPL), units="N")
        self.add_input("Fz", val=np.zeros(nPL), units="N")
        self.add_input("Mxx", val=np.zeros(nPL), units="N*m")
        self.add_input("Myy", val=np.zeros(nPL), units="N*m")
        self.add_input("Mzz", val=np.zeros(nPL), units="N*m")

        # combined wind-water distributed loads
        self.add_input("Px", val=np.zeros(npts), units="N/m")
        self.add_input("Py", val=np.zeros(npts), units="N/m")
        self.add_input("Pz", val=np.zeros(npts), units="N/m")
        self.add_input("qdyn", val=np.zeros(npts), units="N/m**2")

        NFREQ2 = int(NFREQ / 2)
        self.add_output("mass", val=0.0, units="kg")
        self.add_output("f1", val=0.0, units="Hz")
        self.add_output("f2", val=0.0, units="Hz")
        self.add_output("freqs", val=np.zeros(NFREQ), units="Hz")
        self.add_output("x_mode_shapes", val=np.zeros((NFREQ2, 5)))
        self.add_output("y_mode_shapes", val=np.zeros((NFREQ2, 5)))
        self.add_output("x_mode_freqs", val=np.zeros(NFREQ2))
        self.add_output("y_mode_freqs", val=np.zeros(NFREQ2))
        self.add_output("cylinder_deflection", val=np.zeros(npts), units="m")
        self.add_output("Fz_out", val=np.zeros(npts - 1), units="N")
        self.add_output("Vx_out", val=np.zeros(npts - 1), units="N")
        self.add_output("Vy_out", val=np.zeros(npts - 1), units="N")
        self.add_output("Mxx_out", val=np.zeros(npts - 1), units="N*m")
        self.add_output("Myy_out", val=np.zeros(npts - 1), units="N*m")
        self.add_output("Mzz_out", val=np.zeros(npts - 1), units="N*m")
        self.add_output("base_F", val=np.zeros(3), units="N")
        self.add_output("base_M", val=np.zeros(3), units="N*m")
        self.add_output("axial_stress", val=np.zeros(npts - 1), units="N/m**2")
        self.add_output("shear_stress", val=np.zeros(npts - 1), units="N/m**2")
        self.add_output("hoop_stress", val=np.zeros(npts - 1), units="N/m**2")
        self.add_output("hoop_stress_euro", val=np.zeros(npts - 1), units="N/m**2")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):

        frame3dd_opt = self.options["frame3dd_opt"]

        # ------- node data ----------------
        z = inputs["z"]
        n = len(z)
        node = np.arange(1, n + 1)
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)

        nodes = pyframe3dd.NodeData(node, x, y, z, r)
        # -----------------------------------

        # ------ reaction data ------------

        # rigid base
        node = inputs["kidx"] + np.ones(len(inputs["kidx"]))  # add one because 0-based index but 1-based node numbering
        rigid = RIGID

        reactions = pyframe3dd.ReactionData(
            node, inputs["kx"], inputs["ky"], inputs["kz"], inputs["ktx"], inputs["kty"], inputs["ktz"], rigid
        )
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)

        roll = np.zeros(n - 1)

        # average across element b.c. frame3dd uses constant section elements
        # TODO: Use nodal2sectional
        Az = inputs["Az"]
        Asx = inputs["Asx"]
        Asy = inputs["Asy"]
        Jz = inputs["Jz"]
        Ixx = inputs["Ixx"]
        Iyy = inputs["Iyy"]
        E = inputs["E"]
        G = inputs["G"]
        rho = inputs["rho"]

        elements = pyframe3dd.ElementData(element, N1, N2, Az, Asx, Asy, Jz, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], dx)
        # -----------------------------------

        # initialize frame3dd object
        cylinder = pyframe3dd.Frame(nodes, reactions, elements, options)

        # ------ add extra mass ------------

        # extra node inertia data
        N = inputs["midx"] + np.ones(len(inputs["midx"]))

        add_gravity = True
        cylinder.changeExtraNodeMass(
            N,
            inputs["m"],
            inputs["mIxx"],
            inputs["mIyy"],
            inputs["mIzz"],
            inputs["mIxy"],
            inputs["mIxz"],
            inputs["mIyz"],
            inputs["mrhox"],
            inputs["mrhoy"],
            inputs["mrhoz"],
            add_gravity,
        )

        # ------------------------------------

        # ------- enable dynamic analysis ----------
        Mmethod = 1
        lump = 0
        shift = 0.0
        # Run twice the number of modes to ensure that we can ignore the torsional modes and still get the desired number of fore-aft, side-side modes
        cylinder.enableDynamics(2 * NFREQ, Mmethod, lump, frame3dd_opt["tol"], shift)
        # ----------------------------

        # ------ static load case 1 ------------

        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        load = pyframe3dd.StaticLoadCase(gx, gy, gz)

        # point loads
        nF = inputs["plidx"] + np.ones(len(inputs["plidx"]))
        load.changePointLoads(nF, inputs["Fx"], inputs["Fy"], inputs["Fz"], inputs["Mxx"], inputs["Myy"], inputs["Mzz"])

        # distributed loads
        Px, Py, Pz = inputs["Pz"], inputs["Py"], -inputs["Px"]  # switch to local c.s.
        z = inputs["z"]

        # trapezoidally distributed loads
        EL = np.arange(1, n)
        xx1 = xy1 = xz1 = np.zeros(n - 1)
        xx2 = xy2 = xz2 = np.diff(z) - 1e-6  # subtract small number b.c. of precision
        wx1 = Px[:-1]
        wx2 = Px[1:]
        wy1 = Py[:-1]
        wy2 = Py[1:]
        wz1 = Pz[:-1]
        wz2 = Pz[1:]

        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        cylinder.addLoadCase(load)
        # Debugging
        # cylinder.write('temp.3dd')
        # -----------------------------------
        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = cylinder.run()
        iCase = 0

        # mass
        outputs["mass"] = mass.struct_mass

        # natural frequncies
        outputs["f1"] = modal.freq[0]
        outputs["f2"] = modal.freq[1]
        outputs["freqs"] = modal.freq[:NFREQ]

        # Get all mode shapes in batch
        NFREQ2 = int(NFREQ / 2)
        freq_x, freq_y, mshapes_x, mshapes_y = util.get_xy_mode_shapes(
            z, modal.freq, modal.xdsp, modal.ydsp, modal.zdsp, modal.xmpf, modal.ympf, modal.zmpf
        )
        outputs["x_mode_freqs"] = freq_x[:NFREQ2]
        outputs["y_mode_freqs"] = freq_y[:NFREQ2]
        outputs["x_mode_shapes"] = mshapes_x[:NFREQ2, :]
        outputs["y_mode_shapes"] = mshapes_y[:NFREQ2, :]

        # deflections due to loading (from cylinder top and wind/wave loads)
        outputs["cylinder_deflection"] = np.sqrt(
            displacements.dx[iCase, :] ** 2 + displacements.dy[iCase, :] ** 2
        )  # in yaw-aligned direction

        # shear and bending, one per element (convert from local to global c.s.)
        Fz = forces.Nx[iCase, 1::2]
        Vy = forces.Vy[iCase, 1::2]
        Vx = -forces.Vz[iCase, 1::2]

        Mzz = forces.Txx[iCase, 1::2]
        Myy = forces.Myy[iCase, 1::2]
        Mxx = -forces.Mzz[iCase, 1::2]

        # Record total forces and moments
        base_idx = 2 * int(inputs["kidx"].max())
        outputs["base_F"] = (
            -1.0 * np.r_[-forces.Vz[iCase, base_idx], forces.Vy[iCase, base_idx], forces.Nx[iCase, base_idx]]
        )
        outputs["base_M"] = (
            -1.0 * np.r_[-forces.Mzz[iCase, base_idx], forces.Myy[iCase, base_idx], forces.Txx[iCase, base_idx]]
        )

        outputs["Fz_out"] = Fz
        outputs["Vx_out"] = Vx
        outputs["Vy_out"] = Vy
        outputs["Mxx_out"] = Mxx
        outputs["Myy_out"] = Myy
        outputs["Mzz_out"] = Mzz

        # axial and shear stress
        d, _ = util.nodal2sectional(inputs["d"])
        qdyn, _ = util.nodal2sectional(inputs["qdyn"])

        ##R = self.d/2.0
        ##x_stress = R*np.cos(self.theta_stress)
        ##y_stress = R*np.sin(self.theta_stress)
        ##axial_stress = Fz/self.Az + Mxx/self.Ixx*y_stress - Myy/self.Iyy*x_stress
        #        V = Vy*x_stress/R - Vx*y_stress/R  # shear stress orthogonal to direction x,y
        #        shear_stress = 2. * V / self.Az  # coefficient of 2 for a hollow circular section, but should be conservative for other shapes
        outputs["axial_stress"] = (
            Fz / inputs["Az"] - np.sqrt(Mxx ** 2 + Myy ** 2) / inputs["Iyy"] * d / 2.0
        )  # More conservative, just use the tilted bending and add total max shear as well at the same point, if you do not like it go back to the previous lines

        outputs["shear_stress"] = (
            2.0 * np.sqrt(Vx ** 2 + Vy ** 2) / inputs["Az"]
        )  # coefficient of 2 for a hollow circular section, but should be conservative for other shapes

        # hoop_stress (Eurocode method)
        L_reinforced = self.options["buckling_length"] * np.ones(Fz.shape)
        outputs["hoop_stress_euro"] = hoopStressEurocode(inputs["z"], d, inputs["t"], L_reinforced, qdyn)

        # Simpler hoop stress used in API calculations
        outputs["hoop_stress"] = hoopStress(d, inputs["t"], qdyn)
