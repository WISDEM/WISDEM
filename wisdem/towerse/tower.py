import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
import wisdem.towerse.tower_props as tp
import wisdem.towerse.tower_struct as ts
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.cross_section as cs
import wisdem.commonse.utilization_dnvgl as util_dnvgl
import wisdem.commonse.utilization_eurocode as util_euro
import wisdem.commonse.utilization_constraints as util_con

# from wisdem.commonse.utilization_eurocode import hoopStressEurocode
from wisdem.commonse import NFREQ, gravity
from wisdem.commonse.cylinder_member import MemberLoads, MemberStandard, get_nfull

RIGID = 1e30
NREFINE = 3


class PreDiscretization(om.ExplicitComponent):
    """
    Process some of the tower YAML inputs.

    Parameters
    ----------
    height : float, [m]
        Scalar of the tower height computed along the z axis.
    foundation_height : float, [m]
        starting height of tower

    Returns
    -------
    height_constraint : float, [m]
        mismatch between tower height and desired hub_height
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    joint1 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for bottom node of member
    joint2 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for top node of member

    """

    def setup(self):
        self.add_input("tower_height", val=0.0, units="m")
        self.add_input("foundation_height", val=0.0, units="m")

        self.add_output("height_constraint", val=0.0, units="m")
        self.add_output("transition_piece_height", 0.0, units="m")
        self.add_output("z_start", 0.0, units="m")
        self.add_output("joint1", val=np.zeros(3), units="m")
        self.add_output("joint2", val=np.zeros(3), units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack values
        h_tow = inputs["tower_height"]
        fh_tow = inputs["foundation_height"]

        outputs["transition_piece_height"] = outputs["z_start"] = fh_tow
        outputs["joint1"] = np.r_[0.0, 0.0, fh_tow]
        outputs["joint2"] = np.r_[0.0, 0.0, fh_tow + h_tow]
        outputs["height_constraint"] = inputs["hub_height"] - outputs["joint2"][-1]


class TurbineMass(om.ExplicitComponent):
    """
    Compute the turbine mass, center of mass, and mass moment of inertia.

    Parameters
    ----------
    hub_height : float, [m]
        Hub-height
    rna_mass : float, [kg]
        Total tower mass
    rna_I : numpy array[6], [kg*m**2]
        Mass moment of inertia of RNA about tower top [xx yy zz xy xz yz]
    rna_cg : numpy array[3], [m]
        xyz-location of RNA cg relative to tower top
    tower_mass : float, [kg]
        Total tower mass
    tower_center_of_mass : float, [m]
        z-position of center of mass of tower
    tower_I_base : numpy array[6], [kg*m**2]
        Mass moment of inertia of tower about base [xx yy zz xy xz yz]

    Returns
    -------
    turbine_mass : float, [kg]
        Total mass of tower+rna
    turbine_center_of_mass : numpy array[3], [m]
        xyz-position of tower+rna center of mass
    turbine_I_base : numpy array[6], [kg*m**2]
        mass moment of inertia of tower about base [xx yy zz xy xz yz]

    """

    def setup(self):
        self.add_input("hub_height", val=0.0, units="m")
        self.add_input("rna_mass", val=0.0, units="kg")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_cg", np.zeros(3), units="m")
        self.add_input("tower_mass", val=0.0, units="kg")
        self.add_input("tower_center_of_mass", val=0.0, units="m")
        self.add_input("tower_I_base", np.zeros(6), units="kg*m**2")

        self.add_output("turbine_mass", val=0.0, units="kg")
        self.add_output("turbine_center_of_mass", val=np.zeros(3), units="m")
        self.add_output("turbine_I_base", np.zeros(6), units="kg*m**2")

        self.declare_partials(
            "turbine_I_base", ["hub_height", "rna_I", "rna_cg", "rna_mass", "tower_I_base"], method="fd"
        )
        self.declare_partials(
            "turbine_center_of_mass",
            ["hub_height", "rna_cg", "rna_mass", "tower_center_of_mass", "tower_mass"],
            method="fd",
        )
        self.declare_partials("turbine_mass", ["rna_mass", "tower_mass"], val=1.0)

    def compute(self, inputs, outputs):
        outputs["turbine_mass"] = inputs["rna_mass"] + inputs["tower_mass"]

        cg_rna = inputs["rna_cg"] + np.r_[0.0, 0.0, inputs["hub_height"]]
        cg_tower = np.r_[0.0, 0.0, inputs["tower_center_of_mass"]]
        outputs["turbine_center_of_mass"] = (inputs["rna_mass"] * cg_rna + inputs["tower_mass"] * cg_tower) / outputs[
            "turbine_mass"
        ]

        R = cg_rna
        I_tower = util.assembleI(inputs["tower_I_base"])
        I_rna = util.assembleI(inputs["rna_I"]) + inputs["rna_mass"] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["turbine_I_base"] = util.unassembleI(I_tower + I_rna)


class TowerFrame(om.ExplicitComponent):
    """
    Run Frame3DD on the tower

    Parameters
    ----------
    z_full : numpy array[npts], [m]
        location along cylinder. start at bottom and go to top
    d_full : numpy array[npts], [m]
        effective cylinder diameter for section
    t_full : numpy array[npts-1], [m]
        effective shell thickness for section
    E_full : numpy array[npts-1], [N/m**2]
        modulus of elasticity
    G_full : numpy array[npts-1], [N/m**2]
        shear modulus
    rho_full : numpy array[npts-1], [kg/m**3]
        material density
    rna_mass : float, [kg]
        added mass
    rna_I : numpy array[6], [kg*m**2]
        mass moment of inertia about some point p [xx yy zz xy xz yz]
    rna_cg : numpy array[3], [m]
        xyz-location of p relative to node
    rna_F : numpy array[3], [N]
        rna force
    rna_M : numpy array[3], [N*m]
        rna moment
    Px : numpy array[n_full], [N/m]
        force per unit length in x-direction
    Py : numpy array[n_full], [N/m]
        force per unit length in y-direction
    Pz : numpy array[n_full], [N/m]
        force per unit length in z-direction

    Returns
    -------
    f1 : float, [Hz]
        First natural frequency
    f2 : float, [Hz]
        Second natural frequency
    structural_frequencies : numpy array[NFREQ], [Hz]
        First and second natural frequency
    fore_aft_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower fore-aft direction
    side_side_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower side-side direction
    torsion_freqs : numpy array[NFREQ2]
        Frequencies associated with mode shapes in the tower torsion direction
    fore_aft_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower fore-aft direction
        (without constant term)
    side_side_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower side-side direction
        (without constant term)
    torsion_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower torsion direction
        (without constant term)
    tower_deflection : numpy array[n_full], [m]
        Deflection of tower nodes in yaw-aligned +x direction
    top_deflection : float, [m]
        Deflection of tower top in yaw-aligned +x direction
    tower_Fz : numpy array[n_full-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    tower_Vx : numpy array[n_full-1], [N]
        Shear force in x-direction in cylinder structure.
    tower_Vy : numpy array[n_full-1], [N]
        Shear force in y-direction in cylinder structure.
    tower_Mxx : numpy array[n_full-1], [N*m]
        Moment about x-axis in cylinder structure.
    tower_Myy : numpy array[n_full-1], [N*m]
        Moment about y-axis in cylinder structure.
    tower_Mzz : numpy array[n_full-1], [N*m]
        Moment about z-axis in cylinder structure.
    base_F : numpy array[3], [N]
        Total force on cylinder
    base_M : numpy array[3], [N*m]
        Total moment on cylinder measured at base
    """

    def initialize(self):
        self.options.declare("n_full")
        self.options.declare("frame3dd_opt")

    def setup(self):
        n_full = self.options["n_full"]

        # cross-sectional data along cylinder.
        self.add_input("nodes_xyz", np.zeros((n_full, 3)), units="m")
        self.add_input("section_A", np.zeros(n_full - 1), units="m**2")
        self.add_input("section_Asx", np.zeros(n_full - 1), units="m**2")
        self.add_input("section_Asy", np.zeros(n_full - 1), units="m**2")
        self.add_input("section_Ixx", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("section_Iyy", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("section_Izz", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("section_rho", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("section_E", np.zeros(n_full - 1), units="Pa")
        self.add_input("section_G", np.zeros(n_full - 1), units="Pa")
        self.add_output("section_L", np.zeros(n_full - 1), units="m")

        # point loads
        self.add_input("rna_F", np.zeros(3), units="N")
        self.add_input("rna_M", np.zeros(3), units="N*m")
        self.add_input("rna_mass", 0.0, units="kg")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_cg", np.zeros(3), units="m")

        # combined wind-water distributed loads
        self.add_input("Px", val=np.zeros(n_full), units="N/m")
        self.add_input("Py", val=np.zeros(n_full), units="N/m")
        self.add_input("Pz", val=np.zeros(n_full), units="N/m")

        # Frequencies
        NFREQ2 = int(NFREQ / 2)
        self.add_output("f1", val=0.0, units="Hz")
        self.add_output("f2", val=0.0, units="Hz")
        self.add_output("structural_frequencies", np.zeros(NFREQ), units="Hz")
        self.add_output("fore_aft_modes", np.zeros((NFREQ2, 5)))
        self.add_output("side_side_modes", np.zeros((NFREQ2, 5)))
        self.add_output("torsion_modes", np.zeros((NFREQ2, 5)))
        self.add_output("fore_aft_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("side_side_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("torsion_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("tower_deflection", np.zeros(n_full), units="m")
        self.add_output("top_deflection", 0.0, units="m")
        self.add_output("tower_Fz", val=np.zeros(n_full - 1), units="N")
        self.add_output("tower_Vx", val=np.zeros(n_full - 1), units="N")
        self.add_output("tower_Vy", val=np.zeros(n_full - 1), units="N")
        self.add_output("tower_Mxx", val=np.zeros(n_full - 1), units="N*m")
        self.add_output("tower_Myy", val=np.zeros(n_full - 1), units="N*m")
        self.add_output("tower_Mzz", val=np.zeros(n_full - 1), units="N*m")
        self.add_output("base_F", val=np.zeros(3), units="N")
        self.add_output("base_M", val=np.zeros(3), units="N*m")

    def compute(self, inputs, outputs):

        frame3dd_opt = self.options["frame3dd_opt"]

        # ------- node data ----------------
        xyz = inputs["nodes_xyz"]
        n = xyz.shape[0]
        node = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node, xyz[:, 0], xyz[:, 1], xyz[:, 2], r)
        # -----------------------------------

        # ------ reaction data ------------
        # rigid base
        rnode = np.array([0], dtype=np.int_) + 1  # 1-based indexing
        kx = ky = kz = ktx = kty = ktz = np.array([RIGID])
        reactions = pyframe3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        element = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)

        # Element properties
        Area = inputs["section_A"]
        Asx = inputs["section_Asx"]
        Asy = inputs["section_Asy"]
        J0 = inputs["section_J0"]
        Ixx = inputs["section_Ixx"]
        Iyy = inputs["section_Iyy"]
        E = inputs["section_E"]
        G = inputs["section_G"]
        rho = inputs["section_rho"]
        outputs["section_L"] = L = np.sqrt(np.sum(np.diff(xyz, axiz=0) ** 2, axis=1))

        elements = pyframe3dd.ElementData(element, N1, N2, Area, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], dx)
        # -----------------------------------

        # initialize frame3dd object
        cylinder = pyframe3dd.Frame(nodes, reactions, elements, options)

        # ------ add extra mass ------------
        # Note, need len()-1 because Frame3DD crashes if mass add at end
        midx = np.array([n - 1], dtype=np.int_)
        mI = inputs["rna_I"]
        mrho = inputs["rna_cg"]

        add_gravity = True
        cylinder.changeExtraNodeMass(
            midx,
            inputs["rna_mass"],
            np.array([mI[0]]).flatten(),
            np.array([mI[1]]).flatten(),
            np.array([mI[2]]).flatten(),
            np.array([mI[3]]).flatten(),
            np.array([mI[4]]).flatten(),
            np.array([mI[5]]).flatten(),
            np.array([mrho[0]]).flatten(),
            np.array([mrho[1]]).flatten(),
            np.array([mrho[2]]).flatten(),
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

        # Prepare point forces at RNA node
        rna_F = inputs["rna_F"]
        rna_M = inputs["rna_M"]
        load.changePointLoads(
            midx,
            np.array([rna_F[0]]).flatten(),
            np.array([rna_F[1]]).flatten(),
            np.array([rna_F[2]]).flatten(),
            np.array([rna_M[0]]).flatten(),
            np.array([rna_M[1]]).flatten(),
            np.array([rna_M[2]]).flatten(),
        )

        # distributed loads
        Px, Py, Pz = inputs["Pz"], inputs["Py"], -inputs["Px"]  # switch to local c.s.

        # trapezoidally distributed loads
        EL = np.arange(1, n)
        xx1 = xy1 = xz1 = np.zeros(n - 1)
        xx2 = xy2 = xz2 = 0.99 * L  # subtract small number b.c. of precision
        wx1 = Px[:-1]
        wx2 = Px[1:]
        wy1 = Py[:-1]
        wy2 = Py[1:]
        wz1 = Pz[:-1]
        wz2 = Pz[1:]

        load.changeTrapezoidalLoads(EL, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        cylinder.addLoadCase(load)
        # Debugging
        # cylinder.write('towerse_debug.3dd')
        # -----------------------------------
        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = cylinder.run()
        ic = 0

        # natural frequncies
        outputs["f1"] = modal.freq[0]
        outputs["f2"] = modal.freq[1]
        outputs["structural_frequencies"] = modal.freq[:NFREQ]

        # Get all mode shapes in batch
        NFREQ2 = int(NFREQ / 2)
        freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
            xyz[:, 2], modal.freq, modal.xdsp, modal.ydsp, modal.zdsp, modal.xmpf, modal.ympf, modal.zmpf
        )
        outputs["fore_aft_freqs"] = freq_x[:NFREQ2]
        outputs["side_side_freqs"] = freq_y[:NFREQ2]
        outputs["torsion_freqs"] = freq_z[:NFREQ2]
        outputs["fore_aft_modes"] = mshapes_x[:NFREQ2, :]
        outputs["side_side_modes"] = mshapes_y[:NFREQ2, :]
        outputs["torsion_modes"] = mshapes_z[:NFREQ2, :]

        # deflections due to loading (from cylinder top and wind/wave loads)
        outputs["tower_deflection"] = np.sqrt(
            displacements.dx[ic, :] ** 2 + displacements.dy[ic, :] ** 2
        )  # in yaw-aligned direction
        outputs["top_deflection"] = outputs["tower_deflection"][-1]

        # Record total forces and moments
        ibase = 2 * int(inputs["kidx"].max())
        outputs["base_F"] = -np.r_[-forces.Vz[ic, ibase], forces.Vy[ic, ibase], forces.Nx[ic, ibase]]
        outputs["base_M"] = -np.r_[-forces.Mzz[ic, ibase], forces.Myy[ic, ibase], forces.Txx[ic, ibase]]

        # Forces and moments along the structure
        outputs["tower_Fz"] = forces.Nx[ic, 1::2]
        outputs["tower_Vx"] = -forces.Vz[ic, 1::2]
        outputs["tower_Vy"] = forces.Vy[ic, 1::2]
        outputs["tower_Mxx"] = -forces.Mzz[ic, 1::2]
        outputs["tower_Myy"] = forces.Myy[ic, 1::2]
        outputs["tower_Mzz"] = forces.Txx[ic, 1::2]


class TowerPostFrame(om.ExplicitComponent):
    """
    Postprocess results from Frame3DD.

    Parameters
    ----------
    z_full : numpy array[n_full], [m]
        location along tower. start at bottom and go to top
    d_full : numpy array[n_full], [m]
        effective tower diameter for section
    t_full : numpy array[n_full-1], [m]
        effective shell thickness for section
    E_full : numpy array[n_full-1], [Pa]
        Isotropic Youngs modulus of the materials along the tower sections.
    G_full : numpy array[n_full-1], [Pa]
        Isotropic shear modulus of the materials along the tower sections.
    rho_full : numpy array[n_full-1], [kg/m**3]
        Density of the materials along the tower sections.
    sigma_y_full : numpy array[n_full-1], [N/m**2]
        yield stress
    tower_Fz : numpy array[n_full-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    tower_Vx : numpy array[n_full-1], [N]
        Shear force in x-direction in cylinder structure.
    tower_Vy : numpy array[n_full-1], [N]
        Shear force in y-direction in cylinder structure.
    tower_Mxx : numpy array[n_full-1], [N*m]
        Moment about x-axis in cylinder structure.
    tower_Myy : numpy array[n_full-1], [N*m]
        Moment about y-axis in cylinder structure.
    tower_Mzz : numpy array[n_full-1], [N*m]
        Moment about z-axis in cylinder structure.
    qdyn : numpy array[n_full], [N/m**2]
        dynamic pressure

    Returns
    -------
    axial_stress : numpy array[n_full-1], [N/m**2]
        Axial stress in cylinder structure
    shear_stress : numpy array[n_full-1], [N/m**2]
        Shear stress in cylinder structure
    hoop_stress : numpy array[n_full-1], [N/m**2]
        Hoop stress in cylinder structure calculated with simple method used in API
        standards
    hoop_stress_euro : numpy array[n_full-1], [N/m**2]
        Hoop stress in cylinder structure calculated with Eurocode method
    stress : numpy array[n_full-1]
        Von Mises stress utilization along tower at specified locations. Includes safety
        factor.
    shell_buckling : numpy array[n_full-1]
        Shell buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    global_buckling : numpy array[n_full-1]
        Global buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    turbine_F : numpy array[3], [N]
        Total force on tower+rna
    turbine_M : numpy array[3], [N*m]
        Total x-moment on tower+rna measured at base

    """

    def initialize(self):
        self.options.declare("modeling_options")
        # self.options.declare('nDEL')

    def setup(self):
        n_height = self.options["modeling_options"]["n_height"]
        n_refine = self.options["modeling_options"]["n_refine"]
        n_full = get_nfull(n_height, nref=n_refine)

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_input("z_full", np.zeros(n_full), units="m")
        self.add_input("d_full", np.zeros(n_full), units="m")
        self.add_input("t_full", np.zeros(n_full - 1), units="m")

        # Material properties
        self.add_input("E_full", np.zeros(n_full - 1), units="N/m**2")
        self.add_input("G_full", np.zeros(n_full - 1), units="Pa")
        self.add_input("rho_full", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("sigma_y_full", np.zeros(n_full - 1), units="N/m**2")

        # Processed Frame3DD/OpenFAST outputs
        self.add_input("tower_Fz", val=np.zeros(n_full - 1), units="N")
        self.add_input("tower_Vx", val=np.zeros(n_full - 1), units="N")
        self.add_input("tower_Vy", val=np.zeros(n_full - 1), units="N")
        self.add_input("tower_Mxx", val=np.zeros(n_full - 1), units="N*m")
        self.add_input("tower_Myy", val=np.zeros(n_full - 1), units="N*m")
        self.add_input("tower_Mzz", val=np.zeros(n_full - 1), units="N*m")
        self.add_input("qdyn", val=np.zeros(n_full), units="N/m**2")

        # Load analysis
        self.add_output("axial_stress", val=np.zeros(n_full - 1), units="N/m**2")
        self.add_output("shear_stress", val=np.zeros(n_full - 1), units="N/m**2")
        self.add_output("hoop_stress", val=np.zeros(n_full - 1), units="N/m**2")

        self.add_output("hoop_stress_euro", val=np.zeros(n_full - 1), units="N/m**2")
        self.add_output("constr_stress", np.zeros(n_full - 1))
        self.add_output("constr_shell_buckling", np.zeros(n_full - 1))
        self.add_output("constr_global_buckling", np.zeros(n_full - 1))
        # self.add_output('constr_damage', np.zeros(n_full-1), desc='Fatigue damage at each tower section')
        self.add_output("turbine_F", val=np.zeros(3), units="N", desc="Total force on tower+rna")
        self.add_output("turbine_M", val=np.zeros(3), units="N*m", desc="Total x-moment on tower+rna measured at base")

    def compute(self, inputs, outputs):
        # Unpack some variables
        sigma_y = inputs["sigma_y_full"]
        E = inputs["E_full"]
        G = inputs["G_full"]
        z = inputs["z_full"]
        t = inputs["t_full"]
        d = inputs["d_full"]
        h = np.diff(z)
        d_sec, _ = util.nodal2sectional(d)
        r_sec = 0.5 * d_sec
        itube = cs.Tube(d_sec, t)

        L_buckling = self.options["modeling_options"]["buckling_length"]
        gamma_f = self.options["modeling_options"]["gamma_f"]
        gamma_m = self.options["modeling_options"]["gamma_m"]
        gamma_n = self.options["modeling_options"]["gamma_n"]
        gamma_b = self.options["modeling_options"]["gamma_b"]

        # axial and shear stress
        qdyn, _ = util.nodal2sectional(inputs["qdyn"])

        ##R = self.d/2.0
        ##x_stress = R*np.cos(self.theta_stress)
        ##y_stress = R*np.sin(self.theta_stress)
        ##axial_stress = Fz/self.Az + Mxx/self.Ixx*y_stress - Myy/Iyy*x_stress
        #        V = Vy*x_stress/R - Vx*y_stress/R  # shear stress orthogonal to direction x,y
        #        shear_stress = 2. * V / self.Az  # coefficient of 2 for a hollow circular section, but should be conservative for other shapes

        # Get loads from Framee3dd/OpenFAST
        Fz = inputs["tower_Fz"]
        Vx = inputs["tower_Vx"]
        Vy = inputs["tower_Vy"]
        Mxx = inputs["tower_Mxx"]
        Myy = inputs["tower_Myy"]
        Mzz = inputs["tower_Mzz"]

        M = np.sqrt(Mxx ** 2 + Myy ** 2)
        V = np.sqrt(Vx ** 2 + Vy ** 2)

        # Geom properties
        Az = itube.Area
        Asx = itube.Asx
        Asy = itube.Asy
        Jz = itube.J0
        Ixx = itube.Ixx
        Iyy = itube.Iyy

        # See http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        outputs["axial_stress"] = axial_stress = Fz / Az + M * r_sec / Iyy
        outputs["shear_stress"] = shear_stress = np.abs(Mzz) / Jz * r_sec + V / Asx
        outputs["hoop_stress"] = hoop_stress = util_con.hoopStress(d_sec, t, qdyn)
        outputs["constr_stress"] = util_con.vonMisesStressUtilization(
            axial_stress, hoop_stress, shear_stress, gamma_f * gamma_m * gamma_n, sigma_y
        )

        if self.options["modeling_options"]["buckling_method"].lower().find("euro") >= 0:
            # Use Euro-code method
            L_buckling = L_buckling * np.ones(axial_stress.shape)
            hoop_euro = util_euro.hoopStressEurocode(d_sec, t, L_buckling, hoop_stress)
            outputs["hoop_stress_euro"] = hoop_euro

            shell_buckling = util_euro.shellBucklingEurocode(
                d, t, axial_stress, hoop_euro, shear_stress, L_buckling, E, sigma_y, gamma_f, gamma_b
            )

            tower_height = z[-1] - z[0]
            global_buckling = util_euro.bucklingGL(d_sec, t, Fz, M, tower_height, E, sigma_y, gamma_f, gamma_b)

        else:
            # Use DNV-GL CP202 Method
            check = util_dnvgl.CylinderBuckling(
                h,
                d,
                t,
                E=E,
                G=G,
                sigma_y=sigma_y,
                gamma=gamma_f * gamma_b,
            )
            results = check.run_buckling_checks(Fz, M, axial_stress, hoop_stress, shear_stress)
            shell_buckling = results["Shell"]
            global_buckling = results["Global"]

        outputs["constr_shell_buckling"] = shell_buckling
        outputs["constr_global_buckling"] = global_buckling


class TowerLeanSE(om.Group):
    """
    This is a geometry preprocessing group for the tower.

    This group contains components that calculate the geometric properties of
    the tower, such as mass and moments of inertia, as well as geometric
    constraints like diameter-to-thickness and taper ratio. No static or dynamic
    analysis of the tower occurs here.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]
        n_mat = self.options["modeling_options"]["materials"]["n_materials"]

        self.add_subsystem("predis", tp.PreDiscretization(), promotes=["*"])

        promlist = [
            "E_mat",
            "G_mat",
            "sigma_y_mat",
            "sigma_ult_mat",
            "wohler_exp_mat",
            "wohler_A_mat",
            "rho_mat",
            "rho_water",
            "unit_cost_mat",
            "material_names",
            "painting_cost_rate",
            "labor_cost_rate",
            "z_param",
            "z_full",
            "d_full",
            "t_full",
            "joint1",
            "joint2",
            "E",
            "G",
            "sigma_y",
            "sigma_ult",
            "wohler_exp",
            "wohler_A",
            "rho",
            "unit_cost",
            "outfitting_factor",
            "constr_taper",
            "constr_d_to_t",
            "slope",
            ("s", "tower_s"),
            ("layer_materials", "tower_layer_materials"),
            ("layer_thickness", "tower_layer_thickness"),
            ("height", "tower_height"),
            ("section_height", "tower_section_height"),
            ("outer_diameter_in", "tower_outer_diameter_in"),
            ("outer_diameter", "tower_outer_diameter"),
            ("wall_thickness", "tower_wall_thickness"),
        ]

        self.add_subsystem(
            "member", MemberStandard(column_options=mod_opt, idx=0, n_mat=n_mat, n_refine=NREFINE), promotes=promlist
        )

        self.add_subsystem(
            "turb",
            tp.TurbineMass(),
            promotes=[
                "turbine_mass",
                "monopile_mass",
                "tower_mass",
                "tower_center_of_mass",
                "tower_I_base",
                "rna_mass",
                "rna_cg",
                "rna_I",
                "hub_height",
            ],
        )


class TowerSE(om.Group):
    """
    This is the main TowerSE group that performs analysis of the tower.

    This group takes in geometric inputs from TowerLeanSE and environmental and
    loading conditions.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["TowerSE"]
        nLC = mod_opt["nLC"]  # not yet supported
        wind = mod_opt["wind"]  # not yet supported
        frame3dd_opt = mod_opt["frame3dd"]
        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_tow = mod_opt["n_height_tower"]
            n_height = mod_opt["n_height"] = n_height_tow
        n_full = get_nfull(n_height, nref=mod_opt["n_refine"])

        # Load baseline discretization
        self.add_subsystem("geom", TowerLeanSE(modeling_options=self.options["modeling_options"]), promotes=["*"])

        self.add_subsystem("loads", MemberLoads(n_full=n_full, n_lc=nLC, wind=wind, hydro=False), promotes=["*"])
        self.connect("z_full", "loads.z")
        self.connect("d_full", "loads.d")

        for iLC in range(nLC):
            lc = "" if nLC == 1 else str(iLC + 1)

            self.add_subsystem(
                f"tower{lc}",
                ts.TowerFrame(n_full=n_full, frame3dd_opt=frame3dd_opt),
                promotes=["z_full", "d_full", "t_full", "rho_full", "E_full", "G_full"],
            )

            self.add_subsystem(
                f"post{lc}",
                ts.TowerPostFrame(modeling_options=mod_opt),
                promotes=[
                    "z_full",
                    "d_full",
                    "t_full",
                    "rho_full",
                    "E_full",
                    "G_full",
                    "sigma_y_full",
                    "suctionpile_depth",
                ],
            )

            self.connect(f"loads.g2e{lc}.Px", f"tower{lc}.Px")
            self.connect(f"loads.g2e{lc}.Py", f"tower{lc}.Py")
            self.connect(f"loads.g2e{lc}.Pz", f"tower{lc}.Pz")
            self.connect(f"loads.g2e{lc}.qdyn", f"post{lc}.qdyn")

            self.connect(f"tower{lc}.tower_Fz", f"post{lc}.tower_Fz")
            self.connect(f"tower{lc}.tower_Vx", f"post{lc}.tower_Vx")
            self.connect(f"tower{lc}.tower_Vy", f"post{lc}.tower_Vy")
            self.connect(f"tower{lc}.tower_Mxx", f"post{lc}.tower_Mxx")
            self.connect(f"tower{lc}.tower_Myy", f"post{lc}.tower_Myy")
            self.connect(f"tower{lc}.tower_Mzz", f"post{lc}.tower_Mzz")
