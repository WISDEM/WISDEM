import copy

import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.utilization_dnvgl as util_dnvgl
import wisdem.commonse.utilization_eurocode as util_euro
import wisdem.commonse.utilization_constraints as util_con
from wisdem.commonse import NFREQ, gravity
from wisdem.commonse.cylinder_member import MemberLoads, MemberStandard, get_nfull
from wisdem.commonse.utilization_eurocode import hoopStressEurocode

RIGID = 1e30
NREFINE = 3
NPTS_SOIL = 10


class PreDiscretization(om.ExplicitComponent):
    """
    Process some of the tower YAML inputs.

    Parameters
    ----------
    tower_foundation_height : float, [m]
        Starting z-coordinate value of the tower with 0 at the water line
    monopile_foundation_height : float, [m]
        Starting z-coordinate value of the monopile with 0 at the water line
    monopile_height : float, [m]
        Scalar of the monopile height computed along the z axis.

    Returns
    -------
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    joint1 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for bottom node of member
    joint2 : numpy array[3], [m]
        Global dimensional coordinates (x-y-z) for top node of member
    suctionpile_depth : float, [m]
        Depth of monopile below sea floor

    """

    def setup(self):
        self.add_input("monopile_height", val=0.0, units="m")
        self.add_input("tower_foundation_height", val=0.0, units="m")
        self.add_input("monopile_foundation_height", val=0.0, units="m")

        self.add_output("transition_piece_height", 0.0, units="m")
        self.add_output("z_start", 0.0, units="m")
        self.add_output("suctionpile_depth", 0.0, units="m")
        self.add_output("s_const1", 0.0)
        self.add_output("joint1", val=np.zeros(3), units="m")
        self.add_output("joint2", val=np.zeros(3), units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack values
        h_mon = inputs["monopile_height"]
        fh_tow = inputs["tower_foundation_height"]
        fh_mon = inputs["monopile_foundation_height"]
        water_depth = inputs["water_depth"]

        outputs["transition_piece_height"] = fh_tow

        if np.abs(fh_tow - fh_mon - h_mon) > 1.0:
            print(
                "WARNING: Monopile length is not consistent with transition piece height and monopile base height\n",
                "         Determining new base height value . . .",
            )
        outputs["z_start"] = fh_tow - h_mon

        pile = h_mon - fh_tow - water_depth
        outputs["suctionpile_depth"] = pile
        outputs["s_const1"] = pile / h_mon  # Ensure that we have only one segment for pile, a current limitation
        outputs["joint1"] = np.r_[0.0, 0.0, fh_tow - h_mon]
        outputs["joint2"] = np.r_[0.0, 0.0, fh_tow]


class TowerMass(om.ExplicitComponent):
    """
    Compute the tower and monopile masses, raw cost, and CG properties.

    Parameters
    ----------
    shell_mass : numpy array[nFull-1], [kg]
        Total cylinder mass
    shell_cost : float, [USD]
        Total cylinder cost
    shell_z_cg : float, [m]
        z position of center of mass of cylinder
    shell_I_base : numpy array[6], [kg*m**2]
        Mass moment of inertia of cylinder about base [xx yy zz xy xz yz]
    transition_piece_height : float, [m]
        Point mass height of transition piece above water line
    transition_piece_mass : float, [kg]
        Point mass of transition piece
    transition_piece_cost : float, [USD]
        Cost of transition piece
    gravity_foundation_mass : float, [kg]
        Extra mass of gravity foundation
    z_full : numpy array[nFull], [m]
        Parameterized locations along tower, linear lofting between
    d_full : numpy array[nFull], [m]
        diameter along tower

    Returns
    -------
    structural_cost : float, [USD]
        Total structural cost (tower+monopile)
    structural_mass : float, [kg]
        Total structural mass (tower+monopile)
    monopile_mass : float, [kg]
        Mass of monopile from bottom of suction pile through transition piece
    monopile_cost : float, [USD]
        Total monopile cost

    """

    def initialize(self):
        self.options.declare("n_height")
        self.options.declare("n_refine")

    def setup(self):
        n_height = self.options["n_height"]
        n_refine = self.options["n_refine"]
        nFull = get_nfull(n_height, nref=n_refine)

        self.add_input("shell_mass", val=0.0, units="kg")
        self.add_input("shell_cost", val=0.0, units="USD")
        self.add_input("shell_z_cg", val=0.0, units="m")
        self.add_input("shell_I_base", np.zeros(6), units="kg*m**2")
        self.add_input("transition_piece_height", 0.0, units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_cost", 0.0, units="USD")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("z_full", val=np.zeros(nFull), units="m")
        self.add_input("d_full", val=np.zeros(nFull), units="m")
        self.add_input("tower_cost", val=0.0, units="USD")
        self.add_input("tower_mass", val=0.0, units="kg")

        self.add_output("structural_cost", val=0.0, units="USD")
        self.add_output("structural_mass", val=0.0, units="kg")
        self.add_output("monopile_mass", val=0.0, units="kg")
        self.add_output("monopile_cost", val=0.0, units="USD")
        self.add_output("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_output("gravity_foundation_I", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack inputs
        z = inputs["z_full"]
        d = inputs["d_full"]
        z_trans = inputs["transition_piece_height"]
        m_trans = inputs["transition_piece_mass"]
        c_trans = inputs["transition_piece_cost"]
        m_grav = inputs["gravity_foundation_mass"]
        m_cyl = inputs["cylinder_mass"]
        c_cyl = inputs["cylinder_mass"]

        outputs["monopile_mass"] = m_cyl + m_trans + m_grav
        outputs["monopile_cost"] = c_cyl + c_trans
        outputs["structural_mass"] = outputs["monopile_mass"] + inputs["tower_mass"]
        outputs["structural_cost"] = outputs["monopile_cost"] + inputs["tower_cost"]

        # Mass properties for transition piece and gravity foundation
        itrans = util.find_nearest(z, z_trans)
        r_trans = 0.5 * d[itrans]
        r_grav = 0.5 * d[0]
        I_trans = m_trans * r_trans ** 2.0 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]  # shell
        I_grav = m_grav * r_grav ** 2.0 * np.r_[0.25, 0.25, 0.5, np.zeros(3)]  # disk
        outputs["transition_piece_I"] = I_trans
        outputs["gravity_foundation_I"] = I_grav


class MonopileFrame(om.ExplicitComponent):
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
        self.options.declare("soil_springs", default=False)
        self.options.declare("gravity_foundation", default=False)

    def setup(self):
        n_full = self.options["n_full"]

        # Monopile handling
        self.add_input("z_soil", np.zeros(NPTS_SOIL), units="N/m")
        self.add_input("k_soil", np.zeros((NPTS_SOIL, 6)), units="N/m")

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
        self.add_input("transition_F", np.zeros(3), units="N")
        self.add_input("transition_M", np.zeros(3), units="N*m")
        self.add_input("turbine_mass", 0.0, units="kg")
        self.add_input("turbine_base_I", np.zeros(6), units="kg*m**2")
        self.add_input("turbine_cg", np.zeros(3), units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("transition_piece_height", 0.0, units="m")
        self.add_input("suctionpile_depth", 0.0, units="m")

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

        # Prepare for reactions: rigid at tower base
        if self.options["monopile"] and not self.options["gravity_foundation"]:
            if self.options["soil_springs"]:
                z_soil = inputs["z_soil"]
                k_soil = inputs["k_soil"]
                z_pile = z[z <= (z[0] + 1e-1 + np.abs(z_soil[0]))]
                if z_pile.size != 4:
                    print(z)
                    print(z_soil)
                    print(z_pile)
                    raise ValueError("Please use only one section for submerged pile for now")
                k_mono = np.zeros((z_pile.size, 6))
                for k in range(6):
                    k_mono[:, k] = np.interp(z_pile + np.abs(z_soil[0]), z_soil, k_soil[:, k])
                outputs["kidx"] = np.arange(len(z_pile), dtype=np.int_)
                outputs["kx"] = np.array([k_mono[:, 0]])
                outputs["ky"] = np.array([k_mono[:, 2]])
                outputs["kz"] = np.zeros(k_mono.shape[0])
                outputs["kz"][0] = np.array([k_mono[0, 4]])
                outputs["ktx"] = np.array([k_mono[:, 1]])
                outputs["kty"] = np.array([k_mono[:, 3]])
                outputs["ktz"] = np.array([k_mono[:, 5]])

            else:
                z_pile = z[z <= (z[0] + 1e-1 + inputs["suctionpile_depth"])]
                npile = z_pile.size
                if npile != 4:
                    print(z)
                    print(z_pile)
                    print(inputs["suctionpile_depth"])
                    raise ValueError("Please use only one section for submerged pile for now")
                outputs["kidx"] = np.arange(npile, dtype=np.int_)
                outputs["kx"] = outputs["ky"] = outputs["kz"] = RIGID * np.ones(npile)
                outputs["ktx"] = outputs["kty"] = outputs["ktz"] = RIGID * np.ones(npile)

        else:
            outputs["kidx"] = np.array([0], dtype=np.int_)
            outputs["kx"] = outputs["ky"] = outputs["kz"] = np.array([RIGID])
            outputs["ktx"] = outputs["kty"] = outputs["ktz"] = np.array([RIGID])

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

        # Prepare RNA, transition piece, and gravity foundation (if any applicable) for "extra node mass"
        itrans = util.find_nearest(z, inputs["transition_piece_height"])
        mtrans = inputs["transition_piece_mass"]
        Itrans = inputs["transition_piece_I"]
        mgrav = inputs["gravity_foundation_mass"]
        Igrav = inputs["gravity_foundation_I"]
        # Note, need len()-1 because Frame3DD crashes if mass add at end
        outputs["midx"] = np.array([nFull - 1, itrans, 0], dtype=np.int_)
        outputs["m"] = np.array([inputs["mass"], mtrans, mgrav]).flatten()
        outputs["mIxx"] = np.array([inputs["mI"][0], Itrans[0], Igrav[0]]).flatten()
        outputs["mIyy"] = np.array([inputs["mI"][1], Itrans[1], Igrav[1]]).flatten()
        outputs["mIzz"] = np.array([inputs["mI"][2], Itrans[2], Igrav[2]]).flatten()
        outputs["mIxy"] = np.array([inputs["mI"][3], Itrans[3], Igrav[3]]).flatten()
        outputs["mIxz"] = np.array([inputs["mI"][4], Itrans[4], Igrav[4]]).flatten()
        outputs["mIyz"] = np.array([inputs["mI"][5], Itrans[5], Igrav[5]]).flatten()
        outputs["mrhox"] = np.array([inputs["mrho"][0], 0.0, 0.0]).flatten()
        outputs["mrhoy"] = np.array([inputs["mrho"][1], 0.0, 0.0]).flatten()
        outputs["mrhoz"] = np.array([inputs["mrho"][2], 0.0, 0.0]).flatten()

        # Prepare point forces at RNA node
        outputs["plidx"] = np.array([nFull - 1], dtype=np.int_)  # -1 b/c same reason as above
        outputs["Fx"] = np.array([inputs["rna_F"][0]]).flatten()
        outputs["Fy"] = np.array([inputs["rna_F"][1]]).flatten()
        outputs["Fz"] = np.array([inputs["rna_F"][2]]).flatten()
        outputs["Mxx"] = np.array([inputs["rna_M"][0]]).flatten()
        outputs["Myy"] = np.array([inputs["rna_M"][1]]).flatten()
        outputs["Mzz"] = np.array([inputs["rna_M"][2]]).flatten()


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
    Px : numpy array[nFull], [N/m]
        force per unit length in x-direction
    Py : numpy array[nFull], [N/m]
        force per unit length in y-direction
    Pz : numpy array[nFull], [N/m]
        force per unit length in z-direction

    Returns
    -------
    mass : float, [kg]
        Structural mass computed by Frame3DD
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
    tower_deflection : numpy array[nFull], [m]
        Deflection of tower nodes in yaw-aligned +x direction
    top_deflection : float, [m]
        Deflection of tower top in yaw-aligned +x direction
    tower_Fz : numpy array[nFull-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    tower_Vx : numpy array[nFull-1], [N]
        Shear force in x-direction in cylinder structure.
    tower_Vy : numpy array[nFull-1], [N]
        Shear force in y-direction in cylinder structure.
    tower_Mxx : numpy array[nFull-1], [N*m]
        Moment about x-axis in cylinder structure.
    tower_Myy : numpy array[nFull-1], [N*m]
        Moment about y-axis in cylinder structure.
    tower_Mzz : numpy array[nFull-1], [N*m]
        Moment about z-axis in cylinder structure.
    base_F : numpy array[3], [N]
        Total force on cylinder
    base_M : numpy array[3], [N*m]
        Total moment on cylinder measured at base
    """

    def initialize(self):
        self.options.declare("nFull")
        self.options.declare("nK")
        self.options.declare("nMass")
        self.options.declare("nPL")
        self.options.declare("frame3dd_opt")

    def setup(self):
        nFull = self.options["nFull"]
        nK = self.options["nK"]
        nMass = self.options["nMass"]
        nPL = self.options["nPL"]

        # cross-sectional data along cylinder.
        self.add_input("z", val=np.zeros(nFull), units="m")
        self.add_input("Az", val=np.zeros(nFull - 1), units="m**2")
        self.add_input("Asx", val=np.zeros(nFull - 1), units="m**2")
        self.add_input("Asy", val=np.zeros(nFull - 1), units="m**2")
        self.add_input("Jz", val=np.zeros(nFull - 1), units="m**4")
        self.add_input("Ixx", val=np.zeros(nFull - 1), units="m**4")
        self.add_input("Iyy", val=np.zeros(nFull - 1), units="m**4")
        self.add_input("E", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_input("G", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_input("rho", val=np.zeros(nFull - 1), units="kg/m**3")

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_input("d", val=np.zeros(nFull), units="m")
        self.add_input("t", val=np.zeros(nFull - 1), units="m")

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
        self.add_input("Px", val=np.zeros(nFull), units="N/m")
        self.add_input("Py", val=np.zeros(nFull), units="N/m")
        self.add_input("Pz", val=np.zeros(nFull), units="N/m")

        # Frequencies
        NFREQ2 = int(NFREQ / 2)
        self.add_output("mass", val=0.0, units="kg")
        self.add_output("f1", val=0.0, units="Hz")
        self.add_output("f2", val=0.0, units="Hz")
        self.add_output("structural_frequencies", np.zeros(NFREQ), units="Hz")
        self.add_output("fore_aft_modes", np.zeros((NFREQ2, 5)))
        self.add_output("side_side_modes", np.zeros((NFREQ2, 5)))
        self.add_output("torsion_modes", np.zeros((NFREQ2, 5)))
        self.add_output("fore_aft_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("side_side_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("torsion_freqs", np.zeros(NFREQ2), units="Hz")
        self.add_output("tower_deflection", np.zeros(nFull), units="m")
        self.add_output("top_deflection", 0.0, units="m")
        self.add_output("tower_Fz", val=np.zeros(nFull - 1), units="N")
        self.add_output("tower_Vx", val=np.zeros(nFull - 1), units="N")
        self.add_output("tower_Vy", val=np.zeros(nFull - 1), units="N")
        self.add_output("tower_Mxx", val=np.zeros(nFull - 1), units="N*m")
        self.add_output("tower_Myy", val=np.zeros(nFull - 1), units="N*m")
        self.add_output("tower_Mzz", val=np.zeros(nFull - 1), units="N*m")
        self.add_output("base_F", val=np.zeros(3), units="N")
        self.add_output("base_M", val=np.zeros(3), units="N*m")

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

        # Element properties
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
        ic = 0

        # mass
        outputs["mass"] = mass.struct_mass

        # natural frequncies
        outputs["f1"] = modal.freq[0]
        outputs["f2"] = modal.freq[1]
        outputs["structural_frequencies"] = modal.freq[:NFREQ]

        # Get all mode shapes in batch
        NFREQ2 = int(NFREQ / 2)
        freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
            z, modal.freq, modal.xdsp, modal.ydsp, modal.zdsp, modal.xmpf, modal.ympf, modal.zmpf
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
    z_full : numpy array[nFull], [m]
        location along tower. start at bottom and go to top
    d_full : numpy array[nFull], [m]
        effective tower diameter for section
    t_full : numpy array[nFull-1], [m]
        effective shell thickness for section
    suctionpile_depth : float, [m]
        Depth of monopile below sea floor
    E_full : numpy array[nFull-1], [Pa]
        Isotropic Youngs modulus of the materials along the tower sections.
    G_full : numpy array[nFull-1], [Pa]
        Isotropic shear modulus of the materials along the tower sections.
    rho_full : numpy array[nFull-1], [kg/m**3]
        Density of the materials along the tower sections.
    sigma_y_full : numpy array[nFull-1], [N/m**2]
        yield stress
    tower_Fz : numpy array[nFull-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    tower_Vx : numpy array[nFull-1], [N]
        Shear force in x-direction in cylinder structure.
    tower_Vy : numpy array[nFull-1], [N]
        Shear force in y-direction in cylinder structure.
    tower_Mxx : numpy array[nFull-1], [N*m]
        Moment about x-axis in cylinder structure.
    tower_Myy : numpy array[nFull-1], [N*m]
        Moment about y-axis in cylinder structure.
    tower_Mzz : numpy array[nFull-1], [N*m]
        Moment about z-axis in cylinder structure.
    qdyn : numpy array[nFull], [N/m**2]
        dynamic pressure

    Returns
    -------
    axial_stress : numpy array[nFull-1], [N/m**2]
        Axial stress in cylinder structure
    shear_stress : numpy array[nFull-1], [N/m**2]
        Shear stress in cylinder structure
    hoop_stress : numpy array[nFull-1], [N/m**2]
        Hoop stress in cylinder structure calculated with simple method used in API
        standards
    hoop_stress_euro : numpy array[nFull-1], [N/m**2]
        Hoop stress in cylinder structure calculated with Eurocode method
    stress : numpy array[nFull-1]
        Von Mises stress utilization along tower at specified locations. Includes safety
        factor.
    shell_buckling : numpy array[nFull-1]
        Shell buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    global_buckling : numpy array[nFull-1]
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
        nFull = get_nfull(n_height, nref=n_refine)

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_input("z_full", np.zeros(nFull), units="m")
        self.add_input("d_full", np.zeros(nFull), units="m")
        self.add_input("t_full", np.zeros(nFull - 1), units="m")
        self.add_input("suctionpile_depth", 0.0, units="m")

        self.add_input("Az", val=np.zeros(nFull - 1), units="m**2")
        self.add_input("Asx", val=np.zeros(nFull - 1), units="m**2")
        self.add_input("Asy", val=np.zeros(nFull - 1), units="m**2")
        self.add_input("Jz", val=np.zeros(nFull - 1), units="m**4")
        self.add_input("Ixx", val=np.zeros(nFull - 1), units="m**4")
        self.add_input("Iyy", val=np.zeros(nFull - 1), units="m**4")

        # Material properties
        self.add_input("E_full", np.zeros(nFull - 1), units="N/m**2")
        self.add_input("G_full", np.zeros(nFull - 1), units="Pa")
        self.add_input("rho_full", np.zeros(nFull - 1), units="kg/m**3")
        self.add_input("sigma_y_full", np.zeros(nFull - 1), units="N/m**2")

        # Processed Frame3DD/OpenFAST outputs
        self.add_input("tower_Fz", val=np.zeros(nFull - 1), units="N")
        self.add_input("tower_Vx", val=np.zeros(nFull - 1), units="N")
        self.add_input("tower_Vy", val=np.zeros(nFull - 1), units="N")
        self.add_input("tower_Mxx", val=np.zeros(nFull - 1), units="N*m")
        self.add_input("tower_Myy", val=np.zeros(nFull - 1), units="N*m")
        self.add_input("tower_Mzz", val=np.zeros(nFull - 1), units="N*m")
        self.add_input("qdyn", val=np.zeros(nFull), units="N/m**2")

        # fatigue parameters
        self.add_input("life", 20.0)
        # self.add_input('m_SN', 4, desc='slope of S/N curve')
        # self.add_input('DC', 80.0, desc='standard value of stress')
        # self.add_input('z_DEL', np.zeros(nDEL), units='m', desc='absolute z coordinates of corresponding fatigue parameters')
        # self.add_input('M_DEL', np.zeros(nDEL), desc='fatigue parameters at corresponding z coordinates')

        # Load analysis
        self.add_output("axial_stress", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_output("shear_stress", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_output("hoop_stress", val=np.zeros(nFull - 1), units="N/m**2")

        self.add_output("hoop_stress_euro", val=np.zeros(nFull - 1), units="N/m**2")
        self.add_output("constr_stress", np.zeros(nFull - 1))
        self.add_output("constr_shell_buckling", np.zeros(nFull - 1))
        self.add_output("constr_global_buckling", np.zeros(nFull - 1))
        # self.add_output('constr_damage', np.zeros(nFull-1), desc='Fatigue damage at each tower section')
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
        n_sec = r_sec.size

        L_suction = float(inputs["suctionpile_depth"])
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
        Az = inputs["Az"]
        Asx = inputs["Asx"]
        Asy = inputs["Asy"]
        Jz = inputs["Jz"]
        Ixx = inputs["Ixx"]
        Iyy = inputs["Iyy"]

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

            tower_height = z[-1] - z[0] - L_suction
            global_buckling = util_euro.bucklingGL(d_sec, t, Fz, M, tower_height, E, sigma_y, gamma_f, gamma_b)

        else:
            # Use DNV-GL CP202 Method
            check = util_dnvgl.CylinderBuckling(
                h, d, t, E=E, G=G, sigma_y=sigma_y, gamma=gamma_f * gamma_b, mod_length=L_suction
            )
            results = check.run_buckling_checks(Fz, M, axial_stress, hoop_stress, shear_stress)
            shell_buckling = results["Shell"]
            global_buckling = results["Global"]

        outputs["constr_shell_buckling"] = shell_buckling
        outputs["constr_global_buckling"] = global_buckling

        # fatigue
        # N_DEL = 365.0 * 24.0 * 3600.0 * inputs["life"] * np.ones(len(t))
        # outputs['damage'] = np.zeros(N_DEL.shape)

        # if any(inputs['M_DEL']):
        #    M_DEL = np.interp(z_section, inputs['z_DEL'], inputs['M_DEL'])

        #    outputs['damage'] = util_con.fatigue(M_DEL, N_DEL, d, inputs['t'], inputs['m_SN'],
        #                                      inputs['DC'], gamma_fatigue, stress_factor=1.0, weld_factor=True)
