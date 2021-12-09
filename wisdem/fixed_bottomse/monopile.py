import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.cylinder_member as mem
from wisdem.commonse import NFREQ, gravity
from wisdem.commonse.environment import TowerSoil

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
    bending_height : float, [m]
        Length of monopile above mudline subject to bending

    """

    def setup(self):
        self.add_input("monopile_height", val=0.0, units="m")
        self.add_input("tower_foundation_height", val=0.0, units="m")
        self.add_input("monopile_foundation_height", val=0.0, units="m")
        self.add_input("water_depth", val=0.0, units="m")

        self.add_output("transition_piece_height", 0.0, units="m")
        self.add_output("z_start", 0.0, units="m")
        self.add_output("suctionpile_depth", 0.0, units="m")
        self.add_output("bending_height", 0.0, units="m")
        self.add_output("s_const1", 0.0)
        self.add_output("joint1", val=np.zeros(3), units="m")
        self.add_output("joint2", val=np.zeros(3), units="m")

    def compute(self, inputs, outputs):
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
        outputs["bending_height"] = h_mon - pile
        outputs["s_const1"] = pile / h_mon  # Ensure that we have only one segment for pile, a current limitation
        outputs["joint1"] = np.r_[0.0, 0.0, fh_tow - h_mon]
        outputs["joint2"] = np.r_[0.0, 0.0, fh_tow]


class MonopileMass(om.ExplicitComponent):
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
        self.options.declare("npts")

    def setup(self):
        npts = self.options["npts"]

        self.add_input("cylinder_mass", val=0.0, units="kg")
        self.add_input("cylinder_cost", val=0.0, units="USD")
        self.add_input("cylinder_z_cg", val=0.0, units="m")
        self.add_input("cylinder_I_base", np.zeros(6), units="kg*m**2")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_cost", 0.0, units="USD")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("z_full", val=np.zeros(npts), units="m")
        self.add_input("d_full", val=np.zeros(npts), units="m")
        self.add_input("tower_mass", val=0.0, units="kg")
        self.add_input("tower_cost", val=0.0, units="USD")

        self.add_output("monopile_mass", val=0.0, units="kg")
        self.add_output("monopile_cost", val=0.0, units="USD")
        self.add_output("monopile_z_cg", val=0.0, units="m")
        self.add_output("monopile_I_base", np.zeros(6), units="kg*m**2")
        self.add_output("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_output("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_output("structural_mass", val=0.0, units="kg")
        self.add_output("structural_cost", val=0.0, units="USD")

    def compute(self, inputs, outputs):
        # Unpack inputs
        z = inputs["z_full"]
        d = inputs["d_full"]
        m_trans = inputs["transition_piece_mass"]
        c_trans = inputs["transition_piece_cost"]
        m_grav = inputs["gravity_foundation_mass"]
        m_cyl = inputs["cylinder_mass"]
        c_cyl = inputs["cylinder_cost"]

        outputs["monopile_mass"] = m_mono = m_cyl + m_trans + m_grav
        outputs["monopile_cost"] = c_cyl + c_trans
        outputs["structural_mass"] = outputs["monopile_mass"] + inputs["tower_mass"]
        outputs["structural_cost"] = outputs["monopile_cost"] + inputs["tower_cost"]

        outputs["monopile_z_cg"] = (m_cyl * inputs["cylinder_z_cg"] + m_trans * z[-1] + m_grav * z[0]) / m_mono

        # Mass properties for transition piece and gravity foundation
        r_trans = 0.5 * d[-1]
        r_grav = 0.5 * d[0]
        I_trans = m_trans * r_trans ** 2.0 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]  # shell
        I_grav = m_grav * r_grav ** 2.0 * np.r_[0.25, 0.25, 0.5, np.zeros(3)]  # disk
        outputs["transition_piece_I"] = I_trans
        outputs["gravity_foundation_I"] = I_grav

        I_mono = util.assembleI(inputs["cylinder_I_base"] + I_grav)
        R = np.array([0.0, 0.0, z[-1] - z[0]])
        I_mono += util.assembleI(I_trans) + m_trans * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["monopile_I_base"] = util.unassembleI(I_mono)


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
    turbine_F : numpy array[3], [N]
        Total turbine force at tower base / transition piece
    turbine_M : numpy array[3], [N*m]
        Total turbine moment at tower base / transition piece
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
    monopile_deflection : numpy array[n_full], [m]
        Deflection of tower nodes in yaw-aligned +x direction
    top_deflection : float, [m]
        Deflection of tower top in yaw-aligned +x direction
    monopile_Fz : numpy array[n_full-1], [N]
        Axial foce in vertical z-direction in cylinder structure.
    monopile_Vx : numpy array[n_full-1], [N]
        Shear force in x-direction in cylinder structure.
    monopile_Vy : numpy array[n_full-1], [N]
        Shear force in y-direction in cylinder structure.
    monopile_Mxx : numpy array[n_full-1], [N*m]
        Moment about x-axis in cylinder structure.
    monopile_Myy : numpy array[n_full-1], [N*m]
        Moment about y-axis in cylinder structure.
    monopile_Mzz : numpy array[n_full-1], [N*m]
        Moment about z-axis in cylinder structure.
    base_F : numpy array[3], [N]
        Total force on cylinder
    base_M : numpy array[3], [N*m]
        Total moment on cylinder measured at base
    """

    def initialize(self):
        self.options.declare("n_full")
        self.options.declare("nLC")
        self.options.declare("frame3dd_opt")
        self.options.declare("soil_springs", default=False)
        self.options.declare("gravity_foundation", default=False)

    def setup(self):
        n_full = self.options["n_full"]
        nLC = self.options["nLC"]
        self.frame = None

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
        self.add_input("section_J0", np.zeros(n_full - 1), units="kg*m**2")
        self.add_input("section_rho", np.zeros(n_full - 1), units="kg/m**3")
        self.add_input("section_E", np.zeros(n_full - 1), units="Pa")
        self.add_input("section_G", np.zeros(n_full - 1), units="Pa")
        self.add_output("section_L", np.zeros(n_full - 1), units="m")

        # point loads
        self.add_input("turbine_F", np.zeros((3, nLC)), units="N")
        self.add_input("turbine_M", np.zeros((3, nLC)), units="N*m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_input("transition_piece_height", 0.0, units="m")
        self.add_input("suctionpile_depth", 0.0, units="m")

        # For modal analysis only (loads captured in turbine_F & turbine_M)
        self.add_input("turbine_mass", val=0.0, units="kg")
        self.add_input("turbine_cg", val=np.zeros(3), units="m")
        self.add_input("turbine_I", np.zeros(6), units="kg*m**2")

        # combined wind-water distributed loads
        self.add_input("Px", val=np.zeros((n_full, nLC)), units="N/m")
        self.add_input("Py", val=np.zeros((n_full, nLC)), units="N/m")
        self.add_input("Pz", val=np.zeros((n_full, nLC)), units="N/m")

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
        self.add_output("monopile_deflection", np.zeros((n_full, nLC)), units="m")
        self.add_output("top_deflection", np.zeros(nLC), units="m")
        self.add_output("monopile_Fz", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("monopile_Vx", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("monopile_Vy", val=np.zeros((n_full - 1, nLC)), units="N")
        self.add_output("monopile_Mxx", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("monopile_Myy", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("monopile_Mzz", val=np.zeros((n_full - 1, nLC)), units="N*m")
        self.add_output("mudline_F", val=np.zeros((3, nLC)), units="N")
        self.add_output("mudline_M", val=np.zeros((3, nLC)), units="N*m")

    def compute(self, inputs, outputs):

        frame3dd_opt = self.options["frame3dd_opt"]
        nLC = self.options["nLC"]

        # ------- node data ----------------
        xyz = inputs["nodes_xyz"]
        z = xyz[:, 2]
        n = xyz.shape[0]
        node = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node, xyz[:, 0], xyz[:, 1], z, r)
        # -----------------------------------

        # ------ reaction data ------------
        if not self.options["gravity_foundation"]:
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
                kidx = np.arange(len(z_pile), dtype=np.int_)
                kx = np.array([k_mono[:, 0]])
                ky = np.array([k_mono[:, 2]])
                kz = np.zeros(k_mono.shape[0])
                kz[0] = np.array([k_mono[0, 4]])
                ktx = np.array([k_mono[:, 1]])
                kty = np.array([k_mono[:, 3]])
                ktz = np.array([k_mono[:, 5]])

            else:
                z_pile = z[z <= (z[0] + 1e-1 + inputs["suctionpile_depth"])]
                npile = z_pile.size
                if npile != NREFINE + 1:
                    print(z)
                    print(z_pile)
                    print(inputs["suctionpile_depth"])
                    raise ValueError("Please use only one section for submerged pile for now")
                kidx = np.arange(npile, dtype=np.int_)
                kx = ky = kz = RIGID * np.ones(npile)
                ktx = kty = ktz = RIGID * np.ones(npile)

        else:
            kidx = np.array([0], dtype=np.int_)
            kx = ky = kz = np.array([RIGID])
            ktx = kty = ktz = np.array([RIGID])

        rnode = kidx + 1  # 1-based indexing
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
        outputs["section_L"] = L = np.sqrt(np.sum(np.diff(xyz, axis=0) ** 2, axis=1))

        elements = pyframe3dd.ElementData(element, N1, N2, Area, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)
        # -----------------------------------

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], dx)
        # -----------------------------------

        # initialize frame3dd object
        self.frame = pyframe3dd.Frame(nodes, reactions, elements, options)

        # ------- enable dynamic analysis ----------
        Mmethod = 1
        lump = 0
        shift = 0.0
        # Run twice the number of modes to ensure that we can ignore the torsional modes and still get the desired number of fore-aft, side-side modes
        self.frame.enableDynamics(2 * NFREQ, Mmethod, lump, frame3dd_opt["tol"], shift)
        # ----------------------------

        # ------ static load case 1 ------------
        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        for k in range(nLC):
            load = pyframe3dd.StaticLoadCase(gx, gy, gz)

            # Prepare point forces at transition node
            turb_F = inputs["turbine_F"][:, k]
            turb_M = inputs["turbine_M"][:, k]
            load.changePointLoads(
                np.array([n], dtype=np.int_),  # -1 b/c same reason as above
                np.array([turb_F[0]]).flatten(),
                np.array([turb_F[1]]).flatten(),
                np.array([turb_F[2]]).flatten(),
                np.array([turb_M[0]]).flatten(),
                np.array([turb_M[1]]).flatten(),
                np.array([turb_M[2]]).flatten(),
            )

            # distributed loads
            Px, Py, Pz = inputs["Pz"][:, k], inputs["Py"][:, k], -inputs["Px"][:, k]  # switch to local c.s.

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

            self.frame.addLoadCase(load)

        # ------ add extra mass ------------
        # Prepare transition piece, and gravity foundation (if any applicable) for "extra node mass"
        # Turbine mass added for modal analysis only- gravity loads accounted for in point force
        m_trans = float(inputs["transition_piece_mass"])
        I_trans = inputs["transition_piece_I"].flatten()
        m_grav = float(inputs["gravity_foundation_mass"])
        I_grav = inputs["gravity_foundation_I"].flatten()
        m_turb = float(inputs["turbine_mass"])
        cg_turb = inputs["turbine_cg"].flatten()
        I_turb = inputs["turbine_I"].flatten()
        # Note, need len()-1 because Frame3DD crashes if mass add at end
        midx = np.array([n - 1, n - 2, 1], dtype=np.int_)
        m_add = np.array([m_turb, m_trans, m_grav])
        mI = np.c_[I_turb, I_trans, I_grav]
        mrho = np.c_[cg_turb, np.zeros(3), np.zeros(3)]
        add_gravity = [False, True, True]
        self.frame.changeExtraNodeMass(
            midx,
            m_add,
            mI[0, :],
            mI[1, :],
            mI[2, :],
            mI[3, :],
            mI[4, :],
            mI[5, :],
            mrho[0, :],
            mrho[1, :],
            mrho[2, :],
            add_gravity,
        )

        # ------------------------------------
        # Debugging
        # self.frame.write('monopile_debug.3dd')
        # -----------------------------------
        # run the analysis
        displacements, forces, rxns, internalForces, mass, modal = self.frame.run()

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
        outputs["monopile_deflection"] = np.sqrt(displacements.dx ** 2 + displacements.dy ** 2).T
        outputs["top_deflection"] = outputs["monopile_deflection"][-1, :]

        # Record total forces and moments
        ibase = 2 * int(kidx.max())
        outputs["mudline_F"] = -np.c_[-forces.Vz[:, ibase], forces.Vy[:, ibase], forces.Nx[:, ibase]].T
        outputs["mudline_M"] = -np.c_[-forces.Mzz[:, ibase], forces.Myy[:, ibase], forces.Txx[:, ibase]].T

        Fz = np.zeros((len(forces.Nx[0, 1::2]), nLC))
        Vx = np.zeros(Fz.shape)
        Vy = np.zeros(Fz.shape)
        Mxx = np.zeros(Fz.shape)
        Myy = np.zeros(Fz.shape)
        Mzz = np.zeros(Fz.shape)
        for ic in range(nLC):
            # Forces and moments along the structure
            Fz[:, ic] = forces.Nx[ic, 1::2]
            Vx[:, ic] = -forces.Vz[ic, 1::2]
            Vy[:, ic] = forces.Vy[ic, 1::2]
            Mxx[:, ic] = -forces.Mzz[ic, 1::2]
            Myy[:, ic] = forces.Myy[ic, 1::2]
            Mzz[:, ic] = forces.Txx[ic, 1::2]
        outputs["monopile_Fz"] = Fz
        outputs["monopile_Vx"] = Vx
        outputs["monopile_Vy"] = Vy
        outputs["monopile_Mxx"] = Mxx
        outputs["monopile_Myy"] = Myy
        outputs["monopile_Mzz"] = Mzz


class MonopileSE(om.Group):
    """
    This is the main MonopileSE group that performs analysis of the monopile.

    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]["WISDEM"]["FixedBottomSE"]
        n_mat = self.options["modeling_options"]["materials"]["n_mat"]
        nLC = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        wind = mod_opt["wind"]  # not yet supported
        frame3dd_opt = mod_opt["frame3dd"]
        if "n_height" in mod_opt:
            n_height = mod_opt["n_height"]
        else:
            n_height_mono = mod_opt["n_height_monopile"]
            n_height = mod_opt["n_height"] = n_height_mono
        n_full = mem.get_nfull(n_height, nref=mod_opt["n_refine"])

        self.add_subsystem("predis", PreDiscretization(), promotes=["*"])

        promlist = [
            "E_mat",
            "G_mat",
            "sigma_y_mat",
            "sigma_ult_mat",
            "wohler_exp_mat",
            "wohler_A_mat",
            "rho_mat",
            "unit_cost_mat",
            "material_names",
            "painting_cost_rate",
            "labor_cost_rate",
            "z_param",
            "z_full",
            "s_full",
            "d_full",
            "t_full",
            "rho_full",
            "E_full",
            "G_full",
            "sigma_y_full",
            "outfitting_full",
            "s_const1",
            "s_const2",
            "joint1",
            "joint2",
            "outfitting_factor_in",
            "E_user",
            "constr_taper",
            "constr_d_to_t",
            "slope",
            "nodes_xyz",
            "section_A",
            "section_Asx",
            "section_Asy",
            "section_Ixx",
            "section_Iyy",
            "section_J0",
            "section_rho",
            "section_E",
            "section_G",
            "rho_water",
            ("s_in", "monopile_s"),
            ("layer_materials", "monopile_layer_materials"),
            ("layer_thickness", "monopile_layer_thickness"),
            ("section_height", "monopile_section_height"),
            ("outer_diameter_in", "monopile_outer_diameter_in"),
            ("outer_diameter", "monopile_outer_diameter"),
            ("wall_thickness", "monopile_wall_thickness"),
        ]
        if n_height > 2:
            promlist + ["thickness_slope"]

        temp_opt = mod_opt.copy()
        temp_opt["n_height"] = [n_height]
        temp_opt["n_layers"] = [mod_opt["n_layers"]]
        temp_opt["n_ballasts"] = [0]
        self.add_subsystem(
            "member",
            mem.MemberStandard(column_options=temp_opt, idx=0, n_mat=n_mat, n_refine=NREFINE),
            promotes=promlist,
        )

        self.add_subsystem(
            "mono",
            MonopileMass(npts=n_full),
            promotes=[
                "transition_piece_mass",
                "transition_piece_cost",
                "gravity_foundation_mass",
                "z_full",
                "d_full",
                "tower_mass",
                "tower_cost",
                "monopile_mass",
                "monopile_cost",
                "monopile_z_cg",
                "monopile_I_base",
                "transition_piece_I",
                "gravity_foundation_I",
                "structural_mass",
                "structural_cost",
            ],
        )

        self.connect("member.shell_mass", "mono.cylinder_mass")
        self.connect("member.shell_cost", "mono.cylinder_cost")
        self.connect("member.shell_z_cg", "mono.cylinder_z_cg")
        self.connect("member.shell_I_base", "mono.cylinder_I_base")

        if mod_opt["soil_springs"]:
            self.add_subsystem(
                "soil",
                TowerSoil(npts=NPTS_SOIL),
                promotes=[("G", "G_soil"), ("nu", "nu_soil"), ("depth", "suctionpile_depth")],
            )
            self.connect("d_full", "soil.d0", src_indices=[0])

        self.add_subsystem("loads", mem.MemberLoads(n_full=n_full, n_lc=nLC, wind=wind, hydro=True), promotes=["*"])

        self.add_subsystem(
            "monopile",
            MonopileFrame(
                n_full=n_full,
                frame3dd_opt=frame3dd_opt,
                soil_springs=mod_opt["soil_springs"],
                gravity_foundation=mod_opt["gravity_foundation"],
                nLC=nLC,
            ),
            promotes=[
                "nodes_xyz",
                "section_A",
                "section_Asx",
                "section_Asy",
                "section_Ixx",
                "section_Iyy",
                "section_J0",
                "section_rho",
                "section_E",
                "section_G",
                "transition_piece_height",
                "transition_piece_mass",
                "transition_piece_I",
                "gravity_foundation_mass",
                "gravity_foundation_I",
                "suctionpile_depth",
                "Px",
                "Py",
                "Pz",
                "turbine_mass",
                "turbine_cg",
                "turbine_I",
            ],
        )

        self.add_subsystem(
            "post",
            mem.CylinderPostFrame(modeling_options=mod_opt, n_dlc=nLC),
            promotes=[
                "z_full",
                "d_full",
                "t_full",
                "rho_full",
                "E_full",
                "G_full",
                "sigma_y_full",
                "bending_height",
                "qdyn",
            ],
        )

        if mod_opt["soil_springs"]:
            self.connect("soil.z_k", "monopile.z_soil")
            self.connect("soil.k", "monopile.k_soil")

        self.connect("monopile.monopile_Fz", "post.cylinder_Fz")
        self.connect("monopile.monopile_Vx", "post.cylinder_Vx")
        self.connect("monopile.monopile_Vy", "post.cylinder_Vy")
        self.connect("monopile.monopile_Mxx", "post.cylinder_Mxx")
        self.connect("monopile.monopile_Myy", "post.cylinder_Myy")
        self.connect("monopile.monopile_Mzz", "post.cylinder_Mzz")
