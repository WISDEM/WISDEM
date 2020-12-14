import numpy as np
import wisdem.ccblade._bem as _bem
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
from openmdao.api import Group, ExplicitComponent
from wisdem.rotorse import RPM2RS, RS2RPM
from wisdem.commonse import gravity
from wisdem.commonse.csystem import DirectionVector
from wisdem.ccblade.ccblade_component import AeroHubLoads, CCBladeLoads


class BladeCurvature(ExplicitComponent):
    # OpenMDAO component that computes the 3D curvature of the blade
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        n_span = self.options["modeling_options"]["RotorSE"]["n_span"]

        # Inputs
        self.add_input("r", val=np.zeros(n_span), units="m", desc="location in blade z-coordinate")
        self.add_input("precurve", val=np.zeros(n_span), units="m", desc="location in blade x-coordinate")
        self.add_input("presweep", val=np.zeros(n_span), units="m", desc="location in blade y-coordinate")
        self.add_input("precone", val=0.0, units="deg", desc="precone angle")

        # Outputs
        self.add_output(
            "3d_curv", val=np.zeros(n_span), units="deg", desc="total cone angle from precone and curvature"
        )
        self.add_output(
            "x_az", val=np.zeros(n_span), units="m", desc="location of blade in azimuth x-coordinate system"
        )
        self.add_output(
            "y_az", val=np.zeros(n_span), units="m", desc="location of blade in azimuth y-coordinate system"
        )
        self.add_output(
            "z_az", val=np.zeros(n_span), units="m", desc="location of blade in azimuth z-coordinate system"
        )
        self.add_output("s", val=np.zeros(n_span), units="m", desc="cumulative path length along blade")

    def compute(self, inputs, outputs):

        r = inputs["r"]
        precurve = inputs["precurve"]
        presweep = inputs["presweep"]
        precone = inputs["precone"]

        n = len(r)
        dx_dx = np.eye(3 * n)

        x_az, x_azd, y_az, y_azd, z_az, z_azd, cone, coned, s, sd = _bem.definecurvature_dv2(
            r, dx_dx[:, :n], precurve, dx_dx[:, n : 2 * n], presweep, dx_dx[:, 2 * n :], 0.0, np.zeros(3 * n)
        )

        totalCone = precone + np.degrees(cone)
        s = r[0] + s

        outputs["3d_curv"] = totalCone
        outputs["x_az"] = x_az
        outputs["y_az"] = y_az
        outputs["z_az"] = z_az
        outputs["s"] = s


class TotalLoads(ExplicitComponent):
    # OpenMDAO component that takes as input the rotor configuration (tilt, cone), the blade twist and mass distributions, and the blade aerodynamic loading, and computes the total loading including gravity and centrifugal forces
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        n_span = self.options["modeling_options"]["RotorSE"]["n_span"]

        # Inputs
        self.add_input("r", val=np.zeros(n_span), units="m", desc="radial positions along blade going toward tip")
        self.add_input(
            "aeroloads_Px", val=np.zeros(n_span), units="N/m", desc="distributed loads in blade-aligned x-direction"
        )
        self.add_input(
            "aeroloads_Py", val=np.zeros(n_span), units="N/m", desc="distributed loads in blade-aligned y-direction"
        )
        self.add_input(
            "aeroloads_Pz", val=np.zeros(n_span), units="N/m", desc="distributed loads in blade-aligned z-direction"
        )
        self.add_input("aeroloads_Omega", val=0.0, units="rpm", desc="rotor rotation speed")
        self.add_input("aeroloads_pitch", val=0.0, units="deg", desc="pitch angle")
        self.add_input("aeroloads_azimuth", val=0.0, units="deg", desc="azimuthal angle")
        self.add_input("theta", val=np.zeros(n_span), units="deg", desc="structural twist")
        self.add_input("tilt", val=0.0, units="deg", desc="tilt angle")
        self.add_input("3d_curv", val=np.zeros(n_span), units="deg", desc="total cone angle from precone and curvature")
        self.add_input("z_az", val=np.zeros(n_span), units="m", desc="location of blade in azimuth z-coordinate system")
        self.add_input("rhoA", val=np.zeros(n_span), units="kg/m", desc="mass per unit length")
        self.add_input(
            "dynamicFactor", val=1.0, desc="a dynamic amplification factor to adjust the static deflection calculation"
        )

        # Outputs
        self.add_output("Px_af", val=np.zeros(n_span), desc="total distributed loads in airfoil x-direction")
        self.add_output("Py_af", val=np.zeros(n_span), desc="total distributed loads in airfoil y-direction")
        self.add_output("Pz_af", val=np.zeros(n_span), desc="total distributed loads in airfoil z-direction")

    def compute(self, inputs, outputs):

        dynamicFactor = inputs["dynamicFactor"]
        r = inputs["r"]
        theta = inputs["theta"]
        tilt = inputs["tilt"]
        totalCone = inputs["3d_curv"]
        z_az = inputs["z_az"]
        rhoA = inputs["rhoA"]

        # keep all in blade c.s. then rotate all at end

        # --- aero loads ---
        P_a = DirectionVector(0, 0, 0)
        P_a.x, P_a.y, P_a.z = inputs["aeroloads_Px"], inputs["aeroloads_Py"], inputs["aeroloads_Pz"]

        # --- weight loads ---
        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -rhoA * gravity)
        P_w = weight.yawToHub(tilt).hubToAzimuth(inputs["aeroloads_azimuth"]).azimuthToBlade(totalCone)

        # --- centrifugal loads ---
        # azimuthal c.s.
        Omega = inputs["aeroloads_Omega"] * RPM2RS
        load = DirectionVector(0.0, 0.0, rhoA * Omega ** 2 * z_az)
        P_c = load.azimuthToBlade(totalCone)

        # --- total loads ---
        P = P_a + P_w + P_c

        # rotate to airfoil c.s.
        P = P.bladeToAirfoil(theta + inputs["aeroloads_pitch"])

        outputs["Px_af"] = dynamicFactor * P.x
        outputs["Py_af"] = dynamicFactor * P.y
        outputs["Pz_af"] = dynamicFactor * P.z


class RunFrame3DD(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("pbeam", default=False)  # Recover old pbeam c.s. and accuracy

    def setup(self):
        rotorse_options = self.options["modeling_options"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_freq = n_freq = rotorse_options["n_freq"]

        # Locations of airfoils in global c.s.
        self.add_input(
            "x_az", val=np.zeros(n_span), units="m", desc="location of blade in azimuth x-coordinate system (prebend)"
        )
        self.add_input(
            "y_az", val=np.zeros(n_span), units="m", desc="location of blade in azimuth y-coordinate system (sweep)"
        )
        self.add_input(
            "z_az",
            val=np.zeros(n_span),
            units="m",
            desc="location of blade in azimuth z-coordinate system (from root to tip)",
        )
        self.add_input("theta", val=np.zeros(n_span), units="deg", desc="structural twist")

        # all inputs/outputs in airfoil coordinate system
        self.add_input(
            "Px_af", val=np.zeros(n_span), desc="distributed load (force per unit length) in airfoil x-direction"
        )
        self.add_input(
            "Py_af", val=np.zeros(n_span), desc="distributed load (force per unit length) in airfoil y-direction"
        )
        self.add_input(
            "Pz_af", val=np.zeros(n_span), desc="distributed load (force per unit length) in airfoil z-direction"
        )

        self.add_input(
            "xu_strain_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_input(
            "xl_strain_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_input(
            "yu_strain_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_input(
            "yl_strain_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_input(
            "xu_strain_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_input(
            "xl_strain_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )
        self.add_input(
            "yu_strain_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_input(
            "yl_strain_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )

        self.add_input("r", val=np.zeros(n_span), units="m", desc="locations of properties along beam")
        self.add_input("A", val=np.zeros(n_span), units="m**2", desc="airfoil cross section material area")
        self.add_input("EA", val=np.zeros(n_span), units="N", desc="axial stiffness")
        self.add_input(
            "EIxx",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="edgewise stiffness (bending about :ref:`x-axis of airfoil aligned coordinate system <blade_airfoil_coord>`)",
        )
        self.add_input(
            "EIyy",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="flapwise stiffness (bending about y-axis of airfoil aligned coordinate system)",
        )
        self.add_input("EIxy", val=np.zeros(n_span), units="N*m**2", desc="coupled flap-edge stiffness")
        self.add_input(
            "GJ",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="torsional stiffness (about axial z-direction of airfoil aligned coordinate system)",
        )
        self.add_input("rhoA", val=np.zeros(n_span), units="kg/m", desc="mass per unit length")
        self.add_input("rhoJ", val=np.zeros(n_span), units="kg*m", desc="polar mass moment of inertia per unit length")
        self.add_input(
            "x_ec",
            val=np.zeros(n_span),
            units="m",
            desc="x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)",
        )
        self.add_input(
            "y_ec",
            val=np.zeros(n_span),
            units="m",
            desc="y-distance to elastic center from point about which above structural properties are computed",
        )

        # outputs
        n_freq2 = int(n_freq / 2)
        self.add_output("root_F", np.zeros(3), units="N", desc="Blade root forces in blade c.s.")
        self.add_output("root_M", np.zeros(3), units="N*m", desc="Blade root moment in blade c.s.")
        self.add_output(
            "flap_mode_shapes",
            np.zeros((n_freq2, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the flap direction (x^2..x^6, no linear or constant term)",
        )
        self.add_output(
            "edge_mode_shapes",
            np.zeros((n_freq2, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)",
        )
        self.add_output(
            "all_mode_shapes",
            np.zeros((n_freq, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the edge direction (x^2..x^6, no linear or constant term)",
        )
        self.add_output(
            "flap_mode_freqs",
            np.zeros(n_freq2),
            units="Hz",
            desc="Frequencies associated with mode shapes in the flap direction",
        )
        self.add_output(
            "edge_mode_freqs",
            np.zeros(n_freq2),
            units="Hz",
            desc="Frequencies associated with mode shapes in the edge direction",
        )
        self.add_output(
            "freqs",
            val=np.zeros(n_freq),
            units="Hz",
            desc="ration of 2nd and 1st natural frequencies, should be ratio of edgewise to flapwise",
        )
        self.add_output(
            "freq_distance",
            val=0.0,
            desc="ration of 2nd and 1st natural frequencies, should be ratio of edgewise to flapwise",
        )
        self.add_output(
            "dx", val=np.zeros(n_span), units="m", desc="deflection of blade section in airfoil x-direction"
        )
        self.add_output(
            "dy", val=np.zeros(n_span), units="m", desc="deflection of blade section in airfoil y-direction"
        )
        self.add_output(
            "dz", val=np.zeros(n_span), units="m", desc="deflection of blade section in airfoil z-direction"
        )
        self.add_output(
            "strainU_spar",
            val=np.zeros(n_span),
            desc="strain in spar cap on upper surface at location xu,yu_strain with loads P_strain",
        )
        self.add_output(
            "strainL_spar",
            val=np.zeros(n_span),
            desc="strain in spar cap on lower surface at location xl,yl_strain with loads P_strain",
        )
        self.add_output(
            "strainU_te",
            val=np.zeros(n_span),
            desc="strain in trailing-edge panels on upper surface at location xu,yu_te with loads P_te",
        )
        self.add_output(
            "strainL_te",
            val=np.zeros(n_span),
            desc="strain in trailing-edge panels on lower surface at location xl,yl_te with loads P_te",
        )

    def compute(self, inputs, outputs):

        # Unpack inputs
        r = inputs["r"]
        x_az = inputs["x_az"]
        y_az = inputs["y_az"]
        z_az = inputs["z_az"]
        theta = inputs["theta"]
        x_ec = inputs["x_ec"]
        y_ec = inputs["y_ec"]
        A = inputs["A"]
        rhoA = inputs["rhoA"]
        rhoJ = inputs["rhoJ"]
        GJ = inputs["GJ"]
        EA = inputs["EA"]
        EIxx = inputs["EIxx"]
        EIyy = inputs["EIyy"]
        EIxy = inputs["EIxy"]
        Px_af = inputs["Px_af"]
        Py_af = inputs["Py_af"]
        Pz_af = inputs["Pz_af"]
        xu_strain_spar = inputs["xu_strain_spar"]
        xl_strain_spar = inputs["xl_strain_spar"]
        yu_strain_spar = inputs["yu_strain_spar"]
        yl_strain_spar = inputs["yl_strain_spar"]
        xu_strain_te = inputs["xu_strain_te"]
        xl_strain_te = inputs["xl_strain_te"]
        yu_strain_te = inputs["yu_strain_te"]
        yl_strain_te = inputs["yl_strain_te"]
        # np.savez('nrel5mw_test.npz',r=r,x_az=x_az,y_az=y_az,z_az=z_az,theta=theta,x_ec=x_ec,y_ec=y_ec,A=A,rhoA=rhoA,rhoJ=rhoJ,GJ=GJ,EA=EA,EIxx=EIxx,EIyy=EIyy,EIxy=EIxy,Px_af=Px_af,Py_af=Py_af,Pz_af=Pz_af,xu_strain_spar=xu_strain_spar,xl_strain_spar=xl_strain_spar,yu_strain_spar=yu_strain_spar,yl_strain_spar=yl_strain_spar,xu_strain_te=xu_strain_te,xl_strain_te=xl_strain_te,yu_strain_te=yu_strain_te,yl_strain_te=yl_strain_te)

        # Determine principal C.S. (with swap of x, y for profile c.s.)
        # Can get to Hansen's c.s. from Precomp's c.s. by rotating around z -90 deg, then y by 180 (swap x-y)
        EIxx_cs, EIyy_cs = EIyy.copy(), EIxx.copy()
        x_ec_cs, y_ec_cs = y_ec.copy(), x_ec.copy()
        EIxy_cs = EIxy.copy()

        # translate to elastic center
        EIxx_cs -= y_ec_cs ** 2 * EA
        EIyy_cs -= x_ec_cs ** 2 * EA
        EIxy_cs -= x_ec_cs * y_ec_cs * EA

        # get rotation angle
        alpha = 0.5 * np.arctan(2 * EIxy_cs / (EIyy_cs - EIxx_cs))

        # get moments and positions in principal axes
        EI11 = EIxx_cs - EIxy_cs * np.tan(alpha)
        EI22 = EIyy_cs + EIxy_cs * np.tan(alpha)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        def rotate(x, y):
            x2 = x * ca + y * sa
            y2 = -x * sa + y * ca
            return x2, y2

        # Now store alpha for later use in degrees
        alpha = np.rad2deg(alpha)

        # Frame3dd call
        # ------- node data ----------------
        n = len(z_az)
        rad = np.zeros(n)  # 'radius' of rigidity at node- set to zero
        inode = 1 + np.arange(n)  # Node numbers (1-based indexing)
        if self.options["pbeam"]:
            nodes = pyframe3dd.NodeData(inode, np.zeros(n), np.zeros(n), r, rad)
            L = np.diff(r)
        else:
            nodes = pyframe3dd.NodeData(inode, x_az, y_az, z_az, rad)
            L = np.sqrt(np.diff(x_az) ** 2 + np.diff(y_az) ** 2 + np.diff(z_az) ** 2)
        # -----------------------------------

        # ------ reaction data ------------
        # Pinned at root
        rnode = np.array([1])
        rigid = np.array([1e16])
        reactions = pyframe3dd.ReactionData(rnode, rigid, rigid, rigid, rigid, rigid, rigid, float(rigid))
        # -----------------------------------

        # ------ frame element data ------------
        elem = np.arange(1, n)  # Element Numbers
        N1 = np.arange(1, n)  # Node number start
        N2 = np.arange(2, n + 1)  # Node number finish

        E = EA / A
        rho = rhoA / A
        J = rhoJ / rho
        G = GJ / J
        if self.options["pbeam"]:
            # Use airfoil c.s.
            Ix = EIyy / E
            Iy = EIxx / E
        else:
            # Will further rotate to principle axes
            Ix = EI11 / E
            Iy = EI22 / E

        # Have to convert nodal values to find average at center of element
        Abar, _ = util.nodal2sectional(A)
        Ebar, _ = util.nodal2sectional(E)
        rhobar, _ = util.nodal2sectional(rho)
        Jbar, _ = util.nodal2sectional(J)
        Gbar, _ = util.nodal2sectional(G)
        Ixbar, _ = util.nodal2sectional(Ix)
        Iybar, _ = util.nodal2sectional(Iy)

        # Angle of element principal axes relative to global coordinate system
        if self.options["pbeam"]:
            # Work in airfoil c.s. for both global and local c.s.
            roll = np.zeros(n - 1)
        else:
            # Global c.s. is blade, local element c.s. is airfoil (twist + principle rotation)
            roll, _ = util.nodal2sectional(theta + alpha)

        Asx = Asy = 1e-6 * np.ones(elem.shape)  # Unused when shear=False
        elements = pyframe3dd.ElementData(elem, N1, N2, Abar, Asx, Asy, Jbar, Ixbar, Iybar, Ebar, Gbar, roll, rhobar)
        # -----------------------------------

        # ------ options ------------
        shear = False  # If not false, have to compute Asx or Asy
        geom = not self.options["pbeam"]  # Must be true for spin-stiffening
        dx = -1  # Don't need stress changes within element for now
        options = pyframe3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frame3dd object
        blade = pyframe3dd.Frame(nodes, reactions, elements, options)

        # ------- enable dynamic analysis ----------
        Mmethod = 1  # 1= Subspace-Jacobi iteration, 2= Stodola (matrix iteration) method
        lump = 0  # 0= consistent mass matrix, 1= lumped mass matrix
        tol = 1e-9  # frequency convergence tolerance
        shift = 0.0  # frequency shift-factor for rigid body modes, make 0 for pos.def. [K]
        # Run twice the number of modes to ensure that we can ignore the torsional modes and still get the desired number of fore-aft, side-side modes
        blade.enableDynamics(2 * self.n_freq, Mmethod, lump, tol, shift)
        # ----------------------------

        # ------ load case 1, blade 1 ------------
        # trapezoidally distributed loads- already has gravity, centrifugal, aero, etc.
        gx = gy = gz = 0.0
        load = pyframe3dd.StaticLoadCase(gx, gy, gz)

        if not self.options["pbeam"]:
            # Have to further move the loads into principle directions
            P = DirectionVector(Px_af, Py_af, Pz_af).bladeToAirfoil(alpha)
            Px_af = P.x
            Py_af = P.y
            Pz_af = P.z

        Px, Py, Pz = Pz_af, Py_af, -Px_af  # switch to local c.s.
        xx1 = xy1 = xz1 = np.zeros(n - 1)
        xx2 = xy2 = xz2 = L - 1e-6  # subtract small number b.c. of precision
        wx1 = Px[:-1]
        wx2 = Px[1:]
        wy1 = Py[:-1]
        wy2 = Py[1:]
        wz1 = Pz[:-1]
        wz2 = Pz[1:]
        load.changeTrapezoidalLoads(elem, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)
        blade.addLoadCase(load)

        # Debugging
        # blade.write('blade.3dd')

        # run the analysis
        displacements, forces, reactions, internalForces, mass, modal = blade.run()

        # For now, just 1 load case and blade
        iCase = 0

        # Displacements in global (blade) c.s.
        dx = displacements.dx[iCase, :]
        dy = displacements.dy[iCase, :]
        dz = displacements.dz[iCase, :]

        # Mode shapes and frequencies
        n_freq2 = int(self.n_freq / 2)
        freq_x, freq_y, mshapes_x, mshapes_y = util.get_xy_mode_shapes(
            r, modal.freq, modal.xdsp, modal.ydsp, modal.zdsp, modal.xmpf, modal.ympf, modal.zmpf
        )
        freq_x = freq_x[:n_freq2]
        freq_y = freq_y[:n_freq2]
        mshapes_x = mshapes_x[:n_freq2, :]
        mshapes_y = mshapes_y[:n_freq2, :]

        # shear and bending, one per element (convert from local to global c.s.)
        Fz = np.r_[-forces.Nx[iCase, 0], forces.Nx[iCase, 1::2]]
        Vy = np.r_[-forces.Vy[iCase, 0], forces.Vy[iCase, 1::2]]
        Vx = np.r_[forces.Vz[iCase, 0], -forces.Vz[iCase, 1::2]]

        Tz = np.r_[-forces.Txx[iCase, 0], forces.Txx[iCase, 1::2]]
        My = np.r_[-forces.Myy[iCase, 0], forces.Myy[iCase, 1::2]]
        Mx = np.r_[forces.Mzz[iCase, 0], -forces.Mzz[iCase, 1::2]]

        def strain(xu, yu, xl, yl):
            # use profile c.s. to use Hansen's notation
            xuu, yuu = yu, xu
            xll, yll = yl, xl

            # convert to principal axes, unless already there
            if self.options["pbeam"]:
                M1, M2 = rotate(My, Mx)
            else:
                M1, M2 = My, Mx

            # compute strain
            x, y = rotate(xuu, yuu)
            strainU = -(
                M1 / EI11 * y - M2 / EI22 * x + Fz / EA
            )  # negative sign because Hansen c3 is opposite of Precomp z

            x, y = rotate(xll, yll)
            strainL = -(M1 / EI11 * y - M2 / EI22 * x + Fz / EA)

            return strainU, strainL

        # ----- strain -----
        strainU_spar, strainL_spar = strain(xu_strain_spar, yu_strain_spar, xl_strain_spar, yl_strain_spar)
        strainU_te, strainL_te = strain(xu_strain_te, yu_strain_te, xl_strain_te, yl_strain_te)

        # Store outputs
        outputs["root_F"] = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        outputs["root_M"] = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])
        outputs["freqs"] = modal.freq[: self.n_freq]
        outputs["edge_mode_shapes"] = mshapes_y
        outputs["flap_mode_shapes"] = mshapes_x
        # Dense numpy command that interleaves and alternates flap and edge modes
        outputs["all_mode_shapes"] = np.c_[mshapes_x, mshapes_y].flatten().reshape((self.n_freq, 5))
        outputs["edge_mode_freqs"] = freq_y
        outputs["flap_mode_freqs"] = freq_x
        outputs["freq_distance"] = freq_y[0] / freq_x[0]
        outputs["dx"] = dx
        outputs["dy"] = dy
        outputs["dz"] = dz
        outputs["strainU_spar"] = strainU_spar
        outputs["strainL_spar"] = strainL_spar
        outputs["strainU_te"] = strainU_te
        outputs["strainL_te"] = strainL_te


class TipDeflection(ExplicitComponent):
    # OpenMDAO component that computes the blade deflection at tip in yaw x-direction
    def setup(self):
        # Inputs
        self.add_input("dx_tip", val=0.0, units="m", desc="deflection at tip in blade x-direction")
        self.add_input("dy_tip", val=0.0, units="m", desc="deflection at tip in blade y-direction")
        self.add_input("dz_tip", val=0.0, units="m", desc="deflection at tip in blade z-direction")
        # self.add_input('theta_tip',     val=0.0,    units='deg',    desc='twist at tip section')
        self.add_input("pitch_load", val=0.0, units="deg", desc="blade pitch angle")
        self.add_input("tilt", val=0.0, units="deg", desc="tilt angle")
        self.add_input("3d_curv_tip", val=0.0, units="deg", desc="total coning angle including precone and curvature")
        self.add_input(
            "dynamicFactor", val=1.0, desc="a dynamic amplification factor to adjust the static deflection calculation"
        )  # )
        # Outputs
        self.add_output("tip_deflection", val=0.0, units="m", desc="deflection at tip in yaw x-direction")

    def compute(self, inputs, outputs):

        dx = inputs["dx_tip"]
        dy = inputs["dy_tip"]
        dz = inputs["dz_tip"]
        pitch = inputs["pitch_load"]  # + inputs['theta_tip']
        azimuth = 180.0  # The blade is assumed in front of the tower, although the loading may correspond to another azimuthal position
        tilt = inputs["tilt"]
        totalConeTip = inputs["3d_curv_tip"]
        dynamicFactor = inputs["dynamicFactor"]

        dr = DirectionVector(dx, dy, dz)

        delta = dr.airfoilToBlade(pitch).bladeToAzimuth(totalConeTip).azimuthToHub(azimuth).hubToYaw(tilt)

        outputs["tip_deflection"] = dynamicFactor * delta.x


class DesignConstraints(ExplicitComponent):
    # OpenMDAO component that formulates constraints on user-defined maximum strains, frequencies
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_freq = n_freq = rotorse_options["n_freq"]
        n_freq2 = int(n_freq / 2)
        self.opt_options = opt_options = self.options["opt_options"]
        self.n_opt_spar_cap_ss = n_opt_spar_cap_ss = opt_options["design_variables"]["blade"]["structure"][
            "spar_cap_ss"
        ]["n_opt"]
        self.n_opt_spar_cap_ps = n_opt_spar_cap_ps = opt_options["design_variables"]["blade"]["structure"][
            "spar_cap_ps"
        ]["n_opt"]
        # Inputs strains
        self.add_input(
            "strainU_spar",
            val=np.zeros(n_span),
            desc="strain in spar cap on upper surface at location xu,yu_strain with loads P_strain",
        )
        self.add_input(
            "strainL_spar",
            val=np.zeros(n_span),
            desc="strain in spar cap on lower surface at location xl,yl_strain with loads P_strain",
        )

        self.add_input("min_strainU_spar", val=0.0, desc="minimum strain in spar cap suction side")
        self.add_input("max_strainU_spar", val=0.0, desc="minimum strain in spar cap pressure side")
        self.add_input("min_strainL_spar", val=0.0, desc="maximum strain in spar cap suction side")
        self.add_input("max_strainL_spar", val=0.0, desc="maximum strain in spar cap pressure side")

        self.add_input(
            "s",
            val=np.zeros(n_span),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis (0-blade root, 1-blade tip)",
        )
        self.add_input(
            "s_opt_spar_cap_ss",
            val=np.zeros(n_opt_spar_cap_ss),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side",
        )
        self.add_input(
            "s_opt_spar_cap_ps",
            val=np.zeros(n_opt_spar_cap_ss),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap suction side",
        )

        # Input frequencies
        self.add_input("rated_Omega", val=0.0, units="rpm", desc="rotor rotation speed at rated")
        self.add_input(
            "flap_mode_freqs",
            np.zeros(n_freq2),
            units="Hz",
            desc="Frequencies associated with mode shapes in the flap direction",
        )
        self.add_input(
            "edge_mode_freqs",
            np.zeros(n_freq2),
            units="Hz",
            desc="Frequencies associated with mode shapes in the edge direction",
        )

        self.add_discrete_input("blade_number", 3)

        # Outputs
        # self.add_output('constr_min_strainU_spar',     val=np.zeros(n_opt_spar_cap_ss), desc='constraint for minimum strain in spar cap suction side')
        self.add_output(
            "constr_max_strainU_spar",
            val=np.zeros(n_opt_spar_cap_ss),
            desc="constraint for maximum strain in spar cap suction side",
        )
        # self.add_output('constr_min_strainL_spar',     val=np.zeros(n_opt_spar_cap_ps), desc='constraint for minimum strain in spar cap pressure side')
        self.add_output(
            "constr_max_strainL_spar",
            val=np.zeros(n_opt_spar_cap_ps),
            desc="constraint for maximum strain in spar cap pressure side",
        )
        self.add_output(
            "constr_flap_f_margin",
            val=np.zeros(n_freq2),
            desc="constraint on flap blade frequency such that ratio of 3P/f is above or below gamma with constraint <= 0",
        )
        self.add_output(
            "constr_edge_f_margin",
            val=np.zeros(n_freq2),
            desc="constraint on edge blade frequency such that ratio of 3P/f is above or below gamma with constraint <= 0",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Constraints on blade strains
        s = inputs["s"]
        s_opt_spar_cap_ss = inputs["s_opt_spar_cap_ss"]
        s_opt_spar_cap_ps = inputs["s_opt_spar_cap_ps"]

        strainU_spar = inputs["strainU_spar"]
        strainL_spar = inputs["strainL_spar"]
        # min_strainU_spar = inputs['min_strainU_spar']
        if inputs["max_strainU_spar"] == np.zeros_like(inputs["max_strainU_spar"]):
            max_strainU_spar = np.ones_like(inputs["max_strainU_spar"])
        else:
            max_strainU_spar = inputs["max_strainU_spar"]
        # min_strainL_spar = inputs['min_strainL_spar']
        if inputs["max_strainL_spar"] == np.zeros_like(inputs["max_strainL_spar"]):
            max_strainL_spar = np.ones_like(inputs["max_strainL_spar"])
        else:
            max_strainL_spar = inputs["max_strainL_spar"]

        # outputs['constr_min_strainU_spar'] = abs(np.interp(s_opt_spar_cap_ss, s, strainU_spar)) / abs(min_strainU_spar)
        outputs["constr_max_strainU_spar"] = abs(np.interp(s_opt_spar_cap_ss, s, strainU_spar)) / max_strainU_spar
        # outputs['constr_min_strainL_spar'] = abs(np.interp(s_opt_spar_cap_ps, s, strainL_spar)) / abs(min_strainL_spar)
        outputs["constr_max_strainL_spar"] = abs(np.interp(s_opt_spar_cap_ps, s, strainL_spar)) / max_strainL_spar

        # Constraints on blade frequencies
        threeP = discrete_inputs["blade_number"] * inputs["rated_Omega"] / 60.0
        flap_f = inputs["flap_mode_freqs"]
        edge_f = inputs["edge_mode_freqs"]
        gamma = self.options["modeling_options"]["RotorSE"]["gamma_freq"]
        outputs["constr_flap_f_margin"] = np.array(
            [min([threeP - (2 - gamma) * f, gamma * f - threeP]) for f in flap_f]
        ).flatten()
        outputs["constr_edge_f_margin"] = np.array(
            [min([threeP - (2 - gamma) * f, gamma * f - threeP]) for f in edge_f]
        ).flatten()


# class BladeFatigue(ExplicitComponent):
#     # OpenMDAO component that calculates the Miner's Rule cummulative fatigue damage, given precalculated rainflow counting of bending moments

#     def initialize(self):
#         self.options.declare('modeling_options')
#         self.options.declare('opt_options')


#     def setup(self):
#         rotorse_options   = self.options['modeling_options']['RotorSE']
#         mat_init_options     = self.options['modeling_options']['materials']

#         self.n_span          = n_span   = rotorse_options['n_span']
#         self.n_mat           = n_mat    = mat_init_options['n_mat']
#         self.n_layers        = n_layers = rotorse_options['n_layers']
#         self.FatigueFile     = self.options['modeling_options']['rotorse']['FatigueFile']

#         self.te_ss_var       = self.options['opt_options']['blade_struct']['te_ss_var']
#         self.te_ps_var       = self.options['opt_options']['blade_struct']['te_ps_var']
#         self.spar_cap_ss_var = self.options['opt_options']['blade_struct']['spar_cap_ss_var']
#         self.spar_cap_ps_var = self.options['opt_options']['blade_struct']['spar_cap_ps_var']

#         self.add_input('r',            val=np.zeros(n_span), units='m',      desc='radial locations where blade is defined (should be increasing and not go all the way to hub or tip)')
#         self.add_input('chord',        val=np.zeros(n_span), units='m',      desc='chord length at each section')
#         self.add_input('pitch_axis',   val=np.zeros(n_span),                 desc='1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.')
#         self.add_input('rthick',       val=np.zeros(n_span),                 desc='relative thickness of airfoil distribution')

#         self.add_input('gamma_f',      val=1.35,                             desc='safety factor on loads')
#         self.add_input('gamma_m',      val=1.1,                              desc='safety factor on materials')
#         self.add_input('E',            val=np.zeros([n_mat, 3]), units='Pa', desc='2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.')
#         self.add_input('Xt',           val=np.zeros([n_mat, 3]),             desc='2D array of the Ultimate Tensile Strength (UTS) of the materials. Each row represents a material, the three columns represent Xt12, Xt13 and Xt23.')
#         self.add_input('Xc',           val=np.zeros([n_mat, 3]),             desc='2D array of the Ultimate Compressive Strength (UCS) of the materials. Each row represents a material, the three columns represent Xc12, Xc13 and Xc23.')
#         self.add_input('m',            val=np.zeros([n_mat]),                desc='2D array of the S-N fatigue slope exponent for the materials')

#         self.add_input('x_tc',         val=np.zeros(n_span), units='m',      desc='x-distance to the neutral axis (torsion center)')
#         self.add_input('y_tc',         val=np.zeros(n_span), units='m',      desc='y-distance to the neutral axis (torsion center)')
#         self.add_input('EIxx',         val=np.zeros(n_span), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
#         self.add_input('EIyy',         val=np.zeros(n_span), units='N*m**2', desc='flapwise stiffness (bending about y-direction of airfoil aligned coordinate system)')

#         self.add_input('sc_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
#         self.add_input('sc_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="spar cap, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
#         self.add_input('te_ss_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")
#         self.add_input('te_ps_mats',   val=np.zeros((n_span, n_mat)),        desc="trailing edge reinforcement, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis")

#         self.add_discrete_input('layer_web',        val=n_layers * [''],     desc='1D array of the names of the webs the layer is associated to. If the layer is on the outer profile this entry can simply stay empty.')
#         self.add_discrete_input('layer_name',       val=n_layers * [''],     desc='1D array of the names of the layers modeled in the blade structure.')
#         self.add_discrete_input('layer_mat',        val=n_layers * [''],     desc='1D array of the names of the materials of each layer modeled in the blade structure.')
#         self.add_discrete_input('definition_layer', val=np.zeros(n_layers),  desc='1D array of flags identifying how layers are specified in the yaml. 1) all around (skin, paint, ) 2) offset+rotation twist+width (spar caps) 3) offset+user defined rotation+width 4) midpoint TE+width (TE reinf) 5) midpoint LE+width (LE reinf) 6) layer position fixed to other layer (core fillers) 7) start and width 8) end and width 9) start and end nd 10) web layer')

#         self.add_output('C_miners_SC_SS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Spar Cap, suction side")
#         self.add_output('C_miners_SC_PS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Spar Cap, pressure side")
#         self.add_output('C_miners_TE_SS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Trailing-Edge reinforcement, suction side")
#         self.add_output('C_miners_TE_PS',           val=np.zeros((n_span, n_mat, 2)),    desc="Miner's rule cummulative damage to Trailing-Edge reinforcement, pressure side")

#     def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

#         rainflow = load_yaml(self.FatigueFile, package=1)

#         U       = list(rainflow['cases'].keys())
#         Seeds   = list(rainflow['cases'][U[0]].keys())
#         chans   = list(rainflow['cases'][U[0]][Seeds[0]].keys())
#         r_gage  = np.r_[0., rainflow['r_gage']]
#         simtime = rainflow['simtime']
#         n_seeds = float(len(Seeds))
#         n_gage  = len(r_gage)

#         r       = (inputs['r']-inputs['r'][0])/(inputs['r'][-1]-inputs['r'][0])
#         m_default = 10. # assume default m=10  (8 or 12 also reasonable)
#         m       = [mi if mi > 0. else m_default for mi in inputs['m']]  # Assumption: if no S-N slope is given for a material, use default value TODO: input['m'] is not connected, only using the default currently

#         eps_uts = inputs['Xt'][:,0]/inputs['E'][:,0]
#         eps_ucs = inputs['Xc'][:,0]/inputs['E'][:,0]
#         gamma_m = inputs['gamma_m']
#         gamma_f = inputs['gamma_f']
#         yrs     = 20.  # TODO
#         t_life  = 60.*60.*24*365.24*yrs
#         U_bar   = 10.  # TODO

#         # pdf of wind speeds
#         binwidth = np.diff(U)
#         U_bins   = np.r_[[U[0] - binwidth[0]/2.], [np.mean([U[i-1], U[i]]) for i in range(1,len(U))], [U[-1] + binwidth[-1]/2.]]
#         pdf = np.diff(RayleighCDF(U_bins, xbar=U_bar))
#         if sum(pdf) < 0.9:
#             print('Warning: Cummulative probability of wind speeds in rotor_loads_defl_strains.BladeFatigue is low, sum of weights: %f' % sum(pdf))
#             print('Mean winds speed: %f' % U_bar)
#             print('Simulated wind speeds: ', U)

#         # Materials of analysis layers
#         te_ss_var_ok       = False
#         te_ps_var_ok       = False
#         spar_cap_ss_var_ok = False
#         spar_cap_ps_var_ok = False
#         for i_layer in range(self.n_layers):
#             if self.te_ss_var in discrete_inputs['layer_name']:
#                 te_ss_var_ok        = True
#             if self.te_ps_var in discrete_inputs['layer_name']:
#                 te_ps_var_ok        = True
#             if self.spar_cap_ss_var in discrete_inputs['layer_name']:
#                 spar_cap_ss_var_ok  = True
#             if self.spar_cap_ps_var in discrete_inputs['layer_name']:
#                 spar_cap_ps_var_ok  = True

#         if te_ss_var_ok == False:
#             print('The layer at the trailing edge suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ss_var)
#         if te_ps_var_ok == False:
#             print('The layer at the trailing edge pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.te_ps_var)
#         if spar_cap_ss_var_ok == False:
#             print('The layer at the spar cap suction side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ss_var)
#         if spar_cap_ps_var_ok == False:
#             print('The layer at the spar cap pressure side is set for Fatigue Analysis, but "%s" does not exist in the input yaml. Please check.'%self.spar_cap_ps_var)

#         # Get blade properties at gage locations
#         y_tc       = remap2grid(r, inputs['y_tc'], r_gage)
#         x_tc       = remap2grid(r, inputs['x_tc'], r_gage)
#         chord      = remap2grid(r, inputs['x_tc'], r_gage)
#         rthick     = remap2grid(r, inputs['rthick'], r_gage)
#         pitch_axis = remap2grid(r, inputs['pitch_axis'], r_gage)
#         EIyy       = remap2grid(r, inputs['EIyy'], r_gage)
#         EIxx       = remap2grid(r, inputs['EIxx'], r_gage)

#         te_ss_mats = np.floor(remap2grid(r, inputs['te_ss_mats'], r_gage, axis=0)) # materials is section
#         te_ps_mats = np.floor(remap2grid(r, inputs['te_ps_mats'], r_gage, axis=0))
#         sc_ss_mats = np.floor(remap2grid(r, inputs['sc_ss_mats'], r_gage, axis=0))
#         sc_ps_mats = np.floor(remap2grid(r, inputs['sc_ps_mats'], r_gage, axis=0))

#         c_TE       = chord*(1.-pitch_axis) + y_tc
#         c_SC       = chord*rthick + x_tc #this is overly simplistic, using maximum thickness point, should use the actual profiles

#         C_miners_SC_SS_gage = np.zeros((n_gage, self.n_mat, 2))
#         C_miners_SC_PS_gage = np.zeros((n_gage, self.n_mat, 2))
#         C_miners_TE_SS_gage = np.zeros((n_gage, self.n_mat, 2))
#         C_miners_TE_PS_gage = np.zeros((n_gage, self.n_mat, 2))

#         # Map channels to output matrix
#         chan_map   = {}
#         for i_var, var in enumerate(chans):
#             # Determine spanwise position
#             if 'Root' in var:
#                 i_span = 0
#             elif 'Spn' in var and 'M' in var:
#                 i_span = int(var.strip('Spn').split('M')[0])
#             else:
#                 # not a spanwise output channel, skip
#                 print('Fatigue Model: Skipping channel: %s, not a spanwise moment' % var)
#                 chans.remove(var)
#                 continue
#             # Determine if edgewise of flapwise moment
#             if 'M' in var and 'x' in var:
#                 # Edgewise
#                 axis = 0
#             elif 'M' in var and 'y' in var:
#                 # Flapwise
#                 axis = 1
#             else:
#                 # not an edgewise / flapwise moment, skip
#                 print('Fatigue Model: Skipping channel: "%s", not an edgewise/flapwise moment' % var)
#                 continue

#             chan_map[var] = {}
#             chan_map[var]['i_gage'] = i_span
#             chan_map[var]['axis']   = axis

#         # Map composite sections
#         composite_map = [['TE', 'SS', te_ss_var_ok],
#                          ['TE', 'PS', te_ps_var_ok],
#                          ['SC', 'SS', spar_cap_ss_var_ok],
#                          ['SC', 'PS', spar_cap_ps_var_ok]]

#         ########
#         # Loop through composite sections, materials, output channels, and simulations (wind speeds * seeds)
#         for comp_i in composite_map:

#             #skip this composite section?
#             if not comp_i[2]:
#                 continue

#             #
#             C_miners = np.zeros((n_gage, self.n_mat, 2))
#             if comp_i[0]       == 'TE':
#                 c = c_TE
#                 if comp_i[1]   == 'SS':
#                     mats = te_ss_mats
#                 elif comp_i[1] == 'PS':
#                     mats = te_ps_mats
#             elif comp_i[0]     == 'SC':
#                 c = c_SC
#                 if comp_i[1]   == 'SS':
#                     mats = sc_ss_mats
#                 elif comp_i[1] == 'PS':
#                     mats = sc_ps_mats

#             for i_mat in range(self.n_mat):

#                 for i_var, var in enumerate(chans):
#                     i_gage = chan_map[var]['i_gage']
#                     axis   = chan_map[var]['axis']

#                     # skip if material at this spanwise location is not included in the composite section
#                     if mats[i_gage, i_mat] == 0.:
#                         continue

#                     # Determine if edgewise of flapwise moment
#                     pitch_axis_i = pitch_axis[i_gage]
#                     chord_i      = chord[i_gage]
#                     c_i          = c[i_gage]
#                     if axis == 0:
#                         EI_i     = EIyy[i_gage]
#                     else:
#                         EI_i     = EIxx[i_gage]

#                     for i_u, u in enumerate(U):
#                         for i_s, seed in enumerate(Seeds):
#                             M_mean = np.array(rainflow['cases'][u][seed][var]['rf_mean']) * 1.e3
#                             M_amp  = np.array(rainflow['cases'][u][seed][var]['rf_amp']) * 1.e3

#                             for M_mean_i, M_amp_i in zip(M_mean, M_amp):
#                                 n_cycles = 1.
#                                 eps_mean = M_mean_i*c_i/EI_i
#                                 eps_amp  = M_amp_i*c_i/EI_i

#                                 Nf = ((eps_uts[i_mat] + np.abs(eps_ucs[i_mat]) - np.abs(2.*eps_mean*gamma_m*gamma_f - eps_uts[i_mat] + np.abs(eps_ucs[i_mat]))) / (2.*eps_amp*gamma_m*gamma_f))**m[i_mat]
#                                 n  = n_cycles * t_life * pdf[i_u] / (simtime * n_seeds)
#                                 C_miners[i_gage, i_mat, axis]  += n/Nf

#             # Assign outputs
#             if comp_i[0] == 'SC' and comp_i[1] == 'SS':
#                 outputs['C_miners_SC_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
#             elif comp_i[0] == 'SC' and comp_i[1] == 'PS':
#                 outputs['C_miners_SC_PS'] = remap2grid(r_gage, C_miners, r, axis=0)
#             elif comp_i[0] == 'TE' and comp_i[1] == 'SS':
#                 outputs['C_miners_TE_SS'] = remap2grid(r_gage, C_miners, r, axis=0)
#             elif comp_i[0] == 'TE' and comp_i[1] == 'PS':
#                 outputs['C_miners_TE_PS'] = remap2grid(r_gage, C_miners, r, axis=0)


class RotorStructure(Group):
    # OpenMDAO group to compute the blade elastic properties, deflections, and loading
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")
        self.options.declare("freq_run")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]
        freq_run = self.options["freq_run"]

        # Load blade with rated conditions and compute aerodynamic forces
        promoteListAeroLoads = [
            "r",
            "theta",
            "chord",
            "Rtip",
            "Rhub",
            "hub_height",
            "precone",
            "tilt",
            "airfoils_aoa",
            "airfoils_Re",
            "airfoils_cl",
            "airfoils_cd",
            "airfoils_cm",
            "nBlades",
            "rho",
            "mu",
            "Omega_load",
            "pitch_load",
        ]
        # self.add_subsystem('aero_rated',        CCBladeLoads(modeling_options = modeling_options), promotes=promoteListAeroLoads)

        self.add_subsystem("aero_gust", CCBladeLoads(modeling_options=modeling_options), promotes=promoteListAeroLoads)
        # self.add_subsystem('aero_storm_1yr',    CCBladeLoads(modeling_options = modeling_options), promotes=promoteListAeroLoads)
        # self.add_subsystem('aero_storm_50yr',   CCBladeLoads(modeling_options = modeling_options), promotes=promoteListAeroLoads)
        # Add centrifugal and gravity loading to aero loading
        promotes = ["tilt", "theta", "rhoA", "z", "totalCone", "z_az"]
        self.add_subsystem(
            "curvature",
            BladeCurvature(modeling_options=modeling_options),
            promotes=["r", "precone", "precurve", "presweep", "3d_curv", "x_az", "y_az", "z_az"],
        )
        promoteListTotalLoads = ["r", "theta", "tilt", "rhoA", "3d_curv", "z_az"]
        self.add_subsystem(
            "tot_loads_gust", TotalLoads(modeling_options=modeling_options), promotes=promoteListTotalLoads
        )
        # self.add_subsystem('tot_loads_rated',       TotalLoads(modeling_options = modeling_options),      promotes=promoteListTotalLoads)
        # self.add_subsystem('tot_loads_storm_1yr',   TotalLoads(modeling_options = modeling_options),      promotes=promoteListTotalLoads)
        # self.add_subsystem('tot_loads_storm_50yr',  TotalLoads(modeling_options = modeling_options),      promotes=promoteListTotalLoads)
        promoteListFrame3DD = [
            "x_az",
            "y_az",
            "z_az",
            "theta",
            "r",
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
            "xu_strain_spar",
            "xl_strain_spar",
            "yu_strain_spar",
            "yl_strain_spar",
            "xu_strain_te",
            "xl_strain_te",
            "yu_strain_te",
            "yl_strain_te",
        ]
        self.add_subsystem("frame", RunFrame3DD(modeling_options=modeling_options), promotes=promoteListFrame3DD)
        self.add_subsystem("tip_pos", TipDeflection(), promotes=["tilt", "pitch_load"])
        self.add_subsystem(
            "aero_hub_loads", AeroHubLoads(modeling_options=modeling_options), promotes=promoteListAeroLoads
        )
        self.add_subsystem("constr", DesignConstraints(modeling_options=modeling_options, opt_options=opt_options))

        # if modeling_options['rotorse']['FatigueMode'] > 0:
        #     promoteListFatigue = ['r', 'gamma_f', 'gamma_m', 'E', 'Xt', 'Xc', 'x_tc', 'y_tc', 'EIxx', 'EIyy', 'pitch_axis', 'chord', 'layer_name', 'layer_mat', 'definition_layer', 'sc_ss_mats','sc_ps_mats','te_ss_mats','te_ps_mats','rthick']
        #     self.add_subsystem('fatigue', BladeFatigue(modeling_options = modeling_options, opt_options = opt_options), promotes=promoteListFatigue)

        # Aero loads to total loads
        self.connect("aero_gust.loads_Px", "tot_loads_gust.aeroloads_Px")
        self.connect("aero_gust.loads_Py", "tot_loads_gust.aeroloads_Py")
        self.connect("aero_gust.loads_Pz", "tot_loads_gust.aeroloads_Pz")
        # self.connect('aero_rated.loads_Px',     'tot_loads_rated.aeroloads_Px')
        # self.connect('aero_rated.loads_Py',     'tot_loads_rated.aeroloads_Py')
        # self.connect('aero_rated.loads_Pz',     'tot_loads_rated.aeroloads_Pz')
        # self.connect('aero_storm_1yr.loads_Px', 'tot_loads_storm_1yr.aeroloads_Px')
        # self.connect('aero_storm_1yr.loads_Py', 'tot_loads_storm_1yr.aeroloads_Py')
        # self.connect('aero_storm_1yr.loads_Pz', 'tot_loads_storm_1yr.aeroloads_Pz')
        # self.connect('aero_storm_50yr.loads_Px', 'tot_loads_storm_50yr.aeroloads_Px')
        # self.connect('aero_storm_50yr.loads_Py', 'tot_loads_storm_50yr.aeroloads_Py')
        # self.connect('aero_storm_50yr.loads_Pz', 'tot_loads_storm_50yr.aeroloads_Pz')

        # Total loads to strains
        self.connect("tot_loads_gust.Px_af", "frame.Px_af")
        self.connect("tot_loads_gust.Py_af", "frame.Py_af")
        self.connect("tot_loads_gust.Pz_af", "frame.Pz_af")

        # Blade distributed deflections to tip deflection
        self.connect("frame.dx", "tip_pos.dx_tip", src_indices=[-1])
        self.connect("frame.dy", "tip_pos.dy_tip", src_indices=[-1])
        self.connect("frame.dz", "tip_pos.dz_tip", src_indices=[-1])
        self.connect("3d_curv", "tip_pos.3d_curv_tip", src_indices=[-1])

        # Strains from frame3dd to constraint
        self.connect("frame.strainU_spar", "constr.strainU_spar")
        self.connect("frame.strainL_spar", "constr.strainL_spar")
        self.connect("frame.flap_mode_freqs", "constr.flap_mode_freqs")
        self.connect("frame.edge_mode_freqs", "constr.edge_mode_freqs")
