import logging

import numpy as np
from openmdao.api import Group, ExplicitComponent
from scipy.interpolate import interp1d

import wisdem.ccblade._bem as _bem
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
from wisdem.rotorse import RPM2RS, RS2RPM
from wisdem.commonse import gravity
from wisdem.commonse.csystem import DirectionVector
from wisdem.ccblade.ccblade_component import CCBladeLoads, CCBladeEvaluate
from wisdem.rotorse.geometry_tools.geometry import remap2grid

logger = logging.getLogger("wisdem/weis")


class BladeCurvature(ExplicitComponent):
    # OpenMDAO component that computes the 3D curvature of the blade
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        n_span = self.options["modeling_options"]["WISDEM"]["RotorSE"]["n_span"]

        # Inputs
        self.add_input("r", val=np.zeros(n_span), units="m", desc="location in blade z-coordinate")
        self.add_input("precurve", val=np.zeros(n_span), units="m", desc="location in blade x-coordinate")
        self.add_input("presweep", val=np.zeros(n_span), units="m", desc="location in blade y-coordinate")
        self.add_input("precone", val=0.0, units="deg", desc="precone angle")
        self.add_input("Rhub", val=0.0, units="m", desc="hub radius")
        self.add_input("blade_span_cg", val=0.0, units="m", desc="Distance along the blade span for its center of gravity")

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
        self.add_output("blades_cg_hubcc", val=0.0, units="m", desc="cg of all blades relative to hub along shaft axis. Distance is should be interpreted as negative for upwind and positive for downwind turbines")

    def compute(self, inputs, outputs):
        r = inputs["r"]
        precurve = inputs["precurve"]
        presweep = inputs["presweep"]
        precone = inputs["precone"]
        r_cg = inputs["blade_span_cg"]
        Rhub = inputs["Rhub"]
        
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

        # Compute cg location of all blades in hub coordinates
        cone_cg = np.interp(r_cg, r, totalCone)
        cg = (r_cg + Rhub) * np.sin(np.deg2rad(cone_cg))
        outputs["blades_cg_hubcc"] = cg


class TotalBladeLoads(ExplicitComponent):
    # OpenMDAO component that takes as input the rotor configuration (tilt, cone), the blade twist and mass distributions, 
    # and the blade aerodynamic loading, and computes the blade loading including gravity and centrifugal forces
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        n_span = self.options["modeling_options"]["WISDEM"]["RotorSE"]["n_span"]

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
        self.add_output(
            "Px_af", val=np.zeros(n_span), units="N/m", desc="total distributed loads in airfoil x-direction"
        )
        self.add_output(
            "Py_af", val=np.zeros(n_span), units="N/m", desc="total distributed loads in airfoil y-direction"
        )
        self.add_output(
            "Pz_af", val=np.zeros(n_span), units="N/m", desc="total distributed loads in airfoil z-direction"
        )

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
        load = DirectionVector(0.0, 0.0, rhoA * Omega**2 * z_az)
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
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
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
            "Px_af",
            val=np.zeros(n_span),
            units="N/m",
            desc="distributed load (force per unit length) in airfoil x-direction",
        )
        self.add_input(
            "Py_af",
            val=np.zeros(n_span),
            units="N/m",
            desc="distributed load (force per unit length) in airfoil y-direction",
        )
        self.add_input(
            "Pz_af",
            val=np.zeros(n_span),
            units="N/m",
            desc="distributed load (force per unit length) in airfoil z-direction",
        )

        self.add_input("r", val=np.zeros(n_span), units="m", desc="locations of properties along beam")
        self.add_input("A", val=np.zeros(n_span), units="m**2", desc="airfoil cross section material area")
        self.add_input("EA", val=np.zeros(n_span), units="N", desc="axial stiffness")
        self.add_input(
            "EIxx",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section lag (edgewise) bending stiffness about the XE axis",
        )
        self.add_input(
            "EIyy",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section flap bending stiffness about the YE axis",
        )
        self.add_input("EIxy", val=np.zeros(n_span), units="N*m**2", desc="Coupled flap-lag stiffness with respect to the XE-YE frame")
        self.add_input(
            "GJ",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section torsional stiffness with respect to the XE-YE frame",
        )
        self.add_input("rhoA", val=np.zeros(n_span), units="kg/m", desc="mass per unit length")
        self.add_input("rhoJ", val=np.zeros(n_span), units="kg*m", desc="polar mass moment of inertia per unit length")

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
            "tors_mode_shapes",
            np.zeros((n_freq2, 5)),
            desc="6-degree polynomial coefficients of mode shapes in the torsional direction (x^2..x^6, no linear or constant term)",
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
            "tors_mode_freqs",
            np.zeros(n_freq2),
            units="Hz",
            desc="Frequencies associated with mode shapes in the torsional direction",
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
            "EI11",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="stiffness w.r.t principal axis 1",
        )
        self.add_output(
            "EI22",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="stiffness w.r.t principal axis 2",
        )
        self.add_output(
            "alpha",
            val=np.zeros(n_span),
            units="deg",
            desc="Angle between blade c.s. and principal axes",
        )
        self.add_output(
            "M1",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of bending moment w.r.t principal axis 1",
        )
        self.add_output(
            "M2",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of bending moment w.r.t principal axis 2",
        )
        self.add_output(
            "F2",
            val=np.zeros(n_span),
            units="N",
            desc="distribution along blade span of force w.r.t principal axis 2",
        )
        self.add_output(
            "F3",
            val=np.zeros(n_span),
            units="N",
            desc="axial resultant along blade span",
        )

    def compute(self, inputs, outputs):
        # Unpack inputs
        r = inputs["r"]
        x_az = inputs["x_az"]
        y_az = inputs["y_az"]
        z_az = inputs["z_az"]
        theta = inputs["theta"]
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
        #np.savez('nrel5mw_test.npz',
        #         r=r,x_az=x_az,y_az=y_az,z_az=z_az,theta=theta,x_ec=x_ec,y_ec=y_ec,A=A,rhoA=rhoA,
        #         rhoJ=rhoJ,GJ=GJ,EA=EA,EIxx=EIxx,EIyy=EIyy,EIxy=EIxy,Px_af=Px_af,Py_af=Py_af,Pz_af=Pz_af)

        # Determine principal C.S. (with swap of x, y for profile c.s.)
        # Can get to Hansen's c.s. from Precomp's c.s. by rotating around z -90 deg, then y by 180 (swap x-y)
        EIxx_cs, EIyy_cs = EIyy.copy(), EIxx.copy()
        EIxy_cs = EIxy.copy()

        # get rotation angle
        alpha = 0.5 * np.arctan2(2 * EIxy_cs, (EIyy_cs - EIxx_cs))

        # get moments and positions in principal axes
        EI11 = EIxx_cs - EIxy_cs * np.tan(alpha)
        EI22 = EIyy_cs + EIxy_cs * np.tan(alpha)

        # Now store alpha for later use in degrees
        alpha = np.rad2deg(alpha)

        # Frame3dd call
        # ------- node data ----------------
        n = len(z_az)
        rad = np.zeros(n)  # 'radius' of rigidity at node- set to zero
        inode = 1 + np.arange(n)  # Node numbers (1-based indexing)
        # Frame3DD does a coordinate rotation for the local axis and when x_az is negative for precurve, this makes the local axis
        # rotate relative to the global axis and we get inconsistent results.  Best to compute deflections on the reference axis (x=y=0)
        nodes = pyframe3dd.NodeData(inode, np.zeros(n), np.zeros(n), r, rad)
        L = np.diff(r)
        # if self.options["pbeam"]:
        #    nodes = pyframe3dd.NodeData(inode, np.zeros(n), np.zeros(n), r, rad)
        #    L = np.diff(r)
        # else:
        #    nodes = pyframe3dd.NodeData(inode, x_az, y_az, z_az, rad)
        #    L = np.sqrt(np.diff(x_az) ** 2 + np.diff(y_az) ** 2 + np.diff(z_az) ** 2)
        # -----------------------------------

        # ------ reaction data ------------
        # Pinned at root
        rnode = np.array([1])
        rigid = np.array([1e16])
        reactions = pyframe3dd.ReactionData(rnode, rigid, rigid, rigid, rigid, rigid, rigid, float(rigid[0]))
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

        Px, Py, Pz = Pz_af, Py_af, Px_af  # switch to local c.s.
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

        # Mode shapes and frequencies
        n_freq2 = int(self.n_freq / 2)
        freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
            r, modal.freq, modal.xdsp, modal.ydsp, modal.zdsp, modal.xmpf, modal.ympf, modal.zmpf,
        )
        freq_x = freq_x[:n_freq2]
        freq_y = freq_y[:n_freq2]
        freq_z = freq_z[:n_freq2]
        mshapes_x = mshapes_x[:n_freq2, :]
        mshapes_y = mshapes_y[:n_freq2, :]
        mshapes_z = mshapes_z[:n_freq2, :]

        # shear and bending w.r.t. principal axes
        F2 = np.r_[-forces.Vz[iCase, 0], forces.Vz[iCase, 1::2]]  # TODO verify if this is correct
        F3 = np.r_[-forces.Nx[iCase, 0], forces.Nx[iCase, 1::2]]
        M1 = np.r_[-forces.Myy[iCase, 0], forces.Myy[iCase, 1::2]]
        M2 = np.r_[-forces.Mzz[iCase, 0], forces.Mzz[iCase, 1::2]]

        # Store outputs
        outputs["root_F"] = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        outputs["root_M"] = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])
        outputs["freqs"] = modal.freq[: self.n_freq]
        outputs["edge_mode_shapes"] = mshapes_y
        outputs["flap_mode_shapes"] = mshapes_x
        outputs["tors_mode_shapes"] = mshapes_z
        # Dense numpy command that interleaves and alternates flap and edge modes
        outputs["all_mode_shapes"] = np.c_[mshapes_x, mshapes_y].flatten().reshape((self.n_freq, 5))
        outputs["edge_mode_freqs"] = freq_y
        outputs["flap_mode_freqs"] = freq_x
        outputs["tors_mode_freqs"] = freq_z
        outputs["freq_distance"] = freq_y[0] / freq_x[0]
        # Displacements in global (blade) c.s.
        outputs["dx"] = -displacements.dx[iCase, :]
        outputs["dy"] = displacements.dy[iCase, :]
        outputs["dz"] = -displacements.dz[iCase, :]
        outputs["EI11"] = EI11
        outputs["EI22"] = EI22
        outputs["M1"] = M1
        outputs["M2"] = M2
        outputs["F2"] = F2
        outputs["F3"] = F3
        outputs["alpha"] = alpha


class ComputeStrains(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("pbeam", default=False)  # Recover old pbeam c.s. and accuracy

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]

        self.add_input("chord", val=np.zeros(n_span), units="m", desc="chord length at each section")
        self.add_input("EA", val=np.zeros(n_span), units="N", desc="axial stiffness")
        self.add_input("A", val=np.zeros(n_span), units="m**2", desc="airfoil cross section material area")

        self.add_input(
            "EI11",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="stiffness w.r.t principal axis 1",
        )
        self.add_input(
            "EI22",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="stiffness w.r.t principal axis 2",
        )
        self.add_input(
            "alpha",
            val=np.zeros(n_span),
            units="deg",
            desc="Angle between blade c.s. and principal axes",
        )
        self.add_input(
            "M1",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of bending moment w.r.t principal axis 1",
        )
        self.add_input(
            "M2",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of bending moment w.r.t principal axis 2",
        )
        self.add_input(
            "F3",
            val=np.zeros(n_span),
            units="N",
            desc="axial resultant along blade span",
        )
        self.add_input(
            "xu_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_input(
            "xl_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_input(
            "yu_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_input(
            "yl_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_input(
            "xu_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_input(
            "xl_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )
        self.add_input(
            "yu_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_input(
            "yl_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )

        # outputs
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
        self.add_output(
            "axial_root_sparU_load2stress",
            val=np.zeros(6),
            units="m**2",
            desc="Linear conversion factors between loads [Fx-z; Mx-z] and axial stress in the upper spar cap at blade root",
        )
        self.add_output(
            "axial_root_sparL_load2stress",
            val=np.zeros(6),
            units="m**2",
            desc="Linear conversion factors between loads [Fx-z; Mx-z] and axial stress in the lower spar cap at blade root",
        )
        self.add_output(
            "axial_maxc_teU_load2stress",
            val=np.zeros(6),
            units="m**2",
            desc="Linear conversion factors between loads [Fx-z; Mx-z] and axial stress in the upper trailing edge at blade max chord",
        )
        self.add_output(
            "axial_maxc_teL_load2stress",
            val=np.zeros(6),
            units="m**2",
            desc="Linear conversion factors between loads [Fx-z; Mx-z] and axial stress in the lower trailing edge at blade max chord",
        )

    def compute(self, inputs, outputs):
        EA = inputs["EA"]
        A = inputs["A"]
        E = EA / A
        EI11 = inputs["EI11"]
        EI22 = inputs["EI22"]
        xu_spar = inputs["xu_spar"]
        xl_spar = inputs["xl_spar"]
        yu_spar = inputs["yu_spar"]
        yl_spar = inputs["yl_spar"]
        xu_te = inputs["xu_te"]
        xl_te = inputs["xl_te"]
        yu_te = inputs["yu_te"]
        yl_te = inputs["yl_te"]
        F3_principle = inputs["F3"]
        M1_principle = inputs["M1"]
        M2_principle = inputs["M2"]
        alpha = inputs["alpha"]
        n_sec = EA.size
        #np.savez('nrel5mw_test2.npz',
        #         EA=EA,EI11=EI11,EI22=EI22,xu_spar=xu_spar,xl_spar=xl_spar,yu_spar=yu_spar,yl_spar=yl_spar,
        #         xu_te=xu_te,xl_te=xl_te,yu_te=yu_te,yl_te=yl_te, F3=F3_principle, M1=M1_principle, M2=M2_principle, alpha=alpha)

        ca = np.cos(np.deg2rad(alpha))
        sa = np.sin(np.deg2rad(alpha))

        def rotate(x, y):
            x2 = x * ca + y * sa
            y2 = -x * sa + y * ca
            return x2, y2

        def strain(xu, yu, xl, yl, M1in=M1_principle, M2in=M2_principle, F3in=F3_principle):
            # use profile c.s. to use Hansen's notation
            xuu, yuu = yu, xu
            xll, yll = yl, xl

            # convert to principal axes, unless already there
            if self.options["pbeam"]:
                M1, M2 = rotate(M2in, M1in)
            else:
                M1, M2 = M1in, M2in

            # compute strain
            x, y = rotate(xuu, yuu)
            strainU = M1 / EI11 * y - M2 / EI22 * x - F3in / EA

            x, y = rotate(xll, yll)
            strainL = M1 / EI11 * y - M2 / EI22 * x - F3in / EA

            return strainU, strainL

        # ----- strains along the mid-line of the spar caps and at the center of the two trailing edge reinforcement thickness (not the trailing edge) -----
        strainU_spar, strainL_spar = strain(xu_spar, yu_spar, xl_spar, yl_spar)
        strainU_te, strainL_te = strain(xu_te, yu_te, xl_te, yl_te)

        outputs["strainU_spar"] = strainU_spar
        outputs["strainL_spar"] = strainL_spar
        outputs["strainU_te"] = strainU_te
        outputs["strainL_te"] = strainL_te

        # Sensitivities for fatigue calculation
        Espar = E  # Can update with rotor_elasticity later TODO
        Ete = E  # Can update with rotor_elasticity later TODO
        ax_sparU_load2stress = np.zeros((n_sec, 6))
        ax_sparL_load2stress = np.zeros((n_sec, 6))
        ax_teU_load2stress = np.zeros((n_sec, 6))
        ax_teL_load2stress = np.zeros((n_sec, 6))

        # Unit load response for Mxx
        Fz = np.zeros(M1_principle.shape)  # axial
        Mxx = np.ones(M1_principle.shape)  # edgewise
        Myy = np.zeros(M1_principle.shape)  # flapwise
        M1p, M2p = rotate(Myy, Mxx)
        strainU_spar_p, strainL_spar_p = strain(xu_spar, yu_spar, xl_spar, yl_spar, M1in=M1p, M2in=M2p, F3in=Fz)
        strainU_te_p, strainL_te_p = strain(xu_te, yu_te, xl_te, yl_te, M1in=M1p, M2in=M2p, F3in=Fz)
        ax_sparU_load2stress[:, 3] = Espar * strainU_spar_p
        ax_sparL_load2stress[:, 3] = Espar * strainL_spar_p
        ax_teU_load2stress[:, 3] = Ete * strainU_te_p
        ax_teL_load2stress[:, 3] = Ete * strainL_te_p

        # Unit load response for Myy
        Mxx = np.zeros(M1_principle.shape)  # edgewise
        Myy = np.ones(M1_principle.shape)  # flapwise
        M1p, M2p = rotate(Myy, Mxx)
        strainU_spar_p, strainL_spar_p = strain(xu_spar, yu_spar, xl_spar, yl_spar, M1in=M1p, M2in=M2p, F3in=Fz)
        strainU_te_p, strainL_te_p = strain(xu_te, yu_te, xl_te, yl_te, M1in=M1p, M2in=M2p, F3in=Fz)
        ax_sparU_load2stress[:, 4] = Espar * strainU_spar_p
        ax_sparL_load2stress[:, 4] = Espar * strainL_spar_p
        ax_teU_load2stress[:, 4] = Ete * strainU_te_p
        ax_teL_load2stress[:, 4] = Ete * strainL_te_p

        # Unit load response for Fzz
        Fz = np.ones(M1_principle.shape)  # axial
        Mxx = np.zeros(M1_principle.shape)  # edgewise
        Myy = np.zeros(M1_principle.shape)  # flapwise
        M1p, M2p = rotate(Myy, Mxx)
        strainU_spar_p, strainL_spar_p = strain(xu_spar, yu_spar, xl_spar, yl_spar, M1in=M1p, M2in=M2p, F3in=Fz)
        strainU_te_p, strainL_te_p = strain(xu_te, yu_te, xl_te, yl_te, M1in=M1p, M2in=M2p, F3in=Fz)
        ax_sparU_load2stress[:, 2] = Espar * strainU_spar_p
        ax_sparL_load2stress[:, 2] = Espar * strainL_spar_p
        ax_teU_load2stress[:, 2] = Ete * strainU_te_p
        ax_teL_load2stress[:, 2] = Ete * strainL_te_p

        imaxc = np.argmax(inputs["chord"])
        outputs["axial_root_sparU_load2stress"] = ax_sparU_load2stress[0, :]
        outputs["axial_root_sparL_load2stress"] = ax_sparL_load2stress[0, :]
        outputs["axial_maxc_teU_load2stress"] = ax_teU_load2stress[imaxc, :]
        outputs["axial_maxc_teL_load2stress"] = ax_teL_load2stress[imaxc, :]


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
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        n_span = rotorse_options["n_span"]
        n_freq = rotorse_options["n_freq"]
        n_freq2 = int(n_freq / 2)
        opt_options = self.options["opt_options"]
        spars_tereinf = rotorse_options["spars_tereinf"]
        n_opt_spar_cap_ss = opt_options["design_variables"]["blade"]["n_opt_struct"][spars_tereinf[0]]
        n_opt_spar_cap_ps = opt_options["design_variables"]["blade"]["n_opt_struct"][spars_tereinf[1]]
        n_opt_te_ss = opt_options["design_variables"]["blade"]["n_opt_struct"][spars_tereinf[2]]
        n_opt_te_ps = opt_options["design_variables"]["blade"]["n_opt_struct"][spars_tereinf[3]]

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
        self.add_input(
            "strainU_te",
            val=np.zeros(n_span),
            desc="strain in trailing edge on upper surface at location xu,yu_strain with loads P_strain",
        )
        self.add_input(
            "strainL_te",
            val=np.zeros(n_span),
            desc="strain in trailing edge on lower surface at location xl,yl_strain with loads P_strain",
        )

        # self.add_input("min_strainU_spar", val=0.0, desc="minimum strain in spar cap suction side")
        # self.add_input("min_strainL_spar", val=0.0, desc="minimum strain in spar cap pressure side")
        self.add_input("max_strainU_spar", val=1.0, desc="maximum strain in spar cap suction side")
        self.add_input("max_strainL_spar", val=1.0, desc="maximum strain in spar cap pressure side")
        self.add_input("max_strainU_te", val=1.0, desc="maximum strain in spar cap suction side")
        self.add_input("max_strainL_te", val=1.0, desc="maximum strain in spar cap pressure side")

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
            val=np.zeros(n_opt_spar_cap_ps),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade spar cap pressure side",
        )
        self.add_input(
            "s_opt_te_ss",
            val=np.zeros(n_opt_te_ss),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade trailing edge suction side",
        )
        self.add_input(
            "s_opt_te_ps",
            val=np.zeros(n_opt_te_ps),
            desc="1D array of the non-dimensional spanwise grid defined along blade axis to optimize the blade trailing edge pressure side",
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
        self.add_input(
            "tors_mode_freqs",
            np.zeros(n_freq2),
            units="Hz",
            desc="Frequencies associated with mode shapes in the torsional direction",
        )

        self.add_discrete_input("blade_number", 3)

        # Outputs
        # self.add_output('constr_min_strainU_spar',     val=np.zeros(n_opt_spar_cap_ss), desc='constraint for minimum strain in spar cap suction side')
        # self.add_output('constr_min_strainL_spar',     val=np.zeros(n_opt_spar_cap_ps), desc='constraint for minimum strain in spar cap pressure side')
        self.add_output(
            "constr_max_strainU_spar",
            val=np.zeros(n_opt_spar_cap_ss),
            desc="constraint for maximum strain in spar cap suction side",
        )
        self.add_output(
            "constr_max_strainL_spar",
            val=np.zeros(n_opt_spar_cap_ps),
            desc="constraint for maximum strain in spar cap pressure side",
        )
        self.add_output(
            "constr_max_strainU_te",
            val=np.zeros(n_opt_te_ss),
            desc="constraint for maximum strain in trailing edge suction side",
        )
        self.add_output(
            "constr_max_strainL_te",
            val=np.zeros(n_opt_te_ps),
            desc="constraint for maximum strain in trailing edge pressure side",
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
        s_opt_te_ss = inputs["s_opt_te_ss"]
        s_opt_te_ps = inputs["s_opt_te_ps"]

        strainU_spar = inputs["strainU_spar"]
        strainL_spar = inputs["strainL_spar"]
        strainU_te = inputs["strainU_te"]
        strainL_te = inputs["strainL_te"]

        max_strainU_spar = inputs["max_strainU_spar"]
        max_strainL_spar = inputs["max_strainL_spar"]
        max_strainU_te = inputs["max_strainU_te"]
        max_strainL_te = inputs["max_strainL_te"]

        # outputs['constr_min_strainU_spar'] = abs(np.interp(s_opt_spar_cap_ss, s, strainU_spar)) / abs(min_strainU_spar)
        # outputs['constr_min_strainL_spar'] = abs(np.interp(s_opt_spar_cap_ps, s, strainL_spar)) / abs(min_strainL_spar)
        outputs["constr_max_strainU_spar"] = abs(np.interp(s_opt_spar_cap_ss, s, strainU_spar)) / max_strainU_spar
        outputs["constr_max_strainL_spar"] = abs(np.interp(s_opt_spar_cap_ps, s, strainL_spar)) / max_strainL_spar
        outputs["constr_max_strainU_te"] = abs(np.interp(s_opt_te_ss, s, strainU_te)) / max_strainU_te
        outputs["constr_max_strainL_te"] = abs(np.interp(s_opt_te_ps, s, strainL_te)) / max_strainL_te

        # Constraints on blade frequencies
        threeP = discrete_inputs["blade_number"] * inputs["rated_Omega"] / 60.0
        flap_f = inputs["flap_mode_freqs"]
        edge_f = inputs["edge_mode_freqs"]
        tors_f = inputs["tors_mode_freqs"]
        gamma = self.options["modeling_options"]["WISDEM"]["RotorSE"]["gamma_freq"]
        outputs["constr_flap_f_margin"] = np.array(
            [min([threeP - (2 - gamma) * f, gamma * f - threeP]) for f in flap_f]
        ).flatten()
        outputs["constr_edge_f_margin"] = np.array(
            [min([threeP - (2 - gamma) * f, gamma * f - threeP]) for f in edge_f]
        ).flatten()


class BladeRootSizing(ExplicitComponent):
    """
    Compute the minimum blade root fastener circle diameter given the blade root moment

    Parameters
    ----------
    rootD : float, [m]
        Blade root outer diameter / Chord at blade span station 0
    layer_thickness : numpy array[n_layers, n_span], [m]
        Thickness of the blade structural layers along blade span
    layer_start_nd : numpy array[n_layers, n_span]
        Non-dimensional start point defined along the outer profile of a layer along blade span
    layer_end_nd : numpy array[n_layers, n_span]
        Non-dimensional end point defined along the outer profile of a layer along blade span
    root_M : numpy array[3], [N*m]
        Blade root moment in blade coordinate system
    s_f : float
        Safety factor on maximum stress per fastener
    d_f : float, [m]
        Diameter of the fastener
    sigma_max : float , [Pa]
        Maxmim stress per fastener

    Returns
    -------
    d_r : float , [m]
        Recommended diameter of the blade root fastener circle
    ratio : float
        Ratio of recommended diameter over actual diameter. It can be constrained to be smaller than 1

    """

    def initialize(self):
        self.options.declare("rotorse_options")

    def setup(self):
        rotorse_options = self.options["rotorse_options"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_layers = n_layers = rotorse_options["n_layers"]

        self.add_input("rootD", val=0.0, units="m", desc="Blade root outer diameter / Chord at blade span station 0")
        self.add_input(
            "layer_thickness",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_start_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_end_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input("root_M", val=np.zeros(3), units="N*m", desc="Blade root moment in blade c.s.")
        self.add_input("s_f", val=rotorse_options["root_fastener_s_f"], desc="Safety factor")
        self.add_input("d_f", val=0.0, units="m", desc="Diameter of the fastener")
        self.add_input("sigma_max", val=0.0, units="Pa", desc="Max stress on bolt")

        self.add_output("d_r", val=0.0, units="m", desc="Root fastener circle diameter")
        self.add_output(
            "ratio",
            val=0.0,
            desc="Ratio of recommended diameter over actual diameter. It can be constrained to be smaller than 1",
        )

    def compute(self, inputs, outputs):
        Mxy = np.sqrt(inputs["root_M"][0] ** 2.0 + inputs["root_M"][1] ** 2.0)

        d_r = np.sqrt((48.0 * Mxy * inputs["s_f"]) / (np.pi**2.0 * inputs["sigma_max"] * inputs["d_f"]))

        sectors = np.array([])
        for i in range(self.n_layers):
            sectors = np.unique(np.hstack([sectors, inputs["layer_start_nd"][i, 0], inputs["layer_end_nd"][i, 0]]))

        thick = np.zeros(len(sectors))
        for j in range(len(sectors)):
            for i in range(self.n_layers):
                if inputs["layer_start_nd"][i, 0] <= sectors[j] and inputs["layer_end_nd"][i, 0] >= sectors[j]:
                    thick[j] += inputs["layer_thickness"][i, 0]

        # check = np.all(thick == thick[0])
        # if not check:
        #     raise Exception('All Values in Array are not same')
        d_r_actual = inputs["rootD"] - 0.5 * thick[0]

        ratio = d_r / d_r_actual

        outputs["d_r"] = d_r
        outputs["ratio"] = ratio
     
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
            "precurve",
            "precurveTip",
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
            "shearExp",
            "hubloss",
            "tiploss",
            "wakerotation",
            "usecd",
            "yaw",
        ]
        # self.add_subsystem('aero_rated',        CCBladeLoads(modeling_options = modeling_options), promotes=promoteListAeroLoads)

        self.add_subsystem(
            "aero_gust", CCBladeLoads(modeling_options=modeling_options), promotes=promoteListAeroLoads + ["nSector"]
        )
        # self.add_subsystem('aero_storm_1yr',    CCBladeLoads(modeling_options = modeling_options), promotes=promoteListAeroLoads)
        # self.add_subsystem('aero_storm_50yr',   CCBladeLoads(modeling_options = modeling_options), promotes=promoteListAeroLoads)
        # Add centrifugal and gravity loading to aero loading
        #promotes = ["tilt", "theta", "rhoA", "z", "totalCone", "z_az"]
        self.add_subsystem(
            "curvature",
            BladeCurvature(modeling_options=modeling_options),
            promotes=["r", "precone", "precurve", "presweep", "Rhub", "blade_span_cg", "3d_curv", "x_az", "y_az", "z_az"],
        )
        promoteListTotalBladeLoads = ["r", "theta", "tilt", "rhoA", "3d_curv", "z_az"]
        self.add_subsystem(
            "tot_loads_gust", TotalBladeLoads(modeling_options=modeling_options), promotes=promoteListTotalBladeLoads
        )
        # self.add_subsystem('tot_loads_rated',       TotalBladeLoads(modeling_options = modeling_options),      promotes=promoteListTotalBladeLoads)
        # self.add_subsystem('tot_loads_storm_1yr',   TotalBladeLoads(modeling_options = modeling_options),      promotes=promoteListTotalBladeLoads)
        # self.add_subsystem('tot_loads_storm_50yr',  TotalBladeLoads(modeling_options = modeling_options),      promotes=promoteListTotalBladeLoads)
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
        ]
        self.add_subsystem("frame", RunFrame3DD(modeling_options=modeling_options), promotes=promoteListFrame3DD)
        promoteListStrains = [
            "chord",
            "EA",
            "A",
            "xu_spar",
            "xl_spar",
            "yu_spar",
            "yl_spar",
            "xu_te",
            "xl_te",
            "yu_te",
            "yl_te",
        ]

        self.add_subsystem("strains", ComputeStrains(modeling_options=modeling_options), promotes=promoteListStrains)
        self.add_subsystem("tip_pos", TipDeflection(), promotes=["tilt", "pitch_load"])
        self.add_subsystem(
            "aero_hub_loads",
            CCBladeEvaluate(modeling_options=modeling_options),
            promotes=promoteListAeroLoads + ["presweep", "presweepTip"],
        )
        if not modeling_options["user_elastic"]["blade"]:

            self.add_subsystem(
                "constr", DesignConstraints(modeling_options=modeling_options, opt_options=opt_options), promotes=["s"]
            )

            # Can't do sizing with user defined blade elastic properties
            self.add_subsystem("brs", BladeRootSizing(rotorse_options=modeling_options["WISDEM"]["RotorSE"]))


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

        # Total loads to bending moments
        self.connect("tot_loads_gust.Px_af", "frame.Px_af")
        self.connect("tot_loads_gust.Py_af", "frame.Py_af")
        self.connect("tot_loads_gust.Pz_af", "frame.Pz_af")

        # Moments to strains
        self.connect("frame.alpha", "strains.alpha")
        self.connect("frame.M1", "strains.M1")
        self.connect("frame.M2", "strains.M2")
        self.connect("frame.F3", "strains.F3")
        self.connect("frame.EI11", "strains.EI11")
        self.connect("frame.EI22", "strains.EI22")

        # Blade distributed deflections to tip deflection
        self.connect("frame.dx", "tip_pos.dx_tip", src_indices=[-1])
        self.connect("frame.dy", "tip_pos.dy_tip", src_indices=[-1])
        self.connect("frame.dz", "tip_pos.dz_tip", src_indices=[-1])
        self.connect("3d_curv", "tip_pos.3d_curv_tip", src_indices=[-1])

        # Strains from frame3dd to constraint
        if not modeling_options["user_elastic"]["blade"]:
            # TODO YL: no constr component, remove the above conditional when the constr component is enabled for user defined blade
            self.connect("strains.strainU_spar", "constr.strainU_spar")
            self.connect("strains.strainL_spar", "constr.strainL_spar")
            self.connect("strains.strainU_te", "constr.strainU_te")
            self.connect("strains.strainL_te", "constr.strainL_te")

            self.connect("frame.flap_mode_freqs", "constr.flap_mode_freqs")
            self.connect("frame.edge_mode_freqs", "constr.edge_mode_freqs")
            self.connect("frame.tors_mode_freqs", "constr.tors_mode_freqs")

            # Blade root moment to blade root sizing
            self.connect("frame.root_M", "brs.root_M")


def calc_axis_intersection(xy_coord, rotation, offset, p_le_d, side, thk=0.0):
    # dimentional analysis that takes a rotation and offset from the pitch axis and calculates the airfoil intersection
    # rotation
    offset_x = offset * np.cos(rotation) + p_le_d[0]
    offset_y = offset * np.sin(rotation) + p_le_d[1]

    m_rot = np.sin(rotation) / np.cos(rotation)  # slope of rotated axis
    plane_rot = [m_rot, -1 * m_rot * p_le_d[0] + p_le_d[1]]  # coefficients for rotated axis line: a1*x + a0

    m_intersection = np.sin(rotation + np.pi / 2.0) / np.cos(
        rotation + np.pi / 2.0
    )  # slope perpendicular to rotated axis
    plane_intersection = [
        m_intersection,
        -1 * m_intersection * offset_x + offset_y,
    ]  # coefficients for line perpendicular to rotated axis line at the offset: a1*x + a0

    # intersection between airfoil surface and the line perpendicular to the rotated/offset axis
    y_intersection = np.polyval(plane_intersection, xy_coord[:, 0])

    idx_le = np.argmin(xy_coord[:, 0])
    xy_coord_arc = util.arc_length(xy_coord)
    arc_L = xy_coord_arc[-1]
    xy_coord_arc /= arc_L

    idx_inter = np.argwhere(
        np.diff(np.sign(xy_coord[:, 1] - y_intersection))
    ).flatten()  # find closest airfoil surface points to intersection

    midpoint_arc = []
    for sidei in side:
        if sidei.lower() == "suction":
            tangent_line = np.polyfit(
                xy_coord[idx_inter[0] : idx_inter[0] + 2, 0], xy_coord[idx_inter[0] : idx_inter[0] + 2, 1], 1
            )
        elif sidei.lower() == "pressure":
            tangent_line = np.polyfit(
                xy_coord[idx_inter[1] : idx_inter[1] + 2, 0], xy_coord[idx_inter[1] : idx_inter[1] + 2, 1], 1
            )

        midpoint_x = (tangent_line[1] - plane_intersection[1]) / (plane_intersection[0] - tangent_line[0])
        midpoint_y = (
            plane_intersection[0]
            * (tangent_line[1] - plane_intersection[1])
            / (plane_intersection[0] - tangent_line[0])
            + plane_intersection[1]
        )

        # convert to arc position
        if sidei.lower() == "suction":
            x_half = xy_coord[: idx_le + 1, 0]
            arc_half = xy_coord_arc[: idx_le + 1]

        elif sidei.lower() == "pressure":
            x_half = xy_coord[idx_le:, 0]
            arc_half = xy_coord_arc[idx_le:]

        midpoint_arc.append(remap2grid(x_half, arc_half, midpoint_x, spline=interp1d))

    return midpoint_arc
