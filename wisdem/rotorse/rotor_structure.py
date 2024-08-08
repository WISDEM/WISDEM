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
        #np.savez('nrel5mw_test.npz',
        #         r=r,x_az=x_az,y_az=y_az,z_az=z_az,theta=theta,x_ec=x_ec,y_ec=y_ec,A=A,rhoA=rhoA,
        #         rhoJ=rhoJ,GJ=GJ,EA=EA,EIxx=EIxx,EIyy=EIyy,EIxy=EIxy,Px_af=Px_af,Py_af=Py_af,Pz_af=Pz_af)

        # Determine principal C.S. (with swap of x, y for profile c.s.)
        # Can get to Hansen's c.s. from Precomp's c.s. by rotating around z -90 deg, then y by 180 (swap x-y)
        EIxx_cs, EIyy_cs = EIyy.copy(), EIxx.copy()
        x_ec_cs, y_ec_cs = y_ec.copy(), x_ec.copy()
        EIxy_cs = EIxy.copy()

        # translate to elastic center
        EIxx_cs -= y_ec_cs**2 * EA
        EIyy_cs -= x_ec_cs**2 * EA
        EIxy_cs -= x_ec_cs * y_ec_cs * EA

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


class BladeJointSizing(ExplicitComponent):
    """
    Compute the minimum joint size given the blade loading.

    Parameters
    ----------
    load_factor : float
        Factor to multiply input loads by
    nprf : float
        Proof load safety factor for bolts. If this is too high, it seriously constrains the design. Keep it low, <= 1.2
    ny : float
        Yield safety factor for inserts.
    nf : float
        Fatigue safety factor for bolts/inserts.
    n0 : float
        Separation safety factor for joint. If this is too high, it seriously constrains the design. Keep it low, <= 1.2
    ns : float
        Shear safety factor for joint.
    nDLC61 : float
        Safety factor for extreme loads from IEC 61400 design load case 6.1. =1.35

    All other parameters described in setup()


    Returns
    -------
    L_transition_joint : float, [m]
        Required length to accommodate spar cap size increase at segmentation joint
    m_add_joint : float, [kg]
        Mass of bolts + inserts minus mass of spar cap cutouts for segmentation joint (spar cap size change will add mass too)
    n_joint_bolt : int
        Required number of bolts for segmentation joint
    t_joint : float, [m]
        Required thickness of the reinforcement layer at segmentation joint
    w_joint : float, [m]
        Required width of the reinforcement layer at segmentation joint

    """

    def initialize(self):
        self.options.declare("rotorse_options")
        self.options.declare("n_mat")

    def setup(self):
        # options
        n_mat = int(self.options["n_mat"])
        rotorse_options = self.options["rotorse_options"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.nd_span = rotorse_options["nd_span"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_xy = n_xy = rotorse_options["n_xy"]
        self.spar_cap_ss = rotorse_options["spar_cap_ss"]
        self.spar_cap_ps = rotorse_options["spar_cap_ps"]
        self.layer_name = rotorse_options["layer_name"]
        self.layer_mat = rotorse_options["layer_mat"]

        # safety factors. These will eventually come from the modeling yaml, but are currently hardcoded.
        self.add_input(
            "load_factor", val=1.35, desc="input load multiplier. 1.35 recommended for DLC6.1. [float]"
        )  # I find it better to change the load factor than the safety factors. It prevents limit errors (like divide by zero)
        self.add_input(
            "nprf", val=1.2, desc="bolt proof safety factor. [float]"
        )  # bolt ultimate safety factor (proof) [-]. If this is too high, it seriously constrains the design. Keep it low, <= 1.2
        self.add_input("ny", val=1.2, desc="bolt yield safety factor. [float]")
        self.add_input("ns", val=1.2, desc="bolt shear safety factor. [float]")
        self.add_input("n0", val=1.2, desc="bolt separation safety factor. [float]")
        self.add_input("nf", val=1.2, desc="bolt fatigue safety factor. [float]")

        # M48 10.9 bolt properties (Shigley p.433) # kf = 3  # metric 10.9, rolled thread. These are hardcoded.
        self.add_input("Sp_bolt", val=830e6, units="Pa", desc="bolt proof strength. [float]")
        self.add_input("Sy_bolt", val=940e6, units="Pa", desc="bolt yield strength. [float]")
        self.add_input("Su_bolt", val=1040e6, units="Pa", desc="bolt ultimate strength. [float]")
        self.add_input(
            "Se_bolt", val=162e6, units="Pa", desc="bolt endurance strength. [float]"
        )  # (for M1.6-36). Fully corrected so don't need kf
        self.add_input("E_bolt", val=200e9, units="Pa", desc="bolt elastic modulus. [float]")  # medium carbon steel
        self.add_input("d_bolt", val=0.048, units="m", desc="bolt diameter. [float]")
        self.add_input("L_bolt", val=1 / 2, units="m", desc="bolt length. [float]")
        self.add_input("At", val=1470 / 1e6, units="m**2", desc="bolt thread area [float]")
        self.add_input(
            "m_bolt", val=8.03, units="kg", desc="bolt+washer mass. [float]"
        )  # https://www.portlandbolt.com/technical/tools/bolt-weight-calculator

        # stainless steel insert material properties. These mostly come from the geometry yaml.
        # material: Stainless Steel 440A, tempered @315C. Cold worked 304: 515/860. (http://www.matweb.com/search/datasheet_print.aspx?matguid=4f9c4c71102d4cd9b4c52622588e99a0)
        self.add_discrete_input("name_mat", val=n_mat * [""], desc="list of material names. [(n_mat) list of str]")
        self.add_input(
            "rho_mat",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="list of material densities. [(n_mat) 1D array of str]",
        )
        self.add_input(
            "Xt_mat",
            val=np.zeros((n_mat, 3)),
            units="Pa",
            desc="list of material tensile strengths. [(n_mat x 3) 2D array of float]",
        )
        self.add_input(
            "Xy_mat",
            val=np.zeros(n_mat),
            units="Pa",
            desc="list of material yield strengths. [(n_mat x 3) 1D array of float]",
        )
        self.add_input(
            "E_mat",
            val=np.zeros((n_mat, 3)),
            units="Pa",
            desc="list of material elastic moduli. [(n_mat x 3) list of float]",
        )
        self.add_input("S_mat", val=np.zeros((n_mat, 3)), units="Pa")
        self.add_input(
            "unit_cost", val=np.zeros(n_mat), units="USD/kg", desc="1D array of the unit costs of the materials."
        )
        self.add_input(
            "Se_insert", val=316e6, units="Pa", desc="insert material endurance limit. [float]"
        )  # Azom (hardcoded)
        self.add_input(
            "mu_joint", val=0.5, desc="steel insert friction factor. [float]"
        )  # eng toolbox says 0.5-0.8 for clean, dry steel (hardcoded)

        # geometric properties
        self.add_input("joint_position", val=0, desc="nondimensionalized joint position along blade")
        self.add_discrete_input("joint_bolt", val="M48", desc="Type of bolt used in joint")
        self.add_discrete_input(
            "reinforcement_layer_ss",
            val="joint_reinf_ss",
            desc="Layer identifier for the reinforcement layer at the join where bolts are inserted, suction side",
        )
        self.add_discrete_input(
            "reinforcement_layer_ps",
            val="joint_reinf_ps",
            desc="Layer identifier for the reinforcement layer at the join where bolts are inserted, pressure side",
        )
        self.add_input(
            "bolt_spacing_dia", val=3, desc="joint bolt spacing along sparcap in y. [int]"
        )  # units: bolt diameters (hardcoded)
        self.add_input(
            "ply_drop_slope", val=1 / 8, desc="required ply drop slope. [float]"
        )  # max for >45 plies dropped (otherwise 1/3) https://www.sciencedirect.com/science/article/pii/S135983680000038X. (hardcoded)
        self.add_input(
            "t_adhesive", val=0.005, units="m", desc="insert-sparcap adhesive thickness. [float]"
        )  # (hardcoded)
        self.add_input(
            "t_max", val=1 / 4, desc="maximum spar cap thickness per spar cap height. [float]"
        )  # (hardcoded)
        self.add_input(
            "chord",
            val=np.zeros(n_span),
            units="m",
            desc="chord length at joint station. [(n_span x 3) 1D array of float]",
        )
        self.add_input(
            "layer_thickness",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "layer_width",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the width of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.",
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
        self.add_input(
            "blade_length",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of cumulative blade length at each station. n_span long",
        )
        self.add_input(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations.",
        )
        self.add_input(
            "rthick",
            val=np.zeros(n_span),
            desc="1D array of the relative thicknesses of the blade defined along span. n_span long",
        )
        self.add_input(
            "layer_offset_y_pa",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.",
        )
        self.add_input(
            "coord_xy_dim",
            val=np.zeros((n_span, n_xy, 2)),
            units="m",
            desc="3D array of the dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The origin is placed at the pitch axis.",
        )
        self.add_discrete_input(
            "layer_side",
            val=n_layers * [""],
            desc="1D array setting whether the layer is on the suction or pressure side. This entry is only used if definition_layer is equal to 1 or 2.",
        )
        self.add_input(
            "twist",
            val=np.zeros(n_span),
            units="rad",
            desc="1D array of the twist values defined along blade span. The twist is defined positive for negative rotations around the z axis (the same as in BeamDyn).",
        )
        self.add_input(
            "pitch_axis",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )
        # blade ultimate loads
        self.add_input(
            "M1",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of ultimate bending moment w.r.t principal axis 1. [(n_mat) 1D array of float]",
        )
        self.add_input(
            "M2",
            val=np.zeros(n_span),
            units="N*m",
            desc="distribution along blade span of ultimate bending moment w.r.t principal axis 2. [(n_mat) 1D array of float]",
        )
        self.add_input(
            "F2",
            val=np.zeros(n_span),
            units="N",
            desc="distribution along blade span of ultimate force w.r.t principal axis 2. [(n_mat) 1D array of float]",
        )

        # fatigue load calculating factors from https://iopscience.iop.org/article/10.1088/1742-6596/1618/4/042016
        self.add_input("a", val=0.1965)  # (hardcoded)
        self.add_input("b", val=0)  # (hardcoded)
        self.add_input("c", val=0.6448)  # (hardcoded)

        # other
        self.add_input("itermax", val=20, desc="max # calculation iterations. [int]")  # (hardcoded)
        self.add_input(
            "discrete",
            val=False,
            desc="whether discrete calculation is allowed. Set to False if in optimization loop. [bool]",
        )  # (hardcoded) TODO could add as user input
        self.add_input(
            "joint_nonmaterial_cost",
            val=0.0,
            units="USD",
            desc="Non-material joint cost (mfg, assembly, transportation).",
        )

        self.add_output(
            "t_reinf_joint", val=0, units="m", desc="Required reinforcement layer thickness at joint. [float]"
        )
        self.add_output("w_reinf_joint", val=0, units="m", desc="Required reinforcement layer width at joint. [float]")
        self.add_output(
            "w_reinf_ratio_joint", val=0, desc="Ratio of joint-required to nominal reinforcement layer  width"
        )
        self.add_output(
            "t_reinf_ratio_joint", val=0, desc="Ratio of joint-required to nominal reinforcement layer  thickness"
        )
        self.add_output(
            "L_transition_joint",
            val=0,
            units="m",
            desc="Required length to accommodate reinforcement layer  size increase at joint. [float]",
        )
        self.add_output("n_joint_bolt", val=0, desc="Required number of bolts for joint. [float]")
        self.add_output(
            "joint_mass",
            val=0,
            units="kg",
            desc="Mass of bolts + inserts minus mass of reinforcement layer  cutouts at joint. [float]",
        )
        # self.add_output("layer_end_nd_bjs", val=np.zeros((n_layers, n_span)),
        #                desc="2D array of the non-dimensional end point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.")
        # self.add_output("layer_start_nd_bjs", val=np.zeros((n_layers, n_span)),
        #                desc="2D array of the non-dimensional start point defined along the outer profile of a layer. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each layer, the second dimension represents each entry along blade span.")
        # self.add_output("layer_width_bjs", val=np.zeros((n_layers, n_span)), units="m",
        #                desc="2D array of the width of the layers of the blade structure. The first dimension represents each layer, the second dimension represents each entry along blade span.")
        # self.add_output("layer_offset_y_bjs", val=np.zeros((n_layers, n_span)), units="m",
        #            desc="2D array of the offset along the y axis to set the position of a layer. Positive values move the layer towards the trailing edge, negative values towards the leading edge. The first dimension represents each layer, the second dimension represents each entry along blade span.")

        self.add_output(
            "joint_material_cost", val=0, units="USD", desc="cost of joint metallic parts (bolts, washers, and inserts)"
        )
        self.add_output(
            "joint_total_cost",
            val=0,
            units="USD",
            desc="Total cost of the bolted joint (metallic parts and nonmaterial costs)",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # safety factors
        load_factor = inputs["load_factor"]
        nprf = inputs["nprf"]
        ny = inputs["ny"]
        nf = inputs["nf"]
        n0 = inputs["n0"]
        ns = inputs["ns"]

        # bolt parameters
        Sp_bolt = inputs["Sp_bolt"]
        Su_bolt = inputs["Su_bolt"]
        Se_bolt = inputs["Se_bolt"]
        E_bolt = inputs["E_bolt"]
        d_bolt = inputs["d_bolt"]
        L_bolt = inputs["L_bolt"]
        At = inputs["At"]
        m_bolt = inputs["m_bolt"]
        bolt = discrete_inputs["joint_bolt"]

        # material parameters
        name_mat = discrete_inputs["name_mat"]
        rho_mat = inputs["rho_mat"]
        Xt_mat = inputs["Xt_mat"]
        Xy_mat = inputs["Xy_mat"]
        E_mat = inputs["E_mat"]
        S_mat = inputs["S_mat"]
        unit_cost_mat = inputs["unit_cost"]
        reinf_mat_ss = discrete_inputs["reinforcement_layer_ss"]
        reinf_mat_ps = discrete_inputs["reinforcement_layer_ps"]
        if "joint_insert" in name_mat:
            insert_i = name_mat.index("joint_insert")
        else:
            raise Exception("Please add a material named joint_insert to the geometry yaml")
        rho_insert = rho_mat[insert_i]
        pu_cost_insert = unit_cost_mat[insert_i]
        adhesive_i = name_mat.index("Adhesive")
        rho_adhesive = rho_mat[adhesive_i]
        pu_cost_adhesive = unit_cost_mat[adhesive_i]
        Sy_insert = Xy_mat[insert_i]
        Su_insert = Xt_mat[insert_i, 0]
        Se_insert = inputs["Se_insert"]
        E_insert = E_mat[insert_i, 0]
        mu = inputs["mu_joint"]
        rho_ss = 0.0
        for i_lay in range(self.n_layers):
            if self.layer_name[i_lay] == reinf_mat_ss:
                mat_name_joint_ss = self.layer_mat[i_lay]
                ss_mat_i = name_mat.index(mat_name_joint_ss)
                S_ss = S_mat[ss_mat_i, 0]
                rho_ss = rho_mat[ss_mat_i]
                break

        if rho_ss == 0.0:
            raise Exception(
                "The joint reinforcement material ", reinf_mat_ss, " was not found! Check the entry geometry yaml."
            )

        # bolt properties
        Ld = L_bolt * 0.75
        Lt = L_bolt - Ld
        n_bolt = -2  # initialized
        n_bolt_prev = -1  # initialized
        if bolt == "M52":
            At = 1758 / 1e6
            d_bolt = 0.052
            m_bolt = 9.53
        elif bolt == "M48":
            At = 1473 / 1e6
            d_bolt = 0.048
            m_bolt = 8.03
        if bolt == "M42":
            At = 1120 / 1e6
            d_bolt = 0.042
            m_bolt = 6.08
        if bolt == "M36":
            At = 817 / 1e6
            d_bolt = 0.036
            m_bolt = 4.4
        elif bolt == "M30":
            At = 561 / 1e6
            d_bolt = 0.03
            m_bolt = 2.95
        elif bolt == "M24":
            At = 353 / 1e6
            d_bolt = 0.024
            m_bolt = 1.91
        elif bolt == "M18":
            At = 192 / 1e6
            d_bolt = 0.018
            m_bolt = 1.04
        cost_bolt = m_bolt * 10
        Ad = np.pi * d_bolt**2 / 4

        # insert properties
        d_insert = (
            d_bolt * 2
        )  # keeping the insert much larger than the bolt reduces bolt loads, which drive the design.
        A_insert = np.pi * (d_insert**2 - d_bolt**2) / 4
        L_insert = L_bolt
        V_insert = A_insert * L_insert
        m_insert = V_insert * rho_insert

        # geometric properties and calculations
        ratio_SCmax = 0.8  # max sparcap/chord ratio
        joint_position = inputs["joint_position"]
        i_span = util.find_nearest(self.nd_span, joint_position)
        p_le_i = inputs["pitch_axis"][i_span]
        t_layer = inputs["layer_thickness"]
        w_layer = inputs["layer_width"]
        layer_start_nd = inputs["layer_start_nd"]
        layer_end_nd = inputs["layer_end_nd"]
        bolt_spacing_dia = inputs["bolt_spacing_dia"][0]
        ply_drop_slope = inputs["ply_drop_slope"]
        t_adhesive = inputs["t_adhesive"]
        t_max = inputs["t_max"]
        L_blade = inputs["blade_length"]
        eta = self.nd_span[i_span]  # joint location
        chord = inputs["chord"][i_span]
        L_segment = L_blade[-1] / self.n_span  # m

        # Only checking on suction side for now
        ss_layer_i = self.layer_name.index(reinf_mat_ss)
        ps_layer_i = self.layer_name.index(reinf_mat_ps)
        t_reinf = t_layer[ss_layer_i][i_span]
        if t_reinf == 0.0:
            raise Exception("The joint reinforcement layer thickness is 0 mm. Please check the input geometry yaml.")
        bolt_spacing = bolt_spacing_dia * d_bolt
        n_bolt_max = chord * ratio_SCmax // bolt_spacing  # max # bolts that can fit in 0.8 chord
        w_hole_bolthead = d_bolt * 2  # 7 / 3
        h_bolthead_hole_i = d_bolt * 2
        xy_coord = inputs["coord_xy_interp"][i_span, :, :]
        t_airfoil = inputs["rthick"][i_span] * chord
        d_b2b = t_airfoil - t_reinf  # ss bolt to ps bolt distance
        t_max = d_b2b * t_max

        # Compute arc length at joint station
        xy_arc = util.arc_length(xy_coord)
        arc_L = xy_arc[-1]
        w_reinf = w_layer[ss_layer_i, i_span]

        # Loads
        Mflap_ultimate = np.abs(inputs["M1"][i_span]) * load_factor
        Medge_ultimate = np.abs(inputs["M2"][i_span]) * load_factor
        Fflap_ultimate = np.abs(inputs["F2"][i_span]) * load_factor

        # Fatigue factors
        a = inputs["a"]
        b = inputs["b"]
        c = inputs["c"]

        # Fatigue loads. Remember DLC 6.1 ultimate load safety factor = 1.35. Old loads: # Mflap_ultimate = 1.64e6, Fflap_ultimate = 1.5e5, Medge_ultimate = 3.1e5
        kp = a * (1 - eta) ** 2 + b * (1 - eta) + c
        Mflap_fatigue = Mflap_ultimate * kp
        Fflap_fatigue = Fflap_ultimate * kp
        Medge_fatigue = Medge_ultimate * kp

        # Other
        itermax = int(inputs["itermax"][0])
        discrete = inputs["discrete"]

        """ Loading calculations begin"""
        # 1- Calculate joint stiffness constant
        k_bolt = Ad * At * E_bolt / (Ad * Lt + At * Ld)  # bolt stiffness
        k_insert = A_insert * E_insert / L_bolt  # material stiffness.
        C = k_bolt / (k_bolt + k_insert)  # joint stiffness constant

        # 2- Calculate preload (initialize to 70% proof)
        Fi70p = (
            0.7 * Sp_bolt * At
        )  # N, per bolt. Need a high preload like this to resist separation. This means that the ultimate bolt safety factor (nprf) needs to be low
        Fi = Fi70p

        # 3- Calculate # bolts such that axial flap bolt forces > edge bolt forces
        # Loop through number of possible bolts per spar cap, and calculate the following. When the flap ultimate and fatigue loads both
        # dominate, that is the minimum # bolts. Units are force per bolt ONLY HERE.
        for n_bolt_min in range(3, np.min([5, int(np.ceil(n_bolt_max) // 2 * 2 + 1)]), 2):
            N = int(np.floor(n_bolt_min / 2))
            N_array = np.linspace(1, N, N)
            Fax_flap_ultimate_per_bolt = Mflap_ultimate / (d_b2b * n_bolt_min)
            Fax_edge_ultimate_per_bolt = N * Medge_ultimate / (2 * bolt_spacing * np.sum(np.square(N_array)))
            Fax_flap_fatigue_per_bolt = Mflap_fatigue / (d_b2b * n_bolt_min)
            Fax_edge_fatigue_per_bolt = N * Medge_fatigue / (2 * bolt_spacing * np.sum(np.square(N_array)))
            if (
                Fax_flap_fatigue_per_bolt > Fax_edge_fatigue_per_bolt
                and Fax_flap_ultimate_per_bolt > Fax_edge_ultimate_per_bolt
            ):
                break

        # 4- Calculate # fasteners needed to resist extreme and fatigue flap loads, and LATER: satisfy adhesive mean stress limit.

        # a- joint ultimate, fatigue loads per spar cap side
        Fax_flap_ultimate = Mflap_ultimate / d_b2b
        Fax_flap_fatigue = Mflap_fatigue / d_b2b
        Fsh_flap_ultimate = Fflap_ultimate / 2
        Fsh_flap_fatigue = Fflap_fatigue / 2

        i = 0  # while loop counter
        n_bolt_list = []
        while abs(n_bolt - n_bolt_prev) > 0.1:  # loop until converging on a number of bolts
            i += 1
            n_bolt_prev = n_bolt
            # print('n_bolt = ', n_bolt)

            # b- calc # bolts & inserts to resist flap axial ultimate loads
            # bolts. Tensile stress only.
            n_bolt_flap_ultimate = C * Fax_flap_ultimate / (Sp_bolt * At / nprf - Fi)
            # inserts. Von-mises stress. Could consider torsion here. Equation derived with MATLAB symbolic toolbox
            x3 = (
                ny * C**2 * Fax_flap_ultimate**2
                - 2 * ny * C * Fax_flap_ultimate**2
                + ny * Fax_flap_ultimate**2
                + 3 * ny * Fsh_flap_ultimate**2
            ) / (
                np.sqrt(
                    A_insert**2 * C**2 * Fax_flap_ultimate**2 * Sy_insert**2
                    - 2 * A_insert**2 * C * Fax_flap_ultimate**2 * Sy_insert**2
                    + A_insert**2 * Fax_flap_ultimate**2 * Sy_insert**2
                    + 3 * A_insert**2 * Fsh_flap_ultimate**2 * Sy_insert**2
                    - 3 * Fi**2 * Fsh_flap_ultimate**2 * ny**2
                )
                + (-Fax_flap_ultimate * Fi * ny + C * Fax_flap_ultimate * Fi * ny)
            )
            x4 = (
                ny * C**2 * Fax_flap_ultimate**2
                - 2 * ny * C * Fax_flap_ultimate**2
                + ny * Fax_flap_ultimate**2
                + 3 * ny * Fsh_flap_ultimate**2
            ) / (
                np.sqrt(
                    A_insert**2 * C**2 * Fax_flap_ultimate**2 * Sy_insert**2
                    - 2 * A_insert**2 * C * Fax_flap_ultimate**2 * Sy_insert**2
                    + A_insert**2 * Fax_flap_ultimate**2 * Sy_insert**2
                    + 3 * A_insert**2 * Fsh_flap_ultimate**2 * Sy_insert**2
                    - 3 * Fi**2 * Fsh_flap_ultimate**2 * ny**2
                )
                - (-Fax_flap_ultimate * Fi * ny + C * Fax_flap_ultimate * Fi * ny)
            )
            n_insert_flap_ultimate = max([x3, x4])

            # c- calc # bolts & inserts needed to resist flap axial fatigue loads
            # bolts. Tensile only.
            sig_i_bolt = Fi / At
            sig_a_bolt = Fax_flap_fatigue * C / At
            Sa_bolt = Se_bolt - Se_bolt / Su_bolt * sig_i_bolt
            n_bolt_flap_fatigue = nf * sig_a_bolt / Sa_bolt
            # inserts. Von mises. Could consider torsion here. Removing because insert forces are far lower in ultimate, add this back in if
            # that changes.
            sig_i_insert = Fi / A_insert
            Sa_insert = Se_insert - Se_insert / Su_insert * sig_i_insert
            n_insert_flap_fatigue = (nf / Sa_insert) ** (1 / 2) * (
                (Fax_flap_fatigue * (1 - C) / A_insert) ** 2 + 3 * (Fsh_flap_fatigue / A_insert) ** 2
            ) ** (1 / 4)

            # d - calc #bolts/inserts needed for spar cap to resist insert pull-out
            n_bolt_pullout = 2 * ns * Fax_flap_ultimate / (S_ss * np.pi * (d_insert + 2 * t_adhesive))

            # e - take max bolts needed as # bolt-insert pairs
            n_bolt = np.max(
                np.r_[
                    n_bolt_flap_fatigue,
                    n_bolt_flap_ultimate,
                    n_insert_flap_ultimate,
                    n_insert_flap_fatigue,
                    n_bolt_pullout,
                    n_bolt_min,
                ]
            )
            # if discrete:
            #     n_bolt = np.ceil(n_bolt)
            # else:
            #     precision = 2  # round up to two decimal places
            #     n_bolt = np.true_divide(np.ceil(n_bolt * 10**precision), 10**precision)  # round up to nearest 0.01
            n_bolt_list.append(n_bolt)

            # check for negatives. This implies that the loads magnitudes are imbalanced and out of range for the calculation
            if n_bolt_flap_ultimate < 0 or n_bolt_flap_fatigue < 0 or n_bolt_pullout < 0:
                logger.debug(
                    "Warning: negative bolt number found. This implies that the preload exceeds the ultimate load"
                    " requirements. Please check inputs (separation and ultimate safety factors)"
                )

            # 5- calculate preload to prevent separation and bolt shear. Make sure it's not requiring a preload that's too close
            # to proof load....this could constrain the load the joint can handle.
            Fi_sep = (
                n0 * Fax_flap_ultimate * (1 - C) / n_bolt
            )  # because separation is calculated based on extreme ultimate loading, the bolt ultimate safety factor (nprf) needs to be low.
            Fi_sh = Fsh_flap_ultimate / (mu * n_bolt)
            if Fi_sep > Fi70p:
                logger.debug(
                    "Warning, separation preload requirement (",
                    Fi_sep,
                    "N) > 70% bolt proof load and will be limited to prevent overloading.",
                )
            if Fi_sh > Fi70p:
                logger.debug(
                    "Warning, shear preload requirement (",
                    Fi_sh,
                    "N) > 70% bolt proof load and will be limited to prevent overloading.",
                )
            Fi = np.min(np.r_[np.max(np.r_[Fi_sh, Fi_sep]), Fi70p])

            # if iteration is stuck in an Fi-driven loop, then take the max # bolts required by loop
            if i > itermax:
                if discrete:
                    logger.debug("Solution oscillating between bolt numbers. Choosing the maximum of these.")
                    seq = itermax
                    for x in range(2, itermax // 2):
                        if n_bolt_list[-x:] == n_bolt_list[-2 * x : -x]:
                            seq = x
                    n_bolt = max(n_bolt_list[-seq:])
                else:
                    logger.debug(
                        "Solution has not converged to 0.01 bolt. Choosing the maximum of the last three bolt numbers"
                    )
                    n_bolt = max(n_bolt_list[-3:])
            if n_bolt >= n_bolt_max:
                n_bolt_req = n_bolt
                n_bolt = n_bolt_max

            # print('n_bolt_flap_fatigue', n_bolt_flap_fatigue)
            # print('n_bolt_flap_ultimate', n_bolt_flap_ultimate)
            # print('n_insert_flap_ultimate', n_insert_flap_ultimate)
            # print('n_insert_flap_fatigue', n_insert_flap_fatigue)
            # print('n_bolt_pullout', n_bolt_pullout)
            # print('n_bolt_min', n_bolt_min)
            # print('n_bolt', n_bolt)
            #
            # print('Fi_sep', Fi_sep)
            # print('Fi_sh', Fi_sh)
            # print('Fi70p', Fi70p)
            # print('Fi', Fi)
            # print('################################################')

        # 6- loop through steps 4b-5 until n_bolt converges. Result is n_bolt
        if n_bolt >= n_bolt_max:
            logger.debug(
                "Warning. Unable to accommodate # bolts required (",
                n_bolt_req,
                "). Limiting to max # bolts that can fit in " "the cross section (",
                n_bolt_max,
                ").",
            )

        # 7- calc spar cap dimensions needed to resist loads. Neglect fatigue because composites handle it better than
        # metal, generally. Shear will drive the design due to carbon fiber's isotropicity. Consider shearing due to
        # shear out on a z-line on the outside of the adhesive, in the middle of the bolt. Also consider shear at bolt
        # head hole. ***OR, the spar cap could be sized with WISDEM as usual.***

        # a- insert shear out.
        t1 = 2 * Fsh_flap_ultimate * ns / (L_insert * n_bolt * S_ss)

        # b- shear at bolt head hole.
        t2 = (
            Fsh_flap_ultimate * ns / (bolt_spacing * S_ss * n_bolt) + w_hole_bolthead * h_bolthead_hole_i / bolt_spacing
        )

        # c- spar cap dimensions
        t_req_reinf = np.max([t1, t2])
        if t_req_reinf > t_max:
            t_req_reinf = t_max
            # print('Warning, required spar cap thickness (', t_req_reinf, ') is greater than max allowed (', t_max, '). Limiting to max allowed')
        w_req_reinf = n_bolt * bolt_spacing  # required width driven by number of bolts
        w_layer[ss_layer_i, i_span] = w_req_reinf
        w_layer[ps_layer_i, i_span] = w_req_reinf
        # if w_req_reinf > w_reinf:
        # print('Warning, required spar cap width of ', w_req_reinf, ' is greater than nominal. Update input files')

        # check if spar cap reaches TE or LE. If so, change y offset to prevent this
        offset = inputs["layer_offset_y_pa"]
        for i in [ss_layer_i, ps_layer_i]:
            # print('offset =', offset[i, i_span])
            if offset[i, i_span] + 0.5 * w_req_reinf > ratio_SCmax * chord * (1.0 - p_le_i):  # hitting TE?
                offset[i, i_span] = ratio_SCmax * chord * (1.0 - p_le_i) - w_req_reinf - 0.5
            elif offset[i, i_span] - 0.5 * w_req_reinf < -ratio_SCmax * chord * p_le_i:  # hitting LE?
                offset[i, i_span] = -ratio_SCmax * chord * p_le_i + (0.50 * w_req_reinf)
            # calculate layer start, end based on width
            midpoint = calc_axis_intersection(
                inputs["coord_xy_dim"][i_span, :, :],
                inputs["twist"][i_span],
                offset[i, i_span],
                [0.0, 0.0],
                [discrete_inputs["layer_side"][i]],
            )[0]
            layer_start_nd[i, i_span] = midpoint - w_req_reinf / arc_L / chord / 2.0
            layer_end_nd[i, i_span] = midpoint + w_req_reinf / arc_L / chord / 2.0

        # 8- once width and thickness are found, the required ply drop length will determine how long the bulge in the spar cap
        # is, and inform spar cap total mass. ***OR, the spar cap could be sized with WISDEM as usual.***
        h_bolthead_hole = t_req_reinf / 2 + 7 / 6 * d_bolt
        dt = t_req_reinf - t_reinf
        L_transition = dt / ply_drop_slope  # TODO what do I want to do with this?
        if L_transition > L_segment:
            logger.debug("Warning. Segments too short to accommodate ply drop requirements")

        # 9- mass calcs: bolt+insert mass - cutout mass
        n_bolt *= 2  # consider bolts in each spar cap
        m_bolt_tot = n_bolt * m_bolt
        m_insert_tot = n_bolt * m_insert
        V_adhesive = np.pi * ((d_insert + 2 * t_adhesive) ** 2 - d_insert**2) / 4 * L_insert * n_bolt  # m3
        m_adhesive_tot = V_adhesive * rho_adhesive
        V_bolthead_hole = h_bolthead_hole * np.pi * w_hole_bolthead**2 / 4
        V_bolthead_hole_tot = V_bolthead_hole * n_bolt
        V_insert_cutout_tot = np.pi * (d_insert + 2 * t_adhesive) ** 2 / 4 * L_insert * n_bolt  # m3
        V_cutout = V_insert_cutout_tot + V_bolthead_hole_tot  # m3
        m_cutout = V_cutout * rho_ss
        m_add = m_bolt_tot + m_insert_tot + m_adhesive_tot - m_cutout

        cost_adhesive = m_adhesive_tot * pu_cost_adhesive
        m_insert_stock = np.pi * (d_insert**2) / 4 * L_insert * n_bolt
        cost_insert = m_insert_stock * pu_cost_insert
        cost_bolt_tot = cost_bolt * n_bolt
        cost_joint_materials = cost_adhesive + cost_bolt_tot + cost_insert
        t_reinf_ratio = t_req_reinf / t_reinf
        w_reinf_ratio = w_req_reinf / w_reinf

        # outputs['layer_offset_y_bjs'] = offset
        # outputs['layer_start_nd_bjs'] = layer_start_nd
        # outputs['layer_end_nd_bjs'] = layer_end_nd
        # outputs['layer_width_bjs'] = w_layer
        outputs["L_transition_joint"] = L_transition
        outputs["t_reinf_ratio_joint"] = t_reinf_ratio
        outputs["w_reinf_ratio_joint"] = w_reinf_ratio
        outputs["n_joint_bolt"] = n_bolt
        outputs["joint_mass"] = m_add
        outputs["joint_material_cost"] = cost_joint_materials
        outputs["joint_total_cost"] = cost_joint_materials + inputs["joint_nonmaterial_cost"]

    
        
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
            "x_ec",
            "y_ec",
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
        self.add_subsystem(
            "constr", DesignConstraints(modeling_options=modeling_options, opt_options=opt_options), promotes=["s"]
        )
        self.add_subsystem("brs", BladeRootSizing(rotorse_options=modeling_options["WISDEM"]["RotorSE"]))
        if modeling_options["WISDEM"]["RotorSE"]["bjs"]:
            self.add_subsystem(
                "bjs",
                BladeJointSizing(
                    rotorse_options=modeling_options["WISDEM"]["RotorSE"],
                    n_mat=self.options["modeling_options"]["materials"]["n_mat"],
                ),
            )

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
        self.connect("strains.strainU_spar", "constr.strainU_spar")
        self.connect("strains.strainL_spar", "constr.strainL_spar")
        self.connect("strains.strainU_te", "constr.strainU_te")
        self.connect("strains.strainL_te", "constr.strainL_te")
        self.connect("frame.flap_mode_freqs", "constr.flap_mode_freqs")
        self.connect("frame.edge_mode_freqs", "constr.edge_mode_freqs")
        self.connect("frame.tors_mode_freqs", "constr.tors_mode_freqs")

        # Blade root moment to blade root sizing
        self.connect("frame.root_M", "brs.root_M")

        if modeling_options["WISDEM"]["RotorSE"]["bjs"]:
            # Moments to joint sizing
            self.connect("frame.F2", "bjs.F2")
            self.connect("frame.M1", "bjs.M1")
            self.connect("frame.M2", "bjs.M2")


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
