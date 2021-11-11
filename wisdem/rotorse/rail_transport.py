import numpy as np
import scipy.constants as spc
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
from openmdao.api import ExplicitComponent
from scipy.optimize import minimize
from wisdem.commonse.constants import gravity

# This isn't used, but keeping around the code for now
def enforce_length(x, y, z, L0):
    r0 = np.sum(L0)
    xn = x.copy()
    yn = y.copy()
    zn = z.copy()
    rn = util.arc_length(np.c_[xn, yn, zn])

    counter=0
    while np.abs(rn[-1] - r0) > 1e-2 and counter<100:
        counter+=1
        L = np.diff(rn)
        xn[1:] = (L0 / L) * (xn[1:] - xn[:-1]) + xn[:-1]
        zn[1:] = (L0 / L) * (zn[1:] - zn[:-1]) + zn[:-1]
        rn = util.arc_length(np.c_[xn, yn, zn])
    return xn, zn


class RailTransport(ExplicitComponent):
    # Openmdao component to simulate a rail transport of a wind turbine blade
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry

        # Rail configuration
        self.add_input(
            "horizontal_angle_deg",
            val=13.0,
            units="deg",
            desc="Angle of horizontal turn (defined for an chord of 100 feet)",
        )
        self.add_input(
            "min_vertical_radius",
            val=609.6,
            units="m",
            desc="Minimum radius of a vertical curvature (hill or sag) (2000 feet)",
        )
        # self.add_input("lateral_clearance", val=6.7056, units="m", desc="Clearance profile horizontal (22 feet)")
        # self.add_input("vertical_clearance", val=7.0104, units="m", desc="Clearance profile vertical (23 feet)")
        self.add_input("lateral_clearance", val=5.4864, units="m", desc="Clearance profile horizontal (18 feet)")
        self.add_input("vertical_clearance", val=6.096, units="m", desc="Clearance profile vertical (20 feet)")
        self.add_input(
            "deck_height", val=1.19, units="m", desc="Height of the deck of the flatcar from the rails (4 feet)"
        )
        self.add_input("max_strains", val=3500.0 * 1.0e-6, desc="Max allowable strains during transport")
        self.add_input("max_LV", val=0.5, desc="Max allowable ratio between lateral and vertical forces")
        self.add_input(
            "max_flatcar_weight_4axle", val=129727.31, units="kg", desc="Max mass of an 4-axle flatcar (286000 lbm)"
        )
        self.add_input(
            "max_flatcar_weight_8axle", val=217724.16, units="kg", desc="Max mass of an 8-axle flatcar (480000 lbm)"
        )
        self.add_input("max_root_rot_deg", val=15.0, units="deg", desc="Max degree of angle at blade root")
        self.add_input("flatcar_tc_length", val=20.12, units="m", desc="Flatcar truck center to truck center length")

        # Input - Outer blade geometry
        self.add_input(
            "blade_ref_axis",
            val=np.zeros((n_span, 3)),
            units="m",
            desc="2D array of the coordinates (x,y,z) of the blade reference axis, defined along blade span. The coordinate system is the one of BeamDyn: it is placed at blade root with x pointing the suction side of the blade, y pointing the trailing edge and z along the blade span. A standard configuration will have negative x values (prebend), if swept positive y values, and positive z values.",
        )
        self.add_input(
            "theta",
            val=np.zeros(n_span),
            units="deg",
            desc="Twist angle at each section (positive decreases angle of attack)",
        )
        self.add_input("chord", val=np.zeros(n_span), units="m", desc="chord length at each section")
        self.add_input(
            "pitch_axis",
            val=np.zeros(n_span),
            desc="1D array of the chordwise position of the pitch axis (0-LE, 1-TE), defined along blade span.",
        )
        self.add_input(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations. The leading edge is place at x=0 and y=0.",
        )

        # Inputs - Distributed beam properties
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
            desc="flatwise stiffness (bending about y-axis of airfoil aligned coordinate system)",
        )
        self.add_input("EIxy", val=np.zeros(n_span), units="N*m**2", desc="coupled flap-edge stiffness")
        self.add_input(
            "GJ",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="torsional stiffness (about axial z-axis of airfoil aligned coordinate system)",
        )
        self.add_input("rhoA", val=np.zeros(n_span), units="kg/m", desc="mass per unit length")
        self.add_input("rhoJ", val=np.zeros(n_span), units="kg*m", desc="polar mass moment of inertia per unit length")
        self.add_input(
            "x_sc",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the shear-center offset with respect to the XR-YR axes",
        )
        self.add_input(
            "y_sc",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section shear-center with respect to the reference frame, XR-YR",
        )

        # Outputs
        self.add_output(
            "constr_LV_4axle_horiz",
            val=np.zeros(2),
            desc="Constraint for max L/V for a 4-axle flatcar on horiz curves, violated when bigger than 1",
        )
        self.add_output(
            "constr_LV_8axle_horiz",
            val=np.zeros(2),
            desc="Constraint for max L/V for an 8-axle flatcar on horiz curves, violated when bigger than 1",
        )
        self.add_output(
            "constr_LV_4axle_vert",
            val=np.zeros(2),
            desc="Constraint for max L/V for a 4-axle flatcar on vert curves, violated when bigger than 1",
        )
        self.add_output(
            "constr_LV_8axle_vert",
            val=np.zeros(2),
            desc="Constraint for max L/V for an 8-axle flatcar on vert curves, violated when bigger than 1",
        )
        self.add_output(
            "constr_strainPS", val=np.zeros(n_span), desc="Strain along pressure side of blade on a horizontal curve"
        )
        self.add_output(
            "constr_strainSS", val=np.zeros(n_span), desc="Strain along suction side of blade on a horizontal curve"
        )
        self.add_output(
            "constr_strainLE",
            val=np.zeros((n_span, 2)),
            desc="Strain along leading edge side of blade on a vertical curve",
        )
        self.add_output(
            "constr_strainTE",
            val=np.zeros((n_span, 2)),
            desc="Strain along trailing edge side of blade on a vertical curve",
        )

    def compute(self, inputs, outputs):

        PBEAM = False
        _8axle = False

        # Unpack inputs
        x_ref = inputs["blade_ref_axis"][:, 0]  # from PS to SS
        y_ref = inputs["blade_ref_axis"][:, 1]  # from LE to TE
        z_ref = inputs["blade_ref_axis"][:, 2]  # from root to tip
        r = util.arc_length(inputs["blade_ref_axis"])
        blade_length = r[-1]
        theta = inputs["theta"]
        chord = inputs["chord"]
        x_sc = inputs["x_sc"]
        y_sc = inputs["y_sc"]
        A = inputs["A"]
        rhoA = inputs["rhoA"]
        rhoJ = inputs["rhoJ"]
        GJ = inputs["GJ"]
        EA = inputs["EA"]
        EIxx = inputs["EIxx"]  # edge (rotation about x)
        EIyy = inputs["EIyy"]  # flap (rotation about y)
        EIxy = inputs["EIxy"]
        lateral_clearance = 0.5 * float(inputs["lateral_clearance"])
        vertical_clearance = float(inputs["vertical_clearance"])
        max_strains = float(inputs["max_strains"])
        max_rot = float(inputs["max_root_rot_deg"])
        max_LV = float(inputs["max_LV"])
        if _8axle:
            mass_car = float(inputs["max_flatcar_weight_8axle"])
        else:
            mass_car = float(inputs["max_flatcar_weight_4axle"])
        flatcar_tc_length = float(inputs["flatcar_tc_length"])

        # ------- Get turn radius geometry for horizontal and vertical curves
        # Horizontal turns- defined as a degree of arc assuming a 100ft "chord"
        # https://trn.trains.com/railroads/ask-trains/2011/01/measuring-track-curvature
        angleH_rad = np.deg2rad(inputs["horizontal_angle_deg"][0])
        r_curveH = spc.foot * 100.0 / (2.0 * np.sin(0.5 * angleH_rad))
        arcsH = r / r_curveH

        # Vertical curves on hills and sags defined directly by radius
        r_curveV = inputs["min_vertical_radius"][0]
        # ----------

        # ---------- Put airfoil cross sections into principle axes
        # Determine principal C.S. (with swap of x, y for profile c.s.)
        EIxx_cs, EIyy_cs = EIyy.copy(), EIxx.copy()
        x_sc_cs, y_sc_cs = y_sc.copy(), x_sc.copy()
        EIxy_cs = EIxy.copy()

        # translate to elastic center
        EIxx_cs -= y_sc_cs ** 2 * EA
        EIyy_cs -= x_sc_cs ** 2 * EA
        EIxy_cs -= x_sc_cs * y_sc_cs * EA

        # get rotation angle
        alpha = 0.5 * np.arctan2(2 * EIxy_cs, EIyy_cs - EIxx_cs)

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
        # -------------------

        # ---------- Frame3dd blade prep
        # Nodes: Prep data, but node x,y,z will shift for vertical and horizontal curves
        rad = np.zeros(self.n_span)  # 'radius' of rigidity at node- set to zero
        inode = 1 + np.arange(self.n_span)  # Node numbers (will convert to 1-based indexing later)
        L = np.diff(r)

        # Reactions: attachment at root
        ireact = np.array([inode[0]])
        rigid = 1e16
        pinned = rigid * np.ones(1)
        reactions = pyframe3dd.ReactionData(ireact, pinned, pinned, pinned, pinned, pinned, pinned, float(rigid))

        # Element data
        ielem = np.arange(1, self.n_span)  # Element Numbers
        N1 = np.arange(1, self.n_span)  # Node number start
        N2 = np.arange(2, self.n_span + 1)  # Node number finish
        E = EA / A
        rho = rhoA / A
        J = rhoJ / rho
        G = GJ / J
        Ix = EIyy / E if PBEAM else EI11 / E
        Iy = EIxx / E if PBEAM else EI22 / E
        Asx = Asy = 1e-6 * np.ones(ielem.shape)  # Unused when shear=False

        # Have to convert nodal values to find average at center of element
        Abar, _ = util.nodal2sectional(A)
        Ebar, _ = util.nodal2sectional(E)
        rhobar, _ = util.nodal2sectional(rho)
        Jbar, _ = util.nodal2sectional(J)
        Gbar, _ = util.nodal2sectional(G)
        Ixbar, _ = util.nodal2sectional(Ix)
        Iybar, _ = util.nodal2sectional(Iy)

        # Angle of element principal axes relative to global coordinate system
        # Global c.s. is blade with z from root to tip, y from ss to ps, and x from LE to TE (TE points up)
        # Local element c.s. is airfoil (twist + principle rotation)
        if PBEAM:
            roll = np.zeros(theta.shape)
        else:
            roll, _ = util.nodal2sectional(theta + alpha)

        elements = pyframe3dd.ElementData(ielem, N1, N2, Abar, Asx, Asy, Jbar, Ixbar, Iybar, Ebar, Gbar, roll, rhobar)

        # Frame3dd options: no need for shear, axial stiffening, or higher resolution force calculations
        options = pyframe3dd.Options(False, False, -1)
        # -----------

        # ------ Airfoil positions at which to measure strain
        # Find the cross sectional points furthest from the elastic center at each spanwise location to be used for strain measurement
        xps = np.zeros(self.n_span)
        xss = np.zeros(self.n_span)
        yps = np.zeros(self.n_span)
        yss = np.zeros(self.n_span)
        xle = np.zeros(self.n_span)
        xte = np.zeros(self.n_span)
        yle = np.zeros(self.n_span)
        yte = np.zeros(self.n_span)

        for i in range(self.n_span):
            ## Rotate the profiles to the blade reference system
            profile_i = inputs["coord_xy_interp"][i, :, :]
            profile_i_rot = np.column_stack(
                util.rotate(inputs["pitch_axis"][i], 0.0, profile_i[:, 0], profile_i[:, 1], np.radians(theta[i]))
            )
            # normalize
            profile_i_rot[:, 0] -= min(profile_i_rot[:, 0])
            profile_i_rot = profile_i_rot / max(profile_i_rot[:, 0])
            profile_i_rot_precomp = profile_i_rot.copy()
            idx_s = 0
            idx_le_precomp = np.argmax(profile_i_rot_precomp[:, 0])
            if idx_le_precomp != 0:
                if profile_i_rot_precomp[0, 0] == profile_i_rot_precomp[-1, 0]:
                    idx_s = 1
                profile_i_rot_precomp = np.row_stack(
                    (profile_i_rot_precomp[idx_le_precomp:], profile_i_rot_precomp[idx_s:idx_le_precomp, :])
                )
            profile_i_rot_precomp[:, 1] -= profile_i_rot_precomp[np.argmin(profile_i_rot_precomp[:, 0]), 1]

            # # renormalize
            profile_i_rot_precomp[:, 0] -= min(profile_i_rot_precomp[:, 0])
            profile_i_rot_precomp = profile_i_rot_precomp / max(profile_i_rot_precomp[:, 0])

            if profile_i_rot_precomp[-1, 0] != 1.0:
                profile_i_rot_precomp = np.row_stack((profile_i_rot_precomp, profile_i_rot_precomp[0, :]))

            xnode = profile_i_rot_precomp[:, 0]
            xnode_pa = xnode - inputs["pitch_axis"][i]
            ynode = profile_i_rot_precomp[:, 1]
            theta_rad = theta[i] * np.pi / 180.0

            xnode_no_theta = xnode_pa * np.cos(-theta_rad) - ynode * np.sin(-theta_rad)
            ynode_no_theta = xnode_pa * np.sin(-theta_rad) + ynode * np.cos(-theta_rad)

            xnode_dim_no_theta = xnode_no_theta * chord[i]
            ynode_dim_no_theta = ynode_no_theta * chord[i]

            xnode_dim = xnode_dim_no_theta * np.cos(theta_rad) - ynode_dim_no_theta * np.sin(theta_rad)
            ynode_dim = xnode_dim_no_theta * np.sin(theta_rad) + ynode_dim_no_theta * np.cos(theta_rad)

            yss[i] = max(ynode_dim) - y_sc_cs[i]
            yps[i] = y_sc_cs[i] - min(ynode_dim)
            xte[i] = max(xnode_dim) - x_sc_cs[i]
            xle[i] = x_sc_cs[i] - min(xnode_dim)

        # Put these sectional points in airfoil principle directions
        xps_cs, yps_cs = yps, xps
        xss_cs, yss_cs = yss, xss

        ps1, ps2 = rotate(xps_cs, yps_cs)
        ss1, ss2 = rotate(xss_cs, yss_cs)

        # ----------------

        # -------- Horizontal curve where we select blade support nodes on flat cars
        # Gravity field orientation
        gy = -gravity
        gx = gz = 0.0

        # Set clearance boundary
        r_envelopeH = r_curveH + lateral_clearance * np.array([-1, 1])

        # Use blade shape and clearance envelope to determine node position limits
        r_envelopeH_inner1 = r_envelopeH.min() + yss
        r_envelopeH_inner2 = r_envelopeH.min() + yps

        r_envelopeH_outer1 = r_envelopeH.max() - yps
        r_envelopeH_outer2 = r_envelopeH.max() - yss

        # Find rotation angles that keep blade within inner boundary
        # Function that does the structural analysis to be called during optimization
        def rotate_blade(ang):

            # Node location starting points when curving towards SS
            # (towards the LEFT with LE pointed down and standing at the root looking at tip)
            x_rot1, z_rot1 = util.rotate(r_curveH, 0.0, r_curveH + x_ref, z_ref, ang[0])

            # Node location starting points when curving towards PS
            # (towards the RIGHT with LE pointed down and standing at the root looking at tip)
            x_rot2, z_rot2 = util.rotate(-r_curveH, 0.0, -r_curveH + x_ref, z_ref, ang[1])

            # Check solved blade shape against envelope
            r_check1 = np.sqrt(x_rot1 ** 2 + z_rot1 ** 2)
            r_check2 = np.sqrt(x_rot2 ** 2 + z_rot2 ** 2)

            # Formulate as constraints for SLSQP
            cboundary = np.sum(np.maximum(r_envelopeH_inner1 - r_check1, 0.0)) + np.sum(
                np.maximum(r_envelopeH_inner2 - r_check2, 0.0)
            )

            return -cboundary

        # Initiliaze scipy minimization to find the initial blade root flapwise angle - biggest deflection that keeps blade away from inner envelope
        const = {}
        const["type"] = "ineq"
        const["fun"] = rotate_blade

        bounds = [np.deg2rad(max_rot) * np.r_[0, 1], np.deg2rad(max_rot) * np.r_[-1, 0]]
        x0 = np.deg2rad(max_rot * np.array([1.0, -1.0]))
        result = minimize(lambda x: -np.sum(np.abs(x)), x0, method="slsqp", bounds=bounds, tol=1e-3, constraints=const)

        # Now rotate blade at optimized angle



        x_rail_inner  = np.linspace(0., 2.*r_envelopeH.min(), 10000)
        y_rail_inner  = np.sqrt(r_envelopeH.min()**2. - (x_rail_inner-r_envelopeH.min())**2.)
        x_rail_outer  = np.linspace(0., 2.*r_envelopeH.max(), 10000)
        y_rail_outer  = np.sqrt(r_envelopeH.max()**2. - (x_rail_outer-r_envelopeH.max())**2.)

        # Undeflected transport
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1, figsize=(4,8))
        # ax.plot(x_rail_inner+2*lateral_clearance, y_rail_inner, "k:", label='Lateral clearance')
        # ax.plot(x_rail_outer, y_rail_outer, "k:")
        # # ax.plot(x_rot1+x_rot1[0]+lateral_clearance, z_rot1, label='rot1')
        # ax.plot(x_rot2-x_rot2[0]+lateral_clearance + yss, z_rot2, color="tab:red", label='Blade')
        # ax.plot(x_rot2-x_rot2[0]+lateral_clearance - yps, z_rot2, color="tab:red")
        # ax.set_xlim([0,50])
        # # ax.axis('equal')
        # plt.grid(color=[0.8,0.8,0.8], linestyle='--')
        # plt.subplots_adjust(bottom = 0.15, left = 0.15)
        # ax.legend()
        # plt.show()


        

        def get_max_force_h(FrIn):
            # Objective function to minimize the reaction force of the first flatcat, which holds blade root, during a lateral curve
            Fr = FrIn[2:].reshape((self.n_span - 1, 2))
            RF_flatcar_1 = np.zeros(2)
            for k in range(2):
                q_iter    = np.r_[0., Fr[:,k]]
                V_iter    = np.zeros(self.n_span)
                M_iter    = np.zeros(self.n_span)
                for i in range(self.n_span):
                    V_iter[i] = np.trapz(q_iter[i:],r[i:])
                for i in range(self.n_span):
                    M_iter[i] = np.trapz(V_iter[i:],r[i:])
                
                RF_flatcar_1[k] = 0.5 * V_iter[0] + M_iter[0] / flatcar_tc_length

            return np.sum(abs(RF_flatcar_1)*1.e-5)

        # Function that does the structural analysis to be called during optimization
        def run_hcurve(FrIn, optFlag=True, final=False):

            angle = FrIn[0:2]
            if final:
                print("Flapwise rotations at blade root: ", angle)

            # Node location starting points when curving towards SS
            # (towards the LEFT with LE pointed down and standing at the root looking at tip)
            x_rot1, z_rot1 = util.rotate(r_curveH, 0.0, r_curveH + x_ref, z_ref, angle[0])
            nodes1 = pyframe3dd.NodeData(inode, x_rot1, y_ref, z_rot1, rad)

            # Node location starting points when curving towards PS
            # (towards the RIGHT with LE pointed down and standing at the root looking at tip)
            x_rot2, z_rot2 = util.rotate(-r_curveH, 0.0, -r_curveH + x_ref, z_ref, angle[1])
            nodes2 = pyframe3dd.NodeData(inode, x_rot2, y_ref, z_rot2, rad)

            # Initialize Frame3dd objects
            blade1 = pyframe3dd.Frame(nodes1, reactions, elements, options)
            blade2 = pyframe3dd.Frame(nodes2, reactions, elements, options)


            Fr = FrIn[2:].reshape((self.n_span - 1, 2))

            # Will only worry about radial loads
            Fy = Mx = My = Mz = np.zeros(self.n_span - 1)

            # Output containers
            RF_derailH = np.zeros(2)  # Derailment reaction force
            r_check_outer = np.zeros((self.n_span, 2))  # Envelope constraint
            r_check_inner = np.zeros((self.n_span, 2))  # Envelope constraint
            strainPS = np.zeros((self.n_span, 2))
            strainSS = np.zeros((self.n_span, 2))

            # Look over bend to PS/SS cases
            for k in range(2):
                if k == 0:
                    blade, r_outer, r_inner, angs = blade1, r_envelopeH_outer1, r_envelopeH_inner1, arcsH[1:]
                else:
                    blade, r_outer, r_inner, angs = blade2, r_envelopeH_outer2, r_envelopeH_inner2, np.pi - arcsH[1:]

                # Load case: gravity + blade bending to conform to outer boundary
                load = pyframe3dd.StaticLoadCase(gx, gy, gz)

                # Put radial loads in x,z plane
                Fx = -1e4 * Fr[:, k] * np.cos(angs)
                Fz = -1e4 * Fr[:, k] * np.sin(angs)
                load.changePointLoads(inode[1:], Fx, Fy, Fz, Mx, My, Mz)

                # Add load case to Frame3DD objects and run
                blade.clearLoadCases()
                blade.addLoadCase(load)

                # Run the case
                displacements, forces, forces_rxn, _, _, _ = blade.run()
                L0 = np.diff(util.arc_length(np.vstack((blade.nx, blade.nz)).T))
                x = blade.nx + displacements.dx[0, :]
                y = blade.ny + displacements.dy[0, :]
                z = blade.nz + displacements.dz[0, :]
                x_defl , z_defl = enforce_length(x, y, z, L0)

                # Check solved blade shape against envelope
                
                r_check_outer[:, k] = (
                    np.sqrt(x_defl ** 2 + z_defl ** 2)
                    - r_outer
                )
                r_check_inner[:, k] = (
                    r_inner - 
                    np.sqrt(x_defl ** 2 + z_defl ** 2)
                )

                # if final and k==1:
                #     print("Undeflected blade length: ", util.arc_length(np.vstack((blade.nx, blade.nz)).T)[-1])
                #     print("Corrected blade length: ", util.arc_length(np.vstack((x_defl, z_defl)).T)[-1])
                #     # Deflected transport
                #     fig, ax = plt.subplots(1, 1, figsize=(4,6))
                #     ax.plot(x_rail_inner+2*lateral_clearance, y_rail_inner, "k:", label='Lateral clearance')
                #     ax.plot(x_rail_outer, y_rail_outer, "k:")
                #     # ax.plot(x_rot1+x_rot1[0]+lateral_clearance, z_rot1, label='rot1')
                #     ax.plot(x_defl + yss - blade.nx[0] + lateral_clearance, z_defl, color="tab:red", label='Blade')
                #     ax.plot(x_defl - yps - blade.nx[0] + lateral_clearance, z_defl, color="tab:red")
                #     ax.plot(blade.nx+ yss - blade.nx[0] + lateral_clearance, blade.nz, color="tab:blue", label='Undeflected blade')
                #     ax.plot(blade.nx- yps - blade.nx[0] + lateral_clearance, blade.nz, color="tab:blue")
                #     #ax.set_xlim([0,50])
                #     ax.axis('equal')
                #     plt.grid(color=[0.8,0.8,0.8], linestyle='--')
                #     plt.subplots_adjust(bottom = 0.15, left = 0.15)
                #     ax.legend()
                #     plt.show()

                # Derailing reaction force on root node
                #  - Lateral force on wheels (multiply by 0.5 for 2 wheel sets)
                #  - Moment around axis perpendicular to ground
                RF_derailH[k] = (
                    0.5 * np.sqrt(forces_rxn.Fx ** 2 + forces_rxn.Fz ** 2) + np.abs(forces_rxn.Myy) / flatcar_tc_length
                )

                # Element shear and bending, one per element, which are already in principle directions in Hansen's notation
                # Zero-ing out axial stress as there shouldn't be any for pure beam bending
                Fz = np.r_[-forces.Nx[0, 0], forces.Nx[0, 1::2]]
                M1 = np.r_[-forces.Myy[0, 0], forces.Myy[0, 1::2]]
                M2 = np.r_[forces.Mzz[0, 0], -forces.Mzz[0, 1::2]]
                if PBEAM:
                    M1, M2 = rotate(M1, M2)

                # Compute strain at the two points: pressure/suction side extremes
                strainPS[:, k] = -(
                    M1 / EI11 * ps2 - M2 / EI22 * ps1 + Fz / EA
                )  # negative sign because Hansen c3 is opposite of Precomp z
                strainSS[:, k] = -(M1 / EI11 * ss2 - M2 / EI22 * ss1 + Fz / EA)

            if optFlag:
                # First constraint is compliance with outer boundary
                cboundary_outer = np.maximum(r_check_outer, 0)
                cboundary_inner = np.maximum(r_check_inner, 0)

                # Second constraint is reaction forces for derailment:
                crxn = np.maximum(RF_derailH - (0.5 * mass_car * gravity) / max_LV, 0.0)

                # Third constraint is keeping the strains reasonable
                cstrainPS = np.maximum(np.abs(strainPS) - max_strains, 0.0)
                cstrainSS = np.maximum(np.abs(strainSS) - max_strains, 0.0)

                # Accumulate constraints
                cons = np.array([np.sum(cboundary_outer), np.sum(cboundary_inner), np.sum(crxn), np.sum(cstrainPS), np.sum(cstrainSS)])
                # cons = np.array([np.sum(cboundary_outer), np.sum(crxn), np.sum(cstrainPS), np.sum(cstrainSS)])

                return -cons
            else:
                return RF_derailH, strainPS, strainSS

        # Initiliaze scipy minimization: Minimize force applied that keeps blade within all constraints
        const = {}
        const["type"] = "ineq"
        const["fun"] = run_hcurve

        npts = 2 * (self.n_span - 1)
        bounds = [(-max_rot, max_rot),(-max_rot, max_rot)]
        for i in range(npts):
            bounds.append((0, 1e2))
        # bounds = [(-1e2, 1e2)] * (npts + 2)
        x0 = np.r_[result.x, 0. * np.ones(npts)]
        result = minimize(
            lambda x: np.sum(np.abs(x[2:])),
            # get_max_force_h,
            x0,
            method="slsqp",
            bounds=bounds,
            tol=1e-3,
            constraints=const,
            options={"maxiter": 100},
        )

        print("Rail transport module convergence: ", result.success)
        # Evaluate optimized solution
        RF_derailH, strainPS, strainSS = run_hcurve(result.x, optFlag=False, final=False)

        # Express derailing force as a constraint
        if _8axle:
            outputs["constr_LV_8axle_horiz"] = RF_derailH / (0.5 * mass_car * gravity) / max_LV
        else:
            outputs["constr_LV_4axle_horiz"] = RF_derailH / (0.5 * mass_car * gravity) / max_LV

        # Strain constraint outputs
        outputs["constr_strainPS"] = np.max(np.abs(strainPS) / max_strains, axis=1)
        outputs["constr_strainSS"] = np.max(np.abs(strainSS) / max_strains, axis=1)
        # else:
        #     outputs['LV_constraint_4axle_horiz'] = 2.
        #     outputs['LV_constraint_8axle_horiz'] = 2.
        #     outputs['constr_strainPS']           = 2. * np.ones([npts,2])
        #     outputs['constr_strainSS']           = 2. * np.ones([npts,2])
        #     print('The optimization cannot satisfy the blade rail transport constraints.')

        """
        # ------- Vertical hills/sag using best attachment points
        # Set up Frame3DD blade for vertical analysis

        # Node location starting points
        nodes   = pyframe3dd.NodeData(inode, x_ref, r_curveV+y_ref, z_ref, rad)

        # Compute node radii starting points to determine if within clearance boundary
        r_blade = np.sqrt(nodes.y**2 + nodes.z**2)

        # Initialize frame3dd object
        blade = pyframe3dd.Frame(nodes, reactions, elements, options)

        # Hill
        blade_ymax        = y_ref + xte
        r_envelopeV       = r_curveV + vertical_clearance
        r_envelopeV_outer = r_envelopeV - blade_ymax
        node_dr           = np.minimum(r_envelopeV_outer - r_blade, 0)
        node_dy           = node_dr*np.cos(arcsV)
        node_dz           = node_dr*np.sin(arcsV)

        # Load case 1: gravity + hill
        dx = dM = np.zeros(ireact.size)
        load1 = pyframe3dd.StaticLoadCase(gx, gy, gz)
        load1.changePrescribedDisplacements(ireact+1, dx, node_dy[ireact], node_dz[ireact], dM, dM, dM)

        # Sag
        blade_ymin        = y_ref - xte
        r_envelopeV       = r_curveV - vertical_clearance
        r_envelopeV_inner = r_envelopeV.min() + blade_ymin
        node_dr           = np.maximum(r_envelopeV_inner - r_blade, 0)
        node_dy           = node_dr*np.cos(arcsV)
        node_dz           = node_dr*np.sin(arcsV)

        # Load case 2: gravity + sag
        load2 = pyframe3dd.StaticLoadCase(gx, gy, gz)
        load2.changePrescribedDisplacements(ireact+1, dx, node_dy[ireact], node_dz[ireact], dM, dM, dM)

        # Store these load cases and run
        blade.addLoadCase(load1)
        blade.addLoadCase(load2)
        #blade.write('blade.3dd')
        displacements, forces, forces_rxn, internalForces, mass, modal = blade.run()

        # Reaction forces for derailment:
        #  - Lateral force on wheels (multiply by 0.5 for 2 wheel sets)
        #  - Moment around axis perpendicular to ground
        # Should have 2 cases X 3 rxn nodes
        RF_derailV = -0.5*forces_rxn.Fy - forces_rxn.Mxx/flatcar_tc_length

        # Loop over hill & sag cases, then take worst strain case
        strainLE = np.zeros((self.n_span, 2))
        strainTE = np.zeros((self.n_span, 2))
        for k in range(2):
            # Element shear and bending, one per element, with conversion to profile c.s. using Hansen's notation
            # Zero-ing out axial stress as there shouldn't be any for pure beam bending
            Fz = 0.0 #np.r_[-forces.Nx[ 0,0],  forces.Nx[ 0, 1::2]]
            M1 = np.r_[-forces.Myy[k, 0],  forces.Myy[k, 1::2]]
            M2 = np.r_[ forces.Mzz[k, 0], -forces.Mzz[k, 1::2]]

            # compute strain at the two points
            strainLE[:,k] = -(M1/EI11*le2 - M2/EI22*le1 + Fz/EA)
            strainTE[:,k] = -(M1/EI11*te2 - M2/EI22*te1 + Fz/EA)

        # Find best points for middle reaction and formulate as constraints
        constr_derailV_8axle = (np.abs(RF_derailV.T) / (0.5 * mass_car_8axle * gravity)) / max_LV
        constr_derailV_4axle = (np.abs(RF_derailV.T) / (0.5 * mass_car_4axle * gravity)) / max_LV

        outputs['constr_LV_4axle_vert'] = constr_derailV_4axle[0,:]
        outputs['constr_LV_8axle_vert'] = constr_derailV_8axle[0,:]

        # Strain constraint outputs
        outputs['constr_strainLE'] = strainLE / max_strains
        outputs['constr_strainTE'] = strainTE / max_strains
        """
