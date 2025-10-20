import copy

import numpy as np
from openmdao.api import Group, ExplicitComponent
from scipy.interpolate import PchipInterpolator

from wisdem.precomp import PreComp, Profile, CompositeSection, Orthotropic2DMaterial
from wisdem.commonse.utilities import arc_length
from wisdem.precomp.precomp_to_beamdyn import pc2bd_K, pc2bd_I, TransformCrossSectionMatrix
import logging
logger = logging.getLogger("wisdem/weis")


class RunPreComp(ExplicitComponent):
    # Openmdao component to run precomp and generate the elastic properties of a wind turbine blade
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]
        self.n_webs = n_webs = rotorse_options["n_webs"]
        self.n_layers = n_layers = rotorse_options["n_layers"]
        self.n_xy = n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        mat_init_options = self.options["modeling_options"]["materials"]
        self.n_mat = n_mat = mat_init_options["n_mat"]
        self.verbosity = self.options["modeling_options"]["General"]["verbosity"]

        self.te_ss_var = rotorse_options["te_ss"]
        self.te_ps_var = rotorse_options["te_ps"]
        self.spar_cap_ss_var = rotorse_options["spar_cap_ss"]
        self.spar_cap_ps_var = rotorse_options["spar_cap_ps"]

        # Outer geometry
        self.add_input(
            "r",
            val=np.zeros(n_span),
            units="m",
            desc="radial locations where blade is defined (should be increasing and not go all the way to hub or tip)",
        )
        self.add_input(
            "theta",
            val=np.zeros(n_span),
            units="deg",
            desc="Twist angle at each section (positive decreases angle of attack)",
        )
        self.add_input("chord", val=np.zeros(n_span), units="m", desc="chord length at each section")
        self.add_input(
            "section_offset_y",
            val=np.zeros(n_span),
            units="m",
            desc="1D array of the airfoil position relative to the reference axis, specifying the distance in meters along the chordline from the reference axis to the leading edge. 0 means that the airfoil is pinned at the leading edge, a positive offset means that the leading edge is upstream of the reference axis in local chordline coordinates, and a negative offset that the leading edge aft of the reference axis.",
        )
        self.add_input("precurve", val=np.zeros(n_span), units="m", desc="precurve at each section")
        self.add_input("presweep", val=np.zeros(n_span), units="m", desc="presweep at each section")
        self.add_input(
            "coord_xy_interp",
            val=np.zeros((n_span, n_xy, 2)),
            desc="3D array of the non-dimensional x and y airfoil coordinates of the airfoils interpolated along span for n_span stations.",
        )

        # Rotor configuration
        self.add_input(
            "uptilt", val=0.0, units="deg", desc="Nacelle uptilt angle. A standard machine has positive values."
        )
        self.add_discrete_input("n_blades", val=3, desc="Number of blades of the rotor.")

        # Inner structure
        self.add_input(
            "web_start_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional start point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1. The first dimension represents each entry along blade span, the second dimension represents each web.",
        )
        self.add_input(
            "web_end_nd",
            val=np.zeros((n_webs, n_span)),
            desc="2D array of the non-dimensional end point defined along the outer profile of a web. The TE suction side is 0, the TE pressure side is 1.  The first dimension represents each entry along blade span, the second dimension represents each web.",
        )
        self.add_input(
            "layer_thickness",
            val=np.zeros((n_layers, n_span)),
            units="m",
            desc="2D array of the thickness of the layers of the blade structure. The first dimension represents each entry along blade span, the second dimension represents each layer.",
        )
        self.add_input(
            "layer_start_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the start_nd_arc of the anchors. The first dimension represents each entry along blade span, the second dimension represents each layer.",
        )
        self.add_input(
            "layer_end_nd",
            val=np.zeros((n_layers, n_span)),
            desc="2D array of the end_nd_arc of the anchors. The first dimension represents each entry along blade span, the second dimension represents each layer.",
        )
        self.add_input(
            "fiber_orientation",
            val=np.zeros((n_layers, n_span)),
            units="deg",
            desc="2D array of the orientation of the layers of the blade structure. The first dimension represents each entry along blade span, the second dimension represents each layer.",
        )
        self.add_discrete_input(
            "build_layer",
            val=-np.ones(n_layers),
            desc="1D array of boolean values indicating how to build a layer.",
        )

        # Materials
        self.add_discrete_input("mat_name", val=n_mat * [""], desc="1D array of names of materials.")
        self.add_discrete_input(
            "orth",
            val=np.zeros(n_mat),
            desc="1D array of flags to set whether a material is isotropic (0) or orthtropic (1). Each entry represents a material.",
        )
        self.add_input(
            "E",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the Youngs moduli of the materials. Each row represents a material, the three columns represent E11, E22 and E33.",
        )
        self.add_input(
            "G",
            val=np.zeros([n_mat, 3]),
            units="Pa",
            desc="2D array of the shear moduli of the materials. Each row represents a material, the three columns represent G12, G13 and G23.",
        )
        self.add_input(
            "nu",
            val=np.zeros([n_mat, 3]),
            desc="2D array of the Poisson ratio of the materials. Each row represents a material, the three columns represent nu12, nu13 and nu23.",
        )
        self.add_input(
            "rho",
            val=np.zeros(n_mat),
            units="kg/m**3",
            desc="1D array of the density of the materials. For composites, this is the density of the laminate.",
        )

        self.add_input(
            "joint_position",
            val=0.0,
            desc="Spanwise position of the segmentation joint.",
        )
        self.add_input("joint_mass", val=0.0, units="kg", desc="Mass of the joint.")

        # Outputs - Distributed beam properties
        self.add_output("z", val=np.zeros(n_span), units="m", desc="locations of properties along beam")
        self.add_output("A", val=np.zeros(n_span), units="m**2", desc="cross sectional area")
        self.add_output("EA", val=np.zeros(n_span), units="N", desc="axial stiffness")
        self.add_output(
            "EIxx",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section lag (edgewise) bending stiffness about the XE axis",
        )
        self.add_output(
            "EIyy",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section flap bending stiffness about the YE axis",
        )
        self.add_output("EIxy", val=np.zeros(n_span), units="N*m**2", desc="Coupled flap-lag stiffness with respect to the XE-YE frame")
        self.add_output("EA_EIxx", val=np.zeros(n_span), units="N*m", desc="Coupled axial-lag stiffness with respect to the XE-YE frame")
        self.add_output("EA_EIyy", val=np.zeros(n_span), units="N*m", desc="Coupled axial-flap stiffness with respect to the XE-YE frame")
        self.add_output("EIxx_GJ", val=np.zeros(n_span), units="N*m**2", desc="Coupled lag-torsion stiffness with respect to the XE-YE frame")
        self.add_output("EIyy_GJ", val=np.zeros(n_span), units="N*m**2", desc="Coupled flap-torsion stiffness with respect to the XE-YE frame ")
        self.add_output("EA_GJ", val=np.zeros(n_span), units="N*m", desc="Coupled axial-torsion stiffness")
        self.add_output(
            "GJ",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section torsional stiffness with respect to the XE-YE frame",
        )
        self.add_output("rhoA", val=np.zeros(n_span), units="kg/m", desc="Section mass per unit length")
        self.add_output("rhoJ", val=np.zeros(n_span), units="kg*m", desc="polar mass moment of inertia per unit length")
        self.add_output(
            "Tw_iner",
            val=np.zeros(n_span),
            units="deg",
            desc="Orientation of the section principal inertia axes with respect the blade reference plane",
        )
        self.add_output(
            "x_tc",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the tension-center offset with respect to the XR-YR axes",
        )
        self.add_output(
            "y_tc",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section tension-center with respect to the XR-YR axes",
        )
        self.add_output(
            "x_sc",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the shear-center offset with respect to the XR-YR axes",
        )
        self.add_output(
            "y_sc",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section shear-center with respect to the reference frame, XR-YR",
        )
        self.add_output(
            "x_cg",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the center-of-mass offset with respect to the XR-YR axes",
        )
        self.add_output(
            "y_cg",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section center of mass with respect to the XR-YR axes",
        )
        self.add_output(
            "flap_iner",
            val=np.zeros(n_span),
            units="kg/m",
            desc="Section flap inertia about the Y_G axis per unit length.",
        )
        self.add_output(
            "edge_iner",
            val=np.zeros(n_span),
            units="kg/m",
            desc="Section lag inertia about the X_G axis per unit length",
        )
        # self.add_output('eps_crit_spar',    val=np.zeros(n_span), desc='critical strain in spar from panel buckling calculation')
        # self.add_output('eps_crit_te',      val=np.zeros(n_span), desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_output(
            "xu_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_output(
            "xl_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_output(
            "yu_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_output(
            "yl_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_output(
            "xu_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_output(
            "xl_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )
        self.add_output(
            "yu_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_output(
            "yl_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )


        self.add_output(
            "sc_ss_mats",
            val=np.zeros((n_span, n_mat)),
            desc="spar cap, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis",
        )
        self.add_output(
            "sc_ps_mats",
            val=np.zeros((n_span, n_mat)),
            desc="spar cap, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis",
        )
        self.add_output(
            "te_ss_mats",
            val=np.zeros((n_span, n_mat)),
            desc="trailing edge reinforcement, suction side,  boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis",
        )
        self.add_output(
            "te_ps_mats",
            val=np.zeros((n_span, n_mat)),
            desc="trailing edge reinforcement, pressure side, boolean of materials in each composite layer spanwise, passed as floats for differentiablity, used for Fatigue Analysis",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        ##############################
        def region_stacking(
            i,
            idx,
            start_nd_arc,
            end_nd_arc,
            layer_name,
            layer_thickness,
            fiber_orientation,
            layer_mat,
            material_dict,
            materials,
            region_loc,
        ):
            # Receive start and end of composite sections chordwise, find which composites layers are in each
            # chordwise regions, generate the precomp composite class instance

            # error handling to makes sure there were no numeric errors causing values very close too, but not exactly, 0 or 1
            start_nd_arc = [
                0.0 if start_nd_arci != 0.0 and np.isclose(start_nd_arci, 0.0) else start_nd_arci
                for start_nd_arci in start_nd_arc
            ]
            end_nd_arc = [
                0.0 if end_nd_arci != 0.0 and np.isclose(end_nd_arci, 0.0) else end_nd_arci
                for end_nd_arci in end_nd_arc
            ]
            start_nd_arc = [
                1.0 if start_nd_arci != 1.0 and np.isclose(start_nd_arci, 1.0) else start_nd_arci
                for start_nd_arci in start_nd_arc
            ]
            end_nd_arc = [
                1.0 if end_nd_arci != 1.0 and np.isclose(end_nd_arci, 1.0) else end_nd_arci
                for end_nd_arci in end_nd_arc
            ]

            # region end points
            dp = sorted(list(set(start_nd_arc + end_nd_arc)))

            # initialize
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            # loop through division points, find what layers make up the stack between those bounds
            for i_reg, (dp0, dp1) in enumerate(zip(dp[0:-1], dp[1:])):
                n_pliesi = []
                thki = []
                thetai = []
                mati = []
                for i_sec, start_nd_arci, end_nd_arci in zip(idx, start_nd_arc, end_nd_arc):
                    name = layer_name[i_sec]
                    if start_nd_arci <= dp0 and end_nd_arci >= dp1:
                        if name in region_loc.keys():
                            if region_loc[name][i] == None:
                                region_loc[name][i] = [i_reg]
                            else:
                                region_loc[name][i].append(i_reg)

                        n_pliesi.append(1.0)
                        thki.append(layer_thickness[i_sec])
                        if fiber_orientation[i_sec] == None:
                            thetai.append(0.0)
                        else:
                            thetai.append(fiber_orientation[i_sec])
                        mati.append(material_dict[layer_mat[i_sec]])

                n_plies.append(np.array(n_pliesi))
                thk.append(np.array(thki))
                theta.append(np.array(thetai))
                mat_idx.append(np.array(mati))

            # print('----------------------')
            # print('dp', dp)
            # print('n_plies', n_plies)
            # print('thk', thk)
            # print('theta', theta)
            # print('mat_idx', mat_idx)
            # print('materials', materials)

            sec = CompositeSection(dp, n_plies, thk, theta, mat_idx, materials)
            return sec, region_loc
            ##############################

        def web_stacking(
            i,
            web_idx,
            web_start_nd_arc,
            web_end_nd_arc,
            layer_thickness,
            fiber_orientation,
            layer_mat,
            material_dict,
            materials,
            flatback,
            upperCSi,
        ):
            dp = []
            n_plies = []
            thk = []
            theta = []
            mat_idx = []

            if len(web_idx) > 0:
                dp = np.mean((np.abs(web_start_nd_arc), np.abs(web_start_nd_arc)), axis=0).tolist()

                dp_all = [
                    [start_nd_arci, end_nd_arci]
                    for start_nd_arci, end_nd_arci in zip(web_start_nd_arc, web_end_nd_arc)
                ]
                _, web_ids = np.unique(dp_all, axis=0, return_inverse=True)
                for webi in np.unique(web_ids):
                    # store variable values (thickness, orientation, material) for layers that make up each web, based on the mapping array web_ids
                    n_pliesi = [1.0 for i_reg, web_idi in zip(web_idx, web_ids) if web_idi == webi]
                    thki = [layer_thickness[i_reg] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi == webi]
                    thetai = [fiber_orientation[i_reg] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi == webi]
                    thetai = [0.0 if theta_ij == None else theta_ij for theta_ij in thetai]
                    mati = [
                        material_dict[layer_mat[i_reg]] for i_reg, web_idi in zip(web_idx, web_ids) if web_idi == webi
                    ]

                    n_plies.append(np.array(n_pliesi))
                    thk.append(np.array(thki))
                    theta.append(np.array(thetai))
                    mat_idx.append(np.array(mati))

            if flatback:
                dp.append(1.0)
                n_plies.append(upperCSi.n_plies[-1])
                thk.append(upperCSi.t[-1])
                theta.append(upperCSi.theta[-1])
                mat_idx.append(upperCSi.mat_idx[-1])

            dp_out = sorted(list(set(dp)))

            sec = CompositeSection(dp_out, n_plies, thk, theta, mat_idx, materials)
            return sec
            ##############################

        layer_name = self.options["modeling_options"]["WISDEM"]["RotorSE"]["layer_name"]
        layer_mat = self.options["modeling_options"]["WISDEM"]["RotorSE"]["layer_mat"]

        upperCS = [None] * self.n_span
        lowerCS = [None] * self.n_span
        websCS = [None] * self.n_span
        profile = [None] * self.n_span
        chord = inputs["chord"]
        area = np.zeros_like(chord)
        region_loc_vars = [self.te_ss_var, self.te_ps_var, self.spar_cap_ss_var, self.spar_cap_ps_var]

        region_loc_ss = {}  # track precomp regions for user selected composite layers
        region_loc_ps = {}
        for var in region_loc_vars:
            region_loc_ss[var] = [None] * self.n_span
            region_loc_ps[var] = [None] * self.n_span

        ## Materials
        material_dict = {}
        materials = []
        for i_mat in range(self.n_mat):
            materials.append(
                Orthotropic2DMaterial(
                    inputs["E"][i_mat, 0],
                    inputs["E"][i_mat, 1],
                    inputs["G"][i_mat, 0],
                    inputs["nu"][i_mat, 0],
                    inputs["rho"][i_mat],
                    discrete_inputs["mat_name"][i_mat],
                )
            )
            material_dict[discrete_inputs["mat_name"][i_mat]] = i_mat

        ## Spanwise
        for i in range(self.n_span):
            # time0 = time.time()

            ## Profiles
            # rotate
            profile_i = inputs["coord_xy_interp"][i, :, :]
            profile_i_rot = profile_i

            # normalize
            profile_i_rot[:, 0] -= min(profile_i_rot[:, 0])
            profile_i_rot = profile_i_rot / max(profile_i_rot[:, 0])

            profile_i_rot_precomp = copy.copy(profile_i_rot)
            idx_s = 0
            idx_le_precomp = np.argmax(profile_i_rot_precomp[:, 0])
            if idx_le_precomp != 0:
                if profile_i_rot_precomp[0, 0] == profile_i_rot_precomp[-1, 0]:
                    idx_s = 1
                profile_i_rot_precomp = np.vstack(
                    (profile_i_rot_precomp[idx_le_precomp:], profile_i_rot_precomp[idx_s:idx_le_precomp, :])
                )
            profile_i_rot_precomp[:, 1] -= profile_i_rot_precomp[np.argmin(profile_i_rot_precomp[:, 0]), 1]

            # # renormalize
            profile_i_rot_precomp[:, 0] -= min(profile_i_rot_precomp[:, 0])
            profile_i_rot_precomp = profile_i_rot_precomp / max(profile_i_rot_precomp[:, 0])

            if profile_i_rot_precomp[-1, 0] != 1.0:
                profile_i_rot_precomp = np.vstack((profile_i_rot_precomp, profile_i_rot_precomp[0, :]))

            # 'web' at trailing edge needed for flatback airfoils
            if (
                profile_i_rot_precomp[0, 1] != profile_i_rot_precomp[-1, 1]
                and profile_i_rot_precomp[0, 0] == profile_i_rot_precomp[-1, 0]
            ):
                flatback = True
            else:
                flatback = False

            profile[i] = Profile.initWithTEtoTEdata(profile_i_rot_precomp[:, 0], profile_i_rot_precomp[:, 1])

            # import matplotlib.pyplot as plt
            # plt.plot(profile_i_rot_precomp[:,0], profile_i_rot_precomp[:,1])
            # plt.axis('equal')
            # plt.title(i)
            # plt.show()

            idx_le = np.argmin(profile_i_rot[:, 0])

            profile_i_arc = arc_length(profile_i_rot)
            arc_L = profile_i_arc[-1]
            profile_i_arc /= arc_L
            arc_L_m = arc_L * chord[i]

            loc_LE = profile_i_arc[idx_le]
            len_PS = 1.0 - loc_LE

            ## Composites
            ss_idx = []
            ss_start_nd_arc = []
            ss_end_nd_arc = []
            ps_idx = []
            ps_start_nd_arc = []
            ps_end_nd_arc = []
            web_start_nd_arc = []
            web_end_nd_arc = []
            web_idx = []

            # Determine spanwise composite layer elements that are non-zero at this spanwise location,
            # determine their chord-wise start and end location on the pressure and suctions side

            spline_arc2xnd = PchipInterpolator(profile_i_arc, profile_i_rot[:, 0])

            # time1 = time.time()
            for idx_sec in range(self.n_layers):
                if discrete_inputs["build_layer"][idx_sec] >= 0:
                    if inputs["layer_thickness"][idx_sec, i] > 1.0e-6:
                        area[i] += arc_L_m * (inputs["layer_end_nd"][idx_sec, i] - 
                                              inputs["layer_start_nd"][idx_sec, i]) * (
                                              inputs["layer_thickness"][idx_sec, i])
                        if inputs["layer_start_nd"][idx_sec, i] < loc_LE or inputs["layer_end_nd"][idx_sec, i] < loc_LE:
                            ss_idx.append(idx_sec)
                            if inputs["layer_start_nd"][idx_sec, i] < loc_LE:
                                # ss_start_nd_arc.append(sec['start_nd_arc']['values'][i])
                                ss_end_nd_arc_temp = float(spline_arc2xnd(inputs["layer_start_nd"][idx_sec, i]))
                                if ss_end_nd_arc_temp > 1:
                                    logger.debug(
                                        "Error in the definition of material "
                                        + layer_name[idx_sec]
                                        + ". It cannot fit in the section number "
                                        + str(i)
                                        + " at span location "
                                        + str(inputs["r"][i] / inputs["r"][-1] * 100.0)
                                        + " %. Variable ss_end_nd_arc_temp was equal "
                                        + " to "
                                        + str(ss_end_nd_arc_temp)
                                        + " and is not set to 1"
                                    )
                                    ss_end_nd_arc_temp = 1.
                                if ss_end_nd_arc_temp < 0:
                                    logger.debug(
                                        "Error in the definition of material "
                                        + layer_name[idx_sec]
                                        + ". It cannot fit in the section number "
                                        + str(i)
                                        + " at span location "
                                        + str(inputs["r"][i] / inputs["r"][-1] * 100.0)
                                        + " %. Variable ss_end_nd_arc_temp was equal "
                                        + " to "
                                        + str(ss_end_nd_arc_temp)
                                        + " and is not set to 0"
                                    )
                                    ss_end_nd_arc_temp = 0.
                                if ss_end_nd_arc_temp == profile_i_rot[0, 0] and profile_i_rot[0, 0] != 1.0:
                                    ss_end_nd_arc_temp = 1.0
                                ss_end_nd_arc.append(ss_end_nd_arc_temp)
                            else:
                                ss_end_nd_arc.append(1.0)
                            # ss_end_nd_arc.append(min(sec['end_nd_arc']['values'][i], loc_LE)/loc_LE)
                            if inputs["layer_end_nd"][idx_sec, i] < loc_LE:
                                ss_start_nd_arc.append(float(spline_arc2xnd(inputs["layer_end_nd"][idx_sec, i])))
                            else:
                                ss_start_nd_arc.append(0.0)

                        if inputs["layer_start_nd"][idx_sec, i] > loc_LE or inputs["layer_end_nd"][idx_sec, i] > loc_LE:
                            ps_idx.append(idx_sec)
                            # ps_start_nd_arc.append((max(sec['start_nd_arc']['values'][i], loc_LE)-loc_LE)/len_PS)
                            # ps_end_nd_arc.append((min(sec['end_nd_arc']['values'][i], 1.)-loc_LE)/len_PS)

                            if (
                                inputs["layer_start_nd"][idx_sec, i] > loc_LE
                                and inputs["layer_end_nd"][idx_sec, i] < loc_LE
                            ):
                                # ps_start_nd_arc.append(float(remap2grid(profile_i_arc, profile_i_rot[:,0], sec['start_nd_arc']['values'][i])))
                                ps_end_nd_arc.append(1.0)
                            else:
                                ps_end_nd_arc_temp = float(spline_arc2xnd(inputs["layer_end_nd"][idx_sec, i]))
                                if (
                                    np.isclose(ps_end_nd_arc_temp, profile_i_rot[-1, 0], atol=1.0e-2)
                                    and profile_i_rot[-1, 0] != 1.0
                                ):
                                    ps_end_nd_arc_temp = 1.0
                                if ps_end_nd_arc_temp > 1.0:
                                    ps_end_nd_arc_temp = 1.0
                                ps_end_nd_arc.append(ps_end_nd_arc_temp)
                            if inputs["layer_start_nd"][idx_sec, i] < loc_LE:
                                ps_start_nd_arc.append(0.0)
                            else:
                                ps_start_nd_arc.append(float(spline_arc2xnd(inputs["layer_start_nd"][idx_sec, i])))
                else:
                    target_idx = - discrete_inputs["build_layer"][idx_sec] - 1

                    if inputs["layer_thickness"][idx_sec, i] > 1.0e-6:
                        web_idx.append(idx_sec)

                        web_start_nd = inputs["web_start_nd"][int(target_idx), i]
                        web_end_nd = inputs["web_end_nd"][int(target_idx), i]

                        start_nd_arc = float(spline_arc2xnd(web_start_nd))
                        end_nd_arc = float(spline_arc2xnd(web_end_nd))

                        web_start_nd_arc.append(start_nd_arc)
                        web_end_nd_arc.append(end_nd_arc)

                        # Compute height the webs along span
                        id_start = np.argmin(abs(profile_i_arc - web_start_nd))
                        id_end = np.argmin(abs(profile_i_arc - web_end_nd))
                        web_height = np.sqrt((profile_i[id_start, 0] - profile_i[id_end, 0])**2 +
                                             (profile_i[id_start, 1] - profile_i[id_end, 1])**2) * (
                                             chord[i])

                        area[i] += web_height * inputs["layer_thickness"][idx_sec, i]

            # time1 = time.time() - time1
            # print(time1)

            # cap layer starts and ends within 0 and 1
            ss_start_nd_arc = [max(0, min(1, value)) for value in ss_start_nd_arc]
            ss_end_nd_arc = [max(0, min(1, value)) for value in ss_end_nd_arc]
            # generate the Precomp composite stacks for chordwise regions
            upperCS[i], region_loc_ss = region_stacking(
                i,
                ss_idx,
                ss_start_nd_arc,
                ss_end_nd_arc,
                layer_name,
                inputs["layer_thickness"][:, i],
                inputs["fiber_orientation"][:, i],
                layer_mat,
                material_dict,
                materials,
                region_loc_ss,
            )
            lowerCS[i], region_loc_ps = region_stacking(
                i,
                ps_idx,
                ps_start_nd_arc,
                ps_end_nd_arc,
                layer_name,
                inputs["layer_thickness"][:, i],
                inputs["fiber_orientation"][:, i],
                layer_mat,
                material_dict,
                materials,
                region_loc_ps,
            )
            if len(web_idx) > 0 or flatback:
                websCS[i] = web_stacking(
                    i,
                    web_idx,
                    web_start_nd_arc,
                    web_end_nd_arc,
                    inputs["layer_thickness"][:, i],
                    inputs["fiber_orientation"][:, i],
                    layer_mat,
                    material_dict,
                    materials,
                    flatback,
                    upperCS[i],
                )
            else:
                websCS[i] = CompositeSection([], [], [], [], [], [])

        sector_idx_spar_cap_ss = [
            None if regs == None else regs[int(len(regs) / 2)] for regs in region_loc_ss[self.spar_cap_ss_var]
        ]
        sector_idx_spar_cap_ps = [
            None if regs == None else regs[int(len(regs) / 2)] for regs in region_loc_ps[self.spar_cap_ps_var]
        ]
        sector_idx_te_ss = [
            None if regs == None else regs[int(len(regs) / 2)] for regs in region_loc_ss[self.te_ss_var]
        ]
        sector_idx_te_ps = [
            None if regs == None else regs[int(len(regs) / 2)] for regs in region_loc_ps[self.te_ps_var]
        ]

        # Get Beam Properties
        beam = PreComp(
            inputs["r"],
            inputs["chord"],
            inputs["theta"],
            inputs["section_offset_y"]/inputs["chord"],
            inputs["precurve"],
            inputs["presweep"],
            profile,
            materials,
            upperCS,
            lowerCS,
            websCS,
            sector_idx_spar_cap_ps,
            sector_idx_spar_cap_ss,
            sector_idx_te_ps,
            sector_idx_te_ss,
        )
        (
            EIxx,
            EIyy,
            GJ,
            EA,
            EIxy,
            EA_EIxx,
            EA_EIyy,
            EIxx_GJ,
            EIyy_GJ,
            EA_GJ,
            rhoA,
            _,
            rhoJ,
            Tw_iner,
            flap_iner,
            edge_iner,
            x_tc,
            y_tc,
            x_sc,
            y_sc,
            x_cg,
            y_cg,
        ) = beam.sectionProperties()

        # outputs['eps_crit_spar'] = beam.panelBucklingStrain(sector_idx_spar_cap_ss)
        # outputs['eps_crit_te'] = beam.panelBucklingStrain(sector_idx_te_ss)

        xu_spar, xl_spar, yu_spar, yl_spar = beam.criticalStrainLocations(
            sector_idx_spar_cap_ss, sector_idx_spar_cap_ps
        )
        xu_te, xl_te, yu_te, yl_te = beam.criticalStrainLocations(sector_idx_te_ss, sector_idx_te_ps)

        # Store what materials make up the composites for SC/TE
        for i in range(self.n_span):
            for j in range(self.n_mat):
                if sector_idx_spar_cap_ss[i]:
                    if j in upperCS[i].mat_idx[sector_idx_spar_cap_ss[i]]:
                        outputs["sc_ss_mats"][i, j] = 1.0
                if sector_idx_spar_cap_ps[i]:
                    if j in lowerCS[i].mat_idx[sector_idx_spar_cap_ps[i]]:
                        outputs["sc_ps_mats"][i, j] = 1.0
                if sector_idx_te_ss[i]:
                    if j in upperCS[i].mat_idx[sector_idx_te_ss[i]]:
                        outputs["te_ss_mats"][i, j] = 1.0
                if sector_idx_te_ps[i]:
                    if j in lowerCS[i].mat_idx[sector_idx_te_ps[i]]:
                        outputs["te_ps_mats"][i, j] = 1.0
        rhoA_joint = copy.copy(rhoA)
        if inputs["joint_mass"] > 0.0:
            s = (inputs["r"] - inputs["r"][0]) / (inputs["r"][-1] - inputs["r"][0])
            id_station = np.argmin(abs(inputs["joint_position"] - s))
            span = np.average(
                [
                    inputs["r"][id_station] - inputs["r"][id_station - 1],
                    inputs["r"][id_station + 1] - inputs["r"][id_station],
                ]
            )
            rhoA_joint[id_station] += inputs["joint_mass"][0] / span

        outputs["z"] = inputs["r"]
        outputs["EIxx"] = EIxx
        outputs["EIyy"] = EIyy
        outputs["GJ"] = GJ
        outputs["EA"] = EA
        outputs["EIxy"] = EIxy
        outputs["EA_EIxx"] = EA_EIxx
        outputs["EA_EIyy"] = EA_EIyy
        outputs["EIxx_GJ"] = EIxx_GJ
        outputs["EIyy_GJ"] = EIyy_GJ
        outputs["EA_GJ"] = EA_GJ
        outputs["rhoA"] = rhoA_joint
        outputs["A"] = area
        outputs["rhoJ"] = rhoJ
        outputs["Tw_iner"] = Tw_iner
        outputs["flap_iner"] = flap_iner
        outputs["edge_iner"] = edge_iner

        outputs["x_tc"] = x_tc
        outputs["y_tc"] = y_tc
        outputs["x_sc"] = x_sc
        outputs["y_sc"] = y_sc
        outputs["x_cg"] = x_cg
        outputs["y_cg"] = y_cg

        outputs["xu_spar"] = xu_spar
        outputs["xl_spar"] = xl_spar
        outputs["yu_spar"] = yu_spar
        outputs["yl_spar"] = yl_spar
        outputs["xu_te"] = xu_te
        outputs["xl_te"] = xl_te
        outputs["yu_te"] = yu_te
        outputs["yl_te"] = yl_te



class TotalBladeProperties(ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        rotorse_options = self.options["modeling_options"]["WISDEM"]["RotorSE"]
        self.n_span = n_span = rotorse_options["n_span"]

        # Inputs
        self.add_input(
            "r",
            val=np.zeros(n_span),
            units="m",
            desc="radial locations where blade is defined (should be increasing and not go all the way to hub or tip)",
        )
        self.add_input("rhoA", val=np.zeros(n_span), units="kg/m", desc="mass per unit length")
        self.add_discrete_input("n_blades", val=3, desc="Number of blades of the rotor.")

        # Outputs - Overall beam properties
        self.add_output("blade_mass", val=0.0, units="kg", desc="mass of one blade")
        self.add_output("blade_span_cg", val=0.0, units="m", desc="Distance along the blade span for its center of gravity")
        self.add_output(
            "blade_moment_of_inertia", val=0.0, units="kg*m**2", desc="mass moment of inertia of blade about hub"
        )
        self.add_output("mass_all_blades", val=0.0, units="kg", desc="mass of all blades")
        self.add_output(
            "I_all_blades",
            shape=6,
            units="kg*m**2",
            desc="mass moments of inertia of all blades in hub c.s. order:Ixx, Iyy, Izz, Ixy, Ixz, Iyz",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        rhoA_joint = inputs["rhoA"]
        try:
            # Numpy v1/2 clash
            blade_mass = np.trapezoid(rhoA_joint, inputs["r"])
            blade_span_cg = np.trapezoid(rhoA_joint * inputs["r"], inputs["r"]) / blade_mass
            blade_moment_of_inertia = np.trapezoid(rhoA_joint * inputs["r"] ** 2.0, inputs["r"])
        except AttributeError:
            blade_mass = np.trapz(rhoA_joint, inputs["r"])
            blade_span_cg = np.trapz(rhoA_joint * inputs["r"], inputs["r"]) / blade_mass
            blade_moment_of_inertia = np.trapz(rhoA_joint * inputs["r"] ** 2.0, inputs["r"])
        # tilt = inputs["uptilt"]
        n_blades = discrete_inputs["n_blades"]
        mass_all_blades = n_blades * blade_mass
        Ibeam = n_blades * blade_moment_of_inertia
        Ixx = Ibeam
        Iyy = Ibeam / 2.0  # azimuthal average for 2 blades, exact for 3+
        Izz = Ibeam / 2.0
        Ixy = 0.0
        Ixz = 0.0
        Iyz = 0.0  # azimuthal average for 2 blades, exact for 3+
        I_all_blades = np.r_[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]

        outputs["blade_mass"] = blade_mass
        outputs["blade_span_cg"] = blade_span_cg
        outputs["blade_moment_of_inertia"] = blade_moment_of_inertia
        outputs["mass_all_blades"] = mass_all_blades
        outputs["I_all_blades"] = I_all_blades
                
class generate_KI(ExplicitComponent):

    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        
        modopt = self.options['modeling_options']
        rotorse_options  = modopt['WISDEM']['RotorSE']
        self.n_span = n_span = rotorse_options['n_span']

        self.add_input(
            "EA", 
            val=np.zeros(n_span), 
            units="N", 
            desc="Axial stiffness at the elastic center, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "EIxx",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section lag (edgewise) bending stiffness about the XE axis, using the convention of WISDEM solver PreComp.",
        )
        self.add_input("EIyy",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section flap bending stiffness about the YE axis, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "EIxy", 
            val=np.zeros(n_span), 
            units="N*m**2", 
            desc="Coupled flap-lag stiffness with respect to the XE-YE frame, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "EA_EIxx", 
            val=np.zeros(n_span), 
            units="N*m", 
            desc="Coupled axial-lag stiffness with respect to the XE-YE frame, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "EA_EIyy", 
            val=np.zeros(n_span), 
            units="N*m", 
            desc="Coupled axial-flap stiffness with respect to the XE-YE frame, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "EIxx_GJ", 
            val=np.zeros(n_span), 
            units="N*m**2", 
            desc="Coupled lag-torsion stiffness with respect to the XE-YE frame, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "EIyy_GJ", 
            val=np.zeros(n_span), 
            units="N*m**2", 
            desc="Coupled flap-torsion stiffness with respect to the XE-YE frame, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "EA_GJ", 
            val=np.zeros(n_span), 
            units="N*m", 
            desc="Coupled axial-torsion stiffness with respect to the XE-YE frame, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "GJ",
            val=np.zeros(n_span),
            units="N*m**2",
            desc="Section torsion stiffness at the elastic center, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "rhoA", 
            val=np.zeros(n_span), 
            units="kg/m", 
            desc="Section mass per unit length, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "rhoJ", 
            val=np.zeros(n_span), 
            units="kg*m", 
            desc="polar mass moment of inertia per unit length, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "Tw_iner",
            val=np.zeros(n_span),
            units="deg",
            desc="Orientation of the section principal inertia axes with respect the blade reference plane, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "x_tc",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the tension-center offset with respect to the XR-YR axes, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "y_tc",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section tension-center with respect to the XR-YR axes, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "x_cg",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the center-of-mass offset with respect to the XR-YR axes, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "y_cg",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section center of mass with respect to the XR-YR axes, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "flap_iner",
            val=np.zeros(n_span),
            units="kg/m",
            desc="Section flap inertia about the Y_G axis per unit length, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "edge_iner",
            val=np.zeros(n_span),
            units="kg/m",
            desc="Section lag inertia about the X_G axis per unit length, using the convention of WISDEM solver PreComp.",
        )
        self.add_input(
            "theta",
            val=np.zeros(n_span),
            units="deg",
            desc="Aerodynamic twist angle at each section (positive decreases angle of attack)",
        )

        # Outputs are 6x6 K and I matrices at the center of the windIO reference axes
        self.add_output(
            "K",
            val=np.zeros((n_span,6,6)), 
            desc="Stiffness matrix at the center of the windIO reference axes."
        )
        self.add_output(
            "I",
            val=np.zeros((n_span,6,6)), 
            desc="Inertia matrix at the center of the windIO reference axes."
        )

    def compute(self, inputs, outputs):

        # Initialize empty 6x6 K and I matrices
        K = np.zeros((self.n_span,6,6))
        I = np.zeros((self.n_span,6,6))
        EA = inputs["EA"]
        EIxx = inputs["EIxx"]
        EIyy = inputs["EIyy"]
        EIxy = inputs["EIxy"]
        GJ = inputs["GJ"]
        EA_EIxx = inputs["EA_EIxx"]
        EA_EIyy = inputs["EA_EIyy"]
        EIxx_GJ = inputs["EIxx_GJ"]
        EIyy_GJ = inputs["EIyy_GJ"]
        EA_GJ = inputs["EA_GJ"]
        rhoA = inputs["rhoA"]
        rhoJ = inputs["rhoJ"]
        Tw_iner = np.deg2rad(inputs["Tw_iner"])
        x_cg = inputs["x_cg"]
        y_cg = inputs["y_cg"]
        x_tc = inputs["x_tc"]
        y_tc = inputs["y_tc"]
        edge_iner = inputs["edge_iner"]
        flap_iner = inputs["flap_iner"]
        aero_twist =  np.deg2rad(inputs["theta"])

        for i in range(self.n_span):
            # Build stiffness matrix at the reference axis
            K[i,:,:] = pc2bd_K(
                EA[i],
                EIxx[i],
                EIyy[i],
                EIxy[i],
                EA_EIxx[i],
                EA_EIyy[i],
                EIxx_GJ[i],
                EIyy_GJ[i],
                EA_GJ[i],
                GJ[i],
                rhoJ[i],
                edge_iner[i],
                flap_iner[i],
                x_tc[i],
                y_tc[i],
                )
            # Build inertia matrix at the reference axis
            I[i,:,:] = pc2bd_I(
                rhoA[i],
                edge_iner[i],
                flap_iner[i],
                rhoJ[i],
                x_cg[i],
                y_cg[i],
                Tw_iner[i],
                aero_twist[i],
                )
        
        outputs["K"] = K
        outputs["I"] = I

class assemble_KI(ExplicitComponent):
    """"
    Assemble the stiffness and inertia matrices at the reference axis for the entire rotor blade"
    """
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        modopt = self.options['modeling_options']
        rotorse_options  = modopt['WISDEM']['RotorSE']
        self.n_span = n_span = rotorse_options['n_span']

        self.add_input("K11", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K11 element of the stiffness matrix along blade span. K11 corresponds to the shear stiffness along the x axis (in a blade, x points to the trailing edge)",
                            units="N")
        self.add_input("K22", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K22 element of the stiffness matrix along blade span. K22 corresponds to the shear stiffness along the y axis (in a blade, y points to the suction side)",
                            units="N")
        self.add_input("K33", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K33 element of the stiffness matrix along blade span. K33 corresponds to the axial stiffness along the z axis (in a blade, z runs along the span and points to the tip)",
                            units="N")
        self.add_input("K44", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K44 element of the stiffness matrix along blade span. K44 corresponds to the bending stiffness around the x axis (in a blade, x points to the trailing edge and K44 corresponds to the flapwise stiffness)",
                            units="N*m**2")
        self.add_input("K55", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K55 element of the stiffness matrix along blade span. K55 corresponds to the bending stiffness around the y axis (in a blade, y points to the suction side and K55 corresponds to the edgewise stiffness)",
                            units="N*m**2")
        self.add_input("K66", 
                            val=np.zeros(n_span),  
                            desc="Distribution of K66 element of the stiffness matrix along blade span. K66 corresponds to the torsional stiffness along the z axis (in a blade, z runs along the span and points to the tip)",
                            units="N*m**2")
        self.add_input("K12", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K12 element of the stiffness matrix along blade span. K12 is a cross term between shear terms",
                            units="N")
        self.add_input("K13", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K13 element of the stiffness matrix along blade span. K13 is a cross term shear - axial",
                            units="N")
        self.add_input("K14", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K14 element of the stiffness matrix along blade span. K14 is a cross term shear - bending",
                            units="N*m**2")
        self.add_input("K15", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K15 element of the stiffness matrix along blade span. K15 is a cross term shear - bending",
                            units="N*m**2")
        self.add_input("K16", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K16 element of the stiffness matrix along blade span. K16 is a cross term shear - torsion",
                            units="N*m**2")
        self.add_input("K23", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K23 element of the stiffness matrix along blade span. K23 is a cross term shear - axial",
                            units="N*m**2")
        self.add_input("K24", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K24 element of the stiffness matrix along blade span. K24 is a cross term shear - bending",
                            units="N/m**2")
        self.add_input("K25", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K25 element of the stiffness matrix along blade span. K25 is a cross term shear - bending",
                            units="N*m**2")
        self.add_input("K26", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K26 element of the stiffness matrix along blade span. K26 is a cross term shear - torsion",
                            units="N*m**2")
        self.add_input("K34", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K34 element of the stiffness matrix along blade span. K34 is a cross term axial - bending",
                            units="N*m**2")
        self.add_input("K35", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K35 element of the stiffness matrix along blade span. K35 is a cross term axial - bending",
                            units="N*m**2")
        self.add_input("K36", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K36 element of the stiffness matrix along blade span. K36 is a cross term axial - torsion",
                            units="N*m**2")
        self.add_input("K45", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K45 element of the stiffness matrix along blade span. K45 is a cross term flapwise bending - edgewise bending",
                            units="N*m**2")
        self.add_input("K46", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K46 element of the stiffness matrix along blade span. K46 is a cross term flapwise bending - torsion",
                            units="N*m**2")
        self.add_input("K56", 
                            val=np.zeros(n_span),  
                            desc="Distribution of the K56 element of the stiffness matrix along blade span. K56 is a cross term edgewise bending - torsion",
                            units="N*m**2")
        
        # mass matrix inputs
        self.add_input("mass", val=np.zeros(n_span),  desc="Mass per unit length along the beam, expressed in kilogram per meter", units="kg/m")
        self.add_input("cm_x", val=np.zeros(n_span),  desc="Distance between the reference axis and the center of mass along the x axis", units="m")
        self.add_input("cm_y", val=np.zeros(n_span),  desc="Distance between the reference axis and the center of mass along the y axis", units="m")
        self.add_input("i_edge", val=np.zeros(n_span),  desc="Edgewise mass moment of inertia per unit span (around y axis)", units="kg*m**2")
        self.add_input("i_flap", val=np.zeros(n_span),  desc="Flapwise mass moment of inertia per unit span (around x axis)", units="kg*m**2")
        self.add_input("i_plr", val=np.zeros(n_span),  desc="Polar moment of inertia per unit span (around z axis). Please note that for beam-like structures iplr must be equal to iedge plus iflap.", units="kg*m**2")
        self.add_input("i_cp", val=np.zeros(n_span),  desc="Sectional cross-product of inertia per unit span (cross term x y)", units="kg*m**2")

        # Outputs
        self.add_output("K", val=np.zeros((n_span,6,6)), desc="Stiffness matrix at the center of the windIO reference axes.")
        self.add_output("I", val=np.zeros((n_span,6,6)), desc="Inertia matrix at the center of the windIO reference axes.")

    def compute(self, inputs, outputs):
        # Initialize empty 6x6 K and I matrices
        K = np.zeros((self.n_span,6,6))
        I = np.zeros((self.n_span,6,6))

        # Extract inputs
        K11 = inputs["K11"]
        K22 = inputs["K22"]
        K33 = inputs["K33"]
        K44 = inputs["K44"]
        K55 = inputs["K55"]
        K66 = inputs["K66"]
        K12 = inputs["K12"]
        K13 = inputs["K13"]
        K14 = inputs["K14"]
        K15 = inputs["K15"]
        K16 = inputs["K16"]
        K23 = inputs["K23"]
        K24 = inputs["K24"]
        K25 = inputs["K25"]
        K26 = inputs["K26"]
        K34 = inputs["K34"]
        K35 = inputs["K35"]
        K36 = inputs["K36"]
        K45 = inputs["K45"]
        K46 = inputs["K46"]
        K56 = inputs["K56"]

        mass = inputs['mass']
        cm_x = inputs['cm_x']
        cm_y = inputs['cm_y']
        i_edge = inputs['i_edge']
        i_flap = inputs['i_flap']
        i_plr  = inputs['i_plr']
        i_cp   = inputs['i_cp']

        
        for i in range(self.n_span):
            # Assemble stiffness matrix at the reference axis
            k_matrix = np.array([
                [K11[i], K12[i], K13[i], K14[i], K15[i], K16[i]],
                [K12[i], K22[i], K23[i], K24[i], K25[i], K26[i]],
                [K13[i], K23[i], K33[i], K34[i], K35[i], K36[i]],
                [K14[i], K24[i], K34[i], K44[i], K45[i], K46[i]],
                [K15[i], K25[i], K35[i], K45[i], K55[i], K56[i]],
                [K16[i], K26[i], K36[i], K46[i], K56[i], K66[i]]
            ])
            K[i,:,:] = k_matrix
            # Assemble inertia matrix at the reference axis
            i_matrix = np.array([
                [mass[i], 0.0, 0.0, 0.0, 0.0, -cm_y[i]*mass[i]],
                [0.0, mass[i], 0.0, 0.0, 0.0, cm_x[i]*mass[i]],
                [0.0, 0.0, mass[i], cm_y[i]*mass[i], -cm_x[i]*mass[i], 0.0],
                [0.0, 0.0, cm_y[i]*mass[i], i_edge[i], -i_cp[i], 0.0],
                [0.0, 0.0, -cm_x[i]*mass[i], -i_cp[i], i_flap[i], 0.0],
                [-cm_y[i]*mass[i], cm_x[i]*mass[i], 0.0, 0.0, 0.0, i_plr[i]]
            ])
            I[i,:,:] = i_matrix

        outputs["K"] = K
        outputs["I"] = I
class KI_to_Elastic(ExplicitComponent):
    def initialize(self):
        self.options.declare('modeling_options')

    def setup(self):
        modopt = self.options['modeling_options']
        rotorse_options  = modopt['WISDEM']['RotorSE']
        self.n_span = n_span = rotorse_options['n_span']

        # Inputs
        self.add_input(
            "K", 
            val=np.zeros((n_span,6,6)), 
            desc="Stiffness matrix at the center of the windIO reference axes."
        )
        self.add_input(
            "I", 
            val=np.zeros((n_span,6,6)), 
            desc="Inertia matrix at the center of the windIO reference axes."
        )
        self.add_input(
            "theta",
            val=np.zeros(n_span),
            units="deg",
            desc="Aerodynamic twist angle at each section (positive decreases angle of attack)",
        )

        # Outputs
        self.add_output("A", val=np.ones(n_span), units="m**2", desc="cross sectional area")
        self.add_output("EA", val=np.zeros(n_span), units="N", desc="Axial stiffness at the elastic center.")
        self.add_output("EIxx", val=np.zeros(n_span), units="N*m**2", desc="Section lag (edgewise) bending stiffness about the XE axis.")
        self.add_output("EIyy", val=np.zeros(n_span), units="N*m**2", desc="Section flap bending stiffness about the YE axis.")
        self.add_output("EIxy", val=np.zeros(n_span), units="N*m**2", desc="Coupled flap-lag stiffness with respect to the XE-YE frame")
        self.add_output("EA_EIxx", val=np.zeros(n_span), units="N*m", desc="Coupled axial-lag stiffness with respect to the XE-YE frame")
        self.add_output("EA_EIyy", val=np.zeros(n_span), units="N*m", desc="Coupled axial-flap stiffness with respect to the XE-YE frame")
        self.add_output("EIxx_GJ", val=np.zeros(n_span), units="N*m**2", desc="Coupled lag-torsion stiffness with respect to the XE-YE frame")
        self.add_output("EIyy_GJ", val=np.zeros(n_span), units="N*m**2", desc="Coupled flap-torsion stiffness with respect to the XE-YE frame ")
        self.add_output("EA_GJ", val=np.zeros(n_span), units="N*m", desc="Coupled axial-torsion stiffness")
        self.add_output("GJ", val=np.zeros(n_span), units="N*m**2", desc="Section torsion stiffness at the elastic center.")

        self.add_output("rhoA", val=np.zeros(n_span), units="kg/m", desc="Section mass per unit length")
        self.add_output("rhoJ", val=np.zeros(n_span), units="kg*m", desc="polar mass moment of inertia per unit length")
        self.add_output(
            "Tw_iner",
            val=np.zeros(n_span),
            units="deg",
            desc="Orientation of the section principal inertia axes with respect the blade reference plane",
        )
        self.add_output(
            "x_tc",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the tension-center offset with respect to the XR-YR axes",
        )
        self.add_output(
            "y_tc",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section tension-center with respect to the XR-YR axes",
        )
        self.add_output(
            "x_sc",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the shear-center offset with respect to the XR-YR axes",
        )
        self.add_output(
            "y_sc",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section shear-center with respect to the reference frame, XR-YR",
        )
        self.add_output(
            "x_cg",
            val=np.zeros(n_span),
            units="m",
            desc="X-coordinate of the center-of-mass offset with respect to the XR-YR axes",
        )
        self.add_output(
            "y_cg",
            val=np.zeros(n_span),
            units="m",
            desc="Chordwise offset of the section center of mass with respect to the XR-YR axes",
        )
        self.add_output(
            "flap_iner",
            val=np.zeros(n_span),
            units="kg/m",
            desc="Section flap inertia about the Y_G axis per unit length.",
        )
        self.add_output(
            "edge_iner",
            val=np.zeros(n_span),
            units="kg/m",
            desc="Section lag inertia about the X_G axis per unit length",
        )

        # the following will be given as pseudo 
        self.add_output(
            "xu_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_output(
            "xl_spar",
            val=np.zeros(n_span),
            desc="x-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_output(
            "yu_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on upper surface for strain calculation",
        )
        self.add_output(
            "yl_spar",
            val=np.zeros(n_span),
            desc="y-position of midpoint of spar cap on lower surface for strain calculation",
        )
        self.add_output(
            "xu_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_output(
            "xl_te",
            val=np.zeros(n_span),
            desc="x-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )
        self.add_output(
            "yu_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on upper surface for strain calculation",
        )
        self.add_output(
            "yl_te",
            val=np.zeros(n_span),
            desc="y-position of midpoint of trailing-edge panel on lower surface for strain calculation",
        )
        
    def compute(self, inputs, outputs):

        # Initialize empty 6x6 K and I matrices
        K_tc = np.zeros((self.n_span,6,6))
        I_cg = np.zeros((self.n_span,6,6))
        aero_twist =  np.deg2rad(inputs["theta"])


        for i in range(self.n_span):
            # Get the stiffness matrix at the reference axis
            K_ref = inputs["K"][i,:,:]
            I_ref = inputs["I"][i,:,:]

            # Find shear, elastic/tension center
            # Reference SONATA anbax
            K1 = np.array([[K_ref[n, m] for m in range(3)] for n in range(3)])
            K3 = np.array([[K_ref[n, m+3] for m in range(3)] for n in range(3)])
            try:
                Y = np.linalg.solve(K1, -K3)
                x_sc = -Y[1, 2]
                y_sc = Y[0, 2]

                x_tc = Y[2, 1]
                y_tc = -Y[2, 0]
            except:
                print(f"Failed to compute shear center and tension center at section {i}.")
                if (K_ref[3,3] == 0) and (K_ref[4,4] == 0):
                    raise Exception("K44 and K55 cannot be zeros!")
                else:
                    x_sc = 0
                    y_sc = 0

                    x_tc = 0
                    y_tc = 0

                    print("Check if K11, K22, K33 are non-zeros and correct. Shear center and tension center valuse are set to zeros.") 
            # Transform stiffness matrix to elastic center/tension center
            transform = TransformCrossSectionMatrix()
            T = transform.CrossSectionTranslationMatrix(x_tc, y_tc)
            K_tc = T.T @ K_ref @ T

            # assign stiffness matrix to outputs
            outputs["EA"][i] = K_tc[2,2]
            outputs["EA_EIxx"][i] = K_tc[2,3]
            outputs["EA_EIyy"][i] = K_tc[2,4]
            outputs["EIxx"][i] = K_tc[3,3]
            outputs["EIyy"][i] = K_tc[4,4]
            outputs["EIxy"][i] = K_tc[3,4]
            outputs["GJ"][i] = K_tc[5,5]
            outputs["EA_GJ"][i] = K_tc[2,5]
            outputs["EIxx_GJ"][i] = K_tc[3,5]
            outputs["EIyy_GJ"][i] = K_tc[4,5]
            outputs["x_tc"][i] = x_tc
            outputs["y_tc"][i] = y_tc
            outputs["x_sc"][i] = x_sc
            outputs["y_sc"][i] = y_sc

            # Transform mass matrix to inertia frame
            x_cg = I_ref[1,5]/I_ref[0,0]
            y_cg = -I_ref[0,5]/I_ref[0,0]


            transform = TransformCrossSectionMatrix()
            T = transform.CrossSectionTranslationMatrix(x_cg, y_cg)
            I_cg = T.T @ I_ref @ T
            # find the inertia twist
            I3 = np.array([[I_cg[n+3, m+3] for m in range(3)] for n in range(3)]) 
            (w3, v3) = np.linalg.eig(I3)

            # This angle solve is likely working only within in [-pi/2, pi/2]
            if np.abs(v3[0,0]) < np.abs(v3[0,1]):
                angle = np.arccos(v3[0,0])
            else:
                angle = -np.arcsin(v3[0,1])
            R = transform.CrossSectionRotationMatrix(-angle) # negative because from ref to inertia
            I_cg = R.T @ I_cg @ R

            # assign inertia matrix to outputs
            
            outputs["rhoA"][i] = I_cg[0,0]
            outputs["rhoJ"][i] = I_cg[5,5] # This is actually J - edge+flap
            outputs["edge_iner"][i] = I_cg[3,3]
            outputs["flap_iner"][i] = I_cg[4,4]
            outputs["Tw_iner"][i] = np.rad2deg(aero_twist[i]-angle)
            outputs["x_cg"][i] = x_cg
            outputs["y_cg"][i] = y_cg

            # Aproximate area A
            E_est_xx = outputs["EIxx"][i] / outputs["edge_iner"][i]
            E_est_yy = outputs["EIyy"][i] / outputs["flap_iner"][i]
            A_est_xx = outputs["EA"][i] / E_est_xx
            A_est_yy = outputs["EA"][i] / E_est_yy
            outputs["A"][i] = 0.5*(A_est_xx+A_est_yy) # Approximate the area          


class RotorElasticity(Group):
    # OpenMDAO group to compute the blade elastic properties and natural frequencies
    def initialize(self):
        self.options.declare("modeling_options")
        self.options.declare("opt_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]
        opt_options = self.options["opt_options"]

        # Get elastic properties by running precomp
        promote_list = [
            "EA",
            "EIxx",
            "EIyy",
            "EIxy",
            "GJ",       
            "EA_EIxx",
            "EA_EIyy",
            "EIxx_GJ",
            "EIyy_GJ",
            "EA_GJ",
            "rhoA",
            "rhoJ",
            "Tw_iner",
            "x_tc",
            "y_tc",
            "x_cg",
            "y_cg",
            "edge_iner",
            "flap_iner",
        ]

        if modeling_options["user_elastic"]["blade"]:
            self.add_subsystem("generate_KI", assemble_KI(modeling_options=modeling_options), promotes=['*'])
            self.add_subsystem("precomp", KI_to_Elastic(modeling_options=modeling_options), promotes=['*'])

        else:

            self.add_subsystem(
                "precomp",
                RunPreComp(modeling_options=modeling_options, opt_options=opt_options),
                promotes=promote_list
                + [
                    "r",
                    "chord",
                    "theta",
                    "A",
                    "precurve",
                    "presweep",
                    "section_offset_y",
                    "coord_xy_interp",
                    "sc_ss_mats",
                    "sc_ps_mats",
                    "te_ss_mats",
                    "te_ps_mats",
                    "xu_spar",
                    "xl_spar",
                    "yu_spar",
                    "yl_spar",
                    "xu_te",
                    "xl_te",
                    "yu_te",
                    "yl_te",
                    "n_blades",
                ],
            )

            self.add_subsystem("generate_KI", generate_KI(modeling_options=modeling_options), promotes=promote_list+["K", "I"])

        # Compute total blade properties
        self.add_subsystem("total_blade_properties",
                           TotalBladeProperties(modeling_options=modeling_options, opt_options=opt_options),
                           promotes=["*"])
            


