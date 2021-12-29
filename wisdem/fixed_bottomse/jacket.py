"""
This Jacket analysis and design module largely follows the work presented in:
"A systematic approach to offshore wind turbine jacket predesign and optimization:
geometry, cost, and surrogate structural code check models"
by Jan Häfele, Rick R. Damiani, Ryan N. King, Cristian G. Gebhardt, and Raimund Rolfes
Accessible via: https://wes.copernicus.org/articles/3/553/2018/
"""

import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.cross_sections as cs
import wisdem.commonse.utilization_dnvgl as util_dnvgl
import wisdem.commonse.utilization_constraints as util_con
from wisdem.commonse import NFREQ, RIGID, gravity


class GetGreekLetters(om.ExplicitComponent):
    """
    This component computes the intermediate values needed to use the jacket
    parameterization outlined in the Häfele paper. Specifically, this follows
    section 2.1: Topology.

    See the geometry_schema.yaml or the paper for an explanation of each of
    these parameters.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]

        self.add_input("r_foot", val=10.0, units="m")
        self.add_input("r_head", val=6.0, units="m")
        self.add_input("height", val=70.0, units="m")
        self.add_input("q", val=0.9)
        self.add_input("l_osg", val=5.0, units="m")
        self.add_input("l_tp", val=4.0, units="m")

        self.add_input("gamma_b", val=6.0)
        self.add_input("gamma_t", val=8.0)
        self.add_input("beta_b", val=0.9)
        self.add_input("beta_t", val=0.8)
        self.add_input("tau_b", val=0.6)
        self.add_input("tau_t", val=0.5)

        self.add_output("gamma_i", val=np.zeros((n_bays + 1)))
        self.add_output("beta_i", val=np.zeros((n_bays)))
        self.add_output("tau_i", val=np.zeros((n_bays)))

        self.add_output("xi", val=0.0)
        self.add_output("nu", val=0.0)
        self.add_output("psi_s", val=0.0)
        self.add_output("psi_p", val=0.0)

        self.add_output("lower_bay_heights", val=np.zeros((n_bays + 1)), units="m")
        self.add_output("lower_bay_radii", val=np.zeros((n_bays + 1)), units="m")
        self.add_output("l_mi", val=np.zeros((n_bays)), units="m")

    def compute(self, inputs, outputs):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]

        r_foot = inputs["r_foot"]
        r_head = inputs["r_head"]
        L = inputs["height"]
        q = inputs["q"]
        l_osg = inputs["l_osg"]
        l_tp = inputs["l_tp"]

        gamma_b = inputs["gamma_b"]
        gamma_t = inputs["gamma_t"]
        beta_b = inputs["beta_b"]
        beta_t = inputs["beta_t"]
        tau_b = inputs["tau_b"]
        tau_t = inputs["tau_t"]

        # Do calculations to get the rough topology
        xi = r_head / r_foot  # must be <= 1; taper towards the top
        nu = 2 * np.pi / n_legs  # angle enclosed by two legs
        psi_s = np.arctan(r_foot * (1 - xi) / L)  # spatial batter angle
        psi_p = np.arctan(r_foot * (1 - xi) * np.sin(nu / 2.0) / L)

        # Compute all bay heights
        tmp = q ** np.arange(n_bays)
        bay_heights = l_i = (L - l_osg - l_tp) / (np.sum(tmp) / tmp)

        # Compute the starting elevations of the bays and the radii
        lower_bay_heights = np.hstack((0.0, bay_heights))
        lower_bay_radii = r_i = r_foot - np.tan(psi_s) * (l_osg + np.cumsum(lower_bay_heights))

        # x joint layer info
        l_mi = l_i * r_i[:-1] / (r_i[:-1] + r_i[1:])

        gamma_i = (gamma_t - gamma_b) * (l_osg + np.cumsum(l_i) + l_mi) / (L - l_i[-1] + l_mi[-1] - l_tp) + gamma_b
        gamma_i = np.hstack((gamma_b, gamma_i))

        length = L - l_i[-1] - l_osg - l_tp
        beta_i = (beta_t - beta_b) / length * np.cumsum(l_i) + beta_b
        tau_i = (tau_t - tau_b) / length * np.cumsum(l_i) + tau_b

        outputs["xi"] = xi
        outputs["nu"] = nu
        outputs["psi_s"] = psi_s
        outputs["psi_p"] = psi_p
        outputs["gamma_i"] = gamma_i
        outputs["beta_i"] = beta_i
        outputs["tau_i"] = tau_i
        outputs["lower_bay_heights"] = lower_bay_heights
        outputs["lower_bay_radii"] = lower_bay_radii
        outputs["l_mi"] = l_mi


class ComputeNodes(om.ExplicitComponent):
    """
    This component computes the xyz locations of each of the bay and leg nodes
    within the jacket structure. Going from the parameterization outlined in the
    Häfele paper to nodal information requires the intermediary values computed
    in the previous component as well as some dimensioned properties of the jacket.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]

        self.add_input("nu", val=0.0)
        self.add_input("psi_p", val=0.0)
        self.add_input("lower_bay_heights", val=np.zeros((n_bays + 1)), units="m")
        self.add_input("lower_bay_radii", val=np.zeros((n_bays + 1)), units="m")
        self.add_input("l_mi", val=np.zeros((n_bays)), units="m")
        self.add_input("l_osg", val=5.0, units="m")
        self.add_input("height", val=70.0, units="m")
        self.add_input("r_foot", val=10.0, units="m")
        self.add_input("r_head", val=6.0, units="m")

        self.add_output("leg_nodes", val=np.zeros((n_legs, n_bays + 2, 3)), units="m")
        self.add_output("bay_nodes", val=np.zeros((n_legs, n_bays + 1, 3)), units="m")

    def compute(self, inputs, outputs):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]

        nu = inputs["nu"]
        psi_p = inputs["psi_p"]
        lower_bay_heights = inputs["lower_bay_heights"]
        lower_bay_radii = inputs["lower_bay_radii"]
        l_mi = inputs["l_mi"]
        l_osg = inputs["l_osg"]
        L = inputs["height"]
        r_foot = inputs["r_foot"]
        r_head = inputs["r_head"]

        # Take in member properties, like diameters and thicknesses
        bay_nodes = np.zeros((n_legs, n_bays + 1, 3))
        bay_nodes[:, :, 2] = np.cumsum(lower_bay_heights)
        bay_nodes[:, :, 2] += l_osg

        for idx in range(n_legs):
            tmp = np.outer(lower_bay_radii, np.array([np.cos(nu * idx), np.sin(nu * idx)]))
            bay_nodes[idx, :, 0:2] = tmp

        leg_nodes = np.zeros((n_legs, n_bays + 2, 3))
        tmp = l_mi / np.cos(psi_p)
        leg_nodes[:, 1:-1, 2] = bay_nodes[:, :-1, 2] + tmp
        leg_nodes[:, -1, 2] = L

        leg_radii = np.interp(
            leg_nodes[0, :, 2],
            np.linspace(leg_nodes[0, 0, 2], leg_nodes[0, -1, 2], n_bays + 2),
            np.linspace(r_foot[0], r_head[0], n_bays + 2),
        )
        for idx in range(n_legs):
            tmp = np.outer(leg_radii, np.array([np.cos(nu * idx), np.sin(nu * idx)]))
            leg_nodes[idx, :, 0:2] = tmp

        outputs["bay_nodes"] = bay_nodes
        outputs["leg_nodes"] = leg_nodes


class ComputeDiameterAndThicknesses(om.ExplicitComponent):
    """
    This component computes the diameters and thicknesses of the legs and braces
    following section 2.2 of the Häfele paper.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]

        self.add_input("d_l", val=1.4, units="m")
        self.add_input("gamma_i", val=np.zeros((n_bays + 1)))
        self.add_input("beta_i", val=np.zeros((n_bays)))
        self.add_input("tau_i", val=np.zeros((n_bays)))

        self.add_output("leg_thicknesses", val=np.zeros((n_bays + 1)), units="m")
        self.add_output("brace_diameters", val=np.zeros((n_bays)), units="m")
        self.add_output("brace_thicknesses", val=np.zeros((n_bays)), units="m")

    def compute(self, inputs, outputs):
        d_l = inputs["d_l"]
        gamma_i = inputs["gamma_i"]
        beta_i = inputs["beta_i"]
        tau_i = inputs["tau_i"]

        outputs["leg_thicknesses"] = t_l = d_l / (2 * gamma_i)
        outputs["brace_diameters"] = d_b = beta_i * d_l
        outputs["brace_thicknesses"] = t_b = tau_i * t_l[:-1]


class ComputeFrame3DD(om.ExplicitComponent):
    """
    Now that we have modal information of the jacket structure, we construct the
    Frame3DD problem to solve for the structural properties of the jacket under
    the designated loading conditions.

    This is a lengthy process that requires creating a singular nodal array,
    member information for each brace and leg segment, and bringing in the
    loading from the tower.

    There are n_legs "ghost" members at the top of the jacket structure to connect
    to a ghost node that receives the turbine F and M values. These members are
    rigid and are needed to transmit the loads, but are not included in the mass,
    stress, and buckling calculations.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]
        x_mb = mod_opt["WISDEM"]["FixedBottomSE"]["mud_brace"]
        n_mat = mod_opt["materials"]["n_mat"]

        self.add_input("leg_nodes", val=np.zeros((n_legs, n_bays + 2, 3)), units="m")
        self.add_input("bay_nodes", val=np.zeros((n_legs, n_bays + 1, 3)), units="m")
        self.add_input("d_l", val=1.4, units="m")
        self.add_input("leg_thicknesses", val=np.zeros((n_bays + 1)), units="m")
        self.add_input("brace_diameters", val=np.zeros((n_bays)), units="m")
        self.add_input("brace_thicknesses", val=np.zeros((n_bays)), units="m")

        self.add_discrete_input("material_names", val=n_mat * [""])
        self.add_input("E_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("G_mat", val=np.zeros([n_mat, 3]), units="Pa")
        self.add_input("rho_mat", val=np.zeros(n_mat), units="kg/m**3")
        self.add_input("sigma_y_mat", val=np.zeros(n_mat), units="Pa")

        self.add_input("turbine_F", np.zeros((3, n_dlc)), units="N")
        self.add_input("turbine_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("gravity_foundation_mass", 0.0, units="kg")
        self.add_input("gravity_foundation_I", np.zeros(6), units="kg*m**2")
        self.add_input("tower_mass", val=0.0, units="kg")
        self.add_input("tower_cost", val=0.0, units="USD")

        # For modal analysis only (loads captured in turbine_F & turbine_M)
        self.add_input("turbine_mass", val=0.0, units="kg")
        self.add_input("turbine_cg", val=np.zeros(3), units="m")
        self.add_input("turbine_I", np.zeros(6), units="kg*m**2")

        n_elem = 2 * (n_legs * (n_bays + 1)) + 4 * (n_legs * n_bays) + int(x_mb) * n_legs + n_legs

        self.add_output("jacket_elem_L", np.zeros(n_elem), units="m")
        self.add_output("jacket_elem_D", np.zeros(n_elem), units="m")
        self.add_output("jacket_elem_t", np.zeros(n_elem), units="m")
        self.add_output("jacket_elem_A", np.zeros(n_elem), units="m**2")
        self.add_output("jacket_elem_Asx", np.zeros(n_elem), units="m**2")
        self.add_output("jacket_elem_Asy", np.zeros(n_elem), units="m**2")
        self.add_output("jacket_elem_Ixx", np.zeros(n_elem), units="kg*m**2")
        self.add_output("jacket_elem_Iyy", np.zeros(n_elem), units="kg*m**2")
        self.add_output("jacket_elem_J0", np.zeros(n_elem), units="kg*m**2")
        self.add_output("jacket_elem_E", np.zeros(n_elem), units="Pa")
        self.add_output("jacket_elem_G", np.zeros(n_elem), units="Pa")
        self.add_output("jacket_elem_sigma_y", np.zeros(n_elem), units="Pa")
        self.add_output("jacket_elem_qdyn", np.zeros((n_elem, n_dlc)), units="Pa")
        self.add_output("jacket_mass", 0.0, units="kg")
        self.add_output("jacket_cost", 0.0, units="USD")
        self.add_output("structural_mass", val=0.0, units="kg")
        self.add_output("structural_cost", val=0.0, units="USD")

        self.add_output("jacket_base_F", np.zeros((3, n_dlc)), units="N")
        self.add_output("jacket_base_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_output("jacket_Fz", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("jacket_Vx", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("jacket_Vy", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("jacket_Mxx", np.zeros((n_elem, n_dlc)), units="N*m")
        self.add_output("jacket_Myy", np.zeros((n_elem, n_dlc)), units="N*m")
        self.add_output("jacket_Mzz", np.zeros((n_elem, n_dlc)), units="N*m")

        self.idx = 0

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]
        x_mb = mod_opt["WISDEM"]["FixedBottomSE"]["mud_brace"]
        material_name = mod_opt["WISDEM"]["FixedBottomSE"]["material"]

        leg_nodes = inputs["leg_nodes"]
        bay_nodes = inputs["bay_nodes"]

        # Add center of x joint nodes, in between bay members.
        x_nodes = np.zeros((n_legs, n_bays, 3))
        for jdx in range(n_bays):
            for idx in range(n_legs):
                n1 = bay_nodes[idx, jdx]
                n2 = bay_nodes[(idx + 1) % n_legs, (jdx + 1) % (n_bays + 1)]

                n3 = bay_nodes[(idx + 1) % n_legs, jdx]
                n4 = bay_nodes[idx, (jdx + 1) % (n_bays + 1)]

                # Filter out division by 0 warnings
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = (n4 - n2) / ((n1 - n2) - (n3 - n4))
                alpha = alpha[0]
                new_node = n4 + alpha * (n3 - n4)
                x_nodes[idx, jdx, :] = new_node

        # Add ghost node for transition piece
        ghost_nodes = np.mean(leg_nodes[:, -1, :], axis=0)
        ghost_nodes[2] += 2.0  # add two meters in the z-direction
        ghost_nodes = ghost_nodes.reshape(1, 1, 3)

        # Concatenate all nodal information together
        xyz = np.vstack(
            (leg_nodes.reshape(-1, 3), bay_nodes.reshape(-1, 3), x_nodes.reshape(-1, 3), ghost_nodes.reshape(-1, 3))
        )
        n = xyz.shape[0]
        node_indices = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node_indices, xyz[:, 0], xyz[:, 1], xyz[:, 2], r)

        # Create arrays to later reference the indices for each node. Needed because
        # Frame3DD expects a singular nodal array but we want to keep information
        # about which nodes belong to the legs, bays, or ghost nodes.
        leg_indices = node_indices[: leg_nodes.size // 3].reshape((n_legs, n_bays + 2))
        bay_indices = node_indices[leg_nodes.size // 3 : leg_nodes.size // 3 + bay_nodes.size // 3].reshape(
            (n_legs, n_bays + 1)
        )
        x_indices = node_indices[leg_nodes.size // 3 + bay_nodes.size // 3 : -1].reshape((n_legs, n_bays))
        ghost_indices = np.atleast_2d(node_indices[-1])

        rnode = np.array(leg_indices[:, 0], dtype=np.int_)
        kx = ky = kz = ktx = kty = ktz = np.array([RIGID] * len(rnode))
        reactions = pyframe3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid=RIGID)

        # ------ frame element data ------------

        self.num_elements = 0
        self.N1 = []
        self.N2 = []
        self.L = []
        self.D = []
        self.t = []
        self.Area = []
        self.Asx = []
        self.Asy = []
        self.J0 = []
        self.Ixx = []
        self.Iyy = []
        self.vol = []

        # Helper function to add an element to all relevant lists.
        def add_element(n1_nodes, n1_indices, n2_nodes, n2_indices, itube, idx1, idx2, idx3, idx4):
            n1 = n1_nodes[idx1, idx2]
            n2 = n2_nodes[idx3, idx4]
            self.N1.append(n1_indices[idx1, idx2])
            self.N2.append(n2_indices[idx3, idx4])
            length = np.linalg.norm(n2 - n1)
            self.L.append(length)
            self.D.append(itube.D)
            self.t.append(itube.t)
            self.Area.append(itube.Area)
            self.Asx.append(itube.Asx)
            self.Asy.append(itube.Asy)
            self.J0.append(itube.J0)
            self.Ixx.append(itube.Ixx)
            self.Iyy.append(itube.Iyy)
            self.vol.append(itube.Area * length)
            self.num_elements += 1

        # Naive for loops to make sure we get indexing right.
        # Could vectorize later as needed.

        # Add leg members.
        for jdx in range(n_bays + 1):
            itube = cs.Tube(inputs["d_l"], inputs["leg_thicknesses"][jdx])
            for idx in range(n_legs):
                add_element(leg_nodes, leg_indices, bay_nodes, bay_indices, itube, idx, jdx, idx, jdx)
                add_element(bay_nodes, bay_indices, leg_nodes, leg_indices, itube, idx, jdx, idx, jdx + 1)

        # Add brace (x-joint) members
        for jdx in range(n_bays):
            itube = cs.Tube(inputs["brace_diameters"][jdx], inputs["brace_thicknesses"][jdx])
            for idx in range(n_legs):
                add_element(bay_nodes, bay_indices, x_nodes, x_indices, itube, idx, jdx, idx, jdx)
                add_element(
                    x_nodes,
                    x_indices,
                    bay_nodes,
                    bay_indices,
                    itube,
                    idx,
                    jdx,
                    (idx + 1) % n_legs,
                    (jdx + 1) % (n_bays + 1),
                )

                add_element(bay_nodes, bay_indices, x_nodes, x_indices, itube, (idx + 1) % n_legs, jdx, idx, jdx)
                add_element(
                    x_nodes,
                    x_indices,
                    bay_nodes,
                    bay_indices,
                    itube,
                    idx,
                    jdx,
                    idx,
                    (jdx + 1) % (n_bays + 1),
                )

        # Add mud brace if boolean True
        if x_mb:
            itube = cs.Tube(inputs["brace_diameters"][0], inputs["brace_thicknesses"][0])
            for idx in range(n_legs):
                add_element(bay_nodes, bay_indices, bay_nodes, bay_indices, itube, idx, 0, (idx + 1) % n_legs, 0)

        # Add ghost point where we add the turbine_F and M as well as transition mass
        itube = cs.Tube(1.0e-2, 1.0e-3)
        for idx in range(n_legs):
            add_element(leg_nodes, leg_indices, ghost_nodes, ghost_indices, itube, idx, -1, 0, 0)

        # Grab material information; generally steel will be used
        imat = discrete_inputs["material_names"].index(material_name)
        E = [np.mean(inputs["E_mat"][imat])] * self.num_elements
        G = [np.mean(inputs["G_mat"][imat])] * self.num_elements
        rho = [inputs["rho_mat"][imat]] * self.num_elements

        # Convert all lists to arrays
        Area = np.squeeze(np.array(self.Area, dtype=np.float))
        Asx = np.squeeze(np.array(self.Asx, dtype=np.float))
        Asy = np.squeeze(np.array(self.Asy, dtype=np.float))
        J0 = np.squeeze(np.array(self.J0, dtype=np.float))
        Ixx = np.squeeze(np.array(self.Ixx, dtype=np.float))
        Iyy = np.squeeze(np.array(self.Iyy, dtype=np.float))
        L = np.squeeze(np.array(self.L, dtype=np.float))
        D = np.squeeze(np.array(self.D, dtype=np.float))
        t = np.squeeze(np.array(self.t, dtype=np.float))
        E = np.squeeze(np.array(E, dtype=np.float))
        G = np.squeeze(np.array(G, dtype=np.float))
        rho = np.squeeze(np.array(rho, dtype=np.float))
        N1 = self.N1
        N2 = self.N2

        # Populate mass and cost outputs
        outputs["jacket_mass"] = np.sum(Area[:-n_legs] * rho[:-n_legs] * L[:-n_legs])
        outputs["structural_mass"] = outputs["jacket_mass"] + inputs["tower_mass"]
        outputs["structural_cost"] = (
            outputs["jacket_cost"] + inputs["tower_cost"]
        )  # TODO : actually compute the jacket cost

        # Modify last n_legs elements to make them rigid due to the ghost node
        E[-n_legs:] *= 1.0e8
        G[-n_legs:] *= 1.0e8
        rho[-n_legs:] = 1.0e-2

        outputs["jacket_elem_L"] = L
        outputs["jacket_elem_D"] = D
        outputs["jacket_elem_t"] = t
        outputs["jacket_elem_A"] = Area
        outputs["jacket_elem_Asx"] = Asx
        outputs["jacket_elem_Asy"] = Asy
        outputs["jacket_elem_Ixx"] = Ixx
        outputs["jacket_elem_Iyy"] = Iyy
        outputs["jacket_elem_J0"] = J0
        outputs["jacket_elem_E"] = E
        outputs["jacket_elem_G"] = G
        outputs["jacket_elem_sigma_y"] = inputs["sigma_y_mat"][imat]
        outputs["jacket_elem_qdyn"] = 1.0e2  # hardcoded value for now

        outputs["jacket_elem_sigma_y"][-n_legs:] *= 1e6
        outputs["jacket_elem_qdyn"][-n_legs:] *= 1e4

        element = np.arange(1, self.num_elements + 1)
        roll = np.zeros(self.num_elements - 1)

        elements = pyframe3dd.ElementData(element, N1, N2, Area, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(
            mod_opt["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"],
            mod_opt["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"],
            dx,
        )
        # -----------------------------------

        # initialize frame3dd object
        self.frame = pyframe3dd.Frame(nodes, reactions, elements, options)

        if mod_opt["WISDEM"]["FixedBottomSE"]["save_truss_figures"]:
            if not self.under_approx:
                self.frame.draw(savefig=True, fig_idx=self.idx)
                self.idx += 1

        # ------- enable dynamic analysis ----------
        Mmethod = 1
        lump = 0
        shift = 0.0
        # Run twice the number of modes to ensure that we can ignore the torsional modes and still get the desired number of fore-aft, side-side modes
        self.frame.enableDynamics(2 * NFREQ, Mmethod, lump, 1e-4, shift)
        # ----------------------------

        # ------ static load case 1 ------------
        # gravity in the X, Y, Z, directions (global)
        gx = 0.0
        gy = 0.0
        gz = -gravity

        for k in range(n_dlc):
            load_obj = pyframe3dd.StaticLoadCase(gx, gy, gz)

            # Prepare point forces at transition node
            turb_F = inputs["turbine_F"][:, k]
            turb_M = inputs["turbine_M"][:, k]
            load_obj.changePointLoads(
                np.array([n], dtype=np.int_),  # -1 b/c same reason as above
                np.array([turb_F[0]]).flatten(),
                np.array([turb_F[1]]).flatten(),
                np.array([turb_F[2]]).flatten(),
                np.array([turb_M[0]]).flatten(),
                np.array([turb_M[1]]).flatten(),
                np.array([turb_M[2]]).flatten(),
            )

            # # trapezoidally distributed loads
            # xx1 = xy1 = xz1 = np.zeros(ielem.size)
            # xx2 = xy2 = xz2 = 0.99 * L  # multiply slightly less than unity b.c. of precision
            # wx1 = inputs["jacket_elem_Px1"][:nelem, k]
            # wx2 = inputs["jacket_elem_Px2"][:nelem, k]
            # wy1 = inputs["jacket_elem_Py1"][:nelem, k]
            # wy2 = inputs["jacket_elem_Py2"][:nelem, k]
            # wz1 = inputs["jacket_elem_Pz1"][:nelem, k]
            # wz2 = inputs["jacket_elem_Pz2"][:nelem, k]
            # load_obj.changeTrapezoidalLoads(ielem, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

            # Add the load case and run
            self.frame.addLoadCase(load_obj)

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
        midx = np.array([n, n - 1, 1], dtype=np.int_)
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

        # self.frame.write("system.3dd")
        # self.frame.draw()
        displacements, forces, reactions, internalForces, mass, modal = self.frame.run()

        # Determine reaction forces
        outputs["jacket_base_F"] = -np.c_[
            reactions.Fx.sum(axis=1), reactions.Fy.sum(axis=1), reactions.Fz.sum(axis=1)
        ].T
        outputs["jacket_base_M"] = -np.c_[
            reactions.Mxx.sum(axis=1), reactions.Myy.sum(axis=1), reactions.Mzz.sum(axis=1)
        ].T

        for ic in range(n_dlc):
            # Forces and moments along the structure
            outputs["jacket_Fz"][:, ic] = forces.Nx[ic, 1::2]
            outputs["jacket_Vx"][:, ic] = -forces.Vz[ic, 1::2]
            outputs["jacket_Vy"][:, ic] = forces.Vy[ic, 1::2]
            outputs["jacket_Mxx"][:, ic] = -forces.Mzz[ic, 1::2]
            outputs["jacket_Myy"][:, ic] = forces.Myy[ic, 1::2]
            outputs["jacket_Mzz"][:, ic] = forces.Txx[ic, 1::2]


class JacketPost(om.ExplicitComponent):
    """
    This component computes the stress and buckling utilization values
    for the jacket structure based on the loading computed by Frame3DD.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        n_bays = mod_opt["WISDEM"]["FixedBottomSE"]["n_bays"]
        x_mb = mod_opt["WISDEM"]["FixedBottomSE"]["mud_brace"]

        n_elem = 2 * (n_legs * (n_bays + 1)) + 4 * (n_legs * n_bays) + int(x_mb) * n_legs + n_legs

        self.add_input("jacket_elem_L", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_D", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_t", np.zeros(n_elem), units="m")
        self.add_input("jacket_elem_A", np.zeros(n_elem), units="m**2")
        self.add_input("jacket_elem_Asx", np.zeros(n_elem), units="m**2")
        self.add_input("jacket_elem_Asy", np.zeros(n_elem), units="m**2")
        self.add_input("jacket_elem_Ixx", np.zeros(n_elem), units="kg*m**2")
        self.add_input("jacket_elem_Iyy", np.zeros(n_elem), units="kg*m**2")
        self.add_input("jacket_elem_J0", np.zeros(n_elem), units="kg*m**2")
        self.add_input("jacket_elem_E", np.zeros(n_elem), units="Pa")
        self.add_input("jacket_elem_G", np.zeros(n_elem), units="Pa")
        self.add_input("jacket_elem_sigma_y", np.zeros(n_elem), units="Pa")
        self.add_input("jacket_elem_qdyn", np.zeros((n_elem, n_dlc)), units="Pa")

        # Processed Frame3DD/OpenFAST outputs
        self.add_input("jacket_Fz", np.ones((n_elem, n_dlc)), units="N")
        self.add_input("jacket_Vx", np.ones((n_elem, n_dlc)), units="N")
        self.add_input("jacket_Vy", np.ones((n_elem, n_dlc)), units="N")
        self.add_input("jacket_Mxx", np.ones((n_elem, n_dlc)), units="N*m")
        self.add_input("jacket_Myy", np.ones((n_elem, n_dlc)), units="N*m")
        self.add_input("jacket_Mzz", np.ones((n_elem, n_dlc)), units="N*m")

        # We don't care about the last n_leg members because they're connected
        # to the ghost node to transmit loads and masses.
        # Thus, the stress and buckling outputs are smaller dimensions
        # than the full n_elem.
        self.add_output("constr_stress", np.ones((n_elem - n_legs, n_dlc)))
        self.add_output("constr_shell_buckling", np.ones((n_elem - n_legs, n_dlc)))
        self.add_output("constr_global_buckling", np.ones((n_elem - n_legs, n_dlc)))

    def compute(self, inputs, outputs):
        # Unpack some variables
        mod_opt = self.options["modeling_options"]
        n_dlc = mod_opt["WISDEM"]["n_dlc"]
        n_legs = mod_opt["WISDEM"]["FixedBottomSE"]["n_legs"]
        gamma_f = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_f"]
        gamma_m = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_m"]
        gamma_n = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_n"]
        gamma_b = mod_opt["WISDEM"]["FixedBottomSE"]["gamma_b"]

        d = inputs["jacket_elem_D"]
        t = inputs["jacket_elem_t"]
        h = inputs["jacket_elem_L"]
        Az = inputs["jacket_elem_A"]
        Asx = inputs["jacket_elem_Asx"]
        Jz = inputs["jacket_elem_J0"]
        Iyy = inputs["jacket_elem_Iyy"]
        sigy = inputs["jacket_elem_sigma_y"]
        E = inputs["jacket_elem_E"]
        G = inputs["jacket_elem_G"]
        qdyn = inputs["jacket_elem_qdyn"]
        r = 0.5 * d

        # Get loads from Framee3dd/OpenFAST
        Fz = inputs["jacket_Fz"]
        Vx = inputs["jacket_Vx"]
        Vy = inputs["jacket_Vy"]
        Mxx = inputs["jacket_Mxx"]
        Myy = inputs["jacket_Myy"]
        Mzz = inputs["jacket_Mzz"]

        M = np.sqrt(Mxx ** 2 + Myy ** 2)
        V = np.sqrt(Vx ** 2 + Vy ** 2)

        # See http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        # print(Fz.shape, Az.shape, M.shape, r.shape, Iyy.shape)
        axial_stress = Fz / Az[:, np.newaxis] + M * (r / Iyy)[:, np.newaxis]
        shear_stress = np.abs(Mzz) / (Jz * r)[:, np.newaxis] + V / Asx[:, np.newaxis]
        hoop_stress = -qdyn * ((r - 0.5 * t) / t)[:, np.newaxis]  # util_con.hoopStress(d, t, qdyn)
        outputs["constr_stress"] = util_con.vonMisesStressUtilization(
            axial_stress, hoop_stress, shear_stress, gamma_f * gamma_m * gamma_n, sigy.reshape((-1, 1))
        )[:-n_legs]

        # Use DNV-GL CP202 Method
        check = util_dnvgl.CylinderBuckling(h, d, t, E=E, G=G, sigma_y=sigy, gamma=gamma_f * gamma_b)
        for k in range(n_dlc):
            results = check.run_buckling_checks(
                Fz[:, k], M[:, k], axial_stress[:, k], hoop_stress[:, k], shear_stress[:, k]
            )

            outputs["constr_shell_buckling"][:, k] = results["Shell"][:-n_legs]
            outputs["constr_global_buckling"][:, k] = results["Global"][:-n_legs]


# Assemble the system together in an OpenMDAO Group
class JacketSE(om.Group):
    """
    Group to contain all subsystems needed for jacket analysis and design.
    Can be used as a standalone or within the larger WISDEM stack.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modeling_options = self.options["modeling_options"]

        self.add_subsystem("greek", GetGreekLetters(modeling_options=modeling_options), promotes=["*"])
        self.add_subsystem("nodes", ComputeNodes(modeling_options=modeling_options), promotes=["*"])
        self.add_subsystem(
            "properties", ComputeDiameterAndThicknesses(modeling_options=modeling_options), promotes=["*"]
        )
        self.add_subsystem("frame3dd", ComputeFrame3DD(modeling_options=modeling_options), promotes=["*"])
        self.add_subsystem(
            "post",
            JacketPost(modeling_options=modeling_options),
            promotes=["*"],
        )
