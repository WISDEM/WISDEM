import numpy as np
import openmdao.api as om

import wisdem.commonse.utilities as util
from wisdem.commonse import gravity
from wisdem.commonse.cylinder_member import NULL, MEMMAX

NNODES_MAX = 500  # 1000
NELEM_MAX = 500  # 1000
RIGID = 1e30
EPS = 1e-6


class PlatformFrame(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]
        self.shape = opt["floating"]["members"]["outer_shape"]

        for k in range(n_member):
            self.add_input(f"member{k}:nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")
            self.add_input(f"member{k}:nodes_r", NULL * np.ones(MEMMAX), units="m")
            if self.shape[k] == "circular":
                self.add_input(f"member{k}:section_D", NULL * np.ones(MEMMAX), units="m")
            elif self.shape[k] == "rectangular":
                self.add_input(f"member{k}:section_a", NULL * np.ones(MEMMAX), units="m")
                self.add_input(f"member{k}:section_b", NULL * np.ones(MEMMAX), units="m")
            self.add_input(f"member{k}:Iwaterx", 0.0, units="m**4")
            self.add_input(f"member{k}:Iwatery", 0.0, units="m**4")
            self.add_input(f"member{k}:section_t", NULL * np.ones(MEMMAX), units="m")
            self.add_input(f"member{k}:section_A", NULL * np.ones(MEMMAX), units="m**2")
            self.add_input(f"member{k}:section_Asx", NULL * np.ones(MEMMAX), units="m**2")
            self.add_input(f"member{k}:section_Asy", NULL * np.ones(MEMMAX), units="m**2")
            self.add_input(f"member{k}:section_Ixx", NULL * np.ones(MEMMAX), units="kg*m**2")
            self.add_input(f"member{k}:section_Iyy", NULL * np.ones(MEMMAX), units="kg*m**2")
            self.add_input(f"member{k}:section_J0", NULL * np.ones(MEMMAX), units="kg*m**2")
            self.add_input(f"member{k}:section_rho", NULL * np.ones(MEMMAX), units="kg/m**3")
            self.add_input(f"member{k}:section_E", NULL * np.ones(MEMMAX), units="Pa")
            self.add_input(f"member{k}:section_G", NULL * np.ones(MEMMAX), units="Pa")
            self.add_input(f"member{k}:section_TorsC", NULL * np.ones(MEMMAX), units="m**3")
            self.add_input(f"member{k}:section_sigma_y", NULL * np.ones(MEMMAX), units="Pa")
            self.add_input(f"member{k}:idx_cb", 0)
            self.add_input(f"member{k}:buoyancy_force", 0.0, units="N")
            self.add_input(f"member{k}:displacement", 0.0, units="m**3")
            self.add_input(f"member{k}:center_of_buoyancy", np.zeros(3), units="m")
            self.add_input(f"member{k}:center_of_mass", np.zeros(3), units="m")
            self.add_input(f"member{k}:ballast_mass", 0.0, units="kg")
            self.add_input(f"member{k}:total_mass", 0.0, units="kg")
            self.add_input(f"member{k}:total_cost", 0.0, units="USD")
            self.add_input(f"member{k}:I_total", np.zeros(6), units="kg*m**2")
            self.add_input(f"member{k}:Awater", 0.0, units="m**2")
            self.add_input(f"member{k}:added_mass", np.zeros(6), units="kg")
            self.add_input(f"member{k}:waterline_centroid", np.zeros(2), units="m")
            self.add_input(f"member{k}:variable_ballast_capacity", val=0.0, units="m**3")

        self.add_input("transition_node", np.zeros(3), units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_cost", 0.0, units="USD")

        self.add_output("transition_piece_I", np.zeros(6), units="kg*m**2")

        self.add_output("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_output("platform_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_output("platform_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_output("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_output("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_output("platform_elem_L", NULL * np.ones(NELEM_MAX), units="m")
        self.add_output("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m") # elem_D, a, b are added for both circular and rectangular
        self.add_output("platform_elem_a", NULL * np.ones(NELEM_MAX), units="m") # zeros when not the corresponding type
        self.add_output("platform_elem_b", NULL * np.ones(NELEM_MAX), units="m")
        self.add_output("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_output("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_J0", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_output("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("platform_elem_TorsC", NULL * np.ones(NELEM_MAX), units="m**3")
        self.add_output("platform_elem_sigma_y", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_discrete_output("platform_elem_memid", [-1] * NELEM_MAX)
        self.add_output("platform_displacement", 0.0, units="m**3")
        self.add_output("platform_center_of_buoyancy", np.zeros(3), units="m")
        self.add_output("platform_hull_center_of_mass", np.zeros(3), units="m")
        self.add_output("platform_centroid", np.zeros(3), units="m")
        self.add_output("platform_ballast_mass", 0.0, units="kg")
        self.add_output("platform_hull_mass", 0.0, units="kg")
        self.add_output("platform_I_hull", np.zeros(6), units="kg*m**2")
        self.add_output("platform_cost", 0.0, units="USD")
        self.add_output("platform_Awater", 0.0, units="m**2")
        # self.add_output("platform_Iwater", 0.0, units="m**4")
        self.add_output("platform_Iwaterx", 0.0, units="m**4")
        self.add_output("platform_Iwatery", 0.0, units="m**4")
        self.add_output("platform_added_mass", np.zeros(6), units="kg")
        self.add_output("platform_variable_capacity", np.zeros(n_member), units="m**3")

        self.node_mem2glob = {}

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Seems like we have to run this each time as numbering can change during optimization
        self.node_mem2glob = {}
        self.set_connectivity(inputs, outputs)

        self.set_node_props(inputs, outputs)
        self.set_element_props(inputs, outputs, discrete_inputs, discrete_outputs)

    def set_connectivity(self, inputs, outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Initialize running lists across all members
        nodes_temp = np.empty((0, 3))
        elem_n1 = np.array([], dtype=np.int_)
        elem_n2 = np.array([], dtype=np.int_)

        # Look over members and grab all nodes and internal connections
        for k in range(n_member):
            inode_xyz = inputs[f"member{k}:nodes_xyz"]
            inodes = np.where(inode_xyz[:, 0] == NULL)[0][0]
            inode_xyz = inode_xyz[:inodes, :]
            inode_range = np.arange(inodes - 1)

            n = nodes_temp.shape[0]
            for ii in range(inodes):
                self.node_mem2glob[(k, ii)] = n + ii

            elem_n1 = np.append(elem_n1, n + inode_range)
            elem_n2 = np.append(elem_n2, n + inode_range + 1)
            nodes_temp = np.append(nodes_temp, inode_xyz, axis=0)

        # Reveal connectivity by using mapping to unique node positions
        nodes, idx, inv = np.unique(nodes_temp.round(8), axis=0, return_index=True, return_inverse=True)
        nnode = nodes.shape[0]
        outputs["platform_nodes"] = NULL * np.ones((NNODES_MAX, 3))
        outputs["platform_nodes"][:nnode, :] = nodes
        outputs["platform_centroid"] = nodes.mean(axis=0)

        # Use mapping to set references to node joints
        nelem = elem_n1.size
        outputs["platform_elem_n1"] = NULL * np.ones(NELEM_MAX, dtype=np.int_)
        outputs["platform_elem_n2"] = NULL * np.ones(NELEM_MAX, dtype=np.int_)
        outputs["platform_elem_L"] = NULL * np.ones(NELEM_MAX, dtype=np.int_)
        outputs["platform_elem_n1"][:nelem] = idx1 = np.int_(inv[elem_n1])
        outputs["platform_elem_n2"][:nelem] = idx2 = np.int_(inv[elem_n2])
        outputs["platform_elem_L"][:nelem] = np.sqrt(np.sum((nodes[idx2, :] - nodes[idx1, :]) ** 2, axis=1))

        # Update global 2 member mappings
        for k in self.node_mem2glob.keys():
            self.node_mem2glob[k] = inv[self.node_mem2glob[k]]

    def set_node_props(self, inputs, outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Number of valid nodes
        node_platform = outputs["platform_nodes"]
        nnode = np.where(node_platform[:, 0] == NULL)[0][0]
        node_platform = node_platform[:nnode, :]

        # Find greatest radius of all members at node intersections
        Rnode = np.zeros(nnode)
        for k in range(n_member):
            irnode = inputs[f"member{k}:nodes_r"]
            n = np.where(irnode == NULL)[0][0]
            for ii in range(n):
                iglob = self.node_mem2glob[(k, ii)]
                Rnode[iglob] = np.array([Rnode[iglob], irnode[ii]]).max()

        # Find forces on nodes
        Fnode = np.zeros((nnode, 3))
        for k in range(n_member):
            icb = int(inputs[f"member{k}:idx_cb"][0])
            iglob = self.node_mem2glob[(k, icb)]
            Fnode[iglob, 2] += inputs[f"member{k}:buoyancy_force"][0]

        # Get transition piece inertial properties
        itrans_platform = util.closest_node(node_platform, inputs["transition_node"])
        m_trans = float(inputs["transition_piece_mass"][0])
        r_trans = Rnode[itrans_platform]
        I_trans = m_trans * r_trans**2.0 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]
        outputs["transition_piece_I"] = I_trans

        # Store outputs
        outputs["platform_Rnode"] = NULL * np.ones(NNODES_MAX)
        outputs["platform_Rnode"][:nnode] = Rnode
        outputs["platform_Fnode"] = NULL * np.ones((NNODES_MAX, 3))
        outputs["platform_Fnode"][:nnode, :] = Fnode

    def set_element_props(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]
        self.shape = opt["floating"]["members"]["outer_shape"]

        # Initialize running lists across all members
        elem_D = np.array([])
        elem_a = np.array([])
        elem_b = np.array([])
        elem_t = np.array([])
        elem_A = np.array([])
        elem_Asx = np.array([])
        elem_Asy = np.array([])
        elem_Ixx = np.array([])
        elem_Iyy = np.array([])
        elem_J0 = np.array([])
        elem_rho = np.array([])
        elem_E = np.array([])
        elem_G = np.array([])
        elem_TorsC = np.array([])
        elem_sigy = np.array([])
        elem_memid = np.array([], dtype=np.int_)

        mass = 0.0
        m_ball = 0.0
        cost = 0.0
        volume = 0.0
        Awater = 0.0
        # Iwater = 0.0
        Iwaterx = 0.0
        Iwatery = 0.0
        cg_plat = np.zeros(3)
        cb_plat = np.zeros(3)
        centroid = outputs["platform_centroid"][:2]
        variable_capacity = np.zeros(n_member)

        # Append all member data
        for k in range(n_member):
            n = np.where(inputs[f"member{k}:section_A"] == NULL)[0][0]
            if self.shape[k] == "circular":
                elem_D = np.append(elem_D, inputs[f"member{k}:section_D"][:n])
                elem_a = np.append(elem_a, np.zeros(n))
                elem_b = np.append(elem_b, np.zeros(n))
            elif self.shape[k] == "rectangular":
                elem_D = np.append(elem_D, np.zeros(n))
                elem_a = np.append(elem_a, inputs[f"member{k}:section_a"][:n])
                elem_b = np.append(elem_b, inputs[f"member{k}:section_b"][:n])
            elem_t = np.append(elem_t, inputs[f"member{k}:section_t"][:n])
            elem_A = np.append(elem_A, inputs[f"member{k}:section_A"][:n])
            elem_Asx = np.append(elem_Asx, inputs[f"member{k}:section_Asx"][:n])
            elem_Asy = np.append(elem_Asy, inputs[f"member{k}:section_Asy"][:n])
            elem_Ixx = np.append(elem_Ixx, inputs[f"member{k}:section_Ixx"][:n])
            elem_Iyy = np.append(elem_Iyy, inputs[f"member{k}:section_Iyy"][:n])
            elem_J0 = np.append(elem_J0, inputs[f"member{k}:section_J0"][:n])
            elem_rho = np.append(elem_rho, inputs[f"member{k}:section_rho"][:n])
            elem_E = np.append(elem_E, inputs[f"member{k}:section_E"][:n])
            elem_G = np.append(elem_G, inputs[f"member{k}:section_G"][:n])
            elem_TorsC = np.append(elem_TorsC, inputs[f"member{k}:section_TorsC"][:n])
            elem_sigy = np.append(elem_sigy, inputs[f"member{k}:section_sigma_y"][:n])
            elem_memid = np.append(elem_memid, k * np.ones(n, dtype=np.int_))

            # Mass, volume, cost tallies
            imass = inputs[f"member{k}:total_mass"]
            ivol = inputs[f"member{k}:displacement"]

            mass += imass
            volume += ivol
            cost += inputs[f"member{k}:total_cost"]
            m_ball += inputs[f"member{k}:ballast_mass"]
            Awater_k = inputs[f"member{k}:Awater"]
            Awater += Awater_k
            # y-coordinate for roll metacentric height
            # x-coordinate for pitch meracentric height
            Rwaterx = inputs[f"member{k}:waterline_centroid"][1] - centroid[1]
            Rwatery = inputs[f"member{k}:waterline_centroid"][0] - centroid[0]
            Iwaterx += inputs[f"member{k}:Iwaterx"] + Awater_k * Rwaterx**2
            Iwatery += inputs[f"member{k}:Iwatery"] + Awater_k * Rwatery**2
            variable_capacity[k] = inputs[f"member{k}:variable_ballast_capacity"][0]

            # Center of mass / buoyancy tallies
            cg_plat += imass * inputs[f"member{k}:center_of_mass"]
            cb_plat += ivol * inputs[f"member{k}:center_of_buoyancy"]

        # Add transition piece
        m_trans = inputs["transition_piece_mass"]
        cg_trans = inputs["transition_node"]
        I_trans = util.assembleI(outputs["transition_piece_I"])
        mass += m_trans
        cost += inputs["transition_piece_cost"]
        cg_plat += m_trans * cg_trans

        # Finalize outputs
        cg_plat /= mass
        cb_plat /= volume

        # With CG known, loop back through to compute platform I and added mass
        unit_z = np.array([0.0, 0.0, 1.0])
        I_hull = np.zeros((3, 3))
        m_added = np.zeros(6)
        for k in range(n_member):
            xyz_k = inputs[f"member{k}:nodes_xyz"]
            inodes = np.where(xyz_k[:, 0] == NULL)[0][0]
            xyz_k = xyz_k[:inodes, :]

            imass = inputs[f"member{k}:total_mass"]
            cg_k = inputs[f"member{k}:center_of_mass"]
            R = cg_plat - cg_k

            # Figure out angle to make member parallel to global c.s.
            vec_k = xyz_k[-1, :] - xyz_k[0, :]
            T = util.rotate_align_vectors(vec_k, unit_z)

            # Rotate member inertia tensor
            I_k = util.assembleI(inputs[f"member{k}:I_total"])
            I_k_rot = T @ I_k @ T.T

            # Now do parallel axis theorem
            I_hull += np.array(I_k_rot) + imass * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

            # Added mass
            m_add_k = inputs[f"member{k}:added_mass"]
            m_added[:3] += m_add_k[:3]
            m_added[3:] += m_add_k[3:] + np.diag(m_add_k[0] * (np.dot(R, R) * np.eye(3) - np.outer(R, R)))

        # Add in transition piece
        R = cg_plat - cg_trans
        I_hull += I_trans + m_trans * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        # Store outputs
        nelem = elem_A.size
        outputs["platform_elem_D"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_a"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_b"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_t"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_A"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Asx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Asy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Ixx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Iyy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_J0"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_rho"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_E"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_G"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_TorsC"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_sigma_y"] = NULL * np.ones(NELEM_MAX)

        outputs["platform_elem_D"][:nelem] = elem_D
        outputs["platform_elem_a"][:nelem] = elem_a
        outputs["platform_elem_b"][:nelem] = elem_b
        outputs["platform_elem_t"][:nelem] = elem_t
        outputs["platform_elem_A"][:nelem] = elem_A
        outputs["platform_elem_Asx"][:nelem] = elem_Asx
        outputs["platform_elem_Asy"][:nelem] = elem_Asy
        outputs["platform_elem_Ixx"][:nelem] = elem_Ixx
        outputs["platform_elem_Iyy"][:nelem] = elem_Iyy
        outputs["platform_elem_J0"][:nelem] = elem_J0
        outputs["platform_elem_rho"][:nelem] = elem_rho
        outputs["platform_elem_E"][:nelem] = elem_E
        outputs["platform_elem_G"][:nelem] = elem_G
        outputs["platform_elem_TorsC"][:nelem] = elem_TorsC
        outputs["platform_elem_sigma_y"][:nelem] = elem_sigy
        opt['floating']['members']['platform_elem_memid'] = elem_memid  # converted from discrete_output to modeling option

        outputs["platform_ballast_mass"] = m_ball
        outputs["platform_hull_mass"] = mass - m_ball
        outputs["platform_cost"] = cost
        outputs["platform_displacement"] = volume
        outputs["platform_hull_center_of_mass"] = cg_plat
        outputs["platform_center_of_buoyancy"] = cb_plat
        outputs["platform_I_hull"] = util.unassembleI(I_hull)
        outputs["platform_Awater"] = Awater
        outputs["platform_Iwaterx"] = Iwaterx
        outputs["platform_Iwatery"] = Iwatery
        outputs["platform_added_mass"] = m_added
        outputs["platform_variable_capacity"] = variable_capacity


class PlatformTurbineSystem(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]
        n_attach = opt["mooring"]["n_attach"]

        self.add_input("platform_hull_center_of_mass", np.zeros(3), units="m")
        self.add_input("platform_ballast_mass", 0.0, units="kg")
        self.add_input("platform_hull_mass", 0.0, units="kg")
        self.add_input("platform_I_hull", np.zeros(6), units="kg*m**2")
        self.add_input("platform_displacement", 0.0, units="m**3")

        self.add_input("turbine_mass", 0.0, units="kg")
        self.add_input("turbine_cg", np.zeros(3), units="m")
        self.add_input("turbine_I", np.zeros(6), units="kg*m**2")
        self.add_input("transition_node", np.zeros(3), units="m")

        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("mooring_neutral_load", np.zeros((n_attach, 3)), units="N")
        self.add_input("platform_variable_capacity", np.zeros(n_member), units="m**3")

        for k in range(n_member):
            self.add_input(f"member{k}:nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")
            self.add_input(f"member{k}:variable_ballast_Vpts", val=np.zeros(10), units="m**3")
            self.add_input(f"member{k}:variable_ballast_spts", val=np.zeros(10))

        self.add_output("system_structural_center_of_mass", np.zeros(3), units="m")
        self.add_output("system_structural_mass", 0.0, units="kg")
        self.add_output("system_center_of_mass", np.zeros(3), units="m")
        self.add_output("system_mass", 0.0, units="kg")
        self.add_output("system_I", np.zeros(6), units="kg*m**2")
        self.add_output("variable_ballast_mass", 0.0, units="kg")
        self.add_output("variable_center_of_mass", val=np.zeros(3), units="m")
        self.add_output("variable_I", np.zeros(6), units="kg*m**2")
        self.add_output("constr_variable_margin", val=0.0)
        self.add_output("member_variable_volume", val=np.zeros(n_member), units="m**3")
        self.add_output("member_variable_height", val=np.zeros(n_member))
        self.add_output("platform_mass", 0.0, units="kg")
        self.add_output("platform_total_center_of_mass", np.zeros(3), units="m")
        self.add_output("platform_I_total", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Mass summaries
        m_platform = inputs["platform_hull_mass"] + inputs["platform_ballast_mass"]
        cg_platform = inputs["platform_hull_center_of_mass"]
        I_platform = util.assembleI(inputs["platform_I_hull"])
        m_turb = inputs["turbine_mass"]
        cg_turb = inputs["turbine_cg"]
        I_turb = util.assembleI(inputs["turbine_I"])
        m_sys = m_platform + m_turb
        outputs["system_structural_mass"] = m_sys

        outputs["system_structural_center_of_mass"] = (m_platform * cg_platform + m_turb * cg_turb) / m_sys

        # Balance out variable ballast
        mooringFz = inputs["mooring_neutral_load"][:, 2].sum()
        capacity = inputs["platform_variable_capacity"]
        capacity_sum = capacity.sum() + EPS  # Avoid divide by zeros
        rho_water = inputs["rho_water"]
        m_variable = inputs["platform_displacement"] * rho_water - m_sys + mooringFz / gravity
        V_variable = m_variable / rho_water
        outputs["variable_ballast_mass"] = m_variable
        outputs["constr_variable_margin"] = V_variable / capacity_sum
        V_variable_member = V_variable * capacity / capacity_sum
        outputs["member_variable_volume"] = V_variable_member
        m_variable_member = V_variable_member * rho_water

        # Now find the CG of the variable mass assigned to each member
        n_member = capacity.size
        outputs["member_variable_height"] = np.zeros(n_member)
        cg_variable_member = np.zeros((n_member, 3))
        for k in range(n_member):
            if V_variable_member[k] == 0.0:
                continue

            xyz = inputs[f"member{k}:nodes_xyz"]
            inodes = np.where(xyz[:, 0] == NULL)[0][0]
            xyz = xyz[:inodes, :]
            dxyz = xyz[-1, :] - xyz[0, :]

            spts = inputs[f"member{k}:variable_ballast_spts"]
            Vpts = inputs[f"member{k}:variable_ballast_Vpts"]

            s_cg = np.interp(0.5 * V_variable_member[k], Vpts, spts)
            cg_variable_member[k, :] = xyz[0, :] + s_cg * dxyz

            s_end = np.interp(V_variable_member[k], Vpts, spts)
            outputs["member_variable_height"][k] = s_end - spts[0]

        cg_variable = np.dot(V_variable_member, cg_variable_member) / V_variable
        outputs["variable_center_of_mass"] = cg_variable

        # Now find total system mass
        outputs["platform_mass"] = m_platform + m_variable
        outputs["system_mass"] = m_sys_total = m_sys + m_variable
        outputs["system_center_of_mass"] = cg_sys_total = (
            m_sys * outputs["system_structural_center_of_mass"] + m_variable * cg_variable
        ) / (m_sys + m_variable)

        # Compute the total cg for the platform and the variable ballast together using a weighted sum approach
        cg_plat_total = (m_variable * cg_variable + m_platform * cg_platform) / (m_variable + m_platform)
        outputs["platform_total_center_of_mass"] = cg_plat_total

        # Now loop again to compute variable I
        unit_z = np.array([0.0, 0.0, 1.0])
        I_variable = np.zeros((3, 3))
        for k in range(n_member):
            if V_variable_member[k] == 0.0:
                continue

            xyz = inputs[f"member{k}:nodes_xyz"]
            inodes = np.where(xyz[:, 0] == NULL)[0][0]
            xyz = xyz[:inodes, :]
            vec_k = xyz[-1, :] - xyz[0, :]

            ds = outputs["member_variable_height"][k]

            # Compute I aligned with member
            h_k = ds * np.sqrt(np.sum(vec_k**2))
            if h_k == 0.0:
                continue
            r_k = np.sqrt(V_variable_member[k] / h_k / np.pi)
            I_k = (
                m_variable_member[k] * np.r_[(3 * r_k**2 + h_k**2) / 12.0 * np.ones(2), 0.5 * r_k**2, np.ones(3)]
            )

            # Rotate I to global c.s.
            T = util.rotate_align_vectors(vec_k, unit_z)
            I_k_rot = T @ util.assembleI(I_k) @ T.T

            # Now do parallel axis theorem
            R = cg_variable - cg_variable_member[k, :]
            I_variable += np.array(I_k_rot) + m_variable_member[k] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        # Find platform I with variable contribution
        I_total = np.zeros((3, 3))

        # Compute the full moment of inertia for the platform and variable ballast
        R = cg_plat_total - cg_platform
        I_total += I_platform + m_platform * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        R = cg_plat_total - cg_variable
        I_total += I_variable + m_variable * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["platform_I_total"] = util.unassembleI(I_total)
        outputs["variable_I"] = util.unassembleI(I_variable)

        # Now full system moments of inertia
        I_sys = np.zeros((3, 3))

        R = cg_sys_total - cg_plat_total
        I_sys += I_total + outputs["platform_mass"] * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        R = cg_sys_total - inputs["transition_node"]  # turbine I is already at base, not at its cg
        I_sys += I_turb + m_turb * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["system_I"] = util.unassembleI(I_sys)


class FloatingSystem(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]

        self.add_subsystem("plat", PlatformFrame(options=opt), promotes=["*"])
        self.add_subsystem("mux", PlatformTurbineSystem(options=opt), promotes=["*"])
