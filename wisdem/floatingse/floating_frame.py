import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
from wisdem.commonse import NFREQ, gravity
from wisdem.floatingse.member import NULL, MEMMAX, Member

NNODES_MAX = 1000
NELEM_MAX = 1000
RIGID = 1e30
EPS = 1e-6

# TODO:
# - Added mass, hydro stiffness for tower sim
# - Stress or buckling?


class PlatformFrame(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]

        for k in range(n_member):
            self.add_input(f"member{k}:nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")
            self.add_input(f"member{k}:nodes_r", NULL * np.ones(MEMMAX), units="m")
            self.add_input(f"member{k}:section_D", NULL * np.ones(MEMMAX), units="m")
            self.add_input(f"member{k}:section_t", NULL * np.ones(MEMMAX), units="m")
            self.add_input(f"member{k}:section_A", NULL * np.ones(MEMMAX), units="m**2")
            self.add_input(f"member{k}:section_Asx", NULL * np.ones(MEMMAX), units="m**2")
            self.add_input(f"member{k}:section_Asy", NULL * np.ones(MEMMAX), units="m**2")
            self.add_input(f"member{k}:section_Ixx", NULL * np.ones(MEMMAX), units="kg*m**2")
            self.add_input(f"member{k}:section_Iyy", NULL * np.ones(MEMMAX), units="kg*m**2")
            self.add_input(f"member{k}:section_Izz", NULL * np.ones(MEMMAX), units="kg*m**2")
            self.add_input(f"member{k}:section_rho", NULL * np.ones(MEMMAX), units="kg/m**3")
            self.add_input(f"member{k}:section_E", NULL * np.ones(MEMMAX), units="Pa")
            self.add_input(f"member{k}:section_G", NULL * np.ones(MEMMAX), units="Pa")
            self.add_input(f"member{k}:idx_cb", 0)
            self.add_input(f"member{k}:buoyancy_force", 0.0, units="N")
            self.add_input(f"member{k}:displacement", 0.0, units="m**3")
            self.add_input(f"member{k}:center_of_buoyancy", np.zeros(3), units="m")
            self.add_input(f"member{k}:center_of_mass", np.zeros(3), units="m")
            self.add_input(f"member{k}:total_mass", 0.0, units="kg")
            self.add_input(f"member{k}:total_cost", 0.0, units="USD")
            self.add_input(f"member{k}:I_total", np.zeros(6), units="kg*m**2")
            self.add_input(f"member{k}:Awater", 0.0, units="m**2")
            self.add_input(f"member{k}:Iwater", 0.0, units="m**4")
            self.add_input(f"member{k}:added_mass", np.zeros(6), units="kg")
            self.add_input(f"member{k}:waterline_centroid", np.zeros(2), units="m")
            self.add_input(f"member{k}:variable_ballast_capacity", val=0.0, units="m**3")
            self.add_input(f"member{k}:Px", np.zeros(MEMMAX), units="N/m")
            self.add_input(f"member{k}:Py", np.zeros(MEMMAX), units="N/m")
            self.add_input(f"member{k}:Pz", np.zeros(MEMMAX), units="N/m")

        self.add_output("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_output("platform_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_output("platform_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_output("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_output("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_output("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_output("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_output("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_Izz", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_output("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("platform_elem_Px1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Px2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Py1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Py2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Pz1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Pz2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_displacement", 0.0, units="m**3")
        self.add_output("platform_center_of_buoyancy", np.zeros(3), units="m")
        self.add_output("platform_center_of_mass", np.zeros(3), units="m")
        self.add_output("platform_centroid", np.zeros(3), units="m")
        self.add_output("platform_mass", 0.0, units="kg")
        self.add_output("platform_I_total", np.zeros(6), units="kg*m**2")
        self.add_output("platform_cost", 0.0, units="USD")
        self.add_output("platform_Awater", 0.0, units="m**2")
        self.add_output("platform_Iwater", 0.0, units="m**4")
        self.add_output("platform_added_mass", np.zeros(6), units="kg")
        self.add_output("platform_variable_capacity", np.zeros(n_member), units="m**3")

        self.node_mem2glob = {}
        # self.node_glob2mem = {}

    def compute(self, inputs, outputs):
        # This shouldn't change during an optimization, so save some time?
        if len(self.node_mem2glob) == 0:
            self.set_connectivity(inputs, outputs)

        self.set_node_props(inputs, outputs)
        self.set_element_props(inputs, outputs)

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
        outputs["platform_elem_n1"][:nelem] = inv[elem_n1]
        outputs["platform_elem_n2"][:nelem] = inv[elem_n2]

        # Update global 2 member mappings
        for k in self.node_mem2glob.keys():
            self.node_mem2glob[k] = inv[self.node_mem2glob[k]]

    def set_node_props(self, inputs, outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Number of valid nodes
        nnode = np.where(outputs["platform_nodes"][:, 0] == NULL)[0][0]

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
            icb = int(inputs[f"member{k}:idx_cb"])
            iglob = self.node_mem2glob[(k, icb)]
            Fnode[iglob, 2] += inputs[f"member{k}:buoyancy_force"]

        # Store outputs
        outputs["platform_Rnode"] = NULL * np.ones(NNODES_MAX)
        outputs["platform_Rnode"][:nnode] = Rnode
        outputs["platform_Fnode"] = NULL * np.ones((NNODES_MAX, 3))
        outputs["platform_Fnode"][:nnode, :] = Fnode

    def set_element_props(self, inputs, outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Initialize running lists across all members
        elem_D = np.array([])
        elem_t = np.array([])
        elem_A = np.array([])
        elem_Asx = np.array([])
        elem_Asy = np.array([])
        elem_Ixx = np.array([])
        elem_Iyy = np.array([])
        elem_Izz = np.array([])
        elem_rho = np.array([])
        elem_E = np.array([])
        elem_G = np.array([])
        elem_Px1 = np.array([])
        elem_Px2 = np.array([])
        elem_Py1 = np.array([])
        elem_Py2 = np.array([])
        elem_Pz1 = np.array([])
        elem_Pz2 = np.array([])

        mass = 0.0
        cost = 0.0
        volume = 0.0
        Awater = 0.0
        Iwater = 0.0
        m_added = np.zeros(6)
        cg_plat = np.zeros(3)
        cb_plat = np.zeros(3)
        centroid = outputs["platform_centroid"][:2]
        variable_capacity = np.zeros(n_member)

        # Append all member data
        for k in range(n_member):
            n = np.where(inputs[f"member{k}:section_A"] == NULL)[0][0]
            elem_D = np.append(elem_D, inputs[f"member{k}:section_D"][:n])
            elem_t = np.append(elem_t, inputs[f"member{k}:section_t"][:n])
            elem_A = np.append(elem_A, inputs[f"member{k}:section_A"][:n])
            elem_Asx = np.append(elem_Asx, inputs[f"member{k}:section_Asx"][:n])
            elem_Asy = np.append(elem_Asy, inputs[f"member{k}:section_Asy"][:n])
            elem_Ixx = np.append(elem_Ixx, inputs[f"member{k}:section_Ixx"][:n])
            elem_Iyy = np.append(elem_Iyy, inputs[f"member{k}:section_Iyy"][:n])
            elem_Izz = np.append(elem_Izz, inputs[f"member{k}:section_Izz"][:n])
            elem_rho = np.append(elem_rho, inputs[f"member{k}:section_rho"][:n])
            elem_E = np.append(elem_E, inputs[f"member{k}:section_E"][:n])
            elem_G = np.append(elem_G, inputs[f"member{k}:section_G"][:n])

            # The loads should come in with length n+1
            elem_Px1 = np.append(elem_Px1, inputs[f"member{k}:Px"][:n])
            elem_Px2 = np.append(elem_Px2, inputs[f"member{k}:Px"][1 : (n + 1)])
            elem_Py1 = np.append(elem_Py1, inputs[f"member{k}:Py"][:n])
            elem_Py2 = np.append(elem_Py2, inputs[f"member{k}:Py"][1 : (n + 1)])
            elem_Pz1 = np.append(elem_Pz1, inputs[f"member{k}:Pz"][:n])
            elem_Pz2 = np.append(elem_Pz2, inputs[f"member{k}:Pz"][1 : (n + 1)])

            # Mass, volume, cost tallies
            imass = inputs[f"member{k}:total_mass"]
            ivol = inputs[f"member{k}:displacement"]

            mass += imass
            volume += ivol
            cost += inputs[f"member{k}:total_cost"]
            Awater_k = inputs[f"member{k}:Awater"]
            Awater += Awater_k
            Rwater2 = np.sum((inputs[f"member{k}:waterline_centroid"] - centroid) ** 2)
            Iwater += inputs[f"member{k}:Iwater"] + Awater_k * Rwater2
            m_added += inputs[f"member{k}:added_mass"]
            variable_capacity[k] = inputs[f"member{k}:variable_ballast_capacity"]

            # Center of mass / buoyancy tallies
            cg_plat += imass * inputs[f"member{k}:center_of_mass"]
            cb_plat += ivol * inputs[f"member{k}:center_of_buoyancy"]

        # Finalize outputs
        cg_plat /= mass
        cb_plat /= volume

        # With CG known, loop back through to compute platform I
        unit_z = np.array([0.0, 0.0, 1.0])
        I_total = np.zeros((3, 3))
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
            I_k2 = T @ I_k @ T.T

            # Now do parallel axis theorem
            I_total += np.array(I_k2) + imass * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        # Store outputs
        nelem = elem_A.size
        outputs["platform_elem_D"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_t"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_A"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Asx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Asy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Ixx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Iyy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Izz"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_rho"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_E"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_G"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Px1"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Px2"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Py1"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Py2"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Pz1"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Pz2"] = NULL * np.ones(NELEM_MAX)

        outputs["platform_elem_D"][:nelem] = elem_D
        outputs["platform_elem_t"][:nelem] = elem_t
        outputs["platform_elem_A"][:nelem] = elem_A
        outputs["platform_elem_Asx"][:nelem] = elem_Asx
        outputs["platform_elem_Asy"][:nelem] = elem_Asy
        outputs["platform_elem_Ixx"][:nelem] = elem_Ixx
        outputs["platform_elem_Iyy"][:nelem] = elem_Iyy
        outputs["platform_elem_Izz"][:nelem] = elem_Izz
        outputs["platform_elem_rho"][:nelem] = elem_rho
        outputs["platform_elem_E"][:nelem] = elem_E
        outputs["platform_elem_G"][:nelem] = elem_G
        outputs["platform_elem_Px1"][:nelem] = elem_Px1
        outputs["platform_elem_Px2"][:nelem] = elem_Px2
        outputs["platform_elem_Py1"][:nelem] = elem_Py1
        outputs["platform_elem_Py2"][:nelem] = elem_Py2
        outputs["platform_elem_Pz1"][:nelem] = elem_Pz1
        outputs["platform_elem_Pz2"][:nelem] = elem_Pz2

        outputs["platform_mass"] = mass
        outputs["platform_cost"] = cost
        outputs["platform_displacement"] = volume
        outputs["platform_center_of_mass"] = cg_plat
        outputs["platform_center_of_buoyancy"] = cb_plat
        outputs["platform_I_total"] = util.unassembleI(I_total)
        outputs["platform_Awater"] = Awater
        outputs["platform_Iwater"] = Iwater
        outputs["platform_added_mass"] = m_added
        outputs["platform_variable_capacity"] = variable_capacity


class TowerPreMember(om.ExplicitComponent):
    def setup(self):
        self.add_input("transition_node", np.zeros(3), units="m")
        self.add_input("tower_height", 0.0, units="m")
        self.add_output("tower_top_node", np.zeros(3), units="m")

    def compute(self, inputs, outputs):
        transition_node = inputs["transition_node"]
        tower_top_node = transition_node
        tower_top_node[2] += float(inputs["tower_height"])
        outputs["tower_top_node"] = tower_top_node


class PlatformTowerFrame(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]
        n_attach = opt["mooring"]["n_attach"]

        self.add_input("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_input("platform_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_input("platform_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_input("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Izz", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_Px1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Px2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Py1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Py2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Pz1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Pz2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_center_of_mass", np.zeros(3), units="m")
        self.add_input("platform_mass", 0.0, units="kg")
        self.add_input("platform_displacement", 0.0, units="m**3")

        self.add_input("tower_nodes", NULL * np.ones((MEMMAX, 3)), units="m")
        self.add_output("tower_Fnode", copy_shape="tower_nodes", units="N")
        self.add_input("tower_Rnode", NULL * np.ones(MEMMAX), units="m")
        self.add_output("tower_elem_n1", copy_shape="tower_elem_A")
        self.add_output("tower_elem_n2", copy_shape="tower_elem_A")
        self.add_input("tower_elem_D", NULL * np.ones(MEMMAX), units="m")
        self.add_input("tower_elem_t", NULL * np.ones(MEMMAX), units="m")
        self.add_input("tower_elem_A", NULL * np.ones(MEMMAX), units="m**2")
        self.add_input("tower_elem_Asx", NULL * np.ones(MEMMAX), units="m**2")
        self.add_input("tower_elem_Asy", NULL * np.ones(MEMMAX), units="m**2")
        self.add_input("tower_elem_Ixx", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_input("tower_elem_Iyy", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_input("tower_elem_Izz", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_input("tower_elem_rho", NULL * np.ones(MEMMAX), units="kg/m**3")
        self.add_input("tower_elem_E", NULL * np.ones(MEMMAX), units="Pa")
        self.add_input("tower_elem_G", NULL * np.ones(MEMMAX), units="Pa")
        self.add_input("tower_elem_Px", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("tower_elem_Px1", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("tower_elem_Px2", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_elem_Py", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("tower_elem_Py1", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("tower_elem_Py2", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_elem_Pz", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("tower_elem_Pz1", NULL * np.ones(MEMMAX), units="N/m")
        self.add_output("tower_elem_Pz2", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_center_of_mass", np.zeros(3), units="m")
        self.add_input("tower_mass", 0.0, units="kg")

        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("tower_top_node", np.zeros(3), units="m")
        self.add_input("transition_node", np.zeros(3), units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("rna_mass", 0.0, units="kg")
        self.add_input("rna_cg", np.zeros(3), units="m")
        self.add_input("mooring_neutral_load", np.zeros((n_attach, 3)), units="N")
        self.add_input("platform_variable_capacity", np.zeros(n_member), units="m**3")

        for k in range(n_member):
            self.add_input(f"member{k}:nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")
            self.add_input(f"member{k}:variable_ballast_Vpts", val=np.zeros(10), units="m**3")
            self.add_input(f"member{k}:variable_ballast_spts", val=np.zeros(10))

        self.add_output("system_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_output("system_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_output("system_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_output("system_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_output("system_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_output("system_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_output("system_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_output("system_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("system_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("system_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("system_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("system_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("system_elem_Izz", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("system_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_output("system_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("system_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("system_elem_Px1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("system_elem_Px2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("system_elem_Py1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("system_elem_Py2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("system_elem_Pz1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("system_elem_Pz2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("system_structural_center_of_mass", np.zeros(3), units="m")
        self.add_output("system_structural_mass", 0.0, units="kg")
        self.add_output("system_center_of_mass", np.zeros(3), units="m")
        self.add_output("system_mass", 0.0, units="kg")
        self.add_output("variable_ballast_mass", 0.0, units="kg")
        self.add_output("constr_variable_margin", val=0.0)
        self.add_output("member_variable_volume", val=np.zeros(n_member), units="m**3")
        self.add_output("transition_piece_I", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Combine nodes
        node_platform = inputs["platform_nodes"]
        node_tower = inputs["tower_nodes"]

        nnode_platform = np.where(node_platform[:, 0] == NULL)[0][0]
        nnode_tower = np.where(node_tower[:, 0] == NULL)[0][0]
        nnode_system = nnode_platform + np.maximum(1, nnode_tower) - 1

        nelem_platform = np.where(inputs["platform_elem_A"] == NULL)[0][0]
        nelem_tower = np.where(inputs["tower_elem_A"] == NULL)[0][0]
        nelem_system = nelem_platform + nelem_tower

        # Combine elements indices and have tower base node point to platform transition node
        outputs["tower_Fnode"] = np.zeros(node_tower.shape)
        outputs["tower_elem_n1"] = NULL * np.ones(MEMMAX, dtype=np.int_)
        outputs["tower_elem_n2"] = NULL * np.ones(MEMMAX, dtype=np.int_)
        tower_n1 = np.arange(nelem_tower, dtype=np.int_)
        tower_n2 = np.arange(nelem_tower, dtype=np.int_) + 1
        outputs["tower_elem_n1"][:nelem_tower] = tower_n1.copy()
        outputs["tower_elem_n2"][:nelem_tower] = tower_n2.copy()
        itrans_platform = util.closest_node(node_platform[:nnode_platform, :], inputs["transition_node"])
        tower_n1 += nnode_platform - 1
        tower_n2 += nnode_platform - 1
        tower_n1[0] = itrans_platform

        # Store all outputs
        outputs["system_nodes"] = NULL * np.ones((NNODES_MAX, 3))
        outputs["system_Fnode"] = NULL * np.ones((NNODES_MAX, 3))
        outputs["system_Rnode"] = NULL * np.ones(NNODES_MAX)
        outputs["system_elem_n1"] = NULL * np.ones(NELEM_MAX, dtype=np.int_)
        outputs["system_elem_n2"] = NULL * np.ones(NELEM_MAX, dtype=np.int_)

        outputs["system_nodes"][:nnode_system, :] = np.vstack(
            (node_platform[:nnode_platform, :], node_tower[1:nnode_tower, :])
        )
        outputs["system_Fnode"][:nnode_system, :] = np.vstack(
            (inputs["platform_Fnode"][:nnode_platform, :], outputs["tower_Fnode"][1:nnode_tower, :])
        )
        outputs["system_Rnode"][:nnode_system] = np.r_[
            inputs["platform_Rnode"][:nnode_platform], inputs["tower_Rnode"][1:nnode_tower]
        ]

        outputs["system_elem_n1"][:nelem_system] = np.r_[
            inputs["platform_elem_n1"][:nelem_platform],
            tower_n1,
        ]
        outputs["system_elem_n2"][:nelem_system] = np.r_[
            inputs["platform_elem_n2"][:nelem_platform],
            tower_n2,
        ]

        for var in [
            "elem_D",
            "elem_t",
            "elem_A",
            "elem_Asx",
            "elem_Asy",
            "elem_Ixx",
            "elem_Iyy",
            "elem_Izz",
            "elem_rho",
            "elem_E",
            "elem_G",
        ]:
            outputs["system_" + var] = NULL * np.ones(NELEM_MAX)
            outputs["system_" + var][:nelem_system] = np.r_[
                inputs["platform_" + var][:nelem_platform], inputs["tower_" + var][:nelem_tower]
            ]

        # Have to divide up tower member loads to beginning and end points
        for var in ["elem_Px1", "elem_Py1", "elem_Pz1", "elem_Px2", "elem_Py2", "elem_Pz2"]:
            outputs["system_" + var] = NULL * np.ones(NELEM_MAX)
            outputs["tower_" + var] = NULL * np.ones(MEMMAX)
            tower_P = inputs["tower_" + var[:-1]]
            outputs["tower_" + var][:nelem_tower] = (
                tower_P[:nelem_tower] if var[-1] == "1" else tower_P[1 : (nelem_tower + 1)]
            )
            outputs["system_" + var][:nelem_system] = np.r_[
                inputs["platform_" + var][:nelem_platform], outputs["tower_" + var][:nelem_tower]
            ]

        # Mass summaries
        m_platform = inputs["platform_mass"]
        m_tower = inputs["tower_mass"]
        m_rna = inputs["rna_mass"]
        m_trans = inputs["transition_piece_mass"]
        m_sys = m_platform + m_tower + m_rna + m_trans
        outputs["system_structural_mass"] = m_sys

        outputs["system_structural_center_of_mass"] = (
            m_platform * inputs["platform_center_of_mass"]
            + m_tower * inputs["tower_center_of_mass"]
            + m_rna * (inputs["rna_cg"] + inputs["tower_top_node"])
            + m_trans * inputs["transition_node"]
        ) / m_sys

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

        # Now find the CG of the variable mass assigned to each member
        n_member = capacity.size
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

        cg_variable = np.dot(V_variable_member, cg_variable_member) / V_variable

        # Now find total system mass
        outputs["system_mass"] = m_sys + m_variable
        outputs["system_center_of_mass"] = (
            m_sys * outputs["system_structural_center_of_mass"] + m_variable * cg_variable
        ) / (m_sys + m_variable)

        # Transition piece properties
        m_trans = float(inputs["transition_piece_mass"])
        r_trans = inputs["platform_Rnode"][itrans_platform]
        I_trans = m_trans * r_trans ** 2.0 * np.r_[0.5, 0.5, 1.0, np.zeros(3)]
        outputs["transition_piece_I"] = I_trans


class FrameAnalysis(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_attach = opt["mooring"]["n_attach"]

        self.add_input("platform_mass", 0.0, units="kg")
        self.add_input("platform_center_of_mass", np.zeros(3), units="m")

        self.add_input("tower_nodes", NULL * np.ones((MEMMAX, 3)), units="m")
        self.add_input("tower_Fnode", NULL * np.ones((MEMMAX, 3)), units="N")
        self.add_input("tower_Rnode", NULL * np.ones(MEMMAX), units="m")
        self.add_input("tower_elem_n1", NULL * np.ones(MEMMAX))
        self.add_input("tower_elem_n2", NULL * np.ones(MEMMAX))
        self.add_input("tower_elem_D", NULL * np.ones(MEMMAX), units="m")
        self.add_input("tower_elem_t", NULL * np.ones(MEMMAX), units="m")
        self.add_input("tower_elem_A", NULL * np.ones(MEMMAX), units="m**2")
        self.add_input("tower_elem_Asx", NULL * np.ones(MEMMAX), units="m**2")
        self.add_input("tower_elem_Asy", NULL * np.ones(MEMMAX), units="m**2")
        self.add_input("tower_elem_Ixx", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_input("tower_elem_Iyy", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_input("tower_elem_Izz", NULL * np.ones(MEMMAX), units="kg*m**2")
        self.add_input("tower_elem_rho", NULL * np.ones(MEMMAX), units="kg/m**3")
        self.add_input("tower_elem_E", NULL * np.ones(MEMMAX), units="Pa")
        self.add_input("tower_elem_G", NULL * np.ones(MEMMAX), units="Pa")
        self.add_input("tower_elem_Px1", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_elem_Px2", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_elem_Py1", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_elem_Py2", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_elem_Pz1", NULL * np.ones(MEMMAX), units="N/m")
        self.add_input("tower_elem_Pz2", NULL * np.ones(MEMMAX), units="N/m")

        self.add_input("system_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_input("system_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_input("system_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_input("system_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("system_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("system_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("system_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("system_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("system_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("system_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("system_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("system_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("system_elem_Izz", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("system_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_input("system_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("system_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("system_elem_Px1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("system_elem_Px2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("system_elem_Py1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("system_elem_Py2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("system_elem_Pz1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("system_elem_Pz2", NULL * np.ones(NELEM_MAX), units="N/m")

        self.add_input("transition_node", np.zeros(3), units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("rna_mass", 0.0, units="kg")
        self.add_input("rna_cg", np.zeros(3), units="m")
        self.add_input("rna_F", np.zeros(3), units="N")
        self.add_input("rna_M", np.zeros(3), units="N*m")
        self.add_input("rna_I", np.zeros(6), units="kg*m**2")
        self.add_input("mooring_neutral_load", np.zeros((n_attach, 3)), units="N")
        self.add_input("mooring_fairlead_joints", np.zeros((n_attach, 3)), units="m")

        NFREQ2 = int(NFREQ / 2)
        self.add_output("tower_freqs", val=np.zeros(NFREQ), units="Hz")
        self.add_output("tower_fore_aft_modes", val=np.zeros((NFREQ2, 5)))
        self.add_output("tower_side_side_modes", val=np.zeros((NFREQ2, 5)))
        self.add_output("tower_torsion_modes", val=np.zeros((NFREQ2, 5)))
        self.add_output("tower_fore_aft_freqs", val=np.zeros(NFREQ2))
        self.add_output("tower_side_side_freqs", val=np.zeros(NFREQ2))
        self.add_output("tower_torsion_freqs", val=np.zeros(NFREQ2))

    def compute(self, inputs, outputs):

        # Unpack variables
        opt = self.options["options"]
        n_attach = opt["mooring"]["n_attach"]
        m_rna = float(inputs["rna_mass"])
        cg_rna = inputs["rna_cg"]
        I_rna = inputs["rna_I"]
        I_trans = inputs["transition_piece_I"]

        fairlead_joints = inputs["mooring_fairlead_joints"]
        mooringF = inputs["mooring_neutral_load"]

        # Create frame3dd instance: nodes, elements, reactions, and options
        for frame in ["tower", "system"]:
            nodes = inputs[frame + "_nodes"]
            nnode = np.where(nodes[:, 0] == NULL)[0][0]
            nodes = nodes[:nnode, :]
            rnode = np.zeros(nnode)  # inputs[frame + "_Rnode"][:nnode]
            Fnode = inputs[frame + "_Fnode"][:nnode, :]
            Mnode = np.zeros((nnode, 3))
            ihub = np.argmax(nodes[:, 2]) - 1
            itrans = util.closest_node(nodes, inputs["transition_node"])

            N1 = np.int_(inputs[frame + "_elem_n1"])
            nelem = np.where(N1 == NULL)[0][0]
            N1 = N1[:nelem]
            N2 = np.int_(inputs[frame + "_elem_n2"][:nelem])
            A = inputs[frame + "_elem_A"][:nelem]
            Asx = inputs[frame + "_elem_Asx"][:nelem]
            Asy = inputs[frame + "_elem_Asy"][:nelem]
            Ixx = inputs[frame + "_elem_Ixx"][:nelem]
            Iyy = inputs[frame + "_elem_Iyy"][:nelem]
            Izz = inputs[frame + "_elem_Izz"][:nelem]
            rho = inputs[frame + "_elem_rho"][:nelem]
            E = inputs[frame + "_elem_E"][:nelem]
            G = inputs[frame + "_elem_G"][:nelem]
            roll = np.zeros(nelem)
            L = np.sqrt(np.sum((nodes[N2, :] - nodes[N1, :]) ** 2, axis=1))

            inodes = np.arange(nnode) + 1
            node_obj = pyframe3dd.NodeData(inodes, nodes[:, 0], nodes[:, 1], nodes[:, 2], rnode)

            ielem = np.arange(nelem) + 1
            elem_obj = pyframe3dd.ElementData(ielem, N1 + 1, N2 + 1, A, Asx, Asy, Izz, Ixx, Iyy, E, G, roll, rho)

            # TODO: Hydro_K + Mooring_K for tower (system too?)
            rid = np.array([itrans])  # np.array([np.argmin(nodes[:, 2])])
            Rx = Ry = Rz = Rxx = Ryy = Rzz = np.array([RIGID])
            react_obj = pyframe3dd.ReactionData(rid + 1, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)

            frame3dd_opt = opt["WISDEM"]["FloatingSE"]["frame3dd"]
            opt_obj = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], -1.0)

            myframe = pyframe3dd.Frame(node_obj, react_obj, elem_obj, opt_obj)

            # Added mass
            m_trans = float(inputs["transition_piece_mass"])
            if frame == "tower":
                # TODO: Added mass and stiffness
                m_trans += float(inputs["platform_mass"])
                cg_trans = inputs["transition_node"] - inputs["platform_center_of_mass"]
            else:
                cg_trans = np.zeros(3)
            add_gravity = True
            mID = np.array([itrans, ihub], dtype=np.int_).flatten()
            m_add = np.array([m_trans, m_rna]).flatten()
            I_add = np.c_[I_trans, I_rna]
            cg_add = np.c_[cg_trans, cg_rna]
            myframe.changeExtraNodeMass(
                mID + 1,
                m_add,
                I_add[0, :],
                I_add[1, :],
                I_add[2, :],
                I_add[3, :],
                I_add[4, :],
                I_add[5, :],
                cg_add[0, :],
                cg_add[1, :],
                cg_add[2, :],
                add_gravity,
            )

            # Dynamics
            if frame == "tower" and frame3dd_opt["modal"]:
                Mmethod = 1
                lump = 0
                shift = 0.0
                myframe.enableDynamics(2 * NFREQ, Mmethod, lump, frame3dd_opt["tol"], shift)

            # Initialize loading with gravity, mooring line forces, and buoyancy (already in nodal forces)
            gx = gy = 0.0
            gz = -gravity
            load_obj = pyframe3dd.StaticLoadCase(gx, gy, gz)

            if frame == "system":
                for k in range(n_attach):
                    ind = util.closest_node(nodes, fairlead_joints[k, :])
                    Fnode[ind, :] += mooringF[k, :]
            Fnode[ihub, :] += inputs["rna_F"]
            Mnode[ihub, :] += inputs["rna_M"]
            nF = np.where(np.abs(Fnode).sum(axis=1) > 0.0)[0]
            load_obj.changePointLoads(
                nF + 1, Fnode[nF, 0], Fnode[nF, 1], Fnode[nF, 2], Mnode[nF, 0], Mnode[nF, 1], Mnode[nF, 2]
            )

            # trapezoidally distributed loads
            xx1 = xy1 = xz1 = np.zeros(ielem.size)
            xx2 = xy2 = xz2 = L - 1e-6  # subtract small number b.c. of precision
            wx1 = inputs[frame + "_elem_Px1"][:nelem]
            wx2 = inputs[frame + "_elem_Px2"][:nelem]
            wy1 = inputs[frame + "_elem_Py1"][:nelem]
            wy2 = inputs[frame + "_elem_Py2"][:nelem]
            wz1 = inputs[frame + "_elem_Pz1"][:nelem]
            wz2 = inputs[frame + "_elem_Pz2"][:nelem]
            load_obj.changeTrapezoidalLoads(ielem, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

            # Add the load case and run
            myframe.addLoadCase(load_obj)
            # myframe.write(frame + ".3dd")
            displacements, forces, reactions, internalForces, mass, modal = myframe.run()

            # natural frequncies
            if frame == "tower" and frame3dd_opt["modal"]:
                outputs[frame + "_freqs"] = modal.freq[:NFREQ]

                # Get all mode shapes in batch
                NFREQ2 = int(NFREQ / 2)
                freq_x, freq_y, freq_z, mshapes_x, mshapes_y, mshapes_z = util.get_xyz_mode_shapes(
                    nodes[:, 2], modal.freq, modal.xdsp, modal.ydsp, modal.zdsp, modal.xmpf, modal.ympf, modal.zmpf
                )
                outputs[frame + "_fore_aft_freqs"] = freq_x[:NFREQ2]
                outputs[frame + "_side_side_freqs"] = freq_y[:NFREQ2]
                outputs[frame + "_torsion_freqs"] = freq_z[:NFREQ2]
                outputs[frame + "_fore_aft_modes"] = mshapes_x[:NFREQ2, :]
                outputs[frame + "_side_side_modes"] = mshapes_y[:NFREQ2, :]
                outputs[frame + "_torsion_modes"] = mshapes_z[:NFREQ2, :]

            # Determine forces
            F_sum = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
            M_sum = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])
            L = np.sqrt(np.sum((nodes[N2, :] - nodes[N1, :]) ** 2, axis=1))


class FloatingFrame(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]

        self.add_subsystem("plat", PlatformFrame(options=opt), promotes=["*"])
        self.add_subsystem("pre", TowerPreMember(), promotes=["*"])

        prom = [
            "E_mat",
            "G_mat",
            "sigma_y_mat",
            "rho_mat",
            "rho_water",
            "unit_cost_mat",
            "material_names",
            "painting_cost_rate",
            "labor_cost_rate",
        ]
        prom += [
            ("nodes_xyz", "tower_nodes"),
            ("nodes_r", "tower_Rnode"),
            ("total_mass", "tower_mass"),
            ("total_cost", "tower_cost"),
            ("center_of_mass", "tower_center_of_mass"),
            ("joint1", "transition_node"),
            ("joint2", "tower_top_node"),
            ("Px", "tower_elem_Px"),
            ("Py", "tower_elem_Py"),
            ("Pz", "tower_elem_Pz"),
        ]
        for var in ["D", "t", "A", "Asx", "Asy", "rho", "Ixx", "Iyy", "Izz", "E", "G"]:
            prom += [("section_" + var, "tower_elem_" + var)]

        prom += [
            "Uref",
            "zref",
            "z0",
            "shearExp",
            "cd_usr",
            "cm",
            "beta_wind",
            "rho_air",
            "mu_air",
            "beta_wave",
            "mu_water",
            "Uc",
            "Hsig_wave",
            "Tsig_wave",
            "water_depth",
        ]
        self.add_subsystem(
            "tower",
            Member(column_options=opt["floating"]["tower"], idx=0, n_mat=opt["materials"]["n_mat"]),
            promotes=prom,
        )
        self.add_subsystem("mux", PlatformTowerFrame(options=opt), promotes=["*"])
        self.add_subsystem("frame", FrameAnalysis(options=opt), promotes=["*"])
