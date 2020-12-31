import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
from wisdem.commonse import gravity

"""
from wisdem.commonse.utilities import nodal2sectional

from wisdem.commonse import gravity, eps, NFREQ
import wisdem.commonse.utilization_constraints as util
import wisdem.commonse.manufacturing as manufacture
from wisdem.commonse.wind_wave_drag import CylinderWindDrag
from wisdem.commonse.environment import PowerWind
from wisdem.commonse.vertical_cylinder import get_nfull, RIGID
from wisdem.commonse.cross_sections import Tube
from wisdem.floatingse.map_mooring import NLINES_MAX
"""

NNODES_MAX = 1000
NELEM_MAX = 1000
NULL = -9999
RIGID = 1e30

# TODO:
# - Mass summary
# - System CG
# - System CB
# - Stress or buckling?


class PlatformFrame(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_member = opt["floating"]["n_member"]

        for k in range(n_member):
            self.add_input("member" + str(k) + ":nodes_xyz", shape_by_conn=True, units="m")
            self.add_input("member" + str(k) + ":nodes_r", shape_by_conn=True, units="m")
            self.add_input("member" + str(k) + ":section_A", shape_by_conn=True, units="m**2")
            self.add_input("member" + str(k) + ":section_Asx", shape_by_conn=True, units="m**2")
            self.add_input("member" + str(k) + ":section_Asy", shape_by_conn=True, units="m**2")
            self.add_input("member" + str(k) + ":section_Ixx", shape_by_conn=True, units="kg*m**2")
            self.add_input("member" + str(k) + ":section_Iyy", shape_by_conn=True, units="kg*m**2")
            self.add_input("member" + str(k) + ":section_Izz", shape_by_conn=True, units="kg*m**2")
            self.add_input("member" + str(k) + ":section_rho", shape_by_conn=True, units="kg/m**3")
            self.add_input("member" + str(k) + ":section_E", shape_by_conn=True, units="Pa")
            self.add_input("member" + str(k) + ":section_G", shape_by_conn=True, units="Pa")
            self.add_discrete_input("member" + str(k) + ":idx_cb", 0)
            self.add_input("member" + str(k) + ":buoyancy_force", 0.0, units="N")
            self.add_input("member" + str(k) + ":displacement", 0.0, units="m**3")
            self.add_input("member" + str(k) + ":center_of_buoyancy", np.zeros(3), units="m")
            self.add_input("member" + str(k) + ":center_of_mass", np.zeros(3), units="m")
            self.add_input("member" + str(k) + ":total_mass", 0.0, units="kg")
            self.add_input("member" + str(k) + ":total_cost", 0.0, units="USD")
            self.add_input("member" + str(k) + ":Awater", 0.0, units="m**2")
            self.add_input("member" + str(k) + ":Iwater", 0.0, units="m**4")
            self.add_input("member" + str(k) + ":added_mass", np.zeros(6), units="kg")

        self.add_output("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_output("platform_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_output("platform_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_discrete_output("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_discrete_output("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_output("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_output("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_Izz", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_output("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_output("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_output("platform_displacement", 0.0, units="m**3")
        self.add_output("platform_center_of_buoyancy", np.zeros(3), units="m")
        self.add_output("platform_center_of_mass", np.zeros(3), units="m")
        self.add_output("platform_mass", 0.0, units="kg")
        self.add_output("platform_cost", 0.0, units="USD")
        self.add_output("platform_Awater", 0.0, units="m**2")
        self.add_output("platform_Iwater", 0.0, units="m**4")
        self.add_output("platform_added_mass", np.zeros(6), units="kg")

        self.node_mem2glob = {}
        # self.node_glob2mem = {}

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # This shouldn't change during an optimization, so save some time?
        if len(self.node_mem2glob) == 0:
            self.set_connectivity(inputs, outputs, discrete_inputs, discrete_outputs)

        self.set_node_props(inputs, outputs, discrete_inputs, discrete_outputs)
        self.set_element_props(inputs, outputs)

    def set_connectivity(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["n_member"]

        # Initialize running lists across all members
        nodes_temp = np.empty((0, 3))
        elem_n1 = np.array([], dtype=np.int_)
        elem_n2 = np.array([], dtype=np.int_)

        # Look over members and grab all nodes and internal connections
        for k in range(n_member):
            inode_xyz = inputs["member" + str(k) + ":nodes_xyz"]
            inodes = inode_xyz.shape[0]
            inode_range = np.arange(inodes - 1)

            n = nodes_temp.shape[0]
            for ii in range(inodes):
                self.node_mem2glob[(k, ii)] = n + ii

            elem_n1 = np.append(elem_n1, n + inode_range)
            elem_n2 = np.append(elem_n2, n + inode_range + 1)
            nodes_temp = np.append(nodes_temp, inode_xyz, axis=0)

        # Reveal connectivity by using mapping to unique node positions
        nodes, idx, inv = np.unique(nodes_temp.round(4), axis=0, return_index=True, return_inverse=True)
        nnode = nodes.shape[0]
        outputs["platform_nodes"] = NULL * np.ones((NNODES_MAX, 3))
        outputs["platform_nodes"][:nnode, :] = nodes

        # Use mapping to set references to node joints
        nelem = elem_n1.size
        discrete_outputs["platform_elem_n1"] = NULL * np.ones(NELEM_MAX, dtype=np.int_)
        discrete_outputs["platform_elem_n2"] = NULL * np.ones(NELEM_MAX, dtype=np.int_)
        discrete_outputs["platform_elem_n1"][:nelem] = inv[elem_n1]
        discrete_outputs["platform_elem_n2"][:nelem] = inv[elem_n2]

        # Update global 2 member mappings
        for k in self.node_mem2glob.keys():
            self.node_mem2glob[k] = inv[self.node_mem2glob[k]]

    def set_node_props(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["n_member"]

        # Number of valid nodes
        nnode = np.where(outputs["platform_nodes"][:, 0] == NULL)[0][0]

        # Find greatest radius of all members at node intersections
        Rnode = np.zeros(nnode)
        for k in range(n_member):
            irnode = inputs["member" + str(k) + ":nodes_r"]
            n = irnode.shape[0]
            for ii in range(n):
                iglob = self.node_mem2glob[(k, ii)]
                Rnode[iglob] = np.array([Rnode[iglob], irnode[ii]]).max()

        # Find forces on nodes
        Fnode = np.zeros((nnode, 3))
        for k in range(n_member):
            icb = discrete_inputs["member" + str(k) + ":idx_cb"]
            iglob = self.node_mem2glob[(k, icb)]
            Fnode[iglob, 2] = inputs["member" + str(k) + ":buoyancy_force"]

        # Store outputs
        outputs["platform_Rnode"] = NULL * np.ones(NNODES_MAX)
        outputs["platform_Rnode"][:nnode] = Rnode
        outputs["platform_Fnode"] = NULL * np.ones((NNODES_MAX, 3))
        outputs["platform_Fnode"][:nnode, :] = Fnode

    def set_element_props(self, inputs, outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["n_member"]

        # Initialize running lists across all members
        elem_A = np.array([])
        elem_Asx = np.array([])
        elem_Asy = np.array([])
        elem_Ixx = np.array([])
        elem_Iyy = np.array([])
        elem_Izz = np.array([])
        elem_rho = np.array([])
        elem_E = np.array([])
        elem_G = np.array([])

        mass = 0.0
        cost = 0.0
        volume = 0.0
        Awater = 0.0
        Iwater = 0.0
        m_added = np.zeros(6)
        cg_plat = np.zeros(3)
        cb_plat = np.zeros(3)

        # Append all member data
        for k in range(n_member):
            elem_A = np.append(elem_A, inputs["member" + str(k) + ":section_A"])
            elem_Asx = np.append(elem_Asx, inputs["member" + str(k) + ":section_Asx"])
            elem_Asy = np.append(elem_Asy, inputs["member" + str(k) + ":section_Asy"])
            elem_Ixx = np.append(elem_Ixx, inputs["member" + str(k) + ":section_Ixx"])
            elem_Iyy = np.append(elem_Iyy, inputs["member" + str(k) + ":section_Iyy"])
            elem_Izz = np.append(elem_Izz, inputs["member" + str(k) + ":section_Izz"])
            elem_rho = np.append(elem_rho, inputs["member" + str(k) + ":section_rho"])
            elem_E = np.append(elem_E, inputs["member" + str(k) + ":section_E"])
            elem_G = np.append(elem_G, inputs["member" + str(k) + ":section_G"])

            # Mass, volume, cost tallies
            imass = inputs["member" + str(k) + ":total_mass"]
            ivol = inputs["member" + str(k) + ":displacement"]

            mass += imass
            volume += ivol
            cost += inputs["member" + str(k) + ":total_mass"]
            Awater += inputs["member" + str(k) + ":Awater"]
            Iwater += inputs["member" + str(k) + ":Iwater"]
            m_added += inputs["member" + str(k) + ":added_mass"]

            # Center of mass / buoyancy tallies
            cg_plat += imass * inputs["member" + str(k) + ":center_of_mass"]
            cb_plat += ivol * inputs["member" + str(k) + ":center_of_buoyancy"]

        # Store outputs
        nelem = elem_A.size
        outputs["platform_elem_A"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Asx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Asy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Ixx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Iyy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Izz"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_rho"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_E"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_G"] = NULL * np.ones(NELEM_MAX)

        outputs["platform_elem_A"][:nelem] = elem_A
        outputs["platform_elem_Asx"][:nelem] = elem_Asx
        outputs["platform_elem_Asy"][:nelem] = elem_Asy
        outputs["platform_elem_Ixx"][:nelem] = elem_Ixx
        outputs["platform_elem_Iyy"][:nelem] = elem_Iyy
        outputs["platform_elem_Izz"][:nelem] = elem_Izz
        outputs["platform_elem_rho"][:nelem] = elem_rho
        outputs["platform_elem_E"][:nelem] = elem_E
        outputs["platform_elem_G"][:nelem] = elem_G

        outputs["platform_mass"] = mass
        outputs["platform_cost"] = cost
        outputs["platform_displacement"] = volume
        outputs["platform_center_of_mass"] = cg_plat / mass
        outputs["platform_center_of_buoyancy"] = cb_plat / volume
        outputs["platform_Awater"] = Awater
        outputs["platform_Iwater"] = Iwater
        outputs["platform_added_mass"] = m_added


class PlatformTowerFrame(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]

    def compute(self, inputs, outputs):
        opt = self.options["options"]


class FrameAnalysis(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_lines = opt["mooring"]["n_lines"]

        self.add_input("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_input("platform_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_input("platform_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_discrete_input("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_discrete_input("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Izz", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("mooring_neutral_load", np.zeros((n_lines, 3)), units="N")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        # Unpack variables
        n_lines = self.options["options"]["mooring"]["n_lines"]
        nodes = inputs["platform_nodes"]
        nnode = np.where(nodes[:, 0] == NULL)[0][0]
        nodes = nodes[:nnode, :]
        rnode = inputs["platform_Rnode"][:nnode]
        Fnode = inputs["platform_Fnode"][:nnode, :]

        N1 = discrete_inputs["platform_elem_n1"]
        nelem = np.where(N1 == NULL)[0][0]
        N1 = N1[:nelem]
        N2 = discrete_inputs["platform_elem_n2"][:nelem]
        A = inputs["platform_elem_A"][:nelem]
        Asx = inputs["platform_elem_Asx"][:nelem]
        Asy = inputs["platform_elem_Asy"][:nelem]
        Ixx = inputs["platform_elem_Ixx"][:nelem]
        Iyy = inputs["platform_elem_Iyy"][:nelem]
        Izz = inputs["platform_elem_Izz"][:nelem]
        rho = inputs["platform_elem_rho"][:nelem]
        E = inputs["platform_elem_E"][:nelem]
        G = inputs["platform_elem_G"][:nelem]
        roll = np.zeros(nelem)

        fairlead_joints = inputs["mooring_fairlead_joints"]
        mooringF = inputs["mooring_neutral_load"]

        # Create frame3dd instance: nodes, elements, reactions, and options
        inodes = np.arange(nnode) + 1
        node_obj = pyframe3dd.NodeData(inodes, nodes[:, 0], nodes[:, 1], nodes[:, 2], rnode)

        ielem = np.arange(nelem) + 1
        elem_obj = pyframe3dd.ElementData(ielem, N1 + 1, N2 + 1, A, Asx, Asy, Izz, Ixx, Iyy, E, G, roll, rho)

        rid = np.argmin(nodes[:, 2]) + 1
        Rx = Ry = Rz = Rxx = Ryy = Rzz = np.array([RIGID])
        react_obj = pyframe3dd.ReactionData(rid, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)

        frame3dd_opt = self.options["floating"]["frame3dd"]
        opt_obj = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], -1.0)

        myframe = pyframe3dd.Frame(node_obj, react_obj, elem_obj, opt_obj)

        # Initialize loading with gravity, mooring line forces, and buoyancy (already in nodal forces)
        gx = gy = 0.0
        gz = -gravity
        load_obj = pyframe3dd.StaticLoadCase(gx, gy, gz)

        for k in range(n_lines):
            ind = util.closest_node(nodes, fairlead_joints[k, :])
            Fnode[ind, :] += mooringF[k, :]
        nF = np.where(np.abs(Fnode).sum(axis=1) > 0.0)[0]
        M = np.zeros((nF.size, 3))
        load_obj.changePointLoads(nF + 1, Fnode[nF, 0], Fnode[nF, 1], Fnode[nF, 2], M[:, 0], M[:, 1], M[:, 2])

        # Add the load case and run
        myframe.addLoadCase(load_obj)
        displacements, forces, reactions, internalForces, mass, modal = myframe.run()

        # Determine needed variable ballast
        F_sum = -1.0 * np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        M_sum = -1.0 * np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])
        print(F_sum)
        print(M_sum)


class FloatingFrame(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]

        self.add_subsystem("plat", PlatformFrame(options=opt), promotes=["*"])
        # self.add_subsystem("tow", PlatformTowerFrame(options=opt), promotes=["*"])
        self.add_subsystem("frame", FrameAnalysis(options=opt), promotes=["*"])
