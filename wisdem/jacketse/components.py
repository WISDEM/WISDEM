import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt

import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.cross_sections as cs
from wisdem.commonse import NFREQ, gravity

E_input = 2.1e11
G_input = 8.077e10
rho_input = 7850.0

RIGID = 1e30
NREFINE = 3


class GetGreekLetters(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_legs", types=int)
        self.options.declare("n_bays", types=int)

    def setup(self):
        n_legs = self.options["n_legs"]
        n_bays = self.options["n_bays"]
        self.add_input("r_foot", val=10.0)
        self.add_input("r_head", val=6.0)
        self.add_input("L", val=70.0)
        self.add_input("q", val=0.8)
        self.add_input("l_osg", val=5.0)
        self.add_input("l_tp", val=4.0)

        self.add_input("gamma_b", val=12.0)
        self.add_input("gamma_t", val=18.0)
        self.add_input("beta_b", val=0.5)
        self.add_input("beta_t", val=0.8)
        self.add_input("tau_b", val=0.3)
        self.add_input("tau_t", val=0.6)

        self.add_output("gamma_i", val=np.zeros((n_bays + 1)))
        self.add_output("beta_i", val=np.zeros((n_bays)))
        self.add_output("tau_i", val=np.zeros((n_bays)))

        self.add_output("xi", val=0.0)
        self.add_output("nu", val=0.0)
        self.add_output("psi_s", val=0.0)
        self.add_output("psi_p", val=0.0)

        self.add_output("lower_bay_heights", val=np.zeros((n_bays + 1)))
        self.add_output("lower_bay_radii", val=np.zeros((n_bays + 1)))
        self.add_output("l_mi", val=np.zeros((n_bays)))

    def compute(self, inputs, outputs):
        n_legs = self.options["n_legs"]
        n_bays = self.options["n_bays"]
        r_foot = inputs["r_foot"]
        r_head = inputs["r_head"]
        L = inputs["L"]
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
        xi = r_head / r_foot  # must be <= 1.
        nu = 2 * np.pi / n_legs  # angle enclosed by two legs
        psi_s = np.arctan(r_foot * (1 - xi) / L)  # spatial batter angle
        psi_p = np.arctan(r_foot * (1 - xi) * np.sin(nu / 2.0) / L)

        tmp = q ** np.arange(n_bays)
        bay_heights = l_i = (L - l_osg - l_tp) / (np.sum(tmp) / tmp)
        # TODO : add test to verify np.sum(bay_heights == L)

        lower_bay_heights = np.hstack((0.0, bay_heights))
        lower_bay_radii = r_i = r_foot - np.tan(psi_s) * (l_osg + np.cumsum(lower_bay_heights))

        # x joint layer info
        l_mi = l_i * r_i[:-1] / (r_i[:-1] + r_i[1:])
        # r_mi = r_foot - np.tan(psi_s) * (l_osg + np.cumsum(lower_bay_heights) + l_mi)  # I don't think we actually use this value later
        # print(r_mi)

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
    def initialize(self):
        self.options.declare("n_legs", types=int)
        self.options.declare("n_bays", types=int)

    def setup(self):
        n_legs = self.options["n_legs"]
        n_bays = self.options["n_bays"]

        self.add_input("xi", val=0.0)
        self.add_input("nu", val=0.0)
        self.add_input("psi_s", val=0.0)
        self.add_input("psi_p", val=0.0)
        self.add_input("gamma_i", val=np.zeros((n_bays + 1)))
        self.add_input("beta_i", val=np.zeros((n_bays)))
        self.add_input("tau_i", val=np.zeros((n_bays)))
        self.add_input("lower_bay_heights", val=np.zeros((n_bays + 1)))
        self.add_input("lower_bay_radii", val=np.zeros((n_bays + 1)))
        self.add_input("l_mi", val=np.zeros((n_bays)))
        self.add_input("l_osg", val=5.0)
        self.add_input("L", val=70.0)
        self.add_input("r_foot", val=10.0)
        self.add_input("r_head", val=6.0)

        self.add_output("leg_nodes", val=np.zeros((n_legs, n_bays + 2, 3)))
        self.add_output("bay_nodes", val=np.zeros((n_legs, n_bays + 1, 3)))

    def compute(self, inputs, outputs):
        n_legs = self.options["n_legs"]
        n_bays = self.options["n_bays"]
        xi = inputs["xi"]
        nu = inputs["nu"]
        psi_s = inputs["psi_s"]
        psi_p = inputs["psi_p"]
        gamma_i = inputs["gamma_i"]
        beta_i = inputs["beta_i"]
        tau_i = inputs["tau_i"]
        lower_bay_heights = inputs["lower_bay_heights"]
        lower_bay_radii = inputs["lower_bay_radii"]
        l_mi = inputs["l_mi"]
        l_osg = inputs["l_osg"]
        L = inputs["L"]
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
    def initialize(self):
        self.options.declare("n_legs", types=int)
        self.options.declare("n_bays", types=int)

    def setup(self):
        n_legs = self.options["n_legs"]
        n_bays = self.options["n_bays"]
        self.add_input("d_l", val=0.5)
        self.add_input("gamma_i", val=np.zeros((n_bays + 1)))
        self.add_input("beta_i", val=np.zeros((n_bays)))
        self.add_input("tau_i", val=np.zeros((n_bays)))

        self.add_output("leg_thicknesses", val=np.zeros((n_bays + 1)))
        self.add_output("brace_diameters", val=np.zeros((n_bays)))
        self.add_output("brace_thicknesses", val=np.zeros((n_bays)))

    def compute(self, inputs, outputs):
        d_l = inputs["d_l"]
        gamma_i = inputs["gamma_i"]
        beta_i = inputs["beta_i"]
        tau_i = inputs["tau_i"]

        outputs["leg_thicknesses"] = t_l = d_l / (2 * gamma_i)
        outputs["brace_diameters"] = d_b = beta_i * d_l
        outputs["brace_thicknesses"] = t_b = tau_i * t_l[:-1]


class ComputeFrame3DD(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("n_legs", types=int)
        self.options.declare("n_bays", types=int)
        self.options.declare("x_mb", types=bool)

    def setup(self):
        n_legs = self.options["n_legs"]
        n_bays = self.options["n_bays"]
        x_mb = self.options["x_mb"]

        self.add_input("leg_nodes", val=np.zeros((n_legs, n_bays + 2, 3)))
        self.add_input("bay_nodes", val=np.zeros((n_legs, n_bays + 1, 3)))
        self.add_input("d_l", val=0.5)
        self.add_input("leg_thicknesses", val=np.zeros((n_bays + 1)))
        self.add_input("brace_diameters", val=np.zeros((n_bays)))
        self.add_input("brace_thicknesses", val=np.zeros((n_bays)))

        n_dlc = 1
        self.add_input("turbine_F", np.zeros((3, n_dlc)), units="N")
        self.add_input("turbine_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")

        n_elem = 2 * (n_legs * (n_bays + 1)) + 4 * (n_legs * n_bays) + int(x_mb) * n_legs

        self.add_output("platform_base_F", np.zeros((3, n_dlc)), units="N")
        self.add_output("platform_base_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_output("platform_Fz", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("platform_Vx", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("platform_Vy", np.zeros((n_elem, n_dlc)), units="N")
        self.add_output("platform_Mxx", np.zeros((n_elem, n_dlc)), units="N*m")
        self.add_output("platform_Myy", np.zeros((n_elem, n_dlc)), units="N*m")
        self.add_output("platform_Mzz", np.zeros((n_elem, n_dlc)), units="N*m")

    def compute(self, inputs, outputs):
        n_legs = self.options["n_legs"]
        n_bays = self.options["n_bays"]
        x_mb = self.options["x_mb"]

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

                alpha = (n4 - n2) / ((n1 - n2) - (n3 - n4))
                alpha = alpha[0]
                new_node = n4 + alpha * (n3 - n4)
                x_nodes[idx, jdx, :] = new_node

        xyz = np.vstack((leg_nodes.reshape(-1, 3), bay_nodes.reshape(-1, 3), x_nodes.reshape(-1, 3)))
        n = xyz.shape[0]
        node_indices = np.arange(1, n + 1)
        r = np.zeros(n)
        nodes = pyframe3dd.NodeData(node_indices, xyz[:, 0], xyz[:, 1], xyz[:, 2], r)

        leg_indices = node_indices[: leg_nodes.size // 3].reshape((n_legs, n_bays + 2))
        bay_indices = node_indices[leg_nodes.size // 3 : leg_nodes.size // 3 + bay_nodes.size // 3].reshape(
            (n_legs, n_bays + 1)
        )
        x_indices = node_indices[leg_nodes.size // 3 + bay_nodes.size // 3 :].reshape((n_legs, n_bays))

        rnode = np.array(leg_indices[:, 0], dtype=np.int_)
        kx = ky = kz = ktx = kty = ktz = np.array([RIGID] * len(rnode))
        reactions = pyframe3dd.ReactionData(rnode, kx, ky, kz, ktx, kty, ktz, rigid=RIGID)

        # ------ frame element data ------------

        self.num_elements = 0
        self.N1 = []
        self.N2 = []
        self.L = []
        self.Area = []
        self.Asx = []
        self.Asy = []
        self.J0 = []
        self.Ixx = []
        self.Iyy = []
        self.vol = []

        def add_element(n1_nodes, n1_indices, n2_nodes, n2_indices, itube, idx1, idx2, idx3, idx4):
            n1 = n1_nodes[idx1, idx2]
            n2 = n2_nodes[idx3, idx4]
            self.N1.append(n1_indices[idx1, idx2])
            self.N2.append(n2_indices[idx3, idx4])
            length = np.linalg.norm(n2 - n1)
            self.L.append(length)
            self.Area.append(itube.Area)
            self.Asx.append(itube.Asx)
            self.Asy.append(itube.Asy)
            self.J0.append(itube.J0)
            self.Ixx.append(itube.Ixx)
            self.Iyy.append(itube.Iyy)
            self.vol.append(itube.Area * length)
            self.num_elements += 1

        # Naive for loops to make sure we get indexing right.
        # Can vectorize later as needed.
        for jdx in range(n_bays + 1):
            itube = cs.Tube(inputs["d_l"], inputs["leg_thicknesses"][jdx])
            for idx in range(n_legs):
                add_element(leg_nodes, leg_indices, bay_nodes, bay_indices, itube, idx, jdx, idx, jdx)
                add_element(bay_nodes, bay_indices, leg_nodes, leg_indices, itube, idx, jdx, idx, jdx + 1)

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

        E = [E_input] * self.num_elements
        G = [G_input] * self.num_elements
        rho = [rho_input] * self.num_elements

        Area = np.squeeze(np.array(self.Area, dtype=np.float))
        Asx = np.squeeze(np.array(self.Asx, dtype=np.float))
        Asy = np.squeeze(np.array(self.Asy, dtype=np.float))
        J0 = np.squeeze(np.array(self.J0, dtype=np.float))
        Ixx = np.squeeze(np.array(self.Ixx, dtype=np.float))
        Iyy = np.squeeze(np.array(self.Iyy, dtype=np.float))
        E = np.squeeze(np.array(E, dtype=np.float))
        G = np.squeeze(np.array(G, dtype=np.float))
        rho = np.squeeze(np.array(rho, dtype=np.float))
        N1 = self.N1
        N2 = self.N2

        element = np.arange(1, self.num_elements + 1)
        roll = np.zeros(self.num_elements - 1)

        plot = True

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])

            for (n1, n2) in zip(N1, N2):
                n1 -= 1
                n2 -= 1
                plt.plot([xyz[n1][0], xyz[n2][0]], [xyz[n1][1], xyz[n2][1]], [xyz[n1][2], xyz[n2][2]])

            plt.show()

        elements = pyframe3dd.ElementData(element, N1, N2, Area, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)

        # ------ options ------------
        dx = -1.0
        options = pyframe3dd.Options(True, True, dx)  # TODO : replace with options
        # -----------------------------------

        # initialize frame3dd object
        self.frame = pyframe3dd.Frame(nodes, reactions, elements, options)

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

        n_dlc = 1
        for k in range(n_dlc):
            load_obj = pyframe3dd.StaticLoadCase(gx, gy, gz)

            # Fnode2 = Fnode.copy()
            # Fnode2[ihub, :] += inputs["turbine_F"][:, k]
            # Mnode[ihub, :] = inputs["turbine_M"][:, k]
            # nF = np.where(np.abs(Fnode2).sum(axis=1) > 0.0)[0]
            # load_obj.changePointLoads(
            #     nF + 1, Fnode2[nF, 0], Fnode2[nF, 1], Fnode2[nF, 2], Mnode[nF, 0], Mnode[nF, 1], Mnode[nF, 2]
            # )

            # # trapezoidally distributed loads
            # xx1 = xy1 = xz1 = np.zeros(ielem.size)
            # xx2 = xy2 = xz2 = 0.99 * L  # multiply slightly less than unity b.c. of precision
            # wx1 = inputs["platform_elem_Px1"][:nelem, k]
            # wx2 = inputs["platform_elem_Px2"][:nelem, k]
            # wy1 = inputs["platform_elem_Py1"][:nelem, k]
            # wy2 = inputs["platform_elem_Py2"][:nelem, k]
            # wz1 = inputs["platform_elem_Pz1"][:nelem, k]
            # wz2 = inputs["platform_elem_Pz2"][:nelem, k]
            # load_obj.changeTrapezoidalLoads(ielem, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

            # Add the load case and run
            self.frame.addLoadCase(load_obj)
        # self.frame.write("system.3dd")
        # self.frame.draw()
        displacements, forces, reactions, internalForces, mass, modal = self.frame.run()

        # Determine reaction forces
        outputs["platform_base_F"] = -np.c_[
            reactions.Fx.sum(axis=1), reactions.Fy.sum(axis=1), reactions.Fz.sum(axis=1)
        ].T
        outputs["platform_base_M"] = -np.c_[
            reactions.Mxx.sum(axis=1), reactions.Myy.sum(axis=1), reactions.Mzz.sum(axis=1)
        ].T

        # Forces and moments along the structure
        outputs["platform_Fz"] = forces.Nx[k, 1::2]
        outputs["platform_Vx"] = -forces.Vz[k, 1::2]
        outputs["platform_Vy"] = forces.Vy[k, 1::2]
        outputs["platform_Mxx"] = -forces.Mzz[k, 1::2]
        outputs["platform_Myy"] = forces.Myy[k, 1::2]
        outputs["platform_Mzz"] = forces.Txx[k, 1::2]


# Assemble the system together in an OpenMDAO Group
class JacketSE(om.Group):
    def setup(self):
        x_mb = True  # if there's a mud brace
        n_legs = 4
        n_bays = 4

        self.add_subsystem("greek", GetGreekLetters(n_legs=n_legs, n_bays=n_bays), promotes=["*"])
        self.add_subsystem("nodes", ComputeNodes(n_legs=n_legs, n_bays=n_bays), promotes=["*"])
        self.add_subsystem("properties", ComputeDiameterAndThicknesses(n_legs=n_legs, n_bays=n_bays), promotes=["*"])
        self.add_subsystem("frame3dd", ComputeFrame3DD(n_legs=n_legs, n_bays=n_bays, x_mb=x_mb), promotes=["*"])


if __name__ == "__main__":
    prob = om.Problem(model=JacketSE())

    prob.setup()

    prob.run_model()
