import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
import wisdem.pyframe3dd.pyframe3dd as pyframe3dd
import wisdem.commonse.utilization_dnvgl as util_dnvgl
import wisdem.commonse.utilization_constraints as util_con
from wisdem.commonse import NFREQ, gravity
from wisdem.commonse.cylinder_member import NULL, MEMMAX, MemberLoads, get_nfull

NNODES_MAX = 1000
NELEM_MAX = 1000
RIGID = 1e30
EPS = 1e-6


class PlatformLoads(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]

        for k in range(n_member):
            self.add_input(f"member{k}:Px", np.zeros(MEMMAX), units="N/m")
            self.add_input(f"member{k}:Py", np.zeros(MEMMAX), units="N/m")
            self.add_input(f"member{k}:Pz", np.zeros(MEMMAX), units="N/m")
            self.add_input(f"member{k}:qdyn", np.zeros(MEMMAX), units="Pa")

        self.add_output("platform_elem_Px1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Px2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Py1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Py2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Pz1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_Pz2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_output("platform_elem_qdyn", NULL * np.ones(NELEM_MAX), units="Pa")

    def compute(self, inputs, outputs):
        # Load in number of members
        opt = self.options["options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Initialize running lists across all members
        elem_Px1 = np.array([])
        elem_Px2 = np.array([])
        elem_Py1 = np.array([])
        elem_Py2 = np.array([])
        elem_Pz1 = np.array([])
        elem_Pz2 = np.array([])
        elem_qdyn = np.array([])

        # Append all member data
        for k in range(n_member):
            n = np.where(inputs[f"member{k}:qdyn"] == NULL)[0][0]
            mem_qdyn, _ = util.nodal2sectional(inputs[f"member{k}:qdyn"][:n])
            elem_qdyn = np.append(elem_qdyn, mem_qdyn)

            # The loads should come in with length n+1
            n -= 1
            elem_Px1 = np.append(elem_Px1, inputs[f"member{k}:Px"][:n])
            elem_Px2 = np.append(elem_Px2, inputs[f"member{k}:Px"][1 : (n + 1)])
            elem_Py1 = np.append(elem_Py1, inputs[f"member{k}:Py"][:n])
            elem_Py2 = np.append(elem_Py2, inputs[f"member{k}:Py"][1 : (n + 1)])
            elem_Pz1 = np.append(elem_Pz1, inputs[f"member{k}:Pz"][:n])
            elem_Pz2 = np.append(elem_Pz2, inputs[f"member{k}:Pz"][1 : (n + 1)])

        # Store outputs
        nelem = elem_qdyn.size
        outputs["platform_elem_Px1"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Px2"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Py1"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Py2"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Pz1"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_Pz2"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_elem_qdyn"] = NULL * np.ones(NELEM_MAX)

        outputs["platform_elem_Px1"][:nelem] = elem_Px1
        outputs["platform_elem_Px2"][:nelem] = elem_Px2
        outputs["platform_elem_Py1"][:nelem] = elem_Py1
        outputs["platform_elem_Py2"][:nelem] = elem_Py2
        outputs["platform_elem_Pz1"][:nelem] = elem_Pz1
        outputs["platform_elem_Pz2"][:nelem] = elem_Pz2
        outputs["platform_elem_qdyn"][:nelem] = elem_qdyn


class FrameAnalysis(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        opt = self.options["options"]
        n_attach = opt["mooring"]["n_attach"]

        self.add_input("platform_mass", 0.0, units="kg")
        self.add_input("platform_hull_center_of_mass", np.zeros(3), units="m")
        self.add_input("platform_added_mass", np.zeros(6), units="kg")

        self.add_input("platform_nodes", NULL * np.ones((NNODES_MAX, 3)), units="m")
        self.add_input("platform_Fnode", NULL * np.ones((NNODES_MAX, 3)), units="N")
        self.add_input("platform_Rnode", NULL * np.ones(NNODES_MAX), units="m")
        self.add_input("platform_elem_n1", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_n2", NULL * np.ones(NELEM_MAX, dtype=np.int_))
        self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_L", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_J0", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_rho", NULL * np.ones(NELEM_MAX), units="kg/m**3")
        self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")

        self.add_input("platform_elem_Px1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Px2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Py1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Py2", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Pz1", NULL * np.ones(NELEM_MAX), units="N/m")
        self.add_input("platform_elem_Pz2", NULL * np.ones(NELEM_MAX), units="N/m")

        self.add_input("transition_node", np.zeros(3), units="m")
        self.add_input("transition_piece_mass", 0.0, units="kg")
        self.add_input("transition_piece_I", np.zeros(6), units="kg*m**2")
        self.add_input("turbine_F", np.zeros(3), units="N")
        self.add_input("turbine_M", np.zeros(3), units="N*m")
        self.add_input("mooring_neutral_load", np.zeros((n_attach, 3)), units="N")
        self.add_input("mooring_fairlead_joints", np.zeros((n_attach, 3)), units="m")
        self.add_input("mooring_stiffness", np.zeros((6, 6)), units="N/m")
        self.add_input("variable_ballast_mass", 0.0, units="kg")
        self.add_input("variable_center_of_mass", val=np.zeros(3), units="m")

        self.add_output("platform_base_F", np.zeros(3), units="N")
        self.add_output("platform_base_M", np.zeros(3), units="N*m")
        self.add_output("platform_Fz", NULL * np.ones(NELEM_MAX), units="N")
        self.add_output("platform_Vx", NULL * np.ones(NELEM_MAX), units="N")
        self.add_output("platform_Vy", NULL * np.ones(NELEM_MAX), units="N")
        self.add_output("platform_Mxx", NULL * np.ones(NELEM_MAX), units="N*m")
        self.add_output("platform_Myy", NULL * np.ones(NELEM_MAX), units="N*m")
        self.add_output("platform_Mzz", NULL * np.ones(NELEM_MAX), units="N*m")

    def compute(self, inputs, outputs):

        # Unpack variables
        opt = self.options["options"]
        n_attach = opt["mooring"]["n_attach"]
        m_trans = float(inputs["transition_piece_mass"])
        I_trans = inputs["transition_piece_I"]
        m_variable = float(inputs["variable_ballast_mass"])
        cg_variable = inputs["variable_center_of_mass"]

        fairlead_joints = inputs["mooring_fairlead_joints"]
        mooringF = inputs["mooring_neutral_load"]
        mooringK = np.abs(np.diag(inputs["mooring_stiffness"]))

        # Create frame3dd instance: nodes, elements, reactions, and options
        nodes = inputs["platform_nodes"]
        nnode = np.where(nodes[:, 0] == NULL)[0][0]
        nodes = nodes[:nnode, :]
        rnode = np.zeros(nnode)  # inputs["platform_Rnode"][:nnode]
        Fnode = inputs["platform_Fnode"][:nnode, :]
        Mnode = np.zeros((nnode, 3))
        ihub = np.argmax(nodes[:, 2]) - 1
        itrans = util.closest_node(nodes, inputs["transition_node"])

        N1 = np.int_(inputs["platform_elem_n1"])
        nelem = np.where(N1 == NULL)[0][0]
        N1 = N1[:nelem]
        N2 = np.int_(inputs["platform_elem_n2"][:nelem])
        A = inputs["platform_elem_A"][:nelem]
        Asx = inputs["platform_elem_Asx"][:nelem]
        Asy = inputs["platform_elem_Asy"][:nelem]
        Ixx = inputs["platform_elem_Ixx"][:nelem]
        Iyy = inputs["platform_elem_Iyy"][:nelem]
        J0 = inputs["platform_elem_J0"][:nelem]
        rho = inputs["platform_elem_rho"][:nelem]
        E = inputs["platform_elem_E"][:nelem]
        G = inputs["platform_elem_G"][:nelem]
        roll = np.zeros(nelem)
        L = inputs["platform_elem_L"][:nelem]  # np.sqrt(np.sum((nodes[N2, :] - nodes[N1, :]) ** 2, axis=1))

        inodes = np.arange(nnode) + 1
        node_obj = pyframe3dd.NodeData(inodes, nodes[:, 0], nodes[:, 1], nodes[:, 2], rnode)

        ielem = np.arange(nelem) + 1
        elem_obj = pyframe3dd.ElementData(ielem, N1 + 1, N2 + 1, A, Asx, Asy, J0, Ixx, Iyy, E, G, roll, rho)

        # Use Mooring stiffness (TODO Hydro_K too)
        ind = []
        for k in range(n_attach):
            ind.append(util.closest_node(nodes, fairlead_joints[k, :]))
        rid = np.array([ind])  # np.array([np.argmin(nodes[:, 2])])

        Rx = Ry = Rz = Rxx = Ryy = Rzz = RIGID * np.ones(rid.size)
        # Rx, Ry, Rz = [mooringK[0]], [mooringK[1]], [mooringK[2]]
        # Only this solution works and there isn't much different with fully rigid
        # Rx, Ry, Rz = [RIGID], [RIGID], [mooringK[2]]
        # Rxx, Ryy, Rzz = [RIGID], [RIGID], [RIGID]
        react_obj = pyframe3dd.ReactionData(rid + 1, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)

        frame3dd_opt = opt["WISDEM"]["FloatingSE"]["frame3dd"]
        opt_obj = pyframe3dd.Options(frame3dd_opt["shear"], frame3dd_opt["geom"], -1.0)

        myframe = pyframe3dd.Frame(node_obj, react_obj, elem_obj, opt_obj)

        # Added mass
        cg_add = m_variable * cg_variable / (m_trans + m_variable)
        cg_add = cg_add.reshape((-1, 1))
        add_gravity = True
        mID = np.array([itrans], dtype=np.int_)
        m_add = np.array([m_trans + m_variable])
        I_add = I_trans.reshape((-1, 1))
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

        # Initialize loading with gravity, mooring line forces, and buoyancy (already in nodal forces)
        gx = gy = 0.0
        gz = -gravity
        load_obj = pyframe3dd.StaticLoadCase(gx, gy, gz)

        for k in range(n_attach):
            ind = util.closest_node(nodes, fairlead_joints[k, :])
            Fnode[ind, :] += mooringF[k, :]

        Fnode[ihub, :] += inputs["turbine_F"]
        Mnode[ihub, :] += inputs["turbine_M"]
        nF = np.where(np.abs(Fnode).sum(axis=1) > 0.0)[0]
        load_obj.changePointLoads(
            nF + 1, Fnode[nF, 0], Fnode[nF, 1], Fnode[nF, 2], Mnode[nF, 0], Mnode[nF, 1], Mnode[nF, 2]
        )

        # trapezoidally distributed loads
        xx1 = xy1 = xz1 = np.zeros(ielem.size)
        xx2 = xy2 = xz2 = 0.99 * L  # multiply slightly less than unity b.c. of precision
        wx1 = inputs["platform_elem_Px1"][:nelem]
        wx2 = inputs["platform_elem_Px2"][:nelem]
        wy1 = inputs["platform_elem_Py1"][:nelem]
        wy2 = inputs["platform_elem_Py2"][:nelem]
        wz1 = inputs["platform_elem_Pz1"][:nelem]
        wz2 = inputs["platform_elem_Pz2"][:nelem]
        load_obj.changeTrapezoidalLoads(ielem, xx1, xx2, wx1, wx2, xy1, xy2, wy1, wy2, xz1, xz2, wz1, wz2)

        # Add the load case and run
        myframe.addLoadCase(load_obj)
        # myframe.write("system.3dd")
        # myframe.draw()
        displacements, forces, reactions, internalForces, mass, modal = myframe.run()

        # Determine reaction forces
        outputs["platform_base_F"] = -np.array([reactions.Fx.sum(), reactions.Fy.sum(), reactions.Fz.sum()])
        outputs["platform_base_M"] = -np.array([reactions.Mxx.sum(), reactions.Myy.sum(), reactions.Mzz.sum()])

        # Forces and moments along the structure
        ic = 0  # case number
        outputs["platform_Fz"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_Vx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_Vy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_Mxx"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_Myy"] = NULL * np.ones(NELEM_MAX)
        outputs["platform_Mzz"] = NULL * np.ones(NELEM_MAX)

        outputs["platform_Fz"][:nelem] = forces.Nx[ic, 1::2]
        outputs["platform_Vx"][:nelem] = -forces.Vz[ic, 1::2]
        outputs["platform_Vy"][:nelem] = forces.Vy[ic, 1::2]
        outputs["platform_Mxx"][:nelem] = -forces.Mzz[ic, 1::2]
        outputs["platform_Myy"][:nelem] = forces.Myy[ic, 1::2]
        outputs["platform_Mzz"][:nelem] = forces.Txx[ic, 1::2]


class FloatingPost(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("options")

    def setup(self):
        self.add_input("platform_elem_L", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_D", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_t", NULL * np.ones(NELEM_MAX), units="m")
        self.add_input("platform_elem_A", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asx", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Asy", NULL * np.ones(NELEM_MAX), units="m**2")
        self.add_input("platform_elem_Ixx", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_Iyy", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_J0", NULL * np.ones(NELEM_MAX), units="kg*m**2")
        self.add_input("platform_elem_E", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_G", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_sigma_y", NULL * np.ones(NELEM_MAX), units="Pa")
        self.add_input("platform_elem_qdyn", NULL * np.ones(NELEM_MAX), units="Pa")

        # Processed Frame3DD/OpenFAST outputs
        self.add_input("platform_Fz", NULL * np.ones(NELEM_MAX), units="N")
        self.add_input("platform_Vx", NULL * np.ones(NELEM_MAX), units="N")
        self.add_input("platform_Vy", NULL * np.ones(NELEM_MAX), units="N")
        self.add_input("platform_Mxx", NULL * np.ones(NELEM_MAX), units="N*m")
        self.add_input("platform_Myy", NULL * np.ones(NELEM_MAX), units="N*m")
        self.add_input("platform_Mzz", NULL * np.ones(NELEM_MAX), units="N*m")

        self.add_output("constr_platform_stress", NULL * np.ones(NELEM_MAX))
        self.add_output("constr_platform_shell_buckling", NULL * np.ones(NELEM_MAX))
        self.add_output("constr_platform_global_buckling", NULL * np.ones(NELEM_MAX))

    def compute(self, inputs, outputs):
        # Unpack some variables
        d = inputs["platform_elem_D"]
        nelem = np.where(d == NULL)[0][0]
        d = d[:nelem]
        t = inputs["platform_elem_t"][:nelem]
        h = inputs["platform_elem_L"][:nelem]
        Az = inputs["platform_elem_A"][:nelem]
        Asx = inputs["platform_elem_Asx"][:nelem]
        Jz = inputs["platform_elem_J0"][:nelem]
        Iyy = inputs["platform_elem_Iyy"][:nelem]
        sigy = inputs["platform_elem_sigma_y"][:nelem]
        E = inputs["platform_elem_E"][:nelem]
        G = inputs["platform_elem_G"][:nelem]
        qdyn = inputs["platform_elem_qdyn"][:nelem]
        r = 0.5 * d

        gamma_f = self.options["options"]["gamma_f"]
        gamma_m = self.options["options"]["gamma_m"]
        gamma_n = self.options["options"]["gamma_n"]
        gamma_b = self.options["options"]["gamma_b"]

        # Get loads from Framee3dd/OpenFAST
        Fz = inputs["platform_Fz"][:nelem]
        Vx = inputs["platform_Vx"][:nelem]
        Vy = inputs["platform_Vy"][:nelem]
        Mxx = inputs["platform_Mxx"][:nelem]
        Myy = inputs["platform_Myy"][:nelem]
        Mzz = inputs["platform_Mzz"][:nelem]

        M = np.sqrt(Mxx ** 2 + Myy ** 2)
        V = np.sqrt(Vx ** 2 + Vy ** 2)

        # Initialize outputs
        outputs["constr_platform_stress"] = NULL * np.ones(NELEM_MAX)
        outputs["constr_platform_shell_buckling"] = NULL * np.ones(NELEM_MAX)
        outputs["constr_platform_global_buckling"] = NULL * np.ones(NELEM_MAX)

        # See http://svn.code.sourceforge.net/p/frame3dd/code/trunk/doc/Frame3DD-manual.html#structuralmodeling
        # print(Fz.shape, Az.shape, M.shape, r.shape, Iyy.shape)
        axial_stress = Fz / Az + M * r / Iyy
        shear_stress = np.abs(Mzz) / Jz * r + V / Asx
        hoop_stress = util_con.hoopStress(d, t, qdyn)
        outputs["constr_platform_stress"][:nelem] = util_con.vonMisesStressUtilization(
            axial_stress, hoop_stress, shear_stress, gamma_f * gamma_m * gamma_n, sigy
        )

        # Use DNV-GL CP202 Method
        check = util_dnvgl.CylinderBuckling(h, d, t, E=E, G=G, sigma_y=sigy, gamma=gamma_f * gamma_b)
        results = check.run_buckling_checks(Fz, M, axial_stress, hoop_stress, shear_stress)

        outputs["constr_platform_shell_buckling"][:nelem] = results["Shell"]
        outputs["constr_platform_global_buckling"][:nelem] = results["Global"]


class MaxTurbineLoads(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("nLC")

    def setup(self):
        nLC = self.options["nLC"]
        for iLC in range(nLC):
            lc = "" if nLC == 1 else str(iLC + 1)

            self.add_input(f"lc{lc}:turbine_F", np.zeros(3), units="N")
            self.add_input(f"lc{lc}:turbine_M", np.zeros(3), units="N*m")

        self.add_output("max_F", np.zeros(3), units="N")
        self.add_output("max_M", np.zeros(3), units="N*m")

    def compute(self, inputs, outputs):
        Fmax = np.zeros(3)
        Mmax = np.zeros(3)

        nLC = self.options["nLC"]
        for iLC in range(nLC):
            lc = "" if nLC == 1 else str(iLC + 1)

            Fmax = np.fmax(Fmax, inputs[f"lc{lc}:turbine_F"])
            Mmax = np.fmax(Mmax, inputs[f"lc{lc}:turbine_M"])

        outputs["max_F"] = Fmax
        outputs["max_M"] = Mmax


class FloatingFrame(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]
        nLC = opt["WISDEM"]["n_dlc"]
        n_member = opt["floating"]["members"]["n_members"]

        mem_prom = [
            "wind_reference_height",
            "z0",
            "shearExp",
            "cd_usr",
            "cm",
            "rho_air",
            "rho_water",
            "mu_air",
            "mu_water",
            "beta_wind",
            "beta_wave",
            "Uc",
            "Hsig_wave",
            "Tsig_wave",
            "water_depth",
            "yaw",
        ]

        U_prom = []
        for iLC in range(nLC):
            lc = "" if nLC == 1 else str(iLC + 1)
            U_prom.append(f"env{lc}.Uref")

        plat_prom = [
            "platform_elem_L",
            "platform_elem_D",
            "platform_elem_t",
            "platform_elem_A",
            "platform_elem_Asx",
            "platform_elem_Asy",
            "platform_elem_Ixx",
            "platform_elem_Iyy",
            "platform_elem_J0",
            "platform_elem_E",
            "platform_elem_G",
            "platform_elem_sigma_y",
        ]

        plat_frame = plat_prom[:-1] + [
            "platform_elem_rho",
            "platform_mass",
            "platform_hull_center_of_mass",
            "platform_added_mass",
            "platform_nodes",
            "platform_Fnode",
            "platform_Rnode",
            "platform_elem_n1",
            "platform_elem_n2",
            "transition_node",
            "transition_piece_mass",
            "transition_piece_I",
            "mooring_neutral_load",
            "mooring_fairlead_joints",
            "mooring_stiffness",
            "variable_ballast_mass",
            "variable_center_of_mass",
        ]

        mem_vars = ["Px", "Py", "Pz", "qdyn"]
        mem_vars12 = ["Px1", "Px2", "Py1", "Py2", "Pz1", "Pz2"]
        plat_vars = ["platform_Fz", "platform_Vx", "platform_Vy", "platform_Mxx", "platform_Myy", "platform_Mzz"]

        self.add_subsystem("maxturb", MaxTurbineLoads(nLC=nLC), promotes=["*"])

        for k in range(n_member):
            n_full = get_nfull(opt["floating"]["members"]["n_height"][k])
            self.add_subsystem(
                f"memload{k}",
                MemberLoads(
                    n_full=n_full,
                    n_lc=nLC,
                    hydro=True,
                    memmax=True,
                ),
                promotes=mem_prom + U_prom + [("joint1", f"member{k}:joint1"), ("joint2", f"member{k}:joint2")],
            )

        for iLC in range(nLC):
            lc = "" if nLC == 1 else str(iLC + 1)

            self.add_subsystem(f"loadsys{lc}", PlatformLoads(options=opt))

            self.add_subsystem(
                f"frame{lc}",
                FrameAnalysis(options=opt),
                promotes=plat_frame + [("turbine_F", "lc{lc}:turbine_F"), ("turbine_M", "lc{lc}:turbine_M")],
            )
            self.add_subsystem(f"post{lc}", FloatingPost(options=opt["WISDEM"]["FloatingSE"]), promotes=plat_prom)

            for k in range(n_member):
                for var in mem_vars:
                    self.connect(f"memload{k}.g2e{lc}.{var}", f"loadsys{lc}.member{k}:{var}")

            for var in mem_vars12:
                self.connect(f"loadsys{lc}.platform_elem_{var}", f"frame{lc}.platform_elem_{var}")
            self.connect(f"loadsys{lc}.platform_elem_qdyn", f"post{lc}.platform_elem_qdyn")

            for var in plat_vars:
                self.connect(f"frame{lc}.{var}", f"post{lc}.{var}")
