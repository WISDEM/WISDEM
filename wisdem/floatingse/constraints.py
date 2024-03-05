import numpy as np
import openmdao.api as om

import wisdem.commonse.utilities as util
from wisdem.commonse import eps, gravity
from wisdem.commonse.cylinder_member import NULL, MEMMAX


class FloatingConstraints(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]
        n_dlc = opt["WISDEM"]["n_dlc"]
        n_member = opt["floating"]["members"]["n_members"]

        self.add_input("Hsig_wave", 0.0, units="m")
        self.add_input("variable_ballast_mass", 0.0, units="kg")
        self.add_input("fairlead_radius", 0.0, units="m")
        self.add_input("fairlead", 0.0, units="m")
        self.add_input("survival_heel", 0.0, units="rad")
        tot_ball = 0
        for k in range(n_member):
            n_ball = opt["floating"]["members"]["n_ballasts"][k]
            self.add_input(f"member{k}:nodes_xyz", NULL * np.ones((MEMMAX, 3)), units="m")
            self.add_input(f"member{k}:constr_ballast_capacity", np.zeros(n_ball))
            tot_ball += n_ball
        self.add_input("platform_Iwater", 0.0, units="m**4")
        self.add_input("platform_displacement", 0.0, units="m**3")
        self.add_input("platform_center_of_buoyancy", np.zeros(3), units="m")
        self.add_input("system_center_of_mass", np.zeros(3), units="m")
        self.add_input("transition_node", np.zeros(3), units="m")

        self.add_input("turbine_F", np.zeros((3, n_dlc)), units="N")
        self.add_input("turbine_M", np.zeros((3, n_dlc)), units="N*m")
        self.add_input("max_surge_restoring_force", 0.0, units="N")
        self.add_input("operational_heel_restoring_force", np.zeros(6), units="N")
        self.add_input("survival_heel_restoring_force", np.zeros(6), units="N")

        self.add_output("constr_freeboard_heel_margin", np.zeros(n_member))
        self.add_output("constr_draft_heel_margin", np.zeros(n_member))
        self.add_output("constr_fixed_margin", np.zeros(tot_ball))
        self.add_output("constr_fairlead_wave", 0.0)
        self.add_output("constr_mooring_surge", 0.0)
        self.add_output("constr_mooring_heel", 0.0)
        self.add_output("metacentric_height", 0.0, units="m")

    def compute(self, inputs, outputs):
        opt = self.options["modeling_options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Unpack inputs
        Hsig = float(inputs["Hsig_wave"])
        fairlead = np.abs(float(inputs["fairlead"]))
        R_fairlead = float(inputs["fairlead_radius"])
        max_heel = float(inputs["survival_heel"])
        cg = inputs["system_center_of_mass"]
        gamma = 1.1

        # Make sure there is sufficient margin on offset members
        # where freeboard does not get submerged and keel does not come out water
        freeboard_margin = np.zeros(n_member)
        draft_margin = np.zeros(n_member)
        ballast_margin = []
        for k in range(n_member):
            ballast_margin.extend(inputs[f"member{k}:constr_ballast_capacity"].tolist())

            xyz = inputs[f"member{k}:nodes_xyz"]
            inodes = np.where(xyz[:, 0] == NULL)[0][0]
            xyz = xyz[:inodes, :]
            xyz1 = xyz[0, :]
            xyz2 = xyz[-1, :]

            # Get xp-zp coplanar coordinates relative to cg
            xp1 = np.sqrt(np.sum((xyz1[:2] - cg[:2]) ** 2))
            zp1 = xyz1[2] - cg[2]
            xp2 = np.sqrt(np.sum((xyz2[:2] - cg[:2]) ** 2))
            zp2 = xyz2[2] - cg[2]

            # Coordinate transformation about CG and change in z-position
            xp1_h, zp1_h = util.rotate(0.0, 0.0, xp1, zp1, max_heel)
            xp2_h, zp2_h = util.rotate(0.0, 0.0, xp2, zp2, max_heel)
            dz1 = np.abs((zp1 - zp1_h) / xyz1[2])
            dz2 = np.abs((zp2 - zp2_h) / xyz2[2])

            # See if change in z-coordinate is bigger than freeboard or draft, should be <1
            if xyz1[2] > 0.0 and xyz1[2] > xyz2[2]:
                freeboard_margin[k] = dz1
            elif xyz2[2] > 0.0 and xyz2[2] > xyz1[2]:
                freeboard_margin[k] = dz2

            if xyz1[2] < 0.0 and xyz1[2] < xyz2[2]:
                draft_margin[k] = dz1
            elif xyz2[2] < 0.0 and xyz2[2] < xyz1[2]:
                draft_margin[k] = dz2

        # Ensure members have enough clearance from the waterline
        outputs["constr_freeboard_heel_margin"] = freeboard_margin
        outputs["constr_draft_heel_margin"] = draft_margin
        outputs["constr_fixed_margin"] = np.array(ballast_margin)

        # Make sure the fairlead depth is greater than the wave height with margin.  Should be <1
        outputs["constr_fairlead_wave"] = Hsig * gamma / fairlead

        # Compute the distance from the center of buoyancy to the metacentre (BM is naval architecture)
        # BM = Iw / V where V is the displacement volume (just computed)
        # Iw is the area moment of inertia (meters^4) of the water-plane cross section about the heel axis
        # For a spar, we assume this is just the I of a circle about x or y
        # See https://en.wikipedia.org/wiki/Metacentric_height
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        # and http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node30.html
        # Measure static stability:
        # 1. Center of buoyancy should be above CG (difference should be positive)
        # 2. Metacentric height should be positive
        Iwater_total = inputs["platform_Iwater"]
        V_platform = inputs["platform_displacement"]
        z_cb = inputs["platform_center_of_buoyancy"][2]
        buoyancy2metacentre_BM = Iwater_total / V_platform
        outputs["metacentric_height"] = buoyancy2metacentre_BM - (cg[2] - z_cb)

        # Mooring strength checks
        F_turb = inputs["turbine_F"].max(axis=1)
        M_turb = inputs["turbine_M"].max(axis=1)
        surge_restore = inputs["max_surge_restoring_force"]
        outputs["constr_mooring_surge"] = surge_restore - F_turb[0]
        heel_restore = inputs["operational_heel_restoring_force"]
        # (fairlead is assumed negative, made positive above, would otherwise be cg-fairlead)
        M_heel_restore = R_fairlead * heel_restore[2] + (cg[2] + fairlead) * heel_restore[:2].sum() + heel_restore[4]
        trans2cg = inputs["transition_node"][2] - cg[2]
        outputs["constr_mooring_heel"] = M_heel_restore - F_turb[0] * trans2cg - M_turb[1]


class RigidModes(om.ExplicitComponent):
    def setup(self):
        self.add_input("platform_Awater", 0.0, units="m**2")
        self.add_input("platform_added_mass", np.zeros(6), units="kg")
        self.add_input("system_mass", 0.0, units="kg")
        self.add_input("system_center_of_mass", np.zeros(3), units="m")
        self.add_input("system_I", np.zeros(6), units="kg*m**2")
        self.add_input("rho_water", 0.0, units="kg/m**3")
        self.add_input("platform_displacement", 0.0, units="m**3")
        self.add_input("metacentric_height", 0.0, units="m")
        self.add_input("mooring_stiffness", np.zeros((6, 6)), units="N/m")

        self.add_output(
            "hydrostatic_stiffness", np.zeros(6), units="N/m", desc="Summary hydrostatic stiffness of structure"
        )
        self.add_output("rigid_body_periods", np.zeros(6), units="s", desc="Natural periods of oscillation in 6 DOF")
        self.add_output("surge_period", 0.0, units="s", desc="Surge period of oscillation")
        self.add_output("sway_period", 0.0, units="s", desc="Sway period of oscillation")
        self.add_output("heave_period", 0.0, units="s", desc="Heave period of oscillation")
        self.add_output("roll_period", 0.0, units="s", desc="Roll period of oscillation")
        self.add_output("pitch_period", 0.0, units="s", desc="Pitch period of oscillation")
        self.add_output("yaw_period", 0.0, units="s", desc="Yaw period of oscillation")

    def compute(self, inputs, outputs):
        # Number of degrees of freedom
        nDOF = 6

        # Unpack variables
        rhoWater = inputs["rho_water"]
        m_system = inputs["system_mass"]
        I_system = inputs["system_I"]
        z_cg = inputs["system_center_of_mass"][-1]
        A_mat = inputs["platform_added_mass"]
        V_system = inputs["platform_displacement"]
        h_metacenter = inputs["metacentric_height"]
        Awater = inputs["platform_Awater"]
        K_moor = np.diag(inputs["mooring_stiffness"])

        # Compute elements on mass matrix diagonal
        # Surge, sway, heave just use normal inertia (without mooring according to Senu)
        # Roll, pitch, yaw use system moment of inertia
        M_mat = np.r_[m_system * np.ones(3), I_system[:3]]

        # Structural stiffness
        K_struct = np.zeros(nDOF)
        K_struct[3:5] = np.abs(m_system * gravity * z_cg)

        # Hydrostatic stiffness has contributions in heave (K33) and roll/pitch (K44/55)
        # See DNV-RP-H103: Modeling and Analyis of Marine Operations
        K_hydro = np.zeros(nDOF)
        K_hydro[2] = rhoWater * gravity * Awater
        K_hydro[3:5] = rhoWater * gravity * V_system * h_metacenter
        outputs["hydrostatic_stiffness"] = K_hydro

        # Now compute all six natural periods at once
        K_total = np.maximum(K_struct + K_hydro + K_moor, 0.0)
        outputs["rigid_body_periods"] = 2 * np.pi * np.sqrt((M_mat + A_mat) / (K_total + eps))
        outputs["surge_period"] = outputs["rigid_body_periods"][0]
        outputs["sway_period"] = outputs["rigid_body_periods"][1]
        outputs["heave_period"] = outputs["rigid_body_periods"][2]
        outputs["roll_period"] = outputs["rigid_body_periods"][3]
        outputs["pitch_period"] = outputs["rigid_body_periods"][4]
        outputs["yaw_period"] = outputs["rigid_body_periods"][5]
