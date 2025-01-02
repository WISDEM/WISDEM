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
        self.add_input("platform_Iwaterx", 0.0, units="m**4")
        self.add_input("platform_Iwatery", 0.0, units="m**4")
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
        self.add_output("metacentric_height_roll", 0.0, units="m")
        self.add_output("metacentric_height_pitch", 0.0, units="m")

    def compute(self, inputs, outputs):
        opt = self.options["modeling_options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Unpack inputs
        Hsig = float(inputs["Hsig_wave"][0])
        fairlead = np.abs(float(inputs["fairlead"][0]))
        R_fairlead = float(inputs["fairlead_radius"][0])
        max_heel = float(inputs["survival_heel"][0])
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
            xyz1 = xyz[0, :]  # Should be the draft
            xyz2 = xyz[-1, :] # Should be the freeboard

            # Get xp-zp coplanar coordinates relative to cg
            xp1 = np.sqrt(np.sum((xyz1[:2] - cg[:2]) ** 2))
            zp1 = xyz1[2] - cg[2]
            xp2 = np.sqrt(np.sum((xyz2[:2] - cg[:2]) ** 2))
            zp2 = xyz2[2] - cg[2]

            # Only check this for partially submerged members
            if xyz1[2] * xyz2[2] > 0:  # pos * neg
                continue

            # Simplify by making zp1 above zp2
            if zp2 < zp1: # Our assumption that z1 is draft is incorrect
                # Swap variables
                zp1, zp2 = zp2, zp1
                xp1, xp2 = xp2, xp1
                xyz1, xyz2 = xyz2, xyz1

            # Coordinate transformation about CG and change in z-position
            _, zp1_h = util.rotate(0.0, 0.0, xp1, zp1, max_heel)      # Bottom point, we care about it going up
            _, zp2_h = util.rotate(0.0, 0.0, xp2, zp2, -max_heel)     # Top point, we care about it going down
            
            dz1 = zp1_h - zp1       # change in keel/draft
            dz2 = zp2   - zp2_h     # change in freeboard

            draft_margin[k] = -dz1 / xyz1[2]  # Adding negative because assume xyz1[2] is negative below water
            freeboard_margin[k] = dz2 / xyz2[2]

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
        Iwaterx_total = inputs["platform_Iwaterx"]
        Iwatery_total = inputs["platform_Iwatery"]
        V_platform = inputs["platform_displacement"]
        z_cb = inputs["platform_center_of_buoyancy"][2]
        buoyancy2metacentre_BM_roll = Iwaterx_total / V_platform
        buoyancy2metacentre_BM_pitch = Iwatery_total / V_platform
        outputs["metacentric_height_roll"] = buoyancy2metacentre_BM_roll - (cg[2] - z_cb)
        outputs["metacentric_height_pitch"] = buoyancy2metacentre_BM_pitch - (cg[2] - z_cb)

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
        self.add_input("metacentric_height_roll", 0.0, units="m")
        self.add_input("metacentric_height_pitch", 0.0, units="m")
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
        rhoWater = float(inputs["rho_water"][0])
        m_system = float(inputs["system_mass"][0])
        I_system = inputs["system_I"]
        z_cg = float(inputs["system_center_of_mass"][-1])
        A_mat = inputs["platform_added_mass"]
        V_system = float(inputs["platform_displacement"][0])
        h_metacentric_roll = float(inputs["metacentric_height_roll"][0])
        h_metacentric_pitch = float(inputs["metacentric_height_pitch"][0])
        Awater = float(inputs["platform_Awater"][0])
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
        # Roll
        K_hydro[3] = rhoWater * gravity * V_system * h_metacentric_roll
        # Pitch
        K_hydro[4] = rhoWater * gravity * V_system * h_metacentric_pitch
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
