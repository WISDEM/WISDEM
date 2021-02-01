import numpy as np
import openmdao.api as om
import wisdem.commonse.utilities as util
from wisdem.floatingse.member import NULL, MEMMAX


class FloatingConstraints(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]
        n_member = opt["floating"]["members"]["n_members"]
        n_lines = opt["mooring"]["n_anchors"]

        self.add_input("Hsig_wave", val=0.0, units="m")
        self.add_input("variable_ballast_mass", val=0.0, units="kg")
        self.add_input("fairlead_radius", 0.0, units="m")
        self.add_input("fairlead", 0.0, units="m")
        self.add_input("operational_heel", 0.0, units="deg")
        self.add_input("survival_heel", 0.0, units="deg")
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
        self.add_input("tower_top_node", np.zeros(3), units="m")

        self.add_input("rna_F", np.zeros(3), units="N")
        self.add_input("rna_M", np.zeros(3), units="N*m")
        self.add_input("max_surge_restoring_force", 0.0, units="N")
        self.add_input("operational_heel_restoring_force", np.zeros((n_lines, 3)), units="N")
        self.add_input("survival_heel_restoring_force", np.zeros((n_lines, 3)), units="N")

        self.add_output("constr_freeboard_heel_margin", val=np.zeros(n_member))
        self.add_output("constr_draft_heel_margin", val=np.zeros(n_member))
        self.add_output("constr_fixed_margin", val=np.zeros(tot_ball))
        self.add_output("constr_fairlead_wave", val=0.0)
        self.add_output("constr_mooring_surge", val=0.0)
        self.add_output("constr_mooring_heel", val=0.0)
        self.add_output("metacentric_height", val=0.0, units="m")

    def compute(self, inputs, outputs):
        opt = self.options["modeling_options"]
        n_member = opt["floating"]["members"]["n_members"]

        # Unpack inputs
        Hsig = inputs["Hsig_wave"]
        fairlead = np.abs(inputs["fairlead"])
        R_fairlead = inputs["fairlead_radius"]
        max_heel = np.deg2rad(inputs["survival_heel"])
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
            dz1 = np.abs(xyz1[2] / (zp1 - zp1_h))
            dz2 = np.abs(xyz2[2] / (zp2 - zp2_h))

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
        F_rna = inputs["rna_F"]
        M_rna = inputs["rna_M"]
        surge_restore = inputs["max_surge_restoring_force"]
        heel_restore = inputs["operational_heel_restoring_force"].sum(axis=0)
        outputs["constr_mooring_surge"] = surge_restore - F_rna[0]
        # (fairlead is assumed negative, made positive above, would otherwise be cg-fairlead)
        M_heel_restore = R_fairlead * heel_restore[2] + (cg[2] + fairlead) * heel_restore[:2].sum()
        tt2cg = inputs["tower_top_node"][2] - cg[2]
        outputs["constr_mooring_heel"] = M_heel_restore - F_rna[0] * tt2cg - M_rna[1]
