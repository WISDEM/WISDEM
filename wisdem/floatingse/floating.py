import openmdao.api as om
from wisdem.floatingse.mooring import Mooring
from wisdem.floatingse.constraints import FloatingConstraints
from wisdem.commonse.cylinder_member import MemberDetailed
from wisdem.floatingse.floating_frame import FloatingFrame
from wisdem.floatingse.floating_system import FloatingSystem


class FloatingSE(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]

        n_member = opt["floating"]["members"]["n_members"]
        mem_prom = [
            "E_mat",
            "G_mat",
            "sigma_y_mat",
            "sigma_ult_mat",
            "wohler_exp_mat",
            "wohler_A_mat",
            "rho_mat",
            "rho_water",
            "unit_cost_mat",
            "material_names",
            "painting_cost_rate",
            "labor_cost_rate",
        ]
        for k in range(n_member):
            self.add_subsystem(
                f"member{k}",
                MemberDetailed(
                    column_options=opt["floating"]["members"],
                    idx=k,
                    n_mat=opt["materials"]["n_mat"],
                    memmax=True,
                    n_refine=2,
                ),
                promotes=mem_prom + [("joint1", f"member{k}:joint1"), ("joint2", f"member{k}:joint2")],
            )

        # Combine all members and tower into single system
        self.add_subsystem("sys", FloatingSystem(modeling_options=opt), promotes=["*"])

        self.add_subsystem(
            "mm", Mooring(options=opt["mooring"], gamma=opt["WISDEM"]["FloatingSE"]["gamma_f"]), promotes=["*"]
        )

        # Do the load analysis over one or more load cases
        self.add_subsystem("load", FloatingFrame(modeling_options=opt), promotes=["*"])

        # Evaluate system constraints
        self.add_subsystem("cons", FloatingConstraints(modeling_options=opt), promotes=["*"])

        # Connect all input variables from all models
        mem_vars = [
            "section_D",
            "section_t",
            "section_A",
            "section_Asx",
            "section_Asy",
            "section_Ixx",
            "section_Iyy",
            "section_J0",
            "section_rho",
            "section_E",
            "section_G",
            "section_sigma_y",
            "idx_cb",
            "variable_ballast_capacity",
            "variable_ballast_Vpts",
            "variable_ballast_spts",
            "constr_ballast_capacity",
            "buoyancy_force",
            "displacement",
            "center_of_buoyancy",
            "center_of_mass",
            "ballast_mass",
            "total_mass",
            "total_cost",
            "I_total",
            "Awater",
            "Iwater",
            "added_mass",
            "waterline_centroid",
        ]
        mem_load_vars = ["z_full", "d_full", "s_full", "s_all"]

        for k in range(n_member):
            for var in mem_vars:
                self.connect(f"member{k}.{var}", f"member{k}:{var}")

            self.connect(f"member{k}.nodes_xyz_all", f"member{k}:nodes_xyz")
            self.connect(f"member{k}.nodes_r_all", f"member{k}:nodes_r")

            for var in mem_load_vars:
                self.connect(f"member{k}.{var}", f"memload{k}.{var}")
