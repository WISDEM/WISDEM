import openmdao.api as om
from wisdem.floatingse.member import Member
from wisdem.floatingse.constraints import FloatingConstraints
from wisdem.floatingse.map_mooring import MapMooring
from wisdem.floatingse.floating_frame import FloatingFrame


class FloatingSE(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]

        # self.set_input_defaults("mooring_type", "chain")
        # self.set_input_defaults("anchor_type", "SUCTIONPILE")
        # self.set_input_defaults("loading", "hydrostatic")
        # self.set_input_defaults("wave_period_range_low", 2.0, units="s")
        # self.set_input_defaults("wave_period_range_high", 20.0, units="s")
        # self.set_input_defaults("cd_usr", -1.0)
        # self.set_input_defaults("zref", 100.0)
        # self.set_input_defaults("number_of_offset_columns", 0)
        # self.set_input_defaults("material_names", ["steel"])

        n_member = opt["floating"]["members"]["n_members"]
        mem_prom = [
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
        mem_prom += [
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
        for k in range(n_member):
            self.add_subsystem(
                f"member{k}",
                Member(column_options=opt["floating"]["members"], idx=k, n_mat=opt["materials"]["n_mat"]),
                promotes=mem_prom,
            )

        # Next run MapMooring
        self.add_subsystem(
            "mm", MapMooring(options=opt["mooring"], gamma=opt["WISDEM"]["FloatingSE"]["gamma_f"]), promotes=["*"]
        )

        # Add in the connecting truss
        self.add_subsystem("load", FloatingFrame(modeling_options=opt), promotes=["*"])

        # Evaluate system constraints
        self.add_subsystem("cons", FloatingConstraints(modeling_options=opt), promotes=["*"])

        # Connect all input variables from all models
        mem_vars = [
            "nodes_xyz",
            "nodes_r",
            "section_D",
            "section_t",
            "section_A",
            "section_Asx",
            "section_Asy",
            "section_Ixx",
            "section_Iyy",
            "section_Izz",
            "section_rho",
            "section_E",
            "section_G",
            "idx_cb",
            "variable_ballast_capacity",
            "variable_ballast_Vpts",
            "variable_ballast_spts",
            "constr_ballast_capacity",
            "buoyancy_force",
            "displacement",
            "center_of_buoyancy",
            "center_of_mass",
            "total_mass",
            "total_cost",
            "I_total",
            "Awater",
            "Iwater",
            "added_mass",
            "waterline_centroid",
        ]
        for k in range(n_member):
            for var in mem_vars:
                self.connect(f"member{k}." + var, f"member{k}:" + var)

        """
        self.connect("max_offset_restoring_force", "mooring_surge_restoring_force")
        self.connect("operational_heel_restoring_force", "mooring_pitch_restoring_force")
        """
