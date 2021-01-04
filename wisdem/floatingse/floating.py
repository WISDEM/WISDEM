import openmdao.api as om
from wisdem.towerse.tower import TowerLeanSE
from wisdem.floatingse.member import Member
from wisdem.floatingse.map_mooring import MapMooring
from wisdem.floatingse.floating_frame import FloatingFrame

# from wisdem.floatingse.substructure import Substructure, SubstructureGeometry


class FloatingSE(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]["platform"]

        self.set_input_defaults("mooring_type", "chain")
        self.set_input_defaults("anchor_type", "SUCTIONPILE")
        # self.set_input_defaults("loading", "hydrostatic")
        # self.set_input_defaults("wave_period_range_low", 2.0, units="s")
        # self.set_input_defaults("wave_period_range_high", 20.0, units="s")
        # self.set_input_defaults("cd_usr", -1.0)
        # self.set_input_defaults("zref", 100.0)
        # self.set_input_defaults("number_of_offset_columns", 0)
        # self.set_input_defaults("material_names", ["steel"])

        self.add_subsystem(
            "tow",
            TowerLeanSE(modeling_options=self.options["modeling_options"]),
            promotes=[
                "rna_mass",
                "rna_cg",
                "rna_I",
                "hub_height",
                "material_names",
                "labor_cost_rate",
                "painting_cost_rate",
                "unit_cost_mat",
                "rho_mat",
                "E_mat",
                "G_mat",
                "sigma_y_mat",
            ],
        )

        n_member = opt["floating"]["n_member"]
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
        # mem_prom += ["Uref", "zref", "shearExp", "z0", "cd_usr", "cm", "beta_wind", "rho_air", "mu_air", "beta_water",
        #            "rho_water", "mu_water", "Uc", "Hsig_wave","Tsig_wave","rho_water","water_depth"]
        for k in range(n_member):
            self.add_subsystem(
                "member" + str(k),
                Member(modeling_options=opt, column_options=opt["floating"]["member"][k]),
                promotes=mem_prom,
            )

        # Next run MapMooring
        self.add_subsystem("mm", MapMooring(modeling_options=opt), promotes=["*"])

        # Add in the connecting truss
        self.add_subsystem("load", FloatingFrame(modeling_options=opt), promotes=["*"])

        # Evaluate system constraints
        # self.add_subsystem("cons", FloatingConstraints(modeling_options=opt), promotes=["*"])

        # Connect all input variables from all models
        mem_vars = [
            "nodes_xyz",
            "nodes_r",
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
            "buoyancy_force",
            "displacement",
            "center_of_buoyancy",
            "center_of_mass",
            "total_mass",
            "total_cost",
            "Awater",
            "Iwater",
            "added_mass",
        ]
        for k in range(n_member):
            for var in mem_vars:
                self.connect("member" + str(k) + "." + var, "load.member" + str(k) + ":" + var)

        """
        self.connect("tow.d_full", ["windLoads.d", "tower_d_full"])
        self.connect("tow.d_full", "tower_d_base", src_indices=[0])
        self.connect("tow.t_full", "tower_t_full")
        self.connect("tow.z_full", ["loadingWind.z", "windLoads.z", "tower_z_full"])  # includes tower_z_full
        self.connect("tow.E_full", "tower_E_full")
        self.connect("tow.G_full", "tower_G_full")
        self.connect("tow.rho_full", "tower_rho_full")
        self.connect("tow.sigma_y_full", "tower_sigma_y_full")
        self.connect("tow.cm.mass", "tower_mass_section")
        self.connect("tow.turbine_mass", "main.stack_mass_in")
        self.connect("tow.tower_center_of_mass", "tower_center_of_mass")
        self.connect("tow.tower_cost", "tower_shell_cost")

        self.connect("max_offset_restoring_force", "mooring_surge_restoring_force")
        self.connect("operational_heel_restoring_force", "mooring_pitch_restoring_force")
        """
