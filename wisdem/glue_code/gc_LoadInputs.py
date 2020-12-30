import numpy as np
import wisdem.inputs as sch


class WindTurbineOntologyPython(object):
    # Pure python class to load the input yaml file and break into few sub-dictionaries, namely:
    #   - modeling_options: dictionary with all the inputs that will be passed as options to the openmdao components, such as the length of the arrays
    #   - blade: dictionary representing the entry blade in the yaml file
    #   - tower: dictionary representing the entry tower in the yaml file
    #   - nacelle: dictionary representing the entry nacelle in the yaml file
    #   - materials: dictionary representing the entry materials in the yaml file
    #   - airfoils: dictionary representing the entry airfoils in the yaml file

    def __init__(self, fname_input_wt, fname_input_modeling, fname_input_analysis):

        self.modeling_options = sch.load_modeling_yaml(fname_input_modeling)
        self.analysis_options = sch.load_analysis_yaml(fname_input_analysis)
        if fname_input_wt is None:
            self.wt_init = None
        else:
            self.wt_init = sch.load_geometry_yaml(fname_input_wt)
        self.set_run_flags()
        self.set_openmdao_vectors()
        self.set_opt_flags()

    def get_input_data(self):
        return self.wt_init, self.modeling_options, self.analysis_options

    def set_run_flags(self):
        # Create components flag struct
        self.modeling_options["flags"] = {}

        for k in ["blade", "hub", "nacelle", "tower", "monopile", "floating_platform", "mooring", "RNA"]:
            self.modeling_options["flags"][k] = k in self.wt_init["components"]

        for k in ["assembly", "components", "airfoils", "materials", "control", "environment", "bos", "costs"]:
            self.modeling_options["flags"][k] = k in self.wt_init

        # Generator flag
        self.modeling_options["flags"]["generator"] = False
        if self.modeling_options["flags"]["nacelle"] and "generator" in self.wt_init["components"]["nacelle"]:
            self.modeling_options["flags"]["generator"] = True
            if not "GeneratorSE" in self.modeling_options:
                self.modeling_options["GeneratorSE"] = {}
            self.modeling_options["GeneratorSE"]["type"] = self.wt_init["components"]["nacelle"]["generator"][
                "generator_type"
            ].lower()

        # Offshore flags
        self.modeling_options["flags"]["floating"] = self.modeling_options["flags"]["floating_platform"]
        self.modeling_options["flags"]["offshore"] = (
            self.modeling_options["flags"]["floating"] or self.modeling_options["flags"]["monopile"]
        )

        # Put in some logic about what needs to be in there
        flags = self.modeling_options["flags"]

        # Even if the block is in the inputs, the user can turn off via modeling options
        if flags["bos"]:
            flags["bos"] = self.modeling_options["BOS"]["flag"]
        if flags["blade"]:
            flags["blade"] = self.modeling_options["RotorSE"]["flag"]
        if flags["tower"]:
            flags["tower"] = self.modeling_options["TowerSE"]["flag"]
        if flags["hub"]:
            flags["hub"] = self.modeling_options["DriveSE"]["flag"]
        if flags["nacelle"]:
            flags["nacelle"] = self.modeling_options["DriveSE"]["flag"]
        if flags["generator"]:
            flags["generator"] = self.modeling_options["DriveSE"]["flag"]
        flags["hub"] = flags["nacelle"] = flags["hub"] or flags["nacelle"]  # Hub and nacelle have to go together

        # Blades and airfoils
        if flags["blade"] and not flags["airfoils"]:
            raise ValueError("Blades/rotor analysis is requested but no airfoils are found")
        if flags["airfoils"] and not flags["blade"]:
            print("WARNING: Airfoils provided but no blades/rotor found or RotorSE deactivated")

        # Blades, tower, monopile and environment
        if flags["blade"] and not flags["environment"]:
            raise ValueError("Blades/rotor analysis is requested but no environment input found")
        if flags["tower"] and not flags["environment"]:
            raise ValueError("Tower analysis is requested but no environment input found")
        if flags["monopile"] and not flags["environment"]:
            raise ValueError("Monopile analysis is requested but no environment input found")
        if flags["floating_platform"] and not flags["environment"]:
            raise ValueError("Floating analysis is requested but no environment input found")
        if flags["environment"] and not (
            flags["blade"] or flags["tower"] or flags["monopile"] or flags["floating_platform"]
        ):
            print("WARNING: Environment provided but no related component found found")

        # Floating/monopile
        if flags["floating_platform"] and flags["monopile"]:
            raise ValueError("Cannot have both floating and monopile components")

        # Water depth check
        if "water_depth" in self.wt_init["environment"]:
            if self.wt_init["environment"]["water_depth"] <= 0.0 and flags["offshore"]:
                raise ValueError("Water depth must be > 0 to do monopile or floating analysis")

    def set_openmdao_vectors(self):
        # Class instance to determine all the parameters used to initialize the openmdao arrays, i.e. number of airfoils, number of angles of attack, number of blade spanwise stations, etc
        # ==modeling_options = {}

        # Materials
        self.modeling_options["materials"] = {}
        self.modeling_options["materials"]["n_mat"] = len(self.wt_init["materials"])

        # Airfoils
        if self.modeling_options["flags"]["airfoils"]:
            self.modeling_options["RotorSE"]["n_af"] = len(self.wt_init["airfoils"])
            self.modeling_options["RotorSE"]["n_aoa"] = self.modeling_options["RotorSE"]["n_aoa"]
            if self.modeling_options["RotorSE"]["n_aoa"] / 4.0 == int(self.modeling_options["RotorSE"]["n_aoa"] / 4.0):
                # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi
                self.modeling_options["RotorSE"]["aoa"] = np.unique(
                    np.hstack(
                        [
                            np.linspace(-np.pi, -np.pi / 6.0, int(self.modeling_options["RotorSE"]["n_aoa"] / 4.0 + 1)),
                            np.linspace(
                                -np.pi / 6.0, np.pi / 6.0, int(self.modeling_options["RotorSE"]["n_aoa"] / 2.0)
                            ),
                            np.linspace(np.pi / 6.0, np.pi, int(self.modeling_options["RotorSE"]["n_aoa"] / 4.0 + 1)),
                        ]
                    )
                )
            else:
                self.modeling_options["RotorSE"]["aoa"] = np.linspace(
                    -np.pi, np.pi, self.modeling_options["RotorSE"]["n_aoa"]
                )
                print(
                    "WARNING: If you like a grid of angles of attack more refined between +- 30 deg, please choose a n_aoa in the analysis option input file that is a multiple of 4. The current value of "
                    + str(self.modeling_options["RotorSE"]["n_aoa"])
                    + " is not a multiple of 4 and an equally spaced grid is adopted."
                )
            Re_all = []
            for i in range(self.modeling_options["RotorSE"]["n_af"]):
                for j in range(len(self.wt_init["airfoils"][i]["polars"])):
                    Re_all.append(self.wt_init["airfoils"][i]["polars"][j]["re"])
            self.modeling_options["RotorSE"]["n_Re"] = len(np.unique(Re_all))
            self.modeling_options["RotorSE"]["n_tab"] = 1
            self.modeling_options["RotorSE"]["n_xy"] = self.modeling_options["RotorSE"]["n_xy"]
            self.modeling_options["RotorSE"]["af_used"] = self.wt_init["components"]["blade"]["outer_shape_bem"][
                "airfoil_position"
            ]["labels"]

        # Blade
        if self.modeling_options["flags"]["blade"]:
            self.modeling_options["RotorSE"]["nd_span"] = np.linspace(
                0.0, 1.0, self.modeling_options["RotorSE"]["n_span"]
            )  # Equally spaced non-dimensional spanwise grid
            self.modeling_options["RotorSE"]["n_af_span"] = len(
                self.wt_init["components"]["blade"]["outer_shape_bem"]["airfoil_position"]["labels"]
            )  # This is the number of airfoils defined along blade span and it is often different than n_af, which is the number of airfoils defined in the airfoil database
            self.modeling_options["RotorSE"]["n_webs"] = len(
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"]
            )
            self.modeling_options["RotorSE"]["n_layers"] = len(
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"]
            )
            self.modeling_options["RotorSE"]["lofted_output"] = False
            self.modeling_options["RotorSE"]["n_freq"] = 10  # Number of blade nat frequencies computed

            self.modeling_options["RotorSE"]["layer_name"] = self.modeling_options["RotorSE"]["n_layers"] * [""]
            self.modeling_options["RotorSE"]["layer_mat"] = self.modeling_options["RotorSE"]["n_layers"] * [""]
            for i in range(self.modeling_options["RotorSE"]["n_layers"]):
                self.modeling_options["RotorSE"]["layer_name"][i] = self.wt_init["components"]["blade"][
                    "internal_structure_2d_fem"
                ]["layers"][i]["name"]
                self.modeling_options["RotorSE"]["layer_mat"][i] = self.wt_init["components"]["blade"][
                    "internal_structure_2d_fem"
                ]["layers"][i]["material"]

            self.modeling_options["RotorSE"]["web_name"] = self.modeling_options["RotorSE"]["n_webs"] * [""]
            for i in range(self.modeling_options["RotorSE"]["n_webs"]):
                self.modeling_options["RotorSE"]["web_name"][i] = self.wt_init["components"]["blade"][
                    "internal_structure_2d_fem"
                ]["webs"][i]["name"]

            # Distributed aerodynamic control devices along blade
            self.modeling_options["RotorSE"]["n_te_flaps"] = 0
            if "aerodynamic_control" in self.wt_init["components"]["blade"]:
                if "te_flaps" in self.wt_init["components"]["blade"]["aerodynamic_control"]:
                    self.modeling_options["RotorSE"]["n_te_flaps"] = len(
                        self.wt_init["components"]["blade"]["aerodynamic_control"]["te_flaps"]
                    )
                    self.modeling_options["RotorSE"]["n_tab"] = 3
                else:
                    raise RuntimeError(
                        "A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem."
                    )

        # Drivetrain
        if self.modeling_options["flags"]["nacelle"]:
            self.modeling_options["DriveSE"]["direct"] = self.wt_init["assembly"]["drivetrain"].lower() in [
                "direct",
                "direct_drive",
                "pm_direct_drive",
            ]

        # Tower
        if self.modeling_options["flags"]["tower"]:
            self.modeling_options["TowerSE"]["n_height_tower"] = len(
                self.wt_init["components"]["tower"]["outer_shape_bem"]["outer_diameter"]["grid"]
            )
            self.modeling_options["TowerSE"]["n_layers_tower"] = len(
                self.wt_init["components"]["tower"]["internal_structure_2d_fem"]["layers"]
            )

        # Monopile
        if self.modeling_options["flags"]["monopile"]:
            self.modeling_options["TowerSE"]["n_height_monopile"] = len(
                self.wt_init["components"]["monopile"]["outer_shape_bem"]["outer_diameter"]["grid"]
            )
            self.modeling_options["TowerSE"]["n_layers_monopile"] = len(
                self.wt_init["components"]["monopile"]["internal_structure_2d_fem"]["layers"]
            )

        # Floating platform
        self.modeling_options["floating"] = {}
        if self.modeling_options["flags"]["floating_platform"]:
            n_joints = len(self.wt_init["components"]["floating_platform"]["joints"])
            self.modeling_options["floating"]["joints"] = {}
            self.modeling_options["floating"]["joints"]["n_joints"] = n_joints
            self.modeling_options["floating"]["joints"]["name"] = [""] * n_joints
            self.modeling_options["floating"]["joints"]["transition"] = [False] * n_joints
            self.modeling_options["floating"]["joints"]["cylindrical"] = [False] * n_joints
            for i in range(n_joints):
                self.modeling_options["floating"]["joints"]["name"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["name"]
                self.modeling_options["floating"]["joints"]["transition"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["transition"]
                self.modeling_options["floating"]["joints"]["cylindrical"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["joints"][i]["cylindrical"]

            # Check that there is at most one transition joint
            if self.modeling_options["floating"]["joints"]["transition"].count(True) > 1:
                raise ValueError("Can only support one tower on the floating platform for now")
            try:
                self.modeling_options["floating"]["transition_joint"] = self.modeling_options["floating"]["joints"][
                    "transition"
                ].index(True)
            except:
                self.modeling_options["floating"]["transition_joint"] = None

            n_members = len(self.wt_init["components"]["floating_platform"]["members"])
            self.modeling_options["floating"]["members"] = {}
            self.modeling_options["floating"]["members"]["n_members"] = n_members
            self.modeling_options["floating"]["members"]["name"] = [""] * n_members
            self.modeling_options["floating"]["members"]["joint1"] = [""] * n_members
            self.modeling_options["floating"]["members"]["joint2"] = [""] * n_members
            self.modeling_options["floating"]["members"]["outer_shape"] = [""] * n_members
            self.modeling_options["floating"]["members"]["n_layers"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_ballasts"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_axial_joints"] = np.zeros(n_members, dtype=int)
            for i in range(n_members):
                self.modeling_options["floating"]["members"]["name"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["name"]
                self.modeling_options["floating"]["members"]["joint1"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["joint1"]
                self.modeling_options["floating"]["members"]["joint2"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["joint2"]
                self.modeling_options["floating"]["members"]["outer_shape"][i] = self.wt_init["components"][
                    "floating_platform"
                ]["members"][i]["outer_shape"]["shape"]

                n_layers = len(
                    self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["layers"]
                )
                self.modeling_options["floating"]["members"]["n_layers"][i] = n_layers
                if "ballasts" in self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]:
                    n_ballasts = len(
                        self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["ballasts"]
                    )
                else:
                    n_ballasts = 0
                self.modeling_options["floating"]["members"]["n_ballasts"][i] = n_ballasts
                grid = []
                if "bulkhead" in self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]:
                    grid = np.unique(
                        np.hstack(
                            [
                                self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"][
                                    "outer_diameter"
                                ]["grid"],
                                self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                                    "bulkhead"
                                ]["thickness"]["grid"],
                            ]
                        )
                    )
                else:
                    grid = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"][
                        "outer_diameter"
                    ]["grid"]
                self.modeling_options["floating"]["members"][
                    "layer_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = [""] * n_layers
                for j in range(n_layers):
                    self.modeling_options["floating"]["members"][
                        "layer_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                    ][j] = self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                        "layers"
                    ][
                        j
                    ][
                        "material"
                    ]
                    grid = np.unique(
                        np.hstack(
                            [
                                grid,
                                self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                                    "layers"
                                ][j]["thickness"]["grid"],
                            ]
                        )
                    )
                self.modeling_options["floating"]["members"][
                    "ballast_flag_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = [False] * n_ballasts
                self.modeling_options["floating"]["members"][
                    "ballast_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = [""] * n_ballasts
                for k in range(n_ballasts):
                    self.modeling_options["floating"]["members"][
                        "ballast_flag_member_" + self.modeling_options["floating"]["members"]["name"][i]
                    ][k] = self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                        "ballasts"
                    ][
                        k
                    ][
                        "variable_flag"
                    ]
                    if (
                        self.modeling_options["floating"]["members"][
                            "ballast_flag_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ][k]
                        == False
                    ):
                        self.modeling_options["floating"]["members"][
                            "ballast_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ][k] = self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                            "ballasts"
                        ][
                            k
                        ][
                            "material"
                        ]
                    grid = np.unique(
                        np.hstack(
                            [
                                grid,
                                self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                                    "ballasts"
                                ][k]["grid"],
                            ]
                        )
                    )
                if "axial_joints" in self.wt_init["components"]["floating_platform"]["members"][i]:
                    n_axial_joints = len(self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"])
                    self.modeling_options["floating"]["members"]["n_axial_joints"][i] = n_axial_joints
                    self.modeling_options["floating"]["members"][
                        "axial_joint_name_member_" + self.modeling_options["floating"]["members"]["name"][i]
                    ] = [""] * n_axial_joints
                    for m in range(n_axial_joints):
                        self.modeling_options["floating"]["members"][
                            "axial_joint_name_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ] = self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m]["name"]
                        grid = np.unique(
                            np.hstack(
                                [
                                    grid,
                                    self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m][
                                        "grid"
                                    ],
                                ]
                            )
                        )
                else:
                    self.modeling_options["floating"]["members"]["n_axial_joints"][i] = 0
                self.modeling_options["floating"]["members"][
                    "grid_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = grid

        # Mooring
        self.modeling_options["mooring"] = {}
        if self.modeling_options["flags"]["mooring"]:
            n_nodes = len(self.wt_init["components"]["mooring"]["nodes"])
            n_lines = len(self.wt_init["components"]["mooring"]["lines"])
            n_line_types = len(self.wt_init["components"]["mooring"]["line_types"])
            n_anchor_types = len(self.wt_init["components"]["mooring"]["anchor_types"])
            self.modeling_options["mooring"]["n_nodes"] = n_nodes
            self.modeling_options["mooring"]["n_lines"] = n_lines
            self.modeling_options["mooring"]["n_line_types"] = n_line_types
            self.modeling_options["mooring"]["n_anchor_types"] = n_anchor_types
            self.modeling_options["mooring"]["node_type"] = [""] * n_nodes
            self.modeling_options["mooring"]["anchor_type"] = [""] * n_nodes
            self.modeling_options["mooring"]["fairlead_type"] = [""] * n_nodes
            for i in range(n_nodes):
                self.modeling_options["mooring"]["node_type"][i] = self.wt_init["components"]["mooring"]["nodes"][i][
                    "node_type"
                ]
                self.modeling_options["mooring"]["anchor_type"][i] = self.wt_init["components"]["mooring"]["nodes"][i][
                    "anchor_type"
                ]
                self.modeling_options["mooring"]["fairlead_type"][i] = self.wt_init["components"]["mooring"]["nodes"][
                    i
                ]["fairlead_type"]
            self.modeling_options["mooring"]["node1"] = [""] * n_lines
            self.modeling_options["mooring"]["node2"] = [""] * n_lines
            self.modeling_options["mooring"]["line_type"] = [""] * n_lines
            for i in range(n_lines):
                self.modeling_options["mooring"]["node1"][i] = self.wt_init["components"]["mooring"]["lines"][i][
                    "node1"
                ]
                self.modeling_options["mooring"]["node2"][i] = self.wt_init["components"]["mooring"]["lines"][i][
                    "node2"
                ]
                self.modeling_options["mooring"]["line_type"][i] = self.wt_init["components"]["mooring"]["lines"][i][
                    "line_type"
                ]
            self.modeling_options["mooring"]["line_type_name"] = [""] * n_line_types
            for i in range(n_line_types):
                self.modeling_options["mooring"]["line_type_name"][i] = self.wt_init["components"]["mooring"][
                    "line_types"
                ][i]["name"]
            self.modeling_options["mooring"]["anchor_type_name"] = [""] * n_anchor_types
            for i in range(n_anchor_types):
                self.modeling_options["mooring"]["anchor_type_name"][i] = self.wt_init["components"]["mooring"][
                    "anchor_types"
                ][i]["name"]

        # Assembly
        self.modeling_options["assembly"] = {}
        self.modeling_options["assembly"]["number_of_blades"] = int(self.wt_init["assembly"]["number_of_blades"])

    def set_opt_flags(self):
        # Recursively look for flags to set global optimization flag
        def recursive_flag(d):
            opt_flag = False
            for k, v in d.items():
                if isinstance(v, dict):
                    opt_flag = opt_flag or recursive_flag(v)
                elif k == "flag":
                    opt_flag = opt_flag or v
            return opt_flag

        # The user can provide `opt_flag` in analysis_options.yaml,
        # but if it's not provided, we check the individual opt flags
        # from analysis_options.yaml and set a global `opt_flag`
        if "opt_flag" in self.analysis_options["driver"]:
            self.analysis_options["opt_flag"] = self.analysis_options["driver"]["opt_flag"]
        else:
            self.analysis_options["opt_flag"] = recursive_flag(self.analysis_options["design_variables"])

        # If not an optimization DV, then the number of points should be same as the discretization
        blade_opt_options = self.analysis_options["design_variables"]["blade"]
        if not blade_opt_options["aero_shape"]["twist"]["flag"]:
            blade_opt_options["aero_shape"]["twist"]["n_opt"] = self.modeling_options["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["twist"]["n_opt"] < 4:
            raise ValueError("Cannot optimize twist with less than 4 control points along blade span")

        if not blade_opt_options["aero_shape"]["chord"]["flag"]:
            blade_opt_options["aero_shape"]["chord"]["n_opt"] = self.modeling_options["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["chord"]["n_opt"] < 4:
            raise ValueError("Cannot optimize chord with less than 4 control points along blade span")

        if not blade_opt_options["structure"]["spar_cap_ss"]["flag"]:
            blade_opt_options["structure"]["spar_cap_ss"]["n_opt"] = self.modeling_options["RotorSE"]["n_span"]
        elif blade_opt_options["structure"]["spar_cap_ss"]["n_opt"] < 4:
            raise ValueError("Cannot optimize spar cap suction side with less than 4 control points along blade span")

        if not blade_opt_options["structure"]["spar_cap_ps"]["flag"]:
            blade_opt_options["structure"]["spar_cap_ps"]["n_opt"] = self.modeling_options["RotorSE"]["n_span"]
        elif blade_opt_options["structure"]["spar_cap_ps"]["n_opt"] < 4:
            raise ValueError("Cannot optimize spar cap pressure side with less than 4 control points along blade span")

    def write_ontology(self, wt_opt, fname_output):

        # Update blade
        if self.modeling_options["flags"]["blade"]:
            # Update blade outer shape
            self.wt_init["components"]["blade"]["outer_shape_bem"]["airfoil_position"]["grid"] = wt_opt[
                "blade.opt_var.af_position"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["chord"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["chord"]["values"] = wt_opt[
                "blade.pa.chord_param"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["twist"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["twist"]["values"] = wt_opt[
                "blade.pa.twist_param"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["pitch_axis"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["pitch_axis"]["values"] = wt_opt[
                "blade.outer_shape_bem.pitch_axis"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["x"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["y"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["z"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["x"]["values"] = wt_opt[
                "blade.outer_shape_bem.ref_axis"
            ][:, 0].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["y"]["values"] = wt_opt[
                "blade.outer_shape_bem.ref_axis"
            ][:, 1].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["z"]["values"] = wt_opt[
                "blade.outer_shape_bem.ref_axis"
            ][:, 2].tolist()

            # Update blade structure
            # Reference axis from blade outer shape
            self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["reference_axis"] = self.wt_init[
                "components"
            ]["blade"]["outer_shape_bem"]["reference_axis"]
            # Webs positions
            for i in range(self.modeling_options["RotorSE"]["n_webs"]):
                if "rotation" in self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"]:
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["rotation"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["rotation"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.web_rotation"][i, :].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["offset_y_pa"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["offset_y_pa"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.web_offset_y_pa"][i, :].tolist()
                if "start_nd_arc" not in self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]:
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"] = {}
                if "end_nd_arc" not in self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]:
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"] = {}
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"][
                    "grid"
                ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["start_nd_arc"][
                    "values"
                ] = wt_opt["blade.internal_structure_2d_fem.web_start_nd"][i, :].tolist()
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"][
                    "grid"
                ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"][i]["end_nd_arc"][
                    "values"
                ] = wt_opt["blade.internal_structure_2d_fem.web_end_nd"][i, :].tolist()

            # Structural layers
            for i in range(self.modeling_options["RotorSE"]["n_layers"]):
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "grid"
                ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "values"
                ] = wt_opt["blade.ps.layer_thickness_param"][i, :].tolist()
                if wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] < 7:
                    if (
                        "start_nd_arc"
                        not in self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]
                    ):
                        self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i][
                            "start_nd_arc"
                        ] = {}
                    if (
                        "end_nd_arc"
                        not in self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]
                    ):
                        self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"] = {}
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["start_nd_arc"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.layer_start_nd"][i, :].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["end_nd_arc"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.layer_end_nd"][i, :].tolist()
                if (
                    wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] > 1
                    and wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] < 6
                ):
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["width"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["width"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.layer_width"][i, :].tolist()
                if (
                    wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] == 2
                    or wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] == 3
                ):
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["rotation"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["rotation"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.layer_rotation"][i, :].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["offset_y_pa"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["offset_y_pa"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.layer_offset_y_pa"][i, :].tolist()
                if (
                    wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] == 4
                    or wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] == 5
                ):
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["midpoint_nd_arc"][
                        "grid"
                    ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                    self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["midpoint_nd_arc"][
                        "values"
                    ] = wt_opt["blade.internal_structure_2d_fem.layer_midpoint_nd"][i, :].tolist()

                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"] = {}

                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"][
                    "grid"
                ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["fiber_orientation"][
                    "values"
                ] = np.zeros(len(wt_opt["blade.internal_structure_2d_fem.s"])).tolist()

            # Elastic properties of the blade
            self.wt_init["components"]["blade"]["elastic_properties_mb"] = {}
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"] = {}
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["reference_axis"] = self.wt_init[
                "components"
            ]["blade"]["internal_structure_2d_fem"]["reference_axis"]
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["twist"] = self.wt_init[
                "components"
            ]["blade"]["outer_shape_bem"]["twist"]
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["stiff_matrix"] = {}
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["stiff_matrix"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            K = []
            for i in range(self.modeling_options["RotorSE"]["n_span"]):
                Ki = np.zeros(21)
                Ki[11] = wt_opt["re.EA"][i]
                Ki[15] = wt_opt["re.EIxx"][i]
                Ki[18] = wt_opt["re.EIyy"][i]
                Ki[20] = wt_opt["re.GJ"][i]
                K.append(Ki.tolist())
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["stiff_matrix"]["values"] = K
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["inertia_matrix"] = {}
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["inertia_matrix"][
                "grid"
            ] = wt_opt["blade.outer_shape_bem.s"].tolist()
            I = []
            for i in range(self.modeling_options["RotorSE"]["n_span"]):
                Ii = np.zeros(21)
                Ii[0] = wt_opt["re.rhoA"][i]
                Ii[5] = -wt_opt["re.rhoA"][i] * wt_opt["re.precomp.y_cg"][i]
                Ii[6] = wt_opt["re.rhoA"][i]
                Ii[10] = wt_opt["re.rhoA"][i] * wt_opt["re.precomp.x_cg"][i]
                Ii[11] = wt_opt["re.rhoA"][i]
                Ii[12] = wt_opt["re.rhoA"][i] * wt_opt["re.precomp.y_cg"][i]
                Ii[13] = -wt_opt["re.rhoA"][i] * wt_opt["re.precomp.x_cg"][i]
                Ii[15] = wt_opt["re.precomp.edge_iner"][i]
                Ii[16] = wt_opt["re.precomp.edge_iner"][i]
                # Ii[18] = wt_opt['re.precomp.edge_iner'][i]
                Ii[20] = wt_opt["re.rhoJ"][i]
                I.append(Ii.tolist())
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["inertia_matrix"]["values"] = I

        # Update hub
        if self.modeling_options["flags"]["hub"]:
            # Update hub
            self.wt_init["components"]["hub"]["cone_angle"] = float(wt_opt["hub.cone"])
            self.wt_init["components"]["hub"]["flange_t2shell_t"] = float(wt_opt["hub.flange_t2shell_t"])
            self.wt_init["components"]["hub"]["flange_OD2hub_D"] = float(wt_opt["hub.flange_OD2hub_D"])
            self.wt_init["components"]["hub"]["flange_ID2OD"] = float(wt_opt["hub.flange_ID2flange_OD"])
            self.wt_init["components"]["hub"]["hub_blade_spacing_margin"] = float(wt_opt["hub.hub_in2out_circ"])
            self.wt_init["components"]["hub"]["hub_stress_concentration"] = float(
                wt_opt["hub.hub_stress_concentration"]
            )
            self.wt_init["components"]["hub"]["n_front_brackets"] = int(wt_opt["hub.n_front_brackets"])
            self.wt_init["components"]["hub"]["n_rear_brackets"] = int(wt_opt["hub.n_rear_brackets"])
            self.wt_init["components"]["hub"]["clearance_hub_spinner"] = float(wt_opt["hub.clearance_hub_spinner"])
            self.wt_init["components"]["hub"]["spin_hole_incr"] = float(wt_opt["hub.spin_hole_incr"])
            self.wt_init["components"]["hub"]["pitch_system_scaling_factor"] = float(
                wt_opt["hub.pitch_system_scaling_factor"]
            )
            self.wt_init["components"]["hub"]["spinner_gust_ws"] = float(wt_opt["hub.spinner_gust_ws"])

        # Update nacelle
        if self.modeling_options["flags"]["nacelle"]:
            # Common direct and geared
            self.wt_init["components"]["nacelle"]["drivetrain"]["uptilt"] = float(wt_opt["nacelle.uptilt"])
            self.wt_init["components"]["nacelle"]["drivetrain"]["distance_tt_hub"] = float(
                wt_opt["nacelle.distance_tt_hub"]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["overhang"] = float(wt_opt["nacelle.overhang"])
            self.wt_init["components"]["nacelle"]["drivetrain"]["distance_hub_mb"] = float(
                wt_opt["nacelle.distance_hub2mb"]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["distance_mb_mb"] = float(
                wt_opt["nacelle.distance_mb2mb"]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["generator_length"] = float(
                wt_opt["nacelle.L_generator"]
            )
            s_lss = np.linspace(0.0, 1.0, len(wt_opt["nacelle.lss_diameter"])).tolist()
            self.wt_init["components"]["nacelle"]["drivetrain"]["lss_diameter"] = wt_opt[
                "nacelle.lss_diameter"
            ].tolist()
            self.wt_init["components"]["nacelle"]["drivetrain"]["lss_wall_thickness"] = wt_opt[
                "nacelle.lss_wall_thickness"
            ].tolist()
            self.wt_init["components"]["nacelle"]["drivetrain"]["gear_ratio"] = float(wt_opt["nacelle.gear_ratio"])
            self.wt_init["components"]["nacelle"]["drivetrain"]["gearbox_efficiency"] = float(
                wt_opt["nacelle.gearbox_efficiency"]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["mb1Type"] = wt_opt["nacelle.mb1Type"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["mb2Type"] = wt_opt["nacelle.mb2Type"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["uptower"] = wt_opt["nacelle.uptower"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["lss_material"] = wt_opt["nacelle.lss_material"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_material"] = wt_opt[
                "nacelle.bedplate_material"
            ]

            if self.modeling_options["DriveSE"]["direct"]:
                # Direct only
                s_nose = np.linspace(0.0, 1.0, len(wt_opt["nacelle.nose_diameter"])).tolist()
                s_bed = np.linspace(0.0, 1.0, len(wt_opt["nacelle.bedplate_wall_thickness"])).tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["nose_diameter"] = wt_opt[
                    "nacelle.nose_diameter"
                ].tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["nose_wall_thickness"] = wt_opt[
                    "nacelle.nose_wall_thickness"
                ].tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_wall_thickness"]["grid"] = s_bed
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_wall_thickness"]["values"] = wt_opt[
                    "nacelle.bedplate_wall_thickness"
                ].tolist()
            else:
                # Geared only
                s_hss = np.linspace(0.0, 1.0, len(wt_opt["nacelle.hss_diameter"])).tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_length"] = float(wt_opt["nacelle.hss_length"])
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_diameter"] = wt_opt[
                    "nacelle.hss_diameter"
                ].tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_wall_thickness"] = wt_opt[
                    "nacelle.hss_wall_thickness"
                ].tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_flange_width"] = float(
                    wt_opt["nacelle.bedplate_flange_width"]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_flange_thickness"] = float(
                    wt_opt["nacelle.bedplate_flange_thickness"]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_web_thickness"] = float(
                    wt_opt["nacelle.bedplate_web_thickness"]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["gear_configuration"] = wt_opt[
                    "nacelle.gear_configuration"
                ]
                self.wt_init["components"]["nacelle"]["drivetrain"]["planet_numbers"] = wt_opt["nacelle.planet_numbers"]
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_material"] = wt_opt["nacelle.hss_material"]

        # Update generator
        if self.modeling_options["flags"]["generator"]:

            self.wt_init["components"]["nacelle"]["generator"]["B_r"] = float(wt_opt["generator.B_r"])
            self.wt_init["components"]["nacelle"]["generator"]["P_Fe0e"] = float(wt_opt["generator.P_Fe0e"])
            self.wt_init["components"]["nacelle"]["generator"]["P_Fe0h"] = float(wt_opt["generator.P_Fe0h"])
            self.wt_init["components"]["nacelle"]["generator"]["S_N"] = float(wt_opt["generator.S_N"])
            self.wt_init["components"]["nacelle"]["generator"]["alpha_p"] = float(wt_opt["generator.alpha_p"])
            self.wt_init["components"]["nacelle"]["generator"]["b_r_tau_r"] = float(wt_opt["generator.b_r_tau_r"])
            self.wt_init["components"]["nacelle"]["generator"]["b_ro"] = float(wt_opt["generator.b_ro"])
            self.wt_init["components"]["nacelle"]["generator"]["b_s_tau_s"] = float(wt_opt["generator.b_s_tau_s"])
            self.wt_init["components"]["nacelle"]["generator"]["b_so"] = float(wt_opt["generator.b_so"])
            self.wt_init["components"]["nacelle"]["generator"]["cofi"] = float(wt_opt["generator.cofi"])
            self.wt_init["components"]["nacelle"]["generator"]["freq"] = float(wt_opt["generator.freq"])
            self.wt_init["components"]["nacelle"]["generator"]["h_i"] = float(wt_opt["generator.h_i"])
            self.wt_init["components"]["nacelle"]["generator"]["h_sy0"] = float(wt_opt["generator.h_sy0"])
            self.wt_init["components"]["nacelle"]["generator"]["h_w"] = float(wt_opt["generator.h_w"])
            self.wt_init["components"]["nacelle"]["generator"]["k_fes"] = float(wt_opt["generator.k_fes"])
            self.wt_init["components"]["nacelle"]["generator"]["k_fillr"] = float(wt_opt["generator.k_fillr"])
            self.wt_init["components"]["nacelle"]["generator"]["k_fills"] = float(wt_opt["generator.k_fills"])
            self.wt_init["components"]["nacelle"]["generator"]["k_s"] = float(wt_opt["generator.k_s"])
            self.wt_init["components"]["nacelle"]["generator"]["m"] = float(wt_opt["generator.m"])
            self.wt_init["components"]["nacelle"]["generator"]["mu_0"] = float(wt_opt["generator.mu_0"])
            self.wt_init["components"]["nacelle"]["generator"]["mu_r"] = float(wt_opt["generator.mu_r"])
            self.wt_init["components"]["nacelle"]["generator"]["p"] = float(wt_opt["generator.p"])
            self.wt_init["components"]["nacelle"]["generator"]["phi"] = float(wt_opt["generator.phi"])
            self.wt_init["components"]["nacelle"]["generator"]["q1"] = float(wt_opt["generator.q1"])
            self.wt_init["components"]["nacelle"]["generator"]["q2"] = float(wt_opt["generator.q2"])
            self.wt_init["components"]["nacelle"]["generator"]["ratio_mw2pp"] = float(wt_opt["generator.ratio_mw2pp"])
            self.wt_init["components"]["nacelle"]["generator"]["resist_Cu"] = float(wt_opt["generator.resist_Cu"])
            self.wt_init["components"]["nacelle"]["generator"]["sigma"] = float(wt_opt["generator.sigma"])
            self.wt_init["components"]["nacelle"]["generator"]["y_tau_p"] = float(wt_opt["generator.y_tau_p"])
            self.wt_init["components"]["nacelle"]["generator"]["y_tau_pr"] = float(wt_opt["generator.y_tau_pr"])

            self.wt_init["components"]["nacelle"]["generator"]["I_0"] = float(wt_opt["generator.I_0"])
            self.wt_init["components"]["nacelle"]["generator"]["d_r"] = float(wt_opt["generator.d_r"])
            self.wt_init["components"]["nacelle"]["generator"]["h_m"] = float(wt_opt["generator.h_m"])
            self.wt_init["components"]["nacelle"]["generator"]["h_0"] = float(wt_opt["generator.h_0"])
            self.wt_init["components"]["nacelle"]["generator"]["h_s"] = float(wt_opt["generator.h_s"])
            self.wt_init["components"]["nacelle"]["generator"]["len_s"] = float(wt_opt["generator.len_s"])
            self.wt_init["components"]["nacelle"]["generator"]["n_r"] = float(wt_opt["generator.n_r"])
            self.wt_init["components"]["nacelle"]["generator"]["rad_ag"] = float(wt_opt["generator.rad_ag"])
            self.wt_init["components"]["nacelle"]["generator"]["t_wr"] = float(wt_opt["generator.t_wr"])

            self.wt_init["components"]["nacelle"]["generator"]["n_s"] = float(wt_opt["generator.n_s"])
            self.wt_init["components"]["nacelle"]["generator"]["b_st"] = float(wt_opt["generator.b_st"])
            self.wt_init["components"]["nacelle"]["generator"]["d_s"] = float(wt_opt["generator.d_s"])
            self.wt_init["components"]["nacelle"]["generator"]["t_ws"] = float(wt_opt["generator.t_ws"])

            self.wt_init["components"]["nacelle"]["generator"]["rho_Copper"] = float(wt_opt["generator.rho_Copper"])
            self.wt_init["components"]["nacelle"]["generator"]["rho_Fe"] = float(wt_opt["generator.rho_Fe"])
            self.wt_init["components"]["nacelle"]["generator"]["rho_Fes"] = float(wt_opt["generator.rho_Fes"])
            self.wt_init["components"]["nacelle"]["generator"]["rho_PM"] = float(wt_opt["generator.rho_PM"])

            self.wt_init["components"]["nacelle"]["generator"]["C_Cu"] = float(wt_opt["generator.C_Cu"])
            self.wt_init["components"]["nacelle"]["generator"]["C_Fe"] = float(wt_opt["generator.C_Fe"])
            self.wt_init["components"]["nacelle"]["generator"]["C_Fes"] = float(wt_opt["generator.C_Fes"])
            self.wt_init["components"]["nacelle"]["generator"]["C_PM"] = float(wt_opt["generator.C_PM"])

            if self.modeling_options["GeneratorSE"]["type"] in ["pmsg_outer"]:
                self.wt_init["components"]["nacelle"]["generator"]["N_c"] = float(wt_opt["generator.N_c"])
                self.wt_init["components"]["nacelle"]["generator"]["b"] = float(wt_opt["generator.b"])
                self.wt_init["components"]["nacelle"]["generator"]["c"] = float(wt_opt["generator.c"])
                self.wt_init["components"]["nacelle"]["generator"]["E_p"] = float(wt_opt["generator.E_p"])
                self.wt_init["components"]["nacelle"]["generator"]["h_yr"] = float(wt_opt["generator.h_yr"])
                self.wt_init["components"]["nacelle"]["generator"]["h_ys"] = float(wt_opt["generator.h_ys"])
                self.wt_init["components"]["nacelle"]["generator"]["h_sr"] = float(wt_opt["generator.h_sr"])
                self.wt_init["components"]["nacelle"]["generator"]["h_ss"] = float(wt_opt["generator.h_ss"])
                self.wt_init["components"]["nacelle"]["generator"]["t_r"] = float(wt_opt["generator.t_r"])
                self.wt_init["components"]["nacelle"]["generator"]["t_s"] = float(wt_opt["generator.t_s"])

                self.wt_init["components"]["nacelle"]["generator"]["u_allow_pcent"] = float(
                    wt_opt["generator.u_allow_pcent"]
                )
                self.wt_init["components"]["nacelle"]["generator"]["y_allow_pcent"] = float(
                    wt_opt["generator.y_allow_pcent"]
                )
                self.wt_init["components"]["nacelle"]["generator"]["z_allow_deg"] = float(
                    wt_opt["generator.z_allow_deg"]
                )
                self.wt_init["components"]["nacelle"]["generator"]["B_tmax"] = float(wt_opt["generator.B_tmax"])

            if self.modeling_options["GeneratorSE"]["type"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
                self.wt_init["components"]["nacelle"]["generator"]["tau_p"] = float(wt_opt["generator.tau_p"])
                self.wt_init["components"]["nacelle"]["generator"]["h_ys"] = float(wt_opt["generator.h_ys"])
                self.wt_init["components"]["nacelle"]["generator"]["h_yr"] = float(wt_opt["generator.h_yr"])
                self.wt_init["components"]["nacelle"]["generator"]["b_arm"] = float(wt_opt["generator.b_arm"])

            elif self.modeling_options["GeneratorSE"]["type"] in ["scig", "dfig"]:
                self.wt_init["components"]["nacelle"]["generator"]["B_symax"] = float(wt_opt["generator.B_symax"])
                self.wt_init["components"]["nacelle"]["generator"]["S_Nmax"] = float(wt_opt["generator.S_Nmax"])

        # Update tower
        if self.modeling_options["flags"]["tower"]:
            self.wt_init["components"]["tower"]["outer_shape_bem"]["outer_diameter"]["grid"] = wt_opt[
                "tower_grid.s"
            ].tolist()
            self.wt_init["components"]["tower"]["outer_shape_bem"]["outer_diameter"]["values"] = wt_opt[
                "tower.diameter"
            ].tolist()
            self.wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["x"]["grid"] = wt_opt[
                "tower_grid.s"
            ].tolist()
            self.wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["y"]["grid"] = wt_opt[
                "tower_grid.s"
            ].tolist()
            self.wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["grid"] = wt_opt[
                "tower_grid.s"
            ].tolist()
            self.wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["x"]["values"] = wt_opt[
                "tower.ref_axis"
            ][:, 0].tolist()
            self.wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["y"]["values"] = wt_opt[
                "tower.ref_axis"
            ][:, 1].tolist()
            self.wt_init["components"]["tower"]["outer_shape_bem"]["reference_axis"]["z"]["values"] = wt_opt[
                "tower.ref_axis"
            ][:, 2].tolist()
            self.wt_init["components"]["tower"]["internal_structure_2d_fem"]["outfitting_factor"] = float(
                wt_opt["tower.outfitting_factor"]
            )
            for i in range(self.modeling_options["TowerSE"]["n_layers_tower"]):
                self.wt_init["components"]["tower"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "grid"
                ] = wt_opt["tower_grid.s"].tolist()
                self.wt_init["components"]["tower"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "values"
                ] = wt_opt["tower.layer_thickness"][i, :].tolist()

        # Update monopile
        if self.modeling_options["flags"]["monopile"]:
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["outer_diameter"]["grid"] = wt_opt[
                "monopile.s"
            ].tolist()
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["outer_diameter"]["values"] = wt_opt[
                "monopile.diameter"
            ].tolist()
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["reference_axis"]["x"]["grid"] = wt_opt[
                "monopile.s"
            ].tolist()
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["reference_axis"]["y"]["grid"] = wt_opt[
                "monopile.s"
            ].tolist()
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["reference_axis"]["z"]["grid"] = wt_opt[
                "monopile.s"
            ].tolist()
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["reference_axis"]["x"]["values"] = wt_opt[
                "monopile.ref_axis"
            ][:, 0].tolist()
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["reference_axis"]["y"]["values"] = wt_opt[
                "monopile.ref_axis"
            ][:, 1].tolist()
            self.wt_init["components"]["monopile"]["outer_shape_bem"]["reference_axis"]["z"]["values"] = wt_opt[
                "monopile.ref_axis"
            ][:, 2].tolist()
            self.wt_init["components"]["monopile"]["internal_structure_2d_fem"]["outfitting_factor"] = float(
                wt_opt["monopile.outfitting_factor"]
            )
            for i in range(self.modeling_options["TowerSE"]["n_layers_monopile"]):
                self.wt_init["components"]["monopile"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "grid"
                ] = wt_opt["monopile.s"].tolist()
                self.wt_init["components"]["monopile"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "values"
                ] = wt_opt["monopile.layer_thickness"][i, :].tolist()

        # Update rotor nacelle assembly
        if self.modeling_options["flags"]["RNA"]:
            self.wt_init["components"]["RNA"] = {}
            self.wt_init["components"]["RNA"]["elastic_properties_mb"] = {}
            self.wt_init["components"]["RNA"]["elastic_properties_mb"]["mass"] = float(wt_opt["drivese.rna_mass"])
            self.wt_init["components"]["RNA"]["elastic_properties_mb"]["inertia"] = wt_opt["drivese.rna_I_TT"].tolist()
            self.wt_init["components"]["RNA"]["elastic_properties_mb"]["center_mass"] = wt_opt[
                "drivese.rna_cm"
            ].tolist()

        # Update rotor diameter and hub height
        self.wt_init["assembly"]["rotor_diameter"] = float(wt_opt["assembly.rotor_diameter"])
        self.wt_init["assembly"]["hub_height"] = float(wt_opt["assembly.hub_height"])

        # Update controller
        if self.modeling_options["flags"]["control"]:
            self.wt_init["control"]["torque"]["tsr"] = float(wt_opt["control.rated_TSR"])

        # Write yamls with updated values
        sch.write_geometry_yaml(self.wt_init, fname_output)

    def write_options(self, fname_output):
        sch.write_modeling_yaml(self.modeling_options, fname_output)
        sch.write_analysis_yaml(self.analysis_options, fname_output)
