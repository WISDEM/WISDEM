import numpy as np

import wisdem.inputs as sch


class WindTurbineOntologyPython(object):
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

        # Backwards compatibility
        modules = ["RotorSE", "DriveSE", "GeneratorSE", "TowerSE", "FixedBottomSE", "FloatingSE", "Loading", "BOS"]
        for m in modules:
            if m in self.modeling_options:
                self.modeling_options["WISDEM"][m].update(self.modeling_options[m])

        for k in ["blade", "hub", "nacelle", "tower", "monopile", "jacket", "floating_platform", "mooring", "RNA"]:
            self.modeling_options["flags"][k] = k in self.wt_init["components"]

        for k in ["assembly", "components", "airfoils", "materials", "control", "environment", "bos", "costs"]:
            self.modeling_options["flags"][k] = k in self.wt_init

        # Generator flag
        self.modeling_options["flags"]["generator"] = False
        if self.modeling_options["flags"]["nacelle"] and "generator" in self.wt_init["components"]["nacelle"]:
            self.modeling_options["flags"]["generator"] = True
            if not "GeneratorSE" in self.modeling_options["WISDEM"]:
                self.modeling_options["WISDEM"]["GeneratorSE"] = {}
            self.modeling_options["WISDEM"]["GeneratorSE"]["type"] = self.wt_init["components"]["nacelle"]["generator"][
                "generator_type"
            ].lower()

        # Offshore flags
        self.modeling_options["flags"]["floating"] = self.modeling_options["flags"]["floating_platform"]
        self.modeling_options["flags"]["offshore"] = (
            self.modeling_options["flags"]["floating"]
            or self.modeling_options["flags"]["monopile"]
            or self.modeling_options["flags"]["jacket"]
        )

        # Put in some logic about what needs to be in there
        flags = self.modeling_options["flags"]

        # Even if the block is in the inputs, the user can turn off via modeling options
        if flags["bos"]:
            flags["bos"] = self.modeling_options["WISDEM"]["BOS"]["flag"]
        if flags["blade"]:
            flags["blade"] = self.modeling_options["WISDEM"]["RotorSE"]["flag"]
        if flags["tower"]:
            flags["tower"] = self.modeling_options["WISDEM"]["TowerSE"]["flag"]
        if flags["monopile"]:
            flags["monopile"] = self.modeling_options["WISDEM"]["FixedBottomSE"]["flag"]
        if flags["jacket"]:
            flags["jacket"] = self.modeling_options["WISDEM"]["FixedBottomSE"]["flag"]
        if flags["hub"]:
            flags["hub"] = self.modeling_options["WISDEM"]["DriveSE"]["flag"]
        if flags["nacelle"]:
            flags["nacelle"] = self.modeling_options["WISDEM"]["DriveSE"]["flag"]
        if flags["generator"]:
            flags["generator"] = self.modeling_options["WISDEM"]["DriveSE"]["flag"]
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
        if flags["jacket"] and not flags["environment"]:
            raise ValueError("Jacket analysis is requested but no environment input found")
        if flags["jacket"] and flags["monopile"]:
            raise ValueError("Cannot specify both monopile and jacket support structures")
        if flags["floating_platform"] and not flags["environment"]:
            raise ValueError("Floating analysis is requested but no environment input found")
        if flags["environment"] and not (
            flags["blade"] or flags["tower"] or flags["monopile"] or flags["jacket"] or flags["floating_platform"]
        ):
            print("WARNING: Environment provided but no related component found found")

        # Floating/monopile
        if flags["floating_platform"] and (flags["monopile"] or flags["jacket"]):
            raise ValueError("Cannot have both floating and fixed-bottom components")

        # Water depth check
        if "water_depth" in self.wt_init["environment"]:
            if self.wt_init["environment"]["water_depth"] <= 0.0 and flags["offshore"]:
                raise ValueError("Water depth must be > 0 to do fixed-bottom or floating analysis")

    def set_openmdao_vectors(self):
        # Class instance to determine all the parameters used to initialize the openmdao arrays, i.e. number of airfoils, number of angles of attack, number of blade spanwise stations, etc
        # ==modeling_options = {}

        # Materials
        self.modeling_options["materials"] = {}
        self.modeling_options["materials"]["n_mat"] = len(self.wt_init["materials"])

        # Airfoils
        if self.modeling_options["flags"]["airfoils"]:
            self.modeling_options["WISDEM"]["RotorSE"]["n_af"] = len(self.wt_init["airfoils"])
            self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] = self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"]
            if self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0 == int(
                self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0
            ):
                # One fourth of the angles of attack from -pi to -pi/6, half between -pi/6 to pi/6, and one fourth from pi/6 to pi
                self.modeling_options["WISDEM"]["RotorSE"]["aoa"] = np.unique(
                    np.hstack(
                        [
                            np.linspace(
                                -np.pi, -np.pi / 6.0, int(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0 + 1)
                            ),
                            np.linspace(
                                -np.pi / 6.0,
                                np.pi / 6.0,
                                int(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 2.0),
                            ),
                            np.linspace(
                                np.pi / 6.0, np.pi, int(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"] / 4.0 + 1)
                            ),
                        ]
                    )
                )
            else:
                self.modeling_options["WISDEM"]["RotorSE"]["aoa"] = np.linspace(
                    -np.pi, np.pi, self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"]
                )
                print(
                    "WARNING: If you like a grid of angles of attack more refined between +- 30 deg, please choose a n_aoa in the analysis option input file that is a multiple of 4. The current value of "
                    + str(self.modeling_options["WISDEM"]["RotorSE"]["n_aoa"])
                    + " is not a multiple of 4 and an equally spaced grid is adopted."
                )
            Re_all = []
            self.modeling_options["WISDEM"]["RotorSE"]["AFTabMod"] = 1
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_af"]):
                for j in range(len(self.wt_init["airfoils"][i]["polars"])):
                    Re_all.append(self.wt_init["airfoils"][i]["polars"][j]["re"])
                if len(self.wt_init["airfoils"][i]["polars"]) > 1:
                    self.modeling_options["WISDEM"]["RotorSE"]["AFTabMod"] = 2
            self.modeling_options["WISDEM"]["RotorSE"]["n_Re"] = len(np.unique(Re_all))
            self.modeling_options["WISDEM"]["RotorSE"]["n_tab"] = 1
            self.modeling_options["WISDEM"]["RotorSE"]["n_xy"] = self.modeling_options["WISDEM"]["RotorSE"]["n_xy"]
            self.modeling_options["WISDEM"]["RotorSE"]["af_used"] = self.wt_init["components"]["blade"][
                "outer_shape_bem"
            ]["airfoil_position"]["labels"]

        # Blade
        self.modeling_options["WISDEM"]["RotorSE"]["bjs"] = False
        if self.modeling_options["flags"]["blade"]:
            self.modeling_options["WISDEM"]["RotorSE"]["nd_span"] = np.linspace(
                0.0, 1.0, self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
            )  # Equally spaced non-dimensional spanwise grid
            self.modeling_options["WISDEM"]["RotorSE"]["n_af_span"] = len(
                self.wt_init["components"]["blade"]["outer_shape_bem"]["airfoil_position"]["labels"]
            )  # This is the number of airfoils defined along blade span and it is often different than n_af, which is the number of airfoils defined in the airfoil database
            self.modeling_options["WISDEM"]["RotorSE"]["n_webs"] = len(
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["webs"]
            )
            self.modeling_options["WISDEM"]["RotorSE"]["n_layers"] = len(
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"]
            )
            self.modeling_options["WISDEM"]["RotorSE"]["lofted_output"] = False
            self.modeling_options["WISDEM"]["RotorSE"]["n_freq"] = 10  # Number of blade nat frequencies computed

            self.modeling_options["WISDEM"]["RotorSE"]["layer_name"] = self.modeling_options["WISDEM"]["RotorSE"][
                "n_layers"
            ] * [""]
            self.modeling_options["WISDEM"]["RotorSE"]["layer_mat"] = self.modeling_options["WISDEM"]["RotorSE"][
                "n_layers"
            ] * [""]
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_layers"]):
                self.modeling_options["WISDEM"]["RotorSE"]["layer_name"][i] = self.wt_init["components"]["blade"][
                    "internal_structure_2d_fem"
                ]["layers"][i]["name"]
                self.modeling_options["WISDEM"]["RotorSE"]["layer_mat"][i] = self.wt_init["components"]["blade"][
                    "internal_structure_2d_fem"
                ]["layers"][i]["material"]

            self.modeling_options["WISDEM"]["RotorSE"]["web_name"] = self.modeling_options["WISDEM"]["RotorSE"][
                "n_webs"
            ] * [""]
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_webs"]):
                self.modeling_options["WISDEM"]["RotorSE"]["web_name"][i] = self.wt_init["components"]["blade"][
                    "internal_structure_2d_fem"
                ]["webs"][i]["name"]

            # Distributed aerodynamic control devices along blade
            self.modeling_options["WISDEM"]["RotorSE"]["n_te_flaps"] = 0
            if "aerodynamic_control" in self.wt_init["components"]["blade"]:
                if "te_flaps" in self.wt_init["components"]["blade"]["aerodynamic_control"]:
                    self.modeling_options["WISDEM"]["RotorSE"]["n_te_flaps"] = len(
                        self.wt_init["components"]["blade"]["aerodynamic_control"]["te_flaps"]
                    )
                    self.modeling_options["WISDEM"]["RotorSE"]["n_tab"] = 3
                else:
                    raise RuntimeError(
                        "A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem."
                    )

            joint_pos = self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["joint"]["position"]
            if joint_pos > 0.0:
                self.modeling_options["WISDEM"]["RotorSE"]["bjs"] = True
                # Adjust grid to have grid point at join location
                closest_grid_pt = np.argmin(abs(self.modeling_options["WISDEM"]["RotorSE"]["nd_span"] - joint_pos))
                self.modeling_options["WISDEM"]["RotorSE"]["nd_span"][closest_grid_pt] = joint_pos
                self.modeling_options["WISDEM"]["RotorSE"]["id_joint_position"] = closest_grid_pt

        # Drivetrain
        if self.modeling_options["flags"]["nacelle"]:
            self.modeling_options["WISDEM"]["DriveSE"]["direct"] = self.wt_init["assembly"]["drivetrain"].lower() in [
                "direct",
                "direct_drive",
                "pm_direct_drive",
            ]

        # Tower
        if self.modeling_options["flags"]["tower"]:
            self.modeling_options["WISDEM"]["TowerSE"]["n_height"] = len(
                self.wt_init["components"]["tower"]["outer_shape_bem"]["outer_diameter"]["grid"]
            )
            self.modeling_options["WISDEM"]["TowerSE"]["n_layers"] = len(
                self.wt_init["components"]["tower"]["internal_structure_2d_fem"]["layers"]
            )
            self.modeling_options["WISDEM"]["TowerSE"]["n_height_tower"] = self.modeling_options["WISDEM"]["TowerSE"][
                "n_height"
            ]
            self.modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"] = self.modeling_options["WISDEM"]["TowerSE"][
                "n_layers"
            ]

        # Monopile
        if self.modeling_options["flags"]["monopile"]:
            monopile = self.wt_init["components"]["monopile"]
            svec = np.unique(
                np.r_[
                    monopile["outer_shape_bem"]["outer_diameter"]["grid"],
                    monopile["outer_shape_bem"]["reference_axis"]["x"]["grid"],
                    monopile["outer_shape_bem"]["reference_axis"]["y"]["grid"],
                    monopile["outer_shape_bem"]["reference_axis"]["z"]["grid"],
                ]
            )
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = len(svec)
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = len(
                self.wt_init["components"]["monopile"]["internal_structure_2d_fem"]["layers"]
            )
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_height_monopile"] = self.modeling_options["WISDEM"][
                "FixedBottomSE"
            ]["n_height"]
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers_monopile"] = self.modeling_options["WISDEM"][
                "FixedBottomSE"
            ]["n_layers"]

        # Jacket
        if self.modeling_options["flags"]["jacket"]:
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_legs"] = self.wt_init["components"]["jacket"]["n_legs"]
            self.modeling_options["WISDEM"]["FixedBottomSE"]["n_bays"] = self.wt_init["components"]["jacket"]["n_bays"]
            self.modeling_options["WISDEM"]["FixedBottomSE"]["mud_brace"] = self.wt_init["components"]["jacket"]["x_mb"]
            self.modeling_options["WISDEM"]["FixedBottomSE"]["material"] = self.wt_init["components"]["jacket"][
                "material"
            ]

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

            # Create name->index dictionary for joint names, will add on axial joints later
            name2idx = dict(zip(self.modeling_options["floating"]["joints"]["name"], range(n_joints)))

            # Check that there is at most one transition joint
            if self.modeling_options["floating"]["joints"]["transition"].count(True) > 1:
                raise ValueError("Can only support one tower on the floating platform for now")
            try:
                itrans = self.modeling_options["floating"]["joints"]["transition"].index(True)
                self.modeling_options["floating"]["transition_joint"] = itrans
            except:
                self.modeling_options["floating"]["transition_joint"] = None

            n_members = len(self.wt_init["components"]["floating_platform"]["members"])
            self.modeling_options["floating"]["members"] = {}
            self.modeling_options["floating"]["members"]["n_members"] = n_members
            self.modeling_options["floating"]["members"]["name"] = [""] * n_members
            self.modeling_options["floating"]["members"]["joint1"] = [""] * n_members
            self.modeling_options["floating"]["members"]["joint2"] = [""] * n_members
            self.modeling_options["floating"]["members"]["outer_shape"] = [""] * n_members
            self.modeling_options["floating"]["members"]["n_height"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_geom"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_layers"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_ballasts"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_bulkheads"] = np.zeros(n_members, dtype=int)
            self.modeling_options["floating"]["members"]["n_axial_joints"] = np.zeros(n_members, dtype=int)
            ballast_types = []
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

                # Master grid for all bulkheads, internal joints, ballasts, geometry changes, etc
                member_shape = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"]["shape"]
                if member_shape == "circular":
                    grid = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"]["outer_diameter"][
                        "grid"
                    ][:]
                elif member_shape == "rectangular":
                    grid = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"]["side_length_a"][
                            "grid"
                        ][:]
                    grid_b = self.wt_init["components"]["floating_platform"]["members"][i]["outer_shape"]["side_length_b"][
                            "grid"
                        ][:]
                    assert grid == grid_b, "Side length a and b don't have the same grid but they should."
                    
                # Grid for just diameter / thickness design
                geom_grid = grid[:]

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

                # Add in bulkheads and enforce at least endcaps for submerged environment
                # Don't add to master grid as they are handled differently in FloatingSE
                if "bulkhead" in self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]:
                    bulkgrid = self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                        "bulkhead"
                    ]["thickness"]["grid"]
                    if not 0.0 in bulkgrid:
                        self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                            "thickness"
                        ]["grid"].append(0.0)
                        self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                            "thickness"
                        ]["values"].append(0.02)
                    if not 1.0 in bulkgrid:
                        self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                            "thickness"
                        ]["grid"].append(1.0)
                        self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                            "thickness"
                        ]["values"].append(0.02)
                    # grid += bulkgrid # Handled differently in the floating code
                else:
                    self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"] = {}
                    self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                        "material"
                    ] = self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["layers"][
                        0
                    ][
                        "material"
                    ]
                    self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                        "thickness"
                    ] = {}
                    self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                        "thickness"
                    ]["grid"] = [0.0, 1.0]
                    self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                        "thickness"
                    ]["values"] = [0.02, 0.02]

                n_bulk = len(
                    self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"]["bulkhead"][
                        "thickness"
                    ]["grid"]
                )
                self.modeling_options["floating"]["members"]["n_bulkheads"][i] = n_bulk

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
                    grid += self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                        "layers"
                    ][j]["thickness"]["grid"]
                    geom_grid += self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                        "layers"
                    ][j]["thickness"]["grid"]

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
                        ballast_types.append(
                            self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                                "ballasts"
                            ][k]["material"]
                        )
                        self.modeling_options["floating"]["members"][
                            "ballast_mat_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ][k] = ballast_types[-1]
                    else:
                        ballast_types.append("variable")

                    grid += self.wt_init["components"]["floating_platform"]["members"][i]["internal_structure"][
                        "ballasts"
                    ][k]["grid"]

                if "axial_joints" in self.wt_init["components"]["floating_platform"]["members"][i]:
                    n_axial_joints = len(self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"])
                    self.modeling_options["floating"]["members"]["n_axial_joints"][i] = n_axial_joints
                    self.modeling_options["floating"]["members"][
                        "axial_joint_name_member_" + self.modeling_options["floating"]["members"]["name"][i]
                    ] = [""] * n_axial_joints
                    for m in range(n_axial_joints):
                        self.modeling_options["floating"]["members"][
                            "axial_joint_name_member_" + self.modeling_options["floating"]["members"]["name"][i]
                        ][m] = self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m]["name"]
                        grid.append(
                            self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m]["grid"]
                        )
                        name2idx[
                            self.wt_init["components"]["floating_platform"]["members"][i]["axial_joints"][m]["name"]
                        ] = len(name2idx)
                else:
                    self.modeling_options["floating"]["members"]["n_axial_joints"][i] = 0

                final_grid = np.unique(grid)
                final_geom_grid = np.unique(geom_grid)
                self.modeling_options["floating"]["members"][
                    "grid_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = final_grid
                self.modeling_options["floating"]["members"][
                    "geom_member_" + self.modeling_options["floating"]["members"]["name"][i]
                ] = final_geom_grid
                self.modeling_options["floating"]["members"]["n_height"][i] = len(final_grid)
                self.modeling_options["floating"]["members"]["n_geom"][i] = len(final_geom_grid)

            self.modeling_options["floating"]["members"]["ballast_types"] = set(ballast_types)

            # Store joint info
            self.modeling_options["floating"]["joints"]["name2idx"] = name2idx

            # Floating tower params
            self.modeling_options["floating"]["tower"] = {}
            self.modeling_options["floating"]["tower"]["n_ballasts"] = [0]
            self.modeling_options["floating"]["tower"]["n_bulkheads"] = [0]
            self.modeling_options["floating"]["tower"]["n_axial_joints"] = [0]
            if self.modeling_options["flags"]["tower"]:
                self.modeling_options["floating"]["tower"]["n_height"] = [
                    self.modeling_options["WISDEM"]["TowerSE"]["n_height_tower"]
                ]
                self.modeling_options["floating"]["tower"]["n_layers"] = [
                    self.modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"]
                ]
            else:
                self.modeling_options["floating"]["tower"]["n_height"] = [0]
                self.modeling_options["floating"]["tower"]["n_layers"] = [0]

        # Mooring
        self.modeling_options["mooring"] = {}
        if self.modeling_options["flags"]["mooring"]:
            n_nodes = len(self.wt_init["components"]["mooring"]["nodes"])
            n_lines = len(self.wt_init["components"]["mooring"]["lines"])
            n_line_types = len(self.wt_init["components"]["mooring"]["line_types"])
            n_anchor_types = len(self.wt_init["components"]["mooring"]["anchor_types"])
            self.modeling_options["mooring"]["symmetric"] = self.modeling_options["WISDEM"]["FloatingSE"][
                "symmetric_moorings"
            ]
            self.modeling_options["mooring"]["n_nodes"] = n_nodes
            self.modeling_options["mooring"]["n_lines"] = n_lines
            self.modeling_options["mooring"]["n_anchors"] = n_lines
            self.modeling_options["mooring"]["n_line_types"] = n_line_types
            self.modeling_options["mooring"]["n_anchor_types"] = n_anchor_types
            self.modeling_options["mooring"]["node_type"] = [""] * n_nodes
            self.modeling_options["mooring"]["node_names"] = [""] * n_nodes
            self.modeling_options["mooring"]["anchor_type"] = [""] * n_nodes
            self.modeling_options["mooring"]["fairlead_type"] = [""] * n_nodes
            for i in range(n_nodes):
                self.modeling_options["mooring"]["node_type"][i] = self.wt_init["components"]["mooring"]["nodes"][i][
                    "node_type"
                ]
                self.modeling_options["mooring"]["node_names"][i] = self.wt_init["components"]["mooring"]["nodes"][i][
                    "name"
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
            self.modeling_options["mooring"]["line_material"] = [""] * n_lines
            self.modeling_options["mooring"]["line_anchor"] = [""] * n_lines
            fairlead_nodes = []
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
                # For the vessel attachments, find the list of fairlead nodes on the structure
                node1id = self.modeling_options["mooring"]["node_names"].index(
                    self.modeling_options["mooring"]["node1"][i]
                )
                node2id = self.modeling_options["mooring"]["node_names"].index(
                    self.modeling_options["mooring"]["node2"][i]
                )
                if self.modeling_options["mooring"]["node_type"][node1id] == "vessel":
                    fairlead_nodes.append(self.wt_init["components"]["mooring"]["nodes"][node1id]["joint"])
                if self.modeling_options["mooring"]["node_type"][node2id] == "vessel":
                    fairlead_nodes.append(self.wt_init["components"]["mooring"]["nodes"][node2id]["joint"])
                # Store the anchor type names to start
                if "fix" in self.modeling_options["mooring"]["node_type"][node1id]:
                    self.modeling_options["mooring"]["line_anchor"][i] = self.modeling_options["mooring"][
                        "anchor_type"
                    ][node1id]
                if "fix" in self.modeling_options["mooring"]["node_type"][node2id]:
                    self.modeling_options["mooring"]["line_anchor"][i] = self.modeling_options["mooring"][
                        "anchor_type"
                    ][node2id]

            self.modeling_options["mooring"]["line_type_name"] = [""] * n_line_types
            self.modeling_options["mooring"]["line_type_type"] = [""] * n_line_types
            for i in range(n_line_types):
                self.modeling_options["mooring"]["line_type_name"][i] = self.wt_init["components"]["mooring"][
                    "line_types"
                ][i]["name"]
                self.modeling_options["mooring"]["line_type_type"][i] = self.wt_init["components"]["mooring"][
                    "line_types"
                ][i]["type"].lower()
                for j in range(n_lines):
                    if (
                        self.modeling_options["mooring"]["line_type"][j]
                        == self.modeling_options["mooring"]["line_type_name"][i]
                    ):
                        self.modeling_options["mooring"]["line_material"][j] = self.modeling_options["mooring"][
                            "line_type_type"
                        ][i]
            self.modeling_options["mooring"]["anchor_type_name"] = [""] * n_anchor_types
            self.modeling_options["mooring"]["anchor_type_type"] = [""] * n_anchor_types
            for i in range(n_anchor_types):
                self.modeling_options["mooring"]["anchor_type_name"][i] = self.wt_init["components"]["mooring"][
                    "anchor_types"
                ][i]["name"]
                self.modeling_options["mooring"]["anchor_type_type"][i] = self.wt_init["components"]["mooring"][
                    "anchor_types"
                ][i]["type"].lower()
                for j in range(n_lines):
                    if (
                        self.modeling_options["mooring"]["line_anchor"][j]
                        == self.modeling_options["mooring"]["anchor_type_name"][i]
                    ):
                        self.modeling_options["mooring"]["line_anchor"][j] = self.modeling_options["mooring"][
                            "anchor_type_type"
                        ][i]
            self.modeling_options["mooring"]["n_attach"] = len(set(fairlead_nodes))

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
        if "flag" in self.analysis_options["driver"]["optimization"]:
            self.analysis_options["opt_flag"] = self.analysis_options["driver"]["optimization"]["flag"]
        else:
            self.analysis_options["opt_flag"] = recursive_flag(self.analysis_options["design_variables"])

        if self.analysis_options["opt_flag"] == False and (
            self.analysis_options["driver"]["step_size_study"]["flag"] == True or 
            self.analysis_options["driver"]["design_of_experiments"]["flag"] == True
            ):
            self.analysis_options["opt_flag"] = True

        # Blade design variables
        # If not an optimization DV, then the number of points should be same as the discretization
        blade_opt_options = self.analysis_options["design_variables"]["blade"]
        # Blade aero design variables
        if (
            not blade_opt_options["aero_shape"]["twist"]["flag"]
            and not blade_opt_options["aero_shape"]["twist"]["inverse"]
        ):
            blade_opt_options["aero_shape"]["twist"]["n_opt"] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["twist"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError("you are attempting to do an analysis using fewer analysis points than control points.")
        elif blade_opt_options["aero_shape"]["twist"]["n_opt"] < 4:
            raise ValueError("Cannot optimize twist with less than 4 control points along blade span")
        elif blade_opt_options["aero_shape"]["twist"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError(
                """Please set WISDEM->RotorSE->n_span in the modeling options yaml larger
                than aero_shape->twist->n_opt in the analysis options yaml. n_span and twist n_opt are """,
                self.modeling_options["WISDEM"]["RotorSE"]["n_span"],
                blade_opt_options["aero_shape"]["twist"]["n_opt"],
            )

        if not blade_opt_options["aero_shape"]["chord"]["flag"]:
            blade_opt_options["aero_shape"]["chord"]["n_opt"] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["chord"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError("you are attempting to do an analysis using fewer analysis points than control points.")
        elif blade_opt_options["aero_shape"]["chord"]["n_opt"] < 4:
            raise ValueError("Cannot optimize chord with less than 4 control points along blade span")
        elif blade_opt_options["aero_shape"]["chord"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError(
                """Please set WISDEM->RotorSE->n_span in the modeling options yaml larger
                than aero_shape->chord->n_opt in the analysis options yaml. n_span and chord n_opt are """,
                self.modeling_options["WISDEM"]["RotorSE"]["n_span"],
                blade_opt_options["aero_shape"]["chord"]["n_opt"],
            )

        if not blade_opt_options["aero_shape"]["rthick"]["flag"]:
            blade_opt_options["aero_shape"]["rthick"]["n_opt"] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["rthick"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError("you are attempting to do an analysis using fewer analysis points than control points.")
        elif blade_opt_options["aero_shape"]["rthick"]["n_opt"] < 4:
            raise ValueError("Cannot optimize rthick with less than 4 control points along blade span")
        elif blade_opt_options["aero_shape"]["rthick"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError(
                """Please set WISDEM->RotorSE->n_span in the modeling options yaml larger
                than aero_shape->rthick->n_opt in the analysis options yaml. n_span and rthick n_opt are """,
                self.modeling_options["WISDEM"]["RotorSE"]["n_span"],
                blade_opt_options["aero_shape"]["rthick"]["n_opt"],
            )

        if not blade_opt_options["aero_shape"]["L/D"]["flag"]:
            blade_opt_options["aero_shape"]["L/D"]["n_opt"] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
        elif blade_opt_options["aero_shape"]["L/D"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError("you are attempting to do an analysis using fewer analysis points than control points.")
        elif blade_opt_options["aero_shape"]["L/D"]["n_opt"] < 4:
            raise ValueError("Cannot optimize L/D with less than 4 control points along blade span")
        elif blade_opt_options["aero_shape"]["L/D"]["n_opt"] > self.modeling_options["WISDEM"]["RotorSE"]["n_span"]:
            raise ValueError(
                """Please set WISDEM->RotorSE->n_span in the modeling options yaml larger
                than aero_shape->L/D->n_opt in the analysis options yaml. n_span and L/D n_opt are """,
                self.modeling_options["WISDEM"]["RotorSE"]["n_span"],
                blade_opt_options["aero_shape"]["L/D"]["n_opt"],
            )
        # # Blade structural design variables
        if self.modeling_options["WISDEM"]["RotorSE"]["flag"] and self.modeling_options["flags"]["blade"]:
            n_layers = self.modeling_options["WISDEM"]["RotorSE"]["n_layers"]
            layer_name = self.modeling_options["WISDEM"]["RotorSE"]["layer_name"]
            spars_tereinf = np.zeros(4, dtype=int)
            blade_opt_options["n_opt_struct"] = np.ones(n_layers, dtype=int)
            if "structure" in blade_opt_options:
                n_layers_opt = len(blade_opt_options["structure"])
                blade_opt_options["layer_index_opt"] = np.ones(n_layers_opt, dtype=int)
                for i in range(n_layers):
                    foundit = False
                    for j in range(n_layers_opt):
                        if blade_opt_options["structure"][j]["layer_name"] == layer_name[i]:
                            blade_opt_options["n_opt_struct"][i] = blade_opt_options["structure"][j]["n_opt"]
                            blade_opt_options["layer_index_opt"][j] = i
                            foundit = True
                            break
                    if not foundit:
                        blade_opt_options["n_opt_struct"][i] = self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
                    if layer_name[i] == self.modeling_options["WISDEM"]["RotorSE"]["spar_cap_ss"]:
                        spars_tereinf[0] = i
                    if layer_name[i] == self.modeling_options["WISDEM"]["RotorSE"]["spar_cap_ps"]:
                        spars_tereinf[1] = i
                    if layer_name[i] == self.modeling_options["WISDEM"]["RotorSE"]["te_ss"]:
                        spars_tereinf[2] = i
                    if layer_name[i] == self.modeling_options["WISDEM"]["RotorSE"]["te_ps"]:
                        spars_tereinf[3] = i
            else:
                blade_opt_options["structure"] = []
                blade_opt_options["n_opt_struct"] *= self.modeling_options["WISDEM"]["RotorSE"]["n_span"]
            self.modeling_options["WISDEM"]["RotorSE"]["spars_tereinf"] = spars_tereinf
            
            if any(blade_opt_options["n_opt_struct"]>self.modeling_options["WISDEM"]["RotorSE"]["n_span"]):
                raise ValueError("You are attempting to do a blade structural design optimization with more DVs than spanwise stations.")
            blade_opt_options["n_opt_struct"] = blade_opt_options["n_opt_struct"].tolist()
            if "layer_index_opt" in blade_opt_options:
                blade_opt_options["layer_index_opt"] = blade_opt_options["layer_index_opt"].tolist()

        # Handle linked joints and members in floating platform
        if self.modeling_options["flags"]["floating"]:
            float_opt_options = self.analysis_options["design_variables"]["floating"]

            # First the joints
            dv_info = []
            for c in ["z", "r"]:
                for idv in float_opt_options["joints"][c + "_coordinate"]:
                    inames = idv["names"]
                    idx = [self.modeling_options["floating"]["joints"]["name"].index(m) for m in inames]

                    idict = {}
                    idict["indices"] = idx
                    idict["dimension"] = 0 if c == "r" else 2
                    dv_info.append(idict)

            # Check for r-coordinate dv and cylindrical consistency
            for idict in dv_info:
                if idict["dimension"] != 0:
                    continue
                for k in idict["indices"]:
                    m = self.modeling_options["floating"]["joints"]["name"][k]
                    if not self.modeling_options["floating"]["joints"]["cylindrical"][k]:
                        raise ValueError(f"Cannot optimize r-coordinate of, {m}, becase it is not a cylindrical joint")

            # Store DV information for needed linking and IVC assignment
            self.modeling_options["floating"]["joints"]["design_variable_data"] = dv_info

            # Now the members
            memgrps = [[m] for m in self.modeling_options["floating"]["members"]["name"]]
            for idv in float_opt_options["members"]["groups"]:
                inames = idv["names"]
                idx = [self.modeling_options["floating"]["members"]["name"].index(m) for m in inames]
                for k in range(1, len(idx)):
                    try:
                        memgrps[idx[k]].remove(inames[k])
                        memgrps[idx[0]].append(inames[k])
                    except ValueError:
                        raise ValueError("Cannot put member," + inames[k] + ", as part of multiple groups")

            # Remove entries for members that are now linked with others
            while [] in memgrps:
                memgrps.remove([])
            self.modeling_options["floating"]["members"]["linked_members"] = memgrps

            # Make a name 2 group index lookup
            name2grp = {}
            for k, kgrp in enumerate(memgrps):
                for kname in kgrp:
                    name2grp[kname] = k
            self.modeling_options["floating"]["members"]["name2idx"] = name2grp

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
            self.wt_init["components"]["blade"]["outer_shape_bem"]["twist"]["values"] = wt_opt["rotorse.theta"].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["pitch_axis"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["pitch_axis"]["values"] = wt_opt[
                "blade.outer_shape_bem.pitch_axis"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["rthick"] = {}
            self.wt_init["components"]["blade"]["outer_shape_bem"]["rthick"]["grid"] = wt_opt[
                "blade.outer_shape_bem.s"
            ].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["rthick"]["values"] = wt_opt[
                "blade.interp_airfoils.r_thick_interp"
            ].tolist()
            if self.modeling_options["WISDEM"]["RotorSE"]["inn_af"]:
                self.wt_init["components"]["blade"]["outer_shape_bem"]["L/D"]["grid"] = wt_opt[
                    "blade.outer_shape_bem.s"
                ].tolist()
                self.wt_init["components"]["blade"]["outer_shape_bem"]["L/D"]["values"] = wt_opt[
                    "rotorse.rp.powercurve.L_D"
                ].tolist()
                self.wt_init["components"]["blade"]["outer_shape_bem"]["c_d"]["grid"] = wt_opt[
                    "blade.outer_shape_bem.s"
                ].tolist()
                self.wt_init["components"]["blade"]["outer_shape_bem"]["c_d"]["values"] = wt_opt[
                    "rotorse.rp.powercurve.cd_regII"
                ].tolist()
                self.wt_init["components"]["blade"]["outer_shape_bem"]["stall_margin"]["grid"] = wt_opt[
                    "blade.outer_shape_bem.s"
                ].tolist()
                stall_margin = np.deg2rad(
                    wt_opt["rotorse.stall_check.stall_angle_along_span"] - wt_opt["rotorse.stall_check.aoa_along_span"]
                )
                self.wt_init["components"]["blade"]["outer_shape_bem"]["stall_margin"]["values"] = stall_margin.tolist()
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
                "blade.high_level_blade_props.blade_ref_axis"
            ][:, 0].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["y"]["values"] = wt_opt[
                "blade.high_level_blade_props.blade_ref_axis"
            ][:, 1].tolist()
            self.wt_init["components"]["blade"]["outer_shape_bem"]["reference_axis"]["z"]["values"] = wt_opt[
                "blade.high_level_blade_props.blade_ref_axis"
            ][:, 2].tolist()

            # Update blade structure
            # Reference axis from blade outer shape
            self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["reference_axis"] = self.wt_init[
                "components"
            ]["blade"]["outer_shape_bem"]["reference_axis"]
            # Webs positions
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_webs"]):
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
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_layers"]):
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "grid"
                ] = wt_opt["blade.internal_structure_2d_fem.s"].tolist()
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "values"
                ] = wt_opt["blade.ps.layer_thickness_param"][i, :].tolist()
                if wt_opt["blade.internal_structure_2d_fem.definition_layer"][i] != 10:
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

                    # Check for start and end nd layers outside the 0 to 1 range
                    for j in range(len(self.wt_init["components"]["blade"]["internal_structure_2d_fem"][
                        "layers"][i]["start_nd_arc"]["grid"])):
                        if self.wt_init["components"]["blade"]["internal_structure_2d_fem"][
                                "layers"][i]["start_nd_arc"]["values"][j] < 0.:
                            self.wt_init["components"]["blade"]["internal_structure_2d_fem"][
                                "layers"][i]["start_nd_arc"]["values"][j] = 0.
                        if self.wt_init["components"]["blade"]["internal_structure_2d_fem"][
                                "layers"][i]["end_nd_arc"]["values"][j] > 1.:
                            self.wt_init["components"]["blade"]["internal_structure_2d_fem"][
                                "layers"][i]["end_nd_arc"]["values"][j] = 1.

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

            # TODO assign joint mass to wt_init from rs.bjs
            # Elastic properties of the blade
            if self.modeling_options["WISDEM"]["RotorSE"]["bjs"]:
                self.wt_init["components"]["blade"]["internal_structure_2d_fem"]["joint"]["mass"] = float(
                    wt_opt["rotorse.rs.bjs.joint_mass"]
                )

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
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_span"]):
                Ki = np.zeros(21)
                Ki[11] = wt_opt["rotorse.EA"][i]
                Ki[15] = wt_opt["rotorse.EIxx"][i]
                Ki[18] = wt_opt["rotorse.EIyy"][i]
                Ki[20] = wt_opt["rotorse.GJ"][i]
                K.append(Ki.tolist())
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["stiff_matrix"]["values"] = K
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["inertia_matrix"] = {}
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["inertia_matrix"][
                "grid"
            ] = wt_opt["blade.outer_shape_bem.s"].tolist()
            I = []
            for i in range(self.modeling_options["WISDEM"]["RotorSE"]["n_span"]):
                Ii = np.zeros(21)
                Ii[0] = wt_opt["rotorse.rhoA"][i]
                Ii[5] = -wt_opt["rotorse.rhoA"][i] * wt_opt["rotorse.re.y_cg"][i]
                Ii[6] = wt_opt["rotorse.rhoA"][i]
                Ii[10] = wt_opt["rotorse.rhoA"][i] * wt_opt["rotorse.re.x_cg"][i]
                Ii[11] = wt_opt["rotorse.rhoA"][i]
                Ii[12] = wt_opt["rotorse.rhoA"][i] * wt_opt["rotorse.re.y_cg"][i]
                Ii[13] = -wt_opt["rotorse.rhoA"][i] * wt_opt["rotorse.re.x_cg"][i]
                Ii[15] = wt_opt["rotorse.re.precomp.edge_iner"][i]
                # Ii[16] = wt_opt["rotorse.re.precomp.edge_iner"][i]
                Ii[18] = wt_opt["rotorse.re.precomp.flap_iner"][i]
                Ii[20] = wt_opt["rotorse.rhoJ"][i]
                I.append(Ii.tolist())
            self.wt_init["components"]["blade"]["elastic_properties_mb"]["six_x_six"]["inertia_matrix"]["values"] = I

        # Update hub
        if self.modeling_options["flags"]["hub"]:
            # Update hub
            self.wt_init["components"]["hub"]["cone_angle"] = float(wt_opt["hub.cone"][0])
            self.wt_init["components"]["hub"]["flange_t2shell_t"] = float(wt_opt["hub.flange_t2shell_t"][0])
            self.wt_init["components"]["hub"]["flange_OD2hub_D"] = float(wt_opt["hub.flange_OD2hub_D"][0])
            self.wt_init["components"]["hub"]["flange_ID2OD"] = float(wt_opt["hub.flange_ID2flange_OD"][0])
            self.wt_init["components"]["hub"]["hub_blade_spacing_margin"] = float(wt_opt["hub.hub_in2out_circ"][0])
            self.wt_init["components"]["hub"]["hub_stress_concentration"] = float(
                wt_opt["hub.hub_stress_concentration"][0]
            )
            self.wt_init["components"]["hub"]["n_front_brackets"] = int(wt_opt["hub.n_front_brackets"])
            self.wt_init["components"]["hub"]["n_rear_brackets"] = int(wt_opt["hub.n_rear_brackets"])
            self.wt_init["components"]["hub"]["clearance_hub_spinner"] = float(wt_opt["hub.clearance_hub_spinner"][0])
            self.wt_init["components"]["hub"]["spin_hole_incr"] = float(wt_opt["hub.spin_hole_incr"][0])
            self.wt_init["components"]["hub"]["pitch_system_scaling_factor"] = float(
                wt_opt["hub.pitch_system_scaling_factor"][0]
            )
            self.wt_init["components"]["hub"]["elastic_properties_mb"]['system_mass'] = float(wt_opt["drivese.hub_system_mass"][0])
            self.wt_init["components"]["hub"]["elastic_properties_mb"]['system_inertia'] = wt_opt["drivese.hub_system_I"].tolist()
            # self.wt_init["components"]["hub"]["elastic_properties_mb"]['system_center_mass'] = wt_opt["drivese.hub_system_cm"].tolist()

        # Update nacelle
        if self.modeling_options["flags"]["nacelle"]:
            # Common direct and geared
            self.wt_init["components"]["nacelle"]["drivetrain"]["uptilt"] = float(wt_opt["nacelle.uptilt"][0])
            self.wt_init["components"]["nacelle"]["drivetrain"]["distance_tt_hub"] = float(
                wt_opt["nacelle.distance_tt_hub"][0]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["overhang"] = float(wt_opt["nacelle.overhang"][0])
            self.wt_init["components"]["nacelle"]["drivetrain"]["distance_hub_mb"] = float(
                wt_opt["nacelle.distance_hub_mb"][0]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["distance_mb_mb"] = float(
                wt_opt["nacelle.distance_mb_mb"][0]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["generator_length"] = float(
                wt_opt["nacelle.L_generator"][0]
            )
            if not self.modeling_options["flags"]["generator"]:
                self.wt_init["components"]["nacelle"]["drivetrain"]["generator_mass_user"] = float(
                    wt_opt["generator.generator_mass_user"][0]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["generator_radius_user"] = float(
                    wt_opt["generator.generator_radius_user"][0]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["generator_rpm_efficiency_user"]["grid"] = wt_opt[
                    "generator.generator_efficiency_user"
                ][:, 0].tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["generator_rpm_efficiency_user"]["values"] = wt_opt[
                    "generator.generator_efficiency_user"
                ][:, 1].tolist()
            s_lss = np.linspace(0.0, 1.0, len(wt_opt["nacelle.lss_diameter"])).tolist()
            self.wt_init["components"]["nacelle"]["drivetrain"]["lss_diameter"] = wt_opt[
                "nacelle.lss_diameter"
            ].tolist()
            self.wt_init["components"]["nacelle"]["drivetrain"]["lss_wall_thickness"] = wt_opt[
                "nacelle.lss_wall_thickness"
            ].tolist()
            self.wt_init["components"]["nacelle"]["drivetrain"]["gear_ratio"] = float(wt_opt["nacelle.gear_ratio"][0])
            self.wt_init["components"]["nacelle"]["drivetrain"]["gearbox_efficiency"] = float(
                wt_opt["nacelle.gearbox_efficiency"][0]
            )
            self.wt_init["components"]["nacelle"]["drivetrain"]["mb1Type"] = wt_opt["nacelle.mb1Type"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["mb2Type"] = wt_opt["nacelle.mb2Type"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["uptower"] = wt_opt["nacelle.uptower"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["lss_material"] = wt_opt["nacelle.lss_material"]
            self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_material"] = wt_opt[
                "nacelle.bedplate_material"
            ]
            self.wt_init["components"]["nacelle"]["elastic_properties_mb"]["system_mass"] = float(wt_opt["drivese.above_yaw_mass"][0])
            self.wt_init["components"]["nacelle"]["elastic_properties_mb"]["yaw_mass"] = float(wt_opt["drivese.yaw_mass"][0])
            self.wt_init["components"]["nacelle"]["elastic_properties_mb"]["system_inertia"] = wt_opt["drivese.above_yaw_I"].tolist()
            self.wt_init["components"]["nacelle"]["elastic_properties_mb"]["system_inertia_tt"] = wt_opt["drivese.above_yaw_I_TT"].tolist()
            self.wt_init["components"]["nacelle"]["elastic_properties_mb"]["system_center_mass"] = wt_opt["drivese.above_yaw_cm"].tolist()

            if self.modeling_options["WISDEM"]["DriveSE"]["direct"]:
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
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_length"] = float(wt_opt["nacelle.hss_length"][0])
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_diameter"] = wt_opt[
                    "nacelle.hss_diameter"
                ].tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_wall_thickness"] = wt_opt[
                    "nacelle.hss_wall_thickness"
                ].tolist()
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_flange_width"] = float(
                    wt_opt["nacelle.bedplate_flange_width"][0]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_flange_thickness"] = float(
                    wt_opt["nacelle.bedplate_flange_thickness"][0]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["bedplate_web_thickness"] = float(
                    wt_opt["nacelle.bedplate_web_thickness"][0]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["gear_configuration"] = wt_opt[
                    "nacelle.gear_configuration"
                ]
                self.wt_init["components"]["nacelle"]["drivetrain"]["gearbox_torque_density"] = float(
                    wt_opt["drivese.rated_torque"][0]/wt_opt["drivese.gearbox_mass"][0]
                )
                self.wt_init["components"]["nacelle"]["drivetrain"]["planet_numbers"] = wt_opt["nacelle.planet_numbers"]
                self.wt_init["components"]["nacelle"]["drivetrain"]["hss_material"] = wt_opt["nacelle.hss_material"]

        # Update generator
        if self.modeling_options["flags"]["generator"]:
            self.wt_init["components"]["nacelle"]["generator"]["B_r"] = float(wt_opt["generator.B_r"][0])
            self.wt_init["components"]["nacelle"]["generator"]["P_Fe0e"] = float(wt_opt["generator.P_Fe0e"][0])
            self.wt_init["components"]["nacelle"]["generator"]["P_Fe0h"] = float(wt_opt["generator.P_Fe0h"][0])
            self.wt_init["components"]["nacelle"]["generator"]["S_N"] = float(wt_opt["generator.S_N"][0])
            self.wt_init["components"]["nacelle"]["generator"]["alpha_p"] = float(wt_opt["generator.alpha_p"][0])
            self.wt_init["components"]["nacelle"]["generator"]["b_r_tau_r"] = float(wt_opt["generator.b_r_tau_r"][0])
            self.wt_init["components"]["nacelle"]["generator"]["b_ro"] = float(wt_opt["generator.b_ro"][0])
            self.wt_init["components"]["nacelle"]["generator"]["b_s_tau_s"] = float(wt_opt["generator.b_s_tau_s"][0])
            self.wt_init["components"]["nacelle"]["generator"]["b_so"] = float(wt_opt["generator.b_so"][0])
            self.wt_init["components"]["nacelle"]["generator"]["cofi"] = float(wt_opt["generator.cofi"][0])
            self.wt_init["components"]["nacelle"]["generator"]["freq"] = float(wt_opt["generator.freq"][0])
            self.wt_init["components"]["nacelle"]["generator"]["h_i"] = float(wt_opt["generator.h_i"][0])
            self.wt_init["components"]["nacelle"]["generator"]["h_sy0"] = float(wt_opt["generator.h_sy0"][0])
            self.wt_init["components"]["nacelle"]["generator"]["h_w"] = float(wt_opt["generator.h_w"][0])
            self.wt_init["components"]["nacelle"]["generator"]["k_fes"] = float(wt_opt["generator.k_fes"][0])
            self.wt_init["components"]["nacelle"]["generator"]["k_fillr"] = float(wt_opt["generator.k_fillr"][0])
            self.wt_init["components"]["nacelle"]["generator"]["k_fills"] = float(wt_opt["generator.k_fills"][0])
            self.wt_init["components"]["nacelle"]["generator"]["k_s"] = float(wt_opt["generator.k_s"][0])
            self.wt_init["components"]["nacelle"]["generator"]["m"] = wt_opt["generator.m"]
            self.wt_init["components"]["nacelle"]["generator"]["mu_0"] = float(wt_opt["generator.mu_0"][0])
            self.wt_init["components"]["nacelle"]["generator"]["mu_r"] = float(wt_opt["generator.mu_r"][0])
            self.wt_init["components"]["nacelle"]["generator"]["p"] = float(wt_opt["generator.p"][0])
            self.wt_init["components"]["nacelle"]["generator"]["phi"] = float(wt_opt["generator.phi"][0])
            self.wt_init["components"]["nacelle"]["generator"]["q1"] = wt_opt["generator.q1"]
            self.wt_init["components"]["nacelle"]["generator"]["q2"] = wt_opt["generator.q2"]
            self.wt_init["components"]["nacelle"]["generator"]["ratio_mw2pp"] = float(wt_opt["generator.ratio_mw2pp"][0])
            self.wt_init["components"]["nacelle"]["generator"]["resist_Cu"] = float(wt_opt["generator.resist_Cu"][0])
            self.wt_init["components"]["nacelle"]["generator"]["sigma"] = float(wt_opt["generator.sigma"][0])
            self.wt_init["components"]["nacelle"]["generator"]["y_tau_p"] = float(wt_opt["generator.y_tau_p"][0])
            self.wt_init["components"]["nacelle"]["generator"]["y_tau_pr"] = float(wt_opt["generator.y_tau_pr"][0])

            self.wt_init["components"]["nacelle"]["generator"]["I_0"] = float(wt_opt["generator.I_0"][0])
            self.wt_init["components"]["nacelle"]["generator"]["d_r"] = float(wt_opt["generator.d_r"][0])
            self.wt_init["components"]["nacelle"]["generator"]["h_m"] = float(wt_opt["generator.h_m"][0])
            self.wt_init["components"]["nacelle"]["generator"]["h_0"] = float(wt_opt["generator.h_0"][0])
            self.wt_init["components"]["nacelle"]["generator"]["h_s"] = float(wt_opt["generator.h_s"][0])
            self.wt_init["components"]["nacelle"]["generator"]["len_s"] = float(wt_opt["generator.len_s"][0])
            self.wt_init["components"]["nacelle"]["generator"]["n_r"] = float(wt_opt["generator.n_r"][0])
            self.wt_init["components"]["nacelle"]["generator"]["rad_ag"] = float(wt_opt["generator.rad_ag"][0])
            self.wt_init["components"]["nacelle"]["generator"]["t_wr"] = float(wt_opt["generator.t_wr"][0])

            self.wt_init["components"]["nacelle"]["generator"]["n_s"] = float(wt_opt["generator.n_s"][0])
            self.wt_init["components"]["nacelle"]["generator"]["b_st"] = float(wt_opt["generator.b_st"][0])
            self.wt_init["components"]["nacelle"]["generator"]["d_s"] = float(wt_opt["generator.d_s"][0])
            self.wt_init["components"]["nacelle"]["generator"]["t_ws"] = float(wt_opt["generator.t_ws"][0])

            self.wt_init["components"]["nacelle"]["generator"]["rho_Copper"] = float(wt_opt["generator.rho_Copper"][0])
            self.wt_init["components"]["nacelle"]["generator"]["rho_Fe"] = float(wt_opt["generator.rho_Fe"][0])
            self.wt_init["components"]["nacelle"]["generator"]["rho_Fes"] = float(wt_opt["generator.rho_Fes"][0])
            self.wt_init["components"]["nacelle"]["generator"]["rho_PM"] = float(wt_opt["generator.rho_PM"][0])

            self.wt_init["components"]["nacelle"]["generator"]["C_Cu"] = float(wt_opt["generator.C_Cu"][0])
            self.wt_init["components"]["nacelle"]["generator"]["C_Fe"] = float(wt_opt["generator.C_Fe"][0])
            self.wt_init["components"]["nacelle"]["generator"]["C_Fes"] = float(wt_opt["generator.C_Fes"][0])
            self.wt_init["components"]["nacelle"]["generator"]["C_PM"] = float(wt_opt["generator.C_PM"][0])

            if self.modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["pmsg_outer"]:
                self.wt_init["components"]["nacelle"]["generator"]["N_c"] = float(wt_opt["generator.N_c"][0])
                self.wt_init["components"]["nacelle"]["generator"]["b"] = float(wt_opt["generator.b"][0])
                self.wt_init["components"]["nacelle"]["generator"]["c"] = float(wt_opt["generator.c"][0])
                self.wt_init["components"]["nacelle"]["generator"]["E_p"] = float(wt_opt["generator.E_p"][0])
                self.wt_init["components"]["nacelle"]["generator"]["h_yr"] = float(wt_opt["generator.h_yr"][0])
                self.wt_init["components"]["nacelle"]["generator"]["h_ys"] = float(wt_opt["generator.h_ys"][0])
                self.wt_init["components"]["nacelle"]["generator"]["h_sr"] = float(wt_opt["generator.h_sr"][0])
                self.wt_init["components"]["nacelle"]["generator"]["h_ss"] = float(wt_opt["generator.h_ss"][0])
                self.wt_init["components"]["nacelle"]["generator"]["t_r"] = float(wt_opt["generator.t_r"][0])
                self.wt_init["components"]["nacelle"]["generator"]["t_s"] = float(wt_opt["generator.t_s"][0])

                self.wt_init["components"]["nacelle"]["generator"]["u_allow_pcent"] = float(
                    wt_opt["generator.u_allow_pcent"][0]
                )
                self.wt_init["components"]["nacelle"]["generator"]["y_allow_pcent"] = float(
                    wt_opt["generator.y_allow_pcent"][0]
                )
                self.wt_init["components"]["nacelle"]["generator"]["z_allow_deg"] = float(
                    wt_opt["generator.z_allow_deg"][0]
                )
                self.wt_init["components"]["nacelle"]["generator"]["B_tmax"] = float(wt_opt["generator.B_tmax"][0])

            if self.modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["eesg", "pmsg_arms", "pmsg_disc"]:
                self.wt_init["components"]["nacelle"]["generator"]["tau_p"] = float(wt_opt["generator.tau_p"][0])
                self.wt_init["components"]["nacelle"]["generator"]["h_ys"] = float(wt_opt["generator.h_ys"][0])
                self.wt_init["components"]["nacelle"]["generator"]["h_yr"] = float(wt_opt["generator.h_yr"][0])
                self.wt_init["components"]["nacelle"]["generator"]["b_arm"] = float(wt_opt["generator.b_arm"][0])

            elif self.modeling_options["WISDEM"]["GeneratorSE"]["type"] in ["scig", "dfig"]:
                self.wt_init["components"]["nacelle"]["generator"]["B_symax"] = float(wt_opt["generator.B_symax"][0])
                self.wt_init["components"]["nacelle"]["generator"]["S_Nmax"] = float(wt_opt["generator.S_Nmax"][0])

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
                wt_opt["tower.outfitting_factor"][0]
            )
            for i in range(self.modeling_options["WISDEM"]["TowerSE"]["n_layers_tower"]):
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
                wt_opt["monopile.outfitting_factor"][0]
            )
            for i in range(self.modeling_options["WISDEM"]["FixedBottomSE"]["n_layers_monopile"]):
                self.wt_init["components"]["monopile"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "grid"
                ] = wt_opt["monopile.s"].tolist()
                self.wt_init["components"]["monopile"]["internal_structure_2d_fem"]["layers"][i]["thickness"][
                    "values"
                ] = wt_opt["monopile.layer_thickness"][i, :].tolist()

        # Update jacket
        if self.modeling_options["flags"]["jacket"]:
            self.wt_init["components"]["jacket"]["r_head"] = float(wt_opt["jacket.r_head"][0])
            self.wt_init["components"]["jacket"]["r_foot"] = float(
                wt_opt["jacket.r_head"] * wt_opt["jacket.foot_head_ratio"][0]
            )
            self.wt_init["components"]["jacket"]["height"] = float(wt_opt["jacket.height"][0])
            self.wt_init["components"]["jacket"]["leg_diameter"] = float(wt_opt["jacket.leg_diameter"][0])
            self.wt_init["components"]["jacket"]["leg_thickness"] = float(wt_opt["jacket.leg_thickness"][0])
            self.wt_init["components"]["jacket"]["brace_diameters"] = wt_opt["jacket.brace_diameters"].tolist()
            self.wt_init["components"]["jacket"]["brace_thicknesses"] = wt_opt["jacket.brace_thicknesses"].tolist()
            self.wt_init["components"]["jacket"]["bay_spacing"] = wt_opt["jacket.bay_spacing"].tolist()

        # Update floating platform and mooring
        if self.modeling_options["flags"]["floating"]:
            yaml_out = self.wt_init["components"]["floating_platform"]
            n_joints = self.modeling_options["floating"]["joints"]["n_joints"]
            for i in range(n_joints):
                yaml_out["joints"][i]["location"] = wt_opt["floating.location"][i, :].tolist()

            n_members = self.modeling_options["floating"]["members"]["n_members"]
            for i in range(n_members):
                name_member = self.modeling_options["floating"]["members"]["name"][i]
                idx = self.modeling_options["floating"]["members"]["name2idx"][name_member]
                member_shape = yaml_out["members"][i]["outer_shape"]["shape"]

                s_in = wt_opt[f"floating.memgrp{idx}.s_in"].tolist()
                if member_shape == "circular":
                    yaml_out["members"][i]["outer_shape"]["outer_diameter"]["grid"] = s_in
                    d_in = wt_opt[f"floating.memgrp{idx}.outer_diameter_in"]
                    if len(d_in) == len(s_in):
                        yaml_out["members"][i]["outer_shape"]["outer_diameter"]["values"] = d_in.tolist()
                    else:
                        d_in2 = d_in * np.ones(len(s_in))
                        yaml_out["members"][i]["outer_shape"]["outer_diameter"]["values"] = d_in2.tolist()
                elif member_shape == "rectangular":
                    yaml_out["members"][i]["outer_shape"]["side_length_a"]["grid"] = s_in
                    yaml_out["members"][i]["outer_shape"]["side_length_b"]["grid"] = s_in
                    length_a_in = wt_opt[f"floating.memgrp{idx}.side_length_a_in"]
                    length_b_in = wt_opt[f"floating.memgrp{idx}.side_length_b_in"]
                    if len(length_a_in) == len(s_in):
                        yaml_out["members"][i]["outer_shape"]["side_length_a"]["values"] = length_a_in.tolist()
                        yaml_out["members"][i]["outer_shape"]["side_length_b"]["values"] = length_b_in.tolist()
                    else:
                        length_a_in2 = length_a_in * np.ones(len(s_in))
                        length_b_in2 = length_b_in * np.ones(len(s_in))
                        yaml_out["members"][i]["outer_shape"]["side_length_a"]["values"] = length_a_in2.tolist()
                        yaml_out["members"][i]["outer_shape"]["side_length_b"]["values"] = length_b_in2.tolist()

                istruct = yaml_out["members"][i]["internal_structure"]

                n_layers = self.modeling_options["floating"]["members"]["n_layers"][i]
                for j in range(n_layers):
                    istruct["layers"][j]["thickness"]["grid"] = wt_opt[f"floating.memgrp{idx}.s_in"].tolist()
                    istruct["layers"][j]["thickness"]["values"] = wt_opt[f"floating.memgrp{idx}.layer_thickness_in"][
                        j, :
                    ].tolist()

                if "ring_stiffeners" in istruct:
                    istruct["ring_stiffeners"]["web_height"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_web_height"][0]
                    )
                    istruct["ring_stiffeners"]["web_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_web_thickness"][0]
                    )
                    istruct["ring_stiffeners"]["flange_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_flange_thickness"][0]
                    )
                    istruct["ring_stiffeners"]["flange_width"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_flange_width"][0]
                    )
                    istruct["ring_stiffeners"]["spacing"] = float(
                        wt_opt[f"floating.memgrp{idx}.ring_stiffener_spacing"][0]
                    )

                if "longitudinal_stiffeners" in istruct:
                    istruct["longitudinal_stiffeners"]["web_height"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_web_height"][0]
                    )
                    istruct["longitudinal_stiffeners"]["web_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_web_thickness"][0]
                    )
                    istruct["longitudinal_stiffeners"]["flange_thickness"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_flange_thickness"][0]
                    )
                    istruct["longitudinal_stiffeners"]["flange_width"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_flange_width"][0]
                    )
                    istruct["longitudinal_stiffeners"]["spacing"] = float(
                        wt_opt[f"floating.memgrp{idx}.axial_stiffener_spacing"][0]
                    )

                n_ballasts = self.modeling_options["floating"]["members"]["n_ballasts"][i]
                for j in range(n_ballasts):
                    if self.modeling_options["floating"]["members"]["ballast_flag_member_" + name_member][j] == False:
                        istruct["ballasts"][j]["volume"] = float(wt_opt[f"floating.memgrp{idx}.ballast_volume"][j])

                if self.modeling_options["floating"]["members"]["n_axial_joints"][i] > 0:
                    for j in range(self.modeling_options["floating"]["members"]["n_axial_joints"][i]):
                        yaml_out["members"][i]["axial_joints"][j]["grid"] = float(
                            wt_opt[f"floating.memgrp{idx}.grid_axial_joints"][j]
                        )

        if self.modeling_options["flags"]["mooring"]:
            n_lines = self.modeling_options["mooring"]["n_lines"]
            n_line_types = self.modeling_options["mooring"]["n_line_types"]
            line_names = [self.wt_init["components"]["mooring"]["line_types"][i]["name"] for i in range(n_line_types)]
            line_id = [self.wt_init["components"]["mooring"]["lines"][i]["line_type"] for i in range(n_lines)]

            for i in range(n_lines):
                self.wt_init["components"]["mooring"]["lines"][i]["unstretched_length"] = float(
                    wt_opt["mooring.unstretched_length"][i]
                )

            for jj, jname in enumerate(line_id):
                for ii, iname in enumerate(line_names):
                    if jname == iname:
                        self.wt_init["components"]["mooring"]["line_types"][ii]["diameter"] = float(
                            wt_opt["mooring.line_diameter"][jj]
                        )

        # Update rotor nacelle assembly
        if self.modeling_options["flags"]["RNA"]:
            self.wt_init["components"]["RNA"] = {}
            self.wt_init["components"]["RNA"]["elastic_properties_mb"] = {}
            self.wt_init["components"]["RNA"]["elastic_properties_mb"]["mass"] = float(wt_opt["drivese.rna_mass"][0])
            self.wt_init["components"]["RNA"]["elastic_properties_mb"]["inertia"] = wt_opt["drivese.rna_I_TT"].tolist()
            self.wt_init["components"]["RNA"]["elastic_properties_mb"]["center_mass"] = wt_opt[
                "drivese.rna_cm"
            ].tolist()

        # Update rotor diameter and hub height
        if self.modeling_options["flags"]["blade"]:
            self.wt_init["assembly"]["rotor_diameter"] = float(wt_opt["blade.high_level_blade_props.rotor_diameter"][0])
        self.wt_init["assembly"]["hub_height"] = float(wt_opt["high_level_tower_props.hub_height"][0])

        # Update controller
        if self.modeling_options["flags"]["control"]:
            self.wt_init["control"]["torque"]["tsr"] = float(wt_opt["control.rated_TSR"][0])

        # Update cost coefficients
        if self.modeling_options["flags"]["costs"]:
            if float(wt_opt["tcc.blade_mass"][0]) > 0.0:
                self.wt_init["costs"]["blade_mass_cost_coeff"] = float(
                    (wt_opt["tcc.blade_cost"] / wt_opt["tcc.blade_mass"])[0]
                )
            if float(wt_opt["tcc.hub_mass"][0]) > 0.0:
                self.wt_init["costs"]["hub_mass_cost_coeff"] = float((wt_opt["tcc.hub_cost"] / wt_opt["tcc.hub_mass"])[0])
            if float(wt_opt["tcc.pitch_system_mass"][0]) > 0.0:
                self.wt_init["costs"]["pitch_system_mass_cost_coeff"] = float(
                    (wt_opt["tcc.pitch_system_cost"] / wt_opt["tcc.pitch_system_mass"])[0]
                )
            if float(wt_opt["tcc.spinner_mass"][0]) > 0.0:
                self.wt_init["costs"]["spinner_mass_cost_coeff"] = float(
                    (wt_opt["tcc.spinner_cost"] / wt_opt["tcc.spinner_mass"])[0]
                )
            if float(wt_opt["tcc.lss_mass"][0]) > 0.0:
                self.wt_init["costs"]["lss_mass_cost_coeff"] = float((wt_opt["tcc.lss_cost"] / wt_opt["tcc.lss_mass"])[0])
            if float(wt_opt["tcc.main_bearing_mass"][0]) > 0.0:
                self.wt_init["costs"]["bearing_mass_cost_coeff"] = float(
                    (wt_opt["tcc.main_bearing_cost"] / wt_opt["tcc.main_bearing_mass"])[0]
                )
            if self.modeling_options["flags"]["nacelle"]:
                if float(wt_opt["drivese.gearbox_mass"][0]) > 0.:
                    self.wt_init["costs"]["gearbox_torque_cost"] = float(
                        (wt_opt["tcc.gearbox_cost"]/wt_opt["drivese.rated_torque"])[0]*1.e+3
                    )
            if float(wt_opt["tcc.hss_mass"][0]) > 0.0:
                self.wt_init["costs"]["hss_mass_cost_coeff"] = float((wt_opt["tcc.hss_cost"] / wt_opt["tcc.hss_mass"])[0])
            if float(wt_opt["tcc.generator_mass"][0]) > 0.0:
                self.wt_init["costs"]["generator_mass_cost_coeff"] = float(
                    (wt_opt["tcc.generator_cost"] / wt_opt["tcc.generator_mass"])[0]
                )
            if float(wt_opt["tcc.bedplate_mass"][0]) > 0.0:
                self.wt_init["costs"]["bedplate_mass_cost_coeff"] = float(
                    (wt_opt["tcc.bedplate_cost"] / wt_opt["tcc.bedplate_mass"])[0]
                )
            if float(wt_opt["tcc.yaw_mass"][0]) > 0.0:
                self.wt_init["costs"]["yaw_mass_cost_coeff"] = float(
                    (wt_opt["tcc.yaw_system_cost"] / wt_opt["tcc.yaw_mass"])[0]
                )
            if float(wt_opt["tcc.converter_mass"][0]) > 0.0:
                self.wt_init["costs"]["converter_mass_cost_coeff"] = float(
                    (wt_opt["tcc.converter_cost"] / wt_opt["tcc.converter_mass"])[0]
                )
            if float(wt_opt["tcc.transformer_mass"][0]) > 0.0:
                self.wt_init["costs"]["transformer_mass_cost_coeff"] = float(
                    (wt_opt["tcc.transformer_cost"] / wt_opt["tcc.transformer_mass"])[0]
                )
            if float(wt_opt["tcc.hvac_mass"][0]) > 0.0:
                self.wt_init["costs"]["hvac_mass_cost_coeff"] = float((wt_opt["tcc.hvac_cost"] / wt_opt["tcc.hvac_mass"])[0])
            if float(wt_opt["tcc.cover_mass"][0]) > 0.0:
                self.wt_init["costs"]["cover_mass_cost_coeff"] = float(
                    (wt_opt["tcc.cover_cost"] / wt_opt["tcc.cover_mass"])[0]
                )
            if float(wt_opt["tcc.tower_mass"][0]) > 0.0:
                self.wt_init["costs"]["tower_mass_cost_coeff"] = float(
                    (wt_opt["tcc.tower_cost"] / wt_opt["tcc.tower_mass"])[0]
                )

        # Write yamls with updated values
        sch.write_geometry_yaml(self.wt_init, fname_output)

    def write_options(self, fname_output):
        sch.write_modeling_yaml(self.modeling_options, fname_output)
        sch.write_analysis_yaml(self.analysis_options, fname_output)
