"""WISDEM Monopile API"""

__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import openmdao.api as om

from wisdem.orbit import ProjectManager


class Orbit(om.Group):
    def initialize(self):
        self.options.declare("floating", default=False)
        self.options.declare("jacket", default=False)
        self.options.declare("jacket_legs", default=0)

    def setup(self):

        # Define all input variables from all models
        self.set_input_defaults("wtiv", "example_wtiv")
        self.set_input_defaults("feeder", "example_feeder")
        self.set_input_defaults("num_feeders", 1)
        self.set_input_defaults("num_towing", 1)
        self.set_input_defaults("num_station_keeping", 3)
        self.set_input_defaults("oss_install_vessel", "example_heavy_lift_vessel")
        self.set_input_defaults("site_distance", 40.0, units="km")
        self.set_input_defaults("site_distance_to_landfall", 40.0, units="km")
        self.set_input_defaults("interconnection_distance", 40.0, units="km")
        self.set_input_defaults("plant_turbine_spacing", 7)
        self.set_input_defaults("plant_row_spacing", 7)
        self.set_input_defaults("plant_substation_distance", 1, units="km")
        self.set_input_defaults("num_port_cranes", 1)
        self.set_input_defaults("num_assembly_lines", 1)
        self.set_input_defaults("takt_time", 170.0, units="h")
        self.set_input_defaults("port_cost_per_month", 2e6, units="USD/mo")
        self.set_input_defaults("commissioning_pct", 0.01)
        self.set_input_defaults("decommissioning_pct", 0.15)
        self.set_input_defaults("site_auction_price", 100e6, units="USD")
        self.set_input_defaults("site_assessment_plan_cost", 1e6, units="USD")
        self.set_input_defaults("site_assessment_cost", 25e6, units="USD")
        self.set_input_defaults("construction_operations_plan_cost", 2.5e6, units="USD")
        self.set_input_defaults("design_install_plan_cost", 2.5e6, units="USD")
        self.set_input_defaults("boem_review_cost", 0.0, units="USD")

        self.add_subsystem(
            "orbit",
            OrbitWisdem(
                floating=self.options["floating"],
                jacket=self.options["jacket"],
                jacket_legs=self.options["jacket_legs"],
            ),
            promotes=["*"],
        )


class OrbitWisdem(om.ExplicitComponent):
    """ORBIT-WISDEM Fixed Substructure API"""

    def initialize(self):
        self.options.declare("floating", default=False)
        self.options.declare("jacket", default=False)
        self.options.declare("jacket_legs", default=0)

    def setup(self):
        """"""
        # Inputs
        # self.add_discrete_input('weather_file', 'block_island', desc='Weather file to use for installation times.')

        # Vessels
        self.add_discrete_input(
            "wtiv", "example_wtiv", desc="Vessel configuration to use for installation of foundations and turbines."
        )
        self.add_discrete_input(
            "feeder", "future_feeder", desc="Vessel configuration to use for (optional) feeder barges."
        )
        self.add_discrete_input(
            "num_feeders", 1, desc="Number of feeder barges to use for installation of foundations and turbines."
        )
        self.add_discrete_input(
            "num_towing",
            1,
            desc="Number of towing vessels to use for floating platforms that are assembled at port (with or without the turbine).",
        )
        self.add_discrete_input(
            "num_station_keeping",
            3,
            desc="Number of station keeping vessels that attach to floating platforms under tow-out.",
        )
        self.add_discrete_input(
            "oss_install_vessel",
            "example_heavy_lift_vessel",
            desc="Vessel configuration to use for installation of offshore substations.",
        )

        # Site
        self.add_input("site_depth", 40.0, units="m", desc="Site depth.")
        self.add_input("site_distance", 40.0, units="km", desc="Distance from site to installation port.")
        self.add_input(
            "site_distance_to_landfall", 50.0, units="km", desc="Distance from site to landfall for export cable."
        )
        self.add_input("interconnection_distance", 3.0, units="km", desc="Distance from landfall to interconnection.")
        self.add_input("site_mean_windspeed", 9.0, units="m/s", desc="Mean windspeed of the site.")

        # Plant
        self.add_discrete_input("number_of_turbines", 60, desc="Number of turbines.")
        self.add_input("plant_turbine_spacing", 7, desc="Turbine spacing in rotor diameters.")
        self.add_input("plant_row_spacing", 7, desc="Row spacing in rotor diameters. Not used in ring layouts.")
        self.add_input(
            "plant_substation_distance", 1, units="km", desc="Distance from first turbine in string to substation."
        )

        # Turbine
        self.add_input("turbine_rating", 8.0, units="MW", desc="Rated capacity of a turbine.")
        self.add_input("turbine_rated_windspeed", 11.0, units="m/s", desc="Rated windspeed of the turbine.")
        self.add_input("turbine_capex", 1100, units="USD/kW", desc="Turbine CAPEX")
        self.add_input("hub_height", 100.0, units="m", desc="Turbine hub height.")
        self.add_input("turbine_rotor_diameter", 130, units="m", desc="Turbine rotor diameter.")
        self.add_input("tower_mass", 400.0, units="t", desc="mass of the total tower.")
        self.add_input("tower_length", 100.0, units="m", desc="Total length of the tower.")
        self.add_input(
            "tower_deck_space",
            25.0,
            units="m**2",
            desc="Deck space required to transport the tower. Defaults to 0 in order to not be a constraint on installation.",
        )
        self.add_input("nacelle_mass", 500.0, units="t", desc="mass of the rotor nacelle assembly (RNA).")
        self.add_input(
            "nacelle_deck_space",
            25.0,
            units="m**2",
            desc="Deck space required to transport the rotor nacelle assembly (RNA). Defaults to 0 in order to not be a constraint on installation.",
        )
        self.add_discrete_input("number_of_blades", 3, desc="Number of blades per turbine.")
        self.add_input("blade_mass", 50.0, units="t", desc="mass of an individual blade.")
        self.add_input(
            "blade_deck_space",
            100.0,
            units="m**2",
            desc="Deck space required to transport a blade. Defaults to 0 in order to not be a constraint on installation.",
        )

        # Mooring
        self.add_discrete_input("num_mooring_lines", 3, desc="Number of mooring lines per platform.")
        self.add_input("mooring_line_mass", 1e4, units="kg", desc="Total mass of a mooring line")
        self.add_input("mooring_line_diameter", 0.1, units="m", desc="Cross-sectional diameter of a mooring line")
        self.add_input("mooring_line_length", 1e3, units="m", desc="Unstretched mooring line length")
        self.add_input("anchor_mass", 1e4, units="kg", desc="Total mass of an anchor")
        self.add_input("mooring_line_cost", 0.5e6, units="USD", desc="Mooring line unit cost.")
        self.add_input("mooring_anchor_cost", 0.1e6, units="USD", desc="Mooring line unit cost.")
        self.add_discrete_input("anchor_type", "drag_embedment", desc="Number of mooring lines per platform.")

        # Port
        self.add_input("port_cost_per_month", 2e6, units="USD/mo", desc="Monthly port costs.")
        self.add_input(
            "takt_time", 170.0, units="h", desc="Substructure assembly cycle time when doing assembly at the port."
        )
        self.add_discrete_input(
            "num_assembly_lines", 1, desc="Number of assembly lines used when assembly occurs at the port."
        )
        self.add_discrete_input(
            "num_port_cranes",
            1,
            desc="Number of cranes used at the port to load feeders / WTIVS when assembly occurs on-site or assembly cranes when assembling at port.",
        )

        # Floating Substructures
        self.add_input("floating_substructure_cost", 10e6, units="USD", desc="Floating substructure unit cost.")

        # Monopile
        self.add_input("monopile_length", 100.0, units="m", desc="Length of monopile (including pile).")
        self.add_input("monopile_diameter", 7.0, units="m", desc="Diameter of monopile.")
        self.add_input("monopile_mass", 900.0, units="t", desc="mass of an individual monopile.")
        self.add_input("monopile_cost", 4e6, units="USD", desc="Monopile unit cost.")

        # Jacket
        self.add_input("jacket_length", 65.0, units="m", desc="Length/height of jacket (including pile/buckets).")
        self.add_input("jacket_mass", 900.0, units="t", desc="mass of an individual jacket.")
        self.add_input("jacket_cost", 4e6, units="USD", desc="Jacket unit cost.")
        self.add_input("jacket_r_foot", 10.0, units="m", desc="Radius of jacket legs at base from centeroid.")

        # Generic fixed-bottom
        self.add_input("transition_piece_mass", 250.0, units="t", desc="mass of an individual transition piece.")
        self.add_input(
            "transition_piece_deck_space",
            25.0,
            units="m**2",
            desc="Deck space required to transport a transition piece. Defaults to 0 in order to not be a constraint on installation.",
        )
        self.add_input("transition_piece_cost", 1.5e6, units="USD", desc="Transition piece unit cost.")

        # Project
        self.add_input("site_auction_price", 100e6, units="USD", desc="Cost to secure site lease")
        self.add_input(
            "site_assessment_plan_cost", 1e6, units="USD", desc="Cost to do engineering plan for site assessment"
        )
        self.add_input("site_assessment_cost", 25e6, units="USD", desc="Cost to execute site assessment")
        self.add_input("construction_operations_plan_cost", 2.5e6, units="USD", desc="Cost to do construction planning")
        self.add_input(
            "boem_review_cost",
            0.0,
            units="USD",
            desc="Cost for additional review by U.S. Dept of Interior Bureau of Ocean Energy Management (BOEM)",
        )
        self.add_input("design_install_plan_cost", 2.5e6, units="USD", desc="Cost to do installation planning")

        # Other
        self.add_input("commissioning_pct", 0.01, desc="Commissioning percent.")
        self.add_input("decommissioning_pct", 0.15, desc="Decommissioning percent.")

        # Outputs
        # Totals
        self.add_output(
            "bos_capex", 0.0, units="USD", desc="Total BOS CAPEX not including commissioning or decommissioning."
        )
        self.add_output(
            "total_capex", 0.0, units="USD", desc="Total BOS CAPEX including commissioning and decommissioning."
        )
        self.add_output(
            "total_capex_kW", 0.0, units="USD/kW", desc="Total BOS CAPEX including commissioning and decommissioning."
        )
        self.add_output("installation_time", 0.0, units="h", desc="Total balance of system installation time.")
        self.add_output("installation_capex", 0.0, units="USD", desc="Total balance of system installation cost.")

    def compile_orbit_config_file(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """"""

        floating_flag = self.options["floating"]
        jacket_flag = self.options["jacket"]

        config = {
            # Vessels
            "wtiv": "floating_heavy_lift_vessel" if floating_flag else discrete_inputs["wtiv"],
            "array_cable_install_vessel": "example_cable_lay_vessel",
            "array_cable_bury_vessel": "example_cable_lay_vessel",
            "export_cable_install_vessel": "example_cable_lay_vessel",
            "export_cable_bury_vessel": "example_cable_lay_vessel",
            # Site/plant
            "site": {
                "depth": float(inputs["site_depth"]),
                "distance": float(inputs["site_distance"]),
                "distance_to_landfall": float(inputs["site_distance_to_landfall"]),
                "mean_windspeed": float(inputs["site_mean_windspeed"]),
            },
            "landfall": {
                "interconnection_distance": float(inputs["interconnection_distance"]),
            },
            "plant": {
                "layout": "grid",
                "num_turbines": int(discrete_inputs["number_of_turbines"]),
                "row_spacing": float(inputs["plant_row_spacing"]),
                "turbine_spacing": float(inputs["plant_turbine_spacing"]),
                "substation_distance": float(inputs["plant_substation_distance"]),
            },
            # Turbine + components
            "turbine": {
                "hub_height": float(inputs["hub_height"]),
                "rotor_diameter": float(inputs["turbine_rotor_diameter"]),
                "turbine_rating": float(inputs["turbine_rating"]),
                "rated_windspeed": float(inputs["turbine_rated_windspeed"]),
                "tower": {
                    "type": "Tower",
                    "deck_space": float(inputs["tower_deck_space"]),
                    "mass": float(inputs["tower_mass"]),
                    "length": float(inputs["tower_length"]),
                },
                "nacelle": {
                    "type": "Nacelle",
                    "deck_space": float(inputs["nacelle_deck_space"]),
                    "mass": float(inputs["nacelle_mass"]),
                },
                "blade": {
                    "type": "Blade",
                    "number": int(discrete_inputs["number_of_blades"]),
                    "deck_space": float(inputs["blade_deck_space"]),
                    "mass": float(inputs["blade_mass"]),
                },
            },
            # Substructure components
            "transition_piece": {
                "type": "Transition Piece",
                "deck_space": float(inputs["transition_piece_deck_space"]),
                "mass": float(inputs["transition_piece_mass"]),
                "unit_cost": float(inputs["transition_piece_cost"]),
            },
            # Electrical
            "array_system_design": {
                "cables": ["XLPE_630mm_66kV", "XLPE_185mm_66kV"],
            },
            "export_system_design": {
                "cables": "XLPE_1000m_220kV",
                "interconnection_distance": float(inputs["interconnection_distance"]),
                "percent_added_length": 0.1,
            },
            # Phase Specific
            "OffshoreSubstationInstallation": {
                "oss_install_vessel": "floating_heavy_lift_vessel" if floating_flag else "example_heavy_lift_vessel",
                "feeder": "floating_barge" if floating_flag else "future_feeder",
                "num_feeders": int(discrete_inputs["num_feeders"]),
            },
            # Project development costs
            "project_development": {
                "site_auction_price": float(inputs["site_auction_price"]),  # 100e6,
                "site_assessment_plan_cost": float(inputs["site_assessment_plan_cost"]),  # 1e6,
                "site_assessment_cost": float(inputs["site_assessment_cost"]),  # 25e6,
                "construction_operations_plan_cost": float(inputs["construction_operations_plan_cost"]),  # 2.5e6,
                "boem_review_cost": float(inputs["boem_review_cost"]),  # 0,
                "design_install_plan_cost": float(inputs["design_install_plan_cost"]),  # 2.5e6
            },
            # Other
            "commissioning": float(inputs["commissioning_pct"]),
            "decomissioning": float(inputs["decommissioning_pct"]),
            "turbine_capex": float(inputs["turbine_capex"]),
            # Phases
            # Putting monopile or semisub here would override the inputs we assume to get from WISDEM
            "design_phases": [
                #'MonopileDesign',
                #'SemiSubmersibleDesign',
                #'MooringSystemDesign',
                #'ScourProtectionDesign',
                "ArraySystemDesign",
                "ExportSystemDesign",
                "OffshoreSubstationDesign",
            ],
        }

        # Unique design phases
        if floating_flag:
            config["install_phases"] = {
                "ExportCableInstallation": 0,
                "OffshoreSubstationInstallation": 0,
                "MooringSystemInstallation": 0,
                "MooredSubInstallation": ("MooringSystemInstallation", 0.25),
                "ArrayCableInstallation": ("MooredSubInstallation", 0.25),
            }
        else:
            fixedStr = "JacketInstallation" if jacket_flag else "MonopileInstallation"

            if jacket_flag:
                monopile = config.get("monopile", {})
                monopile["diameter"] = 10
                config["monopile"] = monopile

            config["design_phases"] += ["ScourProtectionDesign"]
            config["install_phases"] = {
                "ExportCableInstallation": 0,
                "OffshoreSubstationInstallation": 0,
                "ArrayCableInstallation": 0,
                fixedStr: 0,
                "ScourProtectionInstallation": 0,
                "TurbineInstallation": (fixedStr, 0.25),
            }

        # Unique vessels
        if floating_flag:
            vessels = {
                "support_vessel": "example_support_vessel",
                "towing_vessel": "example_towing_vessel",
                "mooring_install_vessel": "example_support_vessel",
                "towing_vessel_groups": {
                    "towing_vessels": int(discrete_inputs["num_towing"]),
                    "station_keeping_vessels": int(discrete_inputs["num_station_keeping"]),
                },
            }
        else:
            vessels = {
                "feeder": discrete_inputs["feeder"],
                "num_feeders": int(discrete_inputs["num_feeders"]),
                "spi_vessel": "example_scour_protection_vessel",
            }
        config.update(vessels)

        # Unique support structure design/assembly
        if floating_flag:
            config["port"] = {
                "sub_assembly_lines": int(discrete_inputs["num_assembly_lines"]),
                "turbine_assembly_cranes": int(discrete_inputs["num_port_cranes"]),
                "monthly_rate": float(inputs["port_cost_per_month"]),
            }

            config["substructure"] = {
                "takt_time": float(inputs["takt_time"]),
                "unit_cost": float(inputs["floating_substructure_cost"]),
                "towing_speed": 6,  # km/h
            }

            anchorstr_in = discrete_inputs["anchor_type"].lower()
            if anchorstr_in.find("drag") >= 0:
                anchorstr = "Drag Embedment"
            elif anchorstr_in.find("suction") >= 0:
                anchorstr = "Suction Pile"

            config["mooring_system"] = {
                "num_lines": int(discrete_inputs["num_mooring_lines"]),
                "line_mass": 1e-3 * float(inputs["mooring_line_mass"]),
                "line_diam": float(inputs["mooring_line_diameter"]),
                "line_length": float(inputs["mooring_line_length"]),
                "line_cost": float(inputs["mooring_line_cost"]),
                "anchor_mass": 1e-3 * float(inputs["anchor_mass"]),
                "anchor_type": anchorstr,
                "anchor_cost": float(inputs["mooring_anchor_cost"]),
            }
        else:
            config["port"] = {
                "num_cranes": int(discrete_inputs["num_port_cranes"]),
                "monthly_rate": float(inputs["port_cost_per_month"]),
            }

            config["scour_protection_design"] = {
                "cost_per_tonne": 20,
            }

            if jacket_flag:
                config["jacket"] = {
                    "type": "Jacket",
                    "height": float(inputs["jacket_length"]),
                    "num_legs": int(self.options["jacket_legs"]),
                    "deck_space": 4 * float(inputs["jacket_r_foot"]) ** 2,
                    "mass": float(inputs["jacket_mass"]),
                    "unit_cost": float(inputs["jacket_cost"]),
                }
            else:
                config["monopile"] = {
                    "type": "Monopile",
                    "length": float(inputs["monopile_length"]),
                    "diameter": float(inputs["monopile_diameter"]),
                    "deck_space": 0.25 * float(inputs["monopile_diameter"] * inputs["monopile_length"]),
                    "mass": float(inputs["monopile_mass"]),
                    "unit_cost": float(inputs["monopile_cost"]),
                }

        self._orbit_config = config
        return config

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        config = self.compile_orbit_config_file(inputs, outputs, discrete_inputs, discrete_outputs)

        project = ProjectManager(config)
        project.run()

        outputs["bos_capex"] = project.bos_capex
        outputs["total_capex"] = project.total_capex
        outputs["total_capex_kW"] = project.total_capex_per_kw
        outputs["installation_time"] = project.installation_time
        outputs["installation_capex"] = project.installation_capex
