__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


from copy import deepcopy

from benedict import benedict

from wisdem.orbit import ProjectManager

DEFAULT_MULTIPLIERS = {
    "blades": {"domestic": 0.026, "imported": 0.30},
    "nacelle": {"domestic": 0.025, "imported": 0.10},
    "tower": {
        "domestic": 0.04,
        "imported": 0.20,
        "tariffs": 0.25,
    },
    "monopile": {
        "domestic": 0.085,
        "imported": 0.28,
        "tariffs": 0.25,
    },
    "transition_piece": {
        "domestic": 0.169,
        "imported": 0.17,
        "tariffs": 0.25,
    },
    "array_cable": {"domestic": 0.19, "imported": 0.0},
    "export_cable": {"domestic": 0.231, "imported": 0.0},
    "oss_topside": {"domestic": 0.0, "imported": 0.0},
    "oss_substructure": {"domestic": 0.0, "imported": 0.0},
}


TURBINE_CAPEX_SPLIT = {"blades": 0.135, "nacelle": 0.274, "tower": 0.162}


LABOR_SPLIT = {"tower": 0.5, "monopile": 0.5, "transition_piece": 0.5, "oss_topside": 0.5}


class SupplyChainManager:
    def __init__(self, supply_chain_configuration, **kwargs):
        """
        Creates an instance of `SupplyChainManager`.

        Parameters
        ----------
        configuration : dict
        """

        self.sc_config = supply_chain_configuration
        self.multipliers = kwargs.get("multipliers", DEFAULT_MULTIPLIERS)
        self.labor_split = kwargs.get("labor_split", LABOR_SPLIT)
        self.turbine_split = kwargs.get("turbine_split", TURBINE_CAPEX_SPLIT)

    def run_project(self, config, weather=None, **kwargs):
        """
        Runs an ORBIT configuration, performing any supply chain related pre
        and post processing steps on either side of the ORBIT run.

        Parameters
        ----------
        config : dict
            ORBIT Configuration
        weather : pd.DataFrame | None
            Weather data to use for the ORBIT simulation.
        """

        config = benedict(deepcopy(config))
        config = self.pre_process(config)

        project = ProjectManager(config, weather=weather, **kwargs)
        project.run()

        project = self.post_process(project)

        return project

    def pre_process(self, config):
        """"""

        # Save original plant design
        plant = deepcopy(config["plant"])

        # Run ProjectManager without install phases to generate design results
        install_phases = config["install_phases"]
        config["install_phases"] = []
        project = ProjectManager(config)
        project.run()
        config = deepcopy(project.config)

        # Replace calculated plant design with original
        config["plant"] = plant

        # Run pre ORBIT supply chain adjustments
        config = self.process_turbine_capex(config)
        config = self.process_monopile_capex(config)
        config = self.process_transition_piece_capex(config)
        config = self.process_offshore_substation_topside_capex(config)

        # Add install phases back in
        config["install_phases"] = install_phases
        config["design_phases"] = []

        return config

    def post_process(self, project):
        """"""

        project = self.process_array_cable_capex(project)
        project = self.process_export_cable_capex(project)

        return project

    # Pre Processing Methods
    def process_turbine_capex(self, config):
        """
        Add blade, nacelle and tower cost adders to input `turbine_capex`.

        Parameters
        ----------
        config : dict
            ORBIT configuration.
        """

        blade_scenario = self.sc_config["blades"]
        nacelle_scenario = self.sc_config["nacelle"]
        tower_scenario = self.sc_config["blades"]

        blade_mult = self.multipliers["blades"].get(blade_scenario, None)
        if blade_mult == None:
            print(f"Warning: scenario '{blade_scenario}' not found for category 'blades'.")
            blade_mult = 0.0

        nacelle_mult = self.multipliers["nacelle"].get(nacelle_scenario, None)
        if nacelle_mult == None:
            print(f"Warning: scenario '{nacelle_scenario}' not found for category 'nacelle'.")
            nacelle_mult = 0.0

        raw_cost = config.get("project_parameters.turbine_capex", 1300)
        blade_adder = raw_cost * self.turbine_split["blades"] * blade_mult
        nacelle_adder = raw_cost * self.turbine_split["nacelle"] * nacelle_mult

        if tower_scenario == "domestic, imported steel":
            tower_adder = self.multipliers["tower"]["domestic"] * raw_cost
            tower_tariffs = (
                raw_cost
                * self.turbine_split["tower"]
                * (1 - self.labor_split["tower"])
                * self.multipliers["tower"]["tariffs"]
            )

        else:
            tower_tariffs = 0.0
            tower_mult = self.multipliers["tower"].get(tower_scenario, None)
            if tower_mult == None:
                print(f"Warning: scenario '{tower_scenario}' not found for category 'tower'.")
                tower_mult = 0.0

            tower_adder = raw_cost * self.turbine_split["tower"] * tower_mult

        config["project_parameters.turbine_capex"] = sum(
            [raw_cost, blade_adder, nacelle_adder, tower_adder, tower_tariffs]
        )

        return config

    def process_monopile_capex(self, config):
        """
        Add monopile cost adder and potential tariffs to `monopile.unit_cost`.

        Parameters
        ----------
        config : dict
            ORBIT configuration.
        """

        raw_cost = config["monopile.unit_cost"]
        scenario = self.sc_config["monopile"]

        if scenario == "domestic, imported steel":
            adder = self.multipliers["monopile"]["domestic"] * raw_cost
            tariffs = raw_cost * (1 - self.labor_split["monopile"]) * self.multipliers["monopile"]["tariffs"]

        else:
            tariffs = 0.0
            mult = self.multipliers["monopile"].get(scenario, None)
            if mult == None:
                print(f"Warning: scenario '{scenario}' not found for category 'monopile'.")
                mult = 0.0

            adder = raw_cost * mult

        config["monopile.unit_cost"] = sum([raw_cost, adder, tariffs])

        return config

    def process_transition_piece_capex(self, config):
        """
        Add transition piece cost adder and potential tariffs to
        `transition_piece.unit_cost`.

        Parameters
        ----------
        config : dict
            ORBIT configuration.
        """

        raw_cost = config["transition_piece.unit_cost"]
        scenario = self.sc_config["transition_piece"]

        if scenario == "domestic, imported steel":
            adder = self.multipliers["transition_piece"]["domestic"] * raw_cost
            tariffs = (
                raw_cost * (1 - self.labor_split["transition_piece"]) * self.multipliers["transition_piece"]["tariffs"]
            )

        else:
            tariffs = 0.0
            mult = self.multipliers["transition_piece"].get(scenario, None)
            if mult == None:
                print(f"Warning: scenario '{scenario}' not found for category 'transition_piece'.")
                mult = 0.0

            adder = raw_cost * mult

        config["transition_piece.unit_cost"] = sum([raw_cost, adder, tariffs])

        return config

    def process_offshore_substation_topside_capex(self, config):
        """
        Add OSS topside cost adder and potential tariffs to
        `offshore_substation_topside.unit_cost`.

        Parameters
        ----------
        config : dict
            ORBIT configuration.
        """

        raw_cost = config["offshore_substation_topside.unit_cost"]
        scenario = self.sc_config["oss_topside"]

        if scenario == "domestic, imported steel":
            adder = self.multipliers["oss_topside"]["domestic"] * raw_cost
            tariffs = raw_cost * (1 - self.labor_split["oss_topside"]) * self.multipliers["oss_topside"]["tariffs"]

        else:
            tariffs = 0.0
            mult = self.multipliers["oss_topside"].get(scenario, None)
            if mult == None:
                print(f"Warning: scenario '{scenario}' not found for category 'oss_topside'.")
                mult = 0.0

            adder = raw_cost * mult

        config["offshore_substation_topside.unit_cost"] = sum([raw_cost, adder, tariffs])

        return config

    # Post Processing Methods
    def process_array_cable_capex(self, project):
        """
        Add array cable cost adder.

        Parameters
        ----------
        project : ProjectManager
        """

        scenario = self.sc_config["array_cable"]
        mult = self.multipliers["array_cable"].get(scenario, None)
        if mult == None:
            print(f"Warning: scenario '{scenario}' not found for category 'array_cable'.")
            mult = 0.0

        project.system_costs["ArrayCableInstallation"] *= 1 + mult

        return project

    def process_export_cable_capex(self, project):
        """
        Add export cable cost adder.

        Parameters
        ----------
        project : ProjectManager
        """

        scenario = self.sc_config["export_cable"]
        mult = self.multipliers["export_cable"].get(scenario, None)
        if mult == None:
            print(f"Warning: scenario '{scenario}' not found for category 'export_cable'.")
            mult = 0.0

        project.system_costs["ExportCableInstallation"] *= 1 + mult

        return project
