"""`InstallPhase` base class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Jake Nunemaker", "Rob Hammond"]
__email__ = ["jake.nunemaker@nrel.gov", "robert.hammond@nrel.gov"]


from abc import abstractmethod
from itertools import groupby

import numpy as np
import simpy
import pandas as pd

from wisdem.orbit.core import Port, Vessel, Environment
from wisdem.orbit.phases import BasePhase
from wisdem.orbit.core.defaults import common_costs


class InstallPhase(BasePhase):
    """BasePhase subclass for install modules."""

    def __init__(self, weather, **kwargs):
        """
        Creates an instance of `InstallPhase`.

        Parameters
        ----------
        weather : pd.DataFrame | np.ndarray
            Weather profile at site.
        """

        self.extract_phase_kwargs(**kwargs)
        self.initialize_environment(weather, **kwargs)
        self.availability = kwargs.get("availability", None)

    def initialize_environment(self, weather, **kwargs):
        """
        Initializes a `marmot.Environment` at `self.env`.

        Parameters
        ----------
        weather : np.ndarray
            Weather profile at site.
        """

        if isinstance(weather, pd.DataFrame):
            weather = weather.to_records()

        env_name = kwargs.get("env_name", "Environment")
        self.env = Environment(name=env_name, state=weather, **kwargs)

    def initialize_vessel(self, name, specs):
        """"""

        avail = getattr(self, "availability")
        if avail is None:
            return Vessel(name, specs)

        elif isinstance(avail, dict):
            default = avail.get("default", 1)
            return Vessel(name, specs, avail.get(name, default))

        else:
            return Vessel(name, specs, avail)

    @abstractmethod
    def setup_simulation(self):
        """
        Sets up the required simulation infrastructure

        Generally, this creates the port, initializes the items to be
        installed, and initializes the vessel(s) used for the installation.
        """

        pass

    def initialize_port(self):
        """
        Initializes a Port object with N number of cranes.
        """

        self.port = Port(self.env)

        try:
            cranes = self.config["port"]["num_cranes"]
            self.port.crane = simpy.Resource(self.env, cranes)

        except KeyError:
            self.port.crane = simpy.Resource(self.env, 1)

    def run(self, until=None):
        """
        Runs the simulation on self.env.

        Parameters
        ----------
        until : int, optional
            Number of steps to run.
        """

        self.env._submit_log({"message": "SIMULATION START"}, "DEBUG")
        self.env.run(until=until)
        self.append_phase_info()
        self.env._submit_log({"message": "SIMULATION END"}, "DEBUG")

    def append_phase_info(self):
        """Appends phase information to all logs in `self.env.logs`."""

        for l in self.env.logs:
            l["phase"] = self.phase

    @property
    def port_costs(self):
        """Cost of port rental."""

        port = getattr(self, "port", None)

        if port is None:
            return 0

        else:
            key = "port_cost_per_month"
            port_config = self.config.get("port", {})
            rate = port_config.get("monthly_rate", common_costs[key])

            months = self.total_phase_time / (8760 / 12)
            return months * rate

    @property
    def installation_capex(self):
        """Returns sum of all installation costs in `self.env.actions`."""

        return np.nansum([a["cost"] for a in self.env.actions]) + self.port_costs

    @property
    def total_phase_time(self):
        """Returns total phase time in hours."""

        return max([a["time"] for a in self.env.actions])

    @property
    @abstractmethod
    def detailed_output(self):
        """Returns detailed phase information."""

        pass

    @property
    def agent_efficiencies(self):
        """
        Returns a summary of agent operational efficiencies.
        """

        efficiencies = {}

        s = sorted(self.env.actions, key=lambda x: (x["agent"], x["action"]))
        grouped = {
            k: sum([i["duration"] for i in list(v)]) for k, v in groupby(s, key=lambda x: (x["agent"], x["action"]))
        }
        agents = list(set([k[0] for k in grouped.keys()]))
        for agent in agents:
            total = sum([v for k, v in grouped.items() if k[0] == agent])

            try:
                delay = grouped[(agent, "Delay")]
                e = (total - delay) / total

            except KeyError:
                delay = 0.0
                e = 1.0

            except ZeroDivisionError:
                e = 1.0

            if not 0.0 <= e <= 1.0:
                raise ValueError(f"Invalid efficiency for agent '{agent}'")

            name = str(agent).replace(" ", "_")
            efficiencies[f"{name}_operational_efficiency"] = e

        return efficiencies

    @staticmethod
    def get_max_cargo_mass_utilzations(vessels):
        """
        Returns a summary of cargo mass efficiencies for list of input `vessels`.

        Parameters
        ----------
        vessels : list
            List of vessels to calculate efficiencies for.
        """

        outputs = {}

        for vessel in vessels:
            name = vessel.name.replace(" ", "_")
            storage = getattr(vessel, "storage", None)
            if storage is None:
                print("Vessel does not have storage capacity.")
                continue

            outputs[f"{name}_cargo_mass_utilization"] = vessel.max_cargo_mass_utilization

        return outputs

    @staticmethod
    def get_max_deck_space_utilzations(vessels):
        """
        Returns a summary of deck space efficiencies for list of input `vessels`.

        Parameters
        ----------
        vessels : list
            List of vessels to calculate efficiencies for.
        """

        outputs = {}

        for vessel in vessels:
            name = vessel.name.replace(" ", "_")
            storage = getattr(vessel, "storage", None)
            if storage is None:
                print("Vessel does not have storage capacity.")
                continue

            outputs[f"{name}_deck_space_utilization"] = vessel.max_deck_space_utilization

        return outputs
