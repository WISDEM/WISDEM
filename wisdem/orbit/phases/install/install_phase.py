"""Provides the base `InstallPhase` class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Jake Nunemaker", "Rob Hammond"]
__email__ = ["jake.nunemaker@nrel.gov", "robert.hammond@nrel.gov"]


import sys
import logging
from io import StringIO
from abc import abstractmethod

import numpy as np
import simpy
import pandas as pd

from wisdem.orbit.phases import BasePhase
from wisdem.orbit.simulation.port import Port
from wisdem.orbit.simulation.exceptions import IncorrectLogLevel

level_dict = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class InstallPhase(BasePhase):
    """BasePhase subclass for install modules."""

    _install_phase = True
    env = None
    phase = None
    logger = None
    log_format = (
        "{level},{msg},{time},{agent},{phase},{type},{action},"
        "{duration},{target},{location}"
    )

    log_header = {
        "level": "level",
        "time": "time",
        "agent": "agent",
        "phase": "phase",
        "type": "type",
        "action": "action",
        "duration": "duration",
        "target": "target",
        "location": "location",
    }

    agent_costs = {}
    _df = None
    config = None

    def init_logger(self, **kwargs):
        """
        Initializes logger with input 'log_level'.

        Parameters
        ----------
        log_level : str
            Logging level.
        print_logs : bool
            Toggles log printing.

        Returns
        -------
        logger : logging.logger
            Logger
        """

        log_level = kwargs.get("log_level", "INFO")
        print_logs = kwargs.get("print_logs", False)

        logger = logging.getLogger(self.phase)
        if logger.hasHandlers():
            logger.handlers.clear()

        try:
            _level = level_dict[log_level]
            logger.setLevel(_level)

        except KeyError:
            raise IncorrectLogLevel(log_level)

        # self.logs handler
        self._logs = StringIO()
        _handler = logging.StreamHandler(self._logs)
        logger.addHandler(_handler)

        # sys.stdout handler
        if print_logs:
            _handler = logging.StreamHandler(sys.stdout)
            _handler.flush = sys.stdout.flush
            logger.addHandler(_handler)

        # Handler level, filter and formatter
        _filter = ContextFilter()
        _format = logging.Formatter(self.log_format, style="{")
        for handler in logger.handlers:
            handler.addFilter(_filter)
            handler.setFormatter(_format)
            handler.setLevel(_level)

        logger.critical("message", extra=self.log_header)
        self.logger = logger

        try:
            self.env.logger = logger

        except AttributeError:
            pass

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

        cranes = self.config["port"]["num_cranes"]

        self.port = Port(self.env)
        self.port.crane = simpy.Resource(self.env, cranes)
        self.logger.debug(
            "PORT INITIALIZED",
            extra={"time": self.env.now, "agent": "Director"},
        )

    def run(self, until=0):
        """
        Runs the simulation on self.env.

        Parameters
        ----------
        until : int, optional
            Number of steps to run.
        """

        self._logs.seek(0, 2)

        if until:
            self.logger.debug(
                "SIMULATION START",
                extra={"time": self.env.now, "agent": "Director"},
            )
            self.env.run(until=until)
            self.logger.debug(
                "SIMULATION END",
                extra={"time": self.env.now, "agent": "Director"},
            )

        else:
            self.logger.debug(
                "SIMULATION START",
                extra={"time": self.env.now, "agent": "Director"},
            )
            self.env.run()
            self.logger.debug(
                "SIMULATION END",
                extra={"time": self.env.now, "agent": "Director"},
            )

        self.generate_log_df()

    def generate_log_df(self):
        """
        Returns self._logs as a DataFrame at self._df.
        """

        self._logs.seek(0)
        self._df = pd.read_csv(self._logs)

        self.append_cost_column()
        self.append_phase_info()
        self.append_port_costs()

        self._df = self._df.sort_values("time").reset_index(drop=True)

    @property
    def logs(self):
        """Returns self._df after simulation is ran."""

        return self._df

    @property
    def partial_df(self):
        """
        For debugging.
        """

        self._logs.seek(0)

        return pd.read_csv(self._logs)

    @property
    def phase_total(self):
        """Returns the total time and cost of installation phase"""

        rows = self.logs.loc[self.logs["action"] == "Complete"]

        if len(rows) != 1:
            raise Exception()

        else:
            time = float(rows["time"])
            cost = "Not Implemented"

            output = {"phase": self.phase, "time_hrs": time, "cost_usd": cost}

        return output

    def append_phase_info(self):
        """
        Adds the phase information to the 'phase' column.
        """

        self._df["phase"] = self.phase

    def append_cost_column(self):
        """
        Adds the cost column to 'self._df'.
        """

        df = self.logs.loc[self.logs["level"] == "INFO"].copy()
        for agent in df["agent"].unique():

            try:
                cost_per_day = self.agent_costs[agent]
                cost_per_hour = cost_per_day / 24

            except KeyError:
                print("Cost not found for agent '{}'".format(agent))
                cost_per_hour = np.NaN

            sub = df.loc[df["agent"] == agent]
            self._df.loc[sub.index, "cost"] = (
                sub["duration"].astype(float) * cost_per_hour
            )

    def append_port_costs(self):
        """
        Adds the port costs every month to 'self._df'.
        """

        HOURS_PER_MONTH = 8760.0 / 12.0

        name = self.config["port"].get("name", "Port")
        _key = "port_cost_per_month"
        _cost = self.config["port"].get("monthly_rate", self.defaults[_key])

        _months = self.total_phase_time / HOURS_PER_MONTH
        _remaining = _months % 1

        months = np.repeat(1.0, _months)
        if _remaining > 0:
            months = np.append(months, _remaining)
        monthly_costs = months * _cost

        port_costs = pd.DataFrame(
            {
                "level": "INFO",
                "time": np.cumsum(months) * HOURS_PER_MONTH,
                "duration": months * HOURS_PER_MONTH,
                "agent": np.repeat(name, len(months)),
                "cost": monthly_costs,
                "type": np.repeat("Operations", len(months)),
                "action": np.repeat("PortRental", len(months)),
                "phase": np.repeat(self.phase, len(months)),
                "location": np.repeat("Port", len(months)),
            }
        )

        self._df = pd.concat([self._df, port_costs], sort=True)

    @property
    def total_phase_cost(self):
        """Returns total phase cost in $USD."""

        df = self.logs.loc[self.logs["level"] == "INFO"]

        if any(df["cost"].isnull()):
            raise ValueError(f"Missing values in 'cost' column.")

        cost = df["cost"].sum()
        return cost

    @property
    def total_phase_time(self):
        """Returns total phase time in hours."""

        df = self.logs.loc[self.logs["level"] == "INFO"]

        if any(df["time"].isnull()):
            raise ValueError(f"Missing values in 'time' column.")

        time = df["time"].max()
        if np.isnan(time):
            time = 0.0

        return time

    @property
    @abstractmethod
    def detailed_output(self):
        """Returns detailed phase information."""

        pass

    @property
    def phase_dataframe(self):
        """
        Returns the logging DataFrame that contains all actions taken during
        InstallPhase.

        Returns
        -------
        df : pd.DataFrame
            A collection of all actions that occured throughout InstallPhase.
        """

        df = self.logs.loc[self.logs["level"] == "INFO"]
        return df

    @property
    def agent_efficiencies(self):
        """
        Returns a summary of agent operational efficiencies.
        """

        logs = self.logs.loc[self.logs["level"] == "INFO"]
        grouped = (
            logs.groupby(["agent", "type"])
            .sum()["duration"]
            .unstack()
            .fillna(0.0)
        )

        if "Operations" not in grouped.columns:
            raise Exception("'Operations' not found in action types.")

        if "Delay" not in grouped.columns:
            grouped["Delay"] = 0.0

        grouped["Total"] = grouped["Operations"] + grouped["Delay"]
        _efficiencies = (grouped["Operations"] / grouped["Total"]).to_dict()
        efficiencies = {
            k + "_operational_efficiency": v for k, v in _efficiencies.items()
        }

        return efficiencies

    @staticmethod
    def get_max_cargo_weight_utilzations(vessels):
        """
        Returns a summary of cargo weight efficiencies for list of input `vessels`.

        Parameters
        ----------
        vessels : list
            List of vessels to calculate efficiencies for.
        """

        outputs = {}

        for vessel in vessels:

            storage = getattr(vessel, "storage", None)
            if storage is None:
                print("Vessel does not have storage capacity.")
                continue

            outputs[
                f"{vessel.name}_cargo_weight_utilization"
            ] = vessel.max_cargo_weight_utilization

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

            storage = getattr(vessel, "storage", None)
            if storage is None:
                print("Vessel does not have storage capacity.")
                continue

            outputs[
                f"{vessel.name}_deck_space_utilization"
            ] = vessel.max_deck_space_utilization

        return outputs


class ContextFilter(logging.Filter):
    """Custom logging filter."""

    _defaults = {
        "msg": "",
        "time": np.NaN,
        "agent": "",
        "phase": "",
        "type": "",
        "action": "",
        "duration": np.NaN,
        "target": "",
        "location": "",
    }

    def filter(self, record):
        """
        Adds default context values for expected keys in 'extra' dictionary.

        Parameters
        ----------
        record : LogRecord
        """

        for key, default in self._defaults.items():
            val = getattr(record, key, default)
            setattr(record, key, f'"{val}"')

        level = getattr(record, "level", record.levelname)
        setattr(record, "level", f"{level}")

        return True
