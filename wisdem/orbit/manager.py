__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import os
import re
import datetime as dt
import collections.abc as collections
from copy import deepcopy
from math import ceil
from numbers import Number
from itertools import product

import numpy as np
import pandas as pd
from benedict import benedict

import wisdem.orbit as orbit
from wisdem.orbit.phases import DesignPhase, InstallPhase
from wisdem.orbit.core.library import initialize_library, export_library_specs, extract_library_data
from wisdem.orbit.phases.design import (
    SparDesign,
    MonopileDesign,
    ArraySystemDesign,
    ExportSystemDesign,
    MooringSystemDesign,
    ScourProtectionDesign,
    SemiSubmersibleDesign,
    CustomArraySystemDesign,
    OffshoreSubstationDesign,
)
from wisdem.orbit.phases.install import (
    JacketInstallation,
    TurbineInstallation,
    MonopileInstallation,
    MooredSubInstallation,
    ArrayCableInstallation,
    ExportCableInstallation,
    GravityBasedInstallation,
    MooringSystemInstallation,
    ScourProtectionInstallation,
    FloatingSubstationInstallation,
    OffshoreSubstationInstallation,
)
from wisdem.orbit.core.exceptions import PhaseNotFound, WeatherProfileError, PhaseDependenciesInvalid


class ProjectManager:
    """Base Project Manager Class."""

    date_format_short = "%m/%d/%Y"
    date_format_long = "%m/%d/%Y %H:%M"

    _design_phases = [
        MonopileDesign,
        ArraySystemDesign,
        CustomArraySystemDesign,
        ExportSystemDesign,
        ScourProtectionDesign,
        OffshoreSubstationDesign,
        MooringSystemDesign,
        SemiSubmersibleDesign,
        SparDesign,
    ]

    _install_phases = [
        MonopileInstallation,
        TurbineInstallation,
        OffshoreSubstationInstallation,
        ArrayCableInstallation,
        ExportCableInstallation,
        ScourProtectionInstallation,
        MooredSubInstallation,
        MooringSystemInstallation,
        GravityBasedInstallation,
        FloatingSubstationInstallation,
        JacketInstallation,
    ]

    def __init__(self, config, library_path=None, weather=None):
        """
        Creates and instance of ProjectManager.

        Parameters
        ----------
        config : dict
            Project configuration.
        library_path: str, default: None
            The absolute path to the project library.
        weather : np.ndarray
            Site weather timeseries.
        """

        initialize_library(library_path)
        config = deepcopy(config)
        config = extract_library_data(
            config,
            additional_keys=[
                *config.get("design_phases", []),
                *config.get("install_phases", []),
            ],
        )
        self._phases = {}
        self.config = benedict(config)
        self.resolve_project_capacity()
        self.weather = self.transform_weather_input(weather)

        self.design_results = {}
        self.detailed_outputs = {}

        self.system_costs = {}
        self.installation_costs = {}

        # TODO: Revise:
        self.phase_starts = {}
        self.phase_times = {}
        self._output_logs = []

    def run(self, **kwargs):
        """
        Main project run method.

        Parameters
        ----------
        self.config['design_phases'] : list
            Defines which design phases are ran. These phases are ran before
            install phases and merge the result of the design into self.config.
        self.config['install_phases'] : list | dict
            Defines which installation phases are ran.

            - If ``self.config['install_phases']`` is a list, phases are ran
              sequentially using ``self.run_multiple_phases_in_serial()``.
            - If ``self.config['install_phases']`` is a dict, phases are ran
              using ``self.run_multiple_phases_overlapping()``. The expected
              format for the dictionary is ``{'phase_name': '%m/%d/%Y'}``.
        """

        design_phases = self.config.get("design_phases", [])
        install_phases = self.config.get("install_phases", [])

        if isinstance(design_phases, str):
            design_phases = [design_phases]
        if isinstance(install_phases, str):
            install_phases = [install_phases]

        self.run_all_design_phases(design_phases, **kwargs)

        if isinstance(install_phases, (list, set)):
            self.run_multiple_phases_in_serial(install_phases, **kwargs)

        elif isinstance(install_phases, dict):
            self.run_multiple_phases_overlapping(install_phases, **kwargs)

        if install_phases:
            self.progress = ProjectProgress(self.progress_logs)

        self._print_warnings()

    def _print_warnings(self):

        try:
            df = pd.DataFrame(self.logs)
            df = df.loc[~df["message"].isnull()]
            df = df.loc[df["message"].str.contains("Exceeded")]

            for msg in df["message"].unique():

                idx = df.loc[df["message"] == msg].index[0]
                phase = df.loc[idx, "phase"]
                print(f"{phase}:\n\t {msg}")

        except KeyError:
            pass

    @property
    def phases(self):
        """Returns dict of phases that have been ran."""

        return self._phases

    @property
    def _capex_categories(self):
        """Returns CapEx categories for phases in `self._install_phases`."""

        out = {}
        for p in self._install_phases:
            try:
                out[p.__name__] = p.capex_category

            except AttributeError:
                out[p.__name__] = "Misc."

        return out

    @property
    def project_params(self):
        """Returns defined project parameters, if found."""

        return self.config.get("project_parameters", {})

    @classmethod
    def compile_input_dict(cls, phases):
        """
        Returns a compiled input dictionary given a list of phases to run.

        Parameters
        ----------
        phases : list
            A collection of offshore design or installation phases.
        """

        _phases = {p: cls.find_key_match(p) for p in phases}
        _error = [n for n, c in _phases.items() if not bool(c)]
        if _error:
            raise PhaseNotFound(_error)

        design_phases = {n: c for n, c in _phases.items() if issubclass(c, DesignPhase)}
        install_phases = {n: c for n, c in _phases.items() if issubclass(c, InstallPhase)}

        config = {}
        for i in install_phases.values():
            config = cls.merge_dicts(config, i.expected_config)

        for d in design_phases.values():
            config = cls.merge_dicts(config, d.expected_config)
            config = cls.remove_keys(config, d.output_config)

        config["project_parameters"] = {
            "turbine_capex": "$/kW (optional, default: 1300)",
            "ncf": "float (optional, default: 0.4)",
            "offtake_price": "$/MWh (optional, default: 80)",
            "project_lifetime": "yrs (optional, default: 25)",
            "discount_rate": "yearly (optional, default: .025)",
            "opex_rate": "$/kW/year (optional, default: 150)",
            "construction_insurance": "$/kW (optional, default: 44)",
            "construction_financing": "$/kW (optional, default: 183)",
            "contingency": "$/kW (optional, default: 316)",
            "commissioning": "$/kW (optional, default: 44)",
            "decommissioning": "$/kW (optional, default: 58)",
            "site_auction_price": "$ (optional, default: 100e6)",
            "site_assessment_cost": "$ (optional, default: 50e6)",
            "construction_plan_cost": "$ (optional, default: 1e6)",
            "installation_plan_cost": "$ (optional, default: 0.25e6)",
        }

        config["design_phases"] = [*design_phases.keys()]
        config["install_phases"] = [*install_phases.keys()]
        config["orbit_version"] = str(orbit.__version__)

        return config

    def resolve_project_capacity(self):
        """
        Resolves the relationship between 'project_capacity', 'num_turbines'
        and 'turbine_rating' and verifies that input and calculated values
        match. Adds missing values that can be calculated to the 'self.config'.
        """

        try:
            project_capacity = self.config["plant"]["capacity"]
        except KeyError:
            project_capacity = None

        try:
            turbine_rating = self.config["turbine"]["turbine_rating"]
        except KeyError:
            turbine_rating = None

        try:
            num_turbines = self.config["plant"]["num_turbines"]
        except KeyError:
            num_turbines = None

        if all((project_capacity, turbine_rating, num_turbines)):
            if project_capacity != (turbine_rating * num_turbines):
                raise AttributeError(f"Input and calculated project capacity don't match.")

        else:
            if all((project_capacity, turbine_rating)):
                num_turbines = ceil(project_capacity / turbine_rating)
                self.config["plant"]["num_turbines"] = num_turbines

            elif all((project_capacity, num_turbines)):
                turbine_rating = project_capacity / num_turbines
                try:
                    self.config["turbine"]["turbine_rating"] = turbine_rating

                except KeyError:
                    self.config["turbine"] = {"turbine_rating": turbine_rating}

            elif all((num_turbines, turbine_rating)):
                project_capacity = turbine_rating * num_turbines
                self.config["plant"]["capacity"] = project_capacity

    @classmethod
    def find_key_match(cls, target):
        """
        Searches cls.phase_dict() for a key that matches text in 'target'.

        Parameters
        ----------
        target : str
            Phase name to search for a match with.

        Returns
        -------
        phase_class : BasePhase | None
            Matched module class or None if no match is found.
        """

        phase = re.split("[_ ]", target)[0]
        phase_dict = cls.phase_dict()
        phase_class = phase_dict.get(phase, None)

        return phase_class

    @classmethod
    def phase_dict(cls):
        """
        Returns dictionary of all possible phases with format 'name': 'class'.
        """

        install = {p.__name__: p for p in cls._install_phases}
        design = {p.__name__: p for p in cls._design_phases}

        return {**install, **design}

    @classmethod
    def merge_dicts(cls, left, right, overwrite=True, add_keys=True):
        """
        Merges two dicts (right into left) with an option to add keys of right.

        Parameters
        ----------
        left : dict
        right : dict
        add_keys : bool

        Returns
        -------
        new : dict
            Merged dictionary.
        """
        new = left.copy()
        if not add_keys:
            right = {k: right[k] for k in set(new).intersection(set(right))}

        for k, _ in right.items():
            if k in new and isinstance(new[k], dict) and isinstance(right[k], collections.Mapping):
                new[k] = cls.merge_dicts(new[k], right[k], overwrite=overwrite, add_keys=add_keys)
            elif k in new and isinstance(new[k], list) and isinstance(right[k], list):
                new[k].extend(right[k])
            else:
                if overwrite or k not in new:
                    new[k] = right[k]

                else:
                    continue

        return new

    @classmethod
    def remove_keys(cls, left, right):
        """
        Recursively removes keys from left that are found in right.

        Parameters
        ----------
        left : dict
        right : dict

        Returns
        -------
        new : dict
            Left dictionary with keys of right removed.
        """

        new = left.copy()
        right = {k: right[k] for k in set(new).intersection(set(right))}

        for k, val in right.items():

            if isinstance(new.get(k, None), dict) and isinstance(val, dict):
                new[k] = cls.remove_keys(new[k], val)

                if not new[k]:
                    _ = new.pop(k)

            else:
                _ = new.pop(k)

        return new

    def create_config_for_phase(self, phase):
        """
        Produces a configuration input dictionary for 'phase'.

        This method will pick the most specific definition of each parameter.
        For example, if self.master_config['site']['distance'] and
        self.config['PhaseName']['site']['distance'] are both defined,
        the latter will be chosen as it is more specific. This allows for phase
        specific definitions, eg. distance to port dependent on phase.

        Parameters
        ----------
        phase : str
            Name of phase. Phase specific information will be pulled from
            self.config['PhaseName'] if this key exists.

        Returns
        -------
        phase_config : dict
            Configuration dictionary with phase specific information merged in.
        """

        _specific = self.config.get(phase, {}).copy()
        _general = {k: v for k, v in self.config.items() if k not in set(self.phase_dict())}

        phase_config = self.merge_dicts(_general, _specific)

        return phase_config

    @property
    def phase_ends(self):

        ret = {}
        for k, t in self.phase_times.items():
            try:
                ret[k] = self.phase_starts[k] + t

            except KeyError:
                pass

        return ret

    def run_install_phase(self, name, start, **kwargs):
        """
        Compiles the phase specific configuration input dictionary for input
        'name', checks the input against _class.expected_config and runs the
        phase calculations with 'phase.run()'.

        Parameters
        ----------
        name : str
            Phase to run.
        weather : None | np.ndarray

        Returns
        -------
        time : int | float
            Total phase time.
        logs : list
            List of phase logs.
        """

        if self.weather is not None:
            weather = self.get_weather_profile(start)

        else:
            weather = None

        _catch = kwargs.get("catch_exceptions", False)
        _class = self.get_phase_class(name)
        _config = self.create_config_for_phase(name)
        processes = _config.pop("processes", {})
        kwargs = {**kwargs, **processes}

        if _catch:
            try:
                phase = _class(_config, weather=weather, phase_name=name, **kwargs)
                phase.run()

            except Exception as e:
                print(f"\n\t - {name}: {e}")
                return None, None

        else:
            phase = _class(_config, weather=weather, phase_name=name, **kwargs)
            phase.run()

        self._phases[name] = phase

        time = phase.total_phase_time
        logs = deepcopy(phase.env.logs)

        self.phase_starts[name] = start
        self.phase_times[name] = time
        self.detailed_outputs = self.merge_dicts(self.detailed_outputs, phase.detailed_output)

        if phase.system_capex:
            self.system_costs[name] = phase.system_capex

        if phase.installation_capex:
            self.installation_costs[name] = phase.installation_capex

        return time, logs

    def get_phase_class(self, phase):
        """
        Returns the class object for input 'phase'.

        Parameters
        ----------
        phase : str
            Name of phase. Must match a class name in either
            'self._install_phases' or 'self._design_phases'.

        Returns
        -------
        phase_class : Phase
            Class of base type Phase that represents input 'phase'.
        """

        _dict = self.phase_dict()
        phase_class = self.find_key_match(phase)
        if phase_class is None:
            raise PhaseNotFound(phase)

        return phase_class

    def run_all_design_phases(self, phase_list, **kwargs):
        """
        Runs multiple design phases and adds '.design_result' to self.config.
        """

        for name in phase_list:
            self.run_design_phase(name, **kwargs)

    def run_design_phase(self, name, **kwargs):
        """
        Runs a design phase defined by 'name' and merges the '.design_result'
        into self.config.

        Parameters
        ----------
        name : str
            Name of design phase that partially matches a key in `phase_dict`.
        """

        _catch = kwargs.get("catch_exceptions", False)
        _class = self.get_phase_class(name)
        _config = self.create_config_for_phase(name)

        if _catch:
            try:
                phase = _class(_config)
                phase.run()

            except Exception as e:
                print(f"\n\t - {name}: {e}")
                return

        else:
            phase = _class(_config)
            phase.run()

        self._phases[name] = phase

        self.design_results = self.merge_dicts(self.design_results, phase.design_result, overwrite=False)

        self.config = self.merge_dicts(self.config, phase.design_result, overwrite=False)
        self.detailed_outputs = self.merge_dicts(self.detailed_outputs, phase.detailed_output)

    def run_multiple_phases_in_serial(self, phase_list, **kwargs):
        """
        Runs multiple phases listed in self.config['install_phases'] in serial.

        Parameters
        ----------
        phase_list : list
            List of installation phases to run.
        """

        start = 0

        for name in phase_list:
            time, logs = self.run_install_phase(name, start, **kwargs)

            if logs is None:
                continue

            else:
                for l in logs:
                    try:
                        l["time"] += start
                    except KeyError:
                        pass

                self._output_logs.extend(logs)
                start = ceil(start + time)

    def run_multiple_phases_overlapping(self, phases, **kwargs):
        """
        Runs multiple phases overlapping using a mixture of dates, indices or
        dependencies.

        Parameters
        ----------
        phases : dict
            Dictionary of phases to run.
        """

        defined, variable = self._parse_install_phase_values(phases)
        zero = min(defined.values())

        # Run defined
        for name, start in defined.items():

            _, logs = self.run_install_phase(name, start, **kwargs)

            if logs is None:
                continue

            else:
                for l in logs:
                    try:
                        l["time"] += start - zero
                    except KeyError:
                        pass

                self._output_logs.extend(logs)

        # Run remaining phases
        self.run_dependent_phases(variable, zero, **kwargs)

    def run_dependent_phases(self, _phases, zero, **kwargs):
        """
        Runs remaining phases that depend on other phase times.

        Parameters
        ----------
        _phases : dict
            Dictionary of phases to run.
        zero : int | float
            Zero time for the simulation. Used to aggregate total logs.
        """

        phases = deepcopy(_phases)
        skipped = {}

        while True:

            phases = {**phases, **skipped}
            if not phases:
                break

            skipped = {}
            progress = False

            for name in list(phases.keys()):
                target, perc = phases.pop(name)

                try:
                    start = self.get_dependency_start_time(target, perc)
                    _, logs = self.run_install_phase(name, start, **kwargs)

                    progress = True

                    if logs is None:
                        continue

                    else:
                        for l in logs:
                            try:
                                l["time"] += start - zero
                            except KeyError:
                                pass

                        self._output_logs.extend(logs)

                except KeyError:
                    skipped[name] = (target, perc)

            if not progress:
                raise PhaseDependenciesInvalid(_phases)

    def get_dependency_start_time(self, target, perc):
        """
        Returns start time based on the `perc` complete of `target` phase.

        Parameters
        ----------
        target : str
            Phase that start time is dependent on.
        perc : int | float
            Percentage of the target phase completion time. `0`: starts at the
            same time. `1`: starts when target phase is completed.
        """

        start = self.phase_starts[target]
        elapsed = self.phase_times[target]

        return start + elapsed * perc

    @staticmethod
    def transform_weather_input(weather):
        """
        Checks that an input weather profile matches the required format and
        converts the index to a datetime index if necessary.

        Parameters
        ----------
        weather : pd.DataFrame
        """

        if weather is None:
            return None

        else:
            try:
                weather = weather.set_index("datetime")
                if not isinstance(weather.index, pd.DatetimeIndex):
                    weather.index = pd.to_datetime(weather.index)

            except KeyError:
                pass

            return weather

    def _parse_install_phase_values(self, phases):
        """
        Parses the input dictionary `install_phases`, splitting them into
        phases that have defined start times and ones that rely on other phases.

        Parameters
        ----------
        phases : dict
            Dictionary of installation phases to run.

        Raises
        ------
        ValueError
            Raised if no phases have a defined start date as the project can't
            be tied to a specific part of the weather profile.
        """

        defined = {}
        depends = {}

        for k, v in phases.items():

            if isinstance(v, (int, float)):
                defined[k] = ceil(v)

            elif isinstance(v, str):
                _dt = dt.datetime.strptime(v, self.date_format_short)

                try:
                    i = self.weather.index.get_loc(_dt)
                    defined[k] = i

                except AttributeError:
                    raise ValueError(f"No weather profile configured " f"for '{k}': '{v}' input type.")

                except KeyError:
                    raise WeatherProfileError(_dt, self.weather)

            elif isinstance(v, tuple) and len(v) == 2:
                depends[k] = v

            else:
                raise ValueError(f"Input type '{k}': '{v}' not recognized.")

        if not defined:
            raise ValueError("No phases have a defined start index/date.")

        return defined, depends

    def get_weather_profile(self, start):
        """
        Pulls weather profile from 'self.weather' starting at 'start', raising
        any errors if needed.

        Parameters
        ----------
        start : datetime
            Starting index for output weather profile.

        Returns
        -------
        profile : np.ndarray.
            Weather profile with first index at 'start'.
        """

        return self.weather.iloc[ceil(start) :].copy().to_records()

    def outputs(self, include_logs=False, npv_detailed=False):
        """Returns dict of all available outputs."""

        out = {
            # Times
            "project_time": self.project_time,
            "installation_time": self.installation_time,
            # Costs
            "capex_breakdown": self.capex_breakdown,
            "capex_breakdown_per_kw": self.capex_breakdown_per_kw,
            "turbine_capex": self.turbine_capex,
            "turbine_capex_per_kw": self.turbine_capex_per_kw,
            "installation_capex": self.installation_capex,
            "installation_capex_per_kw": self.installation_capex_per_kw,
            "system_capex": self.system_capex,
            "system_capex_per_kw": self.system_capex_per_kw,
            "overnight_capex": self.overnight_capex,
            "overnight_capex_per_kw": self.overnight_capex_per_kw,
            "soft_capex": self.soft_capex,
            "soft_capex_per_kw": self.soft_capex_per_kw,
            "bos_capex": self.bos_capex,
            "bos_capex_per_kw": self.bos_capex_per_kw,
            "project_capex": self.project_capex,
            "project_capex_per_kw": self.project_capex_per_kw,
            "total_capex": self.total_capex,
            "total_capex_per_kw": self.total_capex_per_kw,
            "npv": self.npv,
        }

        if include_logs:
            out["logs"] = self.actions

        if npv_detailed:
            out = {
                **out,
                **{
                    "cash_flow": self.cash_flow,
                    "monthly_revenue": self.monthly_revenue,
                    "monthly_expenses": self.monthly_expenses,
                },
            }

        return out

    @property
    def capacity(self):
        """Returns project capacity in MW."""

        try:
            capacity = self.config["plant"]["capacity"]

        except KeyError:
            capacity = None

        return capacity

    @property
    def num_turbines(self):
        """Returns number of turbines in the project."""

        try:
            num_turbines = self.config["plant"]["num_turbines"]

        except KeyError:
            num_turbines = None

        return num_turbines

    @property
    def turbine_rating(self):
        """Returns turbine rating in MW"""

        try:
            rating = self.config["turbine"]["turbine_rating"]

        except KeyError:
            rating = None

        return rating

    @property
    def logs(self):
        """Returns list of all logs in the project."""

        return sorted(self._output_logs, key=lambda l: l["time"])

    @property
    def project_time(self):
        """Returns total project time as the time of the last log."""

        return self.actions[-1]["time"]

    @property
    def month_bins(self):
        """Returns bins representing project months."""

        return np.arange(0, self.project_time + 730, 730)

    @property
    def monthly_expenses(self):
        """Returns the monthly expenses of the project from development through
        construction."""

        opex = self.monthly_opex
        lifetime = self.project_params.get("project_lifetime", 25)

        _expense_logs = self._filter_logs(keys=["cost", "time"])
        expenses = np.array(_expense_logs, dtype=[("cost", "f8"), ("time", "i4")])
        dig = np.digitize(expenses["time"], self.month_bins)

        monthly = {}
        for i in range(1, lifetime * 12):
            monthly[i] = sum(expenses["cost"][dig == i]) + opex[i]

        return monthly

    @property
    def monthly_opex(self):
        """Returns the monthly OpEx expenditures based on project size."""

        rate = self.project_params.get("opex_rate", 150)
        lifetime = self.project_params.get("project_lifetime", 25)

        try:
            times, turbines = self.progress.energize_points
            dig = list(np.digitize(times, self.month_bins))

        except ValueError:
            return {i: 0.0 for i in range(1, lifetime * 12)}

        opex = {}
        for i in range(1, lifetime * 12):
            generating_strings = len([t for t in dig if i >= t])
            generating_turbines = sum(turbines[:generating_strings])

            opex[i] = generating_turbines * self.turbine_rating * rate * 1000 / 12

        return opex

    @property
    def monthly_revenue(self):
        """Returns the monthly revenue based on when array system strings can
        be energized, eg. 'self.progress.energize_points'."""

        ncf = self.project_params.get("ncf", 0.4)
        price = self.project_params.get("offtake_price", 80)
        lifetime = self.project_params.get("project_lifetime", 25)

        times, turbines = self.progress.energize_points
        dig = list(np.digitize(times, self.month_bins))

        revenue = {}
        for i in range(1, lifetime * 12):
            generating_strings = len([t for t in dig if i >= t])
            generating_turbines = sum(turbines[:generating_strings])
            production = generating_turbines * self.turbine_rating * ncf * 730  # MWh
            revenue[i] = production * price

        return revenue

    @property
    def cash_flow(self):
        """Returns the net cash flow based on `self.monthly_expenses` and
        `self.monthly_revenue`."""

        try:
            revenue = self.monthly_revenue

        except ValueError:
            revenue = {}

        expenses = self.monthly_expenses
        return {
            i: revenue.get(i, 0.0) - expenses.get(i, 0.0)
            for i in range(1, max([*revenue.keys(), *expenses.keys()]) + 1)
        }

    @property
    def npv(self):
        """Returns the net present value of the project based on
        `self.cash_flow`."""

        dr = self.project_params.get("discount_rate", 0.025)
        pr = (1 + dr) ** (1 / 12) - 1

        cash_flow = self.cash_flow
        _npv = [(cash_flow[i] / (1 + pr) ** (i)) for i in range(1, max(cash_flow.keys()) + 1)]

        return (self.total_capex - self.installation_capex) - sum(_npv)

    @property
    def progress_logs(self):
        """Returns logs of progress points."""

        return self._filter_logs(keys=["progress", "time"])

    def _filter_logs(self, keys):
        """Returns filtered list of logs."""

        filtered = []
        for l in self.logs:
            try:
                filtered.append(tuple(l[k] for k in keys))

            except KeyError:
                pass

        return filtered

    @property
    def progress_summary(self):
        """Returns a summary of progress by month."""

        arr = np.array(self.progress_logs, dtype=[("progress", "U32"), ("time", "i4")])
        dig = np.digitize(arr["time"], self.month_bins)

        summary = {}
        for i in range(1, len(self.month_bins)):

            unique, counts = np.unique(arr["progress"][dig == i], return_counts=True)
            summary[i] = dict(zip(unique, counts))

        return summary

    @property
    def actions(self):
        """Returns list of all actions in the project."""

        actions = [l for l in self.logs if l["level"] == "ACTION"]
        return sorted(actions, key=lambda l: l["time"])

    @staticmethod
    def create_input_xlsx():
        """
        A wrapper around self.compile_input_dict that produces an excel input
        file instead of a .json file.
        """
        pass

    @property
    def phase_dates(self):
        """
        Returns a combination of input start dates and `self.phase_times`.
        """

        if not isinstance(self.config["install_phases"], dict):
            print("Project was not configured with start dates.")
            return None

        dates = {}

        for phase, _start in self.config["install_phases"].items():

            start = dt.datetime.strptime(_start, self.date_format_short)
            end = start + dt.timedelta(hours=ceil(self.phase_times[phase]))

            dates[phase] = {
                "start": start.strftime(self.date_format_long),
                "end": end.strftime(self.date_format_long),
            }

        return dates

    @property
    def installation_time(self):
        """
        Returns sum of installation module times. This does not consider
        overlaps if phase dates are supplied.
        """

        res = sum(
            [v for k, v in self.phase_times.items() if k in self.config["install_phases"] and isinstance(v, Number)]
        )
        return res

    @property
    def project_days(self):
        """
        Returns days elapsed during installation phases accounting for
        overlapping phases.
        """

        dates = self.phase_dates
        starts = [d["start"] for _, d in dates.items()]
        ends = [d["end"] for _, d in dates.items()]
        return max([self._diff_dates_long(*p) for p in product(starts, ends)])

    def _diff_dates_long(self, a, b):
        """Returns the difference of two dates in `self.date_format_long`."""

        if not isinstance(a, dt.datetime):
            a = dt.datetime.strptime(a, self.date_format_long)

        if not isinstance(b, dt.datetime):
            b = dt.datetime.strptime(b, self.date_format_long)

        return abs((a - b).days)

    @property
    def overnight_capex_per_kw(self):
        """
        Returns overnight CAPEX/kW.
        """

        try:
            capex = self.overnight_capex / (self.capacity * 1000)

        except TypeError:
            capex = None

        return capex

    @property
    def system_capex(self):
        """Returns total system procurement CapEx."""

        return np.nansum([c for _, c in self.system_costs.items()])

    @property
    def system_capex_per_kw(self):
        """Returns system CapEx/kW."""

        try:
            capex = self.system_capex / (self.capacity * 1000)

        except TypeError:
            capex = None

        return capex

    @property
    def installation_capex(self):
        """Returns total installation related CapEx."""

        return np.nansum([c for _, c in self.installation_costs.items()])

    @property
    def installation_capex_per_kw(self):
        """Returns installation CapEx/kW."""

        try:
            capex = self.installation_capex / (self.capacity * 1000)

        except TypeError:
            capex = None

        return capex

    @property
    def capex_breakdown(self):
        """Returns CapEx breakdown by category."""

        unique = np.unique([*self.system_costs.keys(), *self.installation_costs.keys()])
        categories = {}

        for phase in unique:
            for base, cat in self._capex_categories.items():
                if base in phase:
                    categories[phase] = cat
                    break

        missing = [p for p in unique if p not in categories.keys()]
        if missing:
            print(f"Warning: CapEx category not found for {missing}. " f"Added to 'Misc.'")

            for phase in missing:
                categories[phase] = "Misc."

        outputs = {}
        for phase, cost in self.system_costs.items():
            name = categories[phase]
            if name in outputs.keys():
                outputs[name] += cost

            else:
                outputs[name] = cost

        for phase, cost in self.installation_costs.items():
            name = categories[phase] + " Installation"
            if name in outputs.keys():
                outputs[name] += cost

            else:
                outputs[name] = cost

        outputs["Turbine"] = self.turbine_capex
        outputs["Soft"] = self.soft_capex
        outputs["Project"] = self.project_capex

        return outputs

    @property
    def capex_breakdown_per_kw(self):
        """Returns CapEx per kW breakdown by category."""

        return {k: v / (self.capacity * 1000) for k, v in self.capex_breakdown.items()}

    @property
    def bos_capex(self):
        """Returns total balance of system CapEx."""

        return self.system_capex + self.installation_capex

    @property
    def bos_capex_per_kw(self):
        """Returns balance of system CapEx/kW."""

        try:
            capex = self.bos_capex / (self.capacity * 1000)

        except TypeError:
            capex = None

        return capex

    @property
    def turbine_capex(self):
        """Returns the total turbine CAPEX."""

        _capex = self.project_params.get("turbine_capex", 1300)
        try:
            num_turbines = self.config["plant"]["num_turbines"]
            rating = self.config["turbine"]["turbine_rating"]

        except KeyError:
            raise KeyError(
                f"Total turbine CAPEX can't be calculated. Required "
                f"parameters 'plant.num_turbines' or 'turbine.turbine_rating' "
                f"not found."
            )

        capex = _capex * num_turbines * rating * 1000
        return capex

    @property
    def turbine_capex_per_kw(self):
        """Returns the turbine CapEx/kW."""

        _capex = self.project_params.get("turbine_capex", 1300)
        return _capex

    @property
    def overnight_capex(self):
        """Returns the overnight capital cost of the project."""

        return self.system_capex + self.turbine_capex

    @property
    def soft_capex(self):
        """Returns total project cost costs."""

        try:
            capex = self.soft_capex_per_kw * self.capacity * 1000

        except TypeError:
            capex = None

        return capex

    @property
    def soft_capex_per_kw(self):
        """
        Returns project soft costs per kW. Default numbers are based on the
        Cost of Energy Review (Stehly and Beiter 2018).
        """

        insurance = self.project_params.get("construction_insurance", 44)
        financing = self.project_params.get("construction_financing", 183)
        contingency = self.project_params.get("contingency", 316)
        commissioning = self.project_params.get("commissioning", 44)
        decommissioning = self.project_params.get("decommissioning", 58)

        return sum([insurance, financing, contingency, commissioning, decommissioning])

    @property
    def project_capex(self):
        """
        Returns project related CapEx line items. To override the defaults,
        the keys below should be passed to the 'project_parameters' subdict.
        """

        site_auction = self.project_params.get("site_auction_price", 100e6)
        site_assessment = self.project_params.get("site_assessment_cost", 50e6)
        construction_plan = self.project_params.get("construction_plan_cost", 1e6)
        installation_plan = self.project_params.get("installation_plan_cost", 0.25e6)

        return sum(
            [
                site_auction,
                site_assessment,
                construction_plan,
                installation_plan,
            ]
        )

    @property
    def project_capex_per_kw(self):
        """Returns project related CapEx per kW."""

        try:
            capex = self.project_capex / (self.capacity * 1000)

        except TypeError:
            capex = None

        return capex

    @property
    def total_capex(self):
        """Returns total project CapEx including soft costs."""

        return self.bos_capex + self.turbine_capex + self.soft_capex + self.project_capex

    @property
    def total_capex_per_kw(self):
        """Returns total CapEx/kW."""

        try:
            capex = self.total_capex / (self.capacity * 1000)

        except TypeError:
            capex = None

        return capex

    def export_configuration(self, file_name):
        """
        Exports the configuration settings for the project to
        `library_path/project/config/file_name.yaml`.

        Parameters
        ----------
        file_name : str
            Name to use for the file.
        """

        export_library_specs("config", file_name, self.config)

    def export_project_logs(self, filepath, level="ACTION"):
        """
        Exports the project logs to a .csv file.

        Parameters
        ----------
        filepath : str
            Filepath to save logs at.
        level : str, optional
            Log level to save.
            Options: 'ACTION' | 'DEBUG'
            Default: 'ACTION'
        """

        dirs = os.path.split(filepath)[0]
        if dirs and not os.path.isdir(dirs):
            os.makedirs(dirs)

        if level == "ACTION":
            out = pd.DataFrame(self.actions)

        elif level == "DEBUG":
            out = pd.DataFrame(self.logs)

        else:
            raise ValueError(f"Unrecognized level '{level}'." " Must be 'ACTION' or 'DEBUG'.")

        out.to_csv(filepath, index=False)


class ProjectProgress:
    """Class to store, parse and return project progress data."""

    def __init__(self, data):
        """
        Creates an instance of `ProjectProgress`.

        Parameters
        ----------
        data : list
            List of tuples representing progress points of the project.
        """

        self.data = data

    @property
    def complete_export_system(self):
        """
        Returns project time when the export system and offshore substations(s)
        installations were completed (max of individual values).
        """

        export = self.parse_logs("Export System")
        substations = self.parse_logs("Offshore Substation")

        return max([*export, *substations])

    @property
    def complete_array_strings(self):
        """
        Returns list of times that the array strings and associated
        substructure/turbine assembly installations were completed.
        """

        strings = self.parse_logs("Array String")
        _subs = self.parse_logs("Substructure")
        _turbines = self.parse_logs("Turbine")

        per_string = len(_turbines) // len(strings)
        if len(_turbines) % len(strings):
            per_string += 1

        subs = self.chunk_max(_subs, per_string)
        turbines = self.chunk_max(_turbines, per_string)
        num_turbines = list(self.chunk_len(_turbines, per_string))

        data = list(zip(strings, subs, turbines))

        return [max(l) for l in data], num_turbines

    @property
    def energize_points(self):
        """
        Returns list of times where an array string can be energized. Max of
        each value in `self.complete_array_strings` and
        `self.complete_export_system`.
        """

        export = self.complete_export_system
        points = []

        times, turbines = self.complete_array_strings
        for t in times:
            points.append(max([t, export]))

        return points, turbines

    def parse_logs(self, k):
        """Parse `self.data` for specific progress points associated key `k`"""

        pts = [p[1] for p in self.data if p[0] == k]
        if not pts:
            raise ValueError(f"Installed '{k}' not found in project logs.")

        return pts

    @staticmethod
    def chunk_max(l, n):
        """Yield max value of successive n-sized chunks from l."""

        for i in range(0, len(l), n):
            yield max(l[i : i + n])

    @staticmethod
    def chunk_len(l, n):
        """Yield successive n-sized chunks from l."""

        for i in range(0, len(l), n):
            yield len(l[i : i + n])
