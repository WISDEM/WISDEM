"""ORBIT specific marmot.Environment."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from bisect import bisect

import numpy as np
from marmot import Environment
from marmot._core import Constraint
from numpy.lib.recfunctions import append_fields


class OrbitEnvironment(Environment):
    """ORBIT Specific Environment."""

    def __init__(self, name="Environment", state=None, **kwargs):
        """
        Creates an instance of Environment.

        Parameters
        ----------
        name : str
            Environment name.
            Default: 'Environment'
        state : array-like
            Time series representing the state of the environment throughout
            time or iterations.
        """

        super().__init__()

        self.name = name
        self.state = self.standarize_state_inputs(state)
        self.alpha = kwargs.get("ws_alpha", 0.1)
        self.default_height = kwargs.get("ws_default_height", 10)

        self._logs = []
        self._agents = {}
        self._objects = []

    def _find_valid_constraints(self, **kwargs):
        """
        Finds any constraitns in `kwargs` where the key matches a column name
        in `self.state` and the value type is `Constraint`.

        This method overrides the default method to handle windspeed
        constraints specifically, interpolating or extrapolating environment
        windspeed profiles to the input constraint height.

        Returns
        -------
        valid : dict
            Valid constraints that apply to a column in `self.state`.
        """

        c = {k: v for k, v in kwargs.items() if isinstance(v, Constraint)}
        constraints = self.resolve_windspeed_constraints(c)

        keys = set(self.state.dtype.names).intersection(
            set(constraints.keys())
        )
        valid = {k: v for k, v in constraints.items() if k in keys}

        return valid

    def standarize_state_inputs(self, _in):
        """
        Standardization routines applied to columns in `self.state`.

        Parameters
        ----------
        _in : array-like
            Time series representing the state of the environment throughout
            time or iterations.

        Returns
        -------
        state : array-like
            Time series with windspeed heights in simplest representation.
        """

        if _in is None:
            return None

        names = []
        for name in list(_in.dtype.names):

            if "windspeed" in name:
                try:
                    val = name.split("_")[1].replace("m", "")
                    new = f"windspeed_{self.simplify_num(val)}m"
                    names.append(new)

                except IndexError:
                    names.append(name)

            else:
                names.append(name)

        state = _in.copy()
        state.dtype.names = names
        return state

    def resolve_windspeed_constraints(self, constraints):
        """
        Resolves the applied windspeed constraints given the windspeed profiles
        that are present in `self.state`.

        Parameters
        ----------
        constraints : dict
            Dictionary of constraints

        Returns
        -------
        constraitns : dict
            Dictionary of constraints with windspeed columns resolved to their
            simplest representation.
        """

        ws = {}
        for k in list(constraints.keys()):
            if "windspeed" in k:
                ws[k] = constraints.pop(k)

        if not ws:
            return constraints

        if "windspeed" in self.state.dtype.names:
            if len(ws) > 1:
                raise ValueError(
                    "Multiple constraints applied to the 'windspeed' column."
                )

            return {**constraints, "windspeed": list(ws.values())[0]}

        for k, v in ws.items():

            if k == "windspeed":
                height = self.simplify_num(self.default_height)

            else:
                val = k.split("_")[1].replace("m", "")
                height = self.simplify_num(val)

            name = f"windspeed_{height}m"
            if name in self.state.dtype.names:
                pass

            else:
                loc = bisect(self.ws_heights, height)
                if loc == 0:
                    self.extrapolate_ws(self.ws_heights[0], height)

                elif loc == len(self.ws_heights):
                    self.extrapolate_ws(self.ws_heights[-1], height)

                else:
                    h1 = self.ws_heights[loc - 1]
                    h2 = self.ws_heights[loc]
                    self.interpolate_ws(h1, h2, height)

            constraints[name] = v

        return constraints

    @property
    def ws_heights(self):
        """Returns heights of available windspeed profiles."""

        columns = [c for c in self.state.dtype.names if "windspeed" in c]

        heights = []
        for c in columns:
            try:
                val = c.split("_")[1].replace("m", "")
                heights.append(float(val) if "." in val else int(val))

            except IndexError:
                pass

        return sorted(heights)

    def interpolate_ws(self, h1, h2, h):
        """
        Interpolates between two windspeed profiles using the power law. This
        method will calculate the power law coefficient between `h1` and `h2`.

        Parameters
        ----------
        h1 : int | float
            Lower measurement height.
        h2 : int | float
            Upper measurement height.
        h : int | float
            Desired profile height.
        """

        ts1 = self.state[f"windspeed_{h1}m"]
        ts2 = self.state[f"windspeed_{h2}m"]
        alpha = np.log(ts2.mean() / ts1.mean()) / np.log(h2 / h1)

        ts = ts1 * (h / h1) ** alpha

        self.state = np.array(append_fields(self.state, f"windspeed_{h}m", ts))

    def extrapolate_ws(self, h1, h):
        """
        Extrapolates a windspeed profile using the power law coefficient
        at `self.alpha`.

        Parameters
        ----------
        h1 : int | float
            Measurement height.
        h : int | float
            Desired profile height.
        """

        ts1 = self.state[f"windspeed_{h1}m"]
        ts = ts1 * (h / h1) ** self.alpha

        self.state = np.array(append_fields(self.state, f"windspeed_{h}m", ts))

    @staticmethod
    def simplify_num(str):
        """Returns the simplest str representation of a number."""

        num = float(str)
        if int(num) == num:
            return int(num)

        else:
            return num
