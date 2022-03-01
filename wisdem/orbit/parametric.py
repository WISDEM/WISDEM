__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov"]


import re
import time
from copy import deepcopy
from random import sample
from itertools import product

import yaml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from yaml import Loader
from benedict import benedict
from wisdem.orbit import ProjectManager


class ParametricManager:
    """Class for configuring parametric ORBIT runs."""

    def __init__(
        self,
        base,
        params,
        funcs,
        weather=None,
        module=None,
        product=False,
        keep_inputs=[],
    ):
        """
        Creates an instance of `ParametricRun`.

        Parameters
        ----------
        base : dict
            Base ORBIT configuration.
        params : dict
            Parameters and their values.
            Format: "subdict.param": [num1, num2, num3]
        funcs : dict
            List of functions used to save results.
            Format: "name": lambda project: project.attr
        weather : DataFrame

        """

        self.base = benedict(base)
        self.params = params
        self.funcs = funcs
        self.weather = weather
        self.results = None
        self.module = module
        self.product = product
        self.keep = keep_inputs

    def run(self, **kwargs):
        """Run the configured parametric runs and save any requested results to
        `self.results`."""

        outputs = []
        for run in self.run_list:
            data = self._run_config(run, **kwargs)
            outputs.append(data)

        self.results = pd.DataFrame(outputs)

    def _run_config(self, run, **kwargs):
        """Run an individual config."""

        config = deepcopy(self.base)
        config.merge(run)

        if self.module is not None:
            project = self.module(config, weather=self.weather, **kwargs)
            project.run()

        else:
            project = ProjectManager(config, weather=self.weather, **kwargs)
            project.run()

        results = self.map_funcs(project, self.funcs)
        kept = self._get_kept_inputs(project.config)
        data = {**run, **kept, **results}

        return data

    def _get_kept_inputs(self, config):
        """
        Extract inputs in `self.keep` from `config`.

        Parameters
        ----------
        config : dict
        """

        kept = {}
        for k in self.keep:
            try:
                kept[k] = config[k]

            except KeyError:
                pass

        return kept

    @property
    def run_list(self):
        """Returns list of configured parametric runs."""

        if self.product:
            runs = list(product(*self.params.values()))

        else:
            runs = list(zip(*self.params.values()))

        return [dict(zip(self.params.keys(), run)) for run in runs]

    @property
    def num_runs(self):
        return len(self.run_list)

    @staticmethod
    def map_funcs(obj, funcs):
        """
        Map `obj` to list of `funcs`.

        Parameters
        ----------
        obj : ProjectManager
            Project instance to run through functions.
        funcs : list
            Functions used to pull results from obj.
        """

        results = {}
        for k, f in funcs.items():
            try:
                res = f(obj)

            except TypeError:
                raise TypeError(
                    f"Result function '{f}' not structured properly. " f"Correct format: 'lambda project: project.{f}'"
                )

            except AttributeError:
                res = np.NaN

            results[k] = res

        return results

    def preview(self, num=10, **kwargs):
        """
        Runs a limited set of runs to preview the results and provide an
        estimate for total run time.

        Parameters
        ----------
        num : int
            Number to run.
        """

        start = time.time()
        outputs = []
        if num > len(self.run_list):
            to_run = self.run_list

        else:
            to_run = sample(self.run_list, num)

        for run in to_run:
            data = self._run_config(run, **kwargs)
            outputs.append(data)

        elapsed = time.time() - start
        estimate = (len(self.run_list) / num) * elapsed
        print(f"{num} runs elapsed time: {elapsed:.2f}s")
        print(f"{self.num_runs} runs estimated time: {estimate:.2f}s")

        return pd.DataFrame(outputs)

    def create_model(self, x, y):
        """"""

        if self.results is None:
            print("`ParametricManager hasn't been ran yet.")

        return LinearModel(self.results, x, y)

    @classmethod
    def from_config(cls, data):
        """"""

        outputs = data.pop("outputs", {})

        funcs = {}
        for k, v in outputs.items():

            split = v.split("[")
            attr = split[0]

            try:
                key = re.sub("[^A-Za-z ]", "", split[1])

            except IndexError:
                key = None

            if key is None:
                funcs[k] = lambda run, a=attr: getattr(run, a)

            else:
                funcs[k] = lambda run, a=attr, k=key: getattr(run, a)[k]

        data["funcs"] = funcs

        module_name = data.pop("module", None)
        if module_name is not None:
            module = ProjectManager.phase_dict()[module_name]
            data["module"] = module

        weather_file = data.pop("weather", None)
        if weather_file is not None:
            weather = pd.read_csv(weather_file, parse_dates=["datetime"])
            data["weather"] = weather

        return cls(**data, product=True)


class LinearModel:
    """Simple linear regression model."""

    def __init__(self, data, x, y):
        """
        Creates an instance of `LinearModel`.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset.
        x : list
            List of regression variables.
        y : str
            Output variable.
        """

        self.data = data
        self.x = x
        self.y = y

        self.X = data[x]
        self.Y = data[y]

        self.X2 = sm.add_constant(self.X)
        self.sm = sm.OLS(self.Y, self.X2).fit()

    def predict(self, inputs):
        """
        Predicts the output value of `inputs` given the underlying linear model
        developed using `self.data`, `self.x` and `self.y`.

        Parameters
        ----------
        inputs : dict
            Inputs with form:
            'key': list | np.array | int | float
        """

        missing = [i for i in self.x if i not in inputs.keys()]
        if len(missing) > 0:
            raise ValueError(f"Missing input(s) '{missing}'")

        inputs = {k: np.array(v) for k, v in inputs.items()}
        params = deepcopy(self.sm.params)
        const = params.pop("const")

        return sum([v * inputs[k] for k, v in params.items()]) + const

    @property
    def as_string(self):
        """
        Returns the linear model represented as a string. To be used for
        interfacing with NRWAL.

        Returns
        -------
        str
        """

        params = deepcopy(self.sm.params)
        const = params.pop("const")

        out = ""
        for i, (k, v) in enumerate(params.items()):

            if i == 0:
                pre = ""

            else:
                if v >= 0:
                    pre = " + "

                else:
                    pre = " - "

            out += pre
            out += f"{abs(v)} * {k}"

        if const < 0:
            out += f" - {abs(const)}"

        else:
            out += f" + {const}"

        return out

    @property
    def vif(self):
        """
        Returns the variance inflation factor of the input data series.

        Returns
        -------
        list
        """

        data = self.X2.copy()

        vif = []
        for name, d in data.iteritems():
            r_sq_i = sm.OLS(d, data.drop(name, axis=1)).fit().rsquared
            vif.append(1.0 / (1.0 - r_sq_i))

        return vif

    @property
    def perc_diff(self):
        """
        Returns the percent difference between the predicted values and the
        output value.

        Returns
        -------
        pd.Series
        """

        inputs = dict(zip(self.X.T.index, self.X.T.values))
        predicted = self.predict(inputs)

        return (self.Y - predicted) / self.Y
