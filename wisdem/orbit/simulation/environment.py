"""Provides the `Environment` class."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import numpy as np
import simpy

from .exceptions import WeatherWindowNotFound, WeatherProfileExhausted


class Environment(simpy.Environment):
    """Extension of SimPy.Environment with weather."""

    logger = None

    def __init__(self, weather=None):
        """
        Creates an instance of Environment.

        Parameters
        ----------
        weather : DataFrame | False
            DataFrame of weather timeseries.
            Currently expects 'windspeed' and 'waveheight'.
        """

        super().__init__()

        self._weather = weather

    def task_handler(self, tasks):
        """
        Handler for an action or a list of actions.

        Parameters
        ----------
        tasks : dict | list
            List of tasks or a single task passed as a dictionary.
        """

        if isinstance(tasks, dict):
            task = tasks
            yield self.process(self._schedule_task(task))

        elif isinstance(tasks, list):
            for task in tasks:
                yield self.process(self._schedule_task(task))

        else:
            raise Exception("'tasks' was not passed as a dict or list.")

    def _schedule_task(self, task):
        """
        Handles a single task passed as a dictionary.

        Parameters
        ----------
        task : dict
            Dictionary of action parameters.
            Required keys:
            - 'agent': str | list
            - 'action': str
            - 'duration': int | float

        Yields
        ------
        delay : simpy.timeout
            Delay until clear weather window (h).
        duration : simpy.timeout
            Action duration (h).
        """

        # Operational limits
        max_windspeed = task.get("max_windspeed", 99)
        max_waveheight = task.get("max_waveheight", 99)

        # Required parameters
        _required = ["agent", "action", "duration"]
        if not all(i in task.keys() for i in _required):
            _missing = [i for i in _required if i not in task.keys()]

            raise Exception(
                "Parameters {} missing from task {}" "".format(_missing, task)
            )

        # Weather
        if self._weather is None:
            yield self.timeout(task["duration"])

        else:
            steps = int(np.ceil(task["duration"]))

            # Retreive weather forecast
            forecast = self.weather_forecast

            if len(forecast) < steps:
                raise WeatherProfileExhausted(len(self._weather))

            # Analyze forecast for task constraints
            forecast["passes"] = True
            forecast.loc[
                (forecast["windspeed"] > max_windspeed)
                | (forecast["waveheight"] > max_waveheight),
                ["passes"],
            ] = False

            # Find delay until next window that satisfies above conditions
            delay = self.find_next_window(forecast["passes"].values, steps)

            if delay is None:
                raise WeatherWindowNotFound(
                    task["agent"],
                    task["duration"],
                    max_windspeed,
                    max_waveheight,
                )

            # For debugging:
            # s = max([0, delay-6])
            # forecast = forecast.reset_index()
            # print(forecast.iloc[s:s+24, :])

            yield self.timeout(delay)
            if delay > 0:

                delay_info = {
                    **task,
                    "type": "Delay",
                    "action": "WaitingForWeather",
                    "time": self.now,
                    "duration": delay,
                }

                if isinstance(task["agent"], str):
                    self.logger.info("", extra=delay_info)

                elif isinstance(task["agent"], list):
                    for a in task["agent"]:
                        delay_info["agent"] = a
                        self.logger.info("", extra=delay_info)

                else:
                    raise TypeError("task['agent'] not recongnized.")

            # Yield task duration after delay
            yield self.timeout(task["duration"])

        # Log task duration and information
        proc_info = {**task, "time": self.now}

        if isinstance(task["agent"], str):
            self.logger.info("", extra=proc_info)

        elif isinstance(task["agent"], list):
            for a in task["agent"]:
                proc_info["agent"] = a
                self.logger.info("", extra=proc_info)

    @staticmethod
    def find_next_window(arr, n):
        """
        Find next installation window of at least 'n' steps.

        Parameters
        ----------
        arr : Array
            Array of True/False.
        n : int
            Number of index steps required to complete task.
        """

        arr = np.append(arr, False)
        false = np.where(~arr)[0]

        if false.size == 0:
            delay = 0

        elif false[0] >= n:
            delay = 0

        else:
            diff = np.where((false[1:] - false[:-1] - 1) >= n)[0]

            try:
                delay = false[diff[0]] + 1

            except IndexError:
                delay = None

        return delay

    @property
    def weather_forecast(self):
        """
        Returns a slice of self._weather that starts at self.now.
        """

        start = int(np.ceil(self.now))
        return self._weather.iloc[start:].copy()
