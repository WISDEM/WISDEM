"""Custom exceptions used throughout ORBIT."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os


class MissingComponent(Exception):
    """Error for a missing component on a vessel."""

    def __init__(self, vessel, component):
        """
        Creates an instance of MissingComponent.

        Parameters
        ----------
        vessel : Vessel
        component : str
            Missing required component.
        """

        self.vessel = vessel
        self.component = component

        self.message = (
            f"{vessel} is missing required component(s) '{component}'."
        )

    def __str__(self):

        return self.message


class ItemNotFound(Exception):
    """Error for when no items in list satisfy rule"""

    def __init__(self, rule):
        """
        Creates an instance of ItemNotFound.

        Parameters
        ----------
        _key : str
        _value : varies
        """

        self.message = f"No items found that satisfy: {rule}"

    def __str__(self):
        return self.message


class DeckSpaceExceeded(Exception):
    """Error for exceeding vessel maximum deck space"""

    def __init__(self, max, current, item):
        """
        Creates an instance of DeckSpaceExceeded.

        Parameters
        ----------
        max : int | float
            Maximum vessel deck space (m2).
        current : int | float
            Vessel deck space currently in use (m2).
        item : dict
            Item that exceeded deck space limit.
        """

        self.max = max
        self.current = current
        self.item = item

        self.message = f"'{self.item['type']}' will exceed maximum deck space."

    def __str__(self):
        return self.message


class CargoMassExceeded(Exception):
    """Error for exceeding vessel maximum cargo mass"""

    def __init__(self, max, current, item):
        """
        Creates an instance of CargoMassExceeded.

        Parameters
        ----------
        max : int | float
            Maximum vessel cargo mass (t).
        current : int | float
            Vessel cargo mass currently in use (t).
        item : dict or str
            Item that exceeded cargo mass limit. Item can either be
            a dictionary with a 'type' or the name of an item.
        """

        self.max = max
        self.current = current
        self.item = item
        self.message = f"'{self.item}' will exceed maximum cargo mass."

    def __str__(self):
        return self.message


class ItemPropertyNotDefined(Exception):
    """Exception for incorrectly defined items."""

    def __init__(self, item, required):
        """
        Creates an instance of ItemPropertyNotDefined.

        Parameters
        ----------
        item : dict
            Bad item that was passed to VesselStorage.
        required : list
            Required keys in 'item'.
        """

        missing = [k for k in required if k not in item.keys()]
        self.missing = missing
        self.message = f"{item} is missing {self.missing}"

    def __str__(self):
        return self.message


class InsufficientAmount(Exception):
    """Error for not containing enough item in storage when requested."""

    def __init__(self, current_amount, item_type, amount_requested):
        """
        Creates an instance of InsufficientAmount.

        Parameters
        ----------
        current_amount : int | float
            Current amount of item in vessel storage.
        item_type : str
            Name or type of item.
        amount_requested : dict or str
            Amount of item attempting to remove from vessel storage.
        """

        self.current_amount = current_amount
        self.item_type = item_type
        self.amount_requested = amount_requested

        required = self.amount_requested - self.current_amount
        self.message = (
            f"Not enough '{self.item_type}' on vessel. At least "
            f"{required:.4e} more units required"
        )

    def __str__(self):
        return self.message


class InsufficientCable(Exception):
    """
    Error raised when a Carousel doesn't have enough cable for next section.
    """

    def __init__(self, current_amount, amount_requested):
        """
        Creates an instance of InsufficientAmount.

        Parameters
        ----------
        current_amount : int | float
            Current length of cable.
        amount_requested : dict or str
            Amount of cable needed
        """

        self.current = current_amount
        self.requested = amount_requested
        self.message = f"Not enough cable on carousel."

    def __str__(self):
        return self.message


class PhaseNotFound(Exception):
    """Exception for missing Phase"""

    def __init__(self, p):
        """
        Creates an instance of PhaseNotFound.

        Parameters
        ----------
        p : str
            Phase name.
        """

        self.phase = p
        self.message = f"Unrecognized phase '{self.phase}'."

    def __str__(self):
        return self.message


class MissingInputs(Exception):
    """Exception for missing input parameters."""

    def __init__(self, k):
        """
        Creates an instance of MissingInputs.

        Parameters
        ----------
        k : str
            Missing keys.
        """

        self.keys = k
        self.message = f"Input(s) '{self.keys}' missing in config."

    def __str__(self):
        return self.message


class WeatherProfileError(Exception):
    """Exception for weather profile errors."""

    def __init__(self, start, weather):
        """
        Creates an instance of WeatherProfileError.

        Parameters
        ----------
        start : datetime
            Starting index for output weather profile.
        weather : DataFrame
            Master weather profile.
        """

        self.start = start
        self.weather = weather
        self.message = (
            f"Timestep '{self.start}' not contained within input weather:\n"
            f"\tStart: '{self.weather.index[0]}'\n"
            f"\tEnd: '{self.weather.index[-1]}'"
        )

    def __str__(self):
        return self.message


class LibraryItemNotFoundError(Exception):
    """Error for missing library data"""

    def __init__(self, sub_dir, name):
        """
        Creates the `dir` object to be referenced by inherited errors.

        Parameters
        ----------
        sub_dir : str
            Library subfolder.
        name : str
            Filename of item to be extracted.
        """

        self.dir = os.path.join(os.environ["DATA_LIBRARY"], sub_dir)
        self.name = name
        self.message = f"{self.name} not found in {self.dir}."

    def __str__(self):
        return self.message


class WeatherWindowNotFound(Exception):
    """Error for tasks that do not have a valid weather window."""

    def __init__(self, agent, duration, max_windspeed, max_waveheight):
        """
        Creates an instance of WeatherWindowNotFound.

        Parameters
        ----------
        agent : str
            Name of agent performing action. For logging the delay.
        duration : int | float
            Time to complete action (h).
            If a float is passed in, it is rounded up to the nearest int.
        max_windspeed : int | float
            Maximum windspeed that action can be completed in (m/s).
        max_waveheight : int | float
            Maximum waveheight that action can be completed in (m).
        """

        self.agent = agent
        self.duration = duration
        self.max_windspeed = max_windspeed
        self.max_waveheight = max_waveheight

        self.message = (
            "No weather window found for '{}' that satisfies:"
            "\n\tMaximum Windspeed: {:.2f}"
            "\n\tMaximum Waveheight: {:.2f}"
            "\n\tDuration: {:.2f}"
            "".format(agent, max_windspeed, max_waveheight, duration)
        )

    def __str__(self):
        return self.message


class WeatherProfileExhausted(Exception):
    """
    Error to be raised at the end of the weather data.
    """

    def __init__(self, length):
        """
        Creates an instance of WeatherProfileExhausted.

        Parameters
        ----------
        length : int
            Total number of elements in the weather profile.
        """

        self.length = length

        self.message = "Weather profile exhausted at element {:,.0f}".format(
            length
        )

    def __str__(self):
        return self.message


class VesselCapacityError(Exception):
    """
    Error for a vessel that isn't configured large enough to carry any sets of
    items from port to site.
    """

    def __init__(self, vessel, items):
        """
        Creates an instance of VesselCapacityError.

        Parameters
        ----------
        vessel : Vessel
        """

        self.vessel = vessel
        self.items = items

        self.message = (
            f"Vessel {self.vessel} does not have the required "
            "cargo mass or deck space capacity to transport a "
            f"whole set of components: {self.items}"
        )

    def __str__(self):
        return self.message


class FastenTimeNotFound(Exception):
    """Error for an item that doesn't have a defined fasten time."""

    def __init__(self, item):
        """
        Creates an instance of FastenTimeNotFound

        Parameters
        ----------
        item : str
            Item name
        """

        self.item = item

        self.message = f"Unknown fasten time for item type '{item}'."

    def __str__(self):
        return self.message


class PhaseDependenciesInvalid(Exception):
    """Error for phase dependencies that can't be resolved."""

    def __init__(self, phases):
        """
        Creates an instance of PhaseDependenciesInvalid.

        Parameters
        ----------
        phases : dict
            Invalid phases.
        """

        self.phases = phases

        self.message = f"Phase dependencies {phases} are not resolvable."

    def __str__(self):
        return self.message
