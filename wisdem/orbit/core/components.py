"""Provides the `Crane` class."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

import simpy

from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.core.exceptions import ItemNotFound, InsufficientCable

# TODO: __str__ methods for Components


class Crane:
    """Base Crane Class"""

    def __init__(self, crane_specs):
        """
        Creates an instance of Crane.

        Parameters
        ----------
        crane_specs : dict
            Dictionary containing crane system specifications.
        """

        self.extract_crane_specs(crane_specs)

    def extract_crane_specs(self, crane_specs):
        """
        Extracts and defines crane specifications.

        Parameters
        ----------
        crane_specs : dict
            Dictionary of crane specifications.
        """

        # Operational Parameters
        self.max_lift = crane_specs.get("max_lift", None)
        self.max_hook_height = crane_specs.get("max_hook_height", None)
        self.max_windspeed = crane_specs.get("max_windspeed", 99)
        self._crane_rate = crane_specs.get("crane_rate", 100)

    def crane_rate(self, **kwargs):
        """Returns `self._crane_rate`."""

        return self._crane_rate

    @staticmethod
    def reequip(**kwargs):
        """
        Calculates time taken to change crane equipment.

        Parameters
        ----------
        crane_reequip_time : int | float
            Time required to change crane equipment (h).

        Returns
        -------
        reequip_time : float
            Time required to change crane equipment (h).
        """

        _key = "crane_reequip_time"
        duration = kwargs.get(_key, pt[_key])

        return duration


class DynamicPositioning:
    """Base Dynamic Positioning Class"""

    def __init__(self, dp_specs):
        """
        Creates an instance of DynamicPositioning.

        Parameters
        ----------
        dp_specs : dict
            Dictionary containing dynamic positioning specs.
        """

        self.extract_dp_specs(dp_specs)

    def extract_dp_specs(self, dp_specs):
        """
        Extracts and defines jacking system specifications.

        Parameters
        ----------
        jacksys_specs : dict
            Dictionary containing jacking system specifications.
        """

        self.dp_class = dp_specs.get("class", 1)


class JackingSys:
    """Base Jacking System Class"""

    def __init__(self, jacksys_specs):
        """
        Creates an instance of JackingSys.

        Parameters
        ----------
        jacksys_specs : dict
            Dictionary containing jacking system specifications.
        """

        self.extract_jacksys_specs(jacksys_specs)

    def extract_jacksys_specs(self, jacksys_specs):
        """
        Extracts and defines jacking system specifications.

        Parameters
        ----------
        jacksys_specs : dict
            Dictionary containing jacking system specifications.
        """

        # Physical Dimensions
        self.leg_length = jacksys_specs.get("leg_length", None)
        self.air_gap = jacksys_specs.get("air_gap", None)
        self.leg_pen = jacksys_specs.get("leg_pen", None)

        # Operational Parameters
        self.max_depth = jacksys_specs.get("max_depth", None)
        self.max_extension = jacksys_specs.get("max_extension", None)
        self.speed_below_depth = jacksys_specs.get("speed_below_depth", None)
        self.speed_above_depth = jacksys_specs.get("speed_above_depth", None)

    def jacking_time(self, extension, depth):
        """
        Calculates jacking time for a given depth.

        Parameters
        ----------
        extension : int | float
            Height to jack-up to or jack-down from (m).
        depth : int | float
            Depth at jack-up location (m).

        Returns
        -------
        extension_time : float
            Time required to jack-up to given extension (h).
        """

        if extension > self.max_extension:
            raise Exception("{} extension is greater than {} maximum" "".format(extension, self.max_extension))

        elif depth > self.max_depth:
            raise Exception("{} is beyond the operating depth {}" "".format(depth, self.max_depth))

        elif depth > extension:
            raise Exception("Extension must be greater than depth")

        else:
            return (depth / self.speed_below_depth + (extension - depth) / self.speed_above_depth) / 60


class VesselStorage(simpy.FilterStore):
    """Vessel Storage Class"""

    required_keys = ["type", "mass", "deck_space"]

    def __init__(self, env, max_cargo, max_deck_space, max_deck_load, **kwargs):
        """
        Creates an instance of VesselStorage.

        Parameters
        ----------
        env : simpy.Environment
            SimPy environment that simulation runs on.
        max_cargo : int | float
            Maximum mass the storage system can carry (t).
        max_deck_space : int | float
            Maximum deck space the storage system can use (m2).
        max_deck_load : int | float
            Maximum deck load that the storage system can apply (t/m2).
        """

        capacity = kwargs.get("capacity", float("inf"))
        super().__init__(env, capacity)

        self.max_cargo_mass = max_cargo
        self.max_deck_space = max_deck_space
        self.max_deck_load = max_deck_load

    @property
    def current_cargo_mass(self):
        """Returns current cargo mass in tons."""

        return sum([item.mass for item in self.items])

    @property
    def current_deck_space(self):
        """Returns current deck space used in m2."""

        return sum([item.deck_space for item in self.items])

    def put_item(self, item):
        """
        Checks VesselStorage specific constraints and triggers self.put()
        if successful.

        Items put into the instance should be a dictionary with the following
        attributes:
        - name
        - mass (t)
        - deck_space (m2)

        Parameters
        ----------
        item : dict
            Dictionary of item properties.
        """

        self.put(item)

    def get_item(self, _type):
        """
        Checks `self.items` for an item satisfying `item.type = _type`. Returns
        item if found, otherwise returns an error.

        Parameters
        ----------
        _type : str
            Type of item to retrieve.
        """

        target = None
        for i in self.items:
            if i.type == _type:
                target = i
                break

        if not target:
            raise ItemNotFound(_type)

        else:
            res = self.get(lambda x: x == target)
            return res.value

    def any_remaining(self, _type):
        """
        Checks `self.items` for an item satisfying `item.type = _type`. Returns
        True/False depending on if an item is found. Used to trigger vessel
        release if empty without having to wait for next self.get_item()
        iteration.

        Parameters
        ----------
        _type : str
            Type of item to retrieve.

        Returns
        -------
        resp : bool
            Indicates if any items in self.items satisfy `_type`.
        """

        target = None
        for i in self.items:
            if i.type == _type:
                target = i
                break

        if target:
            return True

        else:
            return False


class ScourProtectionStorage(simpy.Container):
    """Scour Protection Storage Class"""

    def __init__(self, env, max_mass, **kwargs):
        """
        Creates an instance of VesselStorage.

        Parameters
        ----------
        env : simpy.Environment
            SimPy environment that simulation runs on.
        max_mass : int | float
            Maximum mass the storage system can carry (t).
        """

        self.max_mass = max_mass
        super().__init__(env, self.max_mass)

    @property
    def available_capacity(self):
        """Returns available cargo capacity."""

        return self.max_mass - self.level


class CableCarousel(simpy.Container):
    """Cable Storage Class"""

    def __init__(self, env, max_mass, **kwargs):
        """
        Creates an instance of CableCarousel.

        Parameters
        ----------
        env : simpy.Environment
            SimPy environment that simulation runs on.
        max_mass : int | float
            Maximum mass the storage system can carry (t).
        """

        self.cable = None
        self.max_mass = max_mass
        super().__init__(env)

    @property
    def available_mass(self):
        """Returns available cargo mass capacity."""

        return self.max_mass - self.current_mass

    @property
    def current_mass(self):
        """Returns current cargo mass"""

        try:
            mass = self.level * self.cable.linear_density
            return mass

        except AttributeError:
            return 0

    def available_length(self, cable):
        """Returns available length capacity based on input linear density."""

        return self.available_mass / cable.linear_density

    def reset(self):
        """Resets `self.cable` and empties `self.level`."""

        if self.level != 0.0:
            _ = self.get(self.level)

        self.cable = None

    def load_cable(self, cable, length=None):
        """
        Loads input `cable` type onto `self.level`. If `length` isn't passed,
        defaults to maximum amount of cable that can be loaded.

        Parameters
        ----------
        cable : Cable | SimpleCable
        length : int | float

        Raises
        ------
        ValueError
        """

        if self.cable and self.cable != cable:
            raise AttributeError("Carousel already has a cable type.")

        self.cable = cable
        if length is None:
            # Load maximum amount
            length = self.available_length(self.cable)
            self.put(length)

        else:
            # Load length of cable
            proposed = length * cable.linear_density
            if proposed > self.available_mass:
                raise ValueError(f"Length {length} of {cable} can't be loaded.")

            self.put(length)

    def get_cable(self, length):
        """
        Retrieves `length` of cable from `self.level`.

        Parameters
        ----------
        length : int | float
            Length of cable to retrieve.

        Raises
        ------
        InsufficientCable
        """

        if self.cable is None:
            raise AttributeError("Carousel doesn't have any cable.")

        if length > self.level:
            raise InsufficientCable(self.level, length)

        else:
            return self.get(length).amount
