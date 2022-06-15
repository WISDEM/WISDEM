"""Common processes and cargo types for Monopile installations."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import process

from wisdem.orbit.core import Cargo
from wisdem.orbit.core.logic import jackdown_if_required
from wisdem.orbit.core.defaults import process_times as pt


class Monopile(Cargo):
    """Monopile Cargo"""

    def __init__(self, length=None, diameter=None, mass=None, deck_space=None, **kwargs):
        """
        Creates an instance of `Monopile`.
        """

        self.length = length
        self.diameter = diameter
        self.mass = mass
        self.deck_space = deck_space

    @staticmethod
    def fasten(**kwargs):
        """Returns time required to fasten a monopile at port."""

        key = "mono_fasten_time"
        time = kwargs.get(key, pt[key])

        return "Fasten Monopile", time

    @staticmethod
    def release(**kwargs):
        """Returns time required to release monopile from fastenings."""

        key = "mono_release_time"
        time = kwargs.get(key, pt[key])

        return "Release Monopile", time


class TransitionPiece(Cargo):
    """Transition Piece Cargo"""

    def __init__(self, mass=None, deck_space=None, **kwargs):
        """
        Creates an instance of `TransitionPiece`.
        """

        self.mass = mass
        self.deck_space = deck_space

    @staticmethod
    def fasten(**kwargs):
        """Returns time required to fasten a transition piece at port."""

        key = "tp_fasten_time"
        time = kwargs.get(key, pt[key])

        return "Fasten Transition Piece", time

    @staticmethod
    def release(**kwargs):
        """Returns time required to release transition piece from fastenings."""

        key = "tp_release_time"
        time = kwargs.get(key, pt[key])

        return "Release Transition Piece", time


@process
def upend_monopile(vessel, length, **kwargs):
    """
    Calculates time required to upend monopile to vertical position.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    length : int | float
        Overall length of monopile (m).

    Yields
    ------
    vessel.task representing time to "Upend Monopile".
    """

    crane_rate = vessel.crane.crane_rate(**kwargs)
    upend_time = length / crane_rate

    yield vessel.task_wrapper(
        "Upend Monopile",
        upend_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def lower_monopile(vessel, **kwargs):
    """
    Calculates time required to lower monopile to seafloor.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    site_depth : int | float
        Seafloor depth at site (m).

    Yields
    ------
    vessel.task representing time to "Lower Monopile".
    """

    depth = kwargs.get("site_depth", None)
    rate = vessel.crane.crane_rate(**kwargs)

    height = (depth + 10) / rate  # Assumed 10m deck height added to site depth
    lower_time = height / rate

    yield vessel.task_wrapper(
        "Lower Monopile",
        lower_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def drive_monopile(vessel, **kwargs):
    """
    Calculates time required to drive monopile into seafloor.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    mono_embed_len : int | float
        Monopile embedment length below seafloor (m).
    mono_drive_rate : int | float
        Driving rate (m/hr).

    Yields
    ------
    vessel.task representing time to "Drive Monopile".
    """

    _ = vessel.crane

    mono_embed_len = kwargs.get("mono_embed_len", pt["mono_embed_len"])
    mono_drive_rate = kwargs.get("mono_drive_rate", pt["mono_drive_rate"])

    drive_time = mono_embed_len / mono_drive_rate

    yield vessel.task_wrapper(
        "Drive Monopile",
        drive_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def lower_transition_piece(vessel, **kwargs):
    """
    Calculates time required to lower a transition piece onto monopile.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.

    Yields
    ------
    vessel.task representing time to "Lower Transition Piece".
    """

    yield vessel.task_wrapper("Lower TP", 1, constraints=vessel.operational_limits, **kwargs)


@process
def bolt_transition_piece(vessel, **kwargs):
    """
    Returns time required to bolt transition piece to monopile.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    tp_bolt_time : int | float
        Time required to attach transition piece.

    Yields
    ------
    vessel.task representing time to "Bolt TP".
    """

    key = "tp_bolt_time"
    bolt_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper("Bolt TP", bolt_time, constraints=vessel.operational_limits, **kwargs)


@process
def pump_transition_piece_grout(vessel, **kwargs):
    """
    Returns time required to pump grout at the transition piece interface.

    Parameters
    ----------
    grout_pump_time : int | float
        Time required to pump grout at the interface.

    Yields
    ------
    vessel.task representing time to "Pump TP Grout".
    """

    key = "grout_pump_time"
    pump_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper(
        "Pump TP Grout",
        pump_time,
        constraints=vessel.operational_limits,
        **kwargs,
    )


@process
def cure_transition_piece_grout(vessel, **kwargs):
    """
    Returns time required for the transition piece grout to cure.

    Parameters
    ----------
    grout_cure_time : int | float
        Time required for the grout to cure.

    Yields
    ------
    vessel.task representing time to "Cure TP Grout".
    """

    key = "grout_cure_time"
    cure_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper("Cure TP Grout", cure_time, constraints=vessel.transit_limits, **kwargs)


@process
def install_monopile(vessel, monopile, **kwargs):
    """
    Process logic for installing a monopile at site.

    Subprocesses:

    - Lower monopile, ``tasks.lower_monopile()``
    - Reequip crane, ``vessel.crane.reequip()``
    - Drive monopile, ``tasks.drive_monopile()``

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    monopile : dict
    """

    reequip_time = vessel.crane.reequip(**kwargs)

    yield lower_monopile(vessel, **kwargs)
    yield vessel.task_wrapper(
        "Crane Reequip",
        reequip_time,
        constraints=vessel.transit_limits,
        **kwargs,
    )
    yield drive_monopile(vessel, **kwargs)


@process
def install_transition_piece(vessel, tp, **kwargs):
    """
    Process logic for installing a transition piece on a monopile at site.

    Subprocesses:

    - Reequip crane, ``vessel.crane.reequip()``
    - Lower transition piece, ``tasks.lower_transition_piece()``
    - Install connection, see below.
    - Jackdown, ``vessel.jacksys.jacking_time()`` (if a jackup vessel)

    The transition piece can either be installed with a bolted or a grouted
    connection. By default, ORBIT uses the bolted connection with the following
    task:

    - Bolt transition piece, ``tasks.bolt_transition_piece()``

    ORBIT can also be configured to model a grouted connection by passing in
    `tp_connection_type='grouted'` as a `kwarg`. This process uses the
    following tasks:

    - Pump grout, ``tasks.pump_transition_piece_grout()``
    - Cure grout, ``tasks.cure_transition_piece_grout()``

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    tp : dict
    """

    connection = kwargs.get("tp_connection_type", "bolted")
    reequip_time = vessel.crane.reequip(**kwargs)

    yield vessel.task_wrapper(
        "Crane Reequip",
        reequip_time,
        constraints=vessel.transit_limits,
        **kwargs,
    )
    yield lower_transition_piece(vessel, **kwargs)

    if connection == "bolted":
        yield bolt_transition_piece(vessel, **kwargs)

    elif connection == "grouted":

        yield pump_transition_piece_grout(vessel, **kwargs)
        yield cure_transition_piece_grout(vessel)

    else:
        raise Exception(
            f"Transition piece connection type '{connection}'" "not recognized. Must be 'bolted' or 'grouted'."
        )

    yield jackdown_if_required(vessel, **kwargs)
