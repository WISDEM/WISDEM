"""Common processes and cargo types for Offshore Substation installations."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from marmot import process

from wisdem.orbit.core import Cargo
from wisdem.orbit.core.logic import stabilize, jackdown_if_required
from wisdem.orbit.core.defaults import process_times as pt
from wisdem.orbit.phases.install.monopile_install.common import (
    bolt_transition_piece,
    cure_transition_piece_grout,
    pump_transition_piece_grout,
)


class Topside(Cargo):
    """Topside Cargo"""

    def __init__(self, mass=None, deck_space=None, **kwargs):
        """
        Creates an instance of `Topside`.
        """

        self.mass = mass
        self.deck_space = deck_space

    @staticmethod
    def fasten(**kwargs):
        """Returns time required to fasten a topside at port."""

        key = "topside_fasten_time"
        time = kwargs.get(key, pt[key])

        return "Fasten Topside", time

    @staticmethod
    def release(**kwargs):
        """Returns time required to release topside from fastenings."""

        key = "topside_release_time"
        time = kwargs.get(key, pt[key])

        return "Release Topside", time


class Jacket(Cargo):
    """Jacket Cargo"""

    pass


@process
def lift_topside(vessel, **kwargs):
    """
    Calculates time required to lift topside at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.

    Yields
    ------
    vessel.task representing time to "Lift Topside".
    """

    lift_height = 5  # small lift just to clear the deck
    crane_rate = vessel.crane.crane_rate(**kwargs)
    lift_time = lift_height / crane_rate

    yield vessel.task_wrapper("Lift Topside", lift_time, constraints=vessel.operational_limits)


@process
def attach_topside(vessel, **kwargs):
    """
    Returns time required to attach topside at site.

    Parameters
    ----------
    vessel : Vessel
        Vessel to perform action.
    topside_attach_time : int | float
        Time required to attach topside.

    Yields
    ------
    vessel.task representing time to "Attach Topside".
    """

    _ = vessel.crane

    key = "topside_attach_time"
    attach_time = kwargs.get(key, pt[key])

    yield vessel.task_wrapper("Attach Topside", attach_time, constraints=vessel.operational_limits)


@process
def install_topside(vessel, topside, **kwargs):
    """
    Substation topside installation process.
    Subprocesses:
    - Crane reequip
    - Lift topside
    - Attach topside to substructure
    - Pump grout
    - Cure grout

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    topsdie : dict
    """

    connection = kwargs.get("topside_connection_type", "bolted")
    reequip_time = vessel.crane.reequip(**kwargs)

    yield vessel.task_wrapper(
        "Crane Reequip",
        reequip_time,
        constraints=vessel.transit_limits,
        **kwargs,
    )
    yield lift_topside(vessel)
    yield attach_topside(vessel)

    if connection == "bolted":
        yield bolt_transition_piece(vessel, **kwargs)

    elif connection == "grouted":

        yield pump_transition_piece_grout(vessel, **kwargs)
        yield cure_transition_piece_grout(vessel, **kwargs)

    else:
        raise Exception(
            f"Transition piece connection type '{connection}'" "not recognized. Must be 'bolted' or 'grouted'."
        )

    yield jackdown_if_required(vessel, **kwargs)
