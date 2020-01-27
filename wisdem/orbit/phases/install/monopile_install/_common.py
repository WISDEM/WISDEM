"""Provides common functionality between monopile installation strategies."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.vessels import tasks


def install_monopile(env, vessel, monopile, **kwargs):
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

    lower_mono_time = tasks.lower_monopile(vessel, **kwargs)
    reequip_time = vessel.crane.reequip(**kwargs)
    drive_time = tasks.drive_monopile(vessel, **kwargs)

    _shared = {
        "agent": vessel.name,
        "location": "Site",
        "type": "Operations",
        **vessel.operational_limits,
    }

    task_list = [
        {"action": "LowerMonopile", "duration": lower_mono_time, **_shared},
        {"action": "CraneReequip", "duration": reequip_time, **_shared},
        {"action": "DriveMonopile", "duration": drive_time, **_shared},
    ]

    yield env.process(env.task_handler(task_list))


def install_transition_piece(env, vessel, tp, **kwargs):
    """
    Process logic for installing a transition piece on a monopile at site.

    Subprocesses:

    - Reequip crane, ``vessel.crane.reequip()``
    - Lower transition piece, ``tasks.lower_transition_piece()``
    - Install connection, see below.
    - Jackdown, ``vessel.jacksys.jacking_time()``

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

    reequip_time = vessel.crane.reequip(**kwargs)
    lower_tp_time = tasks.lower_transition_piece(vessel, **kwargs)
    site_depth = kwargs.get("site_depth", None)
    extension = kwargs.get("extension", site_depth + 10)
    jackdown_time = vessel.jacksys.jacking_time(extension, site_depth)

    _shared = {"agent": vessel.name, "location": "Site", "type": "Operations"}

    task_list = [
        {
            "action": "CraneReequip",
            "duration": reequip_time,
            **_shared,
            **vessel.operational_limits,
        },
        {
            "action": "LowerTransitionPiece",
            "duration": lower_tp_time,
            **_shared,
            **vessel.operational_limits,
        },
    ]

    connection = kwargs.get("tp_connection_type", "bolted")
    if connection is "bolted":
        tp_bolt_time = tasks.bolt_transition_piece(**kwargs)

        task_list.append(
            {
                "action": "BoltTransitionPiece",
                "duration": tp_bolt_time,
                **_shared,
                **vessel.operational_limits,
            }
        )

    elif connection is "grouted":

        grout_pump_time = tasks.pump_transition_piece_grout(**kwargs)
        grout_cure_time = tasks.cure_transition_piece_grout(**kwargs)

        task_list.append(
            {
                "action": "PumpGrout",
                "duration": grout_pump_time,
                **_shared,
                **vessel.transit_limits,
            }
        )

        task_list.append(
            {
                "action": "CureGrout",
                "duration": grout_cure_time,
                **_shared,
                **vessel.transit_limits,
            }
        )

    else:
        raise Exception(
            f"Transition piece connection type '{connection}'"
            "not recognized. Must be 'bolted' or 'grouted'."
        )

    task_list.append(
        {
            "action": "Jackdown",
            "duration": jackdown_time,
            **_shared,
            **vessel.transit_limits,
        }
    )

    yield env.process(env.task_handler(task_list))
