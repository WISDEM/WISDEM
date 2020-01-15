"""Provides common functionality between turbine installation strategies."""

__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit.vessels import tasks


def install_tower(env, vessel, tower, **kwargs):
    """
    Process logic for installing a tower at site.

    Subprocesses:

    - Reequip crane, ``vessel.crane.reequip()``
    - Lift tower,  ``tasks.lift_tower()``
    - Attach tower, ``tasks.attach_tower()``

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    tower : dict
    """

    reequip_time = vessel.crane.reequip(**kwargs)
    lift_time = tasks.lift_tower(vessel, **kwargs)
    attach_time = tasks.attach_tower(vessel, **kwargs)

    _shared = {
        "agent": vessel.name,
        "type": "Operations",
        "location": "Site",
        **vessel.operational_limits,
    }

    task_list = [
        {"action": "CraneReequip", "duration": reequip_time, **_shared},
        {"action": "LiftTower", "duration": lift_time, **_shared},
        {"action": "AttachTower", "duration": attach_time, **_shared},
    ]

    yield env.process(env.task_handler(task_list))


def install_nacelle(env, vessel, nacelle, **kwargs):
    """
    Process logic for installing a nacelle on a pre-installed tower.

    Subprocesses:

    - Reequip crane, ``vessel.crane.reequip()``
    - Lift nacelle, ``tasks.lift_nacelle()``
    - Attach nacelle, ``tasks.attach_nacelle()``

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    tower : dictå
    """

    reequip_time = vessel.crane.reequip(**kwargs)
    lift_time = tasks.lift_nacelle(vessel, **kwargs)
    attach_time = tasks.attach_nacelle(vessel, **kwargs)

    _shared = {
        "agent": vessel.name,
        "type": "Operations",
        "location": "Site",
        **vessel.operational_limits,
    }

    task_list = [
        {"action": "CraneReequip", "duration": reequip_time, **_shared},
        {"action": "LiftNacelle", "duration": lift_time, **_shared},
        {"action": "AttachNacelle", "duration": attach_time, **_shared},
    ]

    yield env.process(env.task_handler(task_list))


def install_turbine_blade(env, vessel, blade, **kwargs):
    """
    Process logic for installing a turbine blade on a pre-installed tower and
    nacelle assembly.

    Subprocesses:

    - Lift blade, ``tasks.lift_turbine_blade()``
    - Attach blade, ``tasks.attach_turbine_blade()``

    Parameters
    ----------
    env : Environment
    vessel : Vessel
    tower : dict
    """

    lift_time = tasks.lift_turbine_blade(vessel, **kwargs)
    attach_time = tasks.attach_turbine_blade(vessel, **kwargs)

    _shared = {
        "agent": vessel.name,
        "type": "Operations",
        "location": "Site",
        **vessel.operational_limits,
    }

    task_list = [
        {"action": "LiftBlade", "duration": lift_time, **_shared},
        {"action": "AttachBlade", "duration": attach_time, **_shared},
    ]

    yield env.process(env.task_handler(task_list))
