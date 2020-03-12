"""
The phases package contains `DesignPhase`, `InstallPhase` and any subclasses.
"""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Jake Nunemaker", "Rob Hammond"]
__email__ = ["jake.nunemaker@nrel.gov" "robert.hammond@nrel.gov"]


from .base import BasePhase
from .design import DesignPhase
from .install import InstallPhase
