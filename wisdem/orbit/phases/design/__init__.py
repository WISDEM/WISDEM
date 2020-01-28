"""The design package contains `DesignPhase` and its subclasses."""

__author__ = ["Jake Nunemaker", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = ["Jake Nunemaker", "Rob Hammond"]
__email__ = ["jake.nunemaker@nrel.gov" "robert.hammond@nrel.gov"]


from .design_phase import DesignPhase  # isort:skip
from .oss_design import OffshoreSubstationDesign
from .monopile_design import MonopileDesign
from .array_system_design import ArraySystemDesign, CustomArraySystemDesign
from .project_development import ProjectDevelopment
from .export_system_design import ExportSystemDesign
from .scour_protection_design import ScourProtectionDesign
