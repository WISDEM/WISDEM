__author__ = ["Jake Nunemaker", "Matt Shields", "Rob Hammond"]
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = ["jake.nunemaker@nrel.gov", "robert.hammond@nrel.gov"]
__status__ = "Development"


from .manager import ProjectManager  # isort:skip
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
