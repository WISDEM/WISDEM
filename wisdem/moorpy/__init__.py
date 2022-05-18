# make all classes and functions in MoorPy.py available under the main package
# from wisdem.moorpy.MoorPy import *

import os

from wisdem.moorpy.body import Body

# new core MoorPy imports to eventually replace the above
from wisdem.moorpy.line import Line  # this will make mp.Line available from doing "import wisdem.moorpy as mp"
from wisdem.moorpy.point import Point
from wisdem.moorpy.system import System
from wisdem.moorpy.helpers import *
from wisdem.moorpy.Catenary import catenary
from wisdem.moorpy.lineType import LineType

moorpy_dir = os.path.dirname(os.path.realpath(__file__))
