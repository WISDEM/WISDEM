__all__ = ['NREL_WISDEM']

from openmdao.main.api import Component
from openmdao.lib.datatypes.api import Float


# Make sure that your class has some kind of docstring. Otherwise
# the descriptions for your variables won't show up in the
# source ducumentation.
class NREL_WISDEM(Component):
    """
    """
    # declare inputs and outputs here, for example:
    #x = Float(0.0, iotype='in', desc='description for x')
    #y = Float(0.0, iotype='out', desc='description for y')

    def execute(self):
        """ do your calculations here """
        pass
        