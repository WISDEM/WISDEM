"""

"""

from marmot import Object


class Cargo(Object):
    
    def __repr__(self):
        return self.type

    @property
    def type(self):
        """Returns type of `Cargo`."""
        return self.__class__.__name__
