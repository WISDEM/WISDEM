"""Provides the ``Cargo`` base class."""

from marmot import Object


class Cargo(Object):
    """Base class for working with cargo."""

    def __repr__(self):
        """Overridden __repr__ method."""
        return self.type

    @property
    def type(self):
        """Returns type of `Cargo`."""
        return self.__class__.__name__
