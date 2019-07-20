
# http://stackoverflow.com/a/505457 with GB modifications
class Enum(object):
    def __init__(self, names, separator=None):
        if isinstance(names, type('')):
            names = names.upper().split(separator)
            self.data  = tuple(enumerate(names))
        elif isinstance(names, list):
            for i in names:
                assert len(i) == 2, 'Incorrect input data'
                assert isinstance(i[0], int), 'First index must be an integer'
                assert isinstance(i[1], type('')) or isinstance(i[1], type(u'')), 'Second index must be a string'
            self.data = names
        for i in self.data:
            setattr(self, i[1], i[0]) # name, value
        #for value, name in enumerate(names):
        #    setattr(self, name, value)
    def tuples(self):
        return self.data
    def __getitem__(self, i):
        if isinstance(i, type('')) or isinstance(i, type(u'')):
            return getattr(self, i)
        else:
            return self.data[i][1]
    def __len__(self):
        return len(self.data)



# http://stackoverflow.com/a/1695250
"""
#def enum(*sequential, **named):
#    enums = dict(zip(sequential, range(len(sequential))), **named)
#    return type('Enum', (), enums)
class Enum:
    def __init__(self, names):
        self.names = names.split()
        make(self.names)

    def make(*sequential, **named):
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)
"""


# Taken from http://norvig.com/python-iaq.html
"""
class Enum:

    # Create an enumerated type, then add var/value pairs to it.
    # The constructor and the method .ints(names) take a list of variable names,
    # and assign them consecutive integers as values.    The method .strs(names)
    # assigns each variable name to itself (that is variable 'v' has value 'v').
    # The method .vals(a=99, b=200) allows you to assign any value to variables.
    # A 'list of variable names' can also be a string, which will be .split().
    # The method .end() returns one more than the maximum int value.
    # Example: opcodes = Enum("add sub load store").vals(illegal=255).
  
    def __init__(self, names=[]): self.ints(names)

    def set(self, var, val):
        # Set var to the value val in the enum.
        if var in vars(self).keys(): raise AttributeError("duplicate var in enum")
        if val in vars(self).values(): raise ValueError("duplicate value in enum")
        vars(self)[var] = val
        return self
  
    def strs(self, names):
        # Set each of the names to itself (as a string) in the enum.
        for var in self._parse(names): self.set(var, var)
        return self

    def ints(self, names):
        # Set each of the names to the next highest int in the enum.
        for var in self._parse(names): self.set(var, self.end())
        return self

    def vals(self, **entries):
        # Set each of var=val pairs in the enum.
        for (var, val) in entries.items(): self.set(var, val)
        return self

    def end(self):
        # One more than the largest int value in the enum, or 0 if none.
        try: return max([x for x in vars(self).values() if type(x)==type(0)]) + 1
        except ValueError: return 0

    def _parse(self, names):
        # If names is a string, parse it as a list of names.
        if type(names) == type(""): return names.split()
        else: return names
"""
