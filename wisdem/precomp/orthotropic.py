
def skipLines(f, n):
    for i in range(n):
        f.readline()

class Orthotropic2DMaterial:
    """Represents a homogeneous orthotropic material in a
    plane stress state.

    """

    def __init__(self, E1, E2, G12, nu12, rho, name=""):
        """a struct-like object.  all inputs are also fields.
        The object also has an identification
        number *.mat_idx so unique materials can be identified.

        Parameters
        ----------
        E1 : float (N/m^2)
            Young's modulus in first principal direction
        E2 : float (N/m^2)
            Young's modulus in second principal direction
        G12 : float (N/m^2)
            shear modulus
        nu12 : float
            Poisson's ratio  (nu12*E22 = nu21*E11)
        rho : float (kg/m^3)
            density
        name : str
            an optional identifier

        """
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.rho = rho
        self.name = name

    @classmethod
    def listFromPreCompFile(cls, fname):
        """initialize the object by extracting materials from a PreComp materials file

        Parameters
        ----------
        fname : str
            name of input file

        Returns
        -------
        materials : List(:class:`Orthotropic2DMaterial`)
            a list of all materials gathered from the file.

        """

        f = open(fname)

        skipLines(f, 3)

        materials = []
        for line in f:
            array = line.split()
            mat = cls(float(array[1]), float(array[2]), float(array[3]), float(array[4]), float(array[5]), array[6])

            materials.append(mat)
        f.close()

        return materials
