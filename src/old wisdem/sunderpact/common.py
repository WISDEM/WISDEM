"""
common.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from zope.interface import Attribute, Interface

class SubComponent(Interface):
    ''' 
        Interface for turbine components.  This interface provides a set of attributes for a turbine component.
    '''
 
    mass = Attribute("""  mass of the component [kg] """)
    cm = Attribute(""" center of mass of the component in [x,y,z] for an arbitrary coordinate system """)
    I = Attribute("""  moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass """)
    diameter = Attribute(""" outer diameter of a cylindrical or spherical component [m] """)
    length = Attribute(""" y direction == diameter for spheres or vertical cylinders [m] """)
    width = Attribute(""" x direction == diameter for spheres or veritical cylinders [m] """)
    height = Attribute("""  z direction == diameter for spheres or horizontal cylinders [m] """)
    depth = Attribute(""" critical depth on any dimension for component [m] """)
