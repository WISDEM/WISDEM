"""
Common.py

Created by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

from zope.interface import Attribute, Interface

class ComponentCost(Interface):

    cost = Attribute(""" Cost of component (USD) """)