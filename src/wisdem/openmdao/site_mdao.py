#!/usr/bin/env python
# encoding: utf-8
"""
wind_mdao.py

Created by Andrew Ning on 2013-05-16.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.datatypes.api import Enum

from components import SiteBase
from turbine.rotor import RotorAero


class SiteStandardComp(SiteBase):

    wind_turbine_class = Enum('I', ('I', 'II', 'III'), iotype='in', desc='wind turbine class')


    def execute(self):

        # setup wind speeds based on IEC standards
        if self.wind_turbine_class == 'I':
            Vref = 50.0
        elif self.wind_turbine_class == 'II':
            Vref = 42.5
        else:
            Vref = 37.5

        Vmean = 0.2*Vref
        self.PDF = RotorAero.RayleighPDF(Vmean)
        self.CDF = RotorAero.RayleighCDF(Vmean)
        self.shearExp = 0.2

        self.V_extreme = 1.4*Vref


