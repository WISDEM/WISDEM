#!/usr/bin/env python
# encoding: utf-8
"""
CSMdt.py

Created by Andrew Ning on 2013-04-08.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from zope.interface import implements

from rotoraero import DrivetrainEfficiencyInterface


class NRELCSMDrivetrain:
    """Simple drivetrain efficiency from NREL cost and scaling model"""
    implements(DrivetrainEfficiencyInterface)


    def __init__(self, drivetrainType='geared'):
        """Constructor

        Parameters
        ----------
        drivetrainType : str
            must be one of the following types:
            'geared', 'single-stage', 'multi-drive', 'pm-direct-drive'

        """

        if drivetrainType == 'geared':
            self.constant = 0.01289
            self.linear = 0.08510
            self.quadratic = 0.0

        elif drivetrainType == 'single-stage':
            self.constant = 0.01331
            self.linear = 0.03655
            self.quadratic = 0.06107

        elif drivetrainType == 'multi-drive':
            self.constant = 0.01547
            self.linear = 0.04463
            self.quadratic = 0.05790

        elif drivetrainType == 'pm-direct-drive':
            self.constant = 0.01007
            self.linear = 0.02000
            self.quadratic = 0.06899

        else:
            raise ValueError('input drivetrainType does not exist')



    def efficiency(self, power, ratedpower):
        """see :meth:`DrivetrainEfficiencyInterface.efficiency` """

        Pbar = power / ratedpower

        # handle negative power case
        Pbar = np.abs(Pbar)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar = np.minimum(Pbar, 1.0)

        # compute efficiency
        eta = np.zeros_like(Pbar)
        idx = Pbar != 0

        eta[idx] = 1.0 - (self.constant/Pbar[idx] + self.linear + self.quadratic*Pbar[idx])

        return eta


