# DriveEfficiency.py
# 2012 07 31

""" 
csmDriveEfficiency.py
    
Created by George Scott 2012
Modified  by Katherine Dykes 2012
Copyright (c)  NREL. All rights reserved.

"""
    
from math import *
import numpy as np

# from zope.interface import implements, Attribute, Interface
import abc

class DrivetrainEfficiencyModel(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def getMaxEfficiency(self):
        """Return maximum efficiency of the generator           

        Returns
        -------
        eta : float
          maximum efficiency of drivetrain (unitless)
        

        """ 
        
    @abc.abstractmethod
    def getDrivetrainEfficiency(self, outputPower, ratedHubPower):
        """ Returns efficiency of drivetrain operating at a ratio of outputPower to ratedHubPower
        
        Parameters
        ----------
        outputPower : float
          rotor output power to the drivetrain (W, kW - units should match rated hub power units)
        ratedHubPower : float
          rotor rated hub power to the drivetrain (W, kW - units should match output power units)

        Returns
        -------
        eta : float
          efficiency of drivetrain at a given operating point (unitless)
       
        """
  

class csmDriveEfficiency(DrivetrainEfficiencyModel):
    """
    This class represents a simple drivetrain efficiency model based on the NREL Cost and Scaling Model.
    It uses a simple quadratic formula for drivetrain efficiency based on empirical data.

    """

    def __init__(self, drivetrainDesign):
        """ 
        Set efficiency calculation coefficients based on drivetrain configuration type.
        
        Parameters
        ----------
        drivetrainDesign : int
           type of drivetrain design: 1 = 3-stage gearbox, 2 = single-stage gearbox, 3 = multi-gen, 4 = direct-drive (no gearbox)
                
        """

        #constant, linear and quadratic losses based on empirical CSM models
        self.constantloss = [0.0, 0.0128942744499, 0.0133073345628, 0.0154737171062, 0.0100718545194]
        self.linearloss = [0.0, 0.0850952825355, 0.036470617951, 0.0446307743432, 0.0199954344267]
        self.quadraticloss = [0.0, 0.0000, 0.0610666545019, 0.0578976395058, 0.0689903255878]
        self.drivetrainDesign = drivetrainDesign
        
    # ---------------------------

    def getDrivetrainEfficiency(self, outputPower, ratedHubPower):
        """
        Uses quadratic formula to calculate drivetrain efficiency with coefficients based on drivetrain configuraiton.
        
        Notes
        -----
        See Interfaces
        """

        # check for negative value 
        pwrFraction = outputPower / ratedHubPower
        # set for maximum if above rated
        pwrFraction = np.minimum(pwrFraction, 1.0)

        # check for positive power fraction else set eta = 1.0
        if pwrFraction > 0.0:
            dt = self.drivetrainDesign
            eta = np.zeros_like(pwrFraction)

            eta = 1.0 - (self.constantloss[dt]/pwrFraction + self.linearloss[dt] \
                + self.quadraticloss[dt]*pwrFraction)
        else:
            eta = 1.0
 
        return eta
        
    # ---------------------------

    def getMaxEfficiency(self):
        """
        Uses quadratic formula to calculate maximum drivetrain efficiency with coefficients based on drivetrain configuration.

        Notes
        -----
        See Interfaces
        """
        
        return 1.0 - (self.constantloss[self.drivetrainDesign] + self.linearloss[self.drivetrainDesign] + self.quadraticloss[self.drivetrainDesign])

#------------------------------------------------------------------    

def example():
    # simple test of module
    
    dEfficiency = csmDriveEfficiency(1)
    print 'Max Efficiency: %9.3f' % dEfficiency.getMaxEfficiency()
    
    # test for 89.4% efficiency 
    ratedHubPower = 5543.0
    outputPower = 3422.0   
    print 'Output power: %9.3f  Rated power: %9.3f   Efficiency: %9.3f' \
        % (outputPower,ratedHubPower,dEfficiency.getDrivetrainEfficiency(outputPower, ratedHubPower))

if __name__ == "__main__":
  
  example()