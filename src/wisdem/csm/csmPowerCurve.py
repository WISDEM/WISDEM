""" 
csmPowerCurve.py
    
Created by George Scott 2012
Modified  by Katherine Dykes 2012
Copyright (c)  NREL. All rights reserved.

"""
    
from math import *
from csmDriveEfficiency import DrivetrainEfficiencyModel, csmDriveEfficiency
    
# -------------------------------------------------------------------------------- 

class csmPowerCurve:
    """
    This class is a simplified model used to determine power output and 
    peformance characteristics for a wind turbine rotor given a small set of input parameters.   
    """

    def __init__(self):
        """
           Initialize properties for power curve
        """
        
        # initialize output parameters
        self.ratedPower = 0.0
        self.ratedWindSpeed = 0.0
        self.powerCurve = []
        
        pass

    #--------------------
            
    def compute(self, drivetrain, hubHt=90.0, ratedPower=5000.0,maxTipSpd=80.0,rotorDiam=126.0,  \
                 maxCp=0.482, maxTipSpdRatio = 7.55, cutInWS = 4.0, cutOutWS = 25.0, \
                 altitude=0.0, airDensity = 0.0):
        """
        Computes the ideal power curve for a turbine after drivetrain losses using the NREL Cost and Scaling Model method.
        
        Parameters
        ----------
        drivetrain : drivetrainEfficiencyModel
           drivetrain efficiency object implementing interface in driveEfficiency.py
        hubHt : float
           height of hub above ground / sea-surface [m]
        ratedPower : float
           rated power for the turbine [kW]
        maxTipSpd : float
           maximum allowable tip speed for the rotor [m/s]
        rotorDiam : float
           diameter of wind turbine rotor [m]
        maxCp : float
           the optimal power coefficient for the rotor
        maxTipSpdRatio : float
           the optimal tip speed ratio for the rotor in region 2
        cutInWS : float
           cut in wind speed for the rotor
        cutOutWS : float
           cut out wind speed for the rotor
        altitude : float
           wind plant site altitude
        airDensity : float
           wind plant air density at hub height
        
        """

        # initialize input parameters
        self.drivetrain = drivetrain
        self.hubHt      = hubHt
        self.ratedPower = ratedPower
        self.maxTipSpd  = maxTipSpd
        self.rotorDiam  = rotorDiam
        self.maxCp      = maxCp
        self.maxTipSpdRatio = maxTipSpdRatio
        self.cutInWS    =  cutInWS
        self.cutOutWS   = cutOutWS
        self.altitude   = altitude

        if airDensity == 0.0:      
            # Compute air density 
            ssl_pa     = 101300  # std sea-level pressure in Pa
            gas_const  = 287.15  # gas constant for air in J/kg/K
            gravity    = 9.80665 # standard gravity in m/sec/sec
            lapse_rate = 0.0065  # temp lapse rate in K/m
            ssl_temp   = 288.15  # std sea-level temp in K
        
            self.airDensity = (ssl_pa * (1-((lapse_rate*(self.altitude + self.hubHt))/ssl_temp))**(gravity/(lapse_rate*gas_const))) / \
              (gas_const*(ssl_temp-lapse_rate*(self.altitude + self.hubHt)))
        else:
            self.airDensity = airDensity

        # determine power curve inputs
        self.reg2pt5slope  = 0.05
        
        self.maxEfficiency = self.drivetrain.getMaxEfficiency()
        self.ratedHubPower = self.ratedPower / self.maxEfficiency  # RatedHubPower

        self.omegaM = self.maxTipSpd/(self.rotorDiam/2.)  # Omega M - rated rotor speed
        omega0 = self.omegaM/(1+self.reg2pt5slope)       # Omega 0 - rotor speed at which region 2 hits zero torque
        Tm = self.ratedHubPower*1000/self.omegaM         # Tm - rated torque

        # compute rated rotor speed
        self.ratedRPM = (30/pi) * self.omegaM
        
        # compute variable-speed torque constant k
        kTorque = (self.airDensity*pi*self.rotorDiam**5*self.maxCp)/(64*self.maxTipSpdRatio**3) # k
        
        b = -Tm/(self.omegaM-omega0)                       # b - quadratic formula values to determine omegaT
        c = (Tm*omega0)/(self.omegaM-omega0)               # c
        
        # omegaT is rotor speed at which regions 2 and 2.5 intersect
        # add check for feasibility of omegaT calculation 09/20/2012
        omegaTflag = True
        if (b**2-4*kTorque*c) > 0:
           omegaT = -(b/(2*kTorque))-(sqrt(b**2-4*kTorque*c)/(2*kTorque))  # Omega T
           #print [kTorque, b, c, omegaT]
        
           windOmegaT = (omegaT*self.rotorDiam)/(2*self.maxTipSpdRatio) # Wind  at omegaT (M25)
           pwrOmegaT  = kTorque*omegaT**3/1000                                # Power at ometaT (M26)
        
           # compute rated wind speed
           d = self.airDensity*pi*self.rotorDiam**2.*0.25*self.maxCp
           self.ratedWindSpeed = \
              0.33*( (2.*self.ratedHubPower*1000.      / (    d))**(1./3.) ) + \
              0.67*( (((self.ratedHubPower-pwrOmegaT)*1000.) / (1.5*d*windOmegaT**2.))  + windOmegaT )
        else:
           omegaTflag = False
           windOmegaT = self.ratedRPM
           pwrOmegaT = self.ratedPower


        # set up for idealized power curve
        n = 161 # number of wind speed bins
        itp = [None] * n
        ws_inc = 0.25  # size of wind speed bins for integrating power curve
        Wind = []
        Wval = 0.0
        Wind.append(Wval)
        for i in xrange(1,n):
           Wval += ws_inc
           Wind.append(Wval)

        # determine idealized power curve 
        self.idealPowerCurve (Wind, itp, kTorque, windOmegaT, pwrOmegaT, n , omegaTflag)

        # add a fix for rated wind speed calculation inaccuracies kld 9/21/2012
        ratedWSflag = False
        # determine power curve after losses
        mtp = [None] * n
        for i in xrange(0,n):
           mtp[i] = itp[i] * self.drivetrain.getDrivetrainEfficiency(itp[i],self.ratedHubPower)
           #print [Wind[i],itp[i],self.drivetrain.getDrivetrainEfficiency(itp[i],self.ratedHubPower),mtp[i]] # for testing
           if (mtp[i] > self.ratedPower):
              if not ratedWSflag:
                ratedWSflag = True
                self.ratedWindSpeed = Wind[i]
              mtp[i] = self.ratedPower
        self.powerCurve = [Wind,mtp]

        pass

    # --------------------------
            
    def idealPowerCurve( self, Wind, ITP, kTorque, windOmegaT, pwrOmegaT, n , omegaTflag):
        """
        Determine the ITP (idealized turbine power) array
        """
       
        idealPwr = 0.0

        for i in xrange(0,n):
            if (Wind[i] > self.cutOutWS ) or (Wind[i] < self.cutInWS):
                idealPwr = 0.0  # cut out
            else:
                if omegaTflag:
                    if ( Wind[i] > windOmegaT ):
                       idealPwr = (self.ratedHubPower-pwrOmegaT)/(self.ratedWindSpeed-windOmegaT) * (Wind[i]-windOmegaT) + pwrOmegaT # region 2.5
                    else:
                       idealPwr = kTorque * (Wind[i]*self.maxTipSpdRatio/(self.rotorDiam/2.0))**3 / 1000.0 # region 2             
                else:
                    idealPwr = kTorque * (Wind[i]*self.maxTipSpdRatio/(self.rotorDiam/2.0))**3 / 1000.0 # region 2

            ITP[i] = idealPwr
            #print [Wind[i],ITP[i]]
        
        return

    # -----------------------

    def getRatedWindSpeed(self):
        """ 
        Provides the minimum wind speed required for a turbine to operate at rated power after drivetrain losses are deducted.

        Returns
        -------
        ratedWindSpeed : float
            Wind speed [m/s] at which turbine reaches rated power after drivetrain losses are deducted
        """
      
        return self.ratedWindSpeed

    # ------------------------
    
    def getRatedRotorSpeed(self):
        """ 
        Provides the rotor speed for a tubrine operating at rated power after drivetrain losses are deducted.

        Returns
        -------
        ratedRPM : float
            Rotor speed [rpm] at which turbine reaches rated power after drivetrain losses are deducted
        """
      
        return self.ratedRPM
   
    # ------------------------
    
    def getMaxEfficiency(self):
        """ 
        Provides the maximum drivetrain efficiency for a power curve.

        Returns
        -------
        maxEfficiency : float
            Maximum drivetrain efficiency [unitless] at rated power for the turbine
        """
      
        return self.maxEfficiency

    # ------------------------

    def getPowerCurve(self):
        """ 
        Gives an array of power output as a function of input wind speed for a wind turbine accounting for drivetrain losses.

        Returns
        -------
        powerCurve : ndarray(float)
            Power output for a wind turbine for an array of given input wind speeds.
        """

        return self.powerCurve


#------------------------------------------------------------------

def example():

    pc = csmPowerCurve()

    drivetrain = csmDriveEfficiency(1)
    pc.drivetrain = drivetrain

    # initialize input parameters
    pc.hubHt    = 90.0
    pc.ratedPower = 5000.0
    pc.maxTipSpd = 80
    pc.rotorDiam = 126
    pc.maxCp    = 0.488
    pc.maxTipSpdRatio = 7.525
    pc.cutInWS       =  3.0
    pc.cutOutWS      = 25.0
    pc.altitude = 0.0          
    pc.airDensity = 0.0
    pc.reg2pt5slope  = 0.05
      
    pc.compute(pc.drivetrain, pc.hubHt, pc.ratedPower,pc.maxTipSpd,pc.rotorDiam,  \
                 pc.maxCp, pc.maxTipSpdRatio, pc.cutInWS, pc.cutOutWS, \
                 pc.altitude, pc.airDensity)
    

    print 'Rated Speed:   %9.3f mps'          % pc.getRatedWindSpeed()
    print 'Rated RPM:   %9.3f rpm'          % pc.getRatedRotorSpeed()
    pcurve = pc.getPowerCurve()

    '''# plot
    wind = pcurve[0]
    power = pcurve[1] 
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(wind, power)
    plt.xlabel('Wind Speed [m/s]')
    plt.ylabel('Power Output [W]')
    plt.show()'''
    
def example2():

    pc = csmPowerCurve()

    drivetrain = csmDriveEfficiency(1)
    pc.drivetrain = drivetrain

    # initialize input parameters
    pc.hubHt    = 90.0
    pc.maxTipSpd = 80
    pc.rotorDiam = 126
    pc.maxCp    = 0.488
    pc.maxTipSpdRatio = 7.525
    pc.cutInWS       =  3.0
    pc.cutOutWS      = 25.0
    pc.altitude = 0.0          
    pc.airDensity = 0.0
    pc.reg2pt5slope  = 0.05   
    pc.ratedPower = 750.0

    pc.compute(pc.drivetrain, pc.hubHt, pc.ratedPower,pc.maxTipSpd,pc.rotorDiam,  \
                 pc.maxCp, pc.maxTipSpdRatio, pc.cutInWS, pc.cutOutWS, \
                 pc.altitude, pc.airDensity)
    
    print
    print 'RatedSpd   %9.3f mps'          % pc.getRatedWindSpeed()
    print 'RatedRPM   %9.3f    '          % pc.getRatedRotorSpeed()
    print 'Power curve values'
    pcurve = pc.getPowerCurve()
    print pcurve

    pc.ratedPower = 1000.0
    pc.compute(pc.drivetrain, pc.hubHt, pc.ratedPower,pc.maxTipSpd,pc.rotorDiam,  \
                 pc.maxCp, pc.maxTipSpdRatio, pc.cutInWS, pc.cutOutWS, \
                 pc.altitude, pc.airDensity)
    
    print
    print 'RatedSpd   %9.3f mps'          % pc.getRatedWindSpeed()
    print 'RatedRPM   %9.3f    '          % pc.getRatedRotorSpeed()
    print 'Power curve values'
    pcurve = pc.getPowerCurve()
    print pcurve

    pc.ratedPower = 2000.0
    pc.compute(pc.drivetrain, pc.hubHt, pc.ratedPower,pc.maxTipSpd,pc.rotorDiam,  \
                 pc.maxCp, pc.maxTipSpdRatio, pc.cutInWS, pc.cutOutWS, \
                 pc.altitude, pc.airDensity)
    
    print
    print 'RatedSpd   %9.3f mps'          % pc.getRatedWindSpeed()
    print 'RatedRPM   %9.3f    '          % pc.getRatedRotorSpeed()
    print 'Power curve values'
    pcurve = pc.getPowerCurve()
    print pcurve

if __name__ == "__main__":

    example()

    #example2()