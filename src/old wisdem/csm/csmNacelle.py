"""
csmNacelle.py

Created by George Scott on 2012-08-01.
Modified by Katherine Dykes 2012.
Copyright (c) NREL. All rights reserved.
"""

import sys

from csmPPI import *
from config import *

#-------------------------------------------------------------------------------
        

class LowSpdShaft(object):
    ''' LowSpdShaft class 
          lss = LowSpdShaft()
          lss.compute(RotorDiam, RotorMass, RotorTorque[,debug=1][,verbose=1]) : computes lss.mass, lss.cost
    '''
    def  __init__(self):
        """
        Initialize the parameters for low speed shaft
        
        Parameters
        ----------
        mass : float
          Low speed shaft mass [kg]
        cost : float
          Low speed shaft cost [USD]
        """
        self.mass = 0.0
        self.cost = 0.0
        
    def compute(self,RotorDiam, RotorMass, RotorTorque, curr_yr, curr_mon):
        """
        Compute mass and cost for a low speed shaft by calling computeMass and computeCost
        
        Parameters
        ----------
        RotorDiam : float
          rotor diameter [m] of the turbine
        RotorMass : float
          Mass of rotor [kg] of the turbine
        RotorTorque : float
          Rated torque [N-m] of the turbine
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """        
        self.computeMass(RotorDiam, RotorMass, RotorTorque)
        
        self.computeCost(RotorDiam, curr_yr, curr_mon)
      
    def computeMass(self, RotorDiam, RotorMass, RotorTorque):
        """
        Compute mass for a low speed shaft using the NREL Cost and Scaling Model
        
        Parameters
        ----------
        RotorDiam : float
          rotor diameter [m] of the turbine
        RotorMass : float
          Mass of rotor [kg] of the turbine
        RotorTorque : float
          Rated torque [N-m] of the turbine
        """ 

        self.lenShaft  = 0.03 * RotorDiam                                                                   
        self.mmtArm    = self.lenShaft / 5                                                                 
        self.bendLoad  = 1.25*9.81*RotorMass                                                           
        self.bendMom   = self.bendLoad * self.mmtArm                                                                 
        self.hFact     = 0.1                                                                                    
        self.hollow    = 1/(1-(self.hFact)**4)                                                                   
        self.outDiam   = ((32/pi)*self.hollow*3.25*((RotorTorque*3/371000000.)**2+(self.bendMom/71070000)**2)**(0.5))**(1./3.) 
        self.inDiam    = self.outDiam * self.hFact 
                                                                              
        self.mass      = 1.25*(pi/4)*(self.outDiam**2-self.inDiam**2)*self.lenShaft*7860
                                       
    def computeCost(self, RotorDiam,curr_yr, curr_mon):
        """
        Compute cost for a low speed shaft using the NREL Cost and Scaling Model
        
        Parameters
        ----------
        RotorDiam : float
          rotor diameter [m] of the turbine
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """ 
 
        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon
               
        LowSpeedShaftCost2002 = 0.0998 * RotorDiam ** 2.8873
        lssCostEsc     = ppi.compute('IPPI_LSS')
        
        self.cost = LowSpeedShaftCost2002 * lssCostEsc
        
    def getMass(self):
        """ 
        Provides the mass for the low speed shaft.

        Returns
        -------
        mass : float
            Wind turbine low speed shaft mass [kg]
        """

        return self.mass
        
    def getCost(self):
        """ 
        Provides the cost for the wind turbine low speed shaft.

        Returns
        -------
        cost : float
            Wind turbine low speed shaft cost [USD]
        """

        return self.cost
        

#-------------------------------------------------------------------------------

class GearBox(object):
    ''' GearBox class 
          gbx = GearBox()
          gbx.compute(iDsgn,MachineRating,MaxTipSpd,RotorDiam,RatedHubPower[,verbose=1]) : computes gbx.mass, gbx.cost
    '''
    
    costCoeff = [None, 16.45  , 74.101     ,   15.25697015,  0 ]
    costExp   = [None,  1.2491,  1.002     ,    1.2491    ,  0 ]
    massCoeff = [None, 65.601 , 81.63967335,  129.1702924 ,  0 ]
    massExp   = [None,  0.759 ,  0.7738    ,    0.7738    ,  0 ]
    
    def  __init__(self):
        """
        Initialize the parameters for gearbox
        
        Parameters
        ----------
        mass : float
          Gearbox mass [kg]
        cost : float
          Gearbox cost [USD]
        """
        self.mass = 0.0
        self.cost = 0.0
        
    def compute(self,iDsgn,MachineRating,RotorTorque, curr_yr, curr_mon):
        """
        Compute mass and cost for a gearbox by calling computeMass and computeCost
        
        Parameters
        ----------
        iDsgn : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        MachineRating : float
          Rated power [kW] for wind turbine
        RotorTorque : float
          Rated torque [N-m] of the turbine
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """        
        self.computeMass(iDsgn, RotorTorque)
        
        self.computeCost(iDsgn, MachineRating, curr_yr, curr_mon)
        
    def computeMass(self,iDsgn, RotorTorque):
        """
        Compute mass for a gearbox using the cost and scaling model

        Parameters
        ----------
        iDsgn : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        RotorTorque : float
          Rated torque [N-m] of the turbine
        """
        self.mass = self.__class__.massCoeff[iDsgn] * (RotorTorque/1000) ** self.__class__.massExp[iDsgn] 
        
    def computeCost(self,iDsgn, MachineRating, curr_yr, curr_mon):
        """
        Compute cost for a gearbox using the NREL Cost and Scaling Model
        
        Parameters
        ----------
        iDsgn : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        MachineRating : float
          Rated power [kW] for wind turbine
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """
        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon
        
        gearboxCostEsc     = ppi.compute('IPPI_GRB')        
        Gearbox2002 = self.__class__.costCoeff[iDsgn] * MachineRating ** self.__class__.costExp[iDsgn]  
        self.cost = Gearbox2002 * gearboxCostEsc      
                        
    def getMass(self):
        """ 
        Provides the mass for the gearbox.

        Returns
        -------
        mass : float
            Wind turbine gearbox mass [kg]
        """
      
        return self.mass
        
    def getCost(self):
        """ 
        Provides the cost for the wind turbine gearbox.

        Returns
        -------
        cost : float
            Wind turbine gearbox cost [USD]
        """
      
        return self.cost
        

#-------------------------------------------------------------------------------

class Generator(object):
    ''' Generator class 
          gen = Generator()
          gen.compute(iDsgn,MachineRating,MaxTipSpd,RotorDiam,RatedHubPower[,verbose=1]) : computes gen.mass, gen.cost
    '''
    
    costCoeff = [None, 65    , 54.73 ,  48.03 , 219.33 ] # $/kW - from 'Generators' worksheet
    massCoeff = [None, 6.4737, 10.51 ,  5.34  , 37.68  ]
    massExp   = [None, 0.9223, 0.9223,  0.9223, 1      ]
    
    def  __init__(self):
        """
        Initialize the parameters for the generator
        
        Parameters
        ----------
        mass : float
          Generator mass [kg]
        cost : float
          Generator cost [USD]
        """

        self.mass = 0.0
        self.cost = 0.0
        
    def compute(self,iDsgn,MachineRating,RotorTorque,curr_yr, curr_mon):
        """
        Compute mass and cost for a generator by calling computeMass and computeCost
        
        Parameters
        ----------
        iDsgn : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        MachineRating : float
          Rated power [kW] for wind turbine
        RotorTorque : float
          Rated torque [N-m] of the turbine
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """      
        self.computeMass(iDsgn, MachineRating, RotorTorque)
        
        self.computeCost(iDsgn, MachineRating, curr_yr, curr_mon)

    def computeMass(self,iDsgn, MachineRating, RotorTorque):
        """
        Compute mass for a generator using the NREL Cost and Scaling Model
        
        Parameters
        ----------
        iDsgn : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        MachineRating : float
          Rated power [kW] for wind turbine
        RotorTorque : float
          Rated torque [N-m] of the turbine
        """  
    
        if (iDsgn < 4):
            self.mass = self.__class__.massCoeff[iDsgn] * MachineRating ** self.__class__.massExp[iDsgn]   
        else:  # direct drive
            self.mass = self.__class__.massCoeff[iDsgn] * RotorTorque ** self.__class__.massExp[iDsgn] 

    def computeCost(self,iDsgn, MachineRating, curr_yr, curr_mon):
        """
        Compute mass and cost for a generator using the NREL Cost and Scaling Model
        
        Parameters
        ----------
        iDsgn : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        MachineRating : float
          Rated power [kW] for wind turbine
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """

        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon
      
        generatorCostEsc     = ppi.compute('IPPI_GEN')                                                  
        GeneratorCost2002 = self.__class__.costCoeff[iDsgn] * MachineRating 
        self.cost = GeneratorCost2002 * generatorCostEsc
        
    def getMass(self):
        """ 
        Provides the mass for the generator.

        Returns
        -------
        mass : float
            Wind turbine generator mass [kg]
        """

        return self.mass
        
    def getCost(self):
        """ 
        Provides the cost for the wind turbine generator.

        Returns
        -------
        cost : float
            Wind turbine generator cost [USD]
        """     

        return self.cost
        
                
#-------------------------------------------------------------------------------

class csmNacelle(object):
    ''' Nacelle class 
          (was called DriveTrain)
          nac = Nacelle()
          nac.compute(rtrDiam,mRating,rtrMass,maxTSR,rtdWS,rtpPwr,iDsgn[,verbose]) : computes nac.mass, nac.cost
        contains LowSpdShaft, GearBox, Generator
        2012 04 03 - controls moved into Nacelle
    '''

    def __init__(self):
        """
        Initialize the parameters for the nacelle
        
        Parameters
        ----------
        mass : float
          Generator mass [kg]
        cost : float
          Generator cost [USD]
        lss : LowSpdShaft
          Low Speed Shaft Object
        gear : GearBox
          Gearbox Object
        gen : Generator
          Generator Object
        """

        self.mass = 0.0
        self.cost = 0.0
        self.lss  = LowSpdShaft()
        self.gear = GearBox()
        self.gen  = Generator()

        
        # mfmCoeff[1,4] get modified by compute()
        self.mfmCoeff = [None,22448,1.29490,1.72080,22448 ]
        self.mfmExp   = [None,    0,1.9525, 1.9525 ,    0 ]
        
        pass

    def compute(self, RotorDiam,    MachineRating, RotorMass, RotorSpeed, \
                      MaximumThrust, RotorTorque, iDesign,   offshore, \
                      crane=True, AdvancedBedplate=0, curr_yr=2009,curr_mon=12,verbose=0):                                  
        """
        Compute mass and cost for a full wind turbine Nacelle by calling computeMass and computeCost
        
        Parameters
        ----------
        RotorDiam : float
          rotor diameter [m] of the turbine
        MachineRating : float
          Rated power [kW] for wind turbine
        RotorMass : float
          Mass of rotor [kg] of the turbine
        RotorSpeed : float
          Speed of rotor [rpm] at rated power
        MaximumThrust : float
          Maximum thrust [N] from rotor applied to nacelle
        RotorTorque : float
          Rated torque [N-m] of the turbine
        iDesign : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        offshore : int
          Offshore status: 0 - onshore, 1 - shallow, 2 - transition, 3 - deep
        crane : bool
          Boolean for onboard crane
        AdvancedBedplate : int
          Bedplate configuration: 0 - conventional, 1 - modular, 2 - integrated
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """

        self.computeMass(RotorDiam,    MachineRating, RotorMass, RotorSpeed, \
                      MaximumThrust, RotorTorque, iDesign,   offshore, \
                      crane, AdvancedBedplate)
       
        self.computeCost(RotorDiam,    MachineRating, RotorMass, RotorSpeed, \
                      MaximumThrust, RotorTorque, iDesign,   offshore, self.mainframeMass, \
                      crane, curr_yr, curr_mon)
        
        if (verbose > 0):
            self.dump()        

    def computeMass(self, RotorDiam,    MachineRating, RotorMass, RotorSpeed, \
                      MaximumThrust, RotorTorque, iDesign,   offshore, \
                      crane=True, AdvancedBedplate=0):                                  
        """
        Compute mass for a full wind turbine Nacelle using the NREL Cost and Scaling Model
        
        Parameters
        ----------
        RotorDiam : float
          rotor diameter [m] of the turbine
        MachineRating : float
          Rated power [kW] for wind turbine
        RotorMass : float
          Mass of rotor [kg] of the turbine
        RotorSpeed : float
          Speed of rotor [rpm] at rated power
        MaximumThrust : float
          Maximum thrust [N] from rotor applied to nacelle
        RotorTorque : float
          Rated torque [N-m] of the turbine
        iDesign : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        offshore : int
          Offshore status: 0 - onshore, 1 - shallow, 2 - transition, 3 - deep
        crane : bool
          Boolean for onboard crane
        AdvancedBedplate : int
          Bedplate configuration: 0 - conventional, 1 - modular, 2 - integrated
        """ 

        # initialize input variables
        self.crane = crane

        # These RD functions from spreadsheet don't quite form a continuous composite function        
        if (RotorDiam <= 15.0):
            TowerTopDiam = 0.3
        elif (RotorDiam <= 60.0):
            TowerTopDiam = (0.07042*RotorDiam-0.715)
        else:
            TowerTopDiam = (12.29*RotorDiam+2648)/1000
        
        # --- electrical connections           
        self.econnectionsMass = 0
        
        # --- bearings           
        self.bearingMass = 0.00012266667 * (RotorDiam ** 3.5) - 0.00030360 * (RotorDiam ** 2.5)
        HousingMass  = self.bearingMass 
        self.bhMass  = self.bearingMass + HousingMass
        
        # --- mechanical brake           
        mechBrakeCost2002 = 1.9894 * MachineRating + (-0.1141)
        self.mechBrakeMass = mechBrakeCost2002 * 0.10
        
        # --- variable-speed electronics
        self.vspdEtronicsMass = 0.0

        # --- yaw drive bearings
        self.yawDrvMass = 1.6 * (0.0009 * RotorDiam ** 3.314)
        
        # --- hydraulics, cooling
        self.hydrCoolingMass = 0.08 * MachineRating

        # --- bedplate ---        
        if (AdvancedBedplate == 0):   # not an actual option in cost and scaling model                                           
            BedplateWeightFac = 2.86  # modular
        elif (AdvancedBedplate == 1): # test for mod-adv
            BedplateWeightFac = 2.40  # modular-advanced
        else:
            BedplateWeightFac = 0.71  # advanced

        MassFromTorque = BedplateWeightFac * 0.00368 * RotorTorque
        MassFromThrust      = 0.00158 * BedplateWeightFac * MaximumThrust * TowerTopDiam
        MassFromRotorWeight = 0.015   * BedplateWeightFac * RotorMass     * TowerTopDiam
        
        # Bedplate(Length|Area) added by GNS
        BedplateLength = 1.5874 * 0.052 * RotorDiam
        BedplateArea = 0.5 * BedplateLength * BedplateLength
        MassFromArea = 100 * BedplateWeightFac * BedplateArea
    
        # --- control system ---
        
        initControlCost = [ 35000, 55900 ]  # land, off-shore
        self.controlMass = 0.0
        self.controlCost = initControlCost[offshore] * ppi.compute('IPPI_CTL')

        # --- nacelle totals        
        TotalMass = MassFromTorque + MassFromThrust + MassFromRotorWeight + MassFromArea
        self.mfmCoeff[1] = self.mfmCoeff[4] = TotalMass
        self.mfmCoeff[4] *= 0.55
        
        self.mainframeMass = self.mfmCoeff[iDesign] * RotorDiam ** self.mfmExp[iDesign] 

        NacellePlatformsMass = .125 * self.mainframeMass
        
        # --- nacelle cover ---        
        nacelleCovCost2002 = 11.537 * MachineRating + (3849.7)
        self.nacelleCovMass = nacelleCovCost2002 * 0.111111
        
        # --- crane ---        
        if (self.crane):
            self.craneMass =  3000.
        else:
            self.craneMass = 0.
            
        # --- main frame ---       
        self.mfTotalMass = self.mainframeMass + NacellePlatformsMass + self.craneMass
    
        # compute mass for subcomponents       
        self.lss.computeMass(RotorDiam,RotorMass,RotorTorque)
        self.gear.computeMass(iDesign,RotorTorque)
        self.gen.computeMass(iDesign,MachineRating, RotorTorque)

        # overall mass   
        self.mass = self.lss.mass + \
                    self.bhMass + \
                    self.gear.mass + \
                    self.mechBrakeMass + \
                    self.gen.mass + \
                    self.vspdEtronicsMass + \
                    self.yawDrvMass + \
                    self.mfTotalMass + \
                    self.econnectionsMass + \
                    self.hydrCoolingMass + \
                    self.nacelleCovMass + \
                    self.controlMass


    def computeCost(self, RotorDiam,    MachineRating, RotorMass, RotorSpeed, \
                      MaximumThrust, RotorTorque, iDesign,   offshore, mainframeMass, \
                      crane=True, curr_yr=2009,curr_mon=12):
        """
        Compute cost for a full wind turbine Nacelle using the NREL Cost and Scaling Model
        
        Parameters
        ----------
        RotorDiam : float
          rotor diameter [m] of the turbine
        MachineRating : float
          Rated power [kW] for wind turbine
        RotorMass : float
          Mass of rotor [kg] of the turbine
        RotorSpeed : float
          Speed of rotor [rpm] at rated power
        MaximumThrust : float
          Maximum thrust [N] from rotor applied to nacelle
        RotorTorque : float
          Rated torque [N-m] of the turbine
        iDesign : int
          Drivetrain Design (1 - 3-staged geared, 2 - single-stage, 3 - multi-gen, 4 - direct drive)
        offshore : int
          Offshore status: 0 - onshore, 1 - shallow, 2 - transition, 3 - deep
        mainframeMass : float
          Mass of mainframe [kg]
        crane : bool
          Boolean for onboard crane
        curr_yr : int
          year of project start
        curr_mon : int
          month of project start
        """
        
        # initialize input variables
        self.crane = crane
        self.mainframeMass = mainframeMass
        
        # Cost Escalators - obtained from ppi tables
        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon
        
        bearingCostEsc       = ppi.compute('IPPI_BRN')
        mechBrakeCostEsc     = ppi.compute('IPPI_BRK')
        VspdEtronicsCostEsc  = ppi.compute('IPPI_VSE')
        yawDrvBearingCostEsc = ppi.compute('IPPI_YAW')
        nacelleCovCostEsc    = ppi.compute('IPPI_NAC')
        hydrCoolingCostEsc   = ppi.compute('IPPI_HYD')
        mainFrameCostEsc     = ppi.compute('IPPI_MFM')
        econnectionsCostEsc  = ppi.compute('IPPI_ELC')

        # These RD functions from spreadsheet don't quite form a continuous composite function
        
        if (RotorDiam <= 15.0):
            TowerTopDiam = 0.3
        elif (RotorDiam <= 60.0):
            TowerTopDiam = (0.07042*RotorDiam-0.715)
        else:
            TowerTopDiam = (12.29*RotorDiam+2648)/1000
        
        # --- electrical connections
        self.econnectionsCost = 40.0 * MachineRating  # 2002
        self.econnectionsCost *= econnectionsCostEsc
        
        # --- bearings
        self.bearingMass = 0.00012266667 * (RotorDiam ** 3.5) - 0.00030360 * (RotorDiam ** 2.5)
        HousingMass  = self.bearingMass 
        brngSysCostFactor = 17.6 # $/kg
        Bearings2002 = self.bearingMass * brngSysCostFactor
        Housing2002  = HousingMass      * brngSysCostFactor
        self.bearingsCost = ( Bearings2002 + Housing2002 ) * bearingCostEsc
        
        # --- mechanical brake           
        mechBrakeCost2002 = 1.9894 * MachineRating + (-0.1141)
        self.mechBrakeCost = mechBrakeCostEsc * mechBrakeCost2002
        
        # --- variable-speed electronics           
        VspdEtronics2002 = 79.32 * MachineRating
        self.vspdEtronicsCost = VspdEtronics2002 * VspdEtronicsCostEsc

        # --- yaw drive bearings
        YawDrvBearing2002 = 2 * ( 0.0339 * RotorDiam ** 2.9637 )
        self.yawDrvBearingCost = YawDrvBearing2002 * yawDrvBearingCostEsc
        
        # --- hydraulics, cooling
        self.hydrCoolingCost = 12.0 * MachineRating # 2002
        self.hydrCoolingCost *= hydrCoolingCostEsc 
 
        # --- control system ---   
        initControlCost = [ 35000, 55900 ]  # land, off-shore
        self.controlCost = initControlCost[offshore] * ppi.compute('IPPI_CTL')

        # --- nacelle totals
        NacellePlatformsMass = .125 * self.mainframeMass
        NacellePlatforms2002 = 8.7 * NacellePlatformsMass
        
        # --- nacelle cover ---        
        nacelleCovCost2002 = 11.537 * MachineRating + (3849.7)
        self.nacelleCovCost = nacelleCovCostEsc * nacelleCovCost2002
        
        # --- crane ---
        
        if (self.crane):
            self.craneCost = 12000.
        else:
            self.craneCost = 0.0
            
        # --- main frame ---
        MainFrameCost2002 = 9.48850 * RotorDiam ** 1.9525   # todo : should depend on mainframe design
        BaseHardware2002  = MainFrameCost2002 * 0.7
        MainFrame2002 = ( MainFrameCost2002    + 
                          NacellePlatforms2002 + 
                          self.craneCost       + # service crane 
                          BaseHardware2002 )
        self.mainFrameCost = MainFrame2002 * mainFrameCostEsc
        
        # compute cost for subcomponents
        self.lss.computeCost(RotorDiam,curr_yr,curr_mon)
        self.gear.computeCost(iDesign,MachineRating,curr_yr,curr_mon)
        self.gen.computeCost(iDesign,MachineRating, curr_yr,curr_mon)

        # overall system cost
        self.cost = self.lss.cost + \
                    self.bearingsCost + \
                    self.gear.cost + \
                    self.mechBrakeCost + \
                    self.gen.cost + \
                    self.vspdEtronicsCost + \
                    self.yawDrvBearingCost + \
                    self.mainFrameCost + \
                    self.econnectionsCost + \
                    self.hydrCoolingCost + \
                    self.nacelleCovCost + \
                    self.controlCost

    def getMass(self):
        """ 
        Provides the mass for the overall wind turbine nacelle assembly.

        Returns
        -------
        mass : float
            Wind turbine overall nacelle assembly mass [kg]
        """

        return self.mass
        
    def getCost(self):
        """ 
        Provides the cost for the overall wind turbine nacelle assembly.

        Returns
        -------
        cost : float
            Wind turbine overall nacelle assembly cost [USD]
        """

        return self.cost

    def getNacelleComponentMasses(self):
        """ 
        Provides the mass for the overall wind turbine nacelle components.

        Returns
        -------
        nacelleComponentMasses : float
            Wind turbine nacelle masses [kg]: LSS, Bearings, Gearbox, Brakes, Generator, VS Electronics, Electrical Connections
            Controls, Yaw Drive System, Mainframe, and Nacelle Cover
        """
        
        self.nacelleComponentMasses = [self.lss.mass, self.bhMass, self.gear.mass, self.mechBrakeMass, self.gen.mass, \
                self.vspdEtronicsMass, self.econnectionsMass, self.hydrCoolingMass, \
                self.controlMass, self.yawDrvMass, self.mfTotalMass, self.nacelleCovMass]
        
        return self.nacelleComponentMasses
 
    def getNacelleComponentCosts(self):
        """ 
        Provides the cost for the overall wind turbine nacelle components.

        Returns
        -------
        nacelleComponentCosts : float
            Wind turbine nacelle costs [USD]: LSS, Bearings, Gearbox, Brakes, Generator, VS Electronics, Electrical Connections
            Controls, Yaw Drive System, Mainframe, and Nacelle Cover
        """
        
        self.nacelleComponentCosts = [self.lss.cost , self.bearingsCost , self.gear.cost, self.mechBrakeCost, self.gen.cost, \
                self.vspdEtronicsCost, self.econnectionsCost, self.hydrCoolingCost, \
                self.controlCost, self.yawDrvBearingCost, self.mainFrameCost, self.nacelleCovCost]

        return self.nacelleComponentCosts
                
    def dump(self):
        print "Nacelle Components"
        print '  lss           %6.4f K$  %8.4f kg' % (self.lss.cost         , self.lss.mass        )
        print '  bearings      %6.4f K$  %8.4f kg' % (self.bearingsCost     , self.bhMass          )
        print '  gear          %6.4f K$  %8.4f kg' % (self.gear.cost        , self.gear.mass       )
        print '  mechBrake     %6.4f K$  %8.4f kg' % (self.mechBrakeCost    , self.mechBrakeMass   )
        print '  gen           %6.4f K$  %8.4f kg' % (self.gen.cost         , self.gen.mass        )
        print '  vspdEtronics  %6.4f K$  %8.4f kg' % (self.vspdEtronicsCost , self.vspdEtronicsMass)
        print '  yawDrvBearing %6.4f K$  %8.4f kg' % (self.yawDrvBearingCost, self.yawDrvMass      )
        print '  mainFrame     %6.4f K$  %8.4f kg' % (self.mainFrameCost    , self.mfTotalMass     )
        print '  econnections  %6.4f K$  %8.4f kg' % (self.econnectionsCost , self.econnectionsMass)
        print '  hydrCooling   %6.4f K$  %8.4f kg' % (self.hydrCoolingCost  , self.hydrCoolingMass )
        print '  nacelleCov    %6.4f K$  %8.4f kg' % (self.nacelleCovCost   , self.nacelleCovMass  )
        print '  controls      %6.4f K$  %8.4f kg' % (self.controlCost      , self.controlMass  )
        print 'NACELLE TOTAL   %6.4f K$  %8.4f kg' % (self.cost             , self.mass            )
        print ' '       
        
#-------------------------------------------------------------------------------
        
def example():
  
    # simple test of module

    
    ref_yr  = 2002
    ref_mon =    9
    curr_yr = 2009
    curr_mon =  12
    
    ppi.ref_yr   = ref_yr
    ppi.ref_mon  = ref_mon
    ppi.curr_yr  = curr_yr
    ppi.curr_mon = curr_mon
    
    nac = csmNacelle()
    
    RotorDiam = 126.0
    MachineRating = 5000.0
    RotorMass = 123193.3010
    RotorSpeed = 12.1260909
    MaximumThrust = 500930.0837
    RotorTorque = 4365248.7375
    iDesign = 1
    offshore = 0
    crane=True
    AdvancedBedplate=0
    verbose= 1
    nac.compute(RotorDiam, MachineRating, RotorMass, RotorSpeed, \
                      MaximumThrust, RotorTorque, iDesign, offshore, \
                      crane, AdvancedBedplate, curr_yr, curr_mon, verbose)         

if __name__ == "__main__":  #TODO - update based on changes to csm Turbine

    example() 