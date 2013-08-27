"""
nacelle.py

Created by Ryan King 2013.
Copyright (c) NREL. All rights reserved.
"""

from math import *
from common import SubComponent
import numpy as np
import scipy as scp
import scipy.optimize as opt
from zope.interface import implements

# -------------------------------------------------

class LowSpeedShaft():
    implements(SubComponent)
    ''' LowSpeedShaft class
          The LowSpeedShaft class is used to represent the low speed shaft component of a wind turbine drivetrain. 
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, aerodynamicTorque, aerodynamicBendingMoment, rotorMass, ratedSpeed, angleLSS, lengthLSS, D1, D2, nameplate=0, 
              ratioLSS=0.):
        '''
        Initializes low speed shaft component

        Parameters
        ----------
        aerodynamicTorque : float
          The torque load due to aerodynamic forces on the rotor [N*m]
        aerodynamicBendingMoment : float
          The bending moment from uneven aerodynamic loads [N*m]
        rotorMass : float
          The rotor mass [kg]
        ratedSpeed : float
          The speed of the rotor at rated power [rpm]
        angleLSS : float
          Angle of the LSS inclindation with respect to the horizontal [deg]
        lengthLSS : float
          Length of the LSS [m]
        D1 : float
          Fraction of LSS distance from gearbox to downwind main bearing
        D2 : float
          Fraction of LSS distance from gearbox to upwind main bearing
        nameplate : float
          Nameplate power rating for the turbine [W]
        ratioLSS : float
          Ratio of inner diameter to outer diameter.  Leave zero for solid LSS. 
        '''

        #torque check
        if aerodynamicTorque == 0:
          omega=ratedSpeed/60*(2*pi)      #rotational speed in rad/s at rated power
          eta=0.944                 #drivetrain efficiency
          self.aerodynamicTorque=nameplate/(omega*eta)         #torque  
       

        self.update_mass(RotorDiam, RotorMass, RotorTorque)

    def update_mass(self, aerodynamicTorque, aerodynamicBendingMoment, rotorMass, ratedSpeed, angleLSS, lengthLSS, D1, D2, nameplate=0, 
              ratioLSS=0.):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine low speed shaft component.

        The compute method determines and sets the attributes for the low speed shaft component based on the inputs described in the parameter section below.
        The mass is calculated based on a yield strength analysis about the upwind main bearing.

        Parameters
        ----------
        aerodynamicTorque : float
          The torque load due to aerodynamic forces on the rotor [N*m]
        aerodynamicBendingMoment : float
          The bending moment from uneven aerodynamic loads [N*m]
        rotorMass : float
          The rotor mass [kg]
        ratedSpeed : float
          The speed of the rotor at rated power [rpm]
        angleLSS : float
          Angle of the LSS inclindation with respect to the horizontal [deg]
        lengthLSS : float
          Length of the LSS [m]
        D1 : float
          Fraction of LSS distance from gearbox to downwind main bearing
        D2 : float
          Fraction of LSS distance from hub interface to upwind main bearing
        nameplate : float
          Nameplate power rating for the turbine [W]
        ratioLSS : float
          Ratio of inner diameter to outer diameter.  Leave zero for solid LSS. 
        '''

        # compute masses, dimensions and cost
        #static overhanging rotor moment (need to adjust for CM of rotor not just distance to end of LSS)
        L2=lengthLSS*D2                   #main bearing to end of mainshaft
        alpha=angleLSS*pi/180.0           #shaft angle
        L2=L2*cos(alpha)                  #horizontal distance from main bearing to hub center of mass
        staticRotorMoment=rotorMass*L2*9.81      #static bending moment from rotor
      
        #assuming 38CrMo4 / AISI 4140 from http://www.efunda.com/materials/alloys/alloy_steels/show_alloy.cfm?id=aisi_4140&prop=all&page_title=aisi%204140
        yieldStrength=417.0*10.0**6 #Pa
        steelDensity=8.0*10.0**3
        
        #Safety Factors
        gammaAero=1.35
        gammaGravity=1.35 #some talk of changing this to 1.1
        gammaFavorable=0.9
        gammaMaterial=1.25 #most conservative
        
        maxFactoredStress=yieldStrength/gammaMaterial
        factoredAerodynamicTorque=aerodynamicTorque*gammaAero
        factoredTotalRotorMoment=aerodynamicBendingMoment*gammaAero-staticRotorMoment*gammaFavorable
        
        # Second moment of area for hollow shaft
        def Imoment(d_o,d_i):
            I=(pi/64.0)*(d_o**4-d_i**4)
            return I
        
        # Second polar moment for hollow shaft
        def Jmoment(d_o,d_i):
            J=(pi/32.0)*(d_o**4-d_i**4)
            return J
        
        # Bending stress
        def bendingStress(M, y, I):
            sigma=M*y/I
            return sigma
        
        # Shear stress
        def shearStress(T, r, J):
            tau=T*r/J
            return tau
        
        #Find the necessary outer diameter given a diameter ratio and max stress
        def outerDiameterStrength(ratioLSS,maxFactoredStress):
            D_outer=(16.0/(pi*(1.0-ratioLSS**4.0)*maxFactoredStress)*(factoredTotalRotorMoment+sqrt(factoredTotalRotorMoment**2.0+factoredAerodynamicTorque**2.0)))**(1.0/3.0)
            return D_outer
        
        D_outer=outerDiameterStrength(ratioLSS,maxFactoredStress)
        D_inner=ratioLSS*D_outer
        
        print "LSS outer diameter is %f m, inner diameter is %f m" %(D_outer, D_inner)
        
        J=Jmoment(D_outer,D_inner)
        I=Imoment(D_outer,D_inner)
        
        sigmaX=bendingStress(factoredTotalRotorMoment, D_outer/2.0, I)
        tau=shearStress(aerodynamicTorque, D_outer/2.0, J)
        
        print "Max unfactored normal bending stress is %g MPa" % (sigmaX/1.0e6)
        print "Max unfactored shear stress is %g MPa" % (tau/1.0e6)
        
        volumeLSS=((D_outer/2.0)**2.0-(D_inner/2.0)**2.0)*pi*lengthLSS
        weightLSS=volumeLSS*steelDensity
        
        return np.array([weightLSS,D_outer])

        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0] = - (0.035 - 0.01) * RotorDiam            # cm based on WindPACT work - halfway between locations of two main bearings
        cm[1] = 0.0
        cm[2] = 0.025 * RotorDiam
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        I[0]  = self.mass * (inDiam ** 2 + self.diameter ** 2) / 8
        I[1]  = self.mass * (inDiam ** 2 + self.diameter ** 2 + (4 / 3) * (self.length ** 2)) / 16
        I[2]  = I[1]
        self.I = I

#-------------------------------------------------------------------------------

class Bearing():
	  implements(SubComponent)
	  '''
	     Bearing class for a generic bearing
	  '''

class MainBearings():
    implements(SubComponent)
    ''' MainBearings class
          The MainBearings class is used to represent the main bearing components of a wind turbine drivetrain. It contains two subcomponents (main bearing and second bearing) which also inherit from the SubComponent class.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self,D1, D2, lengthLSS,rotorMass, aerodynamicBendingMoment, D_outer):
        ''' Initializes main bearings component

        Parameters
        ----------
        D1 : float
          Fraction of LSS length from gearbox to downwind main bearing.
        D2 : float
          Fraction of LSS length from gearbox to upwind main bearing.
        lengthLSS : float
          Length of the LSS [m].
        rotorMass : float
          Mass of the rotor [kg].
        aerodynamicBendingMoment : float
          Aerodynamic bending moment [N*m].
        D_outer : float
          Outer diameter of the LSS [m].
        '''

        self.upwindBearing = Bearing()
        self.downwindBearing = Bearing()

        self.update_mass(D1, D2, lengthLSS,rotorMass, aerodynamicBendingMoment, D_outer)

    def update_mass(self,D1, D2, lengthLSS,rotorMass, aerodynamicBendingMoment, D_outer):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine main bearing components.

        The compute method determines and sets the attributes for the main bearing components based on the inputs described in the parameter section below.
        The mass is calculated based on commonly used main bearing catalogue information.

        Parameters
        ----------
        D1 : float
          Fraction of LSS length from gearbox to downwind main bearing.
        D2 : float
          Fraction of LSS length from gearbox to upwind main bearing.
        lengthLSS : float
          Length of the LSS [m].
        rotorMass : float
          Mass of the rotor [kg].
        aerodynamicBendingMoment : float
          Aerodynamic bending moment [N*m].
        D_outer : float
          Outer diameter of the LSS [m].
        '''

        #compute reaction forces
        #Safety Factors
        gammaAero=1.35
        gammaGravity=1.35 #some talk of changing this to 1.1
        gammaFavorable=0.9
        gammaMaterial=1.25 #most conservative
        
        #Bearing 1 is closest to gearbox, Bearing 2 is closest to rotor
        L2=lengthLSS*D2
        L1=lengthLSS*D1

        Fstatic=rotorMass*9.81*gammaFavorable #N
        Mrotor=aerodynamicBendingMoment*gammaAero #Nm

        R2=(-Mrotor+Fstatic*D1)/(D2-D1)
        print "R2: %g" %(R2)

        R1=-Fstatic-R2
        print "R1: %g" %(R1)
        
        

        # compute masses, dimensions and cost

        self.upwindBearing.mass = 485.0
        self.secondBearing.mass = 460.0

        self.mass = (self.mainBearing.mass + self.secondBearing.mass)

        # # calculate mass properties
        # inDiam  = lss.diameter
        # self.depth = (inDiam * 1.5)

        # cmMB = np.array([0.0,0.0,0.0])
        # cmMB = ([- (0.035 * RotorDiam), 0.0, 0.025 * RotorDiam])
        # self.mainBearing.cm = cmMB

        # cmSB = np.array([0.0,0.0,0.0])
        # cmSB = ([- (0.01 * RotorDiam), 0.0, 0.025 * RotorDiam])
        # self.secondBearing.cm = cmSB

        # cm = np.array([0.0,0.0,0.0])
        # for i in (range(0,3)):
        #     # calculate center of mass
        #     cm[i] = (self.mainBearing.mass * self.mainBearing.cm[i] + self.secondBearing.mass * self.secondBearing.cm[i]) \
        #               / (self.mainBearing.mass + self.secondBearing.mass)
        # self.cm = cm

        # self.b1I0 = (b1mass * inDiam ** 2 ) / 4 + (h1mass * self.depth ** 2) / 4
        # self.mainBearing.I = ([self.b1I0, self.b1I0 / 2, self.b1I0 / 2])

        # self.b2I0  = (b2mass * inDiam ** 2 ) / 4 + (h2mass * self.depth ** 2) / 4
        # self.secondBearing.I = ([self.b2I0, self.b2I0 / 2, self.b2I0 / 2])

        # I = np.array([0.0, 0.0, 0.0])
        # for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
        #     # calculate moments around CM
        #     # sum moments around each components CM
        #     I[i]  =  self.mainBearing.I[i] + self.secondBearing.I[i]
        #     # translate to nacelle CM using parallel axis theorem
        #     for j in (range(0,3)):
        #         if i != j:
        #             I[i] +=  (self.mainBearing.mass * (self.mainBearing.cm[i] - self.cm[i]) ** 2) + \
        #                           (self.secondBearing.mass * (self.secondBearing.cm[i] - self.cm[i]) ** 2)
        # self.I = I

#-------------------------------------------------------------------------------

class Gearbox():
    implements(SubComponent)
    ''' Gearbox class
          The Gearbox class is used to represent the gearbox component of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, name, power, ratio, gearConfig, Np, rpm, eff, ratioType='optimal', shType='normal',torque=None):
        '''
        Initializes gearbox component

        Parameters
        ----------
        name : str
          Name of the gearbox.
        power : float
          Rated power of the gearbox [W].
        ratio : float
          Overall gearbox speedup ratio.
        gearConfig : str
          String describing configuration of each gear stage.  Use 'e' for epicyclic and 'p' for parallel, for example 'eep' would be epicyclic-epicyclic-parallel.
        Np : array
          Array describing the number of planets in each stage.  For example if gearConfig is 'eep' Np could be [3 3 1].
        rpm : float
          Rotational speed of the LSS at rated power [rpm].
        eff : float
          Mechanical efficiency of the gearbox.
        ratioType : str
          Describes how individual stage ratios will be calculated.  Can be 'empirical' which uses the Sunderland model, or 'optimal' which finds the stage ratios that minimize overall mass.
        shType : str
          Describes the shaft type and applies a corresponding application factor.  Can be 'normal' or 'short'.
        torque : float
          Mechanical torque applied to gearbox at rated power.
        '''

        def rotorTorque():
            torque = self.power / self.eff / (self.rpm * (pi / 30.0))
            return torque
        
        if torque==None:
            self.torque=rotorTorque()
        else:
            self.torque=torque

        def weightEst():
            weight=gbxWeightEst(self.gearConfig,self.ratio,self.Np,self.ratioType,self.shType,self.torque)
            return weight
        self.weight=weightEst()

    def stageMassCalc(indStageRatio,indNp,indStageType):

        '''
        Computes the mass of an individual gearbox stage.

        Parameters
        ----------
        indStageRatio : str
          Speedup ratio of the individual stage in question.
        indNp : int
          Number of planets for the individual stage.
        indStageType : int
          Type of gear.  Use '1' for parallel and '2' for epicyclic.
        '''

        #Application factor to include ring/housing/carrier weight
        Kr=0.4

        if indNp == 3:
            Kgamma=1.1
        elif indNp == 4:
            Kgamma=1.25
        elif indNp == 5:
            Kgamma=1.35

        if indStageType == 1:
            indStageMass=1.0+indStageRatio+indStageRatio**2+(1.0/indStageRatio)

        elif indStageType == 2:
            sunRatio=0.5*indStageRatio - 1.0
            indStageMass=Kgamma*((1/indNp)+(1/(indNp*sunRatio))+sunRatio+sunRatio**2+Kr*((indStageRatio-1)**2)/indNp+Kr*((indStageRatio-1)**2)/(indNp*sunRatio))

        return indStageMass


    def stageRatioCalc(overallRatio,Np,ratioType,config):
        '''
        Calculates individual stage ratios using either empirical relationships from the Sunderland model or a SciPy constrained optimization routine.
        '''

        K_r=0

        #Assumed we can model everything w/Sunderland model to estimate speed ratio
        if ratioType == 'empirical':
            if config == 'p': 
                stageRatio=[overallRatio]
            if config == 'e':
                stageRatio=[overallRatio]
            elif config == 'pp':
                stageRatio=[overallRatio**0.5,overallRatio**0.5]
            elif config == 'ep':
                stageRatio=[overallRatio/2.5,2.5]
            elif config =='ee':
                stageRatio=[overallRatio**0.5,overallRatio**0.5]
            elif config == 'eep':
                stageRatio=[(overallRatio/3)**0.5,(overallRatio/3)**0.5,3]
            elif config == 'epp':
                stageRatio=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
            elif config == 'eee':
                stageRatio=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
            elif config == 'ppp':
                stageRatio=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
        
        elif ratioType == 'optimal':

            if config == 'eep':
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=0 #2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)                              
                
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2],consargs=[overallRatio],rhoend=1e-7)
        
            elif config == 'eep_3':
                #fixes last stage ratio at 3
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=0.8 #2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)                              
                
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]
                
                def constr3(x,overallRatio):
                    return x[2]-3.0
                
                def constr4(x,overallRatio):
                    return 3.0-x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2,constr3,constr4],consargs=[overallRatio],rhoend=1e-7)
            
            elif config == 'eep_2':
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                B_2=Np[1]
                K_r1=0
                K_r2=1.6 #2nd stage structure weight coefficient

                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1.0)+(x[0]/2.0-1)**2+K_r1*((x[0]-1.0)**2)/B_1 + K_r1*((x[0]-1.0)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*((1.0/B_2)+(1/(B_2*((x[1]/2.0)-1.0)))+(x[1]/2.0-1.0)+(x[1]/2.0-1.0)**2.0+K_r2*((x[1]-1.0)**2.0)/B_2 + K_r2*((x[1]-1.0)**2.0)/(B_2*(x[1]/2.0-1.0))) + (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)                              
                
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2],consargs=[overallRatio],rhoend=1e-7)
        
            else:
                x0=[overallRatio**(1.0/3.0),overallRatio**(1.0/3.0),overallRatio**(1.0/3.0)]
                B_1=Np[0]
                K_r=0.0
                def volume(x):
                    return (1.0/(x[0]))*((1.0/B_1)+(1.0/(B_1*((x[0]/2.0)-1.0)))+(x[0]/2.0-1)+(x[0]/2.0-1.0)**2+K_r*((x[0]-1.0)**2)/B_1 + K_r*((x[0]-1)**2)/(B_1*(x[0]/2.0-1.0))) + (1.0/(x[0]*x[1]))*(1.0+(1.0/x[1])+x[1] + x[1]**2)+ (1.0/(x[0]*x[1]*x[2]))*(1.0+(1.0/x[2])+x[2] + x[2]**2)
                                  
                def constr1(x,overallRatio):
                    return x[0]*x[1]*x[2]-overallRatio
        
                def constr2(x,overallRatio):
                    return overallRatio-x[0]*x[1]*x[2]

                x=opt.fmin_cobyla(volume, x0,[constr1,constr2],consargs=[overallRatio],rhoend=1e-7)
        else:
            stageRatio='fail'
        
        stageRatio=x
        return stageRatio

    def gbxWeightEst(config,overallRatio,Np,ratioType,shType,torque):

      '''
      Computes the gearbox weight based on a surface durability criteria.
      '''
        stageType=[]
        for character in config:
            if character == 'e':
                stageType.append(2)
            if character == 'p':
                stageType.append(1)
        stageRatio=np.array(stageRatioCalc(overallRatio,Np,ratioType,config))

        stageMass=np.zeros([len(stageRatio),1])
        stageTorque=np.zeros([len(stageRatio),1])

        ## Define Application Factors ##

        #Application factor for weight estimate
        Ka=0.6

        #K factor for pitting analysis
        if torque < 200000.0:
            Kfact = 850.0
        elif torque < 700000.0:
            Kfact = 950.0
        else:
            Kfact = 1100.0

        #Unit conversion from Nm to inlb and vice-versa
        Kunit=8.029

        # Shaft length factor
        if shType == 'normal':
            Kshaft = 1.0
        elif shType == 'short':
            Kshaft = 1.25

        #Individual stage torques
        torqueTemp=torque
        for s in range(len(stageTorque)):
            stageTorque[s]=torqueTemp/stageRatio[s]
            torqueTemp=stageTorque[s]
            stageMass[s]=Kunit*Ka/Kfact*stageTorque[s]*stageMassCalc(stageRatio[s],Np[s],stageType[s])

        gbxWeight=(sum(stageMass))*Kshaft
       
        return gbxWeight    
        

#-------------------------------------------------------------------------------

class HighSpeedShaft():
	  implements(SubComponent)
	  '''
	     Basic class for a high speed shaft.
	  '''

class MechanicalBrake():
	  implements(SubComponent)
	  '''
	     Basic class for mechanical brake.
	  '''

class HighSpeedSide():
    implements(SubComponent)
    '''
    HighSpeedShaft class
          The HighSpeedShaft class is used to represent the high speed shaft and mechanical brake components of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self,MachineRating,RotorTorque,GearRatio,RotorDiam,lssOutDiam):
        ''' Initializes high speed shaft and mechanical brake component

        Parameters
        ----------
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        lssOutDiam : float
          outer diameter of low speed shaft [m]
        '''

        self.HSS = HighSpeedShaft()
        self.MechBrake = MechanicalBrake()

        self.update_mass(MachineRating,RotorTorque,GearRatio,RotorDiam,lssOutDiam)

    def update_mass(self,MachineRating,RotorTorque,GearRatio,RotorDiam,lssOutDiam):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine high speed shaft and brake.

        The compute method determines and sets the attributes for the HSS and mehanical brakes based on the inputs described in the parameter section below.
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.


        Parameters
        ----------
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        lssOutDiam : float
          outer diameter of low speed shaft [m]
        '''

        # compute masses, dimensions and cost
        designTQ = RotorTorque / GearRatio               # design torque [Nm] based on rotor torque and Gearbox ratio
        massFact = 0.025                                 # mass matching factor default value
        self.HSS.mass = (massFact * designTQ)

        self.MechBrake.mass = (0.5 * self.HSS.mass)      # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines

        self.mass = (self.MechBrake.mass + self.HSS.mass)

        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0]   = 0.5 * (0.0125 * RotorDiam)
        cm[1]   = 0.0
        cm[2]   = 0.025 * RotorDiam
        self.cm = cm

        self.diameter = (1.5 * lssOutDiam)                     # based on WindPACT relationships for full HSS / mechanical brake assembly
        self.length = (0.025)

        I = np.array([0.0, 0.0, 0.0])
        I[0]    = 0.25 * self.length * 3.14159 * (self.diameter ** 2) * GearRatio * (self.diameter ** 2) / 8
        I[1]    = self.mass * ((3/4) * (self.diameter ** 2) + (self.length ** 2)) / 12
        I[2]    = I[1]
        self.I = I

#-------------------------------------------------------------------------------

class Generator():
    implements(SubComponent)
    '''Generator class
          The Generator class is used to represent the generator of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self,iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio):
        ''' Initializes generator component

        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains.
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        '''

        self.update_mass(iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio)

    def update_mass(self,iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine generator component.

        The compute method determines and sets the attributes for the generator component based on the inputs described in the parameter section below.
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
        Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.


        Parameters
        ----------
        iDsgn : int
          Integer which selects the type of gearbox based on drivetrain type: 1 = standard 3-stage gearbox, 2 = single-stage, 3 = multi-gen, 4 = direct drive
          Method is currently configured only for type 1 drivetrains.
        MachineRating : float
          The power rating for the overall wind turbine [kW]
        RotorTorque : float
          The input torque from the wind turbine at rated power after accounting for drivetrain losses [N*m]
        GearRatio : float
          Ratio of high speed to low speed shaft based on total gear ratio of gearbox
        RotorDiam : float
          The wind turbine rotor diameter [m]
        '''

        massCoeff = [None, 6.4737, 10.51 ,  5.34  , 37.68  ]
        massExp   = [None, 0.9223, 0.9223,  0.9223, 1      ]

        CalcRPM    = 80 / (RotorDiam*0.5*pi/30)
        CalcTorque = (MachineRating*1.1) / (CalcRPM * pi/30)

        if (iDsgn < 4):
            self.mass = (massCoeff[iDsgn] * MachineRating ** massExp[iDsgn])
        else:  # direct drive
            self.mass = (massCoeff[iDsgn] * CalcTorque ** massExp[iDsgn])

        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0]  = 0.0125 * RotorDiam
        cm[1]  = 0.0
        cm[2]  = 0.025 * RotorDiam
        self.cm = cm

        self.length = (1.6 * 0.015 * RotorDiam)
        self.depth = (0.015 * RotorDiam)
        self.width = (0.5 * self.depth)

        I = np.array([0.0, 0.0, 0.0])
        I[0]   = ((4.86 * (10 ** (-5))) * (RotorDiam ** 5.333)) + (((2/3) * self.mass) * (self.depth ** 2 + self.width ** 2) / 8)
        I[1]   = (I[0] / 2) / (GearRatio ** 2) + ((1/3) * self.mass * (self.length ** 2) / 12) + (((2 / 3) * self.mass) * \
                   (self.depth ** 2 + self.width ** 2 + (4/3) * (self.length ** 2)) / 16 )
        I[2]   = I[1]
        self.I = I

#-------------------------------------------------------------------------------

class Bedplate():
    implements(SubComponent)
    ''' Bedplate class
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self,towerTopDiam, lengthLSS, rotorDiameter, uptowerTransformer=0):
        ''' Initializes bedplate component

        Parameters
        ----------
        towerTopDiam : float
          Diameter of the top tower section at the nacelle flange [m].
        lengthLSS : float
          Length of the LSS [m].
        rotorDiameter : float
          The wind turbine rotor diameter [m].
        uptowerTransformer : int
          Determines if the transformer is uptower ('1') or downtower ('0').
        '''

        self.update_mass(towerTopDiam, lengthLSS, rotorDiameter, uptowerTransformer=0)

    def update_mass(self,towerTopDiam, lengthLSS, rotorDiameter, uptowerTransformer=0):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine bedplate component.

        The compute method determines and sets the attributes for the bedplate component based on the inputs described in the parameter section below.
        The mass is calculated based on a cast iron front section in the load path and a rear steel frame holding other components.


        Parameters
        ----------
        towerTopDiam : float
          Diameter of the top tower section at the nacelle flange [m].
        lengthLSS : float
          Length of the LSS [m].
        rotorDiameter : float
          The wind turbine rotor diameter [m].
        uptowerTransformer : int
          Determines if the transformer is uptower ('1') or downtower ('0').
        '''

        #Treat front cast iron main beam in similar manner to hub
        #Use 1/4 volume of annulus given by:
        #length is LSS length plus tower radius
        #diameter is 1.25x tower diameter
        #guess at initial thickness same as hub
        
        castThickness=rotorDiameter/620
        castVolume=(lengthLSS+towerTopDiam/2)*pi*(towerTopDiam*1.25)*0.25*castThickness
        castDensity=7.1*10.0**3 #kg/m^3
        castMass=castDensity*castVolume
        
        #These numbers based off V80, need to update
        if uptowerTransformer == 1:
            steelVolume=1.5*lengthLSS*1.2*0.66/4.0
        elif uptowerTransformer == 0:
            steelVolume=lengthLSS*1.2*0.66/4.0
        steelDensity=7900 #kg/m**3
        steelMass=steelDensity*steelVolume

        self.mass = steelMass + castMass

        # calculate mass properties
        cm = np.array([0.0,0.0,0.0])
        cm[0] = cm[1] = 0.0
        cm[2] = 0.0122 * RotorDiam                             # half distance from shaft to yaw axis
        self.cm = cm

        self.depth = (self.length / 2.0)

        I = np.array([0.0, 0.0, 0.0])
        I[0]  = self.mass * (self.width ** 2 + self.depth ** 2) / 8
        I[1]  = self.mass * (self.depth ** 2 + self.width ** 2 + (4/3) * self.length ** 2) / 16
        I[2]  = I[1]
        self.I = I

#-------------------------------------------------------------------------------

class YawSystem():
    implements(SubComponent)
    ''' YawSystem class
          The YawSystem class is used to represent the yaw system of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self, towerTopDiam, rotorDiam, numYawMotors=0):
        ''' Initializes yaw system

        Parameters
        ----------
        towerTopDiam : float
          Diameter of the tower top section [m]
        rotorDiam : float
          Rotor Diameter [m].
        numYawMotors : int
          Number of yaw motors.
        '''

        if self.numYawMotors == 0 :
          if rotorDiam < 90.0 :
            self.numYawMotors = 4.0
          elif rotorDiam < 120.0 :
            self.numYawMotors = 6.0
          else:
            self.numYawMotors = 8.0


        self.update_mass(towerTopDiam, numYawMotors, rotorDiam)

    def update_mass(self,towerTopDiam, rotorDiam, numYawMotors):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine yaw system.

        The compute method determines and sets the attributes for the yaw system component based on the inputs described in the parameter section below.
        The model assumes a friction plate design that scales with the rotor dimensions.


        Parameters
        ----------
        towerTopDiam : float
          Diameter of the tower top section [m]
        rotorDiam : float
          Rotor Diameter [m].
        numYawMotors : int
          Number of yaw motors.
        '''
        #assume friction plate surface width is 1/10 the diameter
        #assume friction plate thickness scales with rotor diameter
        frictionPlateVol=pi*towerTopDiam*(towerTopDiam*0.10)*(rotorDiam/1000.0)
        steelDensity=8000.0
        frictionPlateMass=frictionPlateVol*steelDensity
        
        #Assume same yaw motors as Vestas V80 for now: Bonfiglioli 709T2M
        yawMotorMass=190.0
        
        totalYawMass=frictionPlateMass + (numYawMotors*yawMotorMass)
        return totalYawMass

        # calculate mass properties
        # yaw system assumed to be collocated to tower top center
        cm = np.array([0.0,0.0,0.0])
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        self.I = I

#-------------------------------------------------------------------------------

class NacelleSystem(): # changed name to nacelle - need to rename, move code pieces, develop configurations ***
    implements(SubComponent)
    ''' NacelleSystem class
          The Nacelle class is used to represent the overall nacelle of a wind turbine.
          It contains the general properties for a wind turbine component as well as additional design load and dimentional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
    '''

    def __init__(self,RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane):
        '''
        Initializes nacelle system

        Parameters
        ----------
        lss  : LowSpeedShaft()
        mbg  : MainBearings()
        gear : Gearbox()
        hss  : HighSpeedShaft()
        gen  : Generator()
        bpl  : Bedplate()
        yaw  : YawSystem()
        crane   : bool
          flag for presence of service crane up-tower
        econnectionsCost : float
          cost for electrical cabling
        econnectionsMass : float
          mass uptower for electrical cabling [kg]
        vspdEtronicsCost : float
          cost for variable speed electronics
        vspdEtronicsMass : float
          mass for variable speed electronics [kg]
        hydrCoolingCost : float
          cost for hydraulics and HVAC system
        hydrCoolingMass : float
          mass for hydraulics and HVAC system [kg]
        ControlsCost : float
          cost for controls up
        ControlsMass : float
          mass uptower for controls [kg]
        nacellePlatformsCost : float
          cost for nacelle platforms
        nacellePlatformsMass : float
          mass for nacelle platforms [kg]
        craneCost : float
          cost for service crane uptower
        craneMass : float
          mass for service crane uptower [kg]
        mainframeCost : float
          cost for mainframe including bedplate, service crane and platforms
        mainframeMass : float
          mass for mainframe including bedplate, service crane and platforms [kg]
        nacelleCovCost : float
          cost for nacelle cover
        nacelleCovMass : float
          mass for nacelle cover [kg]
        '''


        self.update_mass(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    def update_mass(self,RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane):
        '''
        Computes the dimensions, mass, mass properties and costs for the wind turbine nacelle and all of its sub components.

        The compute method determines and sets the attributes for the nacelle and all of its sub components based on the inputs described in the parameter section below.
        The mass is calculated based input loads accoreding to the University of Sunderland model [1] and the dimensions are calculated based on the NREL WindPACT rotor studies.
        Component costs are based on mass vs. cost relationships derived from drivetrain component cost and mass data of the NREL cost and scaling model.


        Parameters
        ----------
        RotorMass : float
            mass of rotor [kg]
        RotorDiam : float
            diameter of rotor [m]
        RotorTorque : float
            input torque from rotor at rated power accounting for drivetrain losses [Nm]
        RotorThrust : float
            maximum input thrust from rotor for extreme conditions [N]
        RotorSpeed : float
            rotor speed at rated power [rpm]
        iDsgn : int
            Drivetrain design type (1 to 4)
        MachineRating : float
            Machine rating for turbine [kW]
        GearRatio : float
            gear ratio for full gearbox (high speed shaft to low speed shaft speed)
        GearConfig : str
            number of gears and types in a string; allowable configurations include 'p', 'e', 'pp', 'ep', 'ee', 'epp', 'eep', 'eee'
        Bevel : int
            flag for presence of a bevel stage (0 if not present, 1 if present)
        TowerTopDiam : float
            diameter of tower top [m]
        crane : bool
            boolean for crane present on-board
        '''
        self.crane = crane
        self.GearConfig = GearConfig


        # create instances of input variables and initialize values
        self.lss = LowSpeedShaft(RotorDiam,RotorMass,RotorTorque)
        self.mbg = MainBearings(self.lss, RotorSpeed, RotorDiam)
        self.gear = Gearbox(iDsgn,RotorTorque,GearRatio,GearConfig,Bevel,RotorDiam)
        self.stagemass = self.gear.getStageMass() #return gearbox stage masses
        self.hss = HighSpeedSide(MachineRating,RotorTorque,GearRatio,RotorDiam,self.lss.diameter)
        self.gen = Generator(iDsgn,MachineRating,RotorSpeed,RotorDiam,GearRatio)
        self.bpl = Bedplate(iDsgn,RotorTorque,RotorMass,RotorThrust,RotorDiam,TowerTopDiam)
        self.aboveYawMass = 0.0
        self.yaw = YawSystem(RotorDiam,RotorThrust,TowerTopDiam,self.aboveYawMass)



        # initialize other drivetrain components
        # self.econnectionsCost     = 0.0
        self.econnectionsMass     = 0.0
        # self.vspdEtronicsCost     = 0.0
        self.vspdEtronicsMass     = 0.0
        # self.hydrCoolingCost      = 0.0
        self.hydrCoolingMass      = 0.0
        # self.ControlsCost         = 0.0
        self.ControlsMass         = 0.0
        # self.nacellePlatformsCost = 0.0
        self.nacellePlatformsMass = 0.0
        # self.craneCost            = 0.0
        self.craneMass            = 0.0
        # self.mainframeCost        = 0.0
        self.mainframeMass        = 0.0
        # self.nacelleCovCost       = 0.0
        self.nacelleCovMass       = 0.0

        # electronic systems, hydraulics and controls

        self.hydrCoolingMass = 0.08 * MachineRating

        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        self.nacellePlatformsMass = 0.125 * self.bpl.mass

        if (self.crane):
            self.craneMass =  3000.0
        else:
            self.craneMass = 0.0

        self.mainframeMass  = self.bpl.mass + self.craneMass + self.nacellePlatformsMass

        nacelleCovArea      = 2 * (self.bpl.length ** 2)              # this calculation is based on Sunderland
        self.nacelleCovMass = (84.1 * nacelleCovArea) / 2          # this calculation is based on Sunderland - divided by 2 in order to approach CSM

        self.length      = self.bpl.length                              # nacelle length [m] based on bedplate length
        self.width       = self.bpl.width                        # nacelle width [m] based on bedplate width
        self.height      = (2.0 / 3.0) * self.length                         # nacelle height [m] calculated based on cladding area

        # yaw system weight calculations based on total system mass above yaw system
        self.aboveYawMass =  self.lss.mass + \
                    self.mbg.mass + \
                    self.gear.mass + \
                    self.hss.mass + \
                    self.gen.mass + \
                    self.mainframeMass + \
                    self.econnectionsMass + \
                    self.vspdEtronicsMass + \
                    self.hydrCoolingMass + \
                    self.nacelleCovMass
        self.yaw.update_mass(RotorDiam,RotorThrust,TowerTopDiam,self.aboveYawMass)   # yaw mass calculation based on Sunderalnd model

        # aggregation of nacelle mass
        self.mass = (self.aboveYawMass + self.yaw.mass)

        # calculation of mass center and moments of inertia
        cm = np.array([0.0,0.0,0.0])
        for i in (range(0,3)):
            # calculate center of mass
            cm[i] = (self.lss.mass * self.lss.cm[i] +
                    self.mbg.mainBearing.mass * self.mbg.mainBearing.cm[i] + self.mbg.secondBearing.mass * self.mbg.secondBearing.cm[i] + \
                    self.gear.mass * self.gear.cm[i] + self.hss.mass * self.hss.cm[i] + \
                    self.gen.mass * self.gen.cm[i] + self.bpl.mass * self.bpl.cm[i] ) / \
                    (self.lss.mass + self.mbg.mainBearing.mass + self.mbg.secondBearing.mass + \
                    self.gear.mass + self.hss.mass + self.gen.mass + self.bpl.mass)
        self.cm = cm

        I = np.array([0.0, 0.0, 0.0])
        for i in (range(0,3)):                        # calculating MOI, at nacelle center of gravity with origin at tower top center / yaw mass center, ignoring masses of non-drivetrain components / auxiliary systems
            # calculate moments around CM
            # sum moments around each components CM
            I[i]  =  self.lss.I[i] + self.mbg.mainBearing.I[i] + self.mbg.secondBearing.I[i] + self.gear.I[i] + \
                          self.hss.I[i] + self.gen.I[i] + self.bpl.I[i]
            # translate to nacelle CM using parallel axis theorem
            for j in (range(0,3)):
                if i != j:
                    I[i] +=  self.lss.mass * (self.lss.cm[i] - cm[i]) ** 2 + \
                                  self.mbg.mainBearing.mass * (self.mbg.mainBearing.cm[i] - cm[i]) ** 2 + \
                                  self.mbg.secondBearing.mass * (self.mbg.secondBearing.cm[i] - cm[i]) ** 2 + \
                                  self.gear.mass * (self.gear.cm[i] - cm[i]) ** 2 + \
                                  self.hss.mass * (self.hss.cm[i] - cm[i]) ** 2 + \
                                  self.gen.mass * (self.gen.cm[i] - cm[i]) ** 2 + \
                                  self.bpl.mass * (self.bpl.cm[i] - cm[i]) ** 2
        self.I = I

    def componentMasses(self):
        return self.lss.mass, self.mbg.mass, self.gear.mass, self.hss.mass, self.gen.mass, self.bpl.mass, self.yaw.mass

    def getNacelleComponentMasses(self):
        """ Returns detailed nacelle assembly masses

        detailedMasses : array_like of float
           detailed masses for nacelle components
        """

        detailedMasses = [self.lss.mass, self.mbg.mass, self.gear.mass, self.hss.mass, self.gen.mass, self.vspdEtronicsMass, \
                self.econnectionsMass, self.hydrCoolingMass, \
                self.ControlsMass, self.yaw.mass, self.mainframeMass, self.nacelleCovMass]

        return detailedMasses

#------------------------------------------------------------------

def example():

    # test of module for turbine data set

    # NREL 5 MW Rotor Variables
    RotorDiam = 126.0 # m
    RotorSpeed = 12.13 # m/s
    RotorTorque = 4365248.74 # Nm
    RotorThrust = 500930.84 # N
    RotorMass = 142585.75 # kg

    # NREL 5 MW Drivetrain variables
    iDsgn = 1 # geared 3-stage Gearbox with induction generator machine
    MachineRating = 5000.0 # kW
    GearRatio = 97.0 # 97:1 as listed in the 5 MW reference document
    GearConfig = 'eep' # epicyclic-epicyclic-parallel
    Bevel = 0 # no bevel stage
    crane = True # onboard crane present

    # NREL 5 MW Tower Variables
    TowerTopDiam = 3.78 # m

    print '----- NREL 5 MW Turbine -----'
    nace = NacelleSystem(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, \
                   MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    print 'Nacelle system model results'
    print 'Low speed shaft %8.1f kg  %6.2f m length %6.2f m OD %6.2f m Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.lss.mass , nace.lss.length,nace.lss.diameter, nace.lss.I[0], nace.lss.I[1], nace.lss.I[2], nace.lss.cm[0], nace.lss.cm[1], nace.lss.cm[2])
    # 31257.3 kg
    print 'Main bearings   %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' % (nace.mbg.mass , nace.mbg.I[0], nace.mbg.I[1], nace.mbg.I[2], nace.mbg.cm[0], nace.mbg.cm[1], nace.mbg.cm[2])
    # 9731.4 kg
    print 'Gearbox         %8.1f kg %6.2f m length %6.2f m diameter %6.2f m height %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gear.mass, nace.gear.length, nace.gear.diameter, nace.gear.height, nace.gear.I[0], nace.gear.I[1], nace.gear.I[2], nace.gear.cm[0], nace.gear.cm[1], nace.gear.cm[2] )
    # 30237.6 kg
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.stagemass[1], nace.stagemass[2], nace.stagemass[3])
    print 'High speed shaft & brakes  %8.1f kg %6.2f m length %6.2f m OD %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.hss.mass, nace.hss.length, nace.hss.diameter, nace.hss.I[0], nace.hss.I[1], nace.hss.I[2], nace.hss.cm[0], nace.hss.cm[1], nace.hss.cm[2])
    # 1492.4 kg
    print 'Generator       %8.1f kg %6.2f m length %6.2f m width %6.2f m depth %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gen.mass, nace.gen.length, nace.gen.width, nace.gen.depth, nace.gen.I[0], nace.gen.I[1], nace.gen.I[2], nace.gen.cm[0], nace.gen.cm[1], nace.gen.cm[2])
    # 16699.9 kg
    print 'Variable speed electronics %8.1f kg' % (nace.vspdEtronicsMass)
    # 0.0 kg
    print 'Overall mainframe %8.1f kg %6.2f m length %6.2f m width' % (nace.mainframeMass, nace.bpl.length, nace.bpl.width)
    # 96932.9 kg
    print '     Bedplate     %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
         % (nace.bpl.mass, nace.bpl.I[0], nace.bpl.I[1], nace.bpl.I[2], nace.bpl.cm[0], nace.bpl.cm[1], nace.bpl.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.econnectionsMass)
    # 0.0 kg
    print 'HVAC system     %8.1f kg' % (nace.hydrCoolingMass )
    # 400.0 kg
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.nacelleCovMass , nace.height, nace.width, nace.length)
    # 9097.4 kg
    print 'Yaw system      %8.1f kg' % (nace.yaw.mass )
    # 11878.2 kg
    print 'Overall nacelle:  %8.1f kg cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.mass, nace.cm[0], nace.cm[1], nace.cm[2], nace.I[0], nace.I[1], nace.I[2]  )
    # 207727.1


def example2():

    # WindPACT 1.5 MW Drivetrain variables
    iDsgn = 1 # geared 3-stage Gearbox with induction generator machine
    MachineRating = 1500 # machine rating [kW]
    GearRatio = 87.965
    GearConfig = 'epp'
    Bevel = 0
    crane = True

    # WindPACT 1.5 MW Rotor Variables
    airdensity = 1.225 # air density [kg / m^3]
    MaxTipSpeed = 80 # max tip speed [m/s]
    RotorDiam = 70 # rotor diameter [m]
    RotorSpeed = 21.830
    DrivetrainEfficiency = 0.95
    RotorTorque = (MachineRating * 1000 / DrivetrainEfficiency) / (RotorSpeed * (pi / 30))
        # rotor torque [Nm] calculated from max / rated rotor speed and machine rating
    RotorThrust = 324000
    RotorMass = 28560 # rotor mass [kg]

    # WindPACT 1.5 MW Tower Variables
    TowerTopDiam = 2.7 # tower top diameter [m]

    print '----- WindPACT 1.5 MW Turbine -----'
    nace = NacelleSystem(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, \
                   MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    print 'Nacelle system model results'
    print 'Low speed shaft %8.1f kg  %6.2f m length %6.2f m OD %6.2f m Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.lss.mass , nace.lss.length,nace.lss.diameter, nace.lss.I[0], nace.lss.I[1], nace.lss.I[2], nace.lss.cm[0], nace.lss.cm[1], nace.lss.cm[2])
    # 31257.3 kg
    print 'Main bearings   %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' % (nace.mbg.mass , nace.mbg.I[0], nace.mbg.I[1], nace.mbg.I[2], nace.mbg.cm[0], nace.mbg.cm[1], nace.mbg.cm[2])
    # 9731.4 kg
    print 'Gearbox         %8.1f kg %6.2f m length %6.2f m diameter %6.2f m height %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gear.mass, nace.gear.length, nace.gear.diameter, nace.gear.height, nace.gear.I[0], nace.gear.I[1], nace.gear.I[2], nace.gear.cm[0], nace.gear.cm[1], nace.gear.cm[2] )
    # 30237.6 kg
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.stagemass[1], nace.stagemass[2], nace.stagemass[3])
    print 'High speed shaft & brakes  %8.1f kg %6.2f m length %6.2f m OD %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.hss.mass, nace.hss.length, nace.hss.diameter, nace.hss.I[0], nace.hss.I[1], nace.hss.I[2], nace.hss.cm[0], nace.hss.cm[1], nace.hss.cm[2])
    # 1492.4 kg
    print 'Generator       %8.1f kg %6.2f m length %6.2f m width %6.2f m depth %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gen.mass, nace.gen.length, nace.gen.width, nace.gen.depth, nace.gen.I[0], nace.gen.I[1], nace.gen.I[2], nace.gen.cm[0], nace.gen.cm[1], nace.gen.cm[2])
    # 16699.9 kg
    print 'Variable speed electronics %8.1f kg' % (nace.vspdEtronicsMass)
    # 0.0 kg
    print 'Overall mainframe %8.1f kg %6.2f m length %6.2f m width' % (nace.mainframeMass, nace.bpl.length, nace.bpl.width)
    # 96932.9 kg
    print '     Bedplate     %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
         % (nace.bpl.mass, nace.bpl.I[0], nace.bpl.I[1], nace.bpl.I[2], nace.bpl.cm[0], nace.bpl.cm[1], nace.bpl.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.econnectionsMass)
    # 0.0 kg
    print 'HVAC system     %8.1f kg' % (nace.hydrCoolingMass )
    # 400.0 kg
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.nacelleCovMass , nace.height, nace.width, nace.length)
    # 9097.4 kg
    print 'Yaw system      %8.1f kg' % (nace.yaw.mass)
    # 11878.2 kg
    print 'Overall nacelle:  %8.1f kg cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.mass, nace.cm[0], nace.cm[1], nace.cm[2], nace.I[0], nace.I[1], nace.I[2]  )
    # 207727.1

    # GRC Drivetrain variables
    nace = Nacelle()
    iDesign = 1 # geared 3-stage Gearbox with induction generator machine
    MachineRating = 750 # machine rating [kW]
    GearRatio = 81.491
    GearConfig = 'epp'
    Bevel = 0

    # GRC Rotor Variables
    airdensity = 1.225 # air density [kg / m^3]
    MaxTipSpeed = 8 # max tip speed [m/s]
    RotorDiam = 48.2 # rotor diameter [m]
    #RotorSpeed = MaxTipSpeed / ((RotorDiam / 2) * (pi / 30)) # max / rated rotor speed [rpm] calculated from max tip speed and rotor diamter
    RotorSpeed = 22
    DrivetrainEfficiency = 0.944
    RotorTorque = (MachineRating * 1000 / DrivetrainEfficiency) / (RotorSpeed * (pi / 30)) # rotor torque [Nm] calculated from max / rated rotor speed and machine rating
    RotorThrust = 159000 # based on windpact 750 kW design (GRC information not available)
    RotorMass = 13200 # rotor mass [kg]

    # Tower Variables
    TowerTopDiam = 2 # tower top diameter [m] - not given

    print '----- GRC 750 kW Turbine -----'
    nace.update_mass(RotorSpeed, RotorTorque, RotorThrust, RotorMass, RotorDiam, iDsgn, \
                   MachineRating, GearRatio, GearConfig, Bevel, TowerTopDiam, crane)

    print 'Nacelle system model results'
    print 'Low speed shaft %8.1f kg  %6.2f m length %6.2f m OD %6.2f m Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.lss.mass , nace.lss.length,nace.lss.diameter, nace.lss.I[0], nace.lss.I[1], nace.lss.I[2], nace.lss.cm[0], nace.lss.cm[1], nace.lss.cm[2])
    # 31257.3 kg
    print 'Main bearings   %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' % (nace.mbg.mass , nace.mbg.I[0], nace.mbg.I[1], nace.mbg.I[2], nace.mbg.cm[0], nace.mbg.cm[1], nace.mbg.cm[2])
    # 9731.4 kg
    print 'Gearbox         %8.1f kg %6.2f m length %6.2f m diameter %6.2f m height %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gear.mass, nace.gear.length, nace.gear.diameter, nace.gear.height, nace.gear.I[0], nace.gear.I[1], nace.gear.I[2], nace.gear.cm[0], nace.gear.cm[1], nace.gear.cm[2] )
    # 30237.6 kg
    print '     gearbox stage masses: %8.1f kg  %8.1f kg %8.1f kg' % (nace.stagemass[1], nace.stagemass[2], nace.stagemass[3])
    print 'High speed shaft & brakes  %8.1f kg %6.2f m length %6.2f m OD %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.hss.mass, nace.hss.length, nace.hss.diameter, nace.hss.I[0], nace.hss.I[1], nace.hss.I[2], nace.hss.cm[0], nace.hss.cm[1], nace.hss.cm[2])
    # 1492.4 kg
    print 'Generator       %8.1f kg %6.2f m length %6.2f m width %6.2f m depth %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
          % (nace.gen.mass, nace.gen.length, nace.gen.width, nace.gen.depth, nace.gen.I[0], nace.gen.I[1], nace.gen.I[2], nace.gen.cm[0], nace.gen.cm[1], nace.gen.cm[2])
    # 16699.9 kg
    print 'Variable speed electronics %8.1f kg' % (nace.vspdEtronicsMass)
    # 0.0 kg
    print 'Overall mainframe %8.1f kg %6.2f m length %6.2f m width' % (nace.mainframeMass, nace.bpl.length, nace.bpl.width)
    # 96932.9 kg
    print '     Bedplate     %8.1f kg %6.2f Ixx %6.2f Iyy %6.2f Izz %6.2f CGx %6.2f CGy %6.2f CGz' \
         % (nace.bpl.mass, nace.bpl.I[0], nace.bpl.I[1], nace.bpl.I[2], nace.bpl.cm[0], nace.bpl.cm[1], nace.bpl.cm[2])
    print 'electrical connections  %8.1f kg' % (nace.econnectionsMass)
    # 0.0 kg
    print 'HVAC system     %8.1f kg' % (nace.hydrCoolingMass )
    # 400.0 kg
    print 'Nacelle cover:   %8.1f kg %6.2f m Height %6.2f m Width %6.2f m Length' % (nace.nacelleCovMass , nace.height, nace.width, nace.length)
    # 9097.4 kg
    print 'Yaw system      %8.1f kg' % (nace.yaw.mass)
    # 11878.2 kg
    print 'Overall nacelle:  %8.1f kg cm %6.2f %6.2f %6.2f I %6.2f %6.2f %6.2f' % (nace.mass, nace.cm[0], nace.cm[1], nace.cm[2], nace.I[0], nace.I[1], nace.I[2]  )
    # 207727.1


if __name__ == '__main__':
    ''' Main runs through tests of several drivetrain configurations with known component masses and dimensions '''

    # todo: adjust to use rotor model interface

    example()

    #example2()