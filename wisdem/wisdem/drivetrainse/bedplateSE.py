# -*- coding: utf-8 -*-
"""
bedplateSE.py

Created on Wed Sep  4 13:49:44 2019

@author: gscott

A generalization of the 2015 bedplate model (2 parallel I-beams, cast in front, steel in rear)
"""

import sys, os
import numpy as np
#import scipy as scp
#import scipy.optimize as opt
from math import pi, cos, sqrt, sin, exp, log10, log

#from wisdem.drivetrainse.drivese_utils import get_rotor_mass, get_distance_hub2mb, get_My, get_Mz, resize_for_bearings, mainshaftFlangeCalc 
#from wisdem.commonse.utilities import assembleI, unassembleI 

#--------------------------------------------

class BPelement(object):
    ''' BPelement class
        The BPelement class represents a section (along the x-axis) of the complete bedplate. Each BPelement has its own
        dimensions, material properties, and loads.
        
        The x-origin is located at the center of the tower top. Negative values are towards rotor; positive towards generator.
        
        Need to include moments from other elements
    '''
    
    #--------------------
    
    def __init__(self, element_name, material_density, material_E, material_name, nBeams, 
                 height=0.0, length=0.0, width=0.0, debug=False):
        self.element_name     = element_name
        self.material_density = material_density
        self.material_E       = material_E
        self.material_name    = material_name
        self.nBeams           = nBeams
        self.height           = height
        self.length           = length
        self.width            = width
        self.debug            = debug
        
        self.mass             = 0.0
        
        self.cm = np.zeros(3)
        self.I  = np.zeros(3)
        
        self.loads = []
        self.loadlocs = []
        # self.moments = []

        # initial I-beam dimensions in m
        self.tf = 0.01905        # flange thickness
        self.tw = 0.0127         # web thickness
        self.h0 = 0.6096         # overall height
        self.b0 = self.h0 / 2.0  # overall width
        
    #--------------------
    
    def characterize(self, stressTol, deflTol):
        # iterate over increasing size
        
        self.bi  = (self.b0 - self.tw) / 2.0
        self.hi  = self.h0 - 2.0 * self.tf
        self.I_b = self.b0 * self.h0**3 / 12.0 - 2 * self.bi * self.hi**3 / 12.0
        self.A   = self.b0 * self.h0 - 2.0 * self.bi * self.hi # cross-sectional area of beam
        self.w   = self.A * self.material_density # weight/len in kg/m

        # total mass
        self.mass = self.w * self.length
        
        # COM
        total_mass = self.mass + np.sum(self.loads)
        xcom = self.mass * (self.xpos - self.xpos) # modify if using origin NOT at xpos
        for i in range(len(self.loads)):
            xcom += self.loads[i] * self.loadlocs[i]
        self.cm[0] = xcom / total_mass
        
    #--------------------
    
    @staticmethod
    def dumphdr():
        return '   Name    matrl  xpos  xmin  xmax    L     W     H    Mass   COM\n'
    
    def dump(self):
        return '{:10s} {:5s} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:6.2f}'.format(self.element_name, 
                self.material_name, self.xpos, self.xpos - 0.5*self.length, self.xpos + 0.5*self.length, 
                self.length, self.width, self.height, self.mass, self.cm[0])
        
#--------------------------------------------
    
class Bedplate(object):
    ''' Bedplate class
          The Bedplate class is used to represent the bedplate of a wind turbine drivetrain.
          It contains the general properties for a wind turbine component as well as additional design load and 
            dimensional attributes as listed below.
          It contains an update method to determine the mass, mass properties, and dimensions of the component.
          
        b0, h0: I-beam width, height
        tw, tf: I-beam web thickness, flange thickness
    '''

    def __init__(self, uptower_transformer=True, debug=False):

        super(Bedplate, self).__init__()

        self.uptower_transformer = uptower_transformer # Bool(iotype = 'in', desc = 'Boolean stating if transformer is uptower')

        self.debug = debug
        
        self.nBeams = 2    # number of parallel I-beams in each element
        self.nElements = 0 # number of distinct bedplate elements (along X axis)
        
        self.elements = [] # list of bedplate elements, ordered from upwind to downwind
    
        #Standard constants and material properties
        self.g = 9.81
        
        self.steelDensity = 7800    # kg/m^3
        self.castDensity  = 7100    # kg/m^3
        self.steelE = 210e9         # Young's modulus of steel     in N/m^2
        self.castE  = 169e9         # Young's modulus of cast iron in N/m^2
        self.steelStressMax = 620e6 # yield strength of alloy steel in MPa (1e6N/m^2)
        self.castStressMax  = 200e6 # yield strength of cast iron   in MPa (1e6N/m^2)

        # parameters for sizing elements
        
        self.defl_denom  = 1500.  # factor in deflection check
        self.stressTol   = 5e5
        self.deflTol     = 1e-4
        self.stress_mult = 8.     # modified to fit industry data

        
    # ------------------
        
    # functions used in bedplate sizing
    def midDeflection(self, totalLength, loadLength, load, E, I):
        ''' Eq. 2.154 - tip deflection for load applied at x (Eq. 2.66 in 2015 rpt) '''
        defl = load * loadLength**2.0 * \
            (3.0 * totalLength - loadLength) / (6.0 * E * I)
        return defl
    
    def distDeflection(self, totalLength, distWeight, E, I):
        ''' Eq. 2.155 - tip deflection for distributed load (Eq. 2.67 in 2015 rpt)'''
        defl = distWeight * totalLength**4 / (8.0 * E * I)
        return defl
    
    # ------------------
        
    def compute(self, \
                      tower_top_diameter):
        ''', gearbox_length, gearbox_location, gearbox_mass, hss_location, hss_mass, generator_location, generator_mass, \
                      lss_location, lss_mass, lss_length, mb1_cm, mb1_facewidth, mb1_mass, mb2_cm, mb2_mass, \
                      transformer_mass, transformer_cm, rotor_diameter, machine_rating, rotor_mass, rotor_bending_moment_y, rotor_force_z, \
                      flange_length, distance_hub2mb):
        '''
        
        '''Model bedplate as 2 parallel I-beams with a rear steel frame and a front cast frame
           Deflection constraints applied at each bedplate end
           Stress constraint checked at root of front and rear bedplate sections'''

        #if self.debug:
        #    sys.stderr.write('GBox loc {} mass {}\n'.format(gearbox_location, gearbox_mass))
        
        #variables
        #self.gearbox_length = gearbox_length #Float(iotype = 'in', units = 'm', desc = 'gearbox length')
        self.tower_top_diameter = tower_top_diameter #Float(iotype = 'in', units = 'm', desc = 'tower_top_diameter')
        
        # find sizes for each element
        
        self.bedplate_length = 0.0
        self.height = 0.0
        self.width = 0.0
        self.mass = 0.0
        self.totalCastMass = 0.0
        self.totalSteelMass = 0.0
        cmx = 0.0 # accumulate COM terms
        
        for element in self.elements:
            element.characterize(self.stressTol, self.deflTol)
            self.height = np.max([self.height, element.height])
            self.width = np.max([self.width, element.width])
            self.bedplate_length += element.length
            self.mass += element.mass
            if element.material_name == 'cast':
                self.totalCastMass += element.mass 
            if element.material_name == 'steel':
                self.totalSteelMass += element.mass 
            cmx += element.mass * element.xpos
            
        #self.bedplate_length = self.frontTotalLength + self.rearTotalLength
        #self.width = self.b0 + self.tower_top_diameter
        self.width += self.tower_top_diameter
        #self.height = np.max([self.frontHeight, self.rearHeight])
  
        # calculate mass properties
        cm = np.zeros(3)
        #cm[0] = (self.totalSteelMass * self.rearTotalLength/2 
        #       - self.totalCastMass  * self.frontTotalLength/2) / self.mass 
        cm[0] = cmx / self.mass
        cm[1] = 0.0
        cm[2] = -self.height/2.
        self.cm = cm
  
        self.depth = self.bedplate_length / 2.0
  
        I = np.zeros(3)
        I[0]  = self.mass * (self.width ** 2 + self.depth ** 2) / 8
        I[1]  = self.mass * (self.depth ** 2 + self.width ** 2 + (4/3) * self.bedplate_length ** 2) / 16
        I[2]  = I[1] # is this correct?
        self.I = I

        if self.debug:
            sys.stderr.write('Bedplate: mass {:.1f} cast {:.1f} steel {:.1f} L {:.1f} m H {:.1f} m W {:.1f} m\n'.format(self.mass, 
                             self.totalCastMass, self.totalSteelMass, self.bedplate_length, self.height, self.width))
            sys.stderr.write(self.elements[0].dumphdr()) #'   Name    matrl  xpos    L     W     H    Mass\n')
            for element in self.elements:
                sys.stderr.write('{}\n'.format(element.dump()))
                
            '''
            sys.stderr.write('Bedplate: frontLen {:.1f} m rearLen {:.1f} m nFront {} nRear {} \n'.format(self.frontTotalLength, 
                             self.rearTotalLength, frontCounter, rearCounter))
            sys.stderr.write('  LSS         {:5.2f} m  {:8.1f} kg\n'.format(self.lss_location, self.lss_mass))
            sys.stderr.write('  HSS         {:5.2f} m  {:8.1f} kg\n'.format(self.hss_location, self.hss_mass))
            sys.stderr.write('  Gearbox     {:5.2f} m  {:8.1f} kg\n'.format(gearbox_location, gearbox_mass))
            sys.stderr.write('  Generator   {:5.2f} m  {:8.1f} kg\n'.format(self.generator_location, self.generator_mass))
            sys.stderr.write('  Transformer {:5.2f} m  {:8.1f} kg\n'.format(self.transformer_location, self.transformer_mass))
            '''
            
        return (self.mass, self.cm, self.I, self.bedplate_length, self.height, self.width)

#%%-----------------------------

if __name__=='__main__':
    
    debug = True
    
    tower_top_diameter = 4.5
    
    bp = Bedplate(debug=debug) 

    # ------ Define bedplate elements ------
    #  This should be the only part that needs to be changed if the bedplate configuration changes
    
    bp.elements.append(BPelement('elemFront', bp.castDensity,  bp.castE,  'cast',  bp.nBeams, debug=debug))
    bp.elements.append(BPelement('elemRear',  bp.steelDensity, bp.steelE, 'steel', bp.nBeams, debug=debug))
    
    bp.elements[0].length = 3.0
    bp.elements[1].length = 5.0
    bp.elements[0].width =  2.0
    bp.elements[1].width =  2.0
    bp.elements[0].xpos =  -1.5
    bp.elements[1].xpos =   2.5
    
    bp.elements[1].loads    = [10000.,  5000., 2000.]
    bp.elements[1].loadlocs = [-2.,     0.,    1.8]  # relative to xpos
    
    # ------ END Define bedplate elements ------
        

    bp.compute(tower_top_diameter)  
     