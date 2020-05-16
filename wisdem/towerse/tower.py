#!/usr/bin/env python
# encoding: utf-8
"""
tower.py

Originally created by Andrew Ning on 2012-01-20.
Copyright (c) NREL. All rights reserved.
"""

from __future__ import print_function

import numpy as np
import openmdao.api as om

from wisdem.commonse.WindWaveDrag import AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag

from wisdem.commonse.environment import LinearWaves, TowerSoil, PowerWind, LogWind
from wisdem.commonse.tube import CylindricalShellProperties
from wisdem.commonse.utilities import assembleI, unassembleI, nodal2sectional, interp_with_deriv, sectionalInterp
from wisdem.commonse import gravity, eps

from wisdem.commonse.vertical_cylinder import CylinderDiscretization, CylinderMass, CylinderFrame3DD, NFREQ

import wisdem.commonse.UtilizationSupplement as Util


def find_nearest(array, value):
    return (np.abs(array-value)).argmin() 

NREFINE = 3
def get_nfull(npts):
    nFull = int( 1 + NREFINE*(npts-1) )
    return nFull

def get_npts(nFull):
    npts = int( 1 + (nFull-1)/NREFINE )
    return npts

# -----------------
#  Components
# -----------------


class DiscretizationYAML(om.ExplicitComponent):
    """
    Process some of the tower YAML inputs.
    
    Parameters
    ----------
    tower_s : numpy array[n_height_tow]
        1D array of the non-dimensional grid defined along the tower axis (0-tower base,
        1-tower top)
    tower_layer_materials : list of strings
        1D array of the names of the materials of each layer modeled in the tower
        structure.
    tower_layer_thickness : numpy array[n_layers_tow, n_height_tow-1]
        2D array of the thickness of the layers of the tower structure. The first
        dimension represents each layer, the second dimension represents each piecewise-
        constant entry of the tower sections.
    tower_height : float
        Scalar of the tower height computed along the z axis.
    tower_outer_diameter_in : numpy array[n_height_tow]
        cylinder diameter at corresponding locations
    tower_outfitting_factor : float
        Multiplier that accounts for secondary structure mass inside of cylinder
    monopile_s : numpy array[n_height_mon]
        1D array of the non-dimensional grid defined along the tower axis (0-tower base,
        1-tower top)
    monopile_layer_materials : list of strings
        1D array of the names of the materials of each layer modeled in the tower
        structure.
    monopile_layer_thickness : numpy array[n_layers_mon, n_height_mon_minus]
        2D array of the thickness of the layers of the tower structure. The first
        dimension represents each layer, the second dimension represents each piecewise-
        constant entry of the tower sections.
    monopile_height : float
        Scalar of the tower height computed along the z axis.
    monopile_outer_diameter_in : numpy array[n_height_tow]
        cylinder diameter at corresponding locations
    monopile_outfitting_factor : float
        Multiplier that accounts for secondary structure mass inside of cylinder
    material_names : list of strings
        1D array of names of materials.
    E_mat : numpy array[n_mat, 3]
        2D array of the Youngs moduli of the materials. Each row represents a material,
        the three columns represent E11, E22 and E33.
    G_mat : numpy array[n_mat, 3]
        2D array of the shear moduli of the materials. Each row represents a material,
        the three columns represent G12, G13 and G23.
    sigma_y_mat : numpy array[n_mat]
        2D array of the yield strength of the materials. Each row represents a material,
        the three columns represent Xt12, Xt13 and Xt23.
    rho_mat : numpy array[n_mat]
        1D array of the density of the materials. For composites, this is the density of
        the laminate.
    unit_cost_mat : numpy array[n_mat]
        1D array of the unit costs of the materials.
    
    Returns
    -------
    tower_section_height : numpy array[n_height-1]
        parameterized section heights along cylinder
    tower_outer_diameter : numpy array[n_height]
        cylinder diameter at corresponding locations
    tower_wall_thickness : numpy array[n_height-1]
        shell thickness at corresponding locations
    outfitting_factor : numpy array[n_height-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    E : numpy array[n_height-1]
        Isotropic Youngs modulus of the materials along the tower sections.
    G : numpy array[n_height-1]
        Isotropic shear modulus of the materials along the tower sections.
    sigma_y : numpy array[n_height-1]
        Isotropic yield strength of the materials along the tower sections.
    rho : numpy array[n_height-1]
        Density of the materials along the tower sections.
    unit_cost : numpy array[n_height-1]
        Unit costs of the materials along the tower sections.
    
    """
    def initialize(self):
        self.options.declare('n_height_tower')
        self.options.declare('n_height_monopile')
        self.options.declare('n_layers_tower')
        self.options.declare('n_layers_monopile')
        self.options.declare('n_mat')
        
    def setup(self):
        n_height_tow = self.options['n_height_tower']
        n_height_mon = self.options['n_height_monopile']
        n_layers_tow = self.options['n_layers_tower']
        n_layers_mon = self.options['n_layers_monopile']
        n_mat        = self.options['n_mat']
        if n_height_mon > 0:
            n_height           = n_height_tow + n_height_mon - 1 # Should have one overlapping point
            n_height_mon_minus = n_height_mon - 1
        else:
            n_height           = n_height_tow
            n_height_mon_minus = 0
            
        # Inputs here are the outputs from the Tower component in load_IEA_yaml
        # TODO: Use reference axis and curvature, s, instead of assuming everything is vertical on z
        self.add_input('tower_s', val=np.zeros(n_height_tow))
        self.add_discrete_input('tower_layer_materials', val=n_layers_tow * [''])
        self.add_input('tower_layer_thickness', val=np.zeros((n_layers_tow, n_height_tow-1)), units='m')
        self.add_input('tower_height', val = 0.0, units='m')
        self.add_input('tower_outer_diameter_in', np.zeros(n_height_tow), units='m')
        self.add_input('tower_outfitting_factor', val=0.0)
        self.add_input('monopile_s', val=np.zeros(n_height_mon))
        self.add_discrete_input('monopile_layer_materials', val=n_layers_tow * [''])
        self.add_input('monopile_layer_thickness', val=np.zeros((n_layers_mon, n_height_mon_minus)), units='m')
        self.add_input('monopile_height', val = 0.0, units='m')
        self.add_input('monopile_outer_diameter_in', np.zeros(n_height_tow), units='m')
        self.add_input('monopile_outfitting_factor', val=0.0)
        self.add_discrete_input('material_names', val=n_mat * [''])
        self.add_input('E_mat', val=np.zeros([n_mat, 3]), units='Pa')
        self.add_input('G_mat', val=np.zeros([n_mat, 3]), units='Pa')
        self.add_input('sigma_y_mat', val=np.zeros(n_mat), units='Pa')
        self.add_input('rho_mat', val=np.zeros(n_mat), units='kg/m**3')
        self.add_input('unit_cost_mat', val=np.zeros(n_mat), units='USD/kg')

        self.add_output('tower_section_height', val=np.zeros(n_height-1), units='m')
        self.add_output('tower_outer_diameter', val=np.zeros(n_height), units='m')
        self.add_output('tower_wall_thickness', val=np.zeros(n_height-1), units='m')
        self.add_output('outfitting_factor', val=np.zeros(n_height-1))
        self.add_output('E', val=np.zeros(n_height-1), units='Pa')
        self.add_output('G', val=np.zeros(n_height-1), units='Pa')
        self.add_output('sigma_y', val=np.zeros(n_height-1), units='Pa')
        self.add_output('rho', val=np.zeros(n_height-1), units='kg/m**3')
        self.add_output('unit_cost', val=np.zeros(n_height-1), units='USD/kg')

        # self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack dimensions
        n_height_tow = self.options['n_height_tower']
        n_height_mon = self.options['n_height_monopile']
        n_layers_tow = self.options['n_layers_tower']
        n_layers_mon = self.options['n_layers_monopile']
        n_height     = n_height_tow if n_height_mon == 0 else n_height_tow + n_height_mon - 1

        # Unpack values
        h_mon = inputs['monopile_height']
        h_tow = inputs['tower_height']
        lthick_mon = inputs['monopile_layer_thickness']
        lthick_tow = inputs['tower_layer_thickness']
        lmat_mon = discrete_inputs['monopile_layer_materials']
        lmat_tow = discrete_inputs['tower_layer_materials']
        
        if n_height_mon > 0:
            # Last monopile point and first tower point are the same
            outputs['tower_section_height'] = np.r_[ np.diff( h_mon * inputs['monopile_s'] ),
                                                     np.diff( h_tow * inputs['tower_s'] ) ]
            outputs['outfitting_factor']    =  np.r_[inputs['monopile_outfitting_factor']*np.ones(n_height_mon-1),
                                                     inputs['tower_outfitting_factor']*np.ones(n_height_tow-1)]
            outputs['tower_outer_diameter'] = np.r_[inputs['monopile_outer_diameter_in'],
                                                    inputs['tower_outer_diameter_in'][1:]]

            # Combine layers into one structure
            layer_mat = []
            twall     = np.zeros((1, n_height-1))
            for k in range(n_layers_mon):
                ilayer = np.zeros(n_height-1)

                ilayer[:(n_height_mon-1)] = lthick_mon[k,:]

                imat_mon = lmat_mon[k]
                layer_mat.append(imat_mon)
                
                if imat_mon in lmat_tow:
                    ktow = lmat_tow.index( imat_mon )
                    ilayer[(n_height_mon-1):] = lthick_tow[ktow,:]

                    # Remove from listing so we don't double count later
                    lmat_tow.pop( ktow )
                    lthick_tow = np.delete(lthick_tow, [ktow], axis=0)

                twall = np.vstack( (twall, ilayer) )

            # If there any uncounted tower layers, add them in
            n_layers_tow = len(lmat_tow)
            for k in range(n_layers_tow):
                ilayer = np.zeros(n_height-1)
                ilayer[(n_height_mon-1):] = lthick_tow[k,:]
                twall  = np.vstack( (twall, ilayer) )
                imat  = lmat_tow[k]
                layer_mat.append(imat)
                
            twall = np.delete(twall, [0], axis=0)
            outputs['tower_wall_thickness'] = np.sum(twall, axis=0)
            
        else:
            outputs['tower_section_height'] = np.diff( h_tow * inputs['tower_s'] )
            outputs['tower_wall_thickness'] = np.sum(inputs['tower_layer_thickness'], axis=0)
            outputs['outfitting_factor']    = inputs['tower_outfitting_factor']*np.ones(n_height-1)
            outputs['tower_outer_diameter'] = inputs['tower_outer_diameter_in']
            twall     = inputs['tower_layer_thickness']
            layer_mat = discrete_inputs['tower_layer_materials']

        # Check to make sure we have good values
        if np.any(outputs['tower_section_height'] <= 0.0):
            raise ValueError('Section height values must be greater than zero, '+str(outputs['tower_section_height']))
        if np.any(outputs['tower_wall_thickness'] <= 0.0):
            raise ValueError('Wall thickness values must be greater than zero, '+str(outputs['tower_wall_thickness']))
        if np.any(outputs['tower_outer_diameter'] <= 0.0):
            raise ValueError('Diameter values must be greater than zero, '+str(outputs['tower_outer_diameter']))

        # DETERMINE MATERIAL PROPERTIES IN EACH SECTION
        # Convert to isotropic material
        E    = np.mean(inputs['E_mat'], axis=1)
        G    = np.mean(inputs['G_mat'], axis=1)
        sigy = inputs['sigma_y_mat']
        rho  = inputs['rho_mat']
        cost = inputs['unit_cost_mat']
        mat_names = discrete_inputs['material_names']

        # Initialize sectional data
        myzeros    = np.zeros(n_height - 1)
        E_param    = myzeros.copy()
        G_param    = myzeros.copy()
        sigy_param = myzeros.copy()
        rho_param  = myzeros.copy()
        cost_param = myzeros.copy()
        for k in range(len(layer_mat)):
            # Get the material name for this layer
            iname = layer_mat[k]

            # Get the index into the material list
            imat  = mat_names.index( iname )

            # Thickness of this layer
            tlay   = twall[k,:]
            
            # Find sections where this layer is non-zero
            ilay  = np.where(tlay > 0.0, 1.0, 0.0)

            # For stiffness properties, take the maximum
            E_param    = np.maximum(E_param, E[imat]*ilay)
            G_param    = np.maximum(G_param, G[imat]*ilay)
            sigy_param = np.maximum(sigy_param, sigy[imat]*ilay)

            # For density, take thickness weighted layer
            rho_param += rho[imat]*tlay

            # For cost, take mass weighted layer
            cost_param += rho[imat]*tlay*cost[imat]

        # Mass weighted cost (should really weight by radius too)
        cost_param /= rho_param
        
        # Thickness weighted density (should really weight by radius too)
        rho_param /= twall.sum(axis=0)

        # Find values at finer grid
        outputs['E']   = E_param
        outputs['G']   = G_param
        outputs['rho'] = rho_param
        outputs['sigma_y']   = sigy_param
        outputs['unit_cost'] = cost_param

        
class MonopileFoundation(om.ExplicitComponent):
    """
    Compute the monopile foundation parameterized section heights.
    
    Parameters
    ----------
    suctionpile_depth : float
        depth of foundation in the soil
    suctionpile_depth_diam_ratio : float
        ratio of sunction pile depth to mudline monopile diameter
    foundation_height : float
        height of foundation (0.0 for land, -water_depth for fixed bottom)
    base_diameter : float
        cylinder diameter at base
    
    Returns
    -------
    z_start : float
        parameterized section heights along cylinder
    
    """
    def initialize(self):
        self.options.declare('monopile')
    
    def setup(self):
        self.add_input('suctionpile_depth', 0.0, units='m')
        self.add_input('suctionpile_depth_diam_ratio', 0.0)
        self.add_input('foundation_height', 0.0, units='m')
        self.add_input('base_diameter', 0.0, units='m')

        self.add_output('z_start', 0.0, units='m')
        
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs['z_start'] = inputs['foundation_height']
        
        if self.options['monopile']:
            pile = inputs['suctionpile_depth']
            if pile == 0.0:
                pile = inputs['suctionpile_depth_diam_ratio'] * inputs['base_diameter']
            outputs['z_start'] -= np.abs(pile)
            
                                                 
class TowerDiscretization(om.ExplicitComponent):
    """
    Compute the full arrays for some measures along the tower by interpolating.
    
    Parameters
    ----------
    hub_height : float
        diameter at tower base
    z_param : numpy array[n_height]
        parameterized locations along tower, linear lofting between
    z_full : numpy array[nFull]
        parameterized locations along tower, linear lofting between
    rho : numpy array[n_height-1]
        Density of the materials along the tower sections.
    unit_cost : numpy array[n_height-1]
        Unit costs of the materials along the tower sections.
    outfitting_factor : numpy array[n_height-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    
    Returns
    -------
    height_constraint : float
        mismatch between tower height and desired hub_height
    rho_full : numpy array[nFull-1]
        Density of the materials along the tower sections.
    unit_cost_full : numpy array[nFull-1]
        Unit costs of the materials along the tower sections.
    outfitting_full : numpy array[nFull-1]
        Multiplier that accounts for secondary structure mass inside of cylinder
    
    """
    def initialize(self):
        self.options.declare('n_height')
    
    def setup(self):
        n_height = self.options['n_height']
        nFull = get_nfull(n_height)

        self.add_input('hub_height', val=0.0, units='m')
        self.add_input('z_param', np.zeros(n_height), units='m')
        self.add_input('z_full', val=np.zeros(nFull), units='m')
        self.add_input('rho', val=np.zeros(n_height-1), units='kg/m**3')
        self.add_input('unit_cost', val=np.zeros(n_height-1), units='USD/kg')
        self.add_input('outfitting_factor', val=np.zeros(n_height-1))

        self.add_output('height_constraint', val=0.0, units='m')
        self.add_output('rho_full', val=np.zeros(nFull-1), units='kg/m**3')
        self.add_output('unit_cost_full', val=np.zeros(nFull-1), units='USD/kg')
        self.add_output('outfitting_full', val=np.zeros(nFull-1))
        
        self.declare_partials('height_constraint', ['hub_height', 'z_param'], method='fd')
        self.declare_partials('outfitting_full', ['outfitting_factor'], method='fd')
        self.declare_partials('rho_full', ['rho'], method='fd')
        self.declare_partials('unit_cost_full', ['unit_cost'], method='fd')
        
    def compute(self, inputs, outputs):
        z_full    = inputs['z_full']
        z_param   = inputs['z_param']
        z_section = 0.5*(z_full[:-1] + z_full[1:])

        outputs['height_constraint'] = inputs['hub_height'] - z_param[-1]
        outputs['rho_full']        = sectionalInterp(z_section, z_param, inputs['rho'])
        outputs['outfitting_full'] = sectionalInterp(z_section, z_param, inputs['outfitting_factor'])
        outputs['unit_cost_full']  = sectionalInterp(z_section, z_param, inputs['unit_cost'])

    #def compute_partials(self, inputs, J):
    #    n_height = self.options['n_height']
        
    #    J['height_constraint','hub_height'] = 1.
    #    J['height_constraint','z_param'] = np.zeros(n_height)
    #    J['height_constraint','z_param'][-1] = -1.
    
            
class TowerMass(om.ExplicitComponent):
    """
    Compute the tower and monopile masses, raw cost, and CG properties.
    
    Parameters
    ----------
    cylinder_mass : numpy array[nFull-1]
        Total cylinder mass
    cylinder_cost : float
        Total cylinder cost
    cylinder_center_of_mass : float
        z position of center of mass of cylinder
    cylinder_section_center_of_mass : numpy array[nFull-1]
        z position of center of mass of each can in the cylinder
    cylinder_I_base : numpy array[6]
        Mass moment of inertia of cylinder about base [xx yy zz xy xz yz]
    transition_piece_height : float
        Point mass height of transition piece above water line
    transition_piece_mass : float
        Point mass of transition piece
    gravity_foundation_mass : float
        Extra mass of gravity foundation
    foundation_height : float 
        Height of foundation (0.0 for land, -water_depth for fixed bottom)
    z_full : numpy array[nFull]
        Parameterized locations along tower, linear lofting between
    
    Returns
    -------
    tower_raw_cost : float
        Total tower cost
    tower_mass : float
        Total tower mass
    tower_center_of_mass : float
        z-position of center of mass of tower
    tower_section_center_of_mass : numpy array[nFull-1]
        z position of center of mass of each can in the tower
    tower_I_base : numpy array[6]
        Mass moment of inertia of tower about base [xx yy zz xy xz yz]
    monopile_mass : float 
        Mass of monopile from bottom of suction pile through transition piece
    monopile_cost : float
        Total monopile cost
    monopile_length : float
        Length of monopile from bottom of suction pile through transition piece
        
    """

    def initialize(self):
        self.options.declare('n_height')
    
    def setup(self):
        n_height = self.options['n_height']
        nFull = get_nfull(n_height)
        
        self.add_input('cylinder_mass', val=np.zeros(nFull-1), units='kg')
        self.add_input('cylinder_cost', val=0.0, units='USD')
        self.add_input('cylinder_center_of_mass', val=0.0, units='m')
        self.add_input('cylinder_section_center_of_mass', val=np.zeros(nFull-1), units='m')
        self.add_input('cylinder_I_base', np.zeros(6), units='kg*m**2')
        self.add_input('transition_piece_height', 0.0, units='m')
        self.add_input('transition_piece_mass', 0.0, units='kg')
        self.add_input('gravity_foundation_mass', 0.0, units='kg')
        self.add_input('foundation_height', 0.0, units='m')
        self.add_input('z_full', val=np.zeros(nFull), units='m')

        self.add_output('tower_raw_cost', val=0.0, units='USD')
        self.add_output('tower_mass', val=0.0, units='kg')
        self.add_output('tower_center_of_mass', val=0.0, units='m')
        self.add_output('tower_section_center_of_mass', val=np.zeros(nFull-1), units='m')
        self.add_output('tower_I_base', np.zeros(6), units='kg*m**2')
        self.add_output('monopile_mass', val=0.0, units='kg')
        self.add_output('monopile_cost', val=0.0, units='USD')
        self.add_output('monopile_length', val=0.0, units='m')
        
        self.declare_partials('monopile_cost', ['transition_piece_height', 'z_full'], method='fd')
        self.declare_partials('monopile_length', ['transition_piece_height', 'z_full'], method='fd')
        self.declare_partials('monopile_mass', ['gravity_foundation_mass', 'transition_piece_height', 'transition_piece_mass', 'z_full'], method='fd')
        self.declare_partials('tower_I_base', ['cylinder_I_base'], method='fd')
        self.declare_partials('tower_center_of_mass', ['cylinder_center_of_mass', 'gravity_foundation_mass', 'transition_piece_mass'], method='fd')
        self.declare_partials('tower_mass', ['cylinder_mass', 'transition_piece_height', 'z_full'], method='fd')
        self.declare_partials('tower_raw_cost', ['cylinder_cost'], method='fd')
        self.declare_partials('tower_section_center_of_mass', ['cylinder_section_center_of_mass'], method='fd')
        
    def compute(self, inputs, outputs):
        outputs['tower_raw_cost']       = inputs['cylinder_cost']
        outputs['tower_mass']           = inputs['cylinder_mass'].sum()
        outputs['tower_center_of_mass'] = ( (inputs['cylinder_center_of_mass']*outputs['tower_mass'] +
                                             inputs['transition_piece_mass']*inputs['transition_piece_height'] +
                                             inputs['gravity_foundation_mass']*inputs['foundation_height']) /
                                            (outputs['tower_mass']+inputs['transition_piece_mass']+inputs['gravity_foundation_mass']) )
        outputs['tower_section_center_of_mass'] = inputs['cylinder_section_center_of_mass']
        outputs['tower_I_base']         = inputs['cylinder_I_base']

        outputs['monopile_mass'],dydx,dydxp,dydyp = interp_with_deriv(inputs['transition_piece_height'],
                                                                      inputs['z_full'],
                                                                      np.r_[0.0, np.cumsum(inputs['cylinder_mass'])])
        outputs['tower_mass']     -= outputs['monopile_mass']
        outputs['monopile_cost']   = inputs['cylinder_cost']*outputs['monopile_mass']/inputs['cylinder_mass'].sum()
        outputs['monopile_mass']  += inputs['transition_piece_mass'] + inputs['gravity_foundation_mass']
        outputs['monopile_length'] = inputs['transition_piece_height'] - inputs['z_full'][0]

        # self.J = {}
        # self.J['monopile_mass', 'z_full'] = dydxp[0,:]
        # self.J['monopile_mass', 'cylinder_mass'] = dydyp[0,1:]
        # self.J['monopile_mass', 'transition_piece_height'] = dydx[0,0]
        # 
        # self.J['monopile_cost', 'z_full'] = inputs['cylinder_cost'] * self.J['monopile_mass', 'z_full'] / inputs['cylinder_mass'].sum()
        # self.J['monopile_cost', 'cylinder_cost'] = outputs['monopile_mass']/ inputs['cylinder_mass']
        # self.J['monopile_cost', 'cylinder_mass'] = inputs['cylinder_cost']*self.J['monopile_mass', 'cylinder_mass']/inputs['cylinder_mass'] - outputs['monopile_cost']/inputs['cylinder_mass']
        # self.J['monopile_cost', 'transition_piece_height'] = inputs['cylinder_cost'] * self.J['monopile_mass', 'transition_piece_height'] / inputs['cylinder_mass']
        
    # def compute_partials(self, inputs, J):
    #     J['tower_mass','cylinder_mass'] = np.ones(len(inputs['cylinder_mass'])) - self.J['monopile_mass', 'cylinder_mass']
    #     J['tower_mass','transition_piece_mass'] = 1.0
    #     J['tower_mass', 'z_full'] = -self.J['monopile_mass', 'z_full']
    #     J['tower_mass', 'transition_piece_height'] = -self.J['monopile_mass', 'transition_piece_height']
    # 
    #     J['tower_raw_cost','cylinder_cost'] = 1.0
    # 
    #     J['tower_center_of_mass','cylinder_center_of_mass'] = 1.0
    # 
    #     J['tower_section_center_of_mass','cylinder_section_center_of_mass'] = np.eye(len(inputs['cylinder_section_center_of_mass']))
    # 
    #     J['tower_I_base','cylinder_I_base'] = np.eye(len(inputs['cylinder_I_base']))
    # 
    #     J['monopile_mass', 'z_full'] = self.J['monopile_mass', 'z_full']
    #     J['monopile_mass', 'cylinder_mass'] = self.J['monopile_mass', 'cylinder_mass']
    #     J['monopile_mass', 'transition_piece_height'] = self.J['monopile_mass', 'transition_piece_height']
    #     J['monopile_mass', 'transition_piece_mass'] = 1.0
    #     J['monopile_mass', 'gravity_foundation_mass'] = 1.0
    # 
    #     J['monopile_cost', 'z_full'] = self.J['monopile_cost', 'z_full']
    #     J['monopile_cost', 'cylinder_cost'] = self.J['monopile_cost', 'cylinder_cost']
    #     J['monopile_cost', 'cylinder_mass'] = self.J['monopile_cost', 'cylinder_mass']
    #     J['monopile_cost', 'transition_piece_height'] = self.J['monopile_cost', 'transition_piece_height']
    # 
    #     J['monopile_length','transition_piece_height'] = 1.
    #     J['monopile_length','z_full'] = np.zeros(inputs['z_full'].size)
    #     J['monopile_length','z_full'][0] = -1.
        
        
class TurbineMass(om.ExplicitComponent):
    """
    Compute the turbine mass, center of mass, and mass moment of inertia.
    
    Parameters
    ----------
    hub_height : float
        Hub height
    rna_mass : float
        Total tower mass
    rna_I : numpy array[6]
        Mass moment of inertia of RNA about tower top [xx yy zz xy xz yz]
    rna_cg : numpy array[3]
        xyz-location of RNA cg relative to tower top
    tower_mass : float
        Total tower mass (not including monopile)
    monopile_mass : float
        Monopile mass
    tower_center_of_mass : float
        z-position of center of mass of tower
    tower_I_base : numpy array[6]
        Mass moment of inertia of tower about base [xx yy zz xy xz yz]
        
    Returns
    -------
    turbine_mass : float
        Total mass of tower and RNA combined
    turbine_center_of_mass : numpy array[3]
        xyz-position of tower and RNA center of mass
    turbine_I_base : numpy array[6]
        Mass moment of inertia of tower about base [xx yy zz xy xz yz]
        
    """

    def setup(self):
        self.add_input('hub_height', val=0.0, units='m')
        self.add_input('rna_mass', val=0.0, units='kg')
        self.add_input('rna_I', np.zeros(6), units='kg*m**2')
        self.add_input('rna_cg', np.zeros(3), units='m')
        self.add_input('tower_mass', val=0.0, units='kg')
        self.add_input('monopile_mass', val=0.0, units='kg')
        self.add_input('tower_center_of_mass', val=0.0, units='m')
        self.add_input('tower_I_base', np.zeros(6), units='kg*m**2')

        self.add_output('turbine_mass', val=0.0, units='kg')
        self.add_output('turbine_center_of_mass', val=np.zeros(3), units='m')
        self.add_output('turbine_I_base', np.zeros(6), units='kg*m**2')
        
        self.declare_partials('turbine_I_base', ['hub_height', 'rna_I', 'rna_cg', 'rna_mass', 'tower_I_base'], method='fd')
        self.declare_partials('turbine_center_of_mass', ['hub_height', 'monopile_mass', 'rna_cg', 'rna_mass', 'tower_center_of_mass', 'tower_mass'], method='fd')
        self.declare_partials('turbine_mass', ['monopile_mass', 'rna_mass', 'tower_mass'], val=1.)
        
    def compute(self, inputs, outputs):
        outputs['turbine_mass'] = inputs['rna_mass'] + inputs['tower_mass'] + inputs['monopile_mass']
        
        cg_rna   = inputs['rna_cg'] + np.array([0.0, 0.0, inputs['hub_height']])
        cg_tower = np.array([0.0, 0.0, inputs['tower_center_of_mass']])
        outputs['turbine_center_of_mass'] = (inputs['rna_mass'] * cg_rna + inputs['tower_mass'] * cg_tower) / outputs['turbine_mass']

        R       = cg_rna
        I_tower = assembleI(inputs['tower_I_base'])
        I_rna   = assembleI(inputs['rna_I']) + inputs['rna_mass']*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        outputs['turbine_I_base'] = unassembleI(I_tower + I_rna)

        
class TowerPreFrame(om.ExplicitComponent):
    """
    Compute some properties of the tower needed for FEM analysis.
    
    This component can be simplified by using src_indices for data-passing.
    At the very least, we can code the sparse derivatives as-is for
    input-output relationships.
    
    Parameters
    ----------
    z_full : numpy array[nFull]
        location along tower. start at bottom and go to top
    d_full : numpy array[nFull]
        diameter along tower
    z_param : numpy array[n_height]
        parameterized locations along tower, linear lofting between
    mass : float
        added mass
    mI : numpy array[6]
        mass moment of inertia about some point p [xx yy zz xy xz yz]
    mrho : numpy array[3]
        xyz-location of p relative to node
    transition_piece_mass : float
        point mass of transition piece
    gravity_foundation_mass : float
        point mass of transition piece
    transition_piece_height : float
        height of transition piece above water line
    foundation_height : float
        height of foundation (0.0 for land, -water_depth for fixed bottom)
    rna_F : numpy array[3]
        rna force
    rna_M : numpy array[3]
        rna moment
    k_monopile : numpy array[6]
        Stiffness BCs for ocean soil. Only used if monoflag inputis True
    E : numpy array[n_height-1]
        Isotropic Youngs modulus of the materials along the tower sections.
    G : numpy array[n_height-1]
        Isotropic shear modulus of the materials along the tower sections.
    sigma_y : numpy array[n_height-1]
        Isotropic yield strength of the materials along the tower sections.
    
    Returns
    -------
    E_full : numpy array[nFull-1]
        Isotropic Youngs modulus of the materials along the tower sections.
    G_full : numpy array[nFull-1]
        Isotropic shear modulus of the materials along the tower sections.
    sigma_y_full : numpy array[nFull-1]
        Isotropic yield strength of the materials along the tower sections.
    kidx : numpy array[np.int_]
        indices of z where external stiffness reactions should be applied.
    kx : numpy array[nK]
        spring stiffness in x-direction
    ky : numpy array[nK]
        spring stiffness in y-direction
    kz : numpy array[nK]
        spring stiffness in z-direction
    ktx : numpy array[nK]
        spring stiffness in theta_x-rotation
    kty : numpy array[nK]
        spring stiffness in theta_y-rotation
    ktz : numpy array[nK]
        spring stiffness in theta_z-rotation
    midx : numpy array[np.int_]
        indices where added mass should be applied.
    m : numpy array[nMass]
        added mass
    mIxx : numpy array[nMass]
        x mass moment of inertia about some point p
    mIyy : numpy array[nMass]
        y mass moment of inertia about some point p
    mIzz : numpy array[nMass]
        z mass moment of inertia about some point p
    mIxy : numpy array[nMass]
        xy mass moment of inertia about some point p
    mIxz : numpy array[nMass]
        xz mass moment of inertia about some point p
    mIyz : numpy array[nMass]
        yz mass moment of inertia about some point p
    mrhox : numpy array[nMass]
        x-location of p relative to node
    mrhoy : numpy array[nMass]
        y-location of p relative to node
    mrhoz : numpy array[nMass]
        z-location of p relative to node
    plidx : numpy array[np.int_]
        indices where point loads should be applied.
    Fx : numpy array[nPL]
        point force in x-direction
    Fy : numpy array[nPL]
        point force in y-direction
    Fz : numpy array[nPL]
        point force in z-direction
    Mxx : numpy array[nPL]
        point moment about x-axis
    Myy : numpy array[nPL]
        point moment about y-axis
    Mzz : numpy array[nPL]
        point moment about z-axis
    
    """
    def initialize(self):
        self.options.declare('n_height')
        self.options.declare('monopile', default=False)
    
    def setup(self):
        n_height = self.options['n_height']
        nFull   = get_nfull(n_height)
        
        self.add_input('z_full', np.zeros(nFull), units='m')
        self.add_input('d_full', np.zeros(nFull), units='m')
        self.add_input('z_param', np.zeros(n_height), units='m')
        
        # extra mass
        self.add_input('mass', 0.0, units='kg')
        self.add_input('mI', np.zeros(6), units='kg*m**2')
        self.add_input('mrho', np.zeros(3), units='m')
        self.add_input('transition_piece_mass', 0.0, units='kg')
        self.add_input('gravity_foundation_mass', 0.0, units='kg')
        self.add_input('transition_piece_height', 0.0, units='m')
        self.add_input('foundation_height', 0.0, units='m')
        
        # point loads
        self.add_input('rna_F', np.zeros(3), units='N')
        self.add_input('rna_M', np.zeros(3), units='N*m')
        
        # Monopile handling
        self.add_input('k_monopile', np.zeros(6), units='N/m')
        
        # Material property discretization
        self.add_input('E', val=np.zeros(n_height-1), units='Pa')
        self.add_input('G', val=np.zeros(n_height-1), units='Pa')
        self.add_input('sigma_y', val=np.zeros(n_height-1), units='Pa')

        self.add_output('E_full', val=np.zeros(nFull-1), units='Pa')
        self.add_output('G_full', val=np.zeros(nFull-1), units='Pa')
        self.add_output('sigma_y_full', val=np.zeros(nFull-1), units='Pa')
        
        # spring reaction data.
        nK = 1
        self.add_output('kidx', np.zeros(nK, dtype=np.int_))
        self.add_output('kx', np.zeros(nK), units='N/m')
        self.add_output('ky', np.zeros(nK), units='N/m')
        self.add_output('kz', np.zeros(nK), units='N/m')
        self.add_output('ktx', np.zeros(nK), units='N/m')
        self.add_output('kty', np.zeros(nK), units='N/m')
        self.add_output('ktz', np.zeros(nK), units='N/m')
        
        # extra mass
        nMass = 3
        self.add_output('midx', np.zeros(nMass, dtype=np.int_))
        self.add_output('m', np.zeros(nMass), units='kg')
        self.add_output('mIxx', np.zeros(nMass), units='kg*m**2')
        self.add_output('mIyy', np.zeros(nMass), units='kg*m**2')
        self.add_output('mIzz', np.zeros(nMass), units='kg*m**2')
        self.add_output('mIxy', np.zeros(nMass), units='kg*m**2')
        self.add_output('mIxz', np.zeros(nMass), units='kg*m**2')
        self.add_output('mIyz', np.zeros(nMass), units='kg*m**2')
        self.add_output('mrhox', np.zeros(nMass), units='m')
        self.add_output('mrhoy', np.zeros(nMass), units='m')
        self.add_output('mrhoz', np.zeros(nMass), units='m')
        
        # point loads (if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also)
        nPL = 1
        self.add_output('plidx', np.zeros(nPL, dtype=np.int_))
        self.add_output('Fx', np.zeros(nPL), units='N')
        self.add_output('Fy', np.zeros(nPL), units='N')
        self.add_output('Fz', np.zeros(nPL), units='N')
        self.add_output('Mxx', np.zeros(nPL), units='N*m')
        self.add_output('Myy', np.zeros(nPL), units='N*m')
        self.add_output('Mzz', np.zeros(nPL), units='N*m')

        self.declare_partials('E_full', [], method='fd')
        self.declare_partials('Fx', ['rna_F'], method='fd')
        self.declare_partials('Fy', ['rna_F'], method='fd')
        self.declare_partials('Fz', ['rna_F'], method='fd')
        self.declare_partials('G_full', [], method='fd')
        self.declare_partials('Mxx', ['rna_M'], method='fd')
        self.declare_partials('Myy', ['rna_M'], method='fd')
        self.declare_partials('Mzz', ['rna_M'], method='fd')
        self.declare_partials('kidx', [], method='fd')
        self.declare_partials('ktx', [], method='fd')
        self.declare_partials('kty', [], method='fd')
        self.declare_partials('ktz', [], method='fd')
        self.declare_partials('kx', [], method='fd')
        self.declare_partials('ky', [], method='fd')
        self.declare_partials('kz', [], method='fd')
        self.declare_partials('m', ['gravity_foundation_mass', 'mass', 'transition_piece_mass'], method='fd')
        self.declare_partials('mIxx', ['gravity_foundation_mass', 'mI', 'transition_piece_mass'], method='fd')
        self.declare_partials('mIxy', ['mI'], method='fd')
        self.declare_partials('mIxz', ['mI'], method='fd')
        self.declare_partials('mIyy', ['gravity_foundation_mass', 'mI', 'transition_piece_mass'], method='fd')
        self.declare_partials('mIyz', ['mI'], method='fd')
        self.declare_partials('mIzz', ['gravity_foundation_mass', 'mI', 'transition_piece_mass'], method='fd')
        self.declare_partials('midx', [], method='fd')
        self.declare_partials('mrhox', ['mrho'], method='fd')
        self.declare_partials('mrhoy', ['mrho'], method='fd')
        self.declare_partials('mrhoz', ['mrho'], method='fd')
        self.declare_partials('plidx', [], method='fd')
        self.declare_partials('sigma_y_full', ['sigma_y'], method='fd')
        
    def compute(self, inputs, outputs):
        n_height = self.options['n_height']
        nFull   = get_nfull(n_height)
        d = inputs['d_full']
        z = inputs['z_full']
        
        # Prepare RNA, transition piece, and gravity foundation (if any applicable) for "extra node mass"
        itrans = find_nearest(z, inputs['transition_piece_height'])
        rtrans = 0.5*d[itrans]
        mtrans = inputs['transition_piece_mass']
        Itrans = mtrans*rtrans**2. * np.r_[0.5, 0.5, 1.0, np.zeros(3)] # shell
        rgrav  = 0.5*d[0]
        mgrav  = inputs['gravity_foundation_mass']
        Igrav  = mgrav*rgrav**2. * np.r_[0.25, 0.25, 0.5, np.zeros(3)] # disk
        # Note, need len()-1 because Frame3DD crashes if mass add at end
        outputs['midx']  = np.array([ nFull-1, itrans, 0 ], dtype=np.int_)
        outputs['m']     = np.array([ inputs['mass'],  mtrans,   mgrav ]).flatten()
        outputs['mIxx']  = np.array([ inputs['mI'][0], Itrans[0], Igrav[0] ]).flatten()
        outputs['mIyy']  = np.array([ inputs['mI'][1], Itrans[1], Igrav[1] ]).flatten()
        outputs['mIzz']  = np.array([ inputs['mI'][2], Itrans[2], Igrav[2] ]).flatten()
        outputs['mIxy']  = np.array([ inputs['mI'][3], Itrans[3], Igrav[3] ]).flatten()
        outputs['mIxz']  = np.array([ inputs['mI'][4], Itrans[4], Igrav[4] ]).flatten()
        outputs['mIyz']  = np.array([ inputs['mI'][5], Itrans[5], Igrav[5] ]).flatten()
        outputs['mrhox'] = np.array([ inputs['mrho'][0], 0.0, 0.0 ]).flatten()
        outputs['mrhoy'] = np.array([ inputs['mrho'][1], 0.0, 0.0 ]).flatten()
        outputs['mrhoz'] = np.array([ inputs['mrho'][2], 0.0, 0.0 ]).flatten()

        # Prepare point forces at RNA node
        outputs['plidx'] = np.array([ nFull-1 ], dtype=np.int_) # -1 b/c same reason as above
        outputs['Fx']    = np.array([ inputs['rna_F'][0] ]).flatten()
        outputs['Fy']    = np.array([ inputs['rna_F'][1] ]).flatten()
        outputs['Fz']    = np.array([ inputs['rna_F'][2] ]).flatten()
        outputs['Mxx']   = np.array([ inputs['rna_M'][0] ]).flatten()
        outputs['Myy']   = np.array([ inputs['rna_M'][1] ]).flatten()
        outputs['Mzz']   = np.array([ inputs['rna_M'][2] ]).flatten()

        # Prepare for reactions: rigid at tower base
        outputs['kidx'] = np.array([ 0 ], dtype=np.int_)
        if self.options['monopile']:
            kmono  = inputs['k_monopile']
            outputs['kx']   = np.array([ kmono[0] ])
            outputs['ky']   = np.array([ kmono[2] ])
            outputs['kz']   = np.array([ kmono[4] ])
            outputs['ktx']  = np.array([ kmono[1] ])
            outputs['kty']  = np.array([ kmono[3] ])
            outputs['ktz']  = np.array([ kmono[5] ])
        else:
            outputs['kx']   = np.array([ 1.e16 ])
            outputs['ky']   = np.array([ 1.e16 ])
            outputs['kz']   = np.array([ 1.e16 ])
            outputs['ktx']  = np.array([ 1.e16 ])
            outputs['kty']  = np.array([ 1.e16 ])
            outputs['ktz']  = np.array([ 1.e16 ])

        # Material property discretization
        z_param   = inputs['z_param']
        z_section = 0.5*(z[:-1] + z[1:])
        outputs['E_full']          = sectionalInterp(z_section, z_param, inputs['E'])
        outputs['G_full']          = sectionalInterp(z_section, z_param, inputs['G'])
        outputs['sigma_y_full']    = sectionalInterp(z_section, z_param, inputs['sigma_y'])

        
class TowerPostFrame(om.ExplicitComponent):
    """
    Postprocess results from Frame3DD.
    
    Parameters
    ----------
    z_full : numpy array[nFull]
        location along tower. start at bottom and go to top
    d_full : numpy array[nFull]
        effective tower diameter for section
    t_full : numpy array[nFull-1]
        effective shell thickness for section
    E_full : numpy array[nFull-1]
        modulus of elasticity
    sigma_y_full : numpy array[nFull-1]
        yield stress
    Fz : numpy array[nFull-1]
        Axial foce in vertical z-direction in cylinder structure.
    Mxx : numpy array[nFull-1]
        Moment about x-axis in cylinder structure.
    Myy : numpy array[nFull-1]
        Moment about y-axis in cylinder structure.
    axial_stress : numpy array[nFull-1]
        axial stress in tower elements
    shear_stress : numpy array[nFull-1]
        shear stress in tower elements
    hoop_stress : numpy array[nFull-1]
        hoop stress in tower elements
    top_deflection_in : float
        Deflection of tower top in yaw-aligned +x direction
    life : float
        fatigue life of tower
    freqs : numpy array[NFREQ]
        Natural frequencies of the structure
    x_mode_shapes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the x-direction
    y_mode_shapes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the x-direction
    
    Returns
    -------
    structural_frequencies : numpy array[NFREQ]
        First and second natural frequency
    fore_aft_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower fore-aft direction
        (without constant term)
    side_side_modes : numpy array[NFREQ2, 5]
        6-degree polynomial coefficients of mode shapes in the tower side-side direction
        (without constant term)
    top_deflection : float
        Deflection of tower top in yaw-aligned +x direction
    stress : numpy array[nFull-1]
        Von Mises stress utilization along tower at specified locations. Includes safety
        factor.
    shell_buckling : numpy array[nFull-1]
        Shell buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    global_buckling : numpy array[nFull-1]
        Global buckling constraint. Should be < 1 for feasibility. Includes safety
        factors
    turbine_F : numpy array[3]
        Total force on tower+rna
    turbine_M : numpy array[3]
        Total x-moment on tower+rna measured at base
    
    """
    def initialize(self):
        self.options.declare('n_height')
        self.options.declare('analysis_options')
        #self.options.declare('nDEL')

    def setup(self):
        n_height = self.options['n_height']
        nFull    = get_nfull(n_height)

        # effective geometry -- used for handbook methods to estimate hoop stress, buckling, fatigue
        self.add_input('z_full', np.zeros(nFull), units='m')
        self.add_input('d_full', np.zeros(nFull), units='m')
        self.add_input('t_full', np.zeros(nFull-1), units='m')
        
        # Material properties
        self.add_input('E_full', np.zeros(nFull-1), units='N/m**2')
        self.add_input('sigma_y_full', np.zeros(nFull-1), units='N/m**2')
        
        # Processed Frame3DD outputs
        self.add_input('Fz', np.zeros(nFull-1), units='N')
        self.add_input('Mxx', np.zeros(nFull-1), units='N*m')
        self.add_input('Myy', np.zeros(nFull-1), units='N*m')
        self.add_input('axial_stress', val=np.zeros(nFull-1), units='N/m**2')
        self.add_input('shear_stress', val=np.zeros(nFull-1), units='N/m**2')
        self.add_input('hoop_stress', val=np.zeros(nFull-1), units='N/m**2')
        self.add_input('top_deflection_in', 0.0, units='m')
        
        # safety factors
        #self.add_input('gamma_f', 1.35, desc='safety factor on loads')
        #self.add_input('gamma_m', 1.1, desc='safety factor on materials')
        #self.add_input('gamma_n', 1.0, desc='safety factor on consequence of failure')
        #self.add_input('gamma_b', 1.1, desc='buckling safety factor')
        #self.add_input('gamma_fatigue', 1.755, desc='total safety factor for fatigue')

        # fatigue parameters
        self.add_input('life', 20.0)
        #self.add_input('m_SN', 4, desc='slope of S/N curve')
        #self.add_input('DC', 80.0, desc='standard value of stress')
        #self.add_input('z_DEL', np.zeros(nDEL), units='m', desc='absolute z coordinates of corresponding fatigue parameters')
        #self.add_input('M_DEL', np.zeros(nDEL), desc='fatigue parameters at corresponding z coordinates')
        
        # Frequencies
        NFREQ2 = int(NFREQ/2)
        self.add_input('freqs', val=np.zeros(NFREQ), units='Hz')
        self.add_input('x_mode_shapes', val=np.zeros((NFREQ2, 5)))
        self.add_input('y_mode_shapes', val=np.zeros((NFREQ2, 5)))

        self.add_output('structural_frequencies', np.zeros(NFREQ), units='Hz')
        self.add_output('fore_aft_modes', np.zeros((NFREQ2, 5)))
        self.add_output('side_side_modes', np.zeros((NFREQ2, 5)))
        self.add_output('top_deflection', 0.0, units='m')
        self.add_output('stress', np.zeros(nFull-1))
        self.add_output('shell_buckling', np.zeros(nFull-1))
        self.add_output('global_buckling', np.zeros(nFull-1))
        #self.add_output('damage', np.zeros(nFull-1), desc='Fatigue damage at each tower section')
        self.add_output('turbine_F', val=np.zeros(3), units='N')
        self.add_output('turbine_M', val=np.zeros(3), units='N*m')

        self.declare_partials('global_buckling', ['Fz', 'Mxx', 'Myy', 'd_full', 'sigma_y_full', 't_full', 'z_full'], method='fd')
        self.declare_partials('shell_buckling', ['axial_stress', 'd_full', 'hoop_stress', 'shear_stress', 'sigma_y_full', 't_full'], method='fd')
        self.declare_partials('stress', ['axial_stress', 'hoop_stress', 'shear_stress', 'sigma_y_full'], method='fd')
        self.declare_partials('structural_frequencies', ['freqs'], method='fd')
        self.declare_partials('fore_aft_modes', ['x_mode_shapes'], method='fd')
        self.declare_partials('side_side_modes', ['y_mode_shapes'], method='fd')
        self.declare_partials('top_deflection', ['top_deflection_in'], method='fd')
        self.declare_partials('turbine_F', [], method='fd')
        self.declare_partials('turbine_M', [], method='fd')
        
    def compute(self, inputs, outputs):
        # Unpack some variables
        axial_stress = inputs['axial_stress']
        shear_stress = inputs['shear_stress']
        hoop_stress  = inputs['hoop_stress']
        sigma_y      = inputs['sigma_y_full']
        E            = inputs['E_full']
        t            = inputs['t_full']
        d,_          = nodal2sectional(inputs['d_full'])
        z_section,_  = nodal2sectional(inputs['z_full'])
        L_reinforced = self.options['analysis_options']['buckling_length'] * np.ones(axial_stress.shape)
        gamma_f      = self.options['analysis_options']['gamma_f']
        gamma_m      = self.options['analysis_options']['gamma_m']
        gamma_n      = self.options['analysis_options']['gamma_n']
        gamma_b      = self.options['analysis_options']['gamma_b']

        # Frequencies and mode shapes (with x^2 term first)
        outputs['structural_frequencies'] = inputs['freqs']
        outputs['fore_aft_modes']         = inputs['x_mode_shapes']
        outputs['side_side_modes']        = inputs['y_mode_shapes']

        # Tower top deflection
        outputs['top_deflection'] = inputs['top_deflection_in']
        
        # von mises stress
        outputs['stress'] = Util.vonMisesStressUtilization(axial_stress, hoop_stress, shear_stress, gamma_f*gamma_m*gamma_n, sigma_y)

        # shell buckling
        outputs['shell_buckling'] = Util.shellBucklingEurocode(d, t, axial_stress, hoop_stress,
                                                                shear_stress, L_reinforced, E, sigma_y, gamma_f, gamma_b)

        # global buckling
        tower_height = inputs['z_full'][-1] - inputs['z_full'][0]
        M = np.sqrt(inputs['Mxx']**2 + inputs['Myy']**2)
        outputs['global_buckling'] = Util.bucklingGL(d, t, inputs['Fz'], M, tower_height, E, sigma_y, gamma_f, gamma_b)

        # fatigue
        N_DEL = 365.0*24.0*3600.0*inputs['life'] * np.ones(len(t))
        #outputs['damage'] = np.zeros(N_DEL.shape)

        #if any(inputs['M_DEL']):
        #    M_DEL = np.interp(z_section, inputs['z_DEL'], inputs['M_DEL'])

        #    outputs['damage'] = Util.fatigue(M_DEL, N_DEL, d, inputs['t'], inputs['m_SN'],
        #                                      inputs['DC'], gamma_fatigue, stress_factor=1.0, weld_factor=True)

# -----------------
#  Assembly
# -----------------

class TowerLeanSE(om.Group):
    """
    This is a geometry preprocessing group.
    """
    
    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('topLevelFlag')
        
    def setup(self):
        toweropt = self.options['analysis_options']['tower']
        n_height = toweropt['n_height']
        monopile = toweropt['monopile']
        nFull    = get_nfull(n_height)
        
        # Independent variables that are only used in the user is calling TowerSE via python directly
        if self.options['topLevelFlag']:
            sharedIndeps = om.IndepVarComp()
            sharedIndeps.add_output('gravity_foundation_mass', 0.0, units='kg')
            sharedIndeps.add_output('transition_piece_mass', 0.0, units='kg')
            sharedIndeps.add_output('transition_piece_height', 0.0, units='m')
            sharedIndeps.add_output('suctionpile_depth', 0.0, units='m')
            sharedIndeps.add_output('suctionpile_depth_diam_ratio', 0.0)
            sharedIndeps.add_output('tower_outer_diameter', np.zeros(n_height), units='m')
            sharedIndeps.add_output('tower_section_height', np.zeros(n_height-1), units='m')
            sharedIndeps.add_output('tower_wall_thickness', np.zeros(n_height-1), units='m')
            sharedIndeps.add_output('outfitting_factor', np.zeros(n_height-1))
            sharedIndeps.add_output('foundation_height', 0.0, units='m')
            sharedIndeps.add_output('hub_height', 0.0, units='m')
            sharedIndeps.add_output('rho', np.zeros(n_height-1), units='kg/m**3')
            sharedIndeps.add_output('unit_cost', np.zeros(n_height-1), units='USD/kg')
            sharedIndeps.add_output('labor_cost_rate', 0.0, units='USD/min')
            sharedIndeps.add_output('painting_cost_rate', 0.0, units='USD/m**2')
            self.add_subsystem('sharedIndepsLean', sharedIndeps, promotes=['*'])
        else:
            # If using YAML for input, unpack to native variables
            n_height_tow = self.options['analysis_options']['tower']['n_height']
            n_height_mon = 0 if not monopile else self.options['analysis_options']['monopile']['n_height']
            n_height     = n_height_tow if n_height_mon==0 else n_height_tow + n_height_mon - 1 # Should have one overlapping point
            n_layers_mon = 0 if not monopile else self.options['analysis_options']['monopile']['n_layers']
            self.add_subsystem('yaml', DiscretizationYAML(n_height_tower=n_height_tow, n_height_monopile=n_height_mon,
                                                          n_layers_tower=toweropt['n_layers'], n_layers_monopile=n_layers_mon,
                                                          n_mat=self.options['analysis_options']['materials']['n_mat']),
                               promotes=['*'])
            
        # If doing fixed bottom monopile, we add an additional point for the pile (even for gravity foundations)
        self.add_subsystem('predisc', MonopileFoundation(monopile=monopile), promotes=['*'])
            
        # Promote all but foundation_height so that we can override
        self.add_subsystem('geometry', CylinderDiscretization(nPoints=n_height, nRefine=NREFINE), promotes=['z_param','z_full','d_full','t_full'])
        
        self.add_subsystem('tgeometry', TowerDiscretization(n_height=n_height), promotes=['*'])
        
        self.add_subsystem('cm', CylinderMass(nPoints=nFull), promotes=['z_full','d_full','t_full',
                                                                        'labor_cost_rate','painting_cost_rate'])
        self.add_subsystem('tm', TowerMass(n_height=n_height), promotes=['z_full',
                                                                   'tower_mass','tower_center_of_mass','tower_section_center_of_mass','tower_I_base',
                                                                   'tower_raw_cost','gravity_foundation_mass','foundation_height',
                                                                   'transition_piece_mass','transition_piece_height',
                                                                   'monopile_mass','monopile_cost','monopile_length'])
        self.add_subsystem('gc', Util.GeometricConstraints(nPoints=n_height), promotes=['min_d_to_t','max_taper','manufacturability','weldability','slope'])
        self.add_subsystem('turb', TurbineMass(), promotes=['turbine_mass','monopile_mass',
                                                            'tower_mass','tower_center_of_mass','tower_I_base',
                                                            'rna_mass', 'rna_cg', 'rna_I','hub_height'])
        
        # Connections for geometry and mass
        self.connect('z_start', 'geometry.foundation_height')
        self.connect('tower_section_height', 'geometry.section_height')
        self.connect('tower_outer_diameter', ['geometry.diameter', 'gc.d'])
        self.connect('tower_outer_diameter', 'base_diameter', src_indices=[0])
        self.connect('tower_wall_thickness', ['geometry.wall_thickness', 'gc.t'])

        self.connect('rho_full', 'cm.rho')
        self.connect('outfitting_full', 'cm.outfitting_factor')
        self.connect('unit_cost_full', 'cm.material_cost_rate')
        self.connect('cm.mass', 'tm.cylinder_mass')
        self.connect('cm.cost', 'tm.cylinder_cost')
        self.connect('cm.center_of_mass', 'tm.cylinder_center_of_mass')
        self.connect('cm.section_center_of_mass','tm.cylinder_section_center_of_mass')
        self.connect('cm.I_base','tm.cylinder_I_base')

        
class TowerSE(om.Group):
    """
    This is the main TowerSE group.
    """

    def initialize(self):
        self.options.declare('analysis_options')
        self.options.declare('topLevelFlag')
        
    def setup(self):
        toweropt = self.options['analysis_options']['tower']
        n_height = toweropt['n_height']
        nFull    = get_nfull(n_height)
        monopile = toweropt['monopile']
        nLC      = toweropt['nLC']
        wind     = toweropt['wind']
        frame3dd_opt = toweropt['frame3dd']
        topLevelFlag = self.options['topLevelFlag']
        
        # Independent variables that are only used in the user is calling TowerSE via python directly
        if topLevelFlag:
            sharedIndeps = om.IndepVarComp()
            sharedIndeps.add_output('rho_air', 1.225, units='kg/m**3')
            sharedIndeps.add_output('mu_air', 1.81206e-5, units='kg/m/s')
            sharedIndeps.add_output('shearExp', 0.0)
            sharedIndeps.add_output('wind_reference_height', 0.0, units='m')
            sharedIndeps.add_output('wind_z0', 0.0, units='m')
            sharedIndeps.add_output('beta_wind', 0.0, units='deg')
            sharedIndeps.add_output('rho_water', 1025.0, units='kg/m**3')
            sharedIndeps.add_output('mu_water', 8.9e-4, units='kg/m/s')
            sharedIndeps.add_output('beta_wave', 0.0, units='deg')
            sharedIndeps.add_output('hsig_wave', 0.0, units='m')
            sharedIndeps.add_output('Tsig_wave', 0.0, units='s')
            sharedIndeps.add_output('cd_usr', -1.)
            sharedIndeps.add_output('yaw', 0.0, units='deg')
            sharedIndeps.add_output('E', np.zeros(n_height-1), units='N/m**2')
            sharedIndeps.add_output('G', np.zeros(n_height-1), units='N/m**2')
            sharedIndeps.add_output('G_soil', 0.0, units='N/m**2')
            sharedIndeps.add_output('nu_soil', 0.0)
            sharedIndeps.add_output('sigma_y', np.zeros(n_height-1), units='N/m**2')
            sharedIndeps.add_output('rna_mass', 0.0, units='kg')
            sharedIndeps.add_output('rna_cg', np.zeros(3), units='m')
            sharedIndeps.add_output('rna_I', np.zeros(6), units='kg*m**2')
            sharedIndeps.add_output('life', 0.0)
            #sharedIndeps.add_output('m_SN', 0.0)
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
        else:
            n_height_tow = self.options['analysis_options']['tower']['n_height']
            n_height_mon = 0 if not monopile else self.options['analysis_options']['monopile']['n_height']
            n_height     = n_height_tow if n_height_mon==0 else n_height_tow + n_height_mon - 1 # Should have one overlapping point

        # Load baseline discretization
        self.add_subsystem('geom', TowerLeanSE(analysis_options=self.options['analysis_options'], topLevelFlag=topLevelFlag), promotes=['*'])
        self.add_subsystem('props', CylindricalShellProperties(nFull=nFull))
        self.add_subsystem('soil', TowerSoil())

        # Connections for geometry and mass
        self.connect('d_full', 'props.d')
        self.connect('t_full', 'props.t')
        if monopile:
            self.connect('d_full', 'soil.d0', src_indices=[0])
            self.connect('suctionpile_depth', 'soil.depth')
            self.connect('G_soil', 'soil.G')
            self.connect('nu_soil', 'soil.nu')
        
        # Add in all Components that drive load cases
        # Note multiple load cases have to be handled by replicating components and not groups/assemblies.
        # Replicating Groups replicates the IndepVarComps which doesn't play nicely in OpenMDAO
        for iLC in range(nLC):
            lc = '' if nLC==1 else str(iLC+1)
            
            if wind is None or wind.lower() in ['power', 'powerwind', '']:
                self.add_subsystem('wind'+lc, PowerWind(nPoints=nFull), promotes=['shearExp'])
            elif wind.lower() == 'logwind':
                self.add_subsystem('wind'+lc, LogWind(nPoints=nFull))
            else:
                raise ValueError('Unknown wind type, '+wind)

            self.add_subsystem('windLoads'+lc, CylinderWindDrag(nPoints=nFull), promotes=['cd_usr','rho_air',
                                                                                           'mu_air','beta_wind'])

            if monopile:
                self.add_subsystem('wave'+lc, LinearWaves(nPoints=nFull), promotes=['z_floor','rho_water',
                                                                                     'hsig_wave','Tsig_wave'])
                self.add_subsystem('waveLoads'+lc, CylinderWaveDrag(nPoints=nFull), promotes=['cm','cd_usr',
                                                                                               'rho_water','mu_water',
                                                                                               'beta_wave'])

            self.add_subsystem('distLoads'+lc, AeroHydroLoads(nPoints=nFull), promotes=['yaw'])

            self.add_subsystem('pre'+lc, TowerPreFrame(n_height=n_height, monopile=monopile), promotes=['transition_piece_mass',
                                                                                                        'transition_piece_height',
                                                                                                        'gravity_foundation_mass',
                                                                                                        'E','G','sigma_y','z_full','d_full'])
            self.add_subsystem('tower'+lc, CylinderFrame3DD(npts=nFull, nK=1, nMass=3, nPL=1,
                                                            frame3dd_opt=frame3dd_opt, buckling_length=toweropt['buckling_length']))
            self.add_subsystem('post'+lc, TowerPostFrame(n_height=n_height, analysis_options=toweropt), promotes=['life','z_full','d_full','t_full'])
            
            self.connect('z_full', ['wind'+lc+'.z', 'windLoads'+lc+'.z', 'distLoads'+lc+'.z', 'tower'+lc+'.z'])
            self.connect('d_full', ['windLoads'+lc+'.d', 'tower'+lc+'.d'])
            self.connect('t_full', 'tower'+lc+'.t')
            if monopile:
                self.connect('z_full', ['wave'+lc+'.z', 'waveLoads'+lc+'.z'])
                self.connect('d_full', 'waveLoads'+lc+'.d')

            if topLevelFlag:
                self.connect('rna_mass', 'pre'+lc+'.mass')
                self.connect('rna_cg', 'pre'+lc+'.mrho')
                self.connect('rna_I', 'pre'+lc+'.mI')

            self.connect('rho_full', 'tower'+lc+'.rho')
            self.connect('pre'+lc+'.E_full', ['tower'+lc+'.E', 'post'+lc+'.E_full'])
            self.connect('pre'+lc+'.G_full', 'tower'+lc+'.G')
            self.connect('pre'+lc+'.sigma_y_full', 'post'+lc+'.sigma_y_full')
            
            self.connect('pre'+lc+'.kidx', 'tower'+lc+'.kidx')
            self.connect('pre'+lc+'.kx', 'tower'+lc+'.kx')
            self.connect('pre'+lc+'.ky', 'tower'+lc+'.ky')
            self.connect('pre'+lc+'.kz', 'tower'+lc+'.kz')
            self.connect('pre'+lc+'.ktx', 'tower'+lc+'.ktx')
            self.connect('pre'+lc+'.kty', 'tower'+lc+'.kty')
            self.connect('pre'+lc+'.ktz', 'tower'+lc+'.ktz')
            self.connect('pre'+lc+'.midx', 'tower'+lc+'.midx')
            self.connect('pre'+lc+'.m', 'tower'+lc+'.m')
            self.connect('pre'+lc+'.mIxx', 'tower'+lc+'.mIxx')
            self.connect('pre'+lc+'.mIyy', 'tower'+lc+'.mIyy')
            self.connect('pre'+lc+'.mIzz', 'tower'+lc+'.mIzz')
            self.connect('pre'+lc+'.mIxy', 'tower'+lc+'.mIxy')
            self.connect('pre'+lc+'.mIxz', 'tower'+lc+'.mIxz')
            self.connect('pre'+lc+'.mIyz', 'tower'+lc+'.mIyz')
            self.connect('pre'+lc+'.mrhox', 'tower'+lc+'.mrhox')
            self.connect('pre'+lc+'.mrhoy', 'tower'+lc+'.mrhoy')
            self.connect('pre'+lc+'.mrhoz', 'tower'+lc+'.mrhoz')

            self.connect('pre'+lc+'.plidx', 'tower'+lc+'.plidx')
            self.connect('pre'+lc+'.Fx', 'tower'+lc+'.Fx')
            self.connect('pre'+lc+'.Fy', 'tower'+lc+'.Fy')
            self.connect('pre'+lc+'.Fz', 'tower'+lc+'.Fz')
            self.connect('pre'+lc+'.Mxx', 'tower'+lc+'.Mxx')
            self.connect('pre'+lc+'.Myy', 'tower'+lc+'.Myy')
            self.connect('pre'+lc+'.Mzz', 'tower'+lc+'.Mzz')
            self.connect('soil.k', 'pre'+lc+'.k_monopile')

            self.connect('tower'+lc+'.freqs', 'post'+lc+'.freqs')
            self.connect('tower'+lc+'.x_mode_shapes', 'post'+lc+'.x_mode_shapes')
            self.connect('tower'+lc+'.y_mode_shapes', 'post'+lc+'.y_mode_shapes')
            self.connect('tower'+lc+'.Fz_out', 'post'+lc+'.Fz')
            self.connect('tower'+lc+'.Mxx_out', 'post'+lc+'.Mxx')
            self.connect('tower'+lc+'.Myy_out', 'post'+lc+'.Myy')
            self.connect('tower'+lc+'.axial_stress', 'post'+lc+'.axial_stress')
            self.connect('tower'+lc+'.shear_stress', 'post'+lc+'.shear_stress')
            self.connect('tower'+lc+'.hoop_stress_euro', 'post'+lc+'.hoop_stress')
            self.connect('tower'+lc+'.top_deflection', 'post'+lc+'.top_deflection_in')
        
            # connections to wind, wave
            if topLevelFlag:
                self.connect('wind_reference_height', 'wind'+lc+'.zref')
                self.connect('wind_z0', 'wind'+lc+'.z0')
                if monopile:
                    self.connect('wind_z0', 'wave'+lc+'.z_surface')
                
            self.connect('wind'+lc+'.U', 'windLoads'+lc+'.U')
            if monopile:
                # connections to waveLoads1
                self.connect('foundation_height', 'z_floor')
                self.connect('wave'+lc+'.U', 'waveLoads'+lc+'.U')
                self.connect('wave'+lc+'.A', 'waveLoads'+lc+'.A')
                self.connect('wave'+lc+'.p', 'waveLoads'+lc+'.p')

            # connections to distLoads1
            self.connect('windLoads'+lc+'.windLoads_Px', 'distLoads'+lc+'.windLoads_Px')
            self.connect('windLoads'+lc+'.windLoads_Py', 'distLoads'+lc+'.windLoads_Py')
            self.connect('windLoads'+lc+'.windLoads_Pz', 'distLoads'+lc+'.windLoads_Pz')
            self.connect('windLoads'+lc+'.windLoads_qdyn', 'distLoads'+lc+'.windLoads_qdyn')
            self.connect('windLoads'+lc+'.windLoads_beta', 'distLoads'+lc+'.windLoads_beta')
            #self.connect('windLoads'+lc+'.windLoads_Px0', 'distLoads'+lc+'.windLoads_Px0')
            #self.connect('windLoads'+lc+'.windLoads_Py0', 'distLoads'+lc+'.windLoads_Py0')
            #self.connect('windLoads'+lc+'.windLoads_Pz0', 'distLoads'+lc+'.windLoads_Pz0')
            #self.connect('windLoads'+lc+'.windLoads_qdyn0', 'distLoads'+lc+'.windLoads_qdyn0')
            #self.connect('windLoads'+lc+'.windLoads_beta0', 'distLoads'+lc+'.windLoads_beta0')
            self.connect('windLoads'+lc+'.windLoads_z', 'distLoads'+lc+'.windLoads_z')
            self.connect('windLoads'+lc+'.windLoads_d', 'distLoads'+lc+'.windLoads_d')

            if monopile:
                self.connect('waveLoads'+lc+'.waveLoads_Px', 'distLoads'+lc+'.waveLoads_Px')
                self.connect('waveLoads'+lc+'.waveLoads_Py', 'distLoads'+lc+'.waveLoads_Py')
                self.connect('waveLoads'+lc+'.waveLoads_Pz', 'distLoads'+lc+'.waveLoads_Pz')
                self.connect('waveLoads'+lc+'.waveLoads_pt', 'distLoads'+lc+'.waveLoads_qdyn')
                self.connect('waveLoads'+lc+'.waveLoads_beta', 'distLoads'+lc+'.waveLoads_beta')
                #self.connect('waveLoads'+lc+'.waveLoads_Px0', 'distLoads'+lc+'.waveLoads_Px0')
                #self.connect('waveLoads'+lc+'.waveLoads_Py0', 'distLoads'+lc+'.waveLoads_Py0')
                #self.connect('waveLoads'+lc+'.waveLoads_Pz0', 'distLoads'+lc+'.waveLoads_Pz0')
                #self.connect('waveLoads'+lc+'.waveLoads_qdyn0', 'distLoads'+lc+'.waveLoads_qdyn0')
                #self.connect('waveLoads'+lc+'.waveLoads_beta0', 'distLoads'+lc+'.waveLoads_beta0')
                self.connect('waveLoads'+lc+'.waveLoads_z', 'distLoads'+lc+'.waveLoads_z')
                self.connect('waveLoads'+lc+'.waveLoads_d', 'distLoads'+lc+'.waveLoads_d')

            self.connect('props.Az', 'tower'+lc+'.Az')
            self.connect('props.Asx', 'tower'+lc+'.Asx')
            self.connect('props.Asy', 'tower'+lc+'.Asy')
            self.connect('props.Jz', 'tower'+lc+'.Jz')
            self.connect('props.Ixx', 'tower'+lc+'.Ixx')
            self.connect('props.Iyy', 'tower'+lc+'.Iyy')

            self.connect('distLoads'+lc+'.Px',   'tower'+lc+'.Px')
            self.connect('distLoads'+lc+'.Py',   'tower'+lc+'.Py')
            self.connect('distLoads'+lc+'.Pz',   'tower'+lc+'.Pz')
            self.connect('distLoads'+lc+'.qdyn', 'tower'+lc+'.qdyn')

        
if __name__ == '__main__':
    # --- tower setup ------
    from wisdem.commonse.environment import PowerWind
    from wisdem.commonse.environment import LogWind
    
    plot = False
    opt  = False

    # --- geometry ----
    n_control_points = 3
    h_param = np.diff(np.linspace(0.0, 87.6, n_control_points))
    d_param = np.linspace(6.0, 3.87, n_control_points)
    t_param = 1.3*np.linspace(0.025, 0.021, n_control_points-1)
    z_foundation = 0.0
    theta_stress = 0.0
    yaw = 0.0
    Koutfitting = 1.07 * np.ones(n_control_points - 1)

    # --- material props ---
    E = 210e9 * np.ones(n_control_points - 1)
    G = 80.8e9 * np.ones(n_control_points - 1)
    rho = 8500.0 * np.ones(n_control_points - 1)
    sigma_y = 450.0e6 * np.ones(n_control_points - 1)

    # --- extra mass ----
    m = np.array([285598.8])
    mIxx = 1.14930678e+08
    mIyy = 2.20354030e+07
    mIzz = 1.87597425e+07
    mIxy = 0.0
    mIxz = 5.03710467e+05
    mIyz = 0.0
    mI = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
    mrho = np.array([-1.13197635, 0.0, 0.50875268])
    # -----------

    # --- wind ---
    wind_zref = 90.0
    wind_z0 = 0.0
    shearExp = 0.2
    cd_usr = -1.
    # ---------------

    # --- wave ---
    hmax = 0.0
    T = 1.0
    cm = 1.0
    suction_depth = 0.0
    soilG = 140e6
    soilnu = 0.4
    # ---------------

    # --- costs ---
    material_cost = 5.0
    labor_cost    = 100.0/60.0
    painting_cost = 30.0
    # ---------------
    
    # two load cases.  TODO: use a case iterator
    
    # # --- loading case 1: max Thrust ---
    wind_Uref1 = 11.73732
    Fx1 = 1284744.19620519
    Fy1 = 0.
    Fz1 = -2914124.84400512 + m*gravity
    Mxx1 = 3963732.76208099
    Myy1 = -2275104.79420872
    Mzz1 = -346781.68192839
    # # ---------------

    # # --- loading case 2: max wind speed ---
    wind_Uref2 = 70.0
    Fx2 = 930198.60063279
    Fy2 = 0.
    Fz2 = -2883106.12368949 + m*gravity
    Mxx2 = -1683669.22411597
    Myy2 = -2522475.34625363
    Mzz2 = 147301.97023764
    # # ---------------

    # Store analysis options
    analysis_options = {}
    analysis_options['tower'] = {}
    analysis_options['tower']['buckling_length'] = 30.0
    analysis_options['tower']['monopile'] = False

    # --- safety factors ---
    analysis_options['tower']['gamma_f'] = 1.35
    analysis_options['tower']['gamma_m'] = 1.3
    analysis_options['tower']['gamma_n'] = 1.0
    analysis_options['tower']['gamma_b'] = 1.1
    # ---------------

    # --- fatigue ---
    #z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    #M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    #nDEL = len(z_DEL)
    analysis_options['tower']['gamma_fatigue'] = 1.35*1.3*1.0
    life = 20.0
    #m_SN = 4
    # ---------------
    
    # -----Frame3DD------
    analysis_options['tower']['frame3dd']            = {}
    analysis_options['tower']['frame3dd']['DC']      = 80.0
    analysis_options['tower']['frame3dd']['shear']   = True
    analysis_options['tower']['frame3dd']['geom']    = True
    analysis_options['tower']['frame3dd']['dx']      = 5.0
    #analysis_options['tower']['frame3dd']['nM']      = 2
    analysis_options['tower']['frame3dd']['Mmethod'] = 1
    analysis_options['tower']['frame3dd']['lump']    = 0
    analysis_options['tower']['frame3dd']['tol']     = 1e-9
    analysis_options['tower']['frame3dd']['shift']   = 0.0
    analysis_options['tower']['frame3dd']['add_gravity'] = True
    # ---------------

    # --- constraints ---
    min_d_to_t   = 120.0
    max_taper    = 0.2
    # ---------------

    # # V_max = 80.0  # tip speed
    # # D = 126.0
    # # .freq1p = V_max / (D/2) / (2*pi)  # convert to Hz

    analysis_options['tower']['n_height'] = len(d_param)
    analysis_options['tower']['wind'] = 'PowerWind'
    analysis_options['tower']['nLC'] = 2
    
    prob = om.Problem()
    prob.model = TowerSE(analysis_options=analysis_options, topLevelFlag=True)
    
    if opt:
        prob.driver = om.pyOptSparseDriver() #om.ScipyOptimizeDriver() # 
        prob.driver.options['optimizer'] = 'SNOPT' #'SLSQP' #'CONMIN'
        # prob.driver.opt_settings['Iterations limit'] = 10
        prob.driver.opt_settings['Verify level'] = -1
        # prob.driver.declare_coloring()

        # --- Objective ---
        prob.model.add_objective('tower_mass', scaler=1e-6)
        # ----------------------

        prob.model.add_design_var('tower_outer_diameter', lower=3.87, upper=10.0)
        prob.model.add_design_var('tower_wall_thickness', lower=4e-3, upper=2e-1)

        # --- Constraints ---
        #prob.model.add_constraint('height_constraint',    lower=-1e-2,upper=1.e-2)
        prob.model.add_constraint('post1.stress',          upper=1.0)
        prob.model.add_constraint('post1.global_buckling', upper=1.0)
        prob.model.add_constraint('post1.shell_buckling',  upper=1.0)
        prob.model.add_constraint('post2.stress',          upper=1.0)
        prob.model.add_constraint('post2.global_buckling', upper=1.0)
        prob.model.add_constraint('post2.shell_buckling',  upper=1.0)
        prob.model.add_constraint('weldability',          upper=0.0)
        prob.model.add_constraint('manufacturability',    lower=0.0)
        prob.model.add_constraint('slope',                upper=1.0)
        prob.model.add_constraint('tower1.f1',             lower=0.13, upper=0.40)
        prob.model.add_constraint('tower2.f1',             lower=0.13, upper=0.40)
        # ----------------------
    
    prob.setup()

    if analysis_options['tower']['wind'] == 'PowerWind':
        prob['shearExp'] = shearExp

    # assign values to params

    # --- geometry ----
    prob['hub_height'] = h_param.sum()
    prob['foundation_height'] = 0.0
    prob['tower_section_height'] = h_param
    prob['tower_outer_diameter'] = d_param
    prob['tower_wall_thickness'] = t_param
    prob['outfitting_factor'] = Koutfitting
    prob['yaw'] = yaw
    prob['suctionpile_depth'] = suction_depth
    prob['suctionpile_depth_diam_ratio'] = 3.25
    prob['G_soil'] = soilG
    prob['nu_soil'] = soilnu
    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['rho'] = rho
    prob['sigma_y'] = sigma_y

    # --- extra mass ----
    prob['rna_mass'] = m
    prob['rna_I'] = mI
    prob['rna_cg'] = mrho
    # -----------

    # --- costs ---
    prob['unit_cost'] = material_cost
    prob['labor_cost_rate']    = labor_cost
    prob['painting_cost_rate'] = painting_cost
    # -----------
    
    # --- wind & wave ---
    prob['wind_reference_height'] = wind_zref
    prob['wind_z0'] = wind_z0
    prob['cd_usr'] = cd_usr
    prob['rho_air'] = 1.225
    prob['mu_air'] = 1.7934e-5
    prob['rho_water'] = 1025.0
    prob['mu_water'] = 1.3351e-3
    prob['beta_wind'] = prob['beta_wave'] = 0.0
    prob['hsig_wave'] = hmax
    prob['Tsig_wave'] = T
    #prob['waveLoads1.U0'] = prob['waveLoads1.A0'] = prob['waveLoads1.beta0'] = prob['waveLoads2.U0'] = prob['waveLoads2.A0'] = prob['waveLoads2.beta0'] = 0.0
    
    # --- fatigue ---
    #prob['tower_z_DEL'] = z_DEL
    #prob['tower_M_DEL'] = M_DEL
    prob['life'] = life
    #prob['m_SN'] = m_SN
    # ---------------

    # --- constraints ---
    prob['min_d_to_t'] = min_d_to_t
    prob['max_taper'] = max_taper
    # ---------------


    # # --- loading case 1: max Thrust ---
    prob['wind1.Uref'] = wind_Uref1

    prob['pre1.rna_F'] = np.array([Fx1, Fy1, Fz1])
    prob['pre1.rna_M'] = np.array([Mxx1, Myy1, Mzz1])
    # # ---------------


    # # --- loading case 2: max Wind Speed ---
    prob['wind2.Uref'] = wind_Uref2

    prob['pre2.rna_F'] = np.array([Fx2, Fy2, Fz2])
    prob['pre2.rna_M' ] = np.array([Mxx2, Myy2, Mzz2])

    # # --- run ---
    # prob.run_model()
    # checks = prob.check_partials(compact_print=True)
    
    prob.model.approx_totals()
    # prob.compute_totals()
    
    prob.run_driver()
    
    z,_ = nodal2sectional(prob['z_full'])

    print('zs=', z)
    print('ds=', prob['d_full'])
    print('ts=', prob['t_full'])
    print('mass (kg) =', prob['tower_mass'])
    print('cg (m) =', prob['tower_center_of_mass'])
    print('weldability =', prob['weldability'])
    print('manufacturability =', prob['manufacturability'])
    print('\nwind: ', prob['wind1.Uref'])
    print('freq (Hz) =', prob['post1.structural_frequencies'])
    print('Fore-aft mode shapes:', prob['post1.fore_aft_modes'])
    print('Side-side mode shapes:', prob['post1.side_side_modes'])
    print('top_deflection1 (m) =', prob['post1.top_deflection'])
    print('stress1 =', prob['post1.stress'])
    print('GL buckling =', prob['post1.global_buckling'])
    print('Shell buckling =', prob['post1.shell_buckling'])
    #print('damage =', prob['post1.damage'])
    print('\nwind: ', prob['wind2.Uref'])
    print('freq (Hz) =', prob['post2.structural_frequencies'])
    print('Fore-aft mode shapes:', prob['post2.fore_aft_modes'])
    print('Side-side mode shapes:', prob['post2.side_side_modes'])
    print('top_deflection2 (m) =', prob['post2.top_deflection'])
    print('stress2 =', prob['post2.stress'])
    print('GL buckling =', prob['post2.global_buckling'])
    print('Shell buckling =', prob['post2.shell_buckling'])
    #print('damage =', prob['post2.damage'])


    stress1 = np.copy( prob['post1.stress'] )
    shellBuckle1 = np.copy( prob['post1.shell_buckling'] )
    globalBuckle1 = np.copy( prob['post1.global_buckling'] )
    #damage1 = np.copy( prob['post1.damage'] )

    stress2 = prob['post2.stress']
    shellBuckle2 = prob['post2.shell_buckling']
    globalBuckle2 = prob['post2.global_buckling']
    #damage2 = prob['post2.damage']

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.0, 3.5))
        plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        plt.plot(stress1, z, label='stress 1')
        plt.plot(stress2, z, label='stress 2')
        plt.plot(shellBuckle1, z, label='shell buckling 1')
        plt.plot(shellBuckle2, z, label='shell buckling 2')
        plt.plot(globalBuckle1, z, label='global buckling 1')
        plt.plot(globalBuckle2, z, label='global buckling 2')
        #plt.plot(damage1, z, label='damage 1')
        #plt.plot(damage2, z, label='damage 2')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
        plt.xlabel('utilization')
        plt.ylabel('height along tower (m)')

        #plt.figure(2)
        #plt.plot(prob['d_full']/2.+max(prob['d_full']), z, 'ok')
        #plt.plot(prob['d_full']/-2.+max(prob['d_full']), z, 'ok')

        #fig = plt.figure(3)
        #ax1 = fig.add_subplot(121)
        #ax2 = fig.add_subplot(122)

        #ax1.plot(prob['wind1.U'], z)
        #ax2.plot(prob['wind2.U'], z)
        #plt.tight_layout()
        plt.show()

        print(prob['tower1.base_F'])
        print(prob['tower1.base_M'])
        print(prob['tower2.base_F'])
        print(prob['tower2.base_M'])
    # ------------





    
    
