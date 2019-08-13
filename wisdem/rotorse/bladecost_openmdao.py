import numpy as np
from openmdao.api import ExplicitComponent
from wisdem.rotorse.bladecostse import blade_cost_model


# Class to initiate the blade cost model
class blade_cost_model_mdao(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
        self.options.declare('name',             default='')
        self.options.declare('verbosity',        defualt=False)
        self.options.declare('tex_table',        default=False)
        self.options.declare('generate_plots',   default=False)
        self.options.declare('show_plots',       default=False)
        self.options.declare('show_warnings',    default=False)
        self.options.declare('discrete',         default=False)
                             
    def setup(self):

        NPTS = self.options['NPTS']
        
        # These parameters will come from outside
        self.add_discrete_input('materials',     val=np.zeros(NPTS), desc='material properties of composite materials')
        self.add_discrete_input('upperCS',       val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for upper surface')
        self.add_discrete_input('lowerCS',       val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for lower surface')
        self.add_discrete_input('websCS',        val=np.zeros(NPTS), desc='list of CompositeSection objections defining the properties for shear webs')
        self.add_discrete_input('profile',       val=np.zeros(NPTS), desc='list of CompositeSection profiles')
        
        self.add_input('Rtip',          val=0.0,            units='m', desc='rotor radius')
        self.add_input('Rhub',          val=0.0,            units='m', desc='hub radius')
        self.add_input('bladeLength',   val=0.0,            units='m', desc='blade length')
        self.add_input('r_pts',         val=np.zeros(NPTS), units='m', desc='blade radial locations, expressed in the rotor system')
        self.add_input('chord',         val=np.zeros(NPTS), desc='Chord distribution')
        self.add_input('le_location',   val=np.zeros(NPTS), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')

        # outputs
        self.add_output('total_blade_cost', val=0.0, units='USD', desc='total blade cost')
        self.add_output('total_blade_mass', val=0.0, units='USD', desc='total blade cost')

    def compute(self, inputs, outputs):
        bcm             = blade_cost_model(options=self.options)
        bcm.name        = self.options['ref_name']
        bcm.materials   = inputs['materials']
        bcm.upperCS     = inputs['upperCS']
        bcm.lowerCS     = inputs['lowerCS']
        bcm.websCS      = inputs['websCS']
        bcm.profile     = inputs['profile']
        bcm.chord       = inputs['chord']
                
        bcm.r           = (inputs['r_pts'] - inputs['Rhub'])/(inputs['Rtip'] - inputs['Rhub']) * float(inputs['bladeLength'])
        bcm.bladeLength = float(inputs['bladeLength'])
        
        bcm.le_location              = inputs['le_location']
        blade_cost, blade_mass       = bcm.execute_blade_cost_model()
        
        outputs['total_blade_cost'] = blade_cost
        outputs['total_blade_mass'] = blade_mass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
