
from openmdao.api import ExplicitComponent

class TurbineClass(ExplicitComponent):
    def setup(self):
        # parameters
        self.add_discrete_input('turbine_class', val='I', desc='IEC turbine class')
        self.add_input('V_mean_overwrite', val=0., desc='overwrite value for mean velocity for using user defined CDFs')

        # outputs should be constant
        self.add_output('V_mean', shape=1, units='m/s', desc='IEC mean wind speed for Rayleigh distribution')
        self.add_output('V_extreme1', shape=1, units='m/s', desc='IEC extreme wind speed at hub height for a 1-year retunr period')
        self.add_output('V_extreme50', shape=1, units='m/s', desc='IEC extreme wind speed at hub height for a 50-year retunr period')
        # self.add_output('V_extreme_full', shape=2, units='m/s', desc='IEC extreme wind speed at hub height')
        
        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        turbine_class = discrete_inputs['turbine_class'].upper()

        if turbine_class == 'I':
            Vref = 50.0
        elif turbine_class == 'II':
            Vref = 42.5
        elif turbine_class == 'III':
            Vref = 37.5
        elif turbine_class == 'IV':
            Vref = 30.0
        else:
            raise ValueError('turbine_class input must be I/II/III/IV')

        if inputs['V_mean_overwrite'] == 0.:
            outputs['V_mean'] = 0.2*Vref
        else:
            outputs['V_mean'] = inputs['V_mean_overwrite']
        outputs['V_extreme1'] = 0.8*Vref
        outputs['V_extreme50'] = 1.4*Vref