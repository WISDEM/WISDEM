import numpy as np
from openmdao.api import Component, Group, ParallelGroup
from airfoilprep import Polar, Airfoil


class AirfoilPreppyPolarExtrapolator(Component):

    def __init__(self, config, sdim, nalpha):

        super(AirfoilPreppyPolarExtrapolator, self).__init__()

        # HAWC2 output requires:
        self.blend_var = config['AirfoilPrep']['blend_var']
        self.naero_coeffs = len(self.blend_var)

        # TODO: predict number of aoa for each airfoil or use same for all afs
        '''
        self.naoa = config['AirfoilPrep']['naoa']
        '''

        self.sdim = sdim
        self.nsec = sdim[1]

        self.nalpha = nalpha
        self.res = config['AirfoilPrep']['res']
        self.nre = len(self.res)

        # airfoil calculation method, e.g. clean, rough, 3D correction method
        self.nmet = len(config['AirfoilPrep']['analysis_methods'])

        self._init_params()
        self._init_unknowns()

        self.useCM = True
        # set standard values
        # maximum drag coefficient
        self.cdmax = 0.0
        # minimum drag coefficient. used to prevent negative values that can
        # sometimes occur with this extrapolation method
        self.cdmin = 0.001
        # number of points to add in each segment of Viterna method
        self.nalpha = 15

        self.plot_polars = False

        for k, w in config['AirfoilPrep'].iteritems():
            try:
                setattr(self, k, w)
            except:
                pass

    def _init_params(self):

        self.add_param('cs_polars', np.zeros(  # nsec, alpha, cl ,cd, cm
            (self.nalpha, 4, self.nsec, self.nre, self.nmet)))
        self.add_param(
            'n_cs_alpha', np.zeros((self.nsec, self.nre, self.nmet)).astype(int))
        self.add_param('cs_polars_tc', np.zeros((self.nsec)))

        self.add_param('AR', 0., desc='aspect ratio = (rotor radius / chord_75% radius)\
            if provided, cdmax is computed from AR')

        self.add_param('rotor_diameter', 0.)
        self.add_param('blade_length', 0.)
        self.add_param('s_st', np.zeros(self.nsec))
        self.add_param('chord_st', np.zeros(self.nsec))

    def _init_unknowns(self):
        # TODO: output as HAWC2/FAST format
        self.add_output('airfoildata:blend_var', np.zeros(self.naero_coeffs))
        '''
        for i in range(self.naero_coeffs):
            self.add_output('airfoildata:aoa%02d' % i, np.zeros(self.naoa))
            self.add_output('airfoildata:cl%02d' % i, np.zeros(self.naoa))
            self.add_output('airfoildata:cd%02d' % i, np.zeros(self.naoa))
            self.add_output('airfoildata:cm%02d' % i, np.zeros(self.naoa))
        '''
        pass

    def solve_nonlinear(self, params, unknowns, resids):

        # determine aspect ratio = (rotor radius / chord_75% radius)\
        #    if provided, cdmax is computed from AR'
        bl = params['blade_length']
        dr = params['rotor_diameter']
        hr = 0.5 * dr - bl
        rotor_radius = 0.5 * dr
        chord = params['chord_st'] * bl
        s = params['s_st']
        r = (s * bl + hr) / rotor_radius
        chord_75 = np.interp(0.75, r, chord)
        AR = rotor_radius / chord_75

        # write aerodyn files
        af_name_base = 'cs_'
        af_name_suffix = '_aerodyn'
        tcs = params['cs_polars_tc']

        n_cs_alpha = params['n_cs_alpha']
        pol = params['cs_polars']
        nmet = 0
        # TODO: blend polars determined with different methods
        for i, tc in enumerate(self.blend_var):
            af_name = af_name_base + \
                '%03d_%04d' % (i, tc * 1000) + af_name_suffix
            re_polars = []
            for nre, re in enumerate(self.res):
                # create polar object
                p = Polar(re,
                          pol[:n_cs_alpha[i, nre, nmet], 0, i, nre, nmet],
                          pol[:n_cs_alpha[i, nre, nmet], 1, i, nre, nmet],
                          pol[:n_cs_alpha[i, nre, nmet], 2, i, nre, nmet],
                          pol[:n_cs_alpha[i, nre, nmet], 3, i, nre, nmet])

                # extrapolate polar
                if tc < self.tc_max:
                    p_extrap = p.extrapolate(self.cdmax,
                                             AR,
                                             self.cdmin,
                                             self.nalpha)
                else:
                    p_extrap = p.extrapolate_as_cylinder()
                p_extrap.useCM = self.useCM
                re_polars.append(p_extrap)

                # TODO: output as HAWC2/FAST format
                # See if HAWC can take several af tables with different Re
                # numbers
                '''
                unknowns['airfoildata:aoa%02d' % i] = p_extrap.alpha
                unknowns['airfoildata:cl%02d' % i] = p_extrap.cl
                unknowns['airfoildata:cd%02d' % i] = p_extrap.cd
                unknowns['airfoildata:cm%02d' % i] = p_extrap.cm
                '''

            # create airfoil object
            af = Airfoil(re_polars)
            af.interpToCommonAlpha()
            af.writeToAerodynFile(af_name + '.dat')

            if self.plot_polars:
                figs = af.plot(single_figure=True)
                titles = ['cl', 'cd', 'cm']
                for (fig, title) in zip(figs, titles):
                    fig.savefig(af_name + '_' + title + '.png', dpi=400)
                    fig.savefig(af_name + '_' + title + '.pdf')

        self._get_unknowns(unknowns)

    def _get_unknowns(self, unknowns):
        unknowns['airfoildata:blend_var'] = self.blend_var
