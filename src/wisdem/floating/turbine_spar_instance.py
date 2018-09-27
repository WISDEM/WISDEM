import numpy as np

from wisdem.floating.floating_turbine_instance import FloatingTurbineInstance, NSECTIONS, NPTS, vecOption
import offshorebos.wind_obos as wind_obos


class TurbineSparInstance(FloatingTurbineInstance):
    def __init__(self, refStr):
        super(TurbineSparInstance, self).__init__(refStr)

        # Parameters beyond those in superclass
        self.params['number_of_offset_columns'] = 0
        self.params['cross_attachment_pontoons_int'] = 0
        self.params['lower_attachment_pontoons_int'] = 0
        self.params['upper_attachment_pontoons_int'] = 0
        self.params['lower_ring_pontoons_int'] = 0
        self.params['upper_ring_pontoons_int'] = 0
        self.params['outer_cross_pontoons_int'] = 0
 
        # Typically design (OC3)
        self.params['main_freeboard'] = 10.0
        self.params['fairlead_location'] = 0.75862 # Want 5m
        self.set_length_main(130.0)
        self.params['main_section_height'] = np.array([36.0, 36.0, 36.0, 8.0, 14.0])
        self.params['main_outer_diameter'] = 2*np.array([4.7, 4.7, 4.7, 4.7, 3.25, 3.25])
        self.params['main_wall_thickness'] = 0.05
        self.params['fairlead_offset_from_shell'] = 5.2-4.7
        self.params['main_permanent_ballast_height'] = 10.0
        
        # OC3
        self.params['water_depth'] = 320.0
        self.params['wave_height'] = 10.8
        self.params['wave_period'] = 9.8
        self.params['wind_reference_speed'] = 11.0
        self.params['wind_reference_height'] = 119.0
        self.params['shearExp'] = 0.11
        self.params['morison_mass_coefficient'] = 2.0

        self.params['number_of_mooring_connections']        = 3
        self.params['mooring_lines_per_connection']         = 1
        self.params['mooring_line_length'] = 902.2
        self.params['anchor_radius'] = 853.87
        self.params['mooring_diameter'] = 0.09
        
        self.params['substructure'] =                 wind_obos.Substructure.SPAR
        self.params['anchor'] =                       wind_obos.Anchor.DRAGEMBEDMENT
        self.params['turbInstallMethod'] =            wind_obos.TurbineInstall.INDIVIDUAL
        self.params['towerInstallMethod'] =           wind_obos.TowerInstall.ONEPIECE
        self.params['installStrategy'] =              wind_obos.InstallStrategy.PRIMARYVESSEL

        # Change scalars to vectors where needed
        self.check_vectors()

        
    def get_constraints(self):
        conlist = super(TurbineSparInstance, self).get_constraints()

        poplist = []
        for k in range(len(conlist)):
            if ( (conlist[k][0].find('aux') >= 0) or
                 (conlist[k][0].find('pontoon') >= 0) or
                 (conlist[k][0].find('main_connection_ratio') >= 0) ):
                poplist.append(k)

        poplist.reverse()
        for k in poplist: conlist.pop(k)

        return conlist

    
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        self.draw_mooring(fig, self.prob['mooring_plot_matrix'])

        self.draw_column(fig, [0.0, 0.0], self.params['main_freeboard'], self.params['main_section_height'],
                           0.5*self.params['main_outer_diameter'], self.params['main_stiffener_spacing'])

        zcut = 1.0 + self.params['main_freeboard']
        self.draw_pontoons(fig, self.prob['plot_matrix'], 0.5*self.params['fairlead_support_outer_diameter'], zcut)
        
        self.draw_ballast(fig, [0.0, 0.0], self.params['main_freeboard'], self.params['main_section_height'],
                          0.5*self.params['main_outer_diameter']-self.params['main_wall_thickness'],
                          self.params['main_permanent_ballast_height'], self.prob['variable_ballast_height'])
        
        self.draw_column(fig, [0.0, 0.0], self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

        if self.prob['main.ballast_heave_box_mass'] > 0.0:
            self.draw_ballast_heave_box(fig, [0.0, 0.0], self.params['main_freeboard'],
                                        self.params['main_section_height'],
                                        self.params['main_ballast_heave_box_location'],
                                        0.5*self.params['main_ballast_heave_box_diameter'],
                                        self.params['main_ballast_heave_box_height'])

        self.draw_rna(fig)
        
        self.set_figure(fig, fname)
        

def deriv_check():
   # ----------------
    myspar = TurbineSparInstance()
    myspar.evaluate('psqp')
    f = open('deriv_total_spar.dat','w')
    out = myspar.prob.check_total_derivatives(f)#, compact_print=True)
    f.close()
    tol = 1e-4
    for comp in out.keys():
        if ( (out[comp]['rel error'][0] > tol) and (out[comp]['abs error'][0] > tol) ):
            print comp, out[comp]['magnitude'][0], out[comp]['rel error'][0], out[comp]['abs error'][0]

    
if __name__ == '__main__':
    #deriv_check()
    pass
