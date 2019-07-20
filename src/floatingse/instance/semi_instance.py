from floating_instance import FloatingInstance
from commonse.utilities import sectional2nodal
import numpy as np
        
class SemiInstance(FloatingInstance):
    def __init__(self):
        super(SemiInstance, self).__init__()

        # Change scalars to vectors where needed
        self.check_vectors()

        
    def visualize(self, fname=None):
        fig = self.init_figure()

        self.draw_ocean(fig)

        mooringMat = self.prob['mooring_plot_matrix']
        self.draw_mooring(fig, mooringMat)

        pontoonMat = self.prob['plot_matrix']
        zcut = 1.0 + np.maximum( self.params['main_freeboard'], self.params['offset_freeboard'] )
        self.draw_pontoons(fig, pontoonMat, 0.5*self.params['pontoon_outer_diameter'], zcut)

        self.draw_column(fig, [0.0, 0.0], self.params['main_freeboard'], self.params['main_section_height'],
                           0.5*self.params['main_outer_diameter'], self.params['main_stiffener_spacing'])

        t_full = sectional2nodal(self.params['main_wall_thickness'])
        self.draw_ballast(fig, [0.0, 0.0], self.params['main_freeboard'], self.params['main_section_height'],
                          0.5*self.params['main_outer_diameter']-t_full,
                          self.params['main_permanent_ballast_height'], self.prob['variable_ballast_height'])
        
        if self.prob['main.buoyancy_tank_mass'] > 0.0:
            self.draw_buoyancy_tank(fig, [0.0, 0.0], self.params['main_freeboard'],
                                        self.params['main_section_height'],
                                        self.params['main_buoyancy_tank_location'],
                                        0.5*self.params['main_buoyancy_tank_diameter'],
                                        self.params['main_buoyancy_tank_height'])

        R_semi  = self.params['radius_to_offset_column']
        ncolumn = int(self.params['number_of_offset_columns'])
        angles = np.linspace(0, 2*np.pi, ncolumn+1)
        x = R_semi * np.cos( angles )
        y = R_semi * np.sin( angles )
        for k in xrange(ncolumn):
            self.draw_column(fig, [x[k], y[k]], self.params['offset_freeboard'], self.params['offset_section_height'],
                               0.5*self.params['offset_outer_diameter'], self.params['offset_stiffener_spacing'])

            t_full = sectional2nodal(self.params['offset_wall_thickness'])
            self.draw_ballast(fig, [x[k], y[k]], self.params['offset_freeboard'], self.params['offset_section_height'],
                              0.5*self.params['offset_outer_diameter']-t_full,
                              self.params['offset_permanent_ballast_height'], 0.0)

            if self.prob['off.buoyancy_tank_mass'] > 0.0:
                self.draw_buoyancy_tank(fig, [x[k], y[k]], self.params['offset_freeboard'],
                                            self.params['offset_section_height'],
                                            self.params['offset_buoyancy_tank_location'],
                                            0.5*self.params['offset_buoyancy_tank_diameter'],
                                            self.params['offset_buoyancy_tank_height'])
            
            
        self.draw_column(fig, [0.0, 0.0], self.params['hub_height'], self.params['tower_section_height'],
                         0.5*self.params['tower_outer_diameter'], None, (0.9,)*3)

            
        self.set_figure(fig, fname)


