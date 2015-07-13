
from collections import OrderedDict
import numpy as np
import itertools

from bokeh.plotting import figure

palette = itertools.cycle(["#f22c40", "#5ab738", "#407ee7", "#df5320", "#00ad9c", "#c33ff3"])

def plot_capex(top):

    def ang2xy(r, ang):
        return r * np.cos(ang), r * np.sin(ang)

    # all this can probably be done smarter with a Pandas DataFrame?! Any takers?
    turbine_capex = OrderedDict(Rotor=top.tcc_a.tcc.rotor_cost * top.turbine_number,
                                Tower=top.tcc_a.tcc.tower_cost * top.turbine_number,
                                Nacelle=top.tcc_a.tcc.nacelle_cost * top.turbine_number)
    infra_capex = OrderedDict(BOS=top.bos_a.bos_costs)

    wt_sum = np.sum(turbine_capex.values())
    infra_sum = np.sum(infra_capex.values())
    total_capex = np.array([wt_sum, infra_sum])

    inner_frac = np.append([0.], 2.0 * np.pi * total_capex / total_capex.sum())
    inner_angles = np.cumsum(inner_frac)
    vals = np.array(turbine_capex.values())
    wt_outer_angles = 2.0 * np.pi * np.cumsum(vals) / total_capex.sum()
    wt_outer_angles = np.append([0], wt_outer_angles)
    vals = np.array(infra_capex.values())
    infra_outer_angles = inner_angles[1] + 2.0 * np.pi * np.cumsum(vals) / total_capex.sum()
    infra_outer_angles = np.append([inner_angles[1]], infra_outer_angles)


    p = figure(title="Capital Costs Breakdown",
        x_axis_type=None, y_axis_type=None,
        x_range=[-420, 420], y_range=[-420, 420],
        min_border=0, outline_line_color="black",
        background_fill="white", border_fill="white")

    # setup the plot
    width = p.plot_width
    height = p.plot_height
    inner_radius = 0.01 * p.plot_width
    middle_radius = 2. / 8. * p.plot_width
    outer_radius = 5./8. * p.plot_width

    # first plot inner wedges
    # turbine CAPEX
    p.annular_wedge([0], [0], inner_radius, middle_radius,
                              inner_angles[0], inner_angles[1],
                              fill_color=palette.next(),
                              line_color='white',
                              line_width=2)
    ang = (inner_angles[0]+ inner_angles[1])/2.
    xp, yp = ang2xy((inner_radius+middle_radius) / 2., ang)
    p.text([xp], [yp], ['Turbine'], angle=0,
                                    text_align="center",
                                    text_baseline="middle",
                                    text_font_size="10pt")

    # Infrastructure CAPEX
    p.annular_wedge([0], [0], inner_radius, middle_radius,
                              inner_angles[1], inner_angles[2],
                              fill_color=palette.next(),
                              line_color='white',
                              line_width=2)
    ang = (inner_angles[1]+ inner_angles[2])/2.
    xp, yp = ang2xy((inner_radius+middle_radius) / 2., ang)
    p.text([xp], [yp], ['Balance of Station'], angle=0,
                                           text_align="center",
                                           text_baseline="middle",
                                           text_font_size="10pt")

    # Then plot outer wedges with cost breakdowns
    # 1) Turbine
    for i, color in zip(range(wt_outer_angles.shape[0]-1), palette):
        p.annular_wedge([0], [0], middle_radius, outer_radius,
                        wt_outer_angles[i], wt_outer_angles[i+1],
                        fill_color=color,line_color='white', line_width=2)
    # text
    for i, color in zip(range(wt_outer_angles.shape[0]-1), palette):

        ang = (wt_outer_angles[i]+ wt_outer_angles[i+1])/2.
        xp, yp = ang2xy((outer_radius+middle_radius) / 2., ang)
        ang = ang + np.pi if np.pi/2 < ang < 3*np.pi/2 else ang

        p.text([xp], [yp], [turbine_capex.keys()[i].replace('_', ' ')],
               angle=ang, text_align="center", text_baseline="middle",
               text_font_size="10pt")

    # 2) Infrastructure
    for i, color in zip(range(infra_outer_angles.shape[0]-1), palette):
        p.annular_wedge([0], [0], middle_radius, outer_radius,
                        infra_outer_angles[i], infra_outer_angles[i+1],
                        fill_color=color,line_color='white', line_width=2)
    # text
    for i, color in zip(range(infra_outer_angles.shape[0]-1), palette):

        ang = (infra_outer_angles[i]+ infra_outer_angles[i+1])/2.
        xp, yp = ang2xy((outer_radius+middle_radius) / 2., ang)
        ang = ang + np.pi if np.pi/2 < ang < 3*np.pi/2 else ang

        p.text([xp], [yp], [infra_capex.keys()[i].replace('_', ' ')],
               angle=ang,text_align="center", text_baseline="middle",
               text_font_size="10pt")

    return p
