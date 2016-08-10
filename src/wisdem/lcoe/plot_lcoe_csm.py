
from collections import OrderedDict
import numpy as np

import itertools
from bokeh.plotting import figure
from bokeh.charts import Bar, output_file, show
from pandas import DataFrame as df
from bokeh.charts.attributes import ColorAttr, CatAttr
from bokeh.palettes import Spectral10
from bokeh.models import HoverTool

palette = itertools.cycle(Spectral10)

def plot_lcoe(top):

    # all this can probably be done smarter with a Pandas DataFrame?! Any takers?
    aep = top.fin_a.net_aep
    fcr = top.fin_a.fixed_charge_rate
    turbine_lcoe = OrderedDict(Rotor=(fcr / aep * top.tcc_a.tcc.blade_cost * top.tcc_a.tcc.blade_number + fcr / aep * top.tcc_a.tcc.hub_system_cost) * top.turbine_number,
                                Tower=fcr / aep * top.tcc_a.tcc.tower_cost * top.turbine_number,
                                Nacelle=fcr / aep * top.tcc_a.tcc.nacelle_cost * top.turbine_number)

    infra_lcoe = OrderedDict(Assembly=fcr / aep * top.bos_breakdown.assembly_and_installation_costs,
                              Development=fcr / aep * top.bos_breakdown.development_costs,
                              Electrical=fcr / aep * top.bos_breakdown.electrical_costs,
                              Substructure=fcr / aep * top.bos_breakdown.foundation_and_substructure_costs,
                              Other=fcr / aep * top.bos_breakdown.foundation_and_substructure_costs,
                              Preparation=fcr / aep * top.bos_breakdown.preparation_and_staging_costs,
                              Soft=fcr / aep * top.bos_breakdown.soft_costs,
                              Transportation=fcr / aep * top.bos_breakdown.transportation_costs)

    opex_lcoe = OrderedDict(Opex=top.opex_a.avg_annual_opex / aep)

    turbine_sum = np.sum(turbine_lcoe.values()) 
    infra_sum = np.sum(infra_lcoe.values())
    opex_sum = np.sum(opex_lcoe.values())

    total_lcoe = np.array([turbine_sum, infra_sum, opex_sum]) # array containing Turbine, BOS, and OPEX lcoe costs 
    lcoe = total_lcoe.sum() 
    total_lcoe = OrderedDict(Total=lcoe)

    everything_lcoe = turbine_lcoe
    everything_lcoe.update(infra_lcoe)
    everything_lcoe.update(opex_lcoe)

    cumSum = 0 
    invisibleBlock = OrderedDict()
    for key in everything_lcoe.keys():    
        invisibleBlock[key] = cumSum
        cumSum += everything_lcoe[key]

    everything_lcoe.update(total_lcoe)
    invisibleBlock['Total'] = 0

    Combined = invisibleBlock.values()


    colors = ['white' for i in range(len(Combined))]
    next_color = palette.next()
    for i in range(3):
      colors.append(next_color)
    next_color = palette.next()
    for i in range(8):
      colors.append(next_color)
    colors.append(palette.next())
    colors.append('black')


    myGroup=[]
    for x in range(2):
      for i in range(3):
        myGroup.append('turbine')
      for i in range(8):
        myGroup.append('infrastructure')
      myGroup.append('opex')
      myGroup.append('total')


    for stuff in everything_lcoe.values():    
        Combined.append(stuff)

    myKeys = OrderedDict(turbine=turbine_lcoe.keys(), infra=infra_lcoe.keys(), myOpex=opex_lcoe.keys())

    myNames = myKeys['turbine']
    for stuff in list(myKeys['turbine']):
        myNames.append(stuff)

    myStack  = []
    for i in range(13):
      myStack.append(1)
    for i in range(13):
      myStack.append(2)


    myDict = dict(Amount=Combined, Group=myGroup,  Names=myNames, stack=myStack, color=colors)

    myDF = df(data=myDict)
    # print myDF
    myBar = Bar(myDF, values='Amount', label=CatAttr(columns=['Names'], sort=False), stack="stack",
     toolbar_location="above", color="color", ygrid=False, legend=None,
      title='LCOE Costs Breakdown', ylabel="LCOE ($/kWh)", xlabel="Component",
      tools="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave",
)

    hover = myBar.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
                ("Component","@Names"),
                # ("Group", "@Group"), # maybe one day bokeh will fix this and it will work. It should show "turbine, infra, or opex"
            ])

    

    return myBar
    
