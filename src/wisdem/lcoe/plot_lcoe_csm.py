
from collections import OrderedDict
import numpy as np

import itertools

from bokeh.plotting import figure
from bokeh._legacy_charts import Bar, output_file, show

palette = ['#FFFFFF','#5ab4ac']

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
    # everything_lcoe.update(total_lcoe)

    cumSum = 0 
    invisibleBlock = OrderedDict()
    for key in everything_lcoe.keys():    
        invisibleBlock[key] = cumSum
        cumSum += everything_lcoe[key]

    everything_lcoe.update(total_lcoe)
    invisibleBlock['Total'] = 0

    myDict = OrderedDict(Invisible=invisibleBlock.values(), Amount=everything_lcoe.values())
    myKeys = OrderedDict(turbine=turbine_lcoe.keys(), infra=infra_lcoe.keys(), myOpex=opex_lcoe.keys())



    bar = Bar(myDict.values(), 
      everything_lcoe.keys(), 
      title="LCOE Costs Breakdown",
      stacked=True,
      palette=palette,
      ygrid=False,
      xlabel="Component",
      ylabel="LCOE ($/kwh)",
      )


    return bar
