from semi_instance import SemiInstance
import numpy as np

        
class TLPInstance(SemiInstance):
    def __init__(self):
        super(TLPInstance, self).__init__()

        self.params['mooring_type']        = 'nylon'
        self.params['anchor_type']         = 'suctionpile'
        self.params['mooring_line_length'] = 0.95 * self.params['water_depth']
        self.params['anchor_radius']       = 10.0
        self.params['mooring_diameter']    = 0.1
        
        # Change scalars to vectors where needed
        self.check_vectors()

    def get_constraints(self):
        conlist = super(TLPInstance, self).get_constraints()

        poplist = []
        for k in range(len(conlist)):
            if ( (conlist[k][0].find('metacentric') >= 0) or
                 (conlist[k][0].find('freeboard_heel') >= 0) ):
                poplist.append(k)

        poplist.reverse()
        for k in poplist: conlist.pop(k)

        return conlist
        
