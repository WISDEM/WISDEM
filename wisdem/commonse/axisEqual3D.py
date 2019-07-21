#-------------------------------------------------------------------------------
# Name:        axisEqual3D.py
# Purpose:    When called with an axis3D instance it will render aspect ratio =1:1:1
#
# Author:      rdamiani- from http://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
#
# Created:     08/12/2013
# Copyright:   (c) rdamiani 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np

def main():
    pass

if __name__ == '__main__':
    main()

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)