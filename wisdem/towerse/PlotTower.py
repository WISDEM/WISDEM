#-------------------------------------------------------------------------------
# Name:        PlotTower.Py
# Purpose:
#
# Author:      rdamiani
#
# Created:     11/24/2014
# Copyright:   (c) rdamiani 2014
# Licence:     <Apache 2014>
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(mytwr,util=False,savefileroot=[]):

    twr_z=mytwr.tower1.z
    twr_D=mytwr.tower1.d

    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')

    plt.plot(-twr_D/2, twr_z)
    plt.plot( twr_D/2, twr_z)
    ax.set_xlim([-60,60]);

#    ax.auto_scale_xyz([min(XYZ1[:,0]),max(XYZ1[:,0])],[min(XYZ1[:,1]),max(XYZ1[:,1])],[min(XYZ1[:,2]),max(XYZ1[:,2])])

    if savefileroot:
        plt.savefig(savefileroot+'_config.png',format='png')
    else:
        plt.show()

    #Plot utilization of Tower if requested
    if util:

        fig2=plt.figure(figsize=(8.0, 5))
        ax2 = fig2.add_subplot(111)
        ax2.set_title('Utilization Ratio');
        #plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

        yrange=(np.min(twr_z),np.max(twr_z));
        ax2.set_xlim([0,2]);
        ax2.set_ylim(yrange);

        ax2.plot(mytwr.stress1, twr_z, label='VonMises Util1')
        ax2.plot(mytwr.stress2, twr_z, label='VonMises Util2')
        ax2.plot(mytwr.shellBuckling1, twr_z, label='EUsh Util1')
        ax2.plot(mytwr.shellBuckling2,twr_z, label='EUsh Util2')
        ax2.plot(mytwr.buckling1, twr_z, label='GL Util1')
        ax2.plot(mytwr.buckling2, twr_z, label='GL Util2')
        ax2.grid()
        if any(mytwr.damage):
            ax2.plot(mytwr.damage, twr_z, label='damage')
        ax2.legend(bbox_to_anchor=(1.0, 1.0), loc=1)
        ax2.set_xlabel('Utilization Ratios ')
        ax2.set_ylabel('z from base of tower [m]')

        #Plot tower profile
        ax3=plt.twiny(ax2)# .axes(ax1.get_position(),frameon=False)
        #hh2=plt.axes('Position', pos,'NextPlot','Add','XtickLabel','','Xtick',[],frameon=False);
        ax3.plot(twr_D/2,twr_z);
        ax3.plot(-twr_D/2,twr_z);
        ax3.set_aspect('equal')
        ax3.set_frame_on(False)
        ax3.set_xticklabels('')
        ax3.set_xticks([])
        ax2.set_xlim([0,2]);
        #hh2.set_aspect('equal')
        ##ax2.axis('equal')

        if savefileroot:
            plt.savefig(savefileroot+'_util.png',format='png')
        else:
            plt.show()


if __name__ == '__main__':

    import MyTowerInputs
    mytwr=MyTowerInputs.main()[0]
    mytwr.run()
    main(mytwr,util=True)
