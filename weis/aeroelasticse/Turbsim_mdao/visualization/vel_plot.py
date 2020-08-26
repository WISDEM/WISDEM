"""
===============
Rain simulation
===============

Simulates rain drops on a surface by animating the scale and opacity
of 50 scatter points.

Author: Nicolas P. Rougier
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

velocities=np.load('velocity.npy')

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
x = np.arange(0,29)
ax.set_xlim(0, 29), ax.set_xticks([])
ax.set_ylim(0, 29), ax.set_yticks([])

div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

cont = ax.contourf(x, x, velocities[0, 0, :, :])
n_frames = len(velocities[:,0,0,0])
#quit()
MULT= int(1/0.05)
def update(frame_number):
    # Get an index which we can use to re-spawn the oldest raindrop.
    current_index = (frame_number * MULT) % n_frames
    cont = ax.contourf(x, x, velocities[current_index, 0, :, :], cmap=plt.cm.coolwarm)
    cax.cla()
    colorbar = fig.colorbar(cont, cax=cax)
    ax.set_title('Time is %f seconds'%(current_index*.05))
    # Update the scatter collection, with the new colors, sizes and positions.
    #scat.set_edgecolors(rain_drops['color'])
    #scat.set_sizes(rain_drops['size'])
    #scat.set_offsets(rain_drops['position'])


# Construct the animation, using the update function as the animation
# director.
animation = FuncAnimation(fig, update, frames=n_frames/MULT)
#plt.show()
animation.save('flow.gif', dpi=80, writer='imagemagick')
