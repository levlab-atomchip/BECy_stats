'''Code to produce a formatted image of the average of 10 OD images acquired using the PIXIS'''

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import ticker
import cloud_distribution

UM_PER_PX = 0.542
xes = [0, 50, 100, 150, 200, 250]
xlabs = ['%d'%n for n in xes]
yes = [0, 5, 10, 15, 20]
ylabs = ['%d'%n for n in yes]

CD = cloud_distribution.CloudDistribution

dist = CD(r'Z:\Data\ACM Data\Imaging system\Kinetics\2014-11-12\39.5', False)
od = dist.get_average_image()

plt.figure()
ax = plt.gca()
im = ax.imshow(od, vmin = 0, vmax = 0.3)
ax.set_xticks(np.array(xes) / UM_PER_PX)
ax.set_xticklabels(xlabs)
ax.set_yticks(np.array(yes) / UM_PER_PX)
ax.set_yticklabels(ylabs)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

cb = plt.colorbar(im, cax=cax, ticks=[0, 0.1, 0.2, 0.3])
cb.ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3'])



# tick_locator = ticker.MaxNLocator(nbins=5)
# cb.locator = tick_locator
# cb.update_ticks()
plt.show()