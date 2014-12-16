from cloud_distribution import *
from cloud_image import CloudImage as ci
import BECphysics as bp
import matplotlib.pyplot as plt
import numpy as np

'''
This script will generate a 2D field map given a data set.

Currently shifting is nor included and the data sets are assume to have features that line up perfectly.

Maybe want to shift 'lds' for some data set
'''
cdist = CD(r'C:\Users\Levlab\Documents\becy_stats_data\statistics\5umMap\2014-10-17')
imgs = [ci(ff) for ff in cdist.filelist]
cdimgs = [img.get_od_image / img.s_lambda for img in imgs]
lds = [bp.line_density(cd) for cd in cdimgs]

plt.plot(lds[0])
plt.show()

fas = [bp.field_array(ld) for ld in lds]

fasarr = np.array(fas)

xaxis = np.cumsum(np.ones(len(fas)) * (3.75 / 6.4))
ylocs = np.cumsum(np.ones(fasarr.shape[1])*2)

X, Y = np.meshgrid(xaxis, ylocs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.contourf(X, Y, fasarr)
plt.show()