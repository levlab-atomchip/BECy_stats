from cloud_distribution import CD
from cloud_image import CloudImage as ci
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import ndimage


img = ci(r'Z:\Data\ACM Data\MoveMicro_h_from_sample\2014-10-15\40.5\2014-10-15_192124.mat')
# img = ci(r'Z:\Data\ACM Data\Statistics\loadmot_number\2014-08-11\2014-08-11_104746.mat')

dimg = img.dark_image
# plt.imshow(dimg)
# plt.show()

# noise_floor = np.mean(dimg) + 50*np.std(dimg)
noise_floor = 2000

print 'mean: %2.2f'%np.mean(dimg)
print 'std: %2.2f'%np.std(dimg)

light_masked = np.ma.masked_less_equal(img.light_image, noise_floor)
od_img = img.get_od_image(trunc_switch = False)
struct = ndimage.generate_binary_structure(2, 10)
Zmask1 = ndimage.morphology.binary_erosion(light_masked.mask, struct)
Zmask2 = light_masked.mask
Zmask3 = np.ma.masked_less_equal(img.atom_image, noise_floor).mask
Zmask = np.logical_or(Zmask1, Zmask2)
Zmask = np.logical_or(Zmask, Zmask3)
# od_masked = np.where(Zmask, 0, od_img)
od_masked = np.ma.masked_where(Zmask, od_img)
plt.imshow(light_masked)
plt.show()
plt.imshow(od_masked, vmax = 1)
plt.show()
plt.plot(np.sum(od_masked, axis = 1))
# plt.plot(np.sum(od_img, axis = 1), color = 'r')
plt.show()


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X = range(od_img.shape[1])
# Y = range(od_img.shape[0])
# X, Y = np.meshgrid(X, Y)
# struct = ndimage.generate_binary_structure(2, 4)
# Zmask1 = ndimage.morphology.binary_erosion(light_masked.mask, struct)
# Zmask2 = light_masked.mask
# Zmask = np.logical_or(Zmask1, Zmask2)
# Z = np.where(Zmask, 0, od_img)
# surf = ax.plot_surface(X, Y, Z, rstride = 50, cstride = 50,cmap=cm.jet, linewidth=0)
# ax.set_ylim(img.trunc_win_y[0],img.trunc_win_y[-1])
# ax.set_xlim(img.trunc_win_x[0],img.trunc_win_x[-1])
# ax.set_zlim(0, 4)
# plt.show()