from cloud_distribution import CloudDistribution as CD
from cloud_image import CloudImage as ci
import numpy as np
import matplotlib.pyplot as plt

gamma = 8.63e-29 # T/m**3, from Wildermuth thesis p 85
rho0 = 2e-6
z0 = 2e-6

# dnoise = CD(r'C:\Users\Levlab\Documents\becy_stats_data\dragonfly_noise\2014-10-16\small_sample')
dnoise = CD(r'Z:\Data\ACM Data\dragonfly_noise\2014-10-20')


cts_per_px = []
ln_per_px = []
dn_per_px = []

for imgfile in dnoise.filelist:
    img = ci(imgfile)
    # img.truncate_image(600, 1000, 500, 600)
    odimg = img.get_od_image()
    # light_noise = img.atom_image_trunc - img.light_image_trunc*img.fluc_cor
    light_noise = img.atom_image_trunc - img.light_image_trunc
    dark_noise = img.dark_image_trunc
    ln_sum = np.sum(light_noise, axis = 0)
    dn_sum = np.sum(dark_noise, axis = 0)
    odsum = np.sum(odimg, axis = 0) * (img.pixel_size / img.magnification)**2 / img.s_lambda
    # plt.plot(odsum); plt.show()
    cts_per_px.append(odsum)
    ln_per_px.append(ln_sum)
    dn_per_px.append(dn_sum)
ct_mu = np.mean(cts_per_px)
delB = gamma * ct_mu / (rho0**2 * z0)
print 'Delta N = %2.2f\n'%ct_mu
print 'Delta B = %2.2e T\n'%delB
print 'Light Noise: \t%2.2f counts'%np.mean(ln_per_px)
print 'Dark Noise: \t%2.2f counts'%np.mean(dn_per_px)