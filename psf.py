'''psf.py - Calculate resolution from distribution of cloud images'''

import matplotlib.pyplot as plt
import numpy as np
import cloud_distribution as cd
from cloud_image import CloudImage as ci
import BECphysics as bp
import bfieldsensitivity as bf
import image_align as ia
import scipy.signal
import math

# Long term we should implement this as a subclass of CloudDistribution
#class AlignedCloudDistribution(cd.CD):
#	'''Container for aligned cloud distribution'''

DEFAULT_MAX_SHIFT = 6
DEFAULT_PIXSIZE = 13.0 / 24 #PIXIS

def next_power_two(n):
    this_exp = int(math.log(n, 2))
    return 2**(this_exp + 1)

def get_aligned_line_densities(dist, max_shift=DEFAULT_MAX_SHIFT, pixsize=DEFAULT_PIXSIZE):
    '''Produce aligned line densities from a distribution.
        Args:
            dist: a CloudDistribution
            max_shift: maximum allowed shift, in pixels.
                        Images that need to be shifted more than this will be discarded
    '''
    imgs = [ci(ff) for ff in dist.filelist]
    cdimgs = [im.get_od_image() / im.s_lambda for im in imgs]
    ldimgs = [bp.line_density(cdim, pixsize) for cdim in cdimgs]
    ldsnorm = [ld/np.sum(ld) for ld in ldimgs]

    aligned = []
    shifts = []
    for nn, ld in enumerate(ldimgs):
        shift = ia.optimal_shift(ldsnorm[0], ldsnorm[nn])
        if abs(shift) > max_shift:
            pass
        else:
            aligned.append(np.roll(ld, -shift))
            shifts.append(shift)
    return aligned, shifts

def get_shift_stats(dist, max_shift=DEFAULT_MAX_SHIFT, pixsize=DEFAULT_PIXSIZE):
    _, shifts = get_aligned_line_densities(dist, max_shift)
    shift_mean = np.mean(shifts)
    print shift_mean

    shift_std = np.std(shifts)
    print shift_std

    shift_um = shift_std * pixsize
    print 'Noise in lateral position: %2.1f um'%shift_um

    plt.hist(np.array(shifts)*pixsize, 10)
    plt.xlim(-5,5)
    plt.xlabel('Shifts / um')
    plt.ylabel('Counts')
    plt.title('Histogram of shifts over %d runs'%len(shifts))

def get_power_spectral_density(dist, pixsize=DEFAULT_PIXSIZE, **kwargs):
    aligned_lds, _ = np.array(align_lds(dist, **kwargs))
    avg_ld = np.mean(aligned_lds, axis=0)
    avg_norm = avg_ld / np.sum(avg_ld)
    window_size = next_power_two(len(avg_norm))
    avg_fft = np.fft.fftshift(np.fft.fft(avg_norm, window_size))

    psd_avg = np.abs(avg_fft)**2
    psd_norm = psd_avg / sum(psd_avg)
    faxis = np.fft.fftshift(np.fft.fftfreq(window_size, pixsize))

    plt.plot(0.5/faxis[window_size/2:], psd_norm[window_size/2:])
    plt.xlabel('Spatial resolution  (um)')
    plt.title('Power Spectrum of Averaged Atom Profiles')
    plt.xlim(0.2,2)
    plt.ylim(0,0.5e-4)
    plt.show()

#aliases
align_lds = get_aligned_line_densities
sh_stats = get_shift_stats
psd = get_power_spectral_density
