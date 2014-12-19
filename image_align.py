"""Image Alignment

functions for alignment of atomic density images"""


import numpy as np
import matplotlib.pyplot as plt
from cloud_distribution import *
from cloud_image import CloudImage as ci
import BECphysics as bp
import bfieldsensitivity as bf

def optimal_shift(img, ref_img, shift_range=range(-30,30)):
    '''returns the best number of pixels to shift img relative to ref_img so as to get fragmentation aligned
    img, ref_img - 1D arrays
    shift_range - range of shifts to try, unit is pixel 
    '''
    s1 = img
    s2 = ref_img
    shifts = np.array(shift_range)
    msds = np.zeros(len(shifts))
    for ii, sh in enumerate(shifts):
        s1_shift = np.roll(s1, sh)
        if sh != 0:
            msds[ii] = np.mean(((s1_shift - s2)**2)[abs(sh):-abs(sh)])
        else:
            msds[ii] = np.mean((s1_shift - s2)**2)
    bestshift = shifts[np.argmin(msds)]
    return bestshift
