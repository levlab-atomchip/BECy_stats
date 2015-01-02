#!/usr/bin/env python

"""
strong_saturation - code to analyse high intensity absorption images
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import cloud_image as ci
import intensity_correction as ic
import re
import glob
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_IMAGE_TIME = 10e-6

def sat_vs_num(data_dir, **kwargs):
    '''Given a directory, return the saturation parameters and intensity-corrected atom numbers from each image'''
    data_filenames = sorted(glob.glob(data_dir + '/*.mat'))

    saturation2num = []

    for index, datum in enumerate(data_filenames):
	print 'Processing Image %d'%(index+1)
        this_img = ci.CloudImage(datum)
        this_number = ic.atom_number(this_img)
        this_saturation = ic.saturation(this_img)
        saturation2num.append((this_saturation, this_number))

    saturations, nums = zip(*saturation2num)
    return (saturations, nums)

def od_parts(data_dir, **kwargs):
    '''Given a directory, return the saturation parameters, optical densities, and intensity correction terms from each image'''
    data_filenames = sorted(glob.glob(data_dir + '/*.mat'))

    saturation2od_parts = []

    for index, datum in enumerate(data_filenames):
        print 'Processing Image %d'%(index+1)
        this_img = ci.CloudImage(datum)
        this_saturation = ic.saturation(this_img)
        this_optdens_part = ic.optdens_number(this_img)
        this_int_part = ic.int_term_number(this_img)
        saturation2od_parts.append((this_saturation, this_optdens_part, this_int_part))

    saturations, optdens_parts, int_parts = zip(*saturation2od_parts)
    return (saturations, optdens_parts, int_parts)
	
def sigma_w_int_corr(data_dir):
    '''DEPRECATED!
    Given a directory, return the saturation parameters and numbers, calculated using an intensity-corrected cross section.
    This is inaccurate for high intensities!'''
    data_filenames = glob.glob(data_dir + '/*.mat')

    saturation2num = []

    for datum in data_filenames:
	this_img = ci.CloudImage(datum)
	this_saturation = np.mean(ic.counts2saturation(this_img.light_image_trunc))
	this_intensity = this_saturation * ic.DEFAULT_I_SAT
	this_x_section = ic.DEFAULT_X_SECTION / (1 + this_intensity / ic.DEFAULT_I_SAT)
	this_number = ic.optdens_number(this_img, x_section = this_x_section)
	saturation2num.append((this_saturation, this_number))

    saturations, nums = zip(*saturation2num)
    return (saturations, nums)
