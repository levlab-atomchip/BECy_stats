#!/usr/bin/env python

"""
intensity_correction - correct for high intensity effects on optical depth calculations
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import cloud_image as ci
from math import pi
from BECphysics import C, H, LAMBDA_RB
import numpy as np

DEFAULT_I_SAT = 30.54 # W/m**2, for pi light
# DEFAULT_PIX_SIZE = 3.75e-6/1.86 #m, dragonfly
# DEFAULT_QUANTUM_EFFICIENCY = 0.25 # dragonfly
DEFAULT_PIX_SIZE = 13.0e-6/24 #m, pixis
DEFAULT_QUANTUM_EFFICIENCY = 0.95 # pixis
DEFAULT_IMAGE_TIME = 10e-6 #s
DEFAULT_X_SECTION = 1.55e-13 #m**2, pi light steady state

def get_image_time(this_image):
    if this_image.cont_par_name == 'ExposeTime':
        return this_image.curr_cont_par
    else:
        try:
            this_image_time = this_img.get_variables_values()['ExposeTime']
        except:
            this_image_time = DEFAULT_IMAGE_TIME
        return this_image_time

def optical_depth(cloudimage
        , saturation_intensity=DEFAULT_I_SAT
        ):
    try:
        optical_density = cloudimage.get_cd_image() * cloudimage.s_lambda
    except ci.FitError:
        optical_density = cloudimage.get_od_image()
    intensity_term = intensity_change(cloudimage) / saturation_intensity
    return optical_density + intensity_term

def intensity_change(cloudimage
        ):
    this_image_time = get_image_time(cloudimage)
    return counts2intensity(cloudimage.light_image_trunc, image_time=this_image_time) - counts2intensity(cloudimage.atom_image_trunc, image_time=this_image_time)

def counts2intensity(rawimage
        , quantum_efficiency=DEFAULT_QUANTUM_EFFICIENCY
        , pix_size=DEFAULT_PIX_SIZE
        , image_time=DEFAULT_IMAGE_TIME
        ):
    return (rawimage / quantum_efficiency)*(H*C/LAMBDA_RB) / (pix_size**2) / image_time

def saturation(cloudimage):
    this_image_time = get_image_time(cloudimage)
    return np.mean(counts2saturation(cloudimage.light_image_trunc, image_time=this_image_time))
    
def counts2saturation(rawimage
        , saturation_intensity=DEFAULT_I_SAT
        , image_time=DEFAULT_IMAGE_TIME
        ):
    return counts2intensity(rawimage, image_time=image_time)/saturation_intensity

def atom_number(cloudimage
        , x_section=DEFAULT_X_SECTION
        , pix_size=DEFAULT_PIX_SIZE
        ):
    return np.sum(optical_depth(cloudimage)) / x_section * pix_size**2

def optdens_number(cloudimage):
    return cloudimage.atom_number()

def int_term_number(cloudimage
        , x_section=DEFAULT_X_SECTION
        , pix_size=DEFAULT_PIX_SIZE
        , saturation_intensity=DEFAULT_I_SAT
        ):
    int_term = intensity_change(cloudimage) / saturation_intensity
    return np.sum(int_term) / x_section * pix_size**2
