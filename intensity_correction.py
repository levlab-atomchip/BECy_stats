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

def optical_depth(cloudimage
        , saturation_intensity=DEFAULT_I_SAT):
    try:
        optical_density = cloudimage.get_cd_image() * cloudimage.s_lambda
    except ci.FitError:
        optical_density = cloudimage.get_od_image()
    intensity_term = intensity_change(cloudimage) / saturation_intensity
    return optical_density + intensity_term

def intensity_change(cloudimage):
    return cloudimage.counts2intensity(cloudimage.light_image_trunc) - cloudimage.counts2intensity(cloudimage.atom_image_trunc)

def saturation(cloudimage):
    return np.mean(cloudimage.counts2saturation(cloudimage.light_image_trunc))

def atom_number(cloudimage):
    return np.sum(optical_depth(cloudimage)) / cloudimage.s_lambda * (cloudimage.pixel_size / cloudimage.magnification)**2

def optdens_number(cloudimage):
    return cloudimage.atom_number()

def int_term_number(cloudimage
        , saturation_intensity=DEFAULT_I_SAT):
    int_term = intensity_change(cloudimage) / saturation_intensity
    return np.sum(int_term) / cloudimage.s_lambda * (cloudimage.pixel_size / cloudimage.magnification)**2
