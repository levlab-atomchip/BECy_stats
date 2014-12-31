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
DEFAULT_PIX_SIZE = 3.75e-6/1.86 #m, dragonfly
DEFAULT_QUANTUM_EFFICIENCY = 0.25 # dragonfly
DEFAULT_IMAGE_TIME = 10e-6 #s
DEFAULT_X_SECTION = 1.55e-13 #m**2, pi light steady state

def optical_depth(cloudimage
        , saturation_intensity=DEFAULT_I_SAT
        ):
    optical_density = cloudimage.get_od_image()
    intensity_term = intensity_change(cloudimage) / saturation_intensity
    return optical_density + intensity_term

def intensity_change(cloudimage):
    return counts2intensity(cloudimage.light_image_trunc) - counts2intensity(cloudimage.atom_image_trunc)

def counts2intensity(rawimage
        , quantum_efficiency=DEFAULT_QUANTUM_EFFICIENCY
        , pix_size=DEFAULT_PIX_SIZE
        , image_time=DEFAULT_IMAGE_TIME
        ):
    return (rawimage / quantum_efficiency)*(H*C/LAMBDA_RB) / (pix_size**2) / image_time

def counts2saturation(rawimage
        , saturation_intensity=DEFAULT_I_SAT
        ):
    return counts2intensity(rawimage)/saturation_intensity

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
