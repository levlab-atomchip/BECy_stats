#!/usr/bin/env python

"""
fit_functions - a collection of functions used to fit data
"""

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import numpy as np
from scipy.optimize import curve_fit
from math import pi

def find_nearest(array, value):
    '''return the index and value of the array element closest to value'''
    idx = (np.abs(array-value)).argmin()
    return [array[idx], idx]

def gaussian_1d(x, Asqrt, mu, sigma, offset, slope):
    '''fitting function for 1D gaussian plus offset and line'''
    return Asqrt**2*np.exp(-1.*(x-mu)**2./(2.*sigma**2.)) + offset + \
                                                    slope*np.array(x)

def gaussian_1d_noline(x, Asqrt, mu, sigma, offset):
    '''fitting function for 1D gaussian plus offset'''
    return Asqrt**2*np.exp(-1.*(x-mu)**2./(2.*sigma**2.)) + offset
    
def bec_1d(x, maxsqrt, Bsqrt, center):
    '''fitting function for a Thomas-Fermi BEC in a harmonic trap'''
    # using sqrt of parameters to force positive values
    rho = maxsqrt**2 - Bsqrt**2 * (x - center)**2
    rho = [r if r > 0 else 0 for r in rho]
    rho = [r**2 for r in rho]
    return  rho
    
def bec_thermal_1d(x, Asqrt, mu, sigma, offset, maxsqrt, Bsqrt):
    '''Simultaneous fit of BEC and gaussian'''
    return gaussian_1d_noline(x, Asqrt, mu, sigma, offset) + bec_1d(x, maxsqrt, Bsqrt, mu)

def gaussian_2d(xdata,
                A_x,
                mu_x,
                sigma_x,
                A_y,
                mu_y,
                sigma_y,
                offset):
    '''fitting function for 2D gaussian plus offset'''
    x = xdata[0]
    y = xdata[1]
    return A_x*np.exp(-1.*(x-mu_x)**2./(2.*sigma_x**2.)) + \
            A_y*np.exp(-1.*(x-mu_y)**2./(2.*sigma_y**2.)) + offset

def fit_gaussian_1d(image):
    '''fits a 1D Gaussian to a 1D image;
    includes constant offset and linear bias'''
    max_value = image.max()
    max_loc = np.argmax(image)
    [_, half_max_ind] = find_nearest(image, max_value/2.)
    hwhm = 1.17*abs(half_max_ind - max_loc) # what is 1.17???
    p_0 = [np.sqrt(max_value), max_loc, hwhm, 0., 0.] #fit guess
    xdata = np.arange(np.size(image))

    coef, _ = curve_fit(gaussian_1d, xdata, image, p0=p_0)
    return coef

def fit_gaussian_1d_wings(image, xdata):
    '''fits a 1D Gaussian to a 1D image;
    includes constant offset and linear bias'''
    max_value = image.max()
    max_loc = np.argmax(image)
    [_, half_max_ind] = find_nearest(image, max_value/2.)
    hwhm = 1.17*abs(half_max_ind - max_loc) # what is 1.17???
    p_0 = [np.sqrt(max_value), max_loc, hwhm, 0., 0.] #fit guess

    coef, _ = curve_fit(gaussian_1d, xdata, image, p0=p_0)
    return coef
    
def fit_gaussian_1d_noline(image):
    '''fits a 1D Gaussian to a 1D image;
    includes constant offset and linear bias'''
    max_value = image.max()
    max_loc = np.argmax(image)
    [_, half_max_ind] = find_nearest(image, max_value/2.)
    hwhm = 1.17*abs(half_max_ind - max_loc) # what is 1.17???
    p_0 = [np.sqrt(max_value), max_loc, hwhm, 0.] #fit guess
    xdata = np.arange(np.size(image))

    coef, _ = curve_fit(gaussian_1d_noline, xdata, image, p0=p_0)
    return coef
    
def fit_gaussian_1d_noline_wings(image, xdata):
    '''fits a 1D Gaussian to a 1D image;
    includes constant offset and linear bias'''
    max_value = image.max()
    max_loc = np.argmax(image)
    [_, half_max_ind] = find_nearest(image, max_value/2.)
    hwhm = 1.17*abs(half_max_ind - max_loc) # what is 1.17???
    p_0 = [np.sqrt(max_value), max_loc, hwhm, 0.] #fit guess

    coef, _ = curve_fit(gaussian_1d_noline, xdata, image, p0=p_0)
    return coef

def fit_bec_thermal(image):
    ''''Simultaneous fit to gaussian and BEC'''
    max_value = image.max()
    max_loc = np.argmax(image)
    [_, half_max_ind] = find_nearest(image, max_value/2.)
    hwhm = abs(half_max_ind - max_loc) # what is 1.17???
    max_guess = np.sqrt(max_value)
    p_0 = [max_guess/2, max_loc, hwhm, min(image),np.sqrt(max_guess/2), np.sqrt(1 / hwhm)] #fit guess
    xdata = np.arange(np.size(image))

    coef, _ = curve_fit(bec_thermal_1d, xdata, image, p0=p_0)
    plt.plot(xdata, image); plt.plot(xdata, bec_thermal_1d(xdata, *coef)); plt.show()
    return coef
    
def fit_partial_bec(image):
    '''fits a gaussian and TF profile to a 1D image by trying a gaussian, 
    then fitting to the wings, and then fitting a TF profile to the remainder.
    It's as bad as it sounds! I think a bayesian method would be better...'''
    WING_DEF = 1.5 #sigma
    gaussian_attempt = fit_gaussian_1d_noline(image)
    center_attempt = gaussian_attempt[1]
    width_attempt = gaussian_attempt[2]
    xaxis = xrange(len(image))
    x_wings = [x for x in xaxis if x < center_attempt - WING_DEF*width_attempt or x > center_attempt + WING_DEF*width_attempt]
    x_cen = [x for x in xaxis if x not in x_wings]
    y_wings = np.array([yy[1] for yy in enumerate(image) if yy[0] in x_wings])
    gaussian_wings = fit_gaussian_1d_noline_wings(y_wings, x_wings)
    gaussian_y = gaussian_1d_noline(xaxis, *gaussian_wings)
    nongaussian = image - gaussian_y
    nongaussian_cen = np.array([yy[1] for yy in enumerate(nongaussian) if yy[0] in x_cen])
    p0 = [np.sqrt(max(nongaussian)), np.sqrt(1/(0.5*(max(x_cen) - min(x_cen)))), gaussian_wings[1]]
    bec_fit, _ = curve_fit(bec_1d, x_cen, nongaussian_cen, p0)
    bec_y = bec_1d(xaxis, *bec_fit)
    bec_y = [b if b > 0 else 0 for b in bec_y]
    
    bec_cen = bec_fit[2]
    bec_hw = bec_fit[0] / bec_fit[1]
    x_wings2 = [x for x in xaxis if x < bec_cen - bec_hw or x > bec_cen + bec_hw]
    y_wings2 = np.array([yy[1] for yy in enumerate(image) if yy[0] in x_wings2])
    gaussian_wings2 = fit_gaussian_1d_noline_wings(y_wings2, x_wings2)
    gaussian_y2 = gaussian_1d_noline(xaxis, *gaussian_wings2)

    plt.plot(xaxis, image); 
    plt.plot(xaxis, bec_y+gaussian_y2); 
    plt.plot(xaxis, gaussian_y2); 
    # plt.ylim((0, 1.2*max(image))); 
    plt.show()
    return (gaussian_wings2, bec_fit)

def fit_gaussian_2d(image):
    '''fits a 2D Gaussian to a 2D Image'''
    img_x = np.sum(image, 0)
    img_y = np.sum(image, 1)
    x_coefs = fit_gaussian_1d(img_x)
    #gets coefficient estimates from 1D fits
    y_coefs = fit_gaussian_1d(img_y)
    x, y = np.meshgrid(np.arange(img_x.size), np.arange(img_y.size))

    coef, _ = curve_fit(gaussian_2d,
                            [x, y],
                            image,
                            p0=np.delete(np.append(x_coefs, y_coefs), 3))
    return coef


def temp_func(t, sigma_0, sigma_v):
    '''fitting function for temperature measurement'''
    return np.sqrt(sigma_0**2 + (sigma_v**2)*(t**2))

def lifetime_func(t, N0, decay_rate, offset):
    '''fitting function for lifetime measurement'''
    return N0 * np.exp(-decay_rate * t) + offset
    
def freq_func(t, omega, amplitude, offset, phase):
    '''fitting function for trap frequency measurement'''
    return offset + amplitude*np.sin(omega*t + phase)
    
def magnif_func(x, a, b, c):
    '''fitting function for magnification measurement'''
    return a*np.square(x) + b*x + c
 
