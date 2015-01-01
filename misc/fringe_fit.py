from cloud_distribution import CD
from cloud_image import CloudImage as ci
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import pi
from scipy.optimize import curve_fit

PIXELSIZE = 3.75e-6 # m
MAGNIFICATION = 6.4
WAVELENGTH = 780e-9 #m

def sinusoid_1d(x, A, k, phi, offset):
    '''fitting function for a sinusoid intensity pattern'''
    return A*np.sin(k*x + phi) + offset
    
def fit_sinusoid_1d(image, xdata):
    '''fits a 1D sinusoid to a 1D image;
    includes constant offset'''
    A_guess = np.max(image)
    theta_guess = 2*pi/180
    k_guess = 4*pi*np.sin(theta_guess) / WAVELENGTH
    p_0 = [A_guess, k_guess, 0, 0] #fit guess for k, phi, offset

    coef, covar = curve_fit(sinusoid_1d, xdata, image, p0=p_0)
    return coef, covar

if __name__ == "__main__":
    # img = ci(r'Z:\Data\ACM Data\MoveMicro_h_from_sample\2014-10-15\40.5\2014-10-15_192124.mat')
    dir = r'D:\ACMData\Imaging System\expose_time\2014-10-16\MoveMicro40.5\fringes_fits'
    imgs = CD(dir, False).filelist
    thetas = []
    
    for f in imgs:
        img = ci(f)
        limg = img.light_image
        plt.imshow(limg)
        plt.show()

        limg_trunc = limg[:200, :]
        plt.imshow(limg_trunc)
        plt.show()

        light_pattern = np.sum(limg_trunc, axis = 1)
        distances = np.cumsum(np.ones(len(light_pattern))*(PIXELSIZE / MAGNIFICATION))
        coef, covar = fit_sinusoid_1d(light_pattern, distances)
        plt.plot(distances, sinusoid_1d(distances, *coef))
        plt.plot(distances, light_pattern)
        plt.show()
        
        theta_meas = np.arcsin(WAVELENGTH * coef[1] / (4*pi)) * 180 / pi
        theta_std = np.arcsin(WAVELENGTH * np.sqrt(covar[1][1]) / (4*pi)) * 180 / pi
        thetas.append(theta_meas)
        print 'Theta: %2.2f'%theta_meas
        print 'Theta Std: %2.2f'%theta_std
    # print 'Mean Theta:\t%2.4f'%np.mean(thetas)
    # print 'Theta Std:\t%2.4f'%np.std(thetas)