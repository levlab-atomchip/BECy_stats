from scipy.ndimage import filters
from scipy.integrate import trapz
import numpy as np
from math import pi

# http://www.variousconsequences.com/2010/01/fft-based-abel-inversion-tutorial.html
def abel(dfdx, x): 
    nx = len(x) 
    # do each half of the signal separately (they should be the same 
    # up to noise) 
    integral = np.zeros((2,nx/2), dtype=float) 
    for i in xrange(nx/2, nx-1): 
        divisor = np.sqrt(x[i:nx]**2 - x[i]**2) 
        integrand = dfdx[i:nx] / divisor 
        integrand[0] = integrand[1] # deal with the singularity at x=r 
        integral[0][i-nx/2] = - trapz(integrand, x[i:nx]) / pi 
    for i in xrange(nx/2, 1, -1): 
        divisor = np.sqrt(x[i:0:-1]**2 - x[i]**2) 
        integrand = dfdx[i:0:-1] / divisor 
        integrand[0] = integrand[1] # deal with the singularity at x=r 
        integral[1][-i+nx/2] = - trapz(integrand, x[i:0:-1]) / pi 
    return(integral)

def centroid(arr):
   return np.argmax(arr)

def recenter(arr):
    this_centroid = centroid(arr)
    halfwidth = min([this_centroid, len(arr) - this_centroid - 1])
    return arr[this_centroid - halfwidth: this_centroid + halfwidth + 1]

def symmetrize(arr):
    arr = recenter(arr)
    symmetrized = 0.5*(arr[((len(arr)-1) / 2):] + arr[:((len(arr)+1)/2)][::-1])
    return np.hstack((symmetrized[::-1], symmetrized[1:]))

def gpe_bfield(img):
    cd_img = img.get_cd_image()
    cd_img = filters.gaussian_filter(cd_img, 2, order=0) #reconstruction filter
    img_cut = np.sum(cd_img, axis=1)
    real_pix = img.pixel_size / img.magnification
    symm = symmetrize(img_cut)
    img_cut_prime = filters.gaussian_filter(symm, 1, order=1)

    cloud_abel = abel(np.array(img_cut_prime), np.array(range(len(img_cut_prime))))
    cloud_abel = (np.nan_to_num(np.hstack((cloud_abel[0], cloud_abel[1])))) #radial density function

    ld_img = np.sum(cd_img, axis=0) * real_pix 
    n1d = ld_img * real_pix
    ff = (cloud_abel / np.sum(cloud_abel)) / (2*pi) #normalize

    psi_1d = np.nan_to_num(np.sqrt(n1d))
    k1 = filters.gaussian_filter(psi_1d, 1, order=2) #kinetic energy, longitudinal

    arrs = np.array(range(len(cloud_abel))) + 0.001 # to avoid singularity at origin
    psi_ff = np.nan_to_num(np.sqrt(ff))
    psi_ff_prime = filters.gaussian_filter(psi_ff, 1, order=1)
    inner = arrs * psi_ff_prime
    psi_ff_prime2 = filters.gaussian_filter(inner, 1, order=1)
    psi_ff_del = arrs * psi_ff_prime2
    k2 = psi_1d * np.sum(psi_ff_del) #kinetic energy, radial

    K_pre = (1.05e-34**2 / (2*87*1.66e-27*9.3e-24)) / real_pix**2
    KK = K_pre * (k1 + k2) / ((psi_1d)) #kinetic energy
    KK[np.isinf(KK)] = 0

    II_pre = (4.0 * pi * 1.05e-34**2 * 5e-9) / (87*1.66e-27 * 9.3e-24) / real_pix**3
    II = II_pre * n1d #interaction energy 

    BB = KK - II #reconstructed field
    return BB
