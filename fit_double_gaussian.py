from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
import numpy as np
import matplotlib.pyplot as plt
#import cloud_image as ci
import time


#file=ci.CloudImage('2014-10-15_192154.mat')
#data=np.sum(np.array(file.get_od_image()),1)

def find_nearest(array, value):
    '''return the index and value of the array element closest to value'''
    idx = (np.abs(array-value)).argmin()
    return [array[idx], idx]

def double_gaussian_1d(x, Asqrt1, Asqrt2, mu1, mu2, sigma1, sigma2, offset, slope):
    '''fitting function for 1D double gaussian plus offset and line'''
    return Asqrt1**2*np.exp(-1.*(x-mu1)**2./(2.*sigma1**2.))+\
    Asqrt2**2*np.exp(-1.*(x-mu2)**2./(2.*sigma2**2.)) + \
    offset + slope*np.array(x)
    
def fit_double_gaussian_1d(image,guess_coef=True,p_0=None):
    '''fit data to a bimodel gaussian'''
    #max_value = image.max()
    #max_loc = np.argmax(image)
    #[_, half_max_ind] = find_nearest(image, max_value/2.)
    #hwhm = 1.17*abs(half_max_ind - max_loc) # what is 1.17???
    #print hwhm
    if guess_coef:
        peaks,max_loc=locate_max(image)#this step may break if 10 is not a good guess for width and it is a first pass
        hwhm=0.5*np.abs(max_loc[0]-max_loc[-1])# this is basically half of the distance between the two peaks
        p_0 = [np.sqrt(peaks[0]), np.sqrt(peaks[1]), max_loc[0], max_loc[-1],hwhm, hwhm, 0., 0.] #fit guess
    else:
        #manually enter guesses
        p_0=p_0
    xdata = np.arange(np.size(image))

    coef, _ = curve_fit(double_gaussian_1d, xdata, image, p0=p_0)
    return coef

def locate_max(data,n_peak=2,width=10):
    '''find the location and values of n_peak local maxima, currently only works for n_peaks=2'''   
    peakindx=find_peaks_cwt(data,np.arange(1,width))
    sd=sorted(data[peakindx])
    indx=np.argwhere(data>=sd[-n_peak])
    #print sd[-n_peak:], indx
    return sd[-n_peak:], indx

def peak_separation(coef):
    return np.abs(coef[2]-coef[3]) #need to change this if the set of coefs get changed
        
#print peakindx
#print data[peakindx]
#locate_max(data)
#start=time.time()
#coef=fit_double_gaussian_1d(data)
#print (time.time()-start)

#xdata = np.arange(np.size(data))

#plt.plot(double_gaussian_1d(xdata,coef[0],coef[1],coef[2],coef[3],coef[4],coef[5],coef[6],coef[7]))
#plt.plot(data)
#print np.abs(coef[2]-coef[3])
#print np.abs(coef[4]-coef[5])

#plt.show()