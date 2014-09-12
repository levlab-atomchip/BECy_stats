from math import sqrt, exp, pi
from scipy.optimize import curve_fit

AMU = 1.66e-27 # kg
M = 87*AMU #Rubidium-87
KB = 1.38e-23 # J/K

MYTEMP = 640e-9 #K
MYFREQ = 2*pi*500 #Hz

def width(t, T = MYTEMP, omega = MYFREQ):
    '''Width of a thermal gas at temperature T in a trap with frequency omega after time of flight t'''
    return sqrt(((KB * T)/(M*omega**2))*(1 + omega**2 * t**2))

def temp_func(tsqrd, w0sqrd, tempscld):
    return w0sqrd + tempscld*tsqrd
    
def temperature(tofs, widths):
    '''Extract temperature of cloud from widths'''
    tofsqrd = [(t*1e-3)**2 for t in tofs]
    widthsqrd = [(w*1e-6)**2 for w in widths]
    coefs, _ = curve_fit(temp_func, tofsqrd, widthsqrd)
    return (coefs[1] * M / KB, coefs)