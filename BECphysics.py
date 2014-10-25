from math import sqrt, exp, pi
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

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
    # p0 = [np.min(widthsqrd), 100e-9*KB/M]
    slope, intercept, _, _, std_err = linregress(tofsqrd, widthsqrd)
    coefs = (slope, intercept)
    xaxis = np.linspace(0, np.max(tofs), 100)
    
    print '\n\tTemperature: %2.2f nK'%(coefs[0] * M * 1e9 / KB)
    print '\tSigma: %2.2f nK'%(std_err * M * 1e9 / KB)
    
    plt.plot(tofs, widths, '.'); plt.plot(xaxis, np.sqrt(slope*(xaxis*1e3)**2 + intercept*1e12)); 
    plt.xlabel('TOF / ms'); plt.ylabel('Sigma_x / um'); plt.title('Temperature Fit');plt.show()
    return (coefs[0] * M / KB, coefs)

def tstar(n0):
    '''Calculate characteristic phonon temperature, using eqn 10.78 in P&S pg
    312'''
    return (n0 * U0 / KB)

def damping(n0, T):
    '''Formulae from Pethick and Smith pg 313'''
    Tstar = tstar(n0)
    if 0.1*Tstar < T < 10*Tstar:
        print "DANGER ZONE!"
    if T < Tstar:
        return (3 *  pi**4.5 / 5) * (n0 * A**3)**0.5 * (T / Tstar)**4 * omega_q
    else:
        return 2.09 * (A / n0)**0.5 * (omega_q / lambda_T**2)
