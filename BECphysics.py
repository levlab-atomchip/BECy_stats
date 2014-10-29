'''BEC Physics

This module collects functions useful for calculating physical properties of the atomic cloud'''


from math import sqrt, exp, pi
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

AMU = 1.66e-27 # kg, atomic mass unit
M = 87*AMU #Rubidium-87 mass
KB = 1.38e-23 # J/K, Boltzmann constant
HBAR = 1.05e-34 # J/sec, Planck constant
MUB = 9.274e-24 # J/T, Bohr Magneton

A = 5.6e-9 # m, Rb-87 scattering length
G = 4*pi*HBAR**2*A / M # J*m^3, coupling constant
MU_PRE = G*(15.0/pi)**(0.4) * (1.0 / 2**1.8) * (M/G)**0.6 # A prefactor for chemical potential calculation

MYTEMP = 640e-9 #K, temperature of the cloud
MYFREQ = 2*pi*500 #Hz, trap frequency of the magnetic trap

PIXELSIZE = 3.75e-6 #m, real size of camera pixel side

def width(t, T = MYTEMP, omega = MYFREQ):
    '''Width of a thermal gas at temperature T in a trap with frequency omega 
    after time of flight t'''
    return sqrt(((KB * T)/(M*omega**2))*(1 + omega**2 * t**2))

def temp_func(tsqrd, w0sqrd, tempscld):
    '''fitting function for temperature measurement'''
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
    
def line_density(column_density, pixel_size=PIXELSIZE):
    '''Given a column density, return the integrated linear density for the 
    long direction of the cloud'''
    return np.sum(column_density, axis = 0) * pixel_size
    
def number_array(line_density, pixel_size=PIXELSIZE):
    '''Given a line density, return an array with the number of atoms in each 
    pixel'''
    return line_density * pixel_size
    
def density_map(cds, xaxis, ylocs, pixel_size=PIXELSIZE):
    '''Given a sequence of column densities (cds) with a given xaxis taken at 
    different sample locations (ylocs) produce a map of the linear density'''
    lds = [line_density(cd) for cd in cds]
    # X, Y = np.meshgrid(xaxis, ylocs)
    ldsarr = np.array(lds)
    return ldsarr
    
def field_array(l_density, omega_rad, omega_long):
    '''Given a linear atom density, produce a map of magnetic field'''
    mu = chemical_potential(l_density, omega_rad, omega_long)
    return (mu - HBAR*omega_rad * np.sqrt(1 + 4*A*l_density)) / MUB
    
def chemical_potential(line_density, omega_rad, omega_long):
    '''Given a line density in a harmonic trap defined by omega_rad, 
    omega_long, return the global chemical potential.
    This is based on the Thomas-Fermi approximation and is intended for use in 
    perturbed harmonic traps, which is an uncontrolled approximation...'''
    num_array = number_array(line_density)
    N_tot = np.sum(num_array)
    return MU_PRE * (N_tot * omega_rad**2 * omega_long)**0.4
    
    
