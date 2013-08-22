# Rb constants file ala Kris Baumann

import scipy

from scipy.constants import hbar,pi,c
from scipy.constants import epsilon_0 as epsilon0

# try:
    # from enthought.traits.api import HasTraits, , Property
# except:
    # from traits.api import HasTraits, , Property

class parameters(): #provide general AMO constants
    c = (2.99792458*10**8) #m/s
    mu_0 = (4*pi*10**-7) #N/A^2
    h = hbar*2*pi #Js
    e = (1.602176487*10**-19) #C
    mu_b = (9.27400915*10**-24) #J/T
    m_e = (9.10938215*10**-31) #kg
    a_0 = (0.52917720859*10**-10) #m Bohr Radius
    a_s = (5.2*10**(-9)) #m scattering length
    k_b = (1.3806504*10**-23) #J/K
    m_Rb87 = (1.443160648*10**-25) #kg

class cp_parameters(parameters): #Provide constants for CP simulation
    N = 7*10**3
    m_Rb87 = 1.443160648*10**-25 #kg
    a_0 = (0.52917720859*10**-10) #m
    a_s = 5.2*10**(-9) #m scattering length
    wx = (10*pi*2)
    wy = (2*pi*3e3)
    wz = (2*pi*3e3)
    wbar = (wx*wy*wz)**(1/3.)
    abar = (hbar/(m_Rb87*wbar))**.5
    a1d =(15*N*a_s/abar)**(2/5.)*(hbar*wbar/2)**(3/2.)*(2*m_Rb87)**.5/(3*wx*hbar**2*pi*N)

class bfield_parameters(parameters): #Provide constants for CP simulation
    N = 7*10**3
    m_Rb87 = 1.443160648*10**-25 #kg
    a_0 = (0.52917720859*10**-10) #m
    a_s = 5.2*10**(-9) #m scattering length
    wx = (10*pi*2)
    wy = (2*pi*3e3)
    wz = (2*pi*3e3)
    wbar = (wx*wy*wz)**(1/3.)
    abar = (hbar/(m_Rb87*wbar))**.5
    w_tr = 2*pi*3e3
    g_F = .5
    a1d =(15*N*a_s/abar)**(2/5.)*(hbar*wbar/2)**(3/2.)*(2*m_Rb87)**.5/(3*wx*hbar**2*pi*N)
