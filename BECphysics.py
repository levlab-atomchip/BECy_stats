from math import sqrt, exp, pi

AMU = 1.66e-27 # kg
M = 87*AMU #Rubidium-87
KB = 1.38e-23 # J/K

MYTEMP = 640e-9 #K
MYFREQ = 2*pi*500 #Hz

def width(t, T = MYTEMP, omega = MYFREQ):
    '''Width of a thermal gas at temperature T in a trap with frequency omega after time of flight t'''
    return sqrt(((KB * T)/(M*omega**2))*(1 + omega**2 * t**2))