import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# T = (np.array([4,5,6,7,8,9])*1e-3) #sec
# X = np.array([])
# Y = 564 - np.array([336,373,409,449,506,564]) #px

T = (np.array([5,10,15,20,25,30])*1e-3) #sec
X = np.array([562,565,569,576,585,595])
Y = np.array([652,602,517,396,240,53]) #px

CAMPIXSIZE = 3.75e-6 #m
G = 9.8 #m/s^2

def fit_func(x, a, b, c):
    return a*x**2 + b*x + c
    
if __name__ == "__main__":
    popt, pcov = curve_fit(fit_func, T, np.max(Y)-Y)
    M = 2*popt[0] * CAMPIXSIZE / G
    sigma = 2*np.sqrt(pcov[0][0]) * CAMPIXSIZE / G
    print "Magnification: %2.2f"%M
    print "Sigma: %2.2f"%sigma
    plt.plot(T, Y, '.')
    xax = np.linspace(np.min(T), np.max(T))
    plt.plot(xax, np.max(Y) - fit_func(xax, *popt))
    plt.xlabel('TOF / ms')
    plt.ylabel('Height / px')
    plt.title('Magnification Fit')
    plt.show()