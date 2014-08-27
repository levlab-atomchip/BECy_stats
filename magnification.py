import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 7, 7, 7, 9, 9, 9])*1e-3 #sec
Y = np.array([400,398,398,391,392,391,387,384,384,382,376,384,372,375,378, 354, 352, 352, 328, 316, 324]) #px

CAMPIXSIZE = 3.75e-6 #m
G = 9.8 #m/s^2

def fit_func(x, a, b, c):
    return a*x**2 + b*x + c
    
if __name__ == "__main__":
    popt, pcov = curve_fit(fit_func, X, 400-Y)
    M = 2*popt[0] * CAMPIXSIZE / G
    sigma = 2*np.sqrt(pcov[0][0]) * CAMPIXSIZE / G
    print "Magnification: %2.1f"%M
    print "Sigma: %2.1f"%sigma
    plt.plot(X, 400-Y, '.')
    xax = np.linspace(np.min(X), np.max(X))
    plt.plot(xax, fit_func(xax, *popt))
    plt.show()