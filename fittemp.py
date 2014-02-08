from scipy import optimize
import numpy as np

#Physical Constants
M = 1.66e-27
KB = 1.38e-23

def tempfunc(t, sigma_0, sigma_v):
    return np.sqrt(sigma_0**2 + (sigma_v**2)*(t**2))

def fittemp(TOFs, widths):
    tempfit_params, tempfit_covars = optimize.curve_fit(tempfunc, np.array(TOFs), np.array(widths), (widths[0], 0.002))
    sigma_v = tempfit_params[1]
    T = M * sigma_v**2 / KB
    return (T, tempfit_params)

def temp_chi_2(TOFs, widths):
    (T, tempfit_params) = fittemp(np.array(TOFs), np.array(widths))
    (sigma_0, sigma_v) = (tempfit_params[0], tempfit_params[1])
    sqr_errs = [(w - tempfunc(t, sigma_0, sigma_v))**2 for (t, w) in zip(TOFs, widths)]
    return (sum(sqr_errs), T)