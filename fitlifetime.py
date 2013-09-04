from scipy import optimize
import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats
import math

#Physical Constants
M = 1.66e-27
KB = 1.38e-23

def lifetimefunc(t, N_0, tau):
    return N_0*np.exp(t / (-tau))

def fitlife(holdtimes, numbers):
    lifefit_params, lifefit_covars = optimize.curve_fit(lifetimefunc, holdtimes, numbers, (numbers[0], 10))
    N_0 = lifefit_params[0]
    tau = lifefit_params[1]
    return (N_0, tau, lifefit_covars[1][1])
    
def life_chi_2(holdtimes, numbers):
    (T, lifefit_params) = fitlife(np.array(holdtimes), np.array(numbers))
    (N_0, tau) = (lifefit_params[0], lifefit_params[1])
    sqr_errs = [(w - lifetimefunc(t, N_0, tau))**2 for (t, w) in zip(holdtimes, numbers)]
    return (sum(sqr_errs), T)
    
if __name__ == "__main__":

    dir = r'C:\Users\Levlab\Documents\becy_stats\090113\movedipole_lifetime\2013-09-01\\'
    ContParName = None

    imagelist = glob.glob(dir + '*.mat')

    holdtimes = []
    numbers = []
    numimgs = len(imagelist)
    imgind = 1
    # print(imagelist)

    for img in imagelist:
        thisimg = CloudImage.CloudImage(img)
        # if thisimg.getAtomNumber() < 4e7:
        holdtimes.append(float(thisimg.CurrContPar[0]))
        numbers.append(float(thisimg.getAtomNumber()))
        print('Processed %d out of %d images'%(imgind, numimgs))
        imgind += 1
    print holdtimes
    print numbers
    (initial_num, lifetime, lifetime_var) = fitlife(np.array(holdtimes), np.array(numbers))
    print("Lifetime: %2.2f sec"%lifetime)
    print("Lifetime Uncertainty: %2.2f sec"%(math.sqrt(lifetime_var)))
    
    uniqueholdtimes = sorted(list(set(holdtimes)))
    
    plt.plot(holdtimes, numbers, linestyle="None", marker="o")
    plt.plot(uniqueholdtimes, [lifetimefunc(t, initial_num, lifetime) for t in uniqueholdtimes])
    plt.title('Lifetime in Dipole')
    plt.xlabel('Time / sec')
    plt.ylabel('Number')
    plt.show()
    