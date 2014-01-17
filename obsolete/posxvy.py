import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats
import math

# dir = r'C:\Users\Will\Documents\becystats\ModTfrSetup\loadmot_number\2013-11-14\\'
dir = r'G:\111513 loadmot number\loadmot_number\2013-11-15\\'

imagelist = glob.glob(dir + '*.mat')

# axis = 1 #0 = x, 1 = y
positionsx = []
positionsy = []
numbers = []
numimgs = len(imagelist)
imgind = 1

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    # thisimg.set_fluc_corr(700,750,100,200)
    thisposx = 1e6*thisimg.getPos(0, flucCor_switch = False)
    thisposz = 1e6*thisimg.getPos(1, flucCor_switch = False)
    # thisnumber = thisimg.getAtomNumber(axis=1, offset_switch = True, flucCor_switch = False)
    # if thisnumber > 1e6: #cheap bad img check
    # numbers.append(thisnumber)
    # if thispos > 1e6: #cheap bad img check
    positionsx.append(thisposx)
    positionsy.append(thisposz)
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

#outlier removal
# positions = hempel.hempel_filter(positions)
# print positions
    
# outputfile = dir + 'numpos' + '.csv'
# with open(outputfile, 'w') as f:
    # writer = csv.writer(f)
    # writer.writerow((ContParName, 'Position'))
    # rows = zip(param_vals, positions)
    # for num in positions:
        # writer.writerow([num])
    
    
# from scipy.stats import gaussian_kde
# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
# density = gaussian_kde(positions)
# xs = np.linspace(.75*np.min(positions),1.25*np.max(positions),200)
# density.covariance_factor = lambda : .25
# density._compute_covariance()
# plt.plot(xs,density(xs))
# plt.xlabel('Position')
# plt.ylabel('Probability Density')
# plt.title('Number Probability Density')
# plt.show()
    
# print(positions)
print dir
print numimgs
print len(positionsx)
# print('%2.2f'%np.mean(positions))
# print('%2.2f'%np.std(positions))
# print('%2.2f'%(np.std(positions)/np.mean(positions)))

# print('Mean: %2.2e'%np.mean(positions))
# print('StdDev: %2.2e'%np.std(positions))
# print('%2.2e'%(2*np.std(numbers)/np.mean(numbers)))
# print('SNR: %2.2f'%stats.signaltonoise(positions))
# print('sigma_SNR: %2.2f'%(math.sqrt((2 + stats.signaltonoise(positions)**2) / len(positions))))

# plt.hist(positions,20)
# plt.show()

plt.plot(positionsx, positionsy, marker='o', linestyle = 'none')
plt.xlabel('X Position')
plt.title('X Position against Z Position')
# plt.title('Y Position')
plt.ylabel('Z Position')
plt.show()