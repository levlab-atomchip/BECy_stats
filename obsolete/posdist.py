import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats
import math

dir = r'D:\ACMData\Statistics\moveDipole_number\2014-01-13\\'

imagelist = glob.glob(dir + '*.mat')

axis = 0 #0 = x, 1 = y
positions = []
numimgs = len(imagelist)
imgind = 1

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    try:
        thispos = 1e6*thisimg.getPos(axis, flucCor_switch = False) #microns
        positions.append(thispos)
    except CloudImage.FitError as e:
        print(e.args)
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

#outlier removal
positions = hempel.hempel_filter(positions)
# print positions

# Subtract out the mean
meanpos = np.mean(positions)
positions = [p - meanpos for p in positions]

# outputfile = dir + 'positions' + '.csv'
# with open(outputfile, 'w') as f:
    # writer = csv.writer(f)
    # writer.writerow((ContParName, 'Position'))
    # rows = zip(param_vals, positions)
    # for num in positions:
        # writer.writerow([num])
    
    
    
# print(positions)
print dir
print numimgs
print len(positions)

# print('Mean: %2.2e'%np.mean(positions))
print('StdDev: %2.2e'%np.std(positions))
# print('SNR: %2.2f'%stats.signaltonoise(positions))
# print('sigma_SNR: %2.2f'%(math.sqrt((2 + stats.signaltonoise(positions)**2) / len(positions))))

plt.hist(positions,20)
plt.xlabel('Cloud Position / um')
plt.ylabel('Counts')
if axis == 0:
    plt.title('X Position Histogram')
elif axis == 1:
    plt.title('Z Position Histogram')
plt.show()

plt.plot(positions, marker='o', linestyle = '--')
plt.xlabel('Run Number')
plt.ylabel('Atom Position')
if axis == 0:
    plt.title('X Position over Time')
elif axis == 1:
    plt.title('Z Position over Time')
plt.show()