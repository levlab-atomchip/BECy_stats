import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats
import math

# dir = r'C:\Users\Will\Documents\becystats\loadmot_many\2340\2013-08-21\\'
dir = r'C:\Users\Will\Documents\becystats\ModTfrSetup\loadmot_number\2013-11-14\\'

imagelist = glob.glob(dir + '*.mat')

intensities = []
numimgs = len(imagelist)
imgind = 1

plt.imshow(CloudImage.CloudImage(imagelist[0]).atomImage)
plt.show()

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    thisintensity = thisimg.getLightCounts()
    # if thisintensity > 1e6: #cheap bad img check
    intensities.append(thisintensity)
    print('Processed %d out of %d images'%(imgind, numimgs))

    imgind += 1

#outlier removal
intensities = hempel.hempel_filter(intensities)
# print intensities

outputfile = dir + 'intensities' + '.csv'
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    # writer.writerow((ContParName, 'Number'))
    # rows = zip(param_vals, intensities)
    for num in intensities:
        writer.writerow([num])
    
    
from scipy.stats import gaussian_kde
density = gaussian_kde(intensities)
xs = np.linspace(.75*np.min(intensities),1.25*np.max(intensities),200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.xlabel('Light Counts')
plt.ylabel('Probability Density')
plt.title('Light Count Probability Density')
plt.show()
    
print(intensities)
print('Mean: %2.2e'%np.mean(intensities))
print('StdDev: %2.2e'%np.std(intensities))
# print('%2.2e'%(2*np.std(intensities)/np.mean(intensities)))
print('SNR: %2.2f'%stats.signaltonoise(intensities))
print('sigma_SNR: %2.2f'%(math.sqrt((2 + stats.signaltonoise(intensities)**2) / len(intensities))))
plt.hist(intensities,20)
plt.show()

plt.plot(intensities, marker='o', linestyle = '--')
plt.show()