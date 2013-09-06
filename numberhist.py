import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats
import math
dir = r'C:\ImagingSave\statistics\dipole_num\2013-09-05\\'
imagelist = glob.glob(dir + '*.mat')

numbers = []
numimgs = len(imagelist)
imgind = 1

# plt.imshow(CloudImage.CloudImage(imagelist[0]).getODImage())
# plt.show()

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    thisnumber = thisimg.getAtomNumber()
    # if thisnumber > 1e6: #cheap bad img check
    numbers.append(thisnumber)
    print('Processed %d out of %d images'%(imgind, numimgs))

    imgind += 1

#outlier removal
numbers = hempel.hempel_filter(numbers)
# numbers = [x for x in numbers if x < 4e6]
# print numbers

# print(thisimg.truncWinX)
# print(thisimg.truncWinY)


outputfile = dir + 'numbers' + '.csv'
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    # writer.writerow((ContParName, 'Number'))
    # rows = zip(param_vals, numbers)
    for num in numbers:
        writer.writerow([num])
    
    
from scipy.stats import gaussian_kde
density = gaussian_kde(numbers)
xs = np.linspace(.75*np.min(numbers),1.25*np.max(numbers),200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.xlabel('Atom Number')
plt.ylabel('Probability Density')
plt.title('Number Probability Density')
plt.show()
    
print(numbers)
print('Mean: %2.2e'%np.mean(numbers))
print('StdDev: %2.2e'%np.std(numbers))
# print('%2.2e'%(2*np.std(numbers)/np.mean(numbers)))
print('SNR: %2.2f'%stats.signaltonoise(numbers))
print('sigma_SNR: %2.2f'%(math.sqrt((2 + stats.signaltonoise(numbers)**2) / len(numbers))))
plt.hist(numbers,20)
plt.show()

plt.plot(numbers, marker='o', linestyle = '--')
plt.show()