import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats
import math
# dir = r'D:\ACMData\Statistics\loadmot_number\2013-11-26\\'
dir = r'D:\ACMData\Statistics\moveDipole_number\2014-01-13\\'

imagelist = glob.glob(dir + '*.mat')

numbers = []
numimgs = len(imagelist)
imgind = 1

# plt.imshow(CloudImage.CloudImage(imagelist[0]).getODImage())
# plt.show()

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    # thisimg.set_fluc_corr(700,750,100,200)
    try:
        thisnumber = thisimg.getAtomNumber(axis=1, offset_switch = True, flucCor_switch = False, debug_flag = True)
        numbers.append(thisnumber)
    except CloudImage.FitError as e:
        print e.args
    # if thisnumber > 1e6: #cheap bad img check
    print('Processed %d out of %d images'%(imgind, numimgs))

    imgind += 1

#outlier removal
#numbers = hempel.hempel_filter(numbers)
# numbers = [x for x in numbers if x > 1e8]
print numbers

# print(thisimg.truncWinX)
# print(thisimg.truncWinY)


outputfile = dir + 'numbers' + '.csv'
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    # writer.writerow((ContParName, 'Number'))
    # rows = zip(param_vals, numbers)
    for num in numbers:
        writer.writerow([num])
    
    
# from scipy.stats import gaussian_kde
# density = gaussian_kde(numbers)
# xs = np.linspace(.75*np.min(numbers),1.25*np.max(numbers),200)
# density.covariance_factor = lambda : .25
# density._compute_covariance()
# plt.plot(xs,density(xs))
# plt.xlabel('Atom Number')
# plt.ylabel('Probability Density')
# plt.title('Number Probability Density')
# plt.show()
    
print(numbers)
print('Mean: %2.2e'%np.mean(numbers))
print('StdDev: %2.2e'%np.std(numbers))
# print('%2.2e'%(2*np.std(numbers)/np.mean(numbers)))
print('SNR: %2.2f'%stats.signaltonoise(numbers))
print('sigma_SNR: %2.2f'%(math.sqrt((2 + stats.signaltonoise(numbers)**2) / len(numbers))))
plt.hist(numbers,20)
plt.xlabel('Atom Number')
plt.ylabel('Counts')
plt.title('Number Histogram')
plt.show()

plt.plot(numbers, marker='o', linestyle = '--')
plt.xlabel('Run Number')
plt.ylabel('Atom Number')
plt.title('Number over Time')
plt.show()