from scipy.stats import gaussian_kde
import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats

dir = r'C:\ImagingSave\statistics\vary_uniblitz_time\2013-09-04\\'
imagelist = glob.glob(dir + '*.mat')

numbers = {}
numimgs = len(imagelist)
imgind = 1

# plt.imshow(CloudImage.CloudImage(imagelist[0]).getODImage())
# plt.show()

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    thisnumber = thisimg.getAtomNumber()
    thiscontpar = thisimg.CurrContPar
    if thiscontpar not in numbers:
        numbers[thiscontpar] = []
    numbers[thiscontpar].append(thisnumber)
    # if thisnumber > 1e6: #cheap bad img check
    # numbers.append(thisnumber)
    print('Processed %d out of %d images'%(imgind, numimgs))

    imgind += 1

#outlier removal
for par_val in numbers:
    numbers[par_val] = hempel.hempel_filter(numbers[par_val])
# numbers = [x for x in numbers if x < 4e6]
# print numbers

# print(thisimg.truncWinX)
# print(thisimg.truncWinY)


# outputfile = dir + 'numbers' + '.csv'
# with open(outputfile, 'w') as f:
    # writer = csv.writer(f)
    # writer.writerow((ContParName, 'Number'))
    # rows = zip(param_vals, numbers)
    # for num in numbers:
        # writer.writerow([num])
    
for par_val in numbers:
    density = gaussian_kde(numbers[par_val])
    xs = np.linspace(.75*np.min(numbers[par_val]),1.25*np.max(numbers[par_val]),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs))
plt.xlabel('Atom Number')
plt.ylabel('Probability Density')
plt.title('Number Probability Density')
plt.show()

for par_val in numbers:
    print(par_val)
    print('%2.2e'%np.mean(numbers[par_val]))
    print('%2.2e'%np.std(numbers[par_val]))
    # print('%2.2e'%(2*np.std(numbers[par_val])/np.mean(numbers[par_val])))
    print('SNR: %2.2f'%stats.signaltonoise(numbers[par_val]))
# plt.hist(numbers,20)
# plt.show()

# plt.plot(numbers, marker='o', linestyle = '--')
# plt.show()