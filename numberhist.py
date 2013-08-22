import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
dir = 'C:\\ImagingSave\\statistics\\loadmot_many\\2340\\2013-08-21\\'

imagelist = glob.glob(dir + '*.mat')

numbers = []
numimgs = len(imagelist)
imgind = 1

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    thisnumber = thisimg.getAtomNumber()
    # if thisnumber > 1e6: #cheap bad img check
    numbers.append(thisnumber)
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

#outlier removal
numbers = hempel.hempel_filter(numbers)
    
    
from scipy.stats import gaussian_kde
# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
density = gaussian_kde(numbers)
xs = np.linspace(.75*np.min(numbers),1.25*np.max(numbers),200)
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.xlabel('Atom Number')
plt.ylabel('Probability Density')
plt.title('Number Probability Density, 100 Runs')
plt.show()
    
print(numbers)
print(np.mean(numbers))
print(np.std(numbers))
print(2*np.std(numbers)/np.mean(numbers))
plt.hist(numbers,20)
plt.show()

plt.plot(numbers, marker='o', linestyle = '--')
plt.show()