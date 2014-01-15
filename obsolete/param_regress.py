import CloudImage
import glob
import matplotlib.pyplot as plt
from scipy import stats
import csv
import numpy as np
dir = r'C:\ImagingSave\statistics\loadmot_varyxbias\2013-09-05\\'
ContParName = None

imagelist = glob.glob(dir + '*.mat')

param_vals = []
numbers = []
numimgs = len(imagelist)
imgind = 1
# print(imagelist)

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    # print(thisimg.CurrContPar)
    if thisimg.getAtomNumber() < 1e8:
        param_vals.append(thisimg.CurrContPar)
        numbers.append(thisimg.getAtomNumber())
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1
    
ContParName = thisimg.ContParName

    
outputfile = dir + ContParName + '.csv'
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow((ContParName, 'Number'))
    rows = zip(param_vals, numbers)
    writer.writerows(rows)
    
slope, intercept, r_value, p_value, std_err = stats.linregress(param_vals, numbers)

print("Slope: %2.2e"%slope)
print("Intercept: %2.2e"%intercept)
print("Standard Error: %2.2e"%std_err)
    
plt.plot(param_vals,numbers,'.')
plt.plot(param_vals, slope*np.array(param_vals) + intercept)
plt.xlabel(ContParName)
plt.ylabel('Atom Number') 
plt.show()

plt.plot(numbers)
plt.show()