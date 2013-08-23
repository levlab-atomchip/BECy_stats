import CloudImage
import glob
import matplotlib.pyplot as plt
from scipy import stats
import csv
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\statistics\\loadmot_varyLoadTime\\2013-08-22\\'
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
    param_vals.append(thisimg.CurrContPar[0])
    # print(param_vals)
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

print("Slope: %f"%slope)
print("Intercept: %f"%intercept)
print("Standard Error: %f"%std_err)
    
plt.plot(param_vals,numbers,'.')
plt.xlabel(ContParName)
plt.ylabel('Atom Number') 
plt.show()

plt.plot(numbers)
plt.show()