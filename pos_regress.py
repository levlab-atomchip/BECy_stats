import CloudImage
import glob
import matplotlib.pyplot as plt
from scipy import stats
import csv
import numpy as np
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\statistics\\loadmot_varyYLoad\\2013-08-21\\'
ContParName = None
unit = None

imagelist = glob.glob(dir + '*.mat')

param_vals = []
posx = []
posy = []
numimgs = len(imagelist)
imgind = 1
# print(imagelist)

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    # print(thisimg.CurrContPar)
    param_vals.append(thisimg.CurrContPar[0])
    # print(param_vals)
    # numbers.append(thisimg.getAtomNumber())
    posx.append(thisimg.getPos(0))
    posy.append(thisimg.getPos(1))
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1
    
ContParName = thisimg.ContParName
# print thisimg.pixel_size

if ContParName == "XLoad":
    param_vals = np.array(param_vals) / -0.479 #convert volts to amps
    unit = ' / A'
elif ContParName == "YLoad":
    param_vals = np.array(param_vals) / -0.479
    unit = ' / A'
elif ContParName == "ZLoad":
    param_vals = -6.0 * np.array(param_vals)
    unit = ' / A'

outputfile = dir + ContParName + 'vsPos.csv'
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow((ContParName, 'X Position', 'Y Position'))
    rows = zip(param_vals, posx, posy)
    writer.writerows(rows)
    
# slope, intercept, r_value, p_value, std_err = stats.linregress(param_vals, numbers)

# print("Slope: %f"%slope)
# print("Intercept: %f"%intercept)
# print("Standard Error: %f"%std_err)
    
plt.plot(param_vals,1e6*np.array(posy),'.')
plt.xlabel(ContParName + unit)
plt.ylabel('Y Position / um')
plt.title('Y Position Vs. ' + ContParName )
plt.show()

plt.plot(param_vals,1e6*np.array(posx),'.')
plt.xlabel(ContParName + unit)
plt.ylabel('X Position / um')
plt.title('X Position Vs. ' + ContParName) 
plt.show()