import CloudImage
import glob
import matplotlib.pyplot as plt
from scipy import stats
import csv
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\statistics\\loadmot_varyLoadTime\\2013-08-22\\'
ContParName = None

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
    posx.append(thisimg.getpos(1))
    posy.append(thisimg.getpos(2))
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1
    
ContParName = thisimg.ContParName

    
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
    
plt.plot(param_vals,posy,'.')
plt.xlabel(ContParName)
plt.ylabel('Y Position')
plt.title('Y Position Vs.' + ContParName 
plt.show()

plt.plot(param_vals,posx,'.')
plt.xlabel(ContParName)
plt.ylabel('X Position')
plt.title('X Position Vs.' + ContParName 
plt.show()