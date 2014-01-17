import CloudImage
import glob
import matplotlib.pyplot as plt
from scipy import stats
import numpy
import csv

#This function takes two parameters and does a linear regression 

#Get inputs
dir = raw_input('Enter directory with data: ')
indvar = raw_input('Enter the independent variable: ')
depvar = raw_input('Enter the dependent variable: ')

#Initialize
imagelist = glob.glob(dir + '*.mat')
ind_vals = []
dep_vals = []

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    ind_vals.append(thisimg.getvalue(indvar))
    dep_vals.append(thisimg.getvalue(depvar))

print ind_vals
print dep_vals
    
#Write output file
outputfile = dir + indvar+depvar + '.csv'
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow((indvar, depvar))
    rows = zip(ind_vals, dep_vals)
    writer.writerows(rows)

slope, intercept, r_value, p_value, std_err = stats.linregress(ind_vals, dep_vals)

line = [slope*x+intercept for x in ind_vals]
print("Slope: %f"%slope)
print("Intercept: %f"%intercept)
print("Standard Error: %f"%std_err)

plt.plot(ind_vals,dep_vals,'.',ind_vals,line,'r-')
plt.xlabel(indvar)
plt.ylabel(depvar)
plt.title('Linear Regression of '+indvar+ ' vs. ' + depvar)
plt.show()