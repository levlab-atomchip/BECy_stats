import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy.stats import gaussian_kde

# Run this script to generate time series and histogram plots of data. 
# You must input the directory containing the data you want to analyze.
# You must input the variables in a comma separated list (no spaces)


#get inputs
dir = raw_input('Enter directory with data: ')
vars = raw_input('Enter variables to plot, comma separated: ')
varlist = vars.split(',')
#C:\Users\Levlab\Documents\becy_stats\magellandata082913\macrocompress_num_TOF4\2013-08-29\

#get images
imagelist = glob.glob(dir + '*.mat')
numimgs = len(imagelist)

#Initialize a dictionary of data lists
data = {}
for var in varlist: 
    data[var] = []

#iterate over images    
for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    for var in varlist:
        thisvalue = thisimg.getvalue(var)
        data[var].append(thisvalue)

#outlier removal, output data as a csv, and plot as a histogram and time series
for var in varlist:
    data[var]=hempel.hempel_filter(data[var])
    outputfile = dir+var+ '.csv'
    
    with open(outputfile, 'w') as f:
        writer = csv.writer(f)
        # writer.writerow((ContParName, 'Position'))
        # rows = zip(param_vals, positions)
        for num in data[var]:
            writer.writerow([num])
    
    density = gaussian_kde(data[var])
    xs = np.linspace(.75*np.min(data[var]),1.25*np.max(data[var]),200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    

    plt.subplot(131)
    plt.hist(data[var],np.ceil(np.power(numimgs,0.33)))
    plt.ylabel('Counts')
    plt.xlabel(var)
    plt.title('Histogram')
    plt.subplot(132)
    plt.plot(xs,density(xs))
    plt.xlabel(var)
    plt.ylabel('Probability Density')
    plt.title('Number Probability Density')
    plt.subplot(133)
    plt.plot(data[var],marker='o',linestyle='--')
    plt.ylabel(var)
    plt.xlabel('Run Number')
    plt.title('Time Series')
    plt.suptitle(dir)
    plt.show()