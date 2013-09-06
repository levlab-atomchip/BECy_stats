import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\movedipole_num_TOF1\\2013-08-29\\'
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\movedipole_num\\2013-08-29\\'

dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocapture_num_TOF1\\2013-08-29\\'
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocapture_num_TOF5\\2013-08-29\\'

dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocompress_num_TOF05\\2013-08-29\\'
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocompress_num_TOF1\\2013-08-29\\'
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocompress_num_TOF2\\2013-08-29\\'
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocompress_num_TOF2_5\\2013-08-29\\'
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocompress_num_TOF3\\2013-08-29\\'
dir = 'C:\\Users\\Levlab\\Documents\\becy_stats\\magellandata082913\\macrocompress_num_TOF4\\2013-08-29\\'



imagelist = glob.glob(dir + '*.mat')

axis = 0 #0 = x, 1 = y
widths = []
numimgs = len(imagelist)
imgind = 1

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    thiswidth = 1e6*thisimg.getWidth(axis)
    # if thiswidth > 1e6: #cheap bad img check
    widths.append(thiswidth)
    # print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

#outlier removal
widths = hempel.hempel_filter(widths)
# print widths
    
outputfile = dir + 'widths' + '.csv'
with open(outputfile, 'w') as f:
    writer = csv.writer(f)
    # writer.writerow((ContParName, 'Position'))
    # rows = zip(param_vals, widths)
    for num in widths:
        writer.writerow([num])
    
    
# from scipy.stats import gaussian_kde
# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
# density = gaussian_kde(widths)
# xs = np.linspace(.75*np.min(widths),1.25*np.max(widths),200)
# density.covariance_factor = lambda : .25
# density._compute_covariance()
# plt.plot(xs,density(xs))
# plt.xlabel('Position')
# plt.ylabel('Probability Density')
# plt.title('Number Probability Density')
# plt.show()
    
# print(widths)
print dir
# print numimgs
print('%2.2f'%np.mean(widths))
print('%2.2f'%np.std(widths))
print('%2.2f'%(np.std(widths)/np.mean(widths)))
plt.hist(widths,20)
plt.show()

plt.plot(widths, marker='o', linestyle = '--')
plt.show()