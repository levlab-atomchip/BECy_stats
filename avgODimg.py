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
odimgs = []
numimgs = len(imagelist)
imgind = 1

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    thisODimg = thisimg.getODImage()
    # if thisODimg > 1e6: #cheap bad img check
    odimgs.append(thisODimg)
    # print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

#outlier removal
# odimgs = hempel.hempel_filter(odimgs)
# print odimgs
    
# outputfile = dir + 'odimgs' + '.csv'
# with open(outputfile, 'w') as f:
    # writer = csv.writer(f)
    # writer.writerow((ContParName, 'Position'))
    # rows = zip(param_vals, odimgs)
    # for num in odimgs:
        # writer.writerow([num])
    
    
# from scipy.stats import gaussian_kde
# data = [1.5]*7 + [2.5]*2 + [3.5]*8 + [4.5]*3 + [5.5]*1 + [6.5]*8
# density = gaussian_kde(odimgs)
# xs = np.linspace(.75*np.min(odimgs),1.25*np.max(odimgs),200)
# density.covariance_factor = lambda : .25
# density._compute_covariance()
# plt.plot(xs,density(xs))
# plt.xlabel('Position')
# plt.ylabel('Probability Density')
# plt.title('Number Probability Density')
# plt.show()
    
# print(odimgs)
print dir
# print numimgs
print('%2.2f'%np.mean(odimgs))
print('%2.2f'%np.std(odimgs))
print('%2.2f'%(np.std(odimgs)/np.mean(odimgs)))
# plt.hist(odimgs,20)
# plt.show()

# plt.plot(odimgs, marker='o', linestyle = '--')
# plt.show()