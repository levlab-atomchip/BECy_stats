import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
import hempel
import csv
from scipy import stats
import math

dir = r'C:\Users\Will\Desktop\Benchmark\2014-01-07\\'

imagelist = glob.glob(dir + '*.mat')

# numbers = []
numimgs = len(imagelist)
imgind = 1



for img in imagelist:
    
    # thisimg.set_fluc_corr(700,750,100,200)
    # thisnumber = thisimg.getAtomNumber(axis=1, offset_switch = True, flucCor_switch = True, debug_flag = True)
    # if thisnumber > 1e6: #cheap bad img check
    # numbers.append(thisnumber)
    try:
        thisimg = CloudImage.CloudImage(img)
        plt.imshow(CloudImage.CloudImage(imagelist[0]).getODImage())
        plt.show()
    except:
        print('fail')
    print('Processed %d out of %d images'%(imgind, numimgs))

    imgind += 1