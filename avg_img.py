'''script for producing average of OD images in directory'''

import glob
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.colors as clr

dir = r'D:\ACMData\Stability\moveDipole_position_instability\NoHoldRoundTrip\2014-01-20\\'
# dir = r'D:\ACMData\Stability\moveDipole_position_instability\Hold6077NoMove\2014-01-20\\'

filelist = glob.glob(dir + '*.mat')


VIEW = 1
ROI = 1

ROIX1 = 300
ROIX2 = 1100
ROIY1 = 0
ROIY2 = 550

avg_img = np.zeros((ROIY2 - ROIY1, ROIX2 - ROIX1))

img_ind = 0

for file in filelist:
    try:
        mat_dict = scipy.io.loadmat(file)
    except:
        print(file + ' failed')
        pass
    rawimages = mat_dict['rawImage']
    if ROI:
        atomImg = np.nan_to_num(rawimages[ROIY1:ROIY2,ROIX1:ROIX2,0])
        lightImg = np.nan_to_num(rawimages[ROIY1:ROIY2,ROIX1:ROIX2,1])
    else:
        atomImg = np.nan_to_num(rawimages[:,:,0])
        lightImg = np.nan_to_num(rawimages[:,:,1])
    ODImg = np.absolute(np.log(atomImg + 1) - np.log(lightImg + 1))
    avg_img = (img_ind * avg_img + ODImg)/(img_ind + 1)
    print(file + ' processed')
    
print('Directory %s Processed'%dir)

plt.imshow(avg_img, cmap = 'jet')
plt.show()