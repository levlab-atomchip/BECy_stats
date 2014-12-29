import numpy as np
dir = raw_input('Enter directory with data: ')

#get images
imagelist = glob.glob(dir + '*.mat')

chisquared_x = [()]
chisquared_y = [()]

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    
    this_x = thisimg.getChiSquared1D(0)
    this_y = thisimg.getChiSquared1D(1)
    
    np.append(chisquared_x, this_x)
    np.append(chisquared_y, this_y)