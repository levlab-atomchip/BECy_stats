import CloudImage
import glob
import matplotlib.pyplot as plt
import numpy as np
dir = 'Z:\\Users\\rwturner\\BECy Characterization\\statistics\\loadmot_poor\\'

imagelist = glob.glob(dir + '*.mat')

numbers = []
numimgs = len(imagelist)
imgind = 1

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    numbers.append(thisimg.getAtomNumber())
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

print(numbers)
print(np.mean(numbers))
print(np.std(numbers))
print(2*np.std(numbers)/np.mean(numbers))
plt.hist(numbers, 5)
plt.show()