import CloudImage
import glob
import matplotlib.pyplot as plt
dir = 'Z:\\Users\\rwturner\\BECy Characterization\\statistics\\loadmot_varyYLoad\\2013-08-21\\'

imagelist = glob.glob(dir + '*.mat')

param_vals = []
numbers = []
numimgs = len(imagelist)
imgind = 1
# print(imagelist)

for img in imagelist:
    thisimg = CloudImage.CloudImage(img)
    # print(thisimg.CurrContPar)
    param_vals.append(thisimg.CurrContPar)
    # print(param_vals)
    numbers.append(thisimg.getAtomNumber())
    print('Processed %d out of %d images'%(imgind, numimgs))
    imgind += 1

plt.plot(param_vals,numbers,'.')
plt.xlabel(thisimg.ContParName)
plt.ylabel('Atom Number') 
plt.show()

plt.plot(numbers)
plt.show()