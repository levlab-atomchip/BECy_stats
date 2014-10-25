# import cloud_distribution as cd
import glob
from cloud_image import CloudImage as ci
import matplotlib.pyplot as plt
import scipy

directory = r'D:\ACMData\Statistics\microRFevap_number\2014-09-26\stability_1_06'
filelist = glob.glob(directory + '\\*.mat')
imgno = 1
for f in filelist:
    print 'Image #%d'%imgno
    img = ci(f)
    # plt.imshow(img.atom_image - img.light_image)
    # plt.show()
    # scipy.misc.imsave(directory+'\\noiseimg%d.jpg'%imgno, img.atom_image - img.light_image)
    scipy.misc.imsave(directory+'\\atomimg%d.jpg'%imgno, img.atom_image)
    scipy.misc.imsave(directory+'\\odimg%d.jpg'%imgno, img.get_od_image())
    imgno += 1