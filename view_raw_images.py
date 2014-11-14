# import cloud_distribution as cd
import glob
from cloud_image import CloudImage as ci
import matplotlib.pyplot as plt
import scipy
import numpy as np

directory = r'C:\Users\LevLab\Desktop\raw_image_appearance'
filelist = glob.glob(directory + '\\*.mat')
imgno = 1
for f in filelist:
    # print 'Image #%d'%imgno
    img = ci(f)
    # plt.imshow(img.atom_image - img.light_image)
    # plt.show()
    # scipy.misc.imsave(directory+'\\noiseimg%d.jpg'%imgno, img.atom_image - img.light_image)
    ai = img.atom_image
    li = img.light_image
    # li_shifted_a = np.roll(img.light_image, 1, 0)
    # li_shifted_b = np.roll(img.light_image, -1, 0)
    # li_shifted_c = np.roll(img.light_image, 1, 1)
    # li_shifted_d = np.roll(img.light_image, -1, 1)
    
    # diff_a = li.astype('float64') - li_shifted_a.astype('float64')
    # diff_b = li.astype('float64') - li_shifted_b.astype('float64')
    # diff_c = li.astype('float64') - li_shifted_c.astype('float64')
    # diff_d = li.astype('float64') - li_shifted_d.astype('float64')

    fluc_corr = np.sum(ai) / np.sum(li)
    li_corr = fluc_corr * li
    motion_im_b = np.abs(ai - li_corr)
    print 'Image #%d:'%imgno
    print f
    # scipy.misc.imsave(directory+'\\shiftedimg_a%d.jpg'%imgno, diff_a)
    # scipy.misc.imsave(directory+'\\shiftedimg_b%d.jpg'%imgno, diff_b)
    # scipy.misc.imsave(directory+'\\shiftedimg_c%d.jpg'%imgno, diff_c)
    # scipy.misc.imsave(directory+'\\shiftedimg_d%d.jpg'%imgno, diff_d)

    # scipy.misc.imsave(directory+'\\motionimg%d.jpg'%imgno, motion_im)
    scipy.misc.imsave(directory+'\\motionimg_b_%d.jpg'%imgno, motion_im_b)
    scipy.misc.imsave(directory+'\\lightimg%d.jpg'%imgno, li)
    scipy.misc.imsave(directory+'\\atomimg%d.jpg'%imgno, img.atom_image)
    scipy.misc.imsave(directory+'\\odimg%d.jpg'%imgno, img.get_od_image())
    imgno += 1