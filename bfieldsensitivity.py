from cloud_distribution import *
from cloud_image import CloudImage as ci
import BECphysics as bp
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from mpl_toolkits.mplot3d.axes3d import Axes3D


def field_dist(dist, UNBIAS=False, **kwargs):
    imgs = [ci(ff) for ff in dist.filelist]
    cdimgs = [img.get_od_image()/ img.s_lambda for img in imgs]
    lds = [bp.line_density(cd) for cd in cdimgs]

#plt.plot(lds[0])
#plt.show()

    fas = [bp.field_array(ld, **kwargs) for ld in lds]
    if UNBIAS:
	    fas = [fa - np.mean(fa) for fa in fas]
    return fas

def field_avg(dist, offdist, **kwargs):
    fas = field_dist(dist, **kwargs)
    fas_off = field_dist(offdist, **kwargs)

    fa_mean = np.mean(np.array(fas), axis=0)
    fa_off_mean = np.mean(np.array(fas_off), axis = 0)

    fa_std = np.std(np.array(fas), axis=0)
    fa_off_std = np.std(np.array(fas_off), axis=0)

    fa_tot = fa_mean - fa_off_mean
    fa_noise = np.sqrt(np.power(fa_std, 2) + np.power(fa_off_std, 2))

    xaxis = np.cumsum(np.ones(len(fas[0])) * (13.0 / 21))
    return fa_tot, fa_noise, xaxis


def field_noise(dist, offdist, **kwargs):
    fas = field_dist(dist, **kwargs)
    fas_off = field_dist(offdist, **kwargs)

    fa_std = np.std(np.array(fas), axis = 0)
    fa_off_std = np.std(np.array(fas_off), axis=0)

    fa_noise = np.sqrt(np.power(fa_std, 2) + np.power(fa_off_std, 2))

    xaxis = np.cumsum(np.ones(len(fas[0])) * (13.0 / 21))
    return fa_noise, xaxis


def ci_to_fa(image):
    cdimg = image.get_od_image() / image.s_lambda
    ldimg = bp.line_density(cdimg, PIXSIZE) 
    faimg = bp.field_array(ldimg)
    return faimg

def main():

    dist = CD(r'/home/will/levlab/data/bfieldnoise_20141201/RRconfig/', False)
    offdist = CD(r'/home/will/levlab/data/bfieldnoise_20141201/off/', False)
    fa_tot, fa_noise, xaxis = field_avg(dist, offdist, UNBIAS=False)
    plt.plot(xaxis, fa_tot*1e9, '.')
    plt.xlabel('X axis / um')
    plt.ylabel('Field / nT')
    plt.title('Averaged field measurement over 5 um wires')
    plt.show()

    plt.plot(xaxis, fa_noise*1e9, '.')
    plt.xlabel('X axis / um')
    plt.ylabel('Field noise / nT')
    plt.title('Field Noise vs Position')
    plt.show()

    plt.errorbar(xaxis, fa_tot*1e9, fa_noise*1e9)
    plt.xlabel('X axis / um')
    plt.ylabel('Field / nT')
    plt.title('Average field measurement over 5 um wires')
    plt.show()

#fas_unbias = [fa - np.mean(fa) for fa in fas]
#fas_unbias_arr = np.array(fas_unbias)
#fa_unbias_mean = np.mean(fas_unbias_arr, axis=0)
#fa_unbias_std = np.std(fas_unbias_arr, axis=0)

#plt.plot(xaxis, fa_unbias_std)
#plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#for fa in fas:
#    plt.plot(xaxis, fa)
#plt.show()

if __name__ == "__main__":
    main()
