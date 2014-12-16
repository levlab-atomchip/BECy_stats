'''bfieldsensitivity.py - code for calculating magnetic field sensitivity from a distribution of nominally identical data'''

from cloud_distribution import *
from cloud_image import CloudImage as ci
import BECphysics as bp
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from mpl_toolkits.mplot3d.axes3d import Axes3D

DEFAULT_PIXSIZE = 13.0 / 24 *1e-6 #PIXIS

def field_dist(dist, unbias=False, **kwargs):
    '''Return a list of magnetic field profiles calculated from data in distribution.
        Args:
            dist: a CloudDistribution
            unbias: if True, subtract the mean magnetic field from each field profile
    '''
    imgs = [ci(ff) for ff in dist.filelist]
    cdimgs = [img.get_od_image()/ img.s_lambda for img in imgs]
    lds = [bp.line_density(cd) for cd in cdimgs]

    fas = [bp.field_array(ld, **kwargs) for ld in lds]
    if unbias:
	    fas = [fa - np.mean(fa) for fa in fas]
    return fas

def field_avg(dist, offdist, pixsize=DEFAULT_PIXSIZE, **kwargs):
    '''Return spatially varying statistics of magnetic field profiles.
        Args:
            dist: CloudDistribution of data with sample on
            offdist: CloudDistribution of data in same trap with sample off
            pixsize: real space pixel length
    '''
    fas = field_dist(dist, **kwargs)
    fas_off = field_dist(offdist, **kwargs)

    fa_mean = np.mean(np.array(fas), axis=0)
    fa_off_mean = np.mean(np.array(fas_off), axis = 0)

    fa_std = np.std(np.array(fas), axis=0)
    fa_off_std = np.std(np.array(fas_off), axis=0)

    fa_tot = fa_mean - fa_off_mean
    fa_noise = np.sqrt(np.power(fa_std, 2) + np.power(fa_off_std, 2))

    xaxis = np.cumsum(np.ones(len(fas[0])) * pixsize)
    return fa_tot, fa_noise, xaxis


def field_noise(dist, offdist, pixsize=DEFAULT_PIXSIZE, **kwargs):
    '''Return spatially varying standard deviation of magnetic field profiles'''
    fas = field_dist(dist, **kwargs)
    fas_off = field_dist(offdist, **kwargs)

    fa_std = np.std(np.array(fas), axis = 0)
    fa_off_std = np.std(np.array(fas_off), axis=0)

    fa_noise = np.sqrt(np.power(fa_std, 2) + np.power(fa_off_std, 2))

    xaxis = np.cumsum(np.ones(len(fas[0])) * pixsize)
    return fa_noise, xaxis


def ci_to_fa(image, pixsize=DEFAULT_PIXSIZE):
    '''Return magnetic field profile calculated from CloudImage'''
    cdimg = image.get_od_image() / image.s_lambda
    ldimg = bp.line_density(cdimg, pixsize) 
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
